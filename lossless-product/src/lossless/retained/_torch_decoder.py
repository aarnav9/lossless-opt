"""Retained campaign-011 decoder boundaries; pinned legacy API only."""

import torch
from torch import nn


class FusedDecoder(nn.Module):
    def __init__(self, original, settings):
        super().__init__()
        object.__setattr__(self, "fallback", original)
        self.self_attn = original.self_attn
        self.input_layernorm = original.input_layernorm
        self.post_attention_layernorm = original.post_attention_layernorm
        self.mlp = original.mlp
        self.algorithm = settings.get("algorithm", 0)
        self.full = settings.get("fuse_mlp", True)
        self.merge = settings.get("merge_projections", False)
        self.attention = settings.get("fuse_attention", False)
        if self.merge:
            assert self.mlp.gate_proj.bias is None and self.mlp.up_proj.bias is None
            self.register_buffer(
                "packed_gate_up",
                torch.cat([self.mlp.gate_proj.weight, self.mlp.up_proj.weight], dim=0).contiguous(),
            )
        assert self.post_attention_layernorm.variance_epsilon == 1e-5

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        output_attentions=False,
        use_cache=False,
        cache_position=None,
        position_embeddings=None,
        **kwargs,
    ):
        if not torch._dynamo.is_compiling() and (
            not torch.is_inference_mode_enabled()
            or hidden_states.dtype != torch.float32
            or hidden_states.device.type != "cpu"
        ):
            return self.fallback(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )
        if (
            self.attention
            and past_key_value is None
            and not use_cache
            and not output_attentions
            and position_embeddings is not None
        ):
            h = torch.ops.lossless_retained.decoder(
                hidden_states,
                self.input_layernorm.weight,
                self.self_attn.q_proj.weight,
                self.self_attn.k_proj.weight,
                self.self_attn.v_proj.weight,
                self.self_attn.o_proj.weight,
                *position_embeddings,
                self.post_attention_layernorm.weight,
                self.packed_gate_up if self.merge else self.mlp.gate_proj.weight,
                self.mlp.up_proj.weight,
                self.mlp.down_proj.weight,
                self.self_attn.num_heads,
                self.self_attn.num_key_value_heads,
                self.algorithm,
                self.merge,
                attention_mask,
            )
            return (h,)
        residual = hidden_states
        h = self.input_layernorm(hidden_states)
        h, attentions, cache = self.self_attn(
            hidden_states=h,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        shape = h.shape
        a = h.reshape(-1, shape[-1])
        r = residual.reshape(-1, shape[-1])
        if self.full:
            operation = (
                torch.ops.lossless_retained.rms_mlp_merged
                if self.merge
                else torch.ops.lossless_retained.rms_mlp
            )
            h = operation(
                a,
                r,
                self.post_attention_layernorm.weight,
                self.packed_gate_up if self.merge else self.mlp.gate_proj.weight,
                self.mlp.gate_proj.bias,
                self.mlp.up_proj.weight,
                self.mlp.up_proj.bias,
                self.mlp.down_proj.weight,
                self.mlp.down_proj.bias,
                self.algorithm,
                0,
            ).reshape(shape)
        else:
            norm, added = torch.ops.lossless_retained.rms_pre(
                a, r, self.post_attention_layernorm.weight, self.algorithm, 0
            )
            h = self.mlp(norm.reshape(shape)) + added.reshape(shape)
        outputs = (h,)
        if output_attentions:
            outputs += (attentions,)
        if use_cache:
            outputs += (cache,)
        return outputs


def patched_model(model, settings):
    import copy

    # Separate module structure, identical immutable parameter tensors.
    memo = {id(t): t for t in list(model.parameters()) + list(model.buffers())}
    candidate = copy.deepcopy(model, memo)
    layers = candidate.lm.model.layers if hasattr(candidate, "lm") else candidate.layers
    for i, layer in enumerate(layers):
        layers[i] = FusedDecoder(layer, settings)
    if settings.get("direct_no_cache"):
        candidate.direct_no_cache = True
    if settings.get("protect_outer"):
        assert hasattr(candidate, "lm")
        candidate.lm.model.norm = OpaqueNorm(candidate.lm.model.norm)
        candidate.lm.model.rotary_emb = OpaqueRotary(candidate.lm.model.rotary_emb)
    return candidate.eval()


class OpaqueNorm(nn.Module):
    def __init__(self, original):
        super().__init__()
        self.weight = original.weight
        self.eps = original.variance_epsilon

    def forward(self, x):
        return torch.ops.lossless_retained.plain_norm(x, self.weight, self.eps)


class OpaqueRotary(nn.Module):
    def __init__(self, original):
        super().__init__()
        assert original.rope_type == "default"
        self.register_buffer("inv_freq", original.inv_freq)
        self.scaling = float(original.attention_scaling)

    def forward(self, x, position_ids):
        return torch.ops.lossless_retained.rope(x, self.inv_freq, position_ids, self.scaling)
