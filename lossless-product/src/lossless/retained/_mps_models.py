import copy
import torch
from torch import nn
from torch.nn import functional as F


class MLP(nn.Module):
    def __init__(self, original, config):
        super().__init__()
        self.original = original
        self.config = config
        if config.get("pack"):
            self.register_buffer(
                "packed",
                torch.cat((original.gate_proj.weight, original.up_proj.weight), 0).contiguous(),
            )

    def forward(self, x):
        m = self.original
        c = self.config
        if (
            c.get("metal_gemv_silu")
            and x.shape[-2] == 1
            and x.numel() // x.shape[-1] <= c.get("gemv_decode_rows", 1)
        ):
            return m.down_proj(
                torch.ops.lossless_mps.gemv_silu(
                    x.contiguous(), self.packed, c.get("gemv_silu_mode", 0)
                )
            )
        if c.get("metal_packed_silu"):
            p = F.linear(x, self.packed)
            return m.down_proj(
                torch.ops.lossless_mps.silu_packed(p, c.get("silu_mode", 0), c.get("threads", 256))
            )
        if c.get("pack"):
            g, u = F.linear(x, self.packed).chunk(2, -1)
        else:
            g, u = m.gate_proj(x), m.up_proj(x)
        if c.get("metal_silu"):
            p = torch.ops.lossless_mps.silu_mul(
                g.contiguous(), u.contiguous(), c.get("threads", 256), c.get("silu_mode", 0)
            )
        else:
            p = F.silu(g) * u
        return m.down_proj(p)


class RMS(nn.Module):
    def __init__(self, original, config):
        super().__init__()
        self.original = original
        self.config = config

    def forward(self, x):
        if self.config.get("native_rms"):
            return F.rms_norm(
                x, (x.shape[-1],), self.original.weight, self.original.variance_epsilon
            )
        if self.config.get("metal_rms_tail"):
            variance = x.pow(2).mean(-1, keepdim=True)
            return torch.ops.lossless_mps.norm_tail(
                x.contiguous(),
                variance.contiguous(),
                self.original.weight,
                self.original.variance_epsilon,
                self.config.get("tail_mode", 0),
                self.config.get("threads", 256),
            )
        if self.config.get("metal_rms_safe"):
            scale = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.original.variance_epsilon)
            return torch.ops.lossless_mps.norm_scale(
                x.contiguous(),
                scale.contiguous(),
                self.original.weight,
                self.config.get("threads", 256),
            )
        return torch.ops.lossless_mps.rms(
            x.contiguous(),
            self.original.weight,
            self.original.variance_epsilon,
            self.config.get("rms_threads", 256),
        )


class Attention(nn.Module):
    def __init__(self, original, config):
        super().__init__()
        self.original = original
        self.config = config
        if config.get("qkv_pack"):
            self.register_buffer(
                "qkv",
                torch.cat(
                    (original.q_proj.weight, original.k_proj.weight, original.v_proj.weight), 0
                ).contiguous(),
            )

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
        a = self.original
        x = hidden_states
        B, L, _ = x.shape
        if output_attentions or position_embeddings is None:
            return a(
                hidden_states=x,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )
        if self.config.get("qkv_pack"):
            projection = F.linear(x, self.qkv)
            q, k, v = projection.split(
                [
                    a.num_heads * a.head_dim,
                    a.num_key_value_heads * a.head_dim,
                    a.num_key_value_heads * a.head_dim,
                ],
                -1,
            )
        else:
            q, k, v = a.q_proj(x), a.k_proj(x), a.v_proj(x)
        cos, sin = position_embeddings
        if self.config.get("metal_qkv_rope"):
            q, k, v = torch.ops.lossless_mps.qkv_rope(
                projection,
                cos.contiguous(),
                sin.contiguous(),
                a.num_heads,
                a.num_key_value_heads,
                self.config.get("rope_threads", 256),
            )
        else:
            q, k = torch.ops.lossless_mps.rope(
                q.contiguous(),
                k.contiguous(),
                cos.contiguous(),
                sin.contiguous(),
                a.num_heads,
                a.num_key_value_heads,
                self.config.get("rope_threads", 256),
            )
            v = v.view(B, L, a.num_key_value_heads, a.head_dim).transpose(1, 2)
        if past_key_value is not None:
            k, v = past_key_value.update(
                k, v, a.layer_idx, {"sin": sin, "cos": cos, "cache_position": cache_position}
            )
        from transformers.models.llama.modeling_llama import repeat_kv

        if not self.config.get("gqa"):
            k = repeat_kv(k, a.num_key_value_groups)
            v = repeat_kv(v, a.num_key_value_groups)
        mask = attention_mask[:, :, :, : k.shape[-2]] if attention_mask is not None else None
        extra = {"enable_gqa": True} if self.config.get("gqa") else {}
        z = F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, dropout_p=0.0, is_causal=mask is None and L > 1, **extra
        )
        return a.o_proj(z.transpose(1, 2).contiguous().view(B, L, -1)), None, past_key_value


class LastLogits(nn.Module):
    """Generation-loop adapter: only last-position logits are part of this API.

    Not a drop-in replacement for arbitrary full-logit forward calls or training.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, past_key_values, use_cache):
        from transformers.modeling_outputs import CausalLMOutputWithPast

        z = self.model.model(
            input_ids=input_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            return_dict=True,
        )
        logits = self.model.lm_head(z.last_hidden_state[:, -1:, :]).float()
        return CausalLMOutputWithPast(logits=logits, past_key_values=z.past_key_values)


class ResidualDecoder(nn.Module):
    def __init__(self, original, config):
        super().__init__()
        self.original = original
        self.config = config

    def forward(self, hidden_states, **kwargs):
        layer = self.original
        if kwargs.get("output_attentions", False):
            return layer(hidden_states, **kwargs)
        x = layer.input_layernorm(hidden_states)
        attention, weights, cache = layer.self_attn(hidden_states=x, **kwargs)
        residual, squared = torch.ops.lossless_mps.add_square(
            attention.contiguous(), hidden_states.contiguous(), self.config.get("threads", 256)
        )
        norm = layer.post_attention_layernorm.original
        variance = squared.mean(-1, keepdim=True)
        x = torch.ops.lossless_mps.norm_tail(
            residual,
            variance.contiguous(),
            norm.weight,
            norm.variance_epsilon,
            self.config.get("tail_mode", 1),
            self.config.get("threads", 256),
        )
        out = (residual + layer.mlp(x),)
        return out + (cache,) if kwargs.get("use_cache", False) else out


class GemmResidualDecoder(nn.Module):
    def __init__(self, original, config):
        super().__init__()
        self.original = original
        self.config = config

    def forward(self, hidden_states, **kwargs):
        layer = self.original
        if (
            kwargs.get("output_attentions", False)
            or hidden_states.shape[-2] != 1
            or hidden_states.numel() // hidden_states.shape[-1]
            > self.config.get("gemv_decode_rows", 1)
        ):
            return layer(hidden_states, **kwargs)
        x = layer.input_layernorm(hidden_states)
        attention, weights, cache = layer.self_attn(hidden_states=x, **kwargs)
        residual = hidden_states + attention
        x = layer.post_attention_layernorm(residual)
        if self.config.get("metal_gemv_silu"):
            z = torch.ops.lossless_mps.gemv_silu(
                x.contiguous(), layer.mlp.packed, self.config.get("gemv_silu_mode", 1)
            )
        else:
            packed = F.linear(x, layer.mlp.packed)
            z = torch.ops.lossless_mps.silu_packed(packed, 1, 256)
        result = torch.ops.lossless_mps.gemv_add(
            z.contiguous(), layer.mlp.original.down_proj.weight, residual.contiguous()
        )
        out = (result,)
        return out + (cache,) if kwargs.get("use_cache", False) else out


def patch(model, config):
    # Share immutable parameters, independently copy the module tree.
    memo = {id(p): p for p in list(model.parameters()) + list(model.buffers())}
    out = copy.deepcopy(model, memo)
    for layer_index, layer in enumerate(out.model.layers):
        if config.get("metal_rope"):
            layer.self_attn = Attention(layer.self_attn, config)
        if config.get("pack") or config.get("metal_silu"):
            layer.mlp = MLP(layer.mlp, config)
        if (
            config.get("metal_rms")
            or config.get("metal_rms_safe")
            or config.get("metal_rms_tail")
            or config.get("native_rms")
        ):
            layer.input_layernorm = RMS(layer.input_layernorm, config)
            layer.post_attention_layernorm = RMS(layer.post_attention_layernorm, config)
        if config.get("metal_residual"):
            out.model.layers[layer_index] = ResidualDecoder(layer, config)
        if config.get("metal_down_residual"):
            out.model.layers[layer_index] = GemmResidualDecoder(layer, config)
    if (
        config.get("metal_rms")
        or config.get("metal_rms_safe")
        or config.get("metal_rms_tail")
        or config.get("native_rms")
    ):
        out.model.norm = RMS(out.model.norm, config)
    out.eval()
    return LastLogits(out) if config.get("last_logits") else out
