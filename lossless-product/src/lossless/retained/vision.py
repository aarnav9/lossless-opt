"""CPU SigLIP sign reuse; exact arithmetic path, component scope only."""


class SiglipLoss:
    """Bounded constant reuse; changing logits and autograd remain uncached.

    Matches ``-logsigmoid((2*eye(n)-1)*logits).sum(-1).mean()``. Supports
    float32/float64 square CPU logits; no claims about optimizer/model state.
    Cache keys include dtype and size, and retained signs have a hard byte cap.
    """

    def __init__(self, *, byte_limit=16 * 1024**2, capacity=2):
        if (
            type(byte_limit) is not int
            or byte_limit < 1
            or type(capacity) is not int
            or not 1 <= capacity <= 128
        ):
            raise ValueError("positive byte limit and capacity in 1..128 required")
        self.byte_limit, self.capacity = byte_limit, capacity
        self._signs = {}

    def __call__(self, logits):
        import torch
        from torch.nn import functional as F

        if (
            logits.device.type != "cpu"
            or logits.dtype not in (torch.float32, torch.float64)
            or logits.ndim != 2
            or logits.shape[0] != logits.shape[1]
            or not logits.numel()
        ):
            raise ValueError("nonempty square CPU float32/float64 logits required")
        # Account for ambient inference mode: tensors created there cannot be
        # saved for a later autograd call. Keep caches distinct across modes.
        key = (logits.shape[0], logits.dtype, torch.is_inference_mode_enabled())
        sign = self._signs.get(key)
        if sign is None:
            sign = 2 * torch.eye(key[0], dtype=logits.dtype, device=logits.device) - 1
            size = sign.numel() * sign.element_size()
            if size <= self.byte_limit:
                while self._signs and (
                    len(self._signs) >= self.capacity
                    or self.retained_bytes + size > self.byte_limit
                ):
                    del self._signs[next(iter(self._signs))]
                self._signs[key] = sign
        return -F.logsigmoid(sign * logits).sum(dim=-1).mean()

    @property
    def retained_bytes(self):
        return sum(t.numel() * t.element_size() for t in self._signs.values())
