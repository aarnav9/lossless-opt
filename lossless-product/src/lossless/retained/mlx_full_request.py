"""Experimental 048 whole-request candidate; not a default promoted recipe."""


def load_candidate(model_path, *, contract):
    """Load an existing local, fingerprinted model. No downloading or conversion.

    Call ``run(token_lists, counts, capture=True)`` for fixed-count generation.
    Returns timing/IDs, independent KV exports, and full probability arrays.
    EOS stopping, text encoding and cancellation are outside this token API.
    Model parameters must remain immutable for the lifetime of the candidate.
    """
    if contract != "exact":
        raise ValueError("this candidate targets the pinned exact MLX contract")
    from pathlib import Path
    import mlx.core as mx
    from mlx_lm import load
    from ..adapters.mlx.runtime import admission
    from ._mlx_full_request import Runner

    path = Path(model_path).resolve()
    if not path.is_dir():
        raise ValueError("an existing local model directory is required")
    admitted, reason = admission(path, mx.device_info()["device_name"])
    if not admitted:
        raise ValueError(reason)
    model, tokenizer = load(str(path))
    mx.eval(model.parameters())
    return _Candidate(Runner(model), tokenizer)


class _Candidate:
    def __init__(self, runner, tokenizer):
        self._runner = runner
        self.tokenizer = tokenizer

    def run(self, prompts, counts, capture=True):
        from ..adapters.mlx.batching import output_counts
        from ..adapters.mlx.runtime import LOCK

        if type(capture) is not bool or not isinstance(prompts, (list, tuple)):
            raise ValueError("token sequences and a boolean capture flag required")
        if not 1 <= len(prompts) <= 64:
            raise ValueError("requires 1..64 independent requests")
        counts = output_counts(counts, len(prompts))
        vocab = self._runner.model.args.vocab_size
        for ids, count in zip(prompts, counts):
            if (
                not isinstance(ids, (list, tuple))
                or not ids
                or count > 256
                or len(ids) + count > 8192
                or any(type(i) is not int or not 0 <= i < vocab for i in ids)
            ):
                raise ValueError("token request outside the pinned candidate domain")
        if not LOCK.acquire(blocking=False):
            raise ValueError("MLX adapter requires exclusive single-worker execution")
        try:
            return self._runner.run(prompts, counts, capture)
        finally:
            LOCK.release()
