"""Starting implementation. Authors may replace this entire module with Python/MLX code."""

from lossless.adapters.mlx.model import GroupedHead
from lossless.adapters.mlx.cache import run
from lossless.adapters.mlx.runtime import allocator, CACHE_RECIPE
from lossless.adapters.mlx.graph import GraphRecipe


class Runner:
    def __init__(self, model):
        self.model = model
        self.grouped = GroupedHead(model, 4)
        self.graph = GraphRecipe(model, "fullhead_027", 64)

    def run(self, prompts, counts, capture=True):
        # No prompts, outputs or KV from earlier requests may be memoized.
        recipe = {**CACHE_RECIPE, "append_metal": False, "append_concat": True}
        with allocator({"allocator_cache_mib": 256}), self.graph.installed():
            return run(self.grouped, prompts, counts, recipe, capture)
