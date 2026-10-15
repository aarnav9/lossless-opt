"""Frozen coverage workloads using the normal project workflow and public adapter.

Run each workload once with the published a3 runtime on PYTHONPATH and once with
current source. The shared harness freezes discovery/evaluation cases before
search, records full requests, and checks new values through the public adapter.
"""

from pathlib import Path
import inspect
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cpu_usability import study


study.WRAPPERS = {
    "filterpy-outer": """import numpy as np
from filterpy.common import outer_product_sum

def run(matrices):
    return np.asarray([outer_product_sum(a) for a in matrices])
""",
    "filterpy-resample": """import numpy as np
from filterpy.monte_carlo import systematic_resample

def run(weights):
    weights = weights / weights.sum(axis=1, keepdims=True)
    return np.asarray([systematic_resample(row) for row in weights])
""",
    "filterpy-ensemble": """import numpy as np
from filterpy.kalman import EnsembleKalmanFilter

def transition(x, dt):
    return x + dt

def measurement(x):
    return x[:1]

def run(measurements):
    f = EnsembleKalmanFilter(np.array([0., 1.]), np.eye(2), 1, .1, 12,
                            measurement, transition)
    values = []
    for z in measurements:
        f.predict()
        f.update(z)
        values.append((f.x.copy(), f.P.copy(), f.sigmas.copy()))
    return values
""",
    "perlin-numpy": study.WRAPPERS["perlin-numpy"],
    "numpy-financial": study.WRAPPERS["numpy-financial"],
}
study.INPUTS.update(
    {
        "filterpy-outer": ([512, 640, 768], "args = [rng.normal(size=(size, 8, 4))]"),
        "filterpy-resample": ([256, 384, 512], "args = [rng.uniform(.1, 1., (size, 16))]"),
        "filterpy-ensemble": ([24, 32, 40], "args = [rng.normal(size=(size, 1))]"),
    }
)
study.PACKAGES.update({name: "filterpy" for name in study.WRAPPERS if name.startswith("filterpy-")})


# a3 has no coverage option. Current source keeps the new recipes opt-in after
# this campaign found no additional qualified application wins.
original_optimize_project = study.optimize_project


def optimize_with_coverage(*args, **kwargs):
    if "recipe_coverage" in inspect.signature(original_optimize_project).parameters:
        kwargs["recipe_coverage"] = "expanded"
    return original_optimize_project(*args, **kwargs)


study.optimize_project = optimize_with_coverage


if __name__ == "__main__":
    study.main()
