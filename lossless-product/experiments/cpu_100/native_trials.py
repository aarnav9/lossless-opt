"""Full native-call overhead candidates against a frozen product implementation."""

import copy
from pathlib import Path
import shutil
import subprocess
import sys


from common import HERE, read, sha, write

GUARD_C = r"""
#include <math.h>
#include <stddef.h>
int valid(const float *x,size_t rows,size_t cols,size_t rs,size_t cs,float bound) {
 for(size_t i=0;i<rows;i++)for(size_t j=0;j<cols;j++){
  float v=x[i*rs+j*cs];if(!(v>=-bound && v<=bound))return 0;
 }return 1;
}
"""


def cases(final=False):
    specs = [
        ("copy_small", "copy", 3, 263, "c"),
        ("copy_large", "copy", 513, 1027, "c"),
        ("softmax_small", "softmax", 7, 257, "c"),
        ("softmax_f", "softmax", 31, 509, "f"),
        ("rmsnorm", "rmsnorm_residual", 33, 1021, "c"),
        ("rmsnorm_slice", "rmsnorm_residual", 19, 257, "slice2"),
    ]
    if final:
        specs += [("copy_f", "copy", 71, 97, "f"), ("softmax_long", "softmax", 13, 4093, "c")]
    return [
        dict(
            id=name,
            operator=op,
            rows=rows + (2 if final else 0),
            columns=cols + (4 if final else 0),
            layout=layout,
            seed=99172 + (3721 if final else 0) + i,
            requests=40,
        )
        for i, (name, op, rows, cols, layout) in enumerate(specs)
    ]


def native_inputs(case):
    from lossless._native import operators

    # Distinct arrays; do not repeatedly reuse one pointer. Allocation stays outside call timing.
    calls = []
    for i in range(case["requests"]):
        arrays = operators.input_arrays(case["operator"], case, case["seed"] + i)
        calls.append(arrays if case["operator"] == "rmsnorm_residual" else (arrays[0],))
    return calls


def make_operation(config, payload):
    from lossless._native import operators
    from lossless._native.deployment import NativeOperation

    case = payload["case"]
    project = Path(payload["project"])
    suffix = ".dylib" if sys.platform == "darwin" else ".so"
    lib = operators.load_library(project / (case["operator"] + suffix))
    op = NativeOperation.from_loaded_library(case["operator"], lib, [case])
    return lambda arrays: op(*arrays)


def freeze_baseline(out):
    root = out / "native_baseline"
    if root.exists():
        if not (root / "complete.json").exists():
            raise ValueError(
                "native baseline build is incomplete; preserve its artifacts before retrying"
            )
        return root
    native = root / "lossless/_native"
    native.mkdir(parents=True)
    (root / "lossless/__init__.py").write_text("")
    (native / "__init__.py").write_text("")
    source = HERE / "frozen_native"
    for name, expected in read(source / "manifest.json")["files"].items():
        if sha(source / name) != expected:
            raise ValueError("frozen native reference source changed")
    for name in ("operators.py", "deployment.py"):
        shutil.copyfile(source / name, native / name)
    from lossless._native import worker

    for operator in ("copy", "softmax", "rmsnorm_residual"):
        c = root / (operator + ".c")
        shutil.copyfile(source / (operator + ".c"), c)
        suffix = ".dylib" if sys.platform == "darwin" else ".so"
        result = worker.compile_job(dict(source=str(c), binary=str(root / (operator + suffix))))
        write(root / (operator + "-build.json"), result)
        if result["status"] != "ok":
            raise ValueError(result)
    write(
        root / "complete.json",
        dict(status="complete", operators=["copy", "softmax", "rmsnorm_residual"]),
    )
    return root


def proposal(number, history):
    exact = [r for r in history if r["decision"]["category"] == "exact"]
    passing = [r for r in exact if r["decision"]["eligible"]]
    case_safe = [r for r in exact if min(r["case_speedups"].values()) >= 1 / 1.03]
    best = max(passing or case_safe or exact, key=lambda r: r["speedup"]) if exact else None
    base = dict(
        raw=False,
        coalesce="none",
        ones=False,
        guard="original",
        direct=False,
        pointer_tuple=False,
        empty_weight=False,
        cache_shape=False,
        ones_limit=0,
    )
    experiments = [
        (
            "Use raw native addresses while preserving buffer ownership and the seven-pointer ABI.",
            dict(raw=True),
        ),
        ("Use tuple construction for bound native pointers.", dict(pointer_tuple=True)),
        (
            "Coallocate all three matrix buffers; test the output-lifetime memory penalty.",
            dict(coalesce="all"),
        ),
        (
            "Coallocate only scratch matrices while keeping returned output independently owned.",
            dict(coalesce="scratch"),
        ),
        ("Reuse one bounded immutable all-ones weight vector per operation.", dict(ones=True)),
        (
            "Fill an uninitialized weight vector instead of calling numpy.ones.",
            dict(empty_weight=True),
        ),
        (
            "Replace finite/absolute scans with allocation-free min/max domain checks.",
            dict(guard="minmax"),
        ),
        (
            "Use bounded-comparison NumPy domain checks and count their allocation cost.",
            dict(guard="compare"),
        ),
        ("Use one compiled strided-domain scan for each numerical input.", dict(guard="native")),
        (
            "Combine raw pointer binding with compiled domain validation.",
            dict(raw=True, guard="native"),
        ),
        (
            "Combine raw pointers with independent-output scratch coallocation.",
            dict(raw=True, coalesce="scratch"),
        ),
        ("Combine raw pointers and bounded reusable ones.", dict(raw=True, ones=True)),
        (
            "Combine min/max checks with raw pointers and immutable ones.",
            dict(raw=True, ones=True, guard="minmax"),
        ),
        (
            "Cache shape/layout admission keyed by immutable metadata; retain mutable-value validation.",
            dict(cache_shape=True),
        ),
        (
            "Native scans won on short strided inputs but lost on large RMSNorm: use native scans up to 8192 values, NumPy bounded comparisons above that, plus raw pointers and reusable weights.",
            dict(raw=True, ones=True, guard="hybrid"),
        ),
        (
            "Use direct immediate native dispatch without constructing the bound-call closure.",
            dict(direct=True, raw=True),
        ),
        (
            "Apply direct dispatch to the strongest exact observed guard strategy.",
            dict(adaptive=True, direct=True, raw=True),
        ),
        (
            "Add scratch coallocation to the best exact complete-call implementation.",
            dict(adaptive=True, coalesce="scratch"),
        ),
        (
            "Bound the incumbent's default-weight cache to 256 KiB; oversized fallback vectors remain ephemeral.",
            dict(adaptive=True, ones=True, ones_limit=65536),
        ),
        (
            "Combine the winning binding and guard choices, retaining independent output allocations.",
            dict(adaptive=True, coalesce="scratch", pointer_tuple=True),
        ),
    ]
    hypothesis, change = experiments[number - 1]
    change = dict(change)
    if change.pop("adaptive", False) and best:
        base.update(copy.deepcopy(best["proposal"]["options"]))
    base.update(change)
    return dict(
        hypothesis=hypothesis,
        options=base,
        parent_experiment=best["number"] if best else None,
        informed_by=[
            dict(experiment=r["number"], speedup=r["speedup"], cases=r["case_speedups"])
            for r in history[-2:]
        ],
        rationale="All input validation, allocations, dispatch and returned output ownership remain in ordinary calls. Exact is relative to the unchanged native implementation; its softmax/RMSNorm oracle contract remains numerical.",
    )


def build(baseline, folder, plan):
    project = folder / "project"
    shutil.copytree(baseline, project)
    native = project / "lossless/_native"
    o = plan["options"]
    operators = (native / "operators.py").read_text()
    deployment = (native / "deployment.py").read_text()
    if o["raw"]:
        operators = operators.replace(
            "ptr = ctypes.POINTER(ctypes.c_float)\n    lib.kernel.argtypes",
            "ptr = ctypes.c_void_p\n    lib.kernel.argtypes",
        )
        operators = operators.replace(
            "pointers = [a.ctypes.data_as(ctypes.POINTER(ctypes.c_float)) for a in owners]",
            "pointers = [int(a.ctypes.data) for a in owners]",
        )
    if o["pointer_tuple"]:
        operators = operators.replace(
            "pointers = [a.ctypes.data_as(ctypes.POINTER(ctypes.c_float)) for a in owners]",
            "pointers = tuple(a.ctypes.data_as(ctypes.POINTER(ctypes.c_float)) for a in owners)",
        )
        operators = operators.replace(
            "pointers = [int(a.ctypes.data) for a in owners]",
            "pointers = tuple(int(a.ctypes.data) for a in owners)",
        )
    original = "out, px, pr = [np.empty((rows, cols), dtype=np.float32) for _ in range(3)]"
    if o["coalesce"] == "all":
        operators = operators.replace(
            original, "out, px, pr = np.empty((3, rows, cols), dtype=np.float32)"
        )
    if o["coalesce"] == "scratch":
        operators = operators.replace(
            original,
            "out = np.empty((rows, cols), dtype=np.float32)\n    px, pr = np.empty((2, rows, cols), dtype=np.float32)",
        )
    if o["ones"]:
        deployment = deployment.replace(
            "weight = np.ones(cols, dtype=np.float32)",
            """weight = getattr(self, '_ones', None)
            if weight is None or weight.shape != (cols,):
                weight = np.ones(cols, dtype=np.float32)
                weight.flags.writeable = False
                self._ones = weight""",
        )
        if o.get("ones_limit"):
            deployment = deployment.replace(
                "                self._ones = weight",
                f"                if cols <= {o['ones_limit']}: self._ones = weight",
            )
    elif o["empty_weight"]:
        deployment = deployment.replace(
            "weight = np.ones(cols, dtype=np.float32)",
            "weight = np.empty(cols, dtype=np.float32)\n            weight.fill(1)",
        )
    if o["guard"] != "original":
        for name, bound in [("x", "bound"), ("residual", "1000"), ("weight", "2")]:
            old = f"not np.isfinite({name}).all() or np.max(np.abs({name})) > {bound}"
            # RMSNorm expression spans source lines.
            if name != "x":
                old = f"not np.isfinite({name}).all()\n                or np.max(np.abs({name})) > {bound}"
            if o["guard"] == "minmax":
                new = f"not (np.min({name}) >= -{bound} and np.max({name}) <= {bound})"
            elif o["guard"] == "compare":
                new = f"not (({name} >= -{bound}) & ({name} <= {bound})).all()"
            else:
                new = f"not _domain_valid({name}, {bound})"
            assert old in deployment, old
            deployment = deployment.replace(old, new)
        if o["guard"] in ("native", "hybrid"):
            c = native / "domain.c"
            c.write_text(GUARD_C)
            suffix = ".dylib" if sys.platform == "darwin" else ".so"
            command = [
                "clang",
                "-O3",
                "-fno-fast-math",
                "-ffp-contract=off",
                "-fPIC",
                "-dynamiclib" if sys.platform == "darwin" else "-shared",
                str(c),
                "-o",
                str(native / ("domain" + suffix)),
            ]
            result = subprocess.run(command, capture_output=True, text=True, timeout=30)
            write(
                folder / "build.json",
                dict(command=command, returncode=result.returncode, diagnostics=result.stderr),
            )
            if result.returncode:
                raise ValueError(result.stderr)
            deployment += """\nimport ctypes
from pathlib import Path
_guard=ctypes.CDLL(str(Path(__file__).with_name('domainSUFFIX')))
_guard.valid.argtypes=[ctypes.c_void_p]+[ctypes.c_size_t]*4+[ctypes.c_float]
_guard.valid.restype=ctypes.c_int
def _domain_valid(x,bound):
    if x.ndim==1:return bool(_guard.valid(int(x.ctypes.data),1,x.size,0,x.strides[0]//4,bound))
    return bool(_guard.valid(int(x.ctypes.data),*x.shape,*(s//4 for s in x.strides),bound))
""".replace("SUFFIX", suffix)
            if o["guard"] == "hybrid":
                deployment = deployment.replace(
                    "def _domain_valid(x,bound):",
                    "def _domain_valid(x,bound):\n    if x.size>8192:return bool(((x>=-bound)&(x<=bound)).all())",
                )
    if o["cache_shape"]:
        old = """compatible = (
            layout is not None and (rows, cols, layout) in self.cases and self.library is not None
        )"""
        new = """metadata = (rows, cols, layout, id(self.library))
        cached = getattr(self, '_admission', None)
        if cached is not None and cached[0] == metadata:
            compatible = cached[1]
        else:
            compatible = layout is not None and (rows, cols, layout) in self.cases and self.library is not None
            self._admission = (metadata, compatible)"""
        assert old in deployment
        deployment = deployment.replace(old, new)
    if o["direct"]:
        # Keep bind() unchanged for its reusable-buffer API. Immediate variant uses the same guards.
        deployment = deployment.replace(
            "def bind(self, x, residual=None, weight=None):",
            "def bind(self, x, residual=None, weight=None, _immediate=False):",
        )
        deployment = deployment.replace(
            "return operators.bind_native(arrays, self.library)",
            "return operators.bind_native(arrays, self.library, immediate=_immediate)",
        )
        deployment = deployment.replace(
            "return self.bind(x, residual, weight)()",
            "result = self.bind(x, residual, weight, _immediate=True)\n        return result() if callable(result) else result",
        )
        operators = operators.replace(
            "def bind_native(arrays, library):",
            "def bind_native(arrays, library, immediate=False):",
        )
        old = """    def call():
        library.kernel(*pointers, rows, cols, rs, cs)
        return owners[3]

    return call"""
        new = (
            """    if immediate:
        library.kernel(*pointers, rows, cols, rs, cs)
        return owners[3]
"""
            + old
        )
        assert old in operators
        operators = operators.replace(old, new)
    (native / "operators.py").write_text(operators)
    (native / "deployment.py").write_text(deployment)
    return dict(project=project, config={})
