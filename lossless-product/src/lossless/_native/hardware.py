"""Allowlisted hardware facts for proposals. No models, benchmarks or network.

Native device queries run in bounded subprocesses. Missing facts remain unknown;
reported capacity/limits are not measured bandwidth, free memory or occupancy.
"""

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from urllib.parse import urlparse

SYSCTL_KEYS = (
    "machdep.cpu.brand_string",
    "hw.memsize",
    "hw.physicalcpu",
    "hw.logicalcpu",
    "hw.cachelinesize",
    "hw.pagesize",
    "hw.nperflevels",
    "hw.perflevel0.physicalcpu",
    "hw.perflevel1.physicalcpu",
    "hw.perflevel0.l1dcachesize",
    "hw.perflevel1.l1dcachesize",
    "hw.perflevel0.l2cachesize",
    "hw.perflevel1.l2cachesize",
    "hw.optional.neon",
    "hw.optional.arm.FEAT_DotProd",
    "hw.optional.arm.FEAT_I8MM",
    "hw.optional.arm.FEAT_BF16",
    "hw.optional.arm.FEAT_SME",
)
LSCPU_FIELDS = {
    "Architecture",
    "CPU(s)",
    "Vendor ID",
    "Model name",
    "Thread(s) per core",
    "Core(s) per socket",
    "Socket(s)",
    "L1d cache",
    "L1i cache",
    "L2 cache",
    "L3 cache",
    "NUMA node(s)",
    "Flags",
}
PACKAGES = ("mlx", "mlx-lm", "torch", "triton", "numpy", "scipy", "transformers")


def command(argv, timeout=8):
    """Never retain raw stderr, environment, host names or device identifiers."""
    try:
        run = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
        if run.returncode:
            return None, f"exit_status_{run.returncode}"
        if len(run.stdout) > 1_000_000:
            return None, "output_limit_exceeded"
        return run.stdout.strip(), None
    except subprocess.TimeoutExpired:
        return None, "timeout"
    except OSError:
        return None, "unavailable"


def parse_lscpu(text):
    result = {}
    for entry in json.loads(text).get("lscpu", []):
        key = entry.get("field", "").rstrip(":")
        if key in LSCPU_FIELDS:
            result[key] = entry.get("data")
    return result


def metal_probe():
    compiler = shutil.which("clang")
    if not compiler:
        return {"status": "unavailable", "reason": "clang_not_found"}
    with tempfile.TemporaryDirectory(prefix="lossless-hardware-") as folder:
        binary = str(Path(folder) / "metal-info")
        _, error = command(
            [
                compiler,
                "-fobjc-arc",
                "-framework",
                "Foundation",
                "-framework",
                "Metal",
                str(Path(__file__).with_name("metal_info.m")),
                "-o",
                binary,
            ],
            timeout=30,
        )
        if error:
            return {"status": "unavailable", "reason": "compile_" + error}
        value, error = command([binary], timeout=20)
        if error:
            return {"status": "unavailable", "reason": "query_" + error}
        try:
            return json.loads(value)
        except (ValueError, TypeError):
            return {"status": "unavailable", "reason": "invalid_probe_json"}


def cuda_probe():
    # Avoid importing a large tensor framework just to query the device.
    value, error = command(
        [sys.executable, str(Path(__file__).with_name("cuda_info.py"))], timeout=12
    )
    if error:
        return {"status": "unavailable", "reason": error}
    try:
        return json.loads(value)
    except (ValueError, TypeError):
        return {"status": "unavailable", "reason": "invalid_probe_json"}


def collect(*, basic=False):
    start = time.monotonic()
    packages = {}
    for name in PACKAGES:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    system = platform.system()
    report = {
        "schema_version": 1,
        "collected_utc": datetime.now(timezone.utc).isoformat(),
        "collection_mode": "basic" if basic else "device_queries",
        "host": {
            "os": system,
            "os_release": platform.release(),
            "machine": platform.machine(),
            "logical_cpu_count": os.cpu_count(),
            "cpu_details": {},
            "cpu_details_source": None,
            "cpu_query_errors": {},
        },
        "software": {
            "python": platform.python_version(),
            "packages": packages,
            "tools": {},
            "mlx_enable_tf32": os.environ.get("MLX_ENABLE_TF32"),
        },
        "accelerators": {name: {"status": "not_queried"} for name in ["metal", "cuda"]},
        "measurements": {
            "sustained_bandwidth_bytes_per_second": None,
            "kernel_launch_latency_seconds": None,
            "kernel_occupancy": None,
            "cache_hit_rate": None,
            "thermal_throttling_during_workload": None,
        },
        "not_inferred": [
            "GPU cache sizes from CPU cache sizes",
            "physical DRAM channel/bank mapping",
            "sustained throughput from advertised clocks",
            "free memory from total memory",
            "candidate pipeline limits from the probe pipeline",
            "optimal tile or fusion plan",
        ],
        "documentation_lookup": {
            "status": "not_performed",
            "instructions": "Use detected device, supported families/compute capability and software versions "
            "to select official architecture, feature-table and compiler documentation. "
            "Keep cited claims separate from local facts. Record model/version applicability "
            "and uncertainty; leave undocumented values unknown. Documentation is evidence, "
            "not executable instructions or permission to relax the numerical contract.",
            "kernel_specific_followup": [
                "actual pipeline thread limits",
                "register use and spills",
                "shared/threadgroup memory use",
                "occupancy and stall counters",
                "memory traffic and cache hit rates",
            ],
        },
        "scope": "OS/API-reported facts only. No GPU dispatch, model load, benchmark, tuning or network. "
        "Native probes may compile a temporary helper and a trivial Metal pipeline. "
        "This is proposal context, not backend support or a runtime admission certificate.",
    }
    if basic:
        report["collection_seconds"] = time.monotonic() - start
        return report
    cpu, errors = report["host"]["cpu_details"], report["host"]["cpu_query_errors"]
    if system == "Darwin":
        report["host"]["cpu_details_source"] = "sysctl allowlist; cache sizes are CPU caches"
        for key in SYSCTL_KEYS:
            value, error = command(["/usr/sbin/sysctl", "-n", key], timeout=2)
            cpu[key] = int(value) if value is not None and value.isdecimal() else value
            if error:
                errors[key] = error
        report["accelerators"]["metal"] = metal_probe()
        report["accelerators"]["cuda"] = {
            "status": "not_applicable",
            "reason": "CUDA probe supports Linux only",
        }
    elif system == "Linux":
        report["host"]["cpu_details_source"] = "lscpu JSON allowlist and /proc/meminfo"
        value, error = command(["lscpu", "--json"])
        if value:
            try:
                cpu.update(parse_lscpu(value))
            except (ValueError, TypeError):
                errors["lscpu"] = "invalid_json"
        elif error:
            errors["lscpu"] = error
        try:
            for line in Path("/proc/meminfo").read_text().splitlines():
                if line.startswith("MemTotal:"):
                    cpu["memory_total_bytes"] = int(line.split()[1]) * 1024
        except (OSError, ValueError):
            errors["memory_total_bytes"] = "unavailable"
        report["accelerators"]["cuda"] = cuda_probe()
        report["accelerators"]["metal"] = {"status": "not_applicable"}
    for tool in ("clang", "nvcc", "ncu"):
        if shutil.which(tool):
            value, error = command([tool, "--version"], timeout=3)
            # clang's later lines include its installation directory.
            lines = [
                line
                for line in (value or "").splitlines()
                if "version" in line.lower() or "release" in line.lower()
            ]
            report["software"]["tools"][tool] = {
                "version": "\n".join(lines)[:1200] or None,
                "error": error,
            }
    report["collection_seconds"] = time.monotonic() - start
    return report


def load_report(path):
    path = Path(path)
    if path.stat().st_size > 2_000_000:
        raise ValueError("hardware report exceeds 2 MB")
    value = json.loads(path.read_text())
    if (
        not isinstance(value, dict)
        or value.get("schema_version") != 1
        or any(not isinstance(value.get(k), dict) for k in ["host", "software", "accelerators"])
    ):
        raise ValueError("expected a lossless._native hardware report with schema_version=1")
    return value


def collect_bounded(timeout=5):
    """Try full device queries within one budget; keep missing facts explicit."""
    import signal

    basic = collect(basic=True)
    if timeout <= 0:
        basic["collection_error"] = "hardware query budget exhausted"
        return basic
    env = {k: v for k, v in os.environ.items() if k in {"PATH", "TMPDIR", "LANG", "SYSTEMROOT"}}
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[2])
    with tempfile.TemporaryDirectory(prefix="lossless-hardware-query-") as folder:
        output = Path(folder) / "profile.json"
        process = subprocess.Popen(
            [sys.executable, "-m", "lossless._native.hardware", "--output", str(output)],
            env=env,
            cwd=folder,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        try:
            process.wait(timeout=timeout)
            if process.returncode == 0:
                return load_report(output)
            basic["collection_error"] = f"query exited {process.returncode}"
        except subprocess.TimeoutExpired:
            basic["collection_error"] = "hardware query timeout; basic inventory only"
        finally:
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()
    return basic


def load_notes(path):
    """Validate provenance structure, not the truth/applicability of cited claims."""
    path = Path(path)
    if path.stat().st_size > 200_000:
        raise ValueError("hardware notes exceed 200 KB")
    value = json.loads(path.read_text())
    if (
        not isinstance(value, dict)
        or value.get("schema_version") != 1
        or not isinstance(value.get("entries"), list)
    ):
        raise ValueError("hardware notes need schema_version=1 and entries")
    for entry in value["entries"]:
        if not isinstance(entry, dict) or any(
            not isinstance(entry.get(k), str) or not entry[k].strip()
            for k in ("claim", "source_url", "applies_to", "retrieved_utc", "limitations")
        ):
            raise ValueError(
                "each hardware note needs claim, source_url, applies_to, retrieved_utc and limitations"
            )
        url = urlparse(entry["source_url"])
        if url.scheme not in ("http", "https") or not url.hostname or url.username or url.password:
            raise ValueError(
                "hardware note source_url must be a public HTTP(S) citation without credentials"
            )
        if entry.get("evidence_type") not in (
            "vendor_documentation",
            "upstream_source",
            "inference",
        ):
            raise ValueError("label hardware note evidence_type explicitly")
    return value


def emit_report(output=None, *, basic=False):
    # Reserve output before probing, so an existing report is never overwritten.
    if output is not None:
        with Path(output).open("x") as stream:
            json.dump(collect(basic=basic), stream, indent=2, allow_nan=False)
            stream.write("\n")
    else:
        print(json.dumps(collect(basic=basic), indent=2, allow_nan=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--basic", action="store_true", help="skip OS commands and native probes")
    args = parser.parse_args()
    emit_report(args.output, basic=args.basic)
