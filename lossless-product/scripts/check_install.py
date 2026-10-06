"""Install only the built wheel into a new environment, outside the checkout."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import venv


def main(directory):
    wheel = next(Path(directory).resolve().glob("*.whl"))
    with tempfile.TemporaryDirectory(prefix="lossless-installed-check-") as folder:
        root = Path(folder)
        venv.create(root / "venv", with_pip=True)
        python = root / "venv/bin/python"
        env = {
            k: v
            for k, v in os.environ.items()
            if k not in {"PYTHONPATH", "PYTHONHOME", "LOSSLESS_TEST_MLX_MODEL"}
        }

        def run(args):
            subprocess.run([str(python), *args], cwd=root, env=env, check=True, timeout=90)

        run(["-m", "pip", "install", str(wheel)])
        run(["-I", "-m", "lossless", "doctor"])
        # Existing applications must work through the installed wheel, without
        # importing any source-checkout experiments or product helper code.
        application = root / "application"
        application.mkdir()
        (application / "app.py").write_text("def run(x):\n    return {'value': x * 3}\n")
        (root / "application-cases.json").write_text(
            '[{"id":"sample","split":"discovery","args":[7]}]\n'
        )
        run(
            [
                "-I",
                "-m",
                "lossless",
                "init",
                "--project",
                "application",
                "--entry",
                "app.py:run",
                "--cases",
                "application-cases.json",
                "--output",
                "application-job",
            ]
        )
        run(["-I", "-m", "lossless", "inspect", "application-job/lossless.json"])
        run(
            [
                "-I",
                "-m",
                "lossless",
                "replay",
                "application-job/lossless.json",
                "--output",
                "application-replay",
            ]
        )
        run(
            [
                "-I",
                "-c",
                "import json; from pathlib import Path; r=json.loads(Path('application-replay/report.json').read_text()); assert r['status']=='replayed' and r['llm_calls']==0 and r['source_unchanged']; print('Installed application replay passed')",
            ]
        )
        run(
            [
                "-I",
                "-m",
                "lossless",
                "profile",
                "application-job/lossless.json",
                "--repeats",
                "3",
                "--no-explain",
                "--output",
                "application-profile",
            ]
        )
        run(
            [
                "-I",
                "-c",
                "import json; from pathlib import Path; r=json.loads(Path('application-profile/report.json').read_text()); assert r['status']=='profiled' and r['llm_calls']==0 and len(r['profiles'])==1; print('Installed application profiling passed')",
            ]
        )
        # Exercise the installed candidate worker without requiring Codex or
        # accepting a noisy speed claim: an intentionally wrong body must fail.
        (root / "application-job/cases.json").write_text(
            json.dumps(
                [
                    {"id": "sample", "split": "discovery", "args": [7]},
                    {"id": "held_out", "split": "evaluation", "args": [9]},
                ]
            )
        )
        (root / "application-plan.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "source_scope": [{"path": "app.py", "symbol": "run"}],
                    "contract": "Exact integer output; reject the deliberately wrong candidate.",
                    "max_candidates": 1,
                    "wall_time_seconds": 30,
                    "final_reserve_seconds": 10,
                    "candidates": [
                        {
                            "schema_version": 1,
                            "hypothesis": "Negative installation control",
                            "edits": [
                                {
                                    "path": "app.py",
                                    "symbol": "run",
                                    "body": "return {'value': x * 3 + 1}\n",
                                }
                            ],
                        }
                    ],
                }
            )
        )
        run(
            [
                "-I",
                "-m",
                "lossless",
                "search-application",
                "application-job/lossless.json",
                "--plan",
                "application-plan.json",
                "--output",
                "application-search",
            ]
        )
        run(
            [
                "-I",
                "-c",
                "import json; from pathlib import Path; r=json.loads(Path('application-search/report.json').read_text()); assert r['status']=='reference_retained' and r['selected']=='reference' and r['llm_calls']==0 and r['candidates'][0]['status']=='mismatch'; assert not Path('application-search/evaluation').exists(); print('Installed application source search passed')",
            ]
        )
        run(["-I", "-m", "lossless", "demo", "--budget", "45s", "--output", "run"])
        run(["-I", "-m", "lossless", "export", "run", "--output", "artifact"])
        run(
            [
                "-I",
                "-m",
                "lossless",
                "profile",
                "run/resolved.json",
                "--artifact",
                "artifact",
                "--repeats",
                "3",
                "--budget",
                "30s",
                "--output",
                "profile",
            ]
        )
        run(
            [
                "-I",
                "-c",
                "import json; from pathlib import Path; r=json.loads(Path('profile/report.json').read_text()); assert r['status']=='profiled' and r['llm_calls']==0 and all(c['case']['split']=='discovery' for c in r['cases']); print('Installed profile passed')",
            ]
        )
        run(["-I", "-m", "lossless", "proofs", "search", "matrix transpose", "--limit", "1"])
        run(
            [
                "-I",
                "-c",
                "import lossless,numpy as np; from pathlib import Path; "
                "assert Path(lossless.__file__).is_relative_to(Path.cwd()/'venv'); "
                "operation=lossless.load('artifact'); "
                "x=np.resize(np.array([0,0x80000000,0x7f800001,0x7fc01234],dtype=np.uint32),(64,257)).view(np.float32); "
                "y=operation(x); assert y.tobytes()==x.tobytes() and not np.shares_memory(x,y); "
                "print('Installed exact export/load passed')",
            ]
        )


if __name__ == "__main__":
    main(sys.argv[1])
