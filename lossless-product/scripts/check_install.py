"""Install only the built wheel into a new environment, outside the checkout."""

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
        run(["-I", "-m", "lossless", "demo", "--budget", "45s", "--output", "run"])
        run(["-I", "-m", "lossless", "export", "run", "--output", "artifact"])
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
