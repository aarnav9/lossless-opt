"""Supply the frozen campaign-046 recipe through the normal proposal boundary.

No LLM or network request occurs. The evaluator still controls acceptance.
"""

import hashlib
import json
from pathlib import Path

SOURCE_SHA256 = "34d962e0ebe3e012ddba475f224d3570114fa4f3f05f8847adbef31e0cd3235d"


def complete(prompt):
    context = json.loads(prompt)
    if context.get("operator") != "softmax":
        raise ValueError("This recipe only supports native softmax")
    for case in context["discovery_cases"]:
        if not (
            case["layout"] == "f"
            and case["rows"] >= 32
            and case["columns"] >= 128
            and case["rows"] * case["columns"] >= 16384
        ):
            raise ValueError("Recipe qualification covers large Fortran-layout cases only")
    source = Path(__file__).with_name("kernel.c").read_bytes()
    if hashlib.sha256(source).hexdigest() != SOURCE_SHA256:
        raise ValueError("Frozen recipe source changed")
    return {
        "schema_version": 1,
        "proposals": [
            {
                "id": "softmax_fortran_046",
                "hypothesis": "Frozen campaign-045 first-round kernel, qualified for the campaign-046 large Fortran scope; independently remeasure on this host.",
                "source": source.decode(),
            }
        ],
        "usage": {"input_tokens": 0, "output_tokens": 0, "cost_usd": 0},
    }
