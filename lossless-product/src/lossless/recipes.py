"""Small candidate library; every proposal still passes the frozen evaluator."""

from pathlib import Path
import platform


def catalog(*, contract=None, backend=None):
    """Describe executable retained mechanisms and their admission limits."""
    import json

    entries = json.loads((Path(__file__).parent / "retained/catalog.json").read_text())["recipes"]
    return [
        e
        for e in entries
        if (contract is None or e["contract"] == contract)
        and (backend is None or e["backend"] == backend)
    ]


def native_candidates(operator, cases=()):
    from ._native.operators import ABI, baseline_source

    resources = Path(__file__).parent / "resources"
    if operator == "copy":
        source = (
            "#include <stddef.h>\n#include <string.h>\nvoid kernel("
            + ABI
            + """){
            if(cs==1){for(size_t i=0;i<rows;++i)memcpy(out+i*cols,x+i*rs,cols*sizeof(float));}
            else {for(size_t i=0;i<rows;++i)for(size_t j=0;j<cols;++j)memcpy(out+i*cols+j,x+i*rs+j*cs,sizeof(float));}
        }\n"""
        )
        yield {
            "id": "contiguous_copy",
            "source": source,
            "hypothesis": "Use a bulk byte copy for contiguous rows; preserve exact bits and retain the strided path.",
        }
    elif operator == "softmax":
        # Campaign 046 is a numerical, large column-major specialization. It
        # must compete through the same evaluator, including fresh final cases.
        if (
            platform.machine() in {"arm64", "aarch64"}
            and cases
            and all(
                c["layout"] == "f"
                and c["rows"] >= 32
                and c["columns"] >= 128
                and c["rows"] * c["columns"] >= 16384
                for c in cases
            )
        ):
            yield {
                "id": "softmax_fortran_046",
                "source": (resources / "softmax_fortran_046.c").read_text(),
                "hypothesis": "Retained campaign-046 numerical specialization; requalify complete calls on this host.",
            }
        stems = ("softmax_row_16", "softmax_column_4")
        # Historical replay harnesses request the original pool without cases.
        # Product searches declare their cases and can requalify this lead.
        if cases:
            stems += ("softmax_dispatch_033",)
        for stem in stems:
            yield {
                "id": stem,
                "source": (resources / (stem + ".c")).read_text(),
                "hypothesis": "Retained native CPU candidate; compare its loop organization against the frozen numerical reference.",
            }
    else:
        source = (
            baseline_source(operator)
            .replace("sums[8]", "sums[16]")
            .replace("j+8<=cols;j+=8", "j+16<=cols;j+=16")
            .replace("k<8", "k<16")
        )
        yield {
            "id": "rmsnorm_unroll16",
            "source": source,
            "hypothesis": "Increase the independent accumulation groups while enforcing the original numerical contract.",
        }
        # Earlier layout-search gains were absent from the small alpha pool.
        # These remain numerical proposals, independently compiled and gated.
        retained = (resources / "rms_candidates_004.c").read_text()
        for name in ("column_g32", "column_g4", "column_g16", "column_g64", "row_u1", "row_u8"):
            yield {
                "id": "rms_004_" + name,
                "source": retained
                + "\nvoid kernel("
                + ABI
                + "){"
                + name
                + "(x,r,w,out,inv,px,pr,rows,cols,rs,cs);}\n",
                "hypothesis": "Retained RMS layout candidate from campaigns 002–005; measure complete calls under the numerical contract.",
            }
