"""Small candidate library; every proposal still passes the frozen evaluator."""

from pathlib import Path


def native_candidates(operator):
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
        for stem in ("softmax_row_16", "softmax_column_4"):
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
