#include <stddef.h>
#include <string.h>
void kernel(const float *restrict x, const float *restrict r, const float *restrict w, float *restrict out, float *restrict inv, float *restrict px, float *restrict pr, size_t rows, size_t cols, size_t rs, size_t cs){for(size_t i=0;i<rows;++i)for(size_t j=0;j<cols;++j)memcpy(out+i*cols+j,x+i*rs+j*cs,sizeof(float));}
