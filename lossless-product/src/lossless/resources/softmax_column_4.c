#include <stddef.h>
#include <math.h>

void kernel(const float *restrict x, const float *restrict r, const float *restrict w, float *restrict out, float *restrict inv, float *restrict px, float *restrict pr, size_t rows, size_t cols, size_t rs, size_t cs){
        for(size_t start=0;start<rows;start+=4){size_t count=rows-start;if(count>4)count=4;
          float maxima[4],scales[4];double sums[4]={0};
          for(size_t q=0;q<count;++q)maxima[q]=-INFINITY;
          for(size_t j=0;j<cols;++j)for(size_t q=0;q<count;++q){float v=x[(start+q)*rs+j*cs];if(v>maxima[q])maxima[q]=v;}
          for(size_t j=0;j<cols;++j)for(size_t q=0;q<count;++q){float e=expf(x[(start+q)*rs+j*cs]-maxima[q]);out[(start+q)*cols+j]=e;sums[q]+=(double)e;}
          for(size_t q=0;q<count;++q)scales[q]=(float)(1.0/sums[q]);
          for(size_t j=0;j<cols;++j)for(size_t q=0;q<count;++q)out[(start+q)*cols+j]*=scales[q];
        }}
