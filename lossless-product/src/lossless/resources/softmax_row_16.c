#include <stddef.h>
#include <math.h>

void kernel(const float *restrict x, const float *restrict r, const float *restrict w, float *restrict out, float *restrict inv, float *restrict px, float *restrict pr, size_t rows, size_t cols, size_t rs, size_t cs){
        for(size_t row=0;row<rows;++row){
          float maximum=-INFINITY;for(size_t j=0;j<cols;++j)if(x[row*rs+j*cs]>maximum)maximum=x[row*rs+j*cs];
          double sums[16]={0};size_t j=0;
          for(;j+16<=cols;j+=16)for(size_t k=0;k<16;++k){float e=expf(x[row*rs+(j+k)*cs]-maximum);out[row*cols+j+k]=e;sums[k]+=(double)e;}
          for(;j<cols;++j){float e=expf(x[row*rs+j*cs]-maximum);out[row*cols+j]=e;sums[0]+=(double)e;}
          double sum=0;for(size_t k=0;k<16;++k)sum+=sums[k];float scale=(float)(1.0/sum);
          for(j=0;j<cols;++j){out[row*cols+j]*=scale;}
        }}
