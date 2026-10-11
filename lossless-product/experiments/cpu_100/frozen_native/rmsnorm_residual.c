#include <stddef.h>
#include <math.h>
void kernel(const float *restrict x, const float *restrict r, const float *restrict w, float *restrict out, float *restrict inv, float *restrict px, float *restrict pr, size_t rows, size_t cols, size_t rs, size_t cs){
        for(size_t row=0;row<rows;++row){double sums[8]={0};size_t j=0;
          for(;j+8<=cols;j+=8)for(size_t k=0;k<8;++k){float v=x[row*rs+(j+k)*cs];float q=v*v;sums[k]+=(double)q;}
          for(;j<cols;++j){float v=x[row*rs+j*cs];float q=v*v;sums[0]+=(double)q;}
          double sum=0;for(size_t k=0;k<8;++k)sum+=sums[k];float scale=(float)(1.0/sqrt(sum/(double)cols+1e-5));
          for(j=0;j<cols;++j){float a=x[row*rs+j*cs]*scale;float b=a*w[j];out[row*cols+j]=b+r[row*rs+j*cs];}
        }}
