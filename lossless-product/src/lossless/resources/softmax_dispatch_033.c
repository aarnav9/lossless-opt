#include <stddef.h>
#include <math.h>
static void tile4(const float *restrict x, const float *restrict r, const float *restrict w, float *restrict out, float *restrict inv, float *restrict px, float *restrict pr, size_t rows, size_t cols, size_t rs, size_t cs){
 size_t start=0;
 for(;start+4<=rows;start+=4){
   float maxima[4];double sums[4]={0};float scales[4];
   for(size_t q=0;q<4;++q)maxima[q]=-INFINITY;
   for(size_t j=0;j<cols;++j)for(size_t q=0;q<4;++q){float v=x[(start+q)*rs+j*cs];if(v>maxima[q])maxima[q]=v;}
   for(size_t j=0;j<cols;++j)for(size_t q=0;q<4;++q){float e=expf(x[(start+q)*rs+j*cs]-maxima[q]);out[(start+q)*cols+j]=e;sums[q]+=e;}
   for(size_t q=0;q<4;++q)scales[q]=(float)(1.0/sums[q]);
   for(size_t j=0;j<cols;++j)for(size_t q=0;q<4;++q)out[(start+q)*cols+j]*=scales[q];
 }
 for(;start<rows;++start){float maximum=-INFINITY;double sum=0;
  for(size_t j=0;j<cols;++j){float v=x[start*rs+j*cs];if(v>maximum)maximum=v;}
  for(size_t j=0;j<cols;++j){float e=expf(x[start*rs+j*cs]-maximum);out[start*cols+j]=e;sum+=e;}
  float scale=(float)(1.0/sum);for(size_t j=0;j<cols;++j)out[start*cols+j]*=scale;
 }
}
#include <stddef.h>
#include <math.h>
static void tile8(const float *restrict x, const float *restrict r, const float *restrict w, float *restrict out, float *restrict inv, float *restrict px, float *restrict pr, size_t rows, size_t cols, size_t rs, size_t cs){
 size_t start=0;
 for(;start+8<=rows;start+=8){
   float maxima[8];double sums[8]={0};float scales[8];
   for(size_t q=0;q<8;++q)maxima[q]=-INFINITY;
   for(size_t j=0;j<cols;++j)for(size_t q=0;q<8;++q){float v=x[(start+q)*rs+j*cs];if(v>maxima[q])maxima[q]=v;}
   for(size_t j=0;j<cols;++j)for(size_t q=0;q<8;++q){float e=expf(x[(start+q)*rs+j*cs]-maxima[q]);out[(start+q)*cols+j]=e;sums[q]+=e;}
   for(size_t q=0;q<8;++q)scales[q]=(float)(1.0/sums[q]);
   for(size_t j=0;j<cols;++j)for(size_t q=0;q<8;++q)out[(start+q)*cols+j]*=scales[q];
 }
 for(;start<rows;++start){float maximum=-INFINITY;double sum=0;
  for(size_t j=0;j<cols;++j){float v=x[start*rs+j*cs];if(v>maximum)maximum=v;}
  for(size_t j=0;j<cols;++j){float e=expf(x[start*rs+j*cs]-maximum);out[start*cols+j]=e;sum+=e;}
  float scale=(float)(1.0/sum);for(size_t j=0;j<cols;++j)out[start*cols+j]*=scale;
 }
}
void kernel(const float *restrict x, const float *restrict r, const float *restrict w, float *restrict out, float *restrict inv, float *restrict px, float *restrict pr, size_t rows, size_t cols, size_t rs, size_t cs){if(rs==1)tile8(x,r,w,out,inv,px,pr,rows,cols,rs,cs);else tile4(x,r,w,out,inv,px,pr,rows,cols,rs,cs);}
