#include <stddef.h>
#include <stdint.h>
#include <math.h>
#include <float.h>
// Named reference: scalar IEEE float32 products/additions, no FMA, then double
// log-sum-exp and float32 normalized scores with first-index argmax ties.
int full_head(const float *w,const float *x,float *scores,size_t v,size_t d){
 float maximum=-INFINITY;
 for(size_t i=0;i<v;i++){float s=0;for(size_t j=0;j<d;j++){float p=w[i*d+j]*x[j];s=s+p;}scores[i]=s;if(s>maximum)maximum=s;}
 double sum=0;for(size_t i=0;i<v;i++)sum+=exp((double)scores[i]-maximum);
 double normalizer=(double)maximum+log(sum);float best=-INFINITY;int winner=0;
 for(size_t i=0;i<v;i++){float z=(float)((double)scores[i]-normalizer);if(z>best){best=z;winner=(int)i;}}
 return winner;
}
void suffixes(const float *w,double *suffix,size_t v,size_t d,size_t block){
 size_t nb=(d+block-1)/block;
 for(size_t i=0;i<v;i++){double s=0;suffix[i*(nb+1)+nb]=0;
  for(size_t j=d;j>0;){--j;s=nextafter(s+fabs((double)w[i*d+j]),INFINITY);if(j%block==0)suffix[i*(nb+1)+j/block]=s;}
 }
}
int certified_head(const float *w,const float *x,const double *suffix,float *partial,double *lo,double *hi,uint8_t *alive,float *scratch,uint64_t *stats,size_t v,size_t d,size_t block){
 size_t nb=(d+block-1)/block;stats[0]=stats[1]=0;
 double maxx=0;for(size_t j=0;j<d;j++)if(fabs((double)x[j])>maxx)maxx=fabs((double)x[j]);
 double totalbound=0;
 for(size_t i=0;i<v;i++){partial[i]=0;alive[i]=1;double b=nextafter(suffix[i*(nb+1)]*maxx,INFINITY);if(b>totalbound)totalbound=b;lo[i]=-b;hi[i]=b;}
 // Bound subtraction rounding in the named normalized-score reference. The
 // large logZ allowance includes double summation/transcendental roundoff for
 // these finite, bounded test domains; certify only gaps comfortably above it.
 double guard=nextafter(8*FLT_EPSILON*(2*totalbound+log((double)v)+2),INFINITY);
 for(size_t step=0;step<nb;step++){
  size_t begin=step*block,end=begin+block;if(end>d)end=d;
  double remaining_max=0;for(size_t j=end;j<d;j++)if(fabs((double)x[j])>remaining_max)remaining_max=fabs((double)x[j]);
  int winner=0;double lower=-INFINITY;
  for(size_t i=0;i<v;i++){
   if(alive[i]){float s=partial[i];for(size_t j=begin;j<end;j++){float p=w[i*d+j]*x[j];s=s+p;stats[0]++;}partial[i]=s;
    double remaining=nextafter(suffix[i*(nb+1)+step+1]*remaining_max,INFINITY);
    double nu=2*(d-end)*(FLT_EPSILON/2.0),gamma=nextafter(nu/(1-nu),INFINITY);
    double error=nextafter(gamma*(fabs((double)s)+remaining)+2*(d-end)*0x1p-149,INFINITY);
    lo[i]=nextafter((double)s-remaining-error,-INFINITY);hi[i]=nextafter((double)s+remaining+error,INFINITY);
   }
   if(lo[i]>lower){lower=lo[i];winner=(int)i;}
  }
  double upper=-INFINITY;for(size_t i=0;i<v;i++)if((int)i!=winner && hi[i]>upper)upper=hi[i];
  if(lower>upper+guard)return winner;
  for(size_t i=0;i<v;i++)if(hi[i]+guard<lower)alive[i]=0;
 }
 stats[1]=1;return full_head(w,x,scratch,v,d);
}
