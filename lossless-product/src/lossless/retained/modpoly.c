#include <math.h>
typedef void (*gemv_fn)(int,int,long long,long long,double,const double*,long long,const double*,long long,double,double*,long long);
typedef double (*dot_fn)(long long,const double*,long long,const double*,long long);
int iterate(void *gemv_address,void *dot_address,const double *source,
            double *b0,double *b1,const double *weight,double *clipped,double *scratch,
            const double *pinv,const double *vander,double *coef,double *history,
            long long n,long long order,long long pinv_ld,long long vander_ld,
            int pinv_layout,int vander_layout,int unit,int maximum,double tol,double minimum_norm) {
    gemv_fn gemv=(gemv_fn)gemv_address; dot_fn dot=(dot_fn)dot_address;
    double *old=b0,*next=b1;
    for(int it=0;it<maximum;it++) {
        if(unit) {
            for(long long j=0;j<n;j++)clipped[j]=source[j]<old[j]?source[j]:old[j];
        } else {
            for(long long j=0;j<n;j++){double v=source[j]<old[j]?source[j]:old[j];clipped[j]=v;scratch[j]=weight[j]*v;}
        }
        gemv(pinv_layout,111,order,n,1.0,pinv,pinv_ld,unit?clipped:scratch,1,0.0,coef,1);
        gemv(vander_layout,111,n,order,1.0,vander,vander_ld,coef,1,0.0,next,1);
        for(long long j=0;j<n;j++)scratch[j]=next[j]-old[j];
        double numerator=sqrt(dot(n,scratch,1,scratch,1));
        double denominator=sqrt(dot(n,old,1,old,1));
        if(denominator<minimum_norm)denominator=minimum_norm;
        double change=numerator/denominator;
        history[it]=change;
        if(change<tol)return it+1;
        double *swap=old;old=next;next=swap;
    }
    return maximum;
}
