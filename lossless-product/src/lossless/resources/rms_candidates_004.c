#include <stddef.h>
#include <math.h>


void row_u1(const float *restrict x, const float *restrict r, const float *restrict w, float *restrict out, float *restrict inv, float *restrict px, float *restrict pr, size_t rows, size_t cols, size_t rs, size_t cs) {
  for (size_t row=0; row<rows; ++row) {
    const float *xr=x+row*rs; const float *rr=r+row*rs;
    float *yr=out+row*cols; double s0=0.0;
    size_t j=0;
    for (;j+1<=cols;j+=1) { float v0=xr[(j+0)*cs]; float q0=v0*v0; s0+=(double)q0; }
    for (;j<cols;++j) { float v=xr[j*cs]; float q=v*v; s0+=(double)q; }
    float scale=(float)(1.0/sqrt((s0)/(double)cols+1e-05)); inv[row]=scale;
    for (j=0;j<cols;++j) { float a=xr[j*cs]*scale; float b=a*w[j]; yr[j]=b+rr[j*cs]; }
  }
}


void row_u8(const float *restrict x, const float *restrict r, const float *restrict w, float *restrict out, float *restrict inv, float *restrict px, float *restrict pr, size_t rows, size_t cols, size_t rs, size_t cs) {
  for (size_t row=0; row<rows; ++row) {
    const float *xr=x+row*rs; const float *rr=r+row*rs;
    float *yr=out+row*cols; double s0=0.0; double s1=0.0; double s2=0.0; double s3=0.0; double s4=0.0; double s5=0.0; double s6=0.0; double s7=0.0;
    size_t j=0;
    for (;j+8<=cols;j+=8) { float v0=xr[(j+0)*cs]; float q0=v0*v0; s0+=(double)q0; float v1=xr[(j+1)*cs]; float q1=v1*v1; s1+=(double)q1; float v2=xr[(j+2)*cs]; float q2=v2*v2; s2+=(double)q2; float v3=xr[(j+3)*cs]; float q3=v3*v3; s3+=(double)q3; float v4=xr[(j+4)*cs]; float q4=v4*v4; s4+=(double)q4; float v5=xr[(j+5)*cs]; float q5=v5*v5; s5+=(double)q5; float v6=xr[(j+6)*cs]; float q6=v6*v6; s6+=(double)q6; float v7=xr[(j+7)*cs]; float q7=v7*v7; s7+=(double)q7; }
    for (;j<cols;++j) { float v=xr[j*cs]; float q=v*v; s0+=(double)q; }
    float scale=(float)(1.0/sqrt((s0+s1+s2+s3+s4+s5+s6+s7)/(double)cols+1e-05)); inv[row]=scale;
    for (j=0;j<cols;++j) { float a=xr[j*cs]*scale; float b=a*w[j]; yr[j]=b+rr[j*cs]; }
  }
}


void column_g4(const float *restrict x, const float *restrict r, const float *restrict w, float *restrict out, float *restrict inv, float *restrict px, float *restrict pr, size_t rows, size_t cols, size_t rs, size_t cs) {
  for (size_t start=0;start<rows;start+=4) {
    size_t count=rows-start; if(count>4) count=4;
    double sums[4]={0};
    for (size_t j=0;j<cols;++j) {
      for (size_t q=0;q<count;++q) { size_t row=start+q;
        float v=x[row*rs+j*cs]; float square=v*v; sums[q]+=(double)square; 
      }
    }
    for (size_t q=0;q<count;++q) inv[start+q]=(float)(1.0/sqrt(sums[q]/(double)cols+1e-05));
    
    for (size_t j=0;j<cols;++j) {
      for (size_t q=0;q<count;++q) { size_t row=start+q;
        float a=x[row*rs+j*cs]*inv[row]; float b=a*w[j]; out[row*cols+j]=b+r[row*rs+j*cs]; }
    }

  }
}


void column_g16(const float *restrict x, const float *restrict r, const float *restrict w, float *restrict out, float *restrict inv, float *restrict px, float *restrict pr, size_t rows, size_t cols, size_t rs, size_t cs) {
  for (size_t start=0;start<rows;start+=16) {
    size_t count=rows-start; if(count>16) count=16;
    double sums[16]={0};
    for (size_t j=0;j<cols;++j) {
      for (size_t q=0;q<count;++q) { size_t row=start+q;
        float v=x[row*rs+j*cs]; float square=v*v; sums[q]+=(double)square; 
      }
    }
    for (size_t q=0;q<count;++q) inv[start+q]=(float)(1.0/sqrt(sums[q]/(double)cols+1e-05));
    
    for (size_t j=0;j<cols;++j) {
      for (size_t q=0;q<count;++q) { size_t row=start+q;
        float a=x[row*rs+j*cs]*inv[row]; float b=a*w[j]; out[row*cols+j]=b+r[row*rs+j*cs]; }
    }

  }
}


void column_g64(const float *restrict x, const float *restrict r, const float *restrict w, float *restrict out, float *restrict inv, float *restrict px, float *restrict pr, size_t rows, size_t cols, size_t rs, size_t cs) {
  for (size_t start=0;start<rows;start+=64) {
    size_t count=rows-start; if(count>64) count=64;
    double sums[64]={0};
    for (size_t j=0;j<cols;++j) {
      for (size_t q=0;q<count;++q) { size_t row=start+q;
        float v=x[row*rs+j*cs]; float square=v*v; sums[q]+=(double)square; 
      }
    }
    for (size_t q=0;q<count;++q) inv[start+q]=(float)(1.0/sqrt(sums[q]/(double)cols+1e-05));
    
    for (size_t j=0;j<cols;++j) {
      for (size_t q=0;q<count;++q) { size_t row=start+q;
        float a=x[row*rs+j*cs]*inv[row]; float b=a*w[j]; out[row*cols+j]=b+r[row*rs+j*cs]; }
    }

  }
}


void column_g128(const float *restrict x, const float *restrict r, const float *restrict w, float *restrict out, float *restrict inv, float *restrict px, float *restrict pr, size_t rows, size_t cols, size_t rs, size_t cs) {
  for (size_t start=0;start<rows;start+=128) {
    size_t count=rows-start; if(count>128) count=128;
    double sums[128]={0};
    for (size_t j=0;j<cols;++j) {
      for (size_t q=0;q<count;++q) { size_t row=start+q;
        float v=x[row*rs+j*cs]; float square=v*v; sums[q]+=(double)square; 
      }
    }
    for (size_t q=0;q<count;++q) inv[start+q]=(float)(1.0/sqrt(sums[q]/(double)cols+1e-05));
    
    for (size_t j=0;j<cols;++j) {
      for (size_t q=0;q<count;++q) { size_t row=start+q;
        float a=x[row*rs+j*cs]*inv[row]; float b=a*w[j]; out[row*cols+j]=b+r[row*rs+j*cs]; }
    }

  }
}


void staged_g16(const float *restrict x, const float *restrict r, const float *restrict w, float *restrict out, float *restrict inv, float *restrict px, float *restrict pr, size_t rows, size_t cols, size_t rs, size_t cs) {
  for (size_t start=0;start<rows;start+=16) {
    size_t count=rows-start; if(count>16) count=16;
    double sums[16]={0};
    for (size_t j=0;j<cols;++j) {
      for (size_t q=0;q<count;++q) { size_t row=start+q;
        float v=x[row*rs+j*cs]; float square=v*v; sums[q]+=(double)square; out[row*cols+j]=v; pr[row*cols+j]=r[row*rs+j*cs];
      }
    }
    for (size_t q=0;q<count;++q) inv[start+q]=(float)(1.0/sqrt(sums[q]/(double)cols+1e-05));
    
    for (size_t q=0;q<count;++q) {
      size_t row=start+q; float scale=inv[row];
      for (size_t j=0;j<cols;++j) { size_t at=row*cols+j;
        float a=out[at]*scale; float b=a*w[j]; out[at]=b+pr[at]; }
    }

  }
}


void staged_g64(const float *restrict x, const float *restrict r, const float *restrict w, float *restrict out, float *restrict inv, float *restrict px, float *restrict pr, size_t rows, size_t cols, size_t rs, size_t cs) {
  for (size_t start=0;start<rows;start+=64) {
    size_t count=rows-start; if(count>64) count=64;
    double sums[64]={0};
    for (size_t j=0;j<cols;++j) {
      for (size_t q=0;q<count;++q) { size_t row=start+q;
        float v=x[row*rs+j*cs]; float square=v*v; sums[q]+=(double)square; out[row*cols+j]=v; pr[row*cols+j]=r[row*rs+j*cs];
      }
    }
    for (size_t q=0;q<count;++q) inv[start+q]=(float)(1.0/sqrt(sums[q]/(double)cols+1e-05));
    
    for (size_t q=0;q<count;++q) {
      size_t row=start+q; float scale=inv[row];
      for (size_t j=0;j<cols;++j) { size_t at=row*cols+j;
        float a=out[at]*scale; float b=a*w[j]; out[at]=b+pr[at]; }
    }

  }
}


void pack_row_u8(const float *restrict x, const float *restrict r, const float *restrict w, float *restrict out, float *restrict inv, float *restrict px, float *restrict pr, size_t rows, size_t cols, size_t rs, size_t cs) {
  if (cs<=rs) {
    for (size_t row=0;row<rows;++row) for(size_t j=0;j<cols;++j) {
      px[row*cols+j]=x[row*rs+j*cs]; pr[row*cols+j]=r[row*rs+j*cs];
    }
  } else {
    for(size_t start=0;start<rows;start+=32) { size_t end=start+32; if(end>rows)end=rows;
      for(size_t j=0;j<cols;++j) for(size_t row=start;row<end;++row) {
        px[row*cols+j]=x[row*rs+j*cs]; pr[row*cols+j]=r[row*rs+j*cs];
      }
    }
  }
  row_u8(px,pr,w,out,inv,(float*)0,(float*)0,rows,cols,cols,1);
}

void row_u16(const float *restrict x, const float *restrict r, const float *restrict w, float *restrict out, float *restrict inv, float *restrict px, float *restrict pr, size_t rows, size_t cols, size_t rs, size_t cs) {

  for(size_t row=0;row<rows;++row) {
    double s0=0; double s1=0; double s2=0; double s3=0; double s4=0; double s5=0; double s6=0; double s7=0; double s8=0; double s9=0; double s10=0; double s11=0; double s12=0; double s13=0; double s14=0; double s15=0; size_t j=0;
    for(;j+16<=cols;j+=16) { float v0=x[row*rs+(j+0)*cs]; float q0=v0*v0; s0+=(double)q0; float v1=x[row*rs+(j+1)*cs]; float q1=v1*v1; s1+=(double)q1; float v2=x[row*rs+(j+2)*cs]; float q2=v2*v2; s2+=(double)q2; float v3=x[row*rs+(j+3)*cs]; float q3=v3*v3; s3+=(double)q3; float v4=x[row*rs+(j+4)*cs]; float q4=v4*v4; s4+=(double)q4; float v5=x[row*rs+(j+5)*cs]; float q5=v5*v5; s5+=(double)q5; float v6=x[row*rs+(j+6)*cs]; float q6=v6*v6; s6+=(double)q6; float v7=x[row*rs+(j+7)*cs]; float q7=v7*v7; s7+=(double)q7; float v8=x[row*rs+(j+8)*cs]; float q8=v8*v8; s8+=(double)q8; float v9=x[row*rs+(j+9)*cs]; float q9=v9*v9; s9+=(double)q9; float v10=x[row*rs+(j+10)*cs]; float q10=v10*v10; s10+=(double)q10; float v11=x[row*rs+(j+11)*cs]; float q11=v11*v11; s11+=(double)q11; float v12=x[row*rs+(j+12)*cs]; float q12=v12*v12; s12+=(double)q12; float v13=x[row*rs+(j+13)*cs]; float q13=v13*v13; s13+=(double)q13; float v14=x[row*rs+(j+14)*cs]; float q14=v14*v14; s14+=(double)q14; float v15=x[row*rs+(j+15)*cs]; float q15=v15*v15; s15+=(double)q15; }
    for(;j<cols;++j) { float v=x[row*rs+j*cs]; float q=v*v; s0+=(double)q; }
    float scale=(float)(1.0/sqrt((s0+s1+s2+s3+s4+s5+s6+s7+s8+s9+s10+s11+s12+s13+s14+s15)/(double)cols+1e-05)); inv[row]=scale;
    for(j=0;j<cols;++j) { float a=x[row*rs+j*cs]*scale; float b=a*w[j]; out[row*cols+j]=b+r[row*rs+j*cs]; }
  }
}

void row_u32(const float *restrict x, const float *restrict r, const float *restrict w, float *restrict out, float *restrict inv, float *restrict px, float *restrict pr, size_t rows, size_t cols, size_t rs, size_t cs) {

  for(size_t row=0;row<rows;++row) {
    double s0=0; double s1=0; double s2=0; double s3=0; double s4=0; double s5=0; double s6=0; double s7=0; double s8=0; double s9=0; double s10=0; double s11=0; double s12=0; double s13=0; double s14=0; double s15=0; double s16=0; double s17=0; double s18=0; double s19=0; double s20=0; double s21=0; double s22=0; double s23=0; double s24=0; double s25=0; double s26=0; double s27=0; double s28=0; double s29=0; double s30=0; double s31=0; size_t j=0;
    for(;j+32<=cols;j+=32) { float v0=x[row*rs+(j+0)*cs]; float q0=v0*v0; s0+=(double)q0; float v1=x[row*rs+(j+1)*cs]; float q1=v1*v1; s1+=(double)q1; float v2=x[row*rs+(j+2)*cs]; float q2=v2*v2; s2+=(double)q2; float v3=x[row*rs+(j+3)*cs]; float q3=v3*v3; s3+=(double)q3; float v4=x[row*rs+(j+4)*cs]; float q4=v4*v4; s4+=(double)q4; float v5=x[row*rs+(j+5)*cs]; float q5=v5*v5; s5+=(double)q5; float v6=x[row*rs+(j+6)*cs]; float q6=v6*v6; s6+=(double)q6; float v7=x[row*rs+(j+7)*cs]; float q7=v7*v7; s7+=(double)q7; float v8=x[row*rs+(j+8)*cs]; float q8=v8*v8; s8+=(double)q8; float v9=x[row*rs+(j+9)*cs]; float q9=v9*v9; s9+=(double)q9; float v10=x[row*rs+(j+10)*cs]; float q10=v10*v10; s10+=(double)q10; float v11=x[row*rs+(j+11)*cs]; float q11=v11*v11; s11+=(double)q11; float v12=x[row*rs+(j+12)*cs]; float q12=v12*v12; s12+=(double)q12; float v13=x[row*rs+(j+13)*cs]; float q13=v13*v13; s13+=(double)q13; float v14=x[row*rs+(j+14)*cs]; float q14=v14*v14; s14+=(double)q14; float v15=x[row*rs+(j+15)*cs]; float q15=v15*v15; s15+=(double)q15; float v16=x[row*rs+(j+16)*cs]; float q16=v16*v16; s16+=(double)q16; float v17=x[row*rs+(j+17)*cs]; float q17=v17*v17; s17+=(double)q17; float v18=x[row*rs+(j+18)*cs]; float q18=v18*v18; s18+=(double)q18; float v19=x[row*rs+(j+19)*cs]; float q19=v19*v19; s19+=(double)q19; float v20=x[row*rs+(j+20)*cs]; float q20=v20*v20; s20+=(double)q20; float v21=x[row*rs+(j+21)*cs]; float q21=v21*v21; s21+=(double)q21; float v22=x[row*rs+(j+22)*cs]; float q22=v22*v22; s22+=(double)q22; float v23=x[row*rs+(j+23)*cs]; float q23=v23*v23; s23+=(double)q23; float v24=x[row*rs+(j+24)*cs]; float q24=v24*v24; s24+=(double)q24; float v25=x[row*rs+(j+25)*cs]; float q25=v25*v25; s25+=(double)q25; float v26=x[row*rs+(j+26)*cs]; float q26=v26*v26; s26+=(double)q26; float v27=x[row*rs+(j+27)*cs]; float q27=v27*v27; s27+=(double)q27; float v28=x[row*rs+(j+28)*cs]; float q28=v28*v28; s28+=(double)q28; float v29=x[row*rs+(j+29)*cs]; float q29=v29*v29; s29+=(double)q29; float v30=x[row*rs+(j+30)*cs]; float q30=v30*v30; s30+=(double)q30; float v31=x[row*rs+(j+31)*cs]; float q31=v31*v31; s31+=(double)q31; }
    for(;j<cols;++j) { float v=x[row*rs+j*cs]; float q=v*v; s0+=(double)q; }
    float scale=(float)(1.0/sqrt((s0+s1+s2+s3+s4+s5+s6+s7+s8+s9+s10+s11+s12+s13+s14+s15+s16+s17+s18+s19+s20+s21+s22+s23+s24+s25+s26+s27+s28+s29+s30+s31)/(double)cols+1e-05)); inv[row]=scale;
    for(j=0;j<cols;++j) { float a=x[row*rs+j*cs]*scale; float b=a*w[j]; out[row*cols+j]=b+r[row*rs+j*cs]; }
  }
}

void column_g8(const float *restrict x, const float *restrict r, const float *restrict w, float *restrict out, float *restrict inv, float *restrict px, float *restrict pr, size_t rows, size_t cols, size_t rs, size_t cs) {

  for(size_t start=0;start<rows;start+=8) {
    size_t count=rows-start; if(count>8)count=8; double sums[8]={0};
    for(size_t j=0;j<cols;++j) for(size_t q=0;q<count;++q) {
      float v=x[(start+q)*rs+j*cs]; float square=v*v; sums[q]+=(double)square;
    }
    for(size_t q=0;q<count;++q)inv[start+q]=(float)(1.0/sqrt(sums[q]/(double)cols+1e-05));
    for(size_t j=0;j<cols;++j) for(size_t q=0;q<count;++q) { size_t row=start+q;
              float a=x[row*rs+j*cs]*inv[row]; float b=a*w[j]; out[row*cols+j]=b+r[row*rs+j*cs]; }
  }
}

void column_g32(const float *restrict x, const float *restrict r, const float *restrict w, float *restrict out, float *restrict inv, float *restrict px, float *restrict pr, size_t rows, size_t cols, size_t rs, size_t cs) {

  for(size_t start=0;start<rows;start+=32) {
    size_t count=rows-start; if(count>32)count=32; double sums[32]={0};
    for(size_t j=0;j<cols;++j) for(size_t q=0;q<count;++q) {
      float v=x[(start+q)*rs+j*cs]; float square=v*v; sums[q]+=(double)square;
    }
    for(size_t q=0;q<count;++q)inv[start+q]=(float)(1.0/sqrt(sums[q]/(double)cols+1e-05));
    for(size_t j=0;j<cols;++j) for(size_t q=0;q<count;++q) { size_t row=start+q;
              float a=x[row*rs+j*cs]*inv[row]; float b=a*w[j]; out[row*cols+j]=b+r[row*rs+j*cs]; }
  }
}

void hybrid_g16(const float *restrict x, const float *restrict r, const float *restrict w, float *restrict out, float *restrict inv, float *restrict px, float *restrict pr, size_t rows, size_t cols, size_t rs, size_t cs) {

  for(size_t start=0;start<rows;start+=16) {
    size_t count=rows-start; if(count>16)count=16; double sums[16]={0};
    for(size_t j=0;j<cols;++j) for(size_t q=0;q<count;++q) {
      float v=x[(start+q)*rs+j*cs]; float square=v*v; sums[q]+=(double)square;
    }
    for(size_t q=0;q<count;++q)inv[start+q]=(float)(1.0/sqrt(sums[q]/(double)cols+1e-05));
    for(size_t q=0;q<count;++q) { size_t row=start+q; float scale=inv[row];
                  for(size_t j=0;j<cols;++j) { float a=x[row*rs+j*cs]*scale; float b=a*w[j]; out[row*cols+j]=b+r[row*rs+j*cs]; }}
  }
}

void hybrid_g64(const float *restrict x, const float *restrict r, const float *restrict w, float *restrict out, float *restrict inv, float *restrict px, float *restrict pr, size_t rows, size_t cols, size_t rs, size_t cs) {

  for(size_t start=0;start<rows;start+=64) {
    size_t count=rows-start; if(count>64)count=64; double sums[64]={0};
    for(size_t j=0;j<cols;++j) for(size_t q=0;q<count;++q) {
      float v=x[(start+q)*rs+j*cs]; float square=v*v; sums[q]+=(double)square;
    }
    for(size_t q=0;q<count;++q)inv[start+q]=(float)(1.0/sqrt(sums[q]/(double)cols+1e-05));
    for(size_t q=0;q<count;++q) { size_t row=start+q; float scale=inv[row];
                  for(size_t j=0;j<cols;++j) { float a=x[row*rs+j*cs]*scale; float b=a*w[j]; out[row*cols+j]=b+r[row*rs+j*cs]; }}
  }
}

void tile_pack_16x16(const float *restrict x, const float *restrict r, const float *restrict w, float *restrict out, float *restrict inv, float *restrict px, float *restrict pr, size_t rows, size_t cols, size_t rs, size_t cs) {

  for(size_t i=0;i<rows;i+=16) { size_t endr=i+16;if(endr>rows)endr=rows;
    for(size_t j=0;j<cols;j+=16) { size_t endc=j+16;if(endc>cols)endc=cols;
      if(rs<cs) {
        for(size_t col=j;col<endc;++col)for(size_t row=i;row<endr;++row) {
          px[row*cols+col]=x[row*rs+col*cs]; pr[row*cols+col]=r[row*rs+col*cs];
        }
      } else {
        for(size_t row=i;row<endr;++row)for(size_t col=j;col<endc;++col) {
          px[row*cols+col]=x[row*rs+col*cs]; pr[row*cols+col]=r[row*rs+col*cs];
        }
      }
    }
  }
  row_u8(px,pr,w,out,inv,(float*)0,(float*)0,rows,cols,cols,1);
}

void tile_pack_32x32(const float *restrict x, const float *restrict r, const float *restrict w, float *restrict out, float *restrict inv, float *restrict px, float *restrict pr, size_t rows, size_t cols, size_t rs, size_t cs) {

  for(size_t i=0;i<rows;i+=32) { size_t endr=i+32;if(endr>rows)endr=rows;
    for(size_t j=0;j<cols;j+=32) { size_t endc=j+32;if(endc>cols)endc=cols;
      if(rs<cs) {
        for(size_t col=j;col<endc;++col)for(size_t row=i;row<endr;++row) {
          px[row*cols+col]=x[row*rs+col*cs]; pr[row*cols+col]=r[row*rs+col*cs];
        }
      } else {
        for(size_t row=i;row<endr;++row)for(size_t col=j;col<endc;++col) {
          px[row*cols+col]=x[row*rs+col*cs]; pr[row*cols+col]=r[row*rs+col*cs];
        }
      }
    }
  }
  row_u8(px,pr,w,out,inv,(float*)0,(float*)0,rows,cols,cols,1);
}

void tile_pack_64x64(const float *restrict x, const float *restrict r, const float *restrict w, float *restrict out, float *restrict inv, float *restrict px, float *restrict pr, size_t rows, size_t cols, size_t rs, size_t cs) {

  for(size_t i=0;i<rows;i+=64) { size_t endr=i+64;if(endr>rows)endr=rows;
    for(size_t j=0;j<cols;j+=64) { size_t endc=j+64;if(endc>cols)endc=cols;
      if(rs<cs) {
        for(size_t col=j;col<endc;++col)for(size_t row=i;row<endr;++row) {
          px[row*cols+col]=x[row*rs+col*cs]; pr[row*cols+col]=r[row*rs+col*cs];
        }
      } else {
        for(size_t row=i;row<endr;++row)for(size_t col=j;col<endc;++col) {
          px[row*cols+col]=x[row*rs+col*cs]; pr[row*cols+col]=r[row*rs+col*cs];
        }
      }
    }
  }
  row_u8(px,pr,w,out,inv,(float*)0,(float*)0,rows,cols,cols,1);
}

void tile_pack_128x32(const float *restrict x, const float *restrict r, const float *restrict w, float *restrict out, float *restrict inv, float *restrict px, float *restrict pr, size_t rows, size_t cols, size_t rs, size_t cs) {

  for(size_t i=0;i<rows;i+=128) { size_t endr=i+128;if(endr>rows)endr=rows;
    for(size_t j=0;j<cols;j+=32) { size_t endc=j+32;if(endc>cols)endc=cols;
      if(rs<cs) {
        for(size_t col=j;col<endc;++col)for(size_t row=i;row<endr;++row) {
          px[row*cols+col]=x[row*rs+col*cs]; pr[row*cols+col]=r[row*rs+col*cs];
        }
      } else {
        for(size_t row=i;row<endr;++row)for(size_t col=j;col<endc;++col) {
          px[row*cols+col]=x[row*rs+col*cs]; pr[row*cols+col]=r[row*rs+col*cs];
        }
      }
    }
  }
  row_u8(px,pr,w,out,inv,(float*)0,(float*)0,rows,cols,cols,1);
}

void tile_stage_16x32(const float *restrict x, const float *restrict r, const float *restrict w, float *restrict out, float *restrict inv, float *restrict px, float *restrict pr, size_t rows, size_t cols, size_t rs, size_t cs) {

  float tilex[16][32], tiler[16][32];
  for(size_t i=0;i<rows;i+=16) { size_t nr=rows-i;if(nr>16)nr=16;
    double sums[16][4]={0};
    for(size_t j=0;j<cols;j+=32) { size_t nc=cols-j;if(nc>32)nc=32;
      for(size_t col=0;col<nc;++col)for(size_t q=0;q<nr;++q) {
        tilex[q][col]=x[(i+q)*rs+(j+col)*cs]; tiler[q][col]=r[(i+q)*rs+(j+col)*cs];
      }
      for(size_t q=0;q<nr;++q) { size_t col=0;
        for(;col+4<=nc;col+=4) {
          for(size_t k=0;k<4;++k) { float v=tilex[q][col+k]; float square=v*v;
            sums[q][k]+=(double)square; out[(i+q)*cols+j+col+k]=v;
            pr[(i+q)*cols+j+col+k]=tiler[q][col+k]; }
        }
        for(;col<nc;++col) {float v=tilex[q][col];float square=v*v;sums[q][0]+=(double)square;
          out[(i+q)*cols+j+col]=v;pr[(i+q)*cols+j+col]=tiler[q][col];}
      }
    }
    for(size_t q=0;q<nr;++q) {size_t row=i+q;
      float scale=(float)(1.0/sqrt((sums[q][0]+sums[q][1]+sums[q][2]+sums[q][3])/(double)cols+1e-05));inv[row]=scale;
      for(size_t j=0;j<cols;++j) {size_t at=row*cols+j;float a=out[at]*scale;float b=a*w[j];out[at]=b+pr[at];}
    }
  }
}

void tile_stage_32x32(const float *restrict x, const float *restrict r, const float *restrict w, float *restrict out, float *restrict inv, float *restrict px, float *restrict pr, size_t rows, size_t cols, size_t rs, size_t cs) {

  float tilex[32][32], tiler[32][32];
  for(size_t i=0;i<rows;i+=32) { size_t nr=rows-i;if(nr>32)nr=32;
    double sums[32][4]={0};
    for(size_t j=0;j<cols;j+=32) { size_t nc=cols-j;if(nc>32)nc=32;
      for(size_t col=0;col<nc;++col)for(size_t q=0;q<nr;++q) {
        tilex[q][col]=x[(i+q)*rs+(j+col)*cs]; tiler[q][col]=r[(i+q)*rs+(j+col)*cs];
      }
      for(size_t q=0;q<nr;++q) { size_t col=0;
        for(;col+4<=nc;col+=4) {
          for(size_t k=0;k<4;++k) { float v=tilex[q][col+k]; float square=v*v;
            sums[q][k]+=(double)square; out[(i+q)*cols+j+col+k]=v;
            pr[(i+q)*cols+j+col+k]=tiler[q][col+k]; }
        }
        for(;col<nc;++col) {float v=tilex[q][col];float square=v*v;sums[q][0]+=(double)square;
          out[(i+q)*cols+j+col]=v;pr[(i+q)*cols+j+col]=tiler[q][col];}
      }
    }
    for(size_t q=0;q<nr;++q) {size_t row=i+q;
      float scale=(float)(1.0/sqrt((sums[q][0]+sums[q][1]+sums[q][2]+sums[q][3])/(double)cols+1e-05));inv[row]=scale;
      for(size_t j=0;j<cols;++j) {size_t at=row*cols+j;float a=out[at]*scale;float b=a*w[j];out[at]=b+pr[at];}
    }
  }
}

void tile_stage_64x32(const float *restrict x, const float *restrict r, const float *restrict w, float *restrict out, float *restrict inv, float *restrict px, float *restrict pr, size_t rows, size_t cols, size_t rs, size_t cs) {

  float tilex[64][32], tiler[64][32];
  for(size_t i=0;i<rows;i+=64) { size_t nr=rows-i;if(nr>64)nr=64;
    double sums[64][4]={0};
    for(size_t j=0;j<cols;j+=32) { size_t nc=cols-j;if(nc>32)nc=32;
      for(size_t col=0;col<nc;++col)for(size_t q=0;q<nr;++q) {
        tilex[q][col]=x[(i+q)*rs+(j+col)*cs]; tiler[q][col]=r[(i+q)*rs+(j+col)*cs];
      }
      for(size_t q=0;q<nr;++q) { size_t col=0;
        for(;col+4<=nc;col+=4) {
          for(size_t k=0;k<4;++k) { float v=tilex[q][col+k]; float square=v*v;
            sums[q][k]+=(double)square; out[(i+q)*cols+j+col+k]=v;
            pr[(i+q)*cols+j+col+k]=tiler[q][col+k]; }
        }
        for(;col<nc;++col) {float v=tilex[q][col];float square=v*v;sums[q][0]+=(double)square;
          out[(i+q)*cols+j+col]=v;pr[(i+q)*cols+j+col]=tiler[q][col];}
      }
    }
    for(size_t q=0;q<nr;++q) {size_t row=i+q;
      float scale=(float)(1.0/sqrt((sums[q][0]+sums[q][1]+sums[q][2]+sums[q][3])/(double)cols+1e-05));inv[row]=scale;
      for(size_t j=0;j<cols;++j) {size_t at=row*cols+j;float a=out[at]*scale;float b=a*w[j];out[at]=b+pr[at];}
    }
  }
}

void tile_stage_32x128(const float *restrict x, const float *restrict r, const float *restrict w, float *restrict out, float *restrict inv, float *restrict px, float *restrict pr, size_t rows, size_t cols, size_t rs, size_t cs) {

  float tilex[32][128], tiler[32][128];
  for(size_t i=0;i<rows;i+=32) { size_t nr=rows-i;if(nr>32)nr=32;
    double sums[32][4]={0};
    for(size_t j=0;j<cols;j+=128) { size_t nc=cols-j;if(nc>128)nc=128;
      for(size_t col=0;col<nc;++col)for(size_t q=0;q<nr;++q) {
        tilex[q][col]=x[(i+q)*rs+(j+col)*cs]; tiler[q][col]=r[(i+q)*rs+(j+col)*cs];
      }
      for(size_t q=0;q<nr;++q) { size_t col=0;
        for(;col+4<=nc;col+=4) {
          for(size_t k=0;k<4;++k) { float v=tilex[q][col+k]; float square=v*v;
            sums[q][k]+=(double)square; out[(i+q)*cols+j+col+k]=v;
            pr[(i+q)*cols+j+col+k]=tiler[q][col+k]; }
        }
        for(;col<nc;++col) {float v=tilex[q][col];float square=v*v;sums[q][0]+=(double)square;
          out[(i+q)*cols+j+col]=v;pr[(i+q)*cols+j+col]=tiler[q][col];}
      }
    }
    for(size_t q=0;q<nr;++q) {size_t row=i+q;
      float scale=(float)(1.0/sqrt((sums[q][0]+sums[q][1]+sums[q][2]+sums[q][3])/(double)cols+1e-05));inv[row]=scale;
      for(size_t j=0;j<cols;++j) {size_t at=row*cols+j;float a=out[at]*scale;float b=a*w[j];out[at]=b+pr[at];}
    }
  }
}
