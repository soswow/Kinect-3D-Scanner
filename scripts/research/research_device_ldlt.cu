// SPDX-License-Identifier: MPL-2.0
// Eigen-derived portions: Copyright (C) 2008-2011 Gael Guennebaud,
// 2009 Keir Mierle, 2009 Benoit Jacob, 2011 Timothy E. Holy (LDLT),
// and 2009 Mathieu Gautier (Quaternion).
// This Source Code Form is subject to the Mozilla Public License v.2.0;
// obtain a copy at https://mozilla.org/MPL/2.0/.
// Separate research port of pinned Eigen LDLT<MatrixXd,Lower> and quaternion
// Euler update semantics. Scalar reduction/libdevice trig differ from Eigen's
// CPU evaluator: fresh CPU-shadow proof is mandatory; no bit-parity claim.
// Reference Eigen da7909592376c893dabbc4b6453a8ffe46b1eb8e:
// Eigen/src/Cholesky/LDLT.h and Eigen/src/Geometry/Quaternion.h (MPL-2.0).
// NVRTC provides device math builtins without host C headers. This exact
// binary64 literal equals DBL_MIN used by pinned Eigen's pseudo-inverse and is
// accepted by NVRTC's default C++11 mode (hex float syntax requires C++17).
constexpr double eigen_double_min = 2.2250738585072014e-308;
__device__ __forceinline__ double rejected_nan(){return __longlong_as_double(0x7ff8000000000000LL);}

__device__ __forceinline__ double add(double a,double b){return __dadd_rn(a,b);}
__device__ __forceinline__ double sub(double a,double b){return __dsub_rn(a,b);}
__device__ __forceinline__ double mul(double a,double b){return __dmul_rn(a,b);}
__device__ __forceinline__ double divn(double a,double b){return __ddiv_rn(a,b);}
__device__ __forceinline__ void swapd(double &a,double &b){double t=a;a=b;b=t;}

// Quaternion arrays are w,x,y,z; same generic Hamilton association as Eigen.
__device__ void product(const double* a,const double* b,double* c){
    c[0]=sub(sub(sub(mul(a[0],b[0]),mul(a[1],b[1])),mul(a[2],b[2])),mul(a[3],b[3]));
    c[1]=sub(add(add(mul(a[0],b[1]),mul(a[1],b[0])),mul(a[2],b[3])),mul(a[3],b[2]));
    c[2]=sub(add(add(mul(a[0],b[2]),mul(a[2],b[0])),mul(a[3],b[1])),mul(a[1],b[3]));
    c[3]=sub(add(add(mul(a[0],b[3]),mul(a[3],b[0])),mul(a[1],b[2])),mul(a[2],b[1]));
}

__device__ void euler_update(const double* x,double* pose){
    double ax=mul(.5,x[0]),ay=mul(.5,x[1]),az=mul(.5,x[2]);
    double sx=sin(ax),sy=sin(ay),sz=sin(az);
    double qx[4]={cos(ax),sx,mul(sx,0.),mul(sx,0.)};
    double qy[4]={cos(ay),mul(sy,0.),sy,mul(sy,0.)};
    double qz[4]={cos(az),mul(sz,0.),mul(sz,0.),sz};
    double first[4],q[4];product(qz,qy,first);product(first,qx,q);
    double tx=mul(2.,q[1]),ty=mul(2.,q[2]),tz=mul(2.,q[3]);
    double twx=mul(tx,q[0]),twy=mul(ty,q[0]),twz=mul(tz,q[0]);
    double txx=mul(tx,q[1]),txy=mul(ty,q[1]),txz=mul(tz,q[1]);
    double tyy=mul(ty,q[2]),tyz=mul(tz,q[2]),tzz=mul(tz,q[3]);
    pose[0]=sub(1.,add(tyy,tzz));pose[1]=sub(txy,twz);pose[2]=add(txz,twy);pose[3]=x[3];
    pose[4]=add(txy,twz);pose[5]=sub(1.,add(txx,tzz));pose[6]=sub(tyz,twx);pose[7]=x[4];
    pose[8]=sub(txz,twy);pose[9]=add(tyz,twx);pose[10]=sub(1.,add(txx,tyy));pose[11]=x[5];
    pose[12]=0.;pose[13]=0.;pose[14]=0.;pose[15]=1.;
}

extern "C" __global__ void device_ldlt_solve(const double* input,const double* gradient,
    const double* previous,int count,int* status,double* step,double* update,double* composed,
    int* pivots_out,double* diagonal_out){
    int row=int(blockIdx.x*blockDim.x+threadIdx.x);
    if(row>=count)return;
    int code=0,sign=0;bool ret=true,zero=false;
    double a[36],b[6],temp[6];int pivots[6];
    for(int i=0;i<36;++i){a[i]=input[36*row+i];if(!isfinite(a[i]))code=2;}
    for(int i=0;i<6;++i){b[i]=-gradient[6*row+i];if(!isfinite(b[i]))code=2;pivots[i]=i;}
    // Preflight the optional composition operand as a separate research input.
    for(int i=0;i<16;++i)if(!isfinite(previous[16*row+i]))code=8;
    if(!code){
        for(int k=0;k<6;++k){
            int p=k;double largest=fabs(a[7*k]);
            for(int i=k+1;i<6;++i)if(fabs(a[7*i])>largest){largest=fabs(a[7*i]);p=i;}
            pivots[k]=p;
            if(p!=k){
                for(int i=0;i<k;++i)swapd(a[6*k+i],a[6*p+i]);
                for(int i=p+1;i<6;++i)swapd(a[6*i+k],a[6*i+p]);
                swapd(a[7*k],a[7*p]);
                for(int i=k+1;i<p;++i)swapd(a[6*i+k],a[6*p+i]);
            }
            if(k){
                for(int j=0;j<k;++j)temp[j]=mul(a[7*j],a[6*k+j]);
                double sum=0.;for(int j=0;j<k;++j)sum=add(sum,mul(a[6*k+j],temp[j]));
                a[7*k]=sub(a[7*k],sum);
                for(int i=k+1;i<6;++i){
                    sum=0.;for(int j=0;j<k;++j)sum=add(sum,mul(a[6*i+j],temp[j]));
                    a[6*i+k]=sub(a[6*i+k],sum);
                }
            }
            double d=a[7*k];bool valid=fabs(d)>0.;
            if(k==0&&!valid){
                sign=0;
                for(int j=0;j<6;++j){pivots[j]=j;for(int i=j+1;i<6;++i)ret=ret&&(a[6*i+j]==0.);}
                break;
            }
            if(valid){for(int i=k+1;i<6;++i)a[6*i+k]=divn(a[6*i+k],d);}
            else for(int i=k+1;i<6;++i)ret=ret&&(a[6*i+k]==0.);
            if(zero&&valid)ret=false;else if(!valid)zero=true;
            if(sign==1&&d<0.)sign=2;
            else if(sign==-1&&d>0.)sign=2;
            else if(sign==0){if(d>0.)sign=1;else if(d<0.)sign=-1;}
        }
        if(!ret||(sign!=0&&sign!=1))code=3;
        if(!code){
            double largest=fabs(a[0]),smallest=largest;
            for(int i=1;i<6;++i){double d=fabs(a[7*i]);if(d>largest)largest=d;if(d<smallest)smallest=d;}
            if(!(largest>0.)||smallest<=mul(largest,1e-12))code=4;
        }
        if(!code){
            for(int k=0;k<6;++k)swapd(b[k],b[pivots[k]]);
            for(int i=0;i<6;++i){double sum=0.;for(int j=0;j<i;++j)sum=add(sum,mul(a[6*i+j],b[j]));b[i]=sub(b[i],sum);}
            for(int i=0;i<6;++i)b[i]=fabs(a[7*i])>eigen_double_min?divn(b[i],a[7*i]):0.;
            for(int i=5;i>=0;--i){double sum=0.;for(int j=i+1;j<6;++j)sum=add(sum,mul(a[6*j+i],b[j]));b[i]=sub(b[i],sum);}
            for(int k=5;k>=0;--k)swapd(b[k],b[pivots[k]]);
            for(int i=0;i<6;++i)if(!isfinite(b[i]))code=5;
        }
    }
    for(int i=0;i<6;++i){step[6*row+i]=code?rejected_nan():b[i];pivots_out[6*row+i]=pivots[i];diagonal_out[6*row+i]=a[7*i];}
    double* pose=update+16*row;double* result=composed+16*row;
    if(!code){
        euler_update(b,pose);
        for(int i=0;i<16;++i)if(!isfinite(pose[i]))code=6;
        if(!code){
            for(int i=0;i<4;++i)for(int j=0;j<4;++j){
                double sum=0.;for(int k=0;k<4;++k)sum=add(sum,mul(pose[4*i+k],previous[16*row+4*k+j]));
                result[4*i+j]=sum;
                if(!isfinite(sum))code=9;
            }
        }
    }
    if(code)for(int i=0;i<16;++i){pose[i]=rejected_nan();result[i]=rejected_nan();}
    status[row]=code;
}
