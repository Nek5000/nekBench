/*

See LICENSE file.

*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>

__forceinline__ __device__ __host__  int ijN(const int i, const int j, const int N){

  return i + j*N;

}

__forceinline__ __device__ __host__ int ijkN(const int i, const int j, const int k, const int N){

  return i + j*N + k*N*N;

}

__forceinline__ __device__ __host__ int ijklN(const int i, const int j, const int k, const int l, const int N){

  return i + j*N + k*N*N + l*N*N*N;

}

#define MAX_DOFS_1D 14
#define MAX_HALF_DOFS_1D 7

#define HALF_DOFS_1D ((p_Nq+1)/2)

#define NUM_DOFS_2D (p_Nq*p_Nq)
#define NUM_DOFS_3D (p_Nq*p_Nq*p_Nq)

#define p_Nblock 1

__constant__ dfloat const_DofToDofD[MAX_DOFS_1D*MAX_DOFS_1D];

__forceinline__ __device__ 
  void axhelmDevice(const int numElements,
		 const int element,
		 const dfloat lambda,
		 const dfloat * __restrict__ op,
		 const dfloat * __restrict__ DofToDofD,
		 dfloat * __restrict__ r_p,
		 dfloat * __restrict__ r_Ap){
  
  __shared__ dfloat s_p[p_Nblock][p_Nq][p_Nq];
  __shared__ dfloat s_Gpr[p_Nblock][p_Nq][p_Nq];
  __shared__ dfloat s_Gps[p_Nblock][p_Nq][p_Nq];
  
  // assumes NUM_DOFS_2D threads
  int t = threadIdx.x;
  int blk = threadIdx.y;
  
  int i = t%p_Nq;
  int j = t/p_Nq;
  
  for(int k = 0; k < p_Nq; k++) {
    r_Ap[k] = 0.f; // zero the accumulator
  }
  
  // Layer by layer
#pragma unroll
  for(int k = 0; k < p_Nq; k++) {

    // share r_p[k]
    __syncthreads();

    s_p[blk][j][i] = r_p[k];

    __syncthreads();
    
    dfloat G00 = 0, G01 =0, G02 =0, G11 =0, G12 =0, G22 =0, GWJ =0;
    
    // prefetch geometric factors
    const int gbase = element*p_Nggeo*NUM_DOFS_3D + ijkN(i,j,k,p_Nq);

    if(element<numElements){
      G00 = op[gbase+p_G00ID*NUM_DOFS_3D];
      G01 = op[gbase+p_G01ID*NUM_DOFS_3D];
      G02 = op[gbase+p_G02ID*NUM_DOFS_3D];
      G11 = op[gbase+p_G11ID*NUM_DOFS_3D];
      G12 = op[gbase+p_G12ID*NUM_DOFS_3D];
      G22 = op[gbase+p_G22ID*NUM_DOFS_3D];
      GWJ = op[gbase+p_GWJID*NUM_DOFS_3D];
    }
    
    dfloat pr = 0.f;
    dfloat ps = 0.f;
    dfloat pt = 0.f;

#pragma unroll
    for(int m = 0; m < p_Nq; m++) {
      int im = ijN(m,i,p_Nq);
      int jm = ijN(m,j,p_Nq);
      int km = ijN(m,k,p_Nq);
      pr += DofToDofD[im]*s_p[blk][j][m];
      ps += DofToDofD[jm]*s_p[blk][m][i];
      pt += DofToDofD[km]*r_p[m];
    }
    
    s_Gpr[blk][j][i] = (G00*pr + G01*ps + G02*pt);
    s_Gps[blk][j][i] = (G01*pr + G11*ps + G12*pt);
    
    dfloat Gpt = (G02*pr + G12*ps + G22*pt);
    
    dfloat Apk = GWJ*lambda*r_p[k];
    
    __syncthreads();
    
#pragma unroll
    for(int m = 0; m < p_Nq; m++){
      int mi = ijN(i,m,p_Nq);
      int mj = ijN(j,m,p_Nq);
      int km = ijN(m,k,p_Nq);
      Apk     += DofToDofD[mi]*s_Gpr[blk][j][m];
      Apk     += DofToDofD[mj]*s_Gps[blk][m][i];
      r_Ap[m] += DofToDofD[km]*Gpt; // DT(m,k)*ut(i,j,k,e)
    }
    
    r_Ap[k] += Apk;
  }
  
}

extern "C" __global__ void axhelm_v0(const int numElements,
		                  const dfloat * __restrict__ op,
		                  const dfloat * __restrict__ DofToDofD,
	       	                  const dfloat lambda,
		                  const dfloat * __restrict__ solIn,
		                  dfloat * __restrict__ solOut){
  
  __shared__ dfloat s_DofToDofD[NUM_DOFS_2D];

  dfloat r_q[p_Nq];
  dfloat r_Aq[p_Nq];

  const unsigned int t = threadIdx.x;
  const int blk = threadIdx.y;
  
  const int element = blockIdx.x*p_Nblock + blk;
  
  const unsigned int a = t%p_Nq;
  const unsigned int b = t/p_Nq;

  s_DofToDofD[t] = DofToDofD[t];
  
  if(element < numElements){
    for(int c=0;c<p_Nq;++c){
      
      int id = ijklN(a,b,c,element,p_Nq);
      
      r_q[c] = solIn[id];
    }
  }
  
  __syncthreads();
  
  axhelmDevice(numElements, element, lambda, op, s_DofToDofD, r_q, r_Aq);
  
  if(element<numElements){
#pragma unroll
    for(int c=0;c<p_Nq;++c){
      int id = ijklN(a,b,c,element,p_Nq);
      solOut[id] = r_Aq[c];
    }
  }
}
