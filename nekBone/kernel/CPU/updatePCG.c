/*

The MIT License (MIT)

Copyright (c) 2017 Tim Warburton, Noel Chalmers, Jesse Chan, Ali Karakus

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

*/

extern "C" 
void BPUpdatePCG(const dlong & Nelements,
                 const dlong & Np,
                 const dlong & Nblocks,
	         const dfloat * __restrict__ cpu_invDegree,
		 const dfloat * __restrict__ cpu_p,
		 const dfloat * __restrict__ cpu_Ap,
		 const dfloat & alpha,
		 dfloat * __restrict__ cpu_x,
		 dfloat * __restrict__ cpu_r,
		 dfloat * __restrict__ redr){

  cpu_p  = (dfloat*)__builtin_assume_aligned(cpu_p,  p_Nalign) ;
  cpu_Ap = (dfloat*)__builtin_assume_aligned(cpu_Ap, p_Nalign) ;
  cpu_x  = (dfloat*)__builtin_assume_aligned(cpu_x,  p_Nalign) ;
  cpu_r  = (dfloat*)__builtin_assume_aligned(cpu_r,  p_Nalign) ;
  cpu_invDegree = (dfloat*)__builtin_assume_aligned(cpu_invDegree,  p_Nalign) ;
  
  dfloat rdotr = 0;
  
  for(dlong e=0;e<Nelements;++e){
    for(int i=0;i<p_Np;++i){
      const dlong n = e*p_Np+i;
      cpu_x[n] += alpha*cpu_p[n];

      const dfloat rn = cpu_r[n] - alpha*cpu_Ap[n];
      rdotr += rn*rn*cpu_invDegree[n];
      cpu_r[n] = rn;
    }
  }
  redr[0] = rdotr;
}

extern "C" 
void BPMultipleUpdatePCG(
             const dlong & Nelements,
             const dlong & Np,
             const dlong & offset,
             const dlong & Nblocks,
             const dfloat * __restrict__ cpu_invDegree,
             const dfloat * __restrict__ cpu_p,
             const dfloat * __restrict__ cpu_Ap,
             const dfloat alpha,
             dfloat * __restrict__ cpu_x,
             dfloat * __restrict__ cpu_r,
             dfloat * __restrict__ redr){

  cpu_p  = (dfloat*)__builtin_assume_aligned(cpu_p,  p_Nalign) ;
  cpu_Ap = (dfloat*)__builtin_assume_aligned(cpu_Ap, p_Nalign) ;
  cpu_x  = (dfloat*)__builtin_assume_aligned(cpu_x,  p_Nalign) ;
  cpu_r  = (dfloat*)__builtin_assume_aligned(cpu_r,  p_Nalign) ;
  cpu_invDegree = (dfloat*)__builtin_assume_aligned(cpu_invDegree,  p_Nalign) ;
  
  dfloat rdotr = 0;
  
  for(int fld=0; fld<p_Nfields; fld++){
  for(dlong e=0;e<Nelements;++e){
    for(int i=0;i<p_Np;++i){
      const dlong n = e*p_Np+i + fld*offset;
      cpu_x[n] += alpha*cpu_p[n];

      const dfloat rn = cpu_r[n] - alpha*cpu_Ap[n];
      rdotr += rn*rn*cpu_invDegree[e*p_Np+i];
      cpu_r[n] = rn;
   }
 }
 }
 redr[0] = rdotr;
}