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
void weightedInnerProduct2(const dlong & N,
                           const dfloat* __restrict__ cpu_w,
                           const dfloat* __restrict__ cpu_a,
                           const dfloat* __restrict__ cpu_b,
                           dfloat* __restrict__ cpu_wab )
{
  dfloat wab = 0;
#pragma omp parallel for reduction(+: wab)
  for(dlong id = 0; id < N; ++id)
    wab += cpu_a[id] * cpu_b[id] * cpu_w[id];

  cpu_wab[0] = wab;
}

extern "C"
void weightedMultipleInnerProduct2(const dlong & N,
                                   const dlong & offset,
                                   const dfloat* __restrict__ cpu_w,
                                   const dfloat* __restrict__ cpu_a,
                                   const dfloat* __restrict__ cpu_b,
                                   dfloat* __restrict__ cpu_wab )
{
  dfloat wab = 0;
  for(dlong fld = 0; fld < p_Nfields; ++fld) {
#pragma omp parallel for reduction(+: wab)
    for(dlong id = 0; id < N; ++id) {
      const dlong iid = id + fld * offset;
      wab += cpu_a[iid] * cpu_b[iid] * cpu_w[id];
    }
  }

  cpu_wab[0] = wab;
}


extern "C"
void weightedInnerProductUpdate(const dlong & N,
                                const dlong & offset,
                                const dlong & Nblock,
                                const dfloat* __restrict__  cpu_w,
                                const dfloat* __restrict__  cpu_a,
                                const dfloat* __restrict__  cpu_b,
                                dfloat* __restrict__  cpu_wab)
{
  dfloat wab[] = {0, 0};
  for(dlong fld = 0; fld < p_Nfields; ++fld) {
#pragma omp parallel for reduction(+: wab)
    for(dlong id = 0; id < N; ++id) { 
      const dlong iid = id + fld * offset;
      wab[0] += cpu_a[iid] * cpu_b[iid] * cpu_w[id];
      wab[1] += cpu_a[iid] * cpu_a[iid] * cpu_w[id];
    }
  }

  cpu_wab[0] = wab[0];
  cpu_wab[1] = wab[1];
}
