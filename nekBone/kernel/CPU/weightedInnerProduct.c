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
                           const dfloat * __restrict__ cpu_w,
			   const dfloat * __restrict__ cpu_a,
			   const dfloat * __restrict__ cpu_b,
			   dfloat * __restrict__ cpu_wab ){

  cpu_w   = (dfloat*)__builtin_assume_aligned(cpu_w, p_Nalign);
  cpu_a   = (dfloat*)__builtin_assume_aligned(cpu_a, p_Nalign);
  cpu_b   = (dfloat*)__builtin_assume_aligned(cpu_b, p_Nalign);
  cpu_wab = (dfloat*)__builtin_assume_aligned(cpu_wab, p_Nalign);

  dfloat wab = 0;
  for(dlong id=0;id<N;++id)
    wab += cpu_a[id]*cpu_b[id]*cpu_w[id];

  cpu_wab[0] = wab;
}

extern "C"
void weightedMultipleInnerProduct2(const dlong & N,
                                   const dlong & offset,
                                   dfloat * __restrict__ cpu_w,
		 	           dfloat * __restrict__ cpu_a,
		       	           dfloat * __restrict__ cpu_b,
			           dfloat * __restrict__ cpu_wab ){

  dfloat wab = 0;
  for(dlong fld=0;fld<p_Nfields;++fld){
    for(dlong id=0;id<N;++id) wab += cpu_a[id+fld*offset]*cpu_b[id+fld*offset]*cpu_w[id+fld*offset];
  }
  cpu_wab[0] = wab;

}
