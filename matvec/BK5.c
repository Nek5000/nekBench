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

#define dfloat double
#define dlong int
#define p_Np (p_Nq*p_Nq*p_Nq)

#define G00ID 0
#define G01ID 1
#define G02ID 2
#define G11ID 3
#define G12ID 4
#define G22ID 5
#define GWJID 6
#define p_Nggeo 7 

template < const int p_Nq >
void kernel(const hlong Nelements,
	    const dfloat * __restrict__ ggeo ,
	    const dfloat * __restrict__ D ,
	    const dfloat * __restrict__ S ,
	    const dfloat lambda,
	    const dfloat * __restrict__ q ,
	    dfloat * __restrict__ Aq ){
  
  D    = (dfloat*)__builtin_assume_aligned(D, USE_OCCA_MEM_BYTE_ALIGN) ;
  S    = (dfloat*)__builtin_assume_aligned(S, USE_OCCA_MEM_BYTE_ALIGN) ;
  q    = (dfloat*)__builtin_assume_aligned(q, USE_OCCA_MEM_BYTE_ALIGN) ;
  Aq   = (dfloat*)__builtin_assume_aligned(Aq, USE_OCCA_MEM_BYTE_ALIGN) ;
  ggeo = (dfloat*)__builtin_assume_aligned(ggeo, USE_OCCA_MEM_BYTE_ALIGN) ;
  
  dfloat s_q  [p_Nq][p_Nq][p_Nq] __attribute__((aligned(USE_OCCA_MEM_BYTE_ALIGN)));
  dfloat s_Gqr[p_Nq][p_Nq][p_Nq] __attribute__((aligned(USE_OCCA_MEM_BYTE_ALIGN)));
  dfloat s_Gqs[p_Nq][p_Nq][p_Nq] __attribute__((aligned(USE_OCCA_MEM_BYTE_ALIGN)));
  dfloat s_Gqt[p_Nq][p_Nq][p_Nq] __attribute__((aligned(USE_OCCA_MEM_BYTE_ALIGN)));

  dfloat s_D[p_Nq][p_Nq]  __attribute__((aligned(USE_OCCA_MEM_BYTE_ALIGN)));
  dfloat s_S[p_Nq][p_Nq]  __attribute__((aligned(USE_OCCA_MEM_BYTE_ALIGN)));

  for(int j=0;j<p_Nq;++j){
    for(int i=0;i<p_Nq;++i){
      s_D[j][i] = D[j*p_Nq+i];
      s_S[j][i] = S[j*p_Nq+i];
    }
  }

  const int c_Np = p_Np;
  const int p_N = p_Nq-1;

  for(dlong e=0; e<Nelements; ++e){
    const dlong element = e;
    for(int k = 0; k < p_Nq; k++) {
      for(int j=0;j<p_Nq;++j){
        for(int i=0;i<p_Nq;++i){
          const dlong base = i + j*p_Nq + k*p_Nq*p_Nq + element*c_Np;
          const dfloat qbase = q[base];
          s_q[k][j][i] = qbase;
        }
      }
    }

    for(int k=0;k<p_Nq;++k){
      for(int j=0;j<p_Nq;++j){
        for(int i=0;i<p_Nq;++i){
          const dlong gbase = element*p_Nggeo*c_Np + k*p_Nq*p_Nq + j*p_Nq + i;
          const dfloat r_G00 = ggeo[gbase+G00ID*p_Np];
          const dfloat r_G01 = ggeo[gbase+G01ID*p_Np];
          const dfloat r_G11 = ggeo[gbase+G11ID*p_Np];
          const dfloat r_G12 = ggeo[gbase+G12ID*p_Np];
          const dfloat r_G02 = ggeo[gbase+G02ID*p_Np];
          const dfloat r_G22 = ggeo[gbase+G22ID*p_Np];

          dfloat qr = 0.f;
          dfloat qs = 0.f;
          dfloat qt = 0.f;

          for(int m = 0; m < p_Nq; m++) {
            qr += s_S[m][i]*s_q[k][j][m];  
            qs += s_S[m][j]*s_q[k][m][i];           
            qt += s_S[m][k]*s_q[m][j][i]; 
          }

          dfloat Gqr = r_G00*qr;
          Gqr += r_G01*qs;
          Gqr += r_G02*qt;
          
          dfloat Gqs = r_G01*qr;
          Gqs += r_G11*qs;
          Gqs += r_G12*qt;

          dfloat Gqt = r_G02*qr;
          Gqt += r_G12*qs;
          Gqt += r_G22*qt;
          
          s_Gqr[k][j][i] = Gqr;
          s_Gqs[k][j][i] = Gqs;
          s_Gqt[k][j][i] = Gqt;
        }
      }
    }

    for(int k = 0;k < p_Nq; k++){
      for(int j=0;j<p_Nq;++j){
        for(int i=0;i<p_Nq;++i){
          const dlong gbase = element*p_Nggeo*p_Np + k*p_Nq*p_Nq + j*p_Nq + i;
          const dfloat r_GwJ = ggeo[gbase+GWJID*p_Np];

          dfloat r_Aq = r_GwJ*lambda*s_q[k][j][i];
          dfloat r_Aqr = 0, r_Aqs = 0, r_Aqt = 0;


          for(int m = 0; m < p_Nq; m++)
            r_Aqr += s_D[m][i]*s_Gqr[k][j][m];
          for(int m = 0; m < p_Nq; m++)
            r_Aqs += s_D[m][j]*s_Gqs[k][m][i];
          for(int m = 0; m < p_Nq; m++)
            r_Aqt += s_D[m][k]*s_Gqt[m][j][i];

          const dlong id = element*p_Np +k*p_Nq*p_Nq+ j*p_Nq + i;
          Aq[id] = r_Aqr + r_Aqs + r_Aqt +r_Aq;
        }
      }
    }
  }
}

extern "C" void BK5(const int &Nq,
               const hlong &Nelements,
               const dfloat *ggeo,
               const dfloat *D,
               const dfloat &lambda,
               const dfloat *q,
               dfloat *Aq) {

  switch(Nq){
  case   5: kernel <   5 > (Nelements, ggeo, D, lambda, q, Aq); break;
  case   6: kernel <   6 > (Nelements, ggeo, D, lambda, q, Aq); break;
  case   7: kernel <   7 > (Nelements, ggeo, D, lambda, q, Aq); break;
  case   8: kernel <   8 > (Nelements, ggeo, D, lambda, q, Aq); break;
  case   9: kernel <   9 > (Nelements, ggeo, D, lambda, q, Aq); break;
  case  10: kernel <  10 > (Nelements, ggeo, D, lambda, q, Aq); break;
  }

}
