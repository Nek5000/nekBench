#define cubNq3 (cubNq * cubNq * cubNq)
#define Nq3 (Nq * Nq * Nq)

static int meshIJN(const int i, const int j, const int N)
{
  return i + j * N;
}

static int meshIJKN(const int i, const int j, const int k, const int N)
{
  return i + j * N + k * N * N;
}

static int meshIJKLN(const int i, const int j, const int k, const int l, const int N)
{
  return i + j * N + k * N * N + l * N * N * N;
}

static void axhelmElementReference(int cubNq,
                                   int element,
                                   dfloat lambda,
                                   const dfloat*  ggeo,
                                   const dfloat*  cubD,
                                   const dfloat* qIII,
                                   dfloat* lapqIII)
{
  dfloat Gqr[cubNq][cubNq][cubNq];
  dfloat Gqs[cubNq][cubNq][cubNq];
  dfloat Gqt[cubNq][cubNq][cubNq];

  for(int k = 0; k < cubNq; ++k)
    for(int j = 0; j < cubNq; ++j)
      for(int i = 0; i < cubNq; ++i) {
        dfloat qr = 0;
        dfloat qs = 0;
        dfloat qt = 0;

        for(int n = 0; n < cubNq; ++n) {
          int in = meshIJN(n,i,cubNq);
          int jn = meshIJN(n,j,cubNq);
          int kn = meshIJN(n,k,cubNq);

          int kjn = meshIJKN(n,j,k,cubNq);
          int kni = meshIJKN(i,n,k,cubNq);
          int nji = meshIJKN(i,j,n,cubNq);

          qr += cubD[in] * qIII[kjn];
          qs += cubD[jn] * qIII[kni];
          qt += cubD[kn] * qIII[nji];
        }

        const int gbase = element * p_Nggeo * cubNq3 + meshIJKN(i,j,k,cubNq);

        dfloat G00 = ggeo[gbase + p_G00ID * cubNq3];
        dfloat G01 = ggeo[gbase + p_G01ID * cubNq3];
        dfloat G02 = ggeo[gbase + p_G02ID * cubNq3];
        dfloat G11 = ggeo[gbase + p_G11ID * cubNq3];
        dfloat G12 = ggeo[gbase + p_G12ID * cubNq3];
        dfloat G22 = ggeo[gbase + p_G22ID * cubNq3];

        Gqr[k][j][i] = (G00 * qr + G01 * qs + G02 * qt);
        Gqs[k][j][i] = (G01 * qr + G11 * qs + G12 * qt);
        Gqt[k][j][i] = (G02 * qr + G12 * qs + G22 * qt);
      }


  for(int k = 0; k < cubNq; ++k)
    for(int j = 0; j < cubNq; ++j)
      for(int i = 0; i < cubNq; ++i) {
        int kji = meshIJKN(i,j,k,cubNq);

        const int gbase = element * p_Nggeo * cubNq3 + meshIJKN(i,j,k,cubNq);

        dfloat GWJ = ggeo[gbase + p_GWJID * cubNq3];
        dfloat lapq = lambda * GWJ * qIII[kji];

        for(int n = 0; n < cubNq; ++n) {
          int ni = meshIJN(i,n,cubNq);
          int nj = meshIJN(j,n,cubNq);
          int nk = meshIJN(k,n,cubNq);

          lapq += cubD[ni] * Gqr[k][j][n];
          lapq += cubD[nj] * Gqs[k][n][i];
          lapq += cubD[nk] * Gqt[n][j][i];
        }

        lapqIII[kji] = lapq;
      }
}

void axhelmReference(int Nq,
                     const int numElements,
                     dfloat lambda,
                     const dfloat*  ggeo,
                     const dfloat*  D,
                     const dfloat*  solIn,
                     dfloat*  solOut)
{
  for(int e = 0; e < numElements; ++e)
    axhelmElementReference(Nq, e, lambda, ggeo, D, solIn + e * Nq3, solOut + e * Nq3);
}
/* offsets for geometric factors */
#define RXID 0
#define RYID 1
#define SXID 2
#define SYID 3
#define  JID 4
#define JWID 5
#define IJWID 6
#define RZID 7
#define SZID 8
#define TXID 9
#define TYID 10
#define TZID 11
void axhelmStressReference(int Nq,
                     const int Nelements,
                     const int offset,
                     const int loffset,
                     const dfloat* lambda,
                     const dfloat*  vgeo,
                     const dfloat*  D,
                     const dfloat*  q,
                     dfloat*  Aq)
{
    dfloat s_D[Nq][Nq];

    dfloat s_U[Nq][Nq][Nq];
    dfloat s_V[Nq][Nq][Nq];
    dfloat s_W[Nq][Nq][Nq];

    dfloat s_SUr[Nq][Nq][Nq];
    dfloat s_SUs[Nq][Nq][Nq];
    dfloat s_SUt[Nq][Nq][Nq];

    dfloat s_SVr[Nq][Nq][Nq];
    dfloat s_SVs[Nq][Nq][Nq];
    dfloat s_SVt[Nq][Nq][Nq];

    dfloat s_SWr[Nq][Nq][Nq];
    dfloat s_SWs[Nq][Nq][Nq];
    dfloat s_SWt[Nq][Nq][Nq];

    for(int j=0;j<Nq;++j){
      for(int i=0;i<Nq;++i){
      s_D[j][i] = D[j*Nq+i];
    }
  }
    

for(dlong e=0; e<Nelements; ++e){

    for(int k=0;k<Nq;++k){ 
      for(int j=0;j<Nq;++j){
        for(int i=0;i<Nq;++i){
            const dlong id = e*Nq*Nq*Nq+k*Nq*Nq+j*Nq+i;
            s_U[k][j][i] = q[id + 0*offset];
            s_V[k][j][i] = q[id + 1*offset];
            s_W[k][j][i] = q[id + 2*offset];
        }
      }
    }
    


    // loop over slabs
     for(int k=0;k<Nq;++k){ 
      for(int j=0;j<Nq;++j){
        for(int i=0;i<Nq;++i){   
          const dlong gid = i + j*Nq + k*Nq*Nq + e*Nq*Nq*Nq*p_Nvgeo;
          const dfloat rx = vgeo[gid + RXID*Nq*Nq*Nq];
          const dfloat ry = vgeo[gid + RYID*Nq*Nq*Nq];
          const dfloat rz = vgeo[gid + RZID*Nq*Nq*Nq];
          
          const dfloat sx = vgeo[gid + SXID*Nq*Nq*Nq];
          const dfloat sy = vgeo[gid + SYID*Nq*Nq*Nq];
          const dfloat sz = vgeo[gid + SZID*Nq*Nq*Nq];
          
          const dfloat tx = vgeo[gid + TXID*Nq*Nq*Nq];
          const dfloat ty = vgeo[gid + TYID*Nq*Nq*Nq];
          const dfloat tz = vgeo[gid + TZID*Nq*Nq*Nq];
          
          const dfloat JW = vgeo[gid + JWID*Nq*Nq*Nq];

          // compute 1D derivatives
          dfloat ur = 0.f, us = 0.f, ut = 0.f;
          dfloat vr = 0.f, vs = 0.f, vt = 0.f;
          dfloat wr = 0.f, ws = 0.f, wt = 0.f;
          for(int m=0;m<Nq;++m){
            const dfloat Dim = s_D[i][m]; // Dr
            const dfloat Djm = s_D[j][m]; // Ds
            const dfloat Dkm = s_D[k][m]; // Dt

            ur += Dim*s_U[k][j][m];
            us += Djm*s_U[k][m][i];
            ut += Dkm*s_U[m][j][i];
            //
            vr += Dim*s_V[k][j][m];
            vs += Djm*s_V[k][m][i];
            vt += Dkm*s_V[m][j][i];
            //
            wr += Dim*s_W[k][j][m];
            ws += Djm*s_W[k][m][i];
            wt += Dkm*s_W[m][j][i];
          }

          const dlong id = e*Nq*Nq*Nq + k*Nq*Nq + j*Nq + i;  
          const dfloat u_lam0 = lambda[id + 0*offset + 0*loffset]; 
          // const dfloat u_lam1 = lambda[id + 1*offset + 0*loffset];
          const dfloat v_lam0 = lambda[id + 0*offset + 1*loffset]; 
          // const dfloat v_lam1 = lambda[id + 1*offset + 1*loffset];
          const dfloat w_lam0 = lambda[id + 0*offset + 2*loffset]; 
          // const dfloat w_lam1 = lambda[id + 1*offset + 2*loffset];

         
          const dfloat dudx = rx*ur + sx*us + tx*ut; 
          const dfloat dudy = ry*ur + sy*us + ty*ut; 
          const dfloat dudz = rz*ur + sz*us + tz*ut; 

          const dfloat dvdx = rx*vr + sx*vs + tx*vt; 
          const dfloat dvdy = ry*vr + sy*vs + ty*vt; 
          const dfloat dvdz = rz*vr + sz*vs + tz*vt; 

          const dfloat dwdx = rx*wr + sx*ws + tx*wt; 
          const dfloat dwdy = ry*wr + sy*ws + ty*wt; 
          const dfloat dwdz = rz*wr + sz*ws + tz*wt; 

          const dfloat s11 = u_lam0*JW*(dudx + dudx); 
          const dfloat s12 = u_lam0*JW*(dudy + dvdx); 
          const dfloat s13 = u_lam0*JW*(dudz + dwdx); 

          const dfloat s21 = v_lam0*JW*(dvdx + dudy); 
          const dfloat s22 = v_lam0*JW*(dvdy + dvdy); 
          const dfloat s23 = v_lam0*JW*(dvdz + dwdy); 

          const dfloat s31 = w_lam0*JW*(dwdx + dudz); 
          const dfloat s32 = w_lam0*JW*(dwdy + dvdz); 
          const dfloat s33 = w_lam0*JW*(dwdz + dwdz); 

          s_SUr[k][j][i] =  rx*s11 + ry*s12 + rz*s13;
          s_SUs[k][j][i] =  sx*s11 + sy*s12 + sz*s13;
          s_SUt[k][j][i] =  tx*s11 + ty*s12 + tz*s13;
          //
          s_SVr[k][j][i] =  rx*s21 + ry*s22 + rz*s23;
          s_SVs[k][j][i] =  sx*s21 + sy*s22 + sz*s23;
          s_SVt[k][j][i] =  tx*s21 + ty*s22 + tz*s23;
          //
          s_SWr[k][j][i] =  rx*s31 + ry*s32 + rz*s33;
          s_SWs[k][j][i] =  sx*s31 + sy*s32 + sz*s33;
          s_SWt[k][j][i] =  tx*s31 + ty*s32 + tz*s33;

         
        }
      }
    }


// loop over slabs
    for(int k=0;k<Nq;++k){ 
      for(int j=0;j<Nq;++j){
        for(int i=0;i<Nq;++i){
          dfloat r_Au = 0.f, r_Av = 0.f, r_Aw = 0.f;
          for(int m = 0; m < Nq; m++) {
            const dfloat Dim = s_D[m][i]; // Dr'
            const dfloat Djm = s_D[m][j]; // Ds'
            const dfloat Dkm = s_D[m][k]; // Dt'

            r_Au += Dim*s_SUr[k][j][m];
            r_Au += Djm*s_SUs[k][m][i];
            r_Au += Dkm*s_SUt[m][j][i];

            r_Av += Dim*s_SVr[k][j][m];
            r_Av += Djm*s_SVs[k][m][i];
            r_Av += Dkm*s_SVt[m][j][i];

            r_Aw += Dim*s_SWr[k][j][m];
            r_Aw += Djm*s_SWs[k][m][i];
            r_Aw += Dkm*s_SWt[m][j][i];
          }
          const dlong id      = e*Nq*Nq*Nq +k*Nq*Nq+ j*Nq + i;
          const dfloat u_lam1 = lambda[id + 1*offset + 0*loffset];
          const dfloat v_lam1 = lambda[id + 1*offset + 1*loffset];
          const dfloat w_lam1 = lambda[id + 1*offset + 2*loffset];
         
          const dlong gid = i + j*Nq + k*Nq*Nq + e*Nq*Nq*Nq*p_Nvgeo;
          const dfloat JW = vgeo[gid + JWID*Nq*Nq*Nq];
           // store in register
          Aq[id+0*offset] =  r_Au + u_lam1*JW*s_U[k][j][i]; 
          Aq[id+1*offset] =  r_Av + v_lam1*JW*s_V[k][j][i];
          Aq[id+2*offset] =  r_Aw + w_lam1*JW*s_W[k][j][i];
        }
      }
    }
  }
}

