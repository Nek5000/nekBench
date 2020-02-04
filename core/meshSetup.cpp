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


#include "omp.h"
#include "mpi.h"
#include "mesh.h"

#if USE_CUDA_NATIVE==1
#include <occa/modes/cuda/utils.hpp>
#include <cuda_runtime_api.h>
#endif

int findBestPeriodicMatch(dfloat xper, dfloat yper, dfloat zper,
			  dfloat x1, dfloat y1, dfloat z1,
			  int Np2, int *nodeList, dfloat *x2, dfloat *y2, dfloat *z2, int *nP){
  
  int matchIndex;
  dfloat mindist2=1e9;
  int isFirst = 1;
  
  for(int n=0;n<Np2;++n){
    
    /* next node */
    const int i2 = nodeList[n];
    for(int zp=0;zp<2;++zp){
      for(int yp=0;yp<2;++yp){
	for(int xp=0;xp<2;++xp){

	  /* distance between target and next node */
	  const dfloat dist2 =
	    pow(fabs(x1-x2[i2])-xp*xper,2) +
	    pow(fabs(y1-y2[i2])-yp*yper,2) +
	    pow(fabs(z1-z2[i2])-zp*zper,2);
	  
	  /* if next node is closer to target update match */
	  if(isFirst==1 || dist2<mindist2){
	    mindist2 = dist2;
	    matchIndex = i2;
	    *nP = n;
	    isFirst=0;
	  }
	}
      }
    }
  }
  if(mindist2>1e-3) printf("arggh - bad match: x,y,z= %g,%g,%g => %g,%g,%g with mindist=%lg\n",
			   x1,y1,z1,  x2[matchIndex], y2[matchIndex],  z2[matchIndex], mindist2);

  return matchIndex;
}
                

// serial face-node to face-node connection
void meshConnectPeriodicFaceNodes3D(mesh3D *mesh, dfloat xper, dfloat yper, dfloat zper){
  
  /* volume indices of the interior and exterior face nodes for each element */
  mesh->vmapM = (dlong*) calloc(mesh->Nfp*mesh->Nfaces*mesh->Nelements, sizeof(dlong));
  mesh->vmapP = (dlong*) calloc(mesh->Nfp*mesh->Nfaces*mesh->Nelements, sizeof(dlong));
  mesh->mapP  = (dlong*) calloc(mesh->Nfp*mesh->Nfaces*mesh->Nelements, sizeof(dlong));
  
  /* assume elements already connected */
  for(dlong e=0;e<mesh->Nelements;++e){
    for(int f=0;f<mesh->Nfaces;++f){
      dlong eP = mesh->EToE[e*mesh->Nfaces+f];
      int fP = mesh->EToF[e*mesh->Nfaces+f];
      if(eP<0 || fP<0){ // fake connections for unconnected faces
        eP = e;
        fP = f;
      }
      /* for each node on this face find the neighbor node */
      for(int n=0;n<mesh->Nfp;++n){
        dlong  idM = mesh->faceNodes[f*mesh->Nfp+n] + e*mesh->Np;
        dfloat xM = mesh->x[idM];
        dfloat yM = mesh->y[idM];
        dfloat zM = mesh->z[idM];
        int nP;
        
        int  idP = findBestPeriodicMatch(xper, yper, zper,
					 xM, yM, zM,
					 mesh->Nfp, 
					 mesh->faceNodes+fP*mesh->Nfp,
					 mesh->x+eP*mesh->Np,
					 mesh->y+eP*mesh->Np,
					 mesh->z+eP*mesh->Np, &nP);
	
        dlong id = mesh->Nfaces*mesh->Nfp*e + f*mesh->Nfp + n;
        mesh->vmapM[id] = idM;
        mesh->vmapP[id] = idP + eP*mesh->Np;
        mesh->mapP[id] = eP*mesh->Nfaces*mesh->Nfp + fP*mesh->Nfp + nP;
      }
    }
  }
}


void meshGeometricFactorsTet3D(mesh3D *mesh){

  /* unified storage array for geometric factors */
  mesh->Nvgeo = 12;
  mesh->vgeo = (dfloat*) calloc(mesh->Nelements*mesh->Nvgeo, 
        sizeof(dfloat));

  /* number of second order geometric factors */
  mesh->Nggeo = 7;
  mesh->ggeo = (dfloat*) calloc(mesh->Nelements*mesh->Nggeo, sizeof(dfloat));

  mesh->cubggeo = (dfloat*) calloc(mesh->Nelements*mesh->Nggeo*mesh->cubNp, sizeof(dfloat));
  

  dfloat minJ = 1e9, maxJ = -1e9;
  for(dlong e=0;e<mesh->Nelements;++e){ /* for each element */

    /* find vertex indices and physical coordinates */
    dlong id = e*mesh->Nverts;

    /* vertex coordinates */
    dfloat xe1 = mesh->EX[id+0], ye1 = mesh->EY[id+0], ze1 = mesh->EZ[id+0];
    dfloat xe2 = mesh->EX[id+1], ye2 = mesh->EY[id+1], ze2 = mesh->EZ[id+1];
    dfloat xe3 = mesh->EX[id+2], ye3 = mesh->EY[id+2], ze3 = mesh->EZ[id+2];
    dfloat xe4 = mesh->EX[id+3], ye4 = mesh->EY[id+3], ze4 = mesh->EZ[id+3];

    /* Jacobian matrix */
    dfloat xr = 0.5*(xe2-xe1), xs = 0.5*(xe3-xe1), xt = 0.5*(xe4-xe1);
    dfloat yr = 0.5*(ye2-ye1), ys = 0.5*(ye3-ye1), yt = 0.5*(ye4-ye1);
    dfloat zr = 0.5*(ze2-ze1), zs = 0.5*(ze3-ze1), zt = 0.5*(ze4-ze1);

    /* compute geometric factors for affine coordinate transform*/
    dfloat J = xr*(ys*zt-zs*yt) - yr*(xs*zt-zs*xt) + zr*(xs*yt-ys*xt);
    
    dfloat rx =  (ys*zt - zs*yt)/J, ry = -(xs*zt - zs*xt)/J, rz =  (xs*yt - ys*xt)/J;
    dfloat sx = -(yr*zt - zr*yt)/J, sy =  (xr*zt - zr*xt)/J, sz = -(xr*yt - yr*xt)/J;
    dfloat tx =  (yr*zs - zr*ys)/J, ty = -(xr*zs - zr*xs)/J, tz =  (xr*ys - yr*xs)/J;

    if(J<0) printf("bugger: got negative geofac\n");
    minJ = mymin(minJ,J);
    maxJ = mymax(maxJ,J);
    
    /* store geometric factors */
    mesh->vgeo[mesh->Nvgeo*e + RXID] = rx;
    mesh->vgeo[mesh->Nvgeo*e + RYID] = ry;
    mesh->vgeo[mesh->Nvgeo*e + RZID] = rz;
    mesh->vgeo[mesh->Nvgeo*e + SXID] = sx;
    mesh->vgeo[mesh->Nvgeo*e + SYID] = sy;
    mesh->vgeo[mesh->Nvgeo*e + SZID] = sz;
    mesh->vgeo[mesh->Nvgeo*e + TXID] = tx;
    mesh->vgeo[mesh->Nvgeo*e + TYID] = ty;
    mesh->vgeo[mesh->Nvgeo*e + TZID] = tz;
    mesh->vgeo[mesh->Nvgeo*e +  JID] = J;
    //    printf("geo: %g,%g,%g - %g,%g,%g - %g,%g,%g\n",
    //     rx,ry,rz, sx,sy,sz, tx,ty,tz);

    /* store second order geometric factors */
    mesh->ggeo[mesh->Nggeo*e + G00ID] = J*(rx*rx + ry*ry + rz*rz);
    mesh->ggeo[mesh->Nggeo*e + G01ID] = J*(rx*sx + ry*sy + rz*sz);
    mesh->ggeo[mesh->Nggeo*e + G02ID] = J*(rx*tx + ry*ty + rz*tz);
    mesh->ggeo[mesh->Nggeo*e + G11ID] = J*(sx*sx + sy*sy + sz*sz);
    mesh->ggeo[mesh->Nggeo*e + G12ID] = J*(sx*tx + sy*ty + sz*tz);
    mesh->ggeo[mesh->Nggeo*e + G22ID] = J*(tx*tx + ty*ty + tz*tz);
    mesh->ggeo[mesh->Nggeo*e + GWJID] = J;

   for(int n=0;n<mesh->cubNp;++n){
      dfloat cubrn = mesh->cubr[n];
      dfloat cubsn = mesh->cubs[n];
      dfloat cubtn = mesh->cubt[n];
      dfloat ar = -2./(cubsn+cubtn);
      dfloat as = 2*(1.+cubrn)/pow(cubsn+cubtn,2);
      dfloat at = 2*(1.+cubrn)/pow(cubsn+cubtn,2);
      dfloat br = 0;
      dfloat bs = 2./(1-cubtn);
      dfloat bt = 2.*(1+cubsn)/pow(1-cubtn,2);
      dfloat cr = 0;
      dfloat cs = 0;
      dfloat ct = 1.0;

      dfloat ax = rx*ar + sx*as + tx*at;
      dfloat bx = rx*br + sx*bs + tx*bt;
      dfloat cx = rx*cr + sx*cs + tx*ct;

      dfloat ay = ry*ar + sy*as + ty*at;
      dfloat by = ry*br + sy*bs + ty*bt;
      dfloat cy = ry*cr + sy*cs + ty*ct;

      dfloat az = rz*ar + sz*as + tz*at;
      dfloat bz = rz*br + sz*bs + tz*bt;
      dfloat cz = rz*cr + sz*cs + tz*ct;

      dfloat JW = J*mesh->cubw[n];
      dfloat *gbase = mesh->cubggeo + mesh->Nggeo*mesh->cubNp*e + n;
      gbase[mesh->cubNp*G00ID] = JW*(ax*ax + ay*ay + az*az); // 
      gbase[mesh->cubNp*G01ID] = JW*(ax*bx + ay*by + az*bz);
      gbase[mesh->cubNp*G02ID] = JW*(ax*cx + ay*cy + az*cz);
      gbase[mesh->cubNp*G11ID] = JW*(bx*bx + by*by + bz*bz);
      gbase[mesh->cubNp*G12ID] = JW*(bx*cx + by*cy + bz*cz);
      gbase[mesh->cubNp*G22ID] = JW*(cx*cx + cy*cy + cz*cz);
      gbase[mesh->cubNp*GWJID] = JW;
      //      printf("% e ", JW);
   }
   //   printf("\n"); 
  }

  //printf("minJ = %g, maxJ = %g\n", minJ, maxJ);
}


void meshGeometricFactorsHex3D(mesh3D *mesh){

  /* unified storage array for geometric factors */
  mesh->Nvgeo   = 12;
  mesh->vgeo    = (dfloat*) calloc(mesh->Nelements*mesh->Nvgeo*mesh->Np,    sizeof(dfloat));
  mesh->cubvgeo = (dfloat*) calloc(mesh->Nelements*mesh->Nvgeo*mesh->cubNp, sizeof(dfloat));

  /* number of second order geometric factors */
  mesh->Nggeo   = 7;
  mesh->ggeo    = (dfloat*) calloc(mesh->Nelements*mesh->Nggeo*mesh->Np,    sizeof(dfloat));
  mesh->cubggeo = (dfloat*) calloc(mesh->Nelements*mesh->Nggeo*mesh->cubNp, sizeof(dfloat));
  
  dfloat minJ = 1e9, maxJ = -1e9, maxSkew = 0;

  dfloat *xre = (dfloat*) calloc(mesh->Np, sizeof(dfloat));
  dfloat *xse = (dfloat*) calloc(mesh->Np, sizeof(dfloat));
  dfloat *xte = (dfloat*) calloc(mesh->Np, sizeof(dfloat));
  dfloat *yre = (dfloat*) calloc(mesh->Np, sizeof(dfloat));
  dfloat *yse = (dfloat*) calloc(mesh->Np, sizeof(dfloat));
  dfloat *yte = (dfloat*) calloc(mesh->Np, sizeof(dfloat));
  dfloat *zre = (dfloat*) calloc(mesh->Np, sizeof(dfloat));
  dfloat *zse = (dfloat*) calloc(mesh->Np, sizeof(dfloat));
  dfloat *zte = (dfloat*) calloc(mesh->Np, sizeof(dfloat));
  
  dfloat *cubxre = (dfloat*) calloc(mesh->cubNp, sizeof(dfloat));
  dfloat *cubxse = (dfloat*) calloc(mesh->cubNp, sizeof(dfloat));
  dfloat *cubxte = (dfloat*) calloc(mesh->cubNp, sizeof(dfloat));
  dfloat *cubyre = (dfloat*) calloc(mesh->cubNp, sizeof(dfloat));
  dfloat *cubyse = (dfloat*) calloc(mesh->cubNp, sizeof(dfloat));
  dfloat *cubyte = (dfloat*) calloc(mesh->cubNp, sizeof(dfloat));
  dfloat *cubzre = (dfloat*) calloc(mesh->cubNp, sizeof(dfloat));
  dfloat *cubzse = (dfloat*) calloc(mesh->cubNp, sizeof(dfloat));
  dfloat *cubzte = (dfloat*) calloc(mesh->cubNp, sizeof(dfloat));

  for(dlong e=0;e<mesh->Nelements;++e){ /* for each element */

    /* find vertex indices and physical coordinates */
    dlong id = e*mesh->Nverts;
    
    dfloat *xe = mesh->EX + id;
    dfloat *ye = mesh->EY + id;
    dfloat *ze = mesh->EZ + id;

    for(int n=0;n<mesh->Np;++n){
      xre[n] = 0; xse[n] = 0; xte[n] = 0;
      yre[n] = 0; yse[n] = 0; yte[n] = 0; 
      zre[n] = 0; zse[n] = 0; zte[n] = 0; 
    }
    
    for(int k=0;k<mesh->Nq;++k){
      for(int j=0;j<mesh->Nq;++j){
        for(int i=0;i<mesh->Nq;++i){
          
          int n = i + j*mesh->Nq + k*mesh->Nq*mesh->Nq;

          /* local node coordinates */
          dfloat rn = mesh->r[n]; 
          dfloat sn = mesh->s[n];
          dfloat tn = mesh->t[n];

	  for(int m=0;m<mesh->Nq;++m){
	    int idr = e*mesh->Np + k*mesh->Nq*mesh->Nq + j*mesh->Nq + m;
	    int ids = e*mesh->Np + k*mesh->Nq*mesh->Nq + m*mesh->Nq + i;
	    int idt = e*mesh->Np + m*mesh->Nq*mesh->Nq + j*mesh->Nq + i;
	    xre[n] += mesh->D[i*mesh->Nq+m]*mesh->x[idr];
	    xse[n] += mesh->D[j*mesh->Nq+m]*mesh->x[ids];
	    xte[n] += mesh->D[k*mesh->Nq+m]*mesh->x[idt];
	    yre[n] += mesh->D[i*mesh->Nq+m]*mesh->y[idr];
	    yse[n] += mesh->D[j*mesh->Nq+m]*mesh->y[ids];
	    yte[n] += mesh->D[k*mesh->Nq+m]*mesh->y[idt];
	    zre[n] += mesh->D[i*mesh->Nq+m]*mesh->z[idr];
	    zse[n] += mesh->D[j*mesh->Nq+m]*mesh->z[ids];
	    zte[n] += mesh->D[k*mesh->Nq+m]*mesh->z[idt];
	  }
	  
	  dfloat xr = xre[n], xs = xse[n], xt = xte[n];
	  dfloat yr = yre[n], ys = yse[n], yt = yte[n];
	  dfloat zr = zre[n], zs = zse[n], zt = zte[n];
	  
          /* compute geometric factors for affine coordinate transform*/
          dfloat J = xr*(ys*zt-zs*yt) - yr*(xs*zt-zs*xt) + zr*(xs*yt-ys*xt);

          dfloat hr = sqrt(xr*xr+yr*yr+zr*zr);
          dfloat hs = sqrt(xs*xs+ys*ys+zs*zs);
          dfloat ht = sqrt(xt*xt+yt*yt+zt*zt);
          minJ = mymin(J, minJ);
          maxJ = mymax(J, maxJ);
          maxSkew = mymax(maxSkew, hr/hs);
          maxSkew = mymax(maxSkew, hr/ht);
          maxSkew = mymax(maxSkew, hs/hr);
          maxSkew = mymax(maxSkew, hs/ht);
          maxSkew = mymax(maxSkew, ht/hr);
          maxSkew = mymax(maxSkew, ht/hs);
          
          if(J<1e-12) printf("J = %g !!!!!!!!!!!!!\n", J);
          
          dfloat rx =  (ys*zt - zs*yt)/J, ry = -(xs*zt - zs*xt)/J, rz =  (xs*yt - ys*xt)/J;
          dfloat sx = -(yr*zt - zr*yt)/J, sy =  (xr*zt - zr*xt)/J, sz = -(xr*yt - yr*xt)/J;
          dfloat tx =  (yr*zs - zr*ys)/J, ty = -(xr*zs - zr*xs)/J, tz =  (xr*ys - yr*xs)/J;
          
          dfloat JW = J*mesh->gllw[i]*mesh->gllw[j]*mesh->gllw[k];
          
          /* store geometric factors */
          mesh->vgeo[mesh->Nvgeo*mesh->Np*e + n + mesh->Np*RXID] = rx;
          mesh->vgeo[mesh->Nvgeo*mesh->Np*e + n + mesh->Np*RYID] = ry;
          mesh->vgeo[mesh->Nvgeo*mesh->Np*e + n + mesh->Np*RZID] = rz;
          
          mesh->vgeo[mesh->Nvgeo*mesh->Np*e + n + mesh->Np*SXID] = sx;
          mesh->vgeo[mesh->Nvgeo*mesh->Np*e + n + mesh->Np*SYID] = sy;
          mesh->vgeo[mesh->Nvgeo*mesh->Np*e + n + mesh->Np*SZID] = sz;
          
          mesh->vgeo[mesh->Nvgeo*mesh->Np*e + n + mesh->Np*TXID] = tx;
          mesh->vgeo[mesh->Nvgeo*mesh->Np*e + n + mesh->Np*TYID] = ty;
          mesh->vgeo[mesh->Nvgeo*mesh->Np*e + n + mesh->Np*TZID] = tz;
          
          mesh->vgeo[mesh->Nvgeo*mesh->Np*e + n + mesh->Np*JID]  = J;
          mesh->vgeo[mesh->Nvgeo*mesh->Np*e + n + mesh->Np*JWID] = JW;
          mesh->vgeo[mesh->Nvgeo*mesh->Np*e + n + mesh->Np*IJWID] = 1./JW;

          /* store second order geometric factors */
          mesh->ggeo[mesh->Nggeo*mesh->Np*e + n + mesh->Np*G00ID] = JW*(rx*rx + ry*ry + rz*rz);
          mesh->ggeo[mesh->Nggeo*mesh->Np*e + n + mesh->Np*G01ID] = JW*(rx*sx + ry*sy + rz*sz);
          mesh->ggeo[mesh->Nggeo*mesh->Np*e + n + mesh->Np*G02ID] = JW*(rx*tx + ry*ty + rz*tz);
          mesh->ggeo[mesh->Nggeo*mesh->Np*e + n + mesh->Np*G11ID] = JW*(sx*sx + sy*sy + sz*sz);
          mesh->ggeo[mesh->Nggeo*mesh->Np*e + n + mesh->Np*G12ID] = JW*(sx*tx + sy*ty + sz*tz);
          mesh->ggeo[mesh->Nggeo*mesh->Np*e + n + mesh->Np*G22ID] = JW*(tx*tx + ty*ty + tz*tz);
          mesh->ggeo[mesh->Nggeo*mesh->Np*e + n + mesh->Np*GWJID] = JW;
        }
      }
    }

    meshInterpolateHex3D(mesh->cubInterp, xre, mesh->Nq, cubxre, mesh->cubNq);
    meshInterpolateHex3D(mesh->cubInterp, xse, mesh->Nq, cubxse, mesh->cubNq);
    meshInterpolateHex3D(mesh->cubInterp, xte, mesh->Nq, cubxte, mesh->cubNq);

    meshInterpolateHex3D(mesh->cubInterp, yre, mesh->Nq, cubyre, mesh->cubNq);
    meshInterpolateHex3D(mesh->cubInterp, yse, mesh->Nq, cubyse, mesh->cubNq);
    meshInterpolateHex3D(mesh->cubInterp, yte, mesh->Nq, cubyte, mesh->cubNq);

    meshInterpolateHex3D(mesh->cubInterp, zre, mesh->Nq, cubzre, mesh->cubNq);
    meshInterpolateHex3D(mesh->cubInterp, zse, mesh->Nq, cubzse, mesh->cubNq);
    meshInterpolateHex3D(mesh->cubInterp, zte, mesh->Nq, cubzte, mesh->cubNq);
    
    //geometric data for quadrature
    for(int k=0;k<mesh->cubNq;++k){
      for(int j=0;j<mesh->cubNq;++j){
        for(int i=0;i<mesh->cubNq;++i){

	  int n = k*mesh->cubNq*mesh->cubNq + j*mesh->cubNq + i;
	  
          dfloat rn = mesh->cubr[i];
          dfloat sn = mesh->cubr[j];
          dfloat tn = mesh->cubr[k];
	  
          /* Jacobian matrix */
	  dfloat xr = cubxre[n], xs = cubxse[n], xt = cubxte[n];
	  dfloat yr = cubyre[n], ys = cubyse[n], yt = cubyte[n];
	  dfloat zr = cubzre[n], zs = cubzse[n], zt = cubzte[n];
	  
	  /* compute geometric factors for affine coordinate transform*/
          dfloat J = xr*(ys*zt-zs*yt) - yr*(xs*zt-zs*xt) + zr*(xs*yt-ys*xt);

          if(J<1e-12) printf("CUBATURE J = %g !!!!!!!!!!!!!\n", J);
	  
          dfloat rx =  (ys*zt - zs*yt)/J, ry = -(xs*zt - zs*xt)/J, rz =  (xs*yt - ys*xt)/J;
          dfloat sx = -(yr*zt - zr*yt)/J, sy =  (xr*zt - zr*xt)/J, sz = -(xr*yt - yr*xt)/J;
          dfloat tx =  (yr*zs - zr*ys)/J, ty = -(xr*zs - zr*xs)/J, tz =  (xr*ys - yr*xs)/J;
          
          dfloat JW = J*mesh->cubw[i]*mesh->cubw[j]*mesh->cubw[k];
          
          /* store geometric factors */
          dlong base = mesh->Nvgeo*mesh->cubNp*e + n;
          mesh->cubvgeo[base + mesh->cubNp*RXID] = rx;
          mesh->cubvgeo[base + mesh->cubNp*RYID] = ry;
          mesh->cubvgeo[base + mesh->cubNp*RZID] = rz;
          
          mesh->cubvgeo[base + mesh->cubNp*SXID] = sx;
          mesh->cubvgeo[base + mesh->cubNp*SYID] = sy;
          mesh->cubvgeo[base + mesh->cubNp*SZID] = sz;
          
          mesh->cubvgeo[base + mesh->cubNp*TXID] = tx;
          mesh->cubvgeo[base + mesh->cubNp*TYID] = ty;
          mesh->cubvgeo[base + mesh->cubNp*TZID] = tz;
          
          mesh->cubvgeo[base + mesh->cubNp*JID]  = J;
          mesh->cubvgeo[base + mesh->cubNp*JWID] = JW;
          mesh->cubvgeo[base + mesh->cubNp*IJWID] = 1./JW;


          /* store second order geometric factors */
	  base = mesh->Nggeo*mesh->cubNp*e + n;
          mesh->cubggeo[base + mesh->cubNp*G00ID] = JW*(rx*rx + ry*ry + rz*rz);
          mesh->cubggeo[base + mesh->cubNp*G01ID] = JW*(rx*sx + ry*sy + rz*sz);
          mesh->cubggeo[base + mesh->cubNp*G02ID] = JW*(rx*tx + ry*ty + rz*tz);
          mesh->cubggeo[base + mesh->cubNp*G11ID] = JW*(sx*sx + sy*sy + sz*sz);
          mesh->cubggeo[base + mesh->cubNp*G12ID] = JW*(sx*tx + sy*ty + sz*tz);
          mesh->cubggeo[base + mesh->cubNp*G22ID] = JW*(tx*tx + ty*ty + tz*tz);
          mesh->cubggeo[base + mesh->cubNp*GWJID] = JW;
	  
        }
      }
    }
  }

  {
    dfloat globalMinJ, globalMaxJ, globalMaxSkew;

    MPI_Reduce(&minJ, &globalMinJ, 1, MPI_DFLOAT, MPI_MIN, 0, mesh->comm);
    MPI_Reduce(&maxJ, &globalMaxJ, 1, MPI_DFLOAT, MPI_MAX, 0, mesh->comm);
    MPI_Reduce(&maxSkew, &globalMaxSkew, 1, MPI_DFLOAT, MPI_MAX, 0, mesh->comm);

    if(mesh->rank==0)
      printf("J in range [%g,%g] and max Skew = %g\n", globalMinJ, globalMaxJ, globalMaxSkew);
  }

  free(xre); free(xse); free(xte);
  free(yre); free(yse); free(yte);
  free(zre); free(zse); free(zte);

  free(cubxre); free(cubxse); free(cubxte);
  free(cubyre); free(cubyse); free(cubyte);
  free(cubzre); free(cubzse); free(cubzte);
  
}

// 20 bits per coordinate
#define bitRange 20

// spread bits of i by introducing zeros between binary bits
unsigned long long int bitSplitter3D(unsigned int i){
  
  unsigned long long int mask = 1;
  unsigned long long int li = i;
  unsigned long long int lj = 0;
  
  for(int b=0;b<bitRange;++b){
    lj |= ((li & mask) << 2*b); // bit b moves to bit 3b
    mask <<= 1;
  }
  
  return lj;
}

// compute Morton index of (ix,iy) relative to a bitRange x bitRange  Morton lattice
unsigned long long int mortonIndex3D(unsigned int ix, unsigned int iy, unsigned int iz){
  
  // spread bits of ix apart (introduce zeros)
  unsigned long long int sx = bitSplitter3D(ix);
  unsigned long long int sy = bitSplitter3D(iy);
  unsigned long long int sz = bitSplitter3D(iz);
  
  // interleave bits of ix and iy
  unsigned long long int mi = sx | (sy<<1) | (sz<<2); 
  
  return mi;
}

// capsule for element vertices + Morton index
typedef struct {
  
  unsigned long long int index;
  
  dlong element;

  int type;

  // use 8 for maximum vertices per element
  hlong v[8];

  dfloat EX[8], EY[8], EZ[8];

}element_t;

// compare the Morton indices for two element capsules
int compareElements(const void *a, const void *b){

  element_t *ea = (element_t*) a;
  element_t *eb = (element_t*) b;
  
  if(ea->index < eb->index) return -1;
  if(ea->index > eb->index) return  1;
  
  return 0;

}

// stub for the match function needed by parallelSort
void bogusMatch3D(void *a, void *b){ }

// geometric partition of elements in 3D mesh using Morton ordering + parallelSort
void meshGeometricPartition3D(mesh3D *mesh){

  int rank, size;
  rank = mesh->rank;
  size = mesh->size;

  dlong maxNelements;
  MPI_Allreduce(&(mesh->Nelements), &maxNelements, 1, MPI_DLONG, MPI_MAX,
		mesh->comm);
  maxNelements = 2*((maxNelements+1)/2);
  
  // fix maxNelements
  element_t *elements 
    = (element_t*) calloc(maxNelements, sizeof(element_t));

  // local bounding box of element centers
  dfloat minvx = 1e9, maxvx = -1e9;
  dfloat minvy = 1e9, maxvy = -1e9;
  dfloat minvz = 1e9, maxvz = -1e9;

  // compute element centers on this process
  for(dlong n=0;n<mesh->Nverts*mesh->Nelements;++n){
    minvx = mymin(minvx, mesh->EX[n]);
    maxvx = mymax(maxvx, mesh->EX[n]);
    minvy = mymin(minvy, mesh->EY[n]);
    maxvy = mymax(maxvy, mesh->EY[n]);
    minvz = mymin(minvz, mesh->EZ[n]);
    maxvz = mymax(maxvz, mesh->EZ[n]);
  }
  
  // find global bounding box of element centers
  dfloat gminvx, gminvy, gminvz, gmaxvx, gmaxvy, gmaxvz;
  MPI_Allreduce(&minvx, &gminvx, 1, MPI_DFLOAT, MPI_MIN, mesh->comm);
  MPI_Allreduce(&minvy, &gminvy, 1, MPI_DFLOAT, MPI_MIN, mesh->comm);
  MPI_Allreduce(&minvz, &gminvz, 1, MPI_DFLOAT, MPI_MIN, mesh->comm);
  MPI_Allreduce(&maxvx, &gmaxvx, 1, MPI_DFLOAT, MPI_MAX, mesh->comm);
  MPI_Allreduce(&maxvy, &gmaxvy, 1, MPI_DFLOAT, MPI_MAX, mesh->comm);
  MPI_Allreduce(&maxvz, &gmaxvz, 1, MPI_DFLOAT, MPI_MAX, mesh->comm);

  // choose sub-range of Morton lattice coordinates to embed element centers in
  unsigned long long int Nboxes = (((unsigned long long int)1)<<(bitRange-1));
  
  // compute Morton index for each element
  for(dlong e=0;e<mesh->Nelements;++e){

    // element center coordinates
    dfloat cx = 0, cy = 0, cz = 0;
    for(int n=0;n<mesh->Nverts;++n){
      cx += mesh->EX[e*mesh->Nverts+n];
      cy += mesh->EY[e*mesh->Nverts+n];
      cz += mesh->EZ[e*mesh->Nverts+n];
    }
    cx /= mesh->Nverts;
    cy /= mesh->Nverts;
    cz /= mesh->Nverts;

    // encapsulate element, vertices, Morton index, vertex coordinates
    elements[e].element = e;
    for(int n=0;n<mesh->Nverts;++n){
      elements[e].v[n] = mesh->EToV[e*mesh->Nverts+n];
      elements[e].EX[n] = mesh->EX[e*mesh->Nverts+n];
      elements[e].EY[n] = mesh->EY[e*mesh->Nverts+n];
      elements[e].EZ[n] = mesh->EZ[e*mesh->Nverts+n];
    }

    elements[e].type = mesh->elementInfo[e];

    dfloat maxlength = mymax(gmaxvx-gminvx, mymax(gmaxvy-gminvy, gmaxvz-gminvz));

    // avoid stretching axes
    unsigned long long int ix = (cx-gminvx)*Nboxes/maxlength;
    unsigned long long int iy = (cy-gminvy)*Nboxes/maxlength;
    unsigned long long int iz = (cz-gminvz)*Nboxes/maxlength;
			
    elements[e].index = mortonIndex3D(ix, iy, iz);
  }

  // pad element array with dummy elements
  for(dlong e=mesh->Nelements;e<maxNelements;++e){
    elements[e].element = -1;
    elements[e].index = mortonIndex3D(Nboxes+1, Nboxes+1, Nboxes+1);
  }

  // odd-even parallel sort of element capsules based on their Morton index
  parallelSort(mesh->size, mesh->rank, mesh->comm,
	       maxNelements, elements, sizeof(element_t),
	       compareElements, 
	       bogusMatch3D);

#if 0
  // count number of elements that end up on this process
  int cnt = 0;
  for(int e=0;e<maxNelements;++e)
    cnt += (elements[e].element != -1);

  // reset number of elements and element-to-vertex connectivity from returned capsules
  free(mesh->EToV);
  free(mesh->EX);
  free(mesh->EY);
  free(mesh->EZ);

  mesh->Nelements = cnt;
  mesh->EToV = (int*) calloc(cnt*mesh->Nverts, sizeof(int));
  mesh->EX = (dfloat*) calloc(cnt*mesh->Nverts, sizeof(dfloat));
  mesh->EY = (dfloat*) calloc(cnt*mesh->Nverts, sizeof(dfloat));
  mesh->EZ = (dfloat*) calloc(cnt*mesh->Nverts, sizeof(dfloat));

  cnt = 0;
  for(int e=0;e<maxNelements;++e){
    if(elements[e].element != -1){
      for(int n=0;n<mesh->Nverts;++n){
	mesh->EToV[cnt*mesh->Nverts + n] = elements[e].v[n];
	mesh->EX[cnt*mesh->Nverts + n]   = elements[e].EX[n];
	mesh->EY[cnt*mesh->Nverts + n]   = elements[e].EY[n];
	mesh->EZ[cnt*mesh->Nverts + n]   = elements[e].EZ[n];
      }
      ++cnt;
    }
  }
#else
  // compress and renumber elements
  dlong sk  = 0;
  for(dlong e=0;e<maxNelements;++e){
    if(elements[e].element != -1){
      elements[sk] = elements[e];
      ++sk;
    }
  }

  dlong localNelements = sk;

  /// redistribute elements to improve balancing
  dlong *globalNelements = (dlong *) calloc(size,sizeof(dlong));
  hlong *starts = (hlong *) calloc(size+1,sizeof(hlong));

  MPI_Allgather(&localNelements, 1, MPI_DLONG, globalNelements, 1,  MPI_DLONG, mesh->comm);

  for(int r=0;r<size;++r)
    starts[r+1] = starts[r]+globalNelements[r];

  hlong allNelements = starts[size];

  // decide how many to keep on each process
  hlong chunk = allNelements/size;
  int remainder = (int) (allNelements - chunk*size);

  int *Nsend = (int *) calloc(size, sizeof(int));
  int *Nrecv = (int *) calloc(size, sizeof(int));
  // int *Ncount = (int *) calloc(size, sizeof(int));
  int *sendOffsets = (int*) calloc(size, sizeof(int));
  int *recvOffsets = (int*) calloc(size, sizeof(int));


  // Make the MPI_ELEMENT_T data type
  MPI_Datatype MPI_ELEMENT_T;
  MPI_Datatype dtype[7] = {MPI_LONG_LONG_INT, MPI_DLONG, MPI_INT,
                            MPI_HLONG, MPI_DFLOAT, MPI_DFLOAT, MPI_DFLOAT};
  int blength[7] = {1, 1, 1, 8, 8, 8, 8};
  MPI_Aint addr[7], displ[7];
  MPI_Get_address ( &(elements[0]        ), addr+0);
  MPI_Get_address ( &(elements[0].element), addr+1);
  MPI_Get_address ( &(elements[0].type   ), addr+2);
  MPI_Get_address ( &(elements[0].v[0]   ), addr+3);
  MPI_Get_address ( &(elements[0].EX[0]  ), addr+4);
  MPI_Get_address ( &(elements[0].EY[0]  ), addr+5);
  MPI_Get_address ( &(elements[0].EZ[0]  ), addr+6);
  displ[0] = 0;
  displ[1] = addr[1] - addr[0];
  displ[2] = addr[2] - addr[0];
  displ[3] = addr[3] - addr[0];
  displ[4] = addr[4] - addr[0];
  displ[5] = addr[5] - addr[0];
  displ[6] = addr[6] - addr[0];
  MPI_Type_create_struct (7, blength, displ, dtype, &MPI_ELEMENT_T);
  MPI_Type_commit (&MPI_ELEMENT_T);


  for(dlong e=0;e<localNelements;++e){

    // global element index
    elements[e].element = starts[rank]+e;

    // 0, chunk+1, 2*(chunk+1) ..., remainder*(chunk+1), remainder*(chunk+1) + chunk
    int r;
    if(elements[e].element<remainder*(chunk+1))
      r = elements[e].element/(chunk+1);
    else
      r = remainder + ((elements[e].element-remainder*(chunk+1))/chunk);

    ++Nsend[r];
  }

  // find send offsets
  for(int r=1;r<size;++r)
    sendOffsets[r] = sendOffsets[r-1] + Nsend[r-1];

  // exchange byte counts
  MPI_Alltoall(Nsend, 1, MPI_INT, Nrecv, 1, MPI_INT, mesh->comm);

  // count incoming clusters
  dlong newNelements = 0;
  for(int r=0;r<size;++r)
    newNelements += Nrecv[r];

  for(int r=1;r<size;++r)
    recvOffsets[r] = recvOffsets[r-1] + Nrecv[r-1];

  element_t *tmpElements = (element_t *) calloc(newNelements, sizeof(element_t));

  // exchange parallel clusters
  MPI_Alltoallv(elements, Nsend, sendOffsets, MPI_ELEMENT_T,
                tmpElements, Nrecv, recvOffsets, MPI_ELEMENT_T, mesh->comm);

  MPI_Barrier(mesh->comm);
  MPI_Type_free(&MPI_ELEMENT_T);

  // replace elements with inbound elements
  if (elements) free(elements);
  elements = tmpElements;

  // reset number of elements and element-to-vertex connectivity from returned capsules
  free(mesh->EToV);
  free(mesh->EX);
  free(mesh->EY);
  free(mesh->EZ);
  free(mesh->elementInfo);

  mesh->Nelements = newNelements;
  mesh->EToV = (hlong*) calloc(newNelements*mesh->Nverts, sizeof(hlong));
  mesh->EX = (dfloat*) calloc(newNelements*mesh->Nverts, sizeof(dfloat));
  mesh->EY = (dfloat*) calloc(newNelements*mesh->Nverts, sizeof(dfloat));
  mesh->EZ = (dfloat*) calloc(newNelements*mesh->Nverts, sizeof(dfloat));
  mesh->elementInfo = (hlong*) calloc(newNelements, sizeof(hlong));

  for(dlong e=0;e<newNelements;++e){
    for(int n=0;n<mesh->Nverts;++n){
      mesh->EToV[e*mesh->Nverts + n] = elements[e].v[n];
      mesh->EX[e*mesh->Nverts + n]   = elements[e].EX[n];
      mesh->EY[e*mesh->Nverts + n]   = elements[e].EY[n];
      mesh->EZ[e*mesh->Nverts + n]   = elements[e].EZ[n];
    }
    mesh->elementInfo[e] = elements[e].type;
  }
  if (elements) free(elements);
#endif
}

typedef struct {
  
  dlong element, elementN;
  int face, faceN, rankN;

}facePair_t;

/* comparison function that orders halo element/face
   based on their indexes */
int compareHaloFaces(const void *a, 
                     const void *b){
  
  facePair_t *fa = (facePair_t*) a;
  facePair_t *fb = (facePair_t*) b;
  
  if(fa->rankN < fb->rankN) return -1;
  if(fa->rankN > fb->rankN) return +1;

  if(fa->elementN < fb->elementN) return -1;
  if(fa->elementN > fb->elementN) return +1;

  if(fa->faceN < fb->faceN) return -1;
  if(fa->faceN > fb->faceN) return +1;
  
  return 0;
}


// set up halo infomation for inter-processor MPI 
// exchange of trace nodes
void meshHaloSetup(mesh_t *mesh){

  // MPI info
  int rank, size;
  rank = mesh->rank;
  size = mesh->size;

  // non-blocking MPI isend/irecv requests (used in meshHaloExchange)
  mesh->haloSendRequests = calloc(size, sizeof(MPI_Request));
  mesh->haloRecvRequests = calloc(size, sizeof(MPI_Request));
  
  // count number of halo element nodes to swap
  mesh->totalHaloPairs = 0;
  mesh->NhaloPairs = (int*) calloc(size, sizeof(int));
  for(dlong e=0;e<mesh->Nelements;++e){
    for(int f=0;f<mesh->Nfaces;++f){
      int r = mesh->EToP[e*mesh->Nfaces+f]; // rank of neighbor
      if(r!=-1){
        mesh->totalHaloPairs += 1;
        mesh->NhaloPairs[r] += 1;
      }
    }
  }

  // count number of MPI messages in halo exchange
  mesh->NhaloMessages = 0;
  for(int r=0;r<size;++r)
    if(mesh->NhaloPairs[r])
      ++mesh->NhaloMessages;

  // create a list of element/faces with halo neighbor
  facePair_t *haloElements = 
    (facePair_t*) calloc(mesh->totalHaloPairs, sizeof(facePair_t));

  dlong cnt = 0;
  for(dlong e=0;e<mesh->Nelements;++e){
    for(int f=0;f<mesh->Nfaces;++f){
      dlong ef = e*mesh->Nfaces+f;
      if(mesh->EToP[ef]!=-1){
        haloElements[cnt].element  = e;
        haloElements[cnt].face     = f;
        haloElements[cnt].elementN = mesh->EToE[ef];
        haloElements[cnt].faceN    = mesh->EToF[ef];
        haloElements[cnt].rankN    = mesh->EToP[ef];
        ++cnt;
      }
    }
  }
  
  // sort the face pairs in order the destination requires
  qsort(haloElements, mesh->totalHaloPairs, sizeof(facePair_t), compareHaloFaces);

  // record the outgoing order for elements
  mesh->haloElementList = (dlong*) calloc(mesh->totalHaloPairs, sizeof(dlong));
  for(dlong i=0;i<mesh->totalHaloPairs;++i){
    dlong e = haloElements[i].element;
    mesh->haloElementList[i] = e;
  }

  // record the outgoing node ids for trace nodes
  mesh->haloGetNodeIds = (dlong*) calloc(mesh->totalHaloPairs*mesh->Nfp, sizeof(dlong));
  mesh->haloPutNodeIds = (dlong*) calloc(mesh->totalHaloPairs*mesh->Nfp, sizeof(dlong));
  
  cnt = 0;
  for(dlong i=0;i<mesh->totalHaloPairs;++i){
    dlong eM = haloElements[i].element;
    int fM = haloElements[i].face;
    int fP = haloElements[i].faceN;
    for(int n=0;n<mesh->Nfp;++n){
      mesh->haloGetNodeIds[cnt] = eM*mesh->Np + mesh->faceNodes[fM*mesh->Nfp+n];
      ++cnt;
    }
  }

  // now arrange for incoming nodes
  cnt = mesh->Nelements;
  dlong ncnt = 0;
  for(int r=0;r<size;++r){
    for(dlong e=0;e<mesh->Nelements;++e){
      for(int f=0;f<mesh->Nfaces;++f){
	dlong ef = e*mesh->Nfaces+f;
	if(mesh->EToP[ef]==r){
	  mesh->EToE[ef] = cnt;
	  int fP = mesh->EToF[ef];
	  for(int n=0;n<mesh->Nfp;++n){
	    mesh->haloPutNodeIds[ncnt] = cnt*mesh->Np + mesh->faceNodes[fP*mesh->Nfp+n];
	    ++ncnt;
	  }
	  ++cnt; // next halo element
	}
      }
    }
  }
  
  // create halo extension for x,y arrays
  dlong totalHaloNodes = mesh->totalHaloPairs*mesh->Np;
  dlong localNodes     = mesh->Nelements*mesh->Np;

  // temporary send buffer
  dfloat *sendBuffer = (dfloat*) calloc(totalHaloNodes, sizeof(dfloat));

  // extend x,y arrays to hold coordinates of node coordinates of elements in halo
  mesh->x = (dfloat*) realloc(mesh->x, (localNodes+totalHaloNodes)*sizeof(dfloat));
  mesh->y = (dfloat*) realloc(mesh->y, (localNodes+totalHaloNodes)*sizeof(dfloat));
  if(mesh->dim==3)
    mesh->z = (dfloat*) realloc(mesh->z, (localNodes+totalHaloNodes)*sizeof(dfloat));
  
  // send halo data and recv into extended part of arrays
  meshHaloExchange(mesh, mesh->Np*sizeof(dfloat), mesh->x, sendBuffer, mesh->x + localNodes);
  meshHaloExchange(mesh, mesh->Np*sizeof(dfloat), mesh->y, sendBuffer, mesh->y + localNodes);
  if(mesh->dim==3)
    meshHaloExchange(mesh, mesh->Np*sizeof(dfloat), mesh->z, sendBuffer, mesh->z + localNodes);   

  // grab EX,EY,EZ from halo
  mesh->EX = (dfloat*) realloc(mesh->EX, (mesh->Nelements+mesh->totalHaloPairs)*mesh->Nverts*sizeof(dfloat));
  mesh->EY = (dfloat*) realloc(mesh->EY, (mesh->Nelements+mesh->totalHaloPairs)*mesh->Nverts*sizeof(dfloat));
  if(mesh->dim==3)
    mesh->EZ = (dfloat*) realloc(mesh->EZ, (mesh->Nelements+mesh->totalHaloPairs)*mesh->Nverts*sizeof(dfloat));

  // send halo data and recv into extended part of arrays
  meshHaloExchange(mesh, mesh->Nverts*sizeof(dfloat), mesh->EX, sendBuffer, mesh->EX + mesh->Nverts*mesh->Nelements);
  meshHaloExchange(mesh, mesh->Nverts*sizeof(dfloat), mesh->EY, sendBuffer, mesh->EY + mesh->Nverts*mesh->Nelements);
  if(mesh->dim==3)
    meshHaloExchange(mesh, mesh->Nverts*sizeof(dfloat), mesh->EZ, sendBuffer, mesh->EZ + mesh->Nverts*mesh->Nelements);
  
  free(haloElements);
  free(sendBuffer);
}
void meshHaloExtract(mesh_t *mesh, size_t Nbytes, void *sourceBuffer, void *haloBuffer){
  
  // copy data from outgoing elements into temporary send buffer
  for(int i=0;i<mesh->totalHaloPairs;++i){
    // outgoing element
    int e = mesh->haloElementList[i];
    memcpy(((char*)haloBuffer)+i*Nbytes, ((char*)sourceBuffer)+e*Nbytes, Nbytes);
  }
  
}

// send data from partition boundary elements
// and receive data to ghost elements
void meshHaloExchange(mesh_t *mesh,
		      size_t Nbytes,         // message size per element
		      void *sourceBuffer,  
		      void *sendBuffer,    // temporary buffer
		      void *recvBuffer){

  // MPI info
  int rank, size;
  rank = mesh->rank;
  size = mesh->size;

  // count outgoing and incoming meshes
  int tag = 999;

  // copy data from outgoing elements into temporary send buffer
  for(int i=0;i<mesh->totalHaloPairs;++i){
    // outgoing element
    int e = mesh->haloElementList[i];
    // copy element e data to sendBuffer
    memcpy(((char*)sendBuffer)+i*Nbytes, ((char*)sourceBuffer)+e*Nbytes, Nbytes);
  }

  // initiate immediate send  and receives to each other process as needed
  int offset = 0, message = 0;
  for(int r=0;r<size;++r){
    if(r!=rank){
      size_t count = mesh->NhaloPairs[r]*Nbytes;
      if(count){
	//	printf("rank %d sending %d bytes to rank %d\n", rank, count, r);
	MPI_Irecv(((char*)recvBuffer)+offset, count, MPI_CHAR, r, tag,
		  mesh->comm, (MPI_Request*)mesh->haloRecvRequests+message);
	
	MPI_Isend(((char*)sendBuffer)+offset, count, MPI_CHAR, r, tag,
		  mesh->comm, (MPI_Request*)mesh->haloSendRequests+message);
	offset += count;
	++message;
      }
    }
  }

  //  printf("mesh->NhaloMessages = %d\n", mesh->NhaloMessages);

  // Wait for all sent messages to have left and received messages to have arrived
  MPI_Status *sendStatus = (MPI_Status*) calloc(mesh->NhaloMessages, sizeof(MPI_Status));
  MPI_Status *recvStatus = (MPI_Status*) calloc(mesh->NhaloMessages, sizeof(MPI_Status));
  
  MPI_Waitall(mesh->NhaloMessages, (MPI_Request*)mesh->haloRecvRequests, recvStatus);
  MPI_Waitall(mesh->NhaloMessages, (MPI_Request*)mesh->haloSendRequests, sendStatus);
  
  free(recvStatus);
  free(sendStatus);
}      


// start halo exchange (for q)
void meshHaloExchangeStart(mesh_t *mesh,
			     size_t Nbytes,       // message size per element
			     void *sendBuffer,    // temporary buffer
			     void *recvBuffer){

  if(mesh->totalHaloPairs>0){
    // MPI info
    //    int rank, size;
    //    MPI_Comm_rank(mesh->comm, &rank);
    //    MPI_Comm_size(mesh->comm, &size);

    int rank = mesh->rank;
    int size = mesh->size;
    
    // count outgoing and incoming meshes
    int tag = 999;
    
    // initiate immediate send  and receives to each other process as needed
    int offset = 0, message = 0;
    for(int r=0;r<size;++r){
      if(r!=rank){
	size_t count = mesh->NhaloPairs[r]*Nbytes;
	if(count){
	  MPI_Irecv(((char*)recvBuffer)+offset, count, MPI_CHAR, r, tag,
		    mesh->comm, ((MPI_Request*)mesh->haloRecvRequests)+message);
	
	  MPI_Isend(((char*)sendBuffer)+offset, count, MPI_CHAR, r, tag,
		    mesh->comm, ((MPI_Request*)mesh->haloSendRequests)+message);
	  offset += count;
	  ++message;
	}
      }
    }
  }  
}

void meshHaloExchangeFinish(mesh_t *mesh){

  if(mesh->totalHaloPairs>0){
    // Wait for all sent messages to have left and received messages to have arrived
    MPI_Status *sendStatus = (MPI_Status*) calloc(mesh->NhaloMessages, sizeof(MPI_Status));
    MPI_Status *recvStatus = (MPI_Status*) calloc(mesh->NhaloMessages, sizeof(MPI_Status));

    MPI_Waitall(mesh->NhaloMessages, (MPI_Request*)mesh->haloRecvRequests, recvStatus);
    MPI_Waitall(mesh->NhaloMessages, (MPI_Request*)mesh->haloSendRequests, sendStatus);

    free(recvStatus);
    free(sendStatus);
  }
}      


// start halo exchange (for q)
void meshHaloExchange(mesh_t *mesh,
		      size_t Nbytes,       // message size per element
		      void *sendBuffer,    // temporary buffer
		      void *recvBuffer){

  if(mesh->totalHaloPairs>0){

    int rank = mesh->rank;
    int size = mesh->size;
    
    // count outgoing and incoming meshes
    int tag = 999;

    int Nmessages = 0;
    
    for(int r=0;r<size;++r){
      if(r!=rank){
	size_t count = mesh->NhaloPairs[r]*Nbytes;
	if(count)
	  ++Nmessages;
      }
    }

    MPI_Request *sendRequests = (MPI_Request*) calloc(Nmessages, sizeof(MPI_Request));
    MPI_Request *recvRequests = (MPI_Request*) calloc(Nmessages, sizeof(MPI_Request));
  
    // initiate immediate send  and receives to each other process as needed
    int offset = 0;
    Nmessages = 0;
    for(int r=0;r<size;++r){
      if(r!=rank){
	size_t count = mesh->NhaloPairs[r]*Nbytes;
	if(count){
	  MPI_Irecv(((char*)recvBuffer)+offset, count, MPI_CHAR, r, tag, mesh->comm, (recvRequests)+Nmessages);	
	  MPI_Isend(((char*)sendBuffer)+offset, count, MPI_CHAR, r, tag, mesh->comm, (sendRequests)+Nmessages);
	  offset += count;
	  ++Nmessages;
	}
      }
    }
    
    // Wait for all sent messages to have left and received messages to have arrived
    MPI_Status *sendStatus = (MPI_Status*) calloc(Nmessages, sizeof(MPI_Status));
    MPI_Status *recvStatus = (MPI_Status*) calloc(Nmessages, sizeof(MPI_Status));
    
    MPI_Waitall(Nmessages, recvRequests, recvStatus);
    MPI_Waitall(Nmessages, sendRequests, sendStatus);
    
    free(recvStatus);
    free(sendStatus);

    free(recvRequests);
    free(sendRequests);
  }
}      

void meshLoadReferenceNodesTet3D(mesh3D *mesh, int N, int cubN){

  dfloat deps = 1.;
  while((1.+deps)>1.)
    deps *= 0.5;

  dfloat NODETOL = 1000.*deps;

  int Np = ((N+1)*(N+2)*(N+3))/6;
  int Nfp = ((N+1)*(N+2))/2;
  
  mesh->N = N;
  mesh->Np = Np;
  mesh->Nfp = Nfp;
  mesh->Nq = N+1;
  
  int Nrows, Ncols;

  // node data
  meshWarpBlendNodesTet3D(N, &(mesh->r), &(mesh->s), &(mesh->t));

  // TW: To do
  // (N+1) GL x (N+1) GJ_1,0 x (N+1) GJ_2,0
  //  mesh->cubNp = meshCollapsedCubatureTet3D(N, &(mesh->cubr), &(mesh->cubs), &(mesh->cubt), &(mesh->cubw));
  dfloat *cubaz, *cubbz, *cubcz;
  dfloat *cubaw, *cubbw, *cubcw;

  // just use same basis for all directions so we can reuse D matrix :-(
  int cubNa = meshJacobiGQ(0.0, 0.0, cubN, &cubaz, &cubaw);
  int cubNb = meshJacobiGQ(0.0, 0.0, cubN, &cubbz, &cubbw);
  int cubNc = meshJacobiGQ(0.0, 0.0, cubN, &cubcz, &cubcw);

  meshJacobiGL(0, 0, N, &(mesh->gllz), &(mesh->gllw));
  meshInterpolationMatrix1D(N, mesh->Nq, mesh->gllz, cubNa, cubaz, &(mesh->cubInterp));
  
  printf("cubNq = %d, cubNa = %d, cubNb = %d, cubNc = %d\n",
	 cubN+1, cubNa, cubNb, cubNc);
  mesh->cubNq = cubNa;
  mesh->cubNp = cubNa*cubNb*cubNc;

  mesh->cubr = (dfloat*) calloc(mesh->cubNp, sizeof(dfloat));
  mesh->cubs = (dfloat*) calloc(mesh->cubNp, sizeof(dfloat));
  mesh->cubt = (dfloat*) calloc(mesh->cubNp, sizeof(dfloat));
  mesh->cubw = (dfloat*) calloc(mesh->cubNp, sizeof(dfloat));

  int cnt = 0;
  for(int k=0;k<cubNc;++k){
    for(int j=0;j<cubNb;++j){
      for(int i=0;i<cubNa;++i){
	dfloat a = cubaz[i];
	dfloat b = cubbz[j];
	dfloat c = cubcz[k];
	mesh->cubr[cnt] = 0.25*(1+a)*(1-b)*(1-c)-1;
	mesh->cubs[cnt] = 0.50*(1+b)*(1-c)-1;
	mesh->cubt[cnt] = c;
	mesh->cubw[cnt] = cubaw[i]*cubbw[j]*cubcw[k]*0.5*(1-b)*pow(0.5*(1-c),2);
	++cnt;
      }
    }
  }
  
  // collocation differentiation matrices
  meshDmatricesTet3D(N, Np, mesh->r, mesh->s, mesh->t, &(mesh->Dr), &(mesh->Ds), &(mesh->Dt));
  meshDmatrix1D(cubN, mesh->cubNq, cubaz, &(mesh->cubD));

  dfloat *V, *Vr, *Vs, *Vt;
  dfloat *cubV, *cubVr, *cubVs, *cubVt;

  // Vandermonde matrices
  meshVandermondeTet3D(N, mesh->cubNp, mesh->cubr, mesh->cubs, mesh->cubt, &(cubV), &(cubVr), &(cubVs), &(cubVt));
  meshVandermondeTet3D(N,    mesh->Np, mesh->r,    mesh->s,    mesh->t,    &(V),    &(Vr),    &(Vs),    &(Vt));

  // interopolation matrix to cubature
  mesh->cubInterp3D = (dfloat*) calloc(mesh->cubNp*mesh->Np, sizeof(dfloat));
  matrixRightSolve(mesh->cubNp, mesh->Np, cubV, mesh->Np, mesh->Np, V, mesh->cubInterp3D);

#if 0
  printf("cubInterp3D:\n");
  for(int n=0;n<mesh->cubNp;++n){
    for(int m=0;m<mesh->Np;++m){
      printf("% e ", mesh->cubInterp3D[n*mesh->Np+m]);
    }
    printf("\n");
  }
#endif

  // mass matrix
  meshMassMatrix(Np, V, &(mesh->MM));
  
  // lift matrix
  //  meshLiftMatrixTet3D(N, Np, mesh->faceNodes, mesh->r, mesh->s, mesh->t, &(mesh->LIFT));

  // interpolation derivative to cubature
  //  meshDmatricesTet3D(N, mesh->cubNp, mesh->cubr, mesh->cubs, mesh->cubt, &(mesh->cubDr), &(mesh->cubDs), &(mesh->cubDt);

  mesh->faceNodes = (int*) calloc(mesh->Nfp*mesh->Nfaces, sizeof(int));

  cnt = 0;
  for(int n=0;n<mesh->Np;++n)
    if(fabs(mesh->t[n]+1)<NODETOL)
      mesh->faceNodes[0*mesh->Nfp+(cnt++)] = n;
  cnt = 0;
  for(int n=0;n<mesh->Np;++n)
    if(fabs(mesh->s[n]+1)<NODETOL)
      mesh->faceNodes[1*mesh->Nfp+(cnt++)] = n;
  cnt = 0;
  for(int n=0;n<mesh->Np;++n)
    if(fabs(mesh->r[n]+mesh->s[n]+mesh->t[n]+1.)<NODETOL)
      mesh->faceNodes[2*mesh->Nfp+(cnt++)] = n;
  cnt = 0;
  for(int n=0;n<mesh->Np;++n)
  if(fabs(mesh->r[n]+1)<NODETOL)
      mesh->faceNodes[3*mesh->Nfp+(cnt++)] = n;

  for(int f=0;f<mesh->Nfaces;++f){
    printf("%d: ", f);
    for(int n=0;n<mesh->Nfp;++n){
      printf("%d ", mesh->faceNodes[f*mesh->Nfp+n]);
    }
    printf("\n");
  }
  
  // find node indices of vertex nodes
  mesh->vertexNodes = (int*) calloc(mesh->Nverts, sizeof(int));
  for(int n=0;n<mesh->Np;++n){
    if( fabs(mesh->r[n]+mesh->s[n]+mesh->t[n]+3)<NODETOL)
      mesh->vertexNodes[0] = n;
    if( fabs(mesh->r[n]-1)<NODETOL)
      mesh->vertexNodes[1] = n;
    if( fabs(mesh->s[n]-1)<NODETOL)
      mesh->vertexNodes[2] = n;
    if( fabs(mesh->t[n]-1)<NODETOL)
      mesh->vertexNodes[3] = n;
  }
}




void meshLoadReferenceNodesHex3D(mesh3D *mesh, int N, int cubN){

  dfloat deps = 1.;
  while((1.+deps)>1.)
    deps *= 0.5;

  dfloat NODETOL = 10.*deps;
  
  mesh->N = N;
  mesh->Nq = N+1;
  mesh->Nfp = (N+1)*(N+1);
  mesh->Np = (N+1)*(N+1)*(N+1);
  mesh->Nverts = 8;

  // GLL nodes
  meshJacobiGL(0, 0, N, &(mesh->gllz), &(mesh->gllw));

  // GLL collocation differentiation matrix
  meshDmatrix1D(N, mesh->Nq, mesh->gllz, &(mesh->D));

  // GLL top C0 mode filter matrix
  meshContinuousFilterMatrix1D(N, N-1, mesh->gllz, &(mesh->filterMatrix));
  
  // quadrature
  mesh->cubNq = cubN +1;
  mesh->cubNfp = mesh->cubNq*mesh->cubNq;
  mesh->cubNp = mesh->cubNq*mesh->cubNq*mesh->cubNq;
  
  // GL quadrature
  meshJacobiGQ(0, 0, cubN, &(mesh->cubr), &(mesh->cubw));

  // GLL to GL interpolation matrix
  meshInterpolationMatrix1D(N, mesh->Nq, mesh->gllz, mesh->cubNq, mesh->cubr, &(mesh->cubInterp));

  // GL to GL differentiation matrix
  meshDmatrix1D(cubN, mesh->cubNq, mesh->cubr, &(mesh->cubD));
  
  // populate r
  mesh->r = (dfloat*) calloc(mesh->Np, sizeof(dfloat));
  mesh->s = (dfloat*) calloc(mesh->Np, sizeof(dfloat));
  mesh->t = (dfloat*) calloc(mesh->Np, sizeof(dfloat));

  int cnt = 0;
  for(int k=0;k<mesh->Nq;++k){
    for(int j=0;j<mesh->Nq;++j){
      for(int i=0;i<mesh->Nq;++i){
	mesh->r[cnt] = mesh->gllz[i];
	mesh->s[cnt] = mesh->gllz[j];
	mesh->t[cnt] = mesh->gllz[k];
	++cnt;
      }
    }
  }

  mesh->faceNodes = (int*) calloc(mesh->Nfp*mesh->Nfaces, sizeof(int));

  cnt = 0;
  for(int n=0;n<mesh->Np;++n)
    if(fabs(mesh->t[n]+1)<NODETOL)
      mesh->faceNodes[0*mesh->Nfp+(cnt++)] = n;
  cnt = 0;
  for(int n=0;n<mesh->Np;++n)
    if(fabs(mesh->s[n]+1)<NODETOL)
      mesh->faceNodes[1*mesh->Nfp+(cnt++)] = n;
  cnt = 0;
  for(int n=0;n<mesh->Np;++n)
    if(fabs(mesh->r[n]-1)<NODETOL)
      mesh->faceNodes[2*mesh->Nfp+(cnt++)] = n;
  cnt = 0;
  for(int n=0;n<mesh->Np;++n)
    if(fabs(mesh->s[n]-1)<NODETOL)
      mesh->faceNodes[3*mesh->Nfp+(cnt++)] = n;
  cnt = 0;
  for(int n=0;n<mesh->Np;++n)
    if(fabs(mesh->r[n]+1)<NODETOL)
      mesh->faceNodes[4*mesh->Nfp+(cnt++)] = n;
  cnt = 0;
  for(int n=0;n<mesh->Np;++n)
    if(fabs(mesh->t[n]-1)<NODETOL)
      mesh->faceNodes[5*mesh->Nfp+(cnt++)] = n;

  // find node indices of vertex nodes
  mesh->vertexNodes = (int*) calloc(mesh->Nverts, sizeof(int));
  for(int n=0;n<mesh->Np;++n){
    if( (mesh->r[n]+1)*(mesh->r[n]+1)+(mesh->s[n]+1)*(mesh->s[n]+1)+(mesh->t[n]+1)*(mesh->t[n]+1)<NODETOL)
      mesh->vertexNodes[0] = n;
    if( (mesh->r[n]-1)*(mesh->r[n]-1)+(mesh->s[n]+1)*(mesh->s[n]+1)+(mesh->t[n]+1)*(mesh->t[n]+1)<NODETOL)
      mesh->vertexNodes[1] = n;
    if( (mesh->r[n]-1)*(mesh->r[n]-1)+(mesh->s[n]-1)*(mesh->s[n]-1)+(mesh->t[n]+1)*(mesh->t[n]+1)<NODETOL)
      mesh->vertexNodes[2] = n;
    if( (mesh->r[n]+1)*(mesh->r[n]+1)+(mesh->s[n]-1)*(mesh->s[n]-1)+(mesh->t[n]+1)*(mesh->t[n]+1)<NODETOL)
      mesh->vertexNodes[3] = n;
    if( (mesh->r[n]+1)*(mesh->r[n]+1)+(mesh->s[n]+1)*(mesh->s[n]+1)+(mesh->t[n]-1)*(mesh->t[n]-1)<NODETOL)
      mesh->vertexNodes[4] = n;
    if( (mesh->r[n]-1)*(mesh->r[n]-1)+(mesh->s[n]+1)*(mesh->s[n]+1)+(mesh->t[n]-1)*(mesh->t[n]-1)<NODETOL)
      mesh->vertexNodes[5] = n;
    if( (mesh->r[n]-1)*(mesh->r[n]-1)+(mesh->s[n]-1)*(mesh->s[n]-1)+(mesh->t[n]-1)*(mesh->t[n]-1)<NODETOL)
      mesh->vertexNodes[6] = n;
    if( (mesh->r[n]+1)*(mesh->r[n]+1)+(mesh->s[n]-1)*(mesh->s[n]-1)+(mesh->t[n]-1)*(mesh->t[n]-1)<NODETOL)
      mesh->vertexNodes[7] = n;
  }
}


void reportMemoryUsage(occa::device &device, const char *mess){

  size_t bytes = device.memoryAllocated();

  printf("%s: bytes allocated = %lu\n", mess, bytes);
}

void meshOccaPopulateDevice3D(mesh3D *mesh, setupAide &newOptions, occa::properties &kernelInfo){

  // find elements that have all neighbors on this process
  dlong *internalElementIds = (dlong*) calloc(mesh->Nelements, sizeof(dlong));
  dlong *notInternalElementIds = (dlong*) calloc(mesh->Nelements, sizeof(dlong));

  dlong Ninterior = 0, NnotInterior = 0;
  for(dlong e=0;e<mesh->Nelements;++e){
    int flag = 0;
    for(int f=0;f<mesh->Nfaces;++f)
      if(mesh->EToP[e*mesh->Nfaces+f]!=-1)
        flag = 1;
    if(!flag)
      internalElementIds[Ninterior++] = e;
    else
      notInternalElementIds[NnotInterior++] = e;
  }

  mesh->NinternalElements = Ninterior;
  mesh->NnotInternalElements = NnotInterior;
  if(Ninterior)
    mesh->o_internalElementIds    = mesh->device.malloc(Ninterior*sizeof(dlong), internalElementIds);

  if(NnotInterior>0)
    mesh->o_notInternalElementIds = mesh->device.malloc(NnotInterior*sizeof(dlong), notInternalElementIds);

  if(mesh->elementType==HEXAHEDRA){
  
    //lumped mass matrix
    mesh->MM = (dfloat *) calloc(mesh->Np*mesh->Np, sizeof(dfloat));
    for (int k=0;k<mesh->Nq;k++) {
      for (int j=0;j<mesh->Nq;j++) {
	for (int i=0;i<mesh->Nq;i++) {
	  int n = i+j*mesh->Nq+k*mesh->Nq*mesh->Nq;
	  mesh->MM[n+n*mesh->Np] = mesh->gllw[i]*mesh->gllw[j]*mesh->gllw[k];
	}
      }
    }
    
    mesh->o_D = mesh->device.malloc(mesh->Nq*mesh->Nq*sizeof(dfloat), mesh->D);
    
    mesh->o_filterMatrix = mesh->device.malloc(mesh->Nq*mesh->Nq*sizeof(dfloat), mesh->filterMatrix);

    mesh->o_vgeo =
      mesh->device.malloc(mesh->Nelements*mesh->Np*mesh->Nvgeo*sizeof(dfloat),
			  mesh->vgeo);
    
    mesh->o_sgeo =
      mesh->device.malloc(mesh->Nelements*mesh->Nfaces*mesh->Nfp*mesh->Nsgeo*sizeof(dfloat),
			  mesh->sgeo);
    
    mesh->o_ggeo =
      mesh->device.malloc(mesh->Nelements*mesh->Np*mesh->Nggeo*sizeof(dfloat),
			  mesh->ggeo);

    mesh->o_cubvgeo =
      mesh->device.malloc(mesh->Nelements*mesh->Nvgeo*mesh->cubNp*sizeof(dfloat),
			  mesh->cubvgeo);
    
    mesh->o_cubsgeo =
      mesh->device.malloc(mesh->Nelements*mesh->Nfaces*mesh->cubNfp*mesh->Nsgeo*sizeof(dfloat),
			  mesh->cubsgeo);
    

  }
  
  if(mesh->elementType==TETRAHEDRA){

    mesh->o_vgeo =
      mesh->device.malloc(mesh->Nelements*mesh->Nvgeo*sizeof(dfloat),  mesh->vgeo);
    
    mesh->o_sgeo =
      mesh->device.malloc(mesh->Nelements*mesh->Nfaces*mesh->Nsgeo*sizeof(dfloat), mesh->sgeo);
    
    mesh->o_ggeo =
      mesh->device.malloc(mesh->Nelements*mesh->Nggeo*sizeof(dfloat), mesh->ggeo);
  }

  
  mesh->o_cubggeo =
    mesh->device.malloc(mesh->Nelements*mesh->Nggeo*mesh->cubNp*sizeof(dfloat),
			mesh->cubggeo);

  mesh->o_cubD =
    mesh->device.malloc(mesh->cubNq*mesh->cubNq*sizeof(dfloat),
			mesh->cubD);
  
  mesh->o_cubInterp =
    mesh->device.malloc(mesh->cubNq*mesh->Nq*sizeof(dfloat),
			mesh->cubInterp);


  if(mesh->elementType==TETRAHEDRA){
    dfloat *cubComboInterp3D = (dfloat*) calloc(2*mesh->cubNp*mesh->Np, sizeof(dfloat));
    int cnt = 0;
    for(int n=0;n<mesh->cubNp;++n){
      for(int m=0;m<mesh->Np;++m){
	cubComboInterp3D[n*mesh->Np+m+0*mesh->cubNp*mesh->Np] = mesh->cubInterp3D[n*mesh->Np+m];
	cubComboInterp3D[m*mesh->cubNp+n+1*mesh->cubNp*mesh->Np] = mesh->cubInterp3D[n*mesh->Np+m];
      }
    }
    mesh->o_cubInterp3D =
      mesh->device.malloc(2*mesh->cubNp*mesh->Np*sizeof(dfloat),
			  cubComboInterp3D);
  }
  
  mesh->o_vmapM =
    mesh->device.malloc(mesh->Nelements*mesh->Nfp*mesh->Nfaces*sizeof(dlong),
                        mesh->vmapM);

  mesh->o_vmapP =
    mesh->device.malloc(mesh->Nelements*mesh->Nfp*mesh->Nfaces*sizeof(dlong),
                        mesh->vmapP);

  mesh->o_EToB =
    mesh->device.malloc(mesh->Nelements*mesh->Nfaces*sizeof(int),
                        mesh->EToB);

  mesh->o_x =
    mesh->device.malloc(mesh->Nelements*mesh->Np*sizeof(dfloat), mesh->x);

  mesh->o_y =
    mesh->device.malloc(mesh->Nelements*mesh->Np*sizeof(dfloat), mesh->y);

  mesh->o_z =
    mesh->device.malloc(mesh->Nelements*mesh->Np*sizeof(dfloat), mesh->z);


  if(mesh->totalHaloPairs>0){
    // copy halo element list to DEVICE
    mesh->o_haloElementList =
      mesh->device.malloc(mesh->totalHaloPairs*sizeof(dlong), mesh->haloElementList);

    // temporary DEVICE buffer for halo (maximum size Nfields*Np for dfloat)
    mesh->o_haloBuffer =
      mesh->device.malloc(mesh->totalHaloPairs*mesh->Np*sizeof(dfloat));

    // node ids 
    mesh->o_haloGetNodeIds = 
      mesh->device.malloc(mesh->Nfp*mesh->totalHaloPairs*sizeof(dlong), mesh->haloGetNodeIds);
    mesh->o_haloPutNodeIds = 
      mesh->device.malloc(mesh->Nfp*mesh->totalHaloPairs*sizeof(dlong), mesh->haloPutNodeIds);

  }

  mesh->o_localizedIds =
    mesh->device.malloc(mesh->Nelements*mesh->Np*sizeof(dlong),
                        mesh->localizedIds);

  
  kernelInfo["defines/" "p_dim"]= 3;
  kernelInfo["defines/" "p_N"]= mesh->N;
  kernelInfo["defines/" "p_Nq"]= mesh->N+1;
  kernelInfo["defines/" "p_Np"]= mesh->Np;
  kernelInfo["defines/" "p_Nfp"]= mesh->Nfp;
  kernelInfo["defines/" "p_Nfaces"]= mesh->Nfaces;
  kernelInfo["defines/" "p_NfacesNfp"]= mesh->Nfp*mesh->Nfaces;
  kernelInfo["defines/" "p_Nvgeo"]= mesh->Nvgeo;
  kernelInfo["defines/" "p_Nsgeo"]= mesh->Nsgeo;
  kernelInfo["defines/" "p_Nggeo"]= mesh->Nggeo;

  kernelInfo["defines/" "p_NXID"]= NXID;
  kernelInfo["defines/" "p_NYID"]= NYID;
  kernelInfo["defines/" "p_NZID"]= NZID;
  kernelInfo["defines/" "p_SJID"]= SJID;
  kernelInfo["defines/" "p_IJID"]= IJID;
  kernelInfo["defines/" "p_IHID"]= IHID;
  kernelInfo["defines/" "p_WSJID"]= WSJID;
  kernelInfo["defines/" "p_WIJID"]= WIJID;

  int maxNodes = mymax(mesh->Np, (mesh->Nfp*mesh->Nfaces));
  kernelInfo["defines/" "p_maxNodes"]= maxNodes;

  kernelInfo["defines/" "p_Lambda2"]= 0.5f;

  kernelInfo["defines/" "p_cubNq"]= mesh->cubNq;
  kernelInfo["defines/" "p_cubNfp"]= mesh->cubNfp;
  kernelInfo["defines/" "p_cubNp"]= mesh->cubNp;

  if(sizeof(dfloat)==4){
    kernelInfo["defines/" "dfloat"]="float";
    kernelInfo["defines/" "dfloat4"]="float4";
    kernelInfo["defines/" "dfloat8"]="float8";
  }
  if(sizeof(dfloat)==8){
    kernelInfo["defines/" "dfloat"]="double";
    kernelInfo["defines/" "dfloat4"]="double4";
    kernelInfo["defines/" "dfloat8"]="double8";
  }

  if(sizeof(dlong)==4){
    kernelInfo["defines/" "dlong"]="int";
  }
  if(sizeof(dlong)==8){
    kernelInfo["defines/" "dlong"]="long long int";
  }

  if(mesh->device.mode()=="CUDA"){ // add backend compiler optimization for CUDA
    kernelInfo["compiler_flags"] += " --ftz=true ";
    kernelInfo["compiler_flags"] += " --prec-div=false ";
    kernelInfo["compiler_flags"] += " --prec-sqrt=false ";
    kernelInfo["compiler_flags"] += " --use_fast_math ";
    kernelInfo["compiler_flags"] += " --fmad=true "; // compiler option for cuda
  }

  kernelInfo["defines/" "p_G00ID"]= G00ID;
  kernelInfo["defines/" "p_G01ID"]= G01ID;
  kernelInfo["defines/" "p_G02ID"]= G02ID;
  kernelInfo["defines/" "p_G11ID"]= G11ID;
  kernelInfo["defines/" "p_G12ID"]= G12ID;
  kernelInfo["defines/" "p_G22ID"]= G22ID;
  kernelInfo["defines/" "p_GWJID"]= GWJID;


  kernelInfo["defines/" "p_RXID"]= RXID;
  kernelInfo["defines/" "p_SXID"]= SXID;
  kernelInfo["defines/" "p_TXID"]= TXID;

  kernelInfo["defines/" "p_RYID"]= RYID;
  kernelInfo["defines/" "p_SYID"]= SYID;
  kernelInfo["defines/" "p_TYID"]= TYID;

  kernelInfo["defines/" "p_RZID"]= RZID;
  kernelInfo["defines/" "p_SZID"]= SZID;
  kernelInfo["defines/" "p_TZID"]= TZID;

  kernelInfo["defines/" "p_JID"]= JID;
  kernelInfo["defines/" "p_JWID"]= JWID;
  kernelInfo["defines/" "p_IJWID"]= IJWID;
}


void meshOccaSetup3D(mesh3D *mesh, setupAide &newOptions, occa::properties &kernelInfo){

  // conigure device
  occaDeviceConfig(mesh, newOptions);
  
  //make seperate stream for halo exchange
  mesh->defaultStream = mesh->device.getStream();
  mesh->dataStream = mesh->device.createStream();
  mesh->computeStream = mesh->device.createStream();
  mesh->device.setStream(mesh->defaultStream);

  meshOccaPopulateDevice3D(mesh, newOptions, kernelInfo);
  
}

void meshOccaCloneDevice(mesh_t *donorMesh, mesh_t *mesh){

  mesh->device = donorMesh->device;

  mesh->defaultStream = donorMesh->defaultStream;
  mesh->dataStream = donorMesh->dataStream;
  mesh->computeStream = donorMesh->computeStream;
  
}

typedef struct{

  int baseRank;
  hlong baseId;
  hlong localizedId;
  
}parallelNode_t;


// uniquely label each node with a global index, used for gatherScatter
void meshParallelConnectNodes(mesh_t *mesh){

  int rank, size;
  rank = mesh->rank; 
  size = mesh->size; 

  dlong localNodeCount = mesh->Np*mesh->Nelements;
  dlong *allLocalNodeCounts = (dlong*) calloc(size, sizeof(dlong));

  MPI_Allgather(&localNodeCount,    1, MPI_DLONG,
                allLocalNodeCounts, 1, MPI_DLONG,
                mesh->comm);
  
  hlong gatherNodeStart = 0;
  for(int r=0;r<rank;++r)
    gatherNodeStart += allLocalNodeCounts[r];
  
  free(allLocalNodeCounts);

  // form continuous node numbering (local=>virtual gather)
  parallelNode_t *localNodes =
    (parallelNode_t*) calloc((mesh->totalHaloPairs+mesh->Nelements)*mesh->Np,
                             sizeof(parallelNode_t));

  // use local numbering
  for(dlong e=0;e<mesh->Nelements;++e){
    for(int n=0;n<mesh->Np;++n){
      dlong id = e*mesh->Np+n;

      localNodes[id].baseRank = rank;
      localNodes[id].baseId = 1 + id + mesh->Nnodes + gatherNodeStart;

    }

    // use vertex ids for vertex nodes to reduce iterations
    for(int v=0;v<mesh->Nverts;++v){
      dlong id = e*mesh->Np + mesh->vertexNodes[v];
      hlong gid = mesh->EToV[e*mesh->Nverts+v] + 1;
      localNodes[id].baseId = gid; 
    }
  }

  dlong localChange = 0, gatherChange = 1;

  parallelNode_t *sendBuffer =
    (parallelNode_t*) calloc(mesh->totalHaloPairs*mesh->Np, sizeof(parallelNode_t));

  // keep comparing numbers on positive and negative traces until convergence
  while(gatherChange>0){

    // reset change counter
    localChange = 0;

    // send halo data and recv into extension of buffer
    meshHaloExchange(mesh, mesh->Np*sizeof(parallelNode_t),
                     localNodes, sendBuffer, localNodes+localNodeCount);

    // compare trace nodes
    for(dlong e=0;e<mesh->Nelements;++e){
      for(int n=0;n<mesh->Nfp*mesh->Nfaces;++n){
        dlong id  = e*mesh->Nfp*mesh->Nfaces + n;
        dlong idM = mesh->vmapM[id];
        dlong idP = mesh->vmapP[id];
        hlong gidM = localNodes[idM].baseId;
        hlong gidP = localNodes[idP].baseId;

        int baseRankM = localNodes[idM].baseRank;
        int baseRankP = localNodes[idP].baseRank;
        
        if(gidM<gidP || (gidP==gidM && baseRankM<baseRankP)){
          ++localChange;
          localNodes[idP].baseRank    = localNodes[idM].baseRank;
          localNodes[idP].baseId      = localNodes[idM].baseId;
        }
        
        if(gidP<gidM || (gidP==gidM && baseRankP<baseRankM)){
          ++localChange;
          localNodes[idM].baseRank    = localNodes[idP].baseRank;
          localNodes[idM].baseId      = localNodes[idP].baseId;
        }
      }
    }

    // sum up changes
    MPI_Allreduce(&localChange, &gatherChange, 1, MPI_DLONG, MPI_SUM, mesh->comm);
  }

  //make a locally-ordered version
  mesh->globalIds = (hlong*) calloc(localNodeCount, sizeof(hlong));
  for(dlong id=0;id<localNodeCount;++id){
    mesh->globalIds[id] = localNodes[id].baseId;    
  }
  
  free(localNodes);
  free(sendBuffer);
}
// structure used to encode vertices that make 
// each face, the element/face indices, and
// the neighbor element/face indices (if any)
typedef struct{

  dlong element;
  int face;

  dlong elementNeighbor; // neighbor element
  int faceNeighbor;    // neighbor face

  int NfaceVertices;
  
  hlong v[4];

}face_t;

// comparison function that orders vertices 
// based on their combined vertex indices
int compareVertices(const void *a, 
                    const void *b){

  face_t *fa = (face_t*) a;
  face_t *fb = (face_t*) b;
  
  for(int n=0;n<fa->NfaceVertices;++n){
    if(fa->v[n] < fb->v[n]) return -1;
    if(fa->v[n] > fb->v[n]) return +1;
  }
  
  return 0;

}
/* comparison function that orders element/face
   based on their indexes */
int compareFaces(const void *a, 
                 const void *b){

  face_t *fa = (face_t*) a;
  face_t *fb = (face_t*) b;

  if(fa->element < fb->element) return -1;
  if(fa->element > fb->element) return +1;

  if(fa->face < fb->face) return -1;
  if(fa->face > fb->face) return +1;

  return 0;

}

/* routine to find EToE (Element To Element)
   and EToF (Element To Local Face) connectivity arrays */
void meshConnect(mesh_t *mesh){

  /* build list of faces */
  face_t *faces = 
    (face_t*) calloc(mesh->Nelements*mesh->Nfaces, sizeof(face_t));
  
  dlong cnt = 0;
  for(dlong e=0;e<mesh->Nelements;++e){
    for(int f=0;f<mesh->Nfaces;++f){
      
      for(int n=0;n<mesh->NfaceVertices;++n){
        dlong vid = e*mesh->Nverts + mesh->faceVertices[f*mesh->NfaceVertices+n];
        faces[cnt].v[n] = mesh->EToV[vid];
      }
      
      mysort(faces[cnt].v, mesh->NfaceVertices, "descending");

      faces[cnt].NfaceVertices = mesh->NfaceVertices;
      
      faces[cnt].element = e;
      faces[cnt].face = f;
      
      faces[cnt].elementNeighbor= -1;
      faces[cnt].faceNeighbor = -1;

      ++cnt;

    }
  }
  
  /* sort faces by their vertex number pairs */
  qsort(faces, 
        mesh->Nelements*mesh->Nfaces,
        sizeof(face_t),
        compareVertices);
  
  /* scan through sorted face lists looking for adjacent
     faces that have the same vertex ids */
  for(cnt=0;cnt<mesh->Nelements*mesh->Nfaces-1;++cnt){
    
    if(!compareVertices(faces+cnt, faces+cnt+1)){
      // match
      faces[cnt].elementNeighbor = faces[cnt+1].element;
      faces[cnt].faceNeighbor = faces[cnt+1].face;

      faces[cnt+1].elementNeighbor = faces[cnt].element;
      faces[cnt+1].faceNeighbor = faces[cnt].face;
    }
    
  }

  /* resort faces back to the original element/face ordering */
  qsort(faces, 
        mesh->Nelements*mesh->Nfaces,
        sizeof(face_t),
        compareFaces);

  /* extract the element to element and element to face connectivity */
  mesh->EToE = (dlong*) calloc(mesh->Nelements*mesh->Nfaces, sizeof(dlong));
  mesh->EToF = (int*)   calloc(mesh->Nelements*mesh->Nfaces, sizeof(int  ));

  cnt = 0;
  for(dlong e=0;e<mesh->Nelements;++e){
    for(int f=0;f<mesh->Nfaces;++f){
      mesh->EToE[cnt] = faces[cnt].elementNeighbor;
      mesh->EToF[cnt] = faces[cnt].faceNeighbor;

      //      printf("EToE(%d,%d) = %d \n", e,f, mesh->EToE[cnt]);
      
      ++cnt;
    }
  }

  // dlong Nbcs = 0;
  // for(dlong e=0;e<mesh->Nelements;++e)
  //   for(int f=0;f<mesh->Nfaces;++f)
  //     if(mesh->EToE[e*mesh->Nfaces+f]==-1)
  //       ++Nbcs;
  //printf("Nelements = %d, Nbcs = %d\n", mesh->Nelements, Nbcs);
}

typedef struct {
  hlong v[4]; // vertices on face
  dlong element, elementN; 
  int NfaceVertices;
  int face, rank;    // face info
  int faceN, rankN; // N for neighbor face info

}parallelFace_t;

// comparison function that orders vertices 
// based on their combined vertex indices
int parallelCompareVertices(const void *a, 
                            const void *b){

  parallelFace_t *fa = (parallelFace_t*) a;
  parallelFace_t *fb = (parallelFace_t*) b;

  for(int n=0;n<fa->NfaceVertices;++n){
    if(fa->v[n] < fb->v[n]) return -1;
    if(fa->v[n] > fb->v[n]) return +1;
  }

  return 0;
}

/* comparison function that orders element/face
   based on their indexes */
int parallelCompareFaces(const void *a, 
                         const void *b){

  parallelFace_t *fa = (parallelFace_t*) a;
  parallelFace_t *fb = (parallelFace_t*) b;

  if(fa->rank < fb->rank) return -1;
  if(fa->rank > fb->rank) return +1;

  if(fa->element < fb->element) return -1;
  if(fa->element > fb->element) return +1;

  if(fa->face < fb->face) return -1;
  if(fa->face > fb->face) return +1;

  return 0;
}
  
// mesh is the local partition
void meshParallelConnect(mesh_t *mesh){

  int rank, size;
  rank = mesh->rank;
  size = mesh->size;

  // serial connectivity on each process
  meshConnect(mesh);

  // count # of elements to send to each rank based on
  // minimum {vertex id % size}
  int *Nsend = (int*) calloc(size, sizeof(int));
  int *Nrecv = (int*) calloc(size, sizeof(int));
  int *sendOffsets = (int*) calloc(size, sizeof(int));
  int *recvOffsets = (int*) calloc(size, sizeof(int));
  
  // WARNING: In some corner cases, the number of faces to send may overrun int storage
  int allNsend = 0;
  for(dlong e=0;e<mesh->Nelements;++e){
    for(int f=0;f<mesh->Nfaces;++f){
      if(mesh->EToE[e*mesh->Nfaces+f]==-1){
        // find rank of destination for sorting based on max(face vertices)%size
        hlong maxv = 0;
        for(int n=0;n<mesh->NfaceVertices;++n){
          int nid = mesh->faceVertices[f*mesh->NfaceVertices+n];
          dlong id = mesh->EToV[e*mesh->Nverts + nid];
          maxv = mymax(maxv, id);
        }
        int destRank = (int) (maxv%size);

        // increment send size for 
        ++Nsend[destRank];
        ++allNsend;
      }
    }
  }
  
  // find send offsets
  for(int r=1;r<size;++r)
    sendOffsets[r] = sendOffsets[r-1] + Nsend[r-1];
  
  // reset counters
  for(int r=0;r<size;++r)
    Nsend[r] = 0;

  // buffer for outgoing data
  parallelFace_t *sendFaces = (parallelFace_t*) calloc(allNsend, sizeof(parallelFace_t));

  // Make the MPI_PARALLELFACE_T data type
  MPI_Datatype MPI_PARALLELFACE_T;
  MPI_Datatype dtype[8] = {MPI_HLONG, MPI_DLONG, MPI_DLONG, MPI_INT, 
                            MPI_INT, MPI_INT, MPI_INT, MPI_INT};
  int blength[8] = {4, 1, 1, 1, 1, 1, 1, 1};
  MPI_Aint addr[8], displ[8];
  MPI_Get_address ( &(sendFaces[0]              ), addr+0);
  MPI_Get_address ( &(sendFaces[0].element      ), addr+1);
  MPI_Get_address ( &(sendFaces[0].elementN     ), addr+2);
  MPI_Get_address ( &(sendFaces[0].NfaceVertices), addr+3);
  MPI_Get_address ( &(sendFaces[0].face         ), addr+4);
  MPI_Get_address ( &(sendFaces[0].rank         ), addr+5);
  MPI_Get_address ( &(sendFaces[0].faceN        ), addr+6);
  MPI_Get_address ( &(sendFaces[0].rankN        ), addr+7);
  displ[0] = 0;
  displ[1] = addr[1] - addr[0];
  displ[2] = addr[2] - addr[0];
  displ[3] = addr[3] - addr[0];
  displ[4] = addr[4] - addr[0];
  displ[5] = addr[5] - addr[0];
  displ[6] = addr[6] - addr[0];
  displ[7] = addr[7] - addr[0];
  MPI_Type_create_struct (8, blength, displ, dtype, &MPI_PARALLELFACE_T);
  MPI_Type_commit (&MPI_PARALLELFACE_T);

  // pack face data
  for(dlong e=0;e<mesh->Nelements;++e){
    for(int f=0;f<mesh->Nfaces;++f){
      if(mesh->EToE[e*mesh->Nfaces+f]==-1){

        // find rank of destination for sorting based on max(face vertices)%size
        hlong maxv = 0;
        for(int n=0;n<mesh->NfaceVertices;++n){
          int nid = mesh->faceVertices[f*mesh->NfaceVertices+n];
          hlong id = mesh->EToV[e*mesh->Nverts + nid];
          maxv = mymax(maxv, id);
        }
        int destRank = (int) (maxv%size);
        
        // populate face to send out staged in segment of sendFaces array
        int id = sendOffsets[destRank]+Nsend[destRank];

        
        sendFaces[id].element = e;
        sendFaces[id].face = f;
        for(int n=0;n<mesh->NfaceVertices;++n){
          int nid = mesh->faceVertices[f*mesh->NfaceVertices+n];
          sendFaces[id].v[n] = mesh->EToV[e*mesh->Nverts + nid];
        }

        mysort(sendFaces[id].v,mesh->NfaceVertices, "descending");

        sendFaces[id].NfaceVertices = mesh->NfaceVertices;
        sendFaces[id].rank = rank;

        sendFaces[id].elementN = -1;
        sendFaces[id].faceN = -1;
        sendFaces[id].rankN = -1;
        
        ++Nsend[destRank];
      }
    }
  }

  // exchange byte counts 
  MPI_Alltoall(Nsend, 1, MPI_INT,
               Nrecv, 1, MPI_INT,
               mesh->comm);
  
  // count incoming faces
  int allNrecv = 0;
  for(int r=0;r<size;++r)
    allNrecv += Nrecv[r];

  // find offsets for recv data
  for(int r=1;r<size;++r)
    recvOffsets[r] = recvOffsets[r-1] + Nrecv[r-1]; // byte offsets
  
  // buffer for incoming face data
  parallelFace_t *recvFaces = (parallelFace_t*) calloc(allNrecv, sizeof(parallelFace_t));
  
  // exchange parallel faces
  MPI_Alltoallv(sendFaces, Nsend, sendOffsets, MPI_PARALLELFACE_T,
                recvFaces, Nrecv, recvOffsets, MPI_PARALLELFACE_T,
                mesh->comm);
  
  // local sort allNrecv received faces
  qsort(recvFaces, allNrecv, sizeof(parallelFace_t), parallelCompareVertices);

  // find matches
  for(int n=0;n<allNrecv-1;++n){
    // since vertices are ordered we just look for pairs
    if(!parallelCompareVertices(recvFaces+n, recvFaces+n+1)){
      recvFaces[n].elementN = recvFaces[n+1].element;
      recvFaces[n].faceN = recvFaces[n+1].face;
      recvFaces[n].rankN = recvFaces[n+1].rank;
      
      recvFaces[n+1].elementN = recvFaces[n].element;
      recvFaces[n+1].faceN = recvFaces[n].face;
      recvFaces[n+1].rankN = recvFaces[n].rank;
    }
  }

  // sort back to original ordering
  qsort(recvFaces, allNrecv, sizeof(parallelFace_t), parallelCompareFaces);

  // send faces back from whence they came
  MPI_Alltoallv(recvFaces, Nrecv, recvOffsets, MPI_PARALLELFACE_T,
                sendFaces, Nsend, sendOffsets, MPI_PARALLELFACE_T,
                mesh->comm);
  
  // extract connectivity info
  mesh->EToP = (int*) calloc(mesh->Nelements*mesh->Nfaces, sizeof(int));
  for(dlong cnt=0;cnt<mesh->Nelements*mesh->Nfaces;++cnt)
    mesh->EToP[cnt] = -1;
  
  for(int cnt=0;cnt<allNsend;++cnt){
    dlong e = sendFaces[cnt].element;
    dlong eN = sendFaces[cnt].elementN;
    int f = sendFaces[cnt].face;
    int fN = sendFaces[cnt].faceN;
    int rN = sendFaces[cnt].rankN;
    
    if(e>=0 && f>=0 && eN>=0 && fN>=0){
      mesh->EToE[e*mesh->Nfaces+f] = eN;
      mesh->EToF[e*mesh->Nfaces+f] = fN;
      mesh->EToP[e*mesh->Nfaces+f] = rN;
    }
  }
  
  MPI_Barrier(mesh->comm);
  MPI_Type_free(&MPI_PARALLELFACE_T);
  free(sendFaces);
  free(recvFaces);
}

void meshPartitionStatistics(mesh_t *mesh){

  /* get MPI rank and size */
  int rank, size;
  rank = mesh->rank;
  size = mesh->size;
  
  /* now gather statistics on connectivity between processes */
  int *comms = (int*) calloc(size, sizeof(int));
  int Ncomms = 0;

  /* count elements with neighbors on each other rank ranks */
  for(dlong e=0;e<mesh->Nelements;++e){
    for(int f=0;f<mesh->Nfaces;++f){
      if(mesh->EToP[e*mesh->Nfaces+f]!=-1){
        ++comms[mesh->EToP[e*mesh->Nfaces+f]];
        ++Ncomms;
      }
    }
  }

  int Nmessages = 0;
  for(int r=0;r<size;++r)
    if(comms[r]>0)
      ++Nmessages;

  for(int r=0;r<size;++r){
    MPI_Barrier(mesh->comm);
    if(r==rank){
      fflush(stdout);
      printf("r: %02d [", rank);
      for(int s=0;s<size;++s){
        printf(" %04d", comms[s]);
      }
      printf("] (Nelements=" dlongFormat ", Nmessages=%d, Ncomms=%d)\n", mesh->Nelements,Nmessages, Ncomms);
      fflush(stdout);
    }
  }
  
  free(comms);
}


void meshPhysicalNodesTet3D(mesh3D *mesh){
  
  mesh->x = (dfloat*) calloc(mesh->Nelements*mesh->Np,sizeof(dfloat));
  mesh->y = (dfloat*) calloc(mesh->Nelements*mesh->Np,sizeof(dfloat));
  mesh->z = (dfloat*) calloc(mesh->Nelements*mesh->Np,sizeof(dfloat));
  
  dlong cnt = 0;
  for(dlong e=0;e<mesh->Nelements;++e){ /* for each element */

    dlong id = e*mesh->Nverts;

    dfloat xe1 = mesh->EX[id+0]; /* x-coordinates of vertices */
    dfloat xe2 = mesh->EX[id+1];
    dfloat xe3 = mesh->EX[id+2];
    dfloat xe4 = mesh->EX[id+3];

    dfloat ye1 = mesh->EY[id+0]; /* y-coordinates of vertices */
    dfloat ye2 = mesh->EY[id+1];
    dfloat ye3 = mesh->EY[id+2];
    dfloat ye4 = mesh->EY[id+3];
    
    dfloat ze1 = mesh->EZ[id+0]; /* z-coordinates of vertices */
    dfloat ze2 = mesh->EZ[id+1];
    dfloat ze3 = mesh->EZ[id+2];
    dfloat ze4 = mesh->EZ[id+3];
    
    for(int n=0;n<mesh->Np;++n){ /* for each node */
      
      /* (r,s,t) coordinates of interpolation nodes*/
      dfloat rn = mesh->r[n]; 
      dfloat sn = mesh->s[n];
      dfloat tn = mesh->t[n];

      /* physical coordinate of interpolation node */
      mesh->x[cnt] = -0.5*(1+rn+sn+tn)*xe1 + 0.5*(1+rn)*xe2 + 0.5*(1+sn)*xe3 + 0.5*(1+tn)*xe4;
      mesh->y[cnt] = -0.5*(1+rn+sn+tn)*ye1 + 0.5*(1+rn)*ye2 + 0.5*(1+sn)*ye3 + 0.5*(1+tn)*ye4;
      mesh->z[cnt] = -0.5*(1+rn+sn+tn)*ze1 + 0.5*(1+rn)*ze2 + 0.5*(1+sn)*ze3 + 0.5*(1+tn)*ze4;
      ++cnt;
    }
  }
}
 
void meshPhysicalNodesHex3D(mesh3D *mesh){
  
  mesh->x = (dfloat*) calloc(mesh->Nelements*mesh->Np,sizeof(dfloat));
  mesh->y = (dfloat*) calloc(mesh->Nelements*mesh->Np,sizeof(dfloat));
  mesh->z = (dfloat*) calloc(mesh->Nelements*mesh->Np,sizeof(dfloat));
  
  dlong cnt = 0;
  for(dlong e=0;e<mesh->Nelements;++e){ /* for each element */

    dlong id = e*mesh->Nverts;

    dfloat xe1 = mesh->EX[id+0]; /* x-coordinates of vertices */
    dfloat xe2 = mesh->EX[id+1];
    dfloat xe3 = mesh->EX[id+2];
    dfloat xe4 = mesh->EX[id+3];
    dfloat xe5 = mesh->EX[id+4]; 
    dfloat xe6 = mesh->EX[id+5];
    dfloat xe7 = mesh->EX[id+6];
    dfloat xe8 = mesh->EX[id+7];
    
    dfloat ye1 = mesh->EY[id+0]; /* y-coordinates of vertices */
    dfloat ye2 = mesh->EY[id+1];
    dfloat ye3 = mesh->EY[id+2];
    dfloat ye4 = mesh->EY[id+3];
    dfloat ye5 = mesh->EY[id+4]; 
    dfloat ye6 = mesh->EY[id+5];
    dfloat ye7 = mesh->EY[id+6];
    dfloat ye8 = mesh->EY[id+7];

    dfloat ze1 = mesh->EZ[id+0]; /* z-coordinates of vertices */
    dfloat ze2 = mesh->EZ[id+1];
    dfloat ze3 = mesh->EZ[id+2];
    dfloat ze4 = mesh->EZ[id+3];
    dfloat ze5 = mesh->EZ[id+4]; 
    dfloat ze6 = mesh->EZ[id+5];
    dfloat ze7 = mesh->EZ[id+6];
    dfloat ze8 = mesh->EZ[id+7];

    for(int n=0;n<mesh->Np;++n){ /* for each node */
      
      /* (r,s,t) coordinates of interpolation nodes*/
      dfloat rn = mesh->r[n]; 
      dfloat sn = mesh->s[n];
      dfloat tn = mesh->t[n];

      /* physical coordinate of interpolation node */
      mesh->x[cnt] = 
        +0.125*(1-rn)*(1-sn)*(1-tn)*xe1
        +0.125*(1+rn)*(1-sn)*(1-tn)*xe2
        +0.125*(1+rn)*(1+sn)*(1-tn)*xe3
        +0.125*(1-rn)*(1+sn)*(1-tn)*xe4
        +0.125*(1-rn)*(1-sn)*(1+tn)*xe5
        +0.125*(1+rn)*(1-sn)*(1+tn)*xe6
        +0.125*(1+rn)*(1+sn)*(1+tn)*xe7
        +0.125*(1-rn)*(1+sn)*(1+tn)*xe8;

      mesh->y[cnt] = 
        +0.125*(1-rn)*(1-sn)*(1-tn)*ye1
        +0.125*(1+rn)*(1-sn)*(1-tn)*ye2
        +0.125*(1+rn)*(1+sn)*(1-tn)*ye3
        +0.125*(1-rn)*(1+sn)*(1-tn)*ye4
        +0.125*(1-rn)*(1-sn)*(1+tn)*ye5
        +0.125*(1+rn)*(1-sn)*(1+tn)*ye6
        +0.125*(1+rn)*(1+sn)*(1+tn)*ye7
        +0.125*(1-rn)*(1+sn)*(1+tn)*ye8;

      mesh->z[cnt] = 
        +0.125*(1-rn)*(1-sn)*(1-tn)*ze1
        +0.125*(1+rn)*(1-sn)*(1-tn)*ze2
        +0.125*(1+rn)*(1+sn)*(1-tn)*ze3
        +0.125*(1-rn)*(1+sn)*(1-tn)*ze4
        +0.125*(1-rn)*(1-sn)*(1+tn)*ze5
        +0.125*(1+rn)*(1-sn)*(1+tn)*ze6
        +0.125*(1+rn)*(1+sn)*(1+tn)*ze7
        +0.125*(1-rn)*(1+sn)*(1+tn)*ze8;

      ++cnt;
    }
  }
}

void meshChooseBoxDimensions(hlong Nelements, int *NX, int *NY, int *NZ){

  int Ncr = pow(Nelements, 1./3);
  
  int bestNX = Ncr, bestNY = Ncr, bestNZ = Ncr;

  int mismatch = Nelements - Ncr*Ncr*Ncr;

  int delta = (Ncr>40) ? 40:Ncr-1;

  for(int k=-delta;k<=delta;++k){
    for(int j=k;j<=delta;++j){
      for(int i=j;i<=delta;++i){
	int nx = Ncr + i;
	int ny = Ncr + j;
	int nz = Ncr + k;
	int newMismatch = Nelements - nx*ny*nz;
	if(fabs(newMismatch)<fabs(mismatch)){
	  bestNX = nx;
	  bestNY = ny;
	  bestNZ = nz;
	  mismatch = newMismatch;
	}
      }
    }
  }
  
  *NX = bestNX;
  *NY = bestNY;
  *NZ = bestNZ;
  
}

mesh3D *meshSetupBoxHex3D(int N, int cubN, setupAide &options){

  mesh_t *mesh = new mesh_t();
  
  int rank, size;
  
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  
  MPI_Comm_dup(MPI_COMM_WORLD, &mesh->comm);
  
  mesh->rank = rank;
  mesh->size = size;
  
  mesh->dim = 3;
  mesh->Nverts = 8; // number of vertices per element
  mesh->Nfaces = 6;
  mesh->NfaceVertices = 4;
  
  // vertices on each face
  int faceVertices[6][4] =
    {{0,1,2,3},{0,1,5,4},{1,2,6,5},{2,3,7,6},{3,0,4,7},{4,5,6,7}};

  mesh->faceVertices =
    (int*) calloc(mesh->NfaceVertices*mesh->Nfaces, sizeof(int));

  memcpy(mesh->faceVertices, faceVertices[0], mesh->NfaceVertices*mesh->Nfaces*sizeof(int));
  
  // build an NX x NY x NZ periodic box grid
  
  hlong NX = 3, NY = 3, NZ = 3; // defaults

  int NdofsTarget = 0;
  options.getArgs("TARGET NODES", NdofsTarget);
  if(NdofsTarget){
    int tmpNp = (N+1)*(N+1)*(N+1);
    int NelementsTarget = (NdofsTarget+tmpNp-1)/tmpNp;
    meshChooseBoxDimensions(NelementsTarget, &NX, &NY, &NZ);
    printf("TARGET NODES = %d, ACTUAL NODES = %d, NELEMENTS = [%d,%d,%d=>%d]\n", NdofsTarget, NX*NY*NZ*tmpNp, NX,NY,NZ, NX*NY*NZ);
  }else{
    options.getArgs("BOX NX", NX);
    options.getArgs("BOX NY", NY);
    options.getArgs("BOX NZ", NZ);
  }
  
  dfloat XMIN = -1, XMAX = +1; // default bi-unit cube
  dfloat YMIN = -1, YMAX = +1;
  dfloat ZMIN = -1, ZMAX = +1;
  
  options.getArgs("BOX XMIN", XMIN);
  options.getArgs("BOX YMIN", YMIN);
  options.getArgs("BOX ZMIN", ZMIN);

  options.getArgs("BOX XMAX", XMAX);
  options.getArgs("BOX YMAX", YMAX);
  options.getArgs("BOX ZMAX", ZMAX);

  hlong allNelements = NX*NY*NZ;

  hlong chunkNelements = allNelements/size;

  hlong start = chunkNelements*rank;
  hlong end   = chunkNelements*(rank+1);
  
  if(mesh->rank==(size-1))
    end = allNelements;
    

  mesh->Nnodes = NX*NY*NZ; // assume periodic and global number of nodes
  mesh->Nelements = end-start;
  mesh->NboundaryFaces = 0;

  printf("Rank %d initially has %d elements\n", mesh->rank, mesh->Nelements);
  
  mesh->EToV = (hlong*) calloc(mesh->Nelements*mesh->Nverts, sizeof(hlong));

  mesh->EX = (dfloat*) calloc(mesh->Nelements*mesh->Nverts, sizeof(dfloat));
  mesh->EY = (dfloat*) calloc(mesh->Nelements*mesh->Nverts, sizeof(dfloat));
  mesh->EZ = (dfloat*) calloc(mesh->Nelements*mesh->Nverts, sizeof(dfloat));

  mesh->elementInfo = (hlong*) calloc(mesh->Nelements, sizeof(hlong));
  
  // [0,NX]
  dfloat dx = (XMAX-XMIN)/NX; // xmin+0*dx, xmin + NX*(XMAX-XMIN)/NX
  dfloat dy = (YMAX-YMIN)/NY;
  dfloat dz = (ZMAX-ZMIN)/NZ;
  for(hlong n=start;n<end;++n){

    int i = n%NX;      // [0, NX)
    int j = (n/NX)%NY; // [0, NY)
    int k = n/(NX*NY); // [0, NZ)

    hlong e = n-start;

    int ip = (i+1)%NX;
    int jp = (j+1)%NY;
    int kp = (k+1)%NZ;

    // do not use for coordinates
    mesh->EToV[e*mesh->Nverts+0] = i  +  j*NX + k*NX*NY;
    mesh->EToV[e*mesh->Nverts+1] = ip +  j*NX + k*NX*NY;
    mesh->EToV[e*mesh->Nverts+2] = ip + jp*NX + k*NX*NY;
    mesh->EToV[e*mesh->Nverts+3] = i  + jp*NX + k*NX*NY;

    mesh->EToV[e*mesh->Nverts+4] = i  +  j*NX + kp*NX*NY;
    mesh->EToV[e*mesh->Nverts+5] = ip +  j*NX + kp*NX*NY;
    mesh->EToV[e*mesh->Nverts+6] = ip + jp*NX + kp*NX*NY;
    mesh->EToV[e*mesh->Nverts+7] = i  + jp*NX + kp*NX*NY;

    dfloat xo = XMIN + dx*i;
    dfloat yo = YMIN + dy*j;
    dfloat zo = ZMIN + dz*k;
    
    dfloat *ex = mesh->EX+e*mesh->Nverts;
    dfloat *ey = mesh->EY+e*mesh->Nverts;
    dfloat *ez = mesh->EZ+e*mesh->Nverts;
    
    ex[0] = xo;    ey[0] = yo;    ez[0] = zo;
    ex[1] = xo+dx; ey[1] = yo;    ez[1] = zo;
    ex[2] = xo+dx; ey[2] = yo+dy; ez[2] = zo;
    ex[3] = xo;    ey[3] = yo+dy; ez[3] = zo;

    ex[4] = xo;    ey[4] = yo;    ez[4] = zo+dz;
    ex[5] = xo+dx; ey[5] = yo;    ez[5] = zo+dz;
    ex[6] = xo+dx; ey[6] = yo+dy; ez[6] = zo+dz;
    ex[7] = xo;    ey[7] = yo+dy; ez[7] = zo+dz;

    mesh->elementInfo[e] = 1; // ?
    
  }
  
  // partition elements using Morton ordering & parallel sort
  meshGeometricPartition3D(mesh);

  mesh->EToB = (int*) calloc(mesh->Nelements*mesh->Nfaces, sizeof(int)); 

  // connect elements using parallel sort
  meshParallelConnect(mesh);
  
  // print out connectivity statistics
  meshPartitionStatistics(mesh);

  // load reference (r,s,t) element nodes
  meshLoadReferenceNodesHex3D(mesh, N, cubN);

  // compute physical (x,y) locations of the element nodes
  meshPhysicalNodesHex3D(mesh);

  // compute geometric factors
  meshGeometricFactorsHex3D(mesh);

  // set up halo exchange info for MPI (do before connect face nodes)
  meshHaloSetup(mesh);

  // connect face nodes (find trace indices)
  meshConnectPeriodicFaceNodes3D(mesh,XMAX-XMIN,YMAX-YMIN,ZMAX-ZMIN); // needs to fix this !

  // compute surface geofacs (including halo)
  meshSurfaceGeometricFactorsHex3D(mesh);
  
  // global nodes
  meshParallelConnectNodes(mesh); 

  // localized numbering (contiguous on node)
  meshLocalizedConnectNodes(mesh);
  
  return mesh;
}

mesh3D *meshSetupBoxTet3D(int N, int cubN, setupAide &options){

  mesh_t *mesh = new mesh_t();
  
  int rank, size;
  
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  
  MPI_Comm_dup(MPI_COMM_WORLD, &mesh->comm);
  
  mesh->rank = rank;
  mesh->size = size;
  
  mesh->dim = 3;
  mesh->Nverts = 4; // number of vertices per element
  mesh->Nfaces = 4;
  mesh->NfaceVertices = 3;
  
  // vertices on each face
  int faceVertices[4][3] =
    {{0,1,2},{0,1,3},{1,2,3},{0,2,3}}; // check

  mesh->faceVertices =
    (int*) calloc(mesh->NfaceVertices*mesh->Nfaces, sizeof(int));

  memcpy(mesh->faceVertices, faceVertices[0], mesh->NfaceVertices*mesh->Nfaces*sizeof(int));
  
  // build an NX x NY x NZ periodic box grid
  
  hlong NX = 3, NY = 3, NZ = 3; // defaults

  int NdofsTarget = 0;
  options.getArgs("TARGET NODES", NdofsTarget);
  if(NdofsTarget){
    int tmpNp = 6*(N+1)*(N+2)*(N+3)/6;
    int NboxesTarget = (NdofsTarget+tmpNp-1)/tmpNp;
    meshChooseBoxDimensions(NboxesTarget, &NX, &NY, &NZ);
    printf("TARGET NODES = %d, ACTUAL NODES = %d, NELEMENTS = [%d,%d,%d=>%d]\n", NdofsTarget, NX*NY*NZ*tmpNp, NX,NY,NZ, NX*NY*NZ);
  }else{
    options.getArgs("BOX NX", NX);
    options.getArgs("BOX NY", NY);
    options.getArgs("BOX NZ", NZ);
  }
  
  dfloat XMIN = -1, XMAX = +1; // default bi-unit cube
  dfloat YMIN = -1, YMAX = +1;
  dfloat ZMIN = -1, ZMAX = +1;
  
  options.getArgs("BOX XMIN", XMIN);
  options.getArgs("BOX YMIN", YMIN);
  options.getArgs("BOX ZMIN", ZMIN);

  options.getArgs("BOX XMAX", XMAX);
  options.getArgs("BOX YMAX", YMAX);
  options.getArgs("BOX ZMAX", ZMAX);

  hlong allNelements = NX*NY*NZ;

  hlong chunkNelements = allNelements/size;

  hlong start = chunkNelements*rank;
  hlong end   = chunkNelements*(rank+1);
  
  if(mesh->rank==(size-1))
    end = allNelements;
    

  mesh->Nnodes = NX*NY*NZ; // assume periodic and global number of nodes
  mesh->Nelements = 6*(end-start);
  mesh->NboundaryFaces = 0;

  printf("Rank %d initially has %d elements\n", mesh->rank, mesh->Nelements);
  
  mesh->EToV = (hlong*) calloc(mesh->Nelements*mesh->Nverts, sizeof(hlong));

  mesh->EX = (dfloat*) calloc(mesh->Nelements*mesh->Nverts, sizeof(dfloat));
  mesh->EY = (dfloat*) calloc(mesh->Nelements*mesh->Nverts, sizeof(dfloat));
  mesh->EZ = (dfloat*) calloc(mesh->Nelements*mesh->Nverts, sizeof(dfloat));

  mesh->elementInfo = (hlong*) calloc(mesh->Nelements, sizeof(hlong));
  
  // [0,NX]
  dfloat dx = (XMAX-XMIN)/NX; // xmin+0*dx, xmin + NX*(XMAX-XMIN)/NX
  dfloat dy = (YMAX-YMIN)/NY;
  dfloat dz = (ZMAX-ZMIN)/NZ;
  for(hlong n=start;n<end;++n){

    int i = n%NX;      // [0, NX)
    int j = (n/NX)%NY; // [0, NY)
    int k = n/(NX*NY); // [0, NZ)

    hlong eo = 6*(n-start);
  
    int ip = (i+1)%NX;
    int jp = (j+1)%NY;
    int kp = (k+1)%NZ;

    int a = i  +  j*NX + k*NX*NY;
    int b = ip +  j*NX + k*NX*NY;
    int c = ip + jp*NX + k*NX*NY;
    int d = i  + jp*NX + k*NX*NY;

    int e = i  +  j*NX + kp*NX*NY;
    int f = ip +  j*NX + kp*NX*NY;
    int g = ip + jp*NX + kp*NX*NY;
    int h = i  + jp*NX + kp*NX*NY;
    
    // do not use for coordinates
    hlong *EToV0 = mesh->EToV+(eo+0)*mesh->Nverts;
    hlong *EToV1 = mesh->EToV+(eo+1)*mesh->Nverts;
    hlong *EToV2 = mesh->EToV+(eo+2)*mesh->Nverts;
    hlong *EToV3 = mesh->EToV+(eo+3)*mesh->Nverts;
    hlong *EToV4 = mesh->EToV+(eo+4)*mesh->Nverts;
    hlong *EToV5 = mesh->EToV+(eo+5)*mesh->Nverts;
    
    EToV0[0] = a; EToV0[1] = b; EToV0[2] = h; EToV0[3] = e;
    EToV1[0] = a; EToV1[1] = b; EToV1[2] = d; EToV1[3] = h;
    EToV2[0] = b; EToV2[1] = c; EToV2[2] = d; EToV2[3] = h;

    EToV3[0] = b; EToV3[1] = h; EToV3[2] = e; EToV3[3] = f;
    EToV4[0] = b; EToV4[1] = h; EToV4[2] = f; EToV4[3] = g;
    EToV5[0] = h; EToV5[1] = b; EToV5[2] = c; EToV5[3] = g;

    dfloat xa = XMIN + i*dx, ya = YMIN + j*dy, za = ZMIN + k*dz;
    dfloat xb = XMIN +(i+1)*dx, yb = YMIN + j*dy, zb = ZMIN + k*dz;
    dfloat xc = XMIN +(i+1)*dx, yc = YMIN +(j+1)*dy, zc = ZMIN + k*dz;
    dfloat xd = XMIN + i*dx, yd = YMIN + (j+1)*dy, zd = ZMIN + k*dz;

    dfloat xe = XMIN + i*dx, ye = YMIN + j*dy, ze = ZMIN + (k+1)*dz;
    dfloat xf = XMIN +(i+1)*dx, yf = YMIN + j*dy, zf = ZMIN + (k+1)*dz;
    dfloat xg = XMIN +(i+1)*dx, yg = YMIN + (j+1)*dy, zg = ZMIN + (k+1)*dz;
    dfloat xh = XMIN + i*dx, yh = YMIN + (j+1)*dy, zh = ZMIN + (k+1)*dz;
    
    dfloat *ex0 = mesh->EX+(eo+0)*mesh->Nverts, *ey0 = mesh->EY+(eo+0)*mesh->Nverts, *ez0 = mesh->EZ+(eo+0)*mesh->Nverts;
    dfloat *ex1 = mesh->EX+(eo+1)*mesh->Nverts, *ey1 = mesh->EY+(eo+1)*mesh->Nverts, *ez1 = mesh->EZ+(eo+1)*mesh->Nverts;
    dfloat *ex2 = mesh->EX+(eo+2)*mesh->Nverts, *ey2 = mesh->EY+(eo+2)*mesh->Nverts, *ez2 = mesh->EZ+(eo+2)*mesh->Nverts;
    dfloat *ex3 = mesh->EX+(eo+3)*mesh->Nverts, *ey3 = mesh->EY+(eo+3)*mesh->Nverts, *ez3 = mesh->EZ+(eo+3)*mesh->Nverts;
    dfloat *ex4 = mesh->EX+(eo+4)*mesh->Nverts, *ey4 = mesh->EY+(eo+4)*mesh->Nverts, *ez4 = mesh->EZ+(eo+4)*mesh->Nverts;
    dfloat *ex5 = mesh->EX+(eo+5)*mesh->Nverts, *ey5 = mesh->EY+(eo+5)*mesh->Nverts, *ez5 = mesh->EZ+(eo+5)*mesh->Nverts;

    ex0[0] = xa; ex0[1] = xb; ex0[2] = xh; ex0[3] = xe;
    ex1[0] = xa; ex1[1] = xb; ex1[2] = xd; ex1[3] = xh;
    ex2[0] = xb; ex2[1] = xc; ex2[2] = xd; ex2[3] = xh;
    ex3[0] = xb; ex3[1] = xh; ex3[2] = xe; ex3[3] = xf;
    ex4[0] = xb; ex4[1] = xh; ex4[2] = xf; ex4[3] = xg;
    ex5[0] = xh; ex5[1] = xb; ex5[2] = xc; ex5[3] = xg;

    ey0[0] = ya; ey0[1] = yb; ey0[2] = yh; ey0[3] = ye;
    ey1[0] = ya; ey1[1] = yb; ey1[2] = yd; ey1[3] = yh;
    ey2[0] = yb; ey2[1] = yc; ey2[2] = yd; ey2[3] = yh;
    ey3[0] = yb; ey3[1] = yh; ey3[2] = ye; ey3[3] = yf;
    ey4[0] = yb; ey4[1] = yh; ey4[2] = yf; ey4[3] = yg;
    ey5[0] = yh; ey5[1] = yb; ey5[2] = yc; ey5[3] = yg;

    ez0[0] = za; ez0[1] = zb; ez0[2] = zh; ez0[3] = ze;
    ez1[0] = za; ez1[1] = zb; ez1[2] = zd; ez1[3] = zh;
    ez2[0] = zb; ez2[1] = zc; ez2[2] = zd; ez2[3] = zh;
    ez3[0] = zb; ez3[1] = zh; ez3[2] = ze; ez3[3] = zf;
    ez4[0] = zb; ez4[1] = zh; ez4[2] = zf; ez4[3] = zg;
    ez5[0] = zh; ez5[1] = zb; ez5[2] = zc; ez5[3] = zg;

    mesh->elementInfo[e] = 1; // ?
    
  }
  
  // partition elements using Morton ordering & parallel sort
  meshGeometricPartition3D(mesh);

  mesh->EToB = (int*) calloc(mesh->Nelements*mesh->Nfaces, sizeof(int)); 

  // connect elements using parallel sort
  meshParallelConnect(mesh);
  
  // print out connectivity statistics
  meshPartitionStatistics(mesh);

  // load reference (r,s,t) element nodes
  meshLoadReferenceNodesTet3D(mesh, N, cubN);

  // compute physical (x,y) locations of the element nodes
  meshPhysicalNodesTet3D(mesh);

  // compute geometric factors
  meshGeometricFactorsTet3D(mesh);

  // set up halo exchange info for MPI (do before connect face nodes)
  meshHaloSetup(mesh);

  // connect face nodes (find trace indices)
  meshConnectPeriodicFaceNodes3D(mesh,XMAX-XMIN,YMAX-YMIN,ZMAX-ZMIN); // needs to fix this !

  // compute surface geofacs (including halo)
  meshSurfaceGeometricFactorsTet3D(mesh);
  
  // global nodes
  meshParallelConnectNodes(mesh); 

  // localized numbering (contiguous on node)
  meshLocalizedConnectNodes(mesh);
  
  return mesh;
}





void interpolateFaceHex3D(int *faceNodes, dfloat *I, dfloat *x, int N, dfloat *Ix, int M){
  
  dfloat *Ix0 = (dfloat*) calloc(N*N, sizeof(dfloat));
  dfloat *Ix1 = (dfloat*) calloc(N*M, sizeof(dfloat));
  
  for(int j=0;j<N;++j){
    for(int i=0;i<N;++i){
      Ix0[j*N+i] = x[faceNodes[j*N+i]];
    }
  }
  
  for(int j=0;j<N;++j){
    for(int i=0;i<M;++i){
      dfloat tmp = 0;
      for(int n=0;n<N;++n){
	tmp += I[i*N + n]*Ix0[j*N+n];
      }
      Ix1[j*M+i] = tmp;
    }
  }

  for(int j=0;j<M;++j){
    for(int i=0;i<M;++i){
      dfloat tmp = 0;
      for(int n=0;n<N;++n){
	tmp += I[j*N + n]*Ix1[n*M+i];
      }
      Ix[j*M+i] = tmp;
    }
  }

  free(Ix0);
  free(Ix1);
  
}


void meshSurfaceGeometricFactorsTet3D(mesh3D *mesh){

  /* unified storage array for geometric factors */
  mesh->Nsgeo = 14;
  mesh->sgeo = (dfloat*) calloc((mesh->Nelements+mesh->totalHaloPairs)*
                            mesh->Nsgeo*mesh->Nfaces, sizeof(dfloat));
  
  for(dlong e=0;e<mesh->Nelements+mesh->totalHaloPairs;++e){ /* for each element */

    /* find vertex indices and physical coordinates */
    dlong id = e*mesh->Nverts;
    dfloat xe1 = mesh->EX[id+0], ye1 = mesh->EY[id+0], ze1 = mesh->EZ[id+0];
    dfloat xe2 = mesh->EX[id+1], ye2 = mesh->EY[id+1], ze2 = mesh->EZ[id+1];
    dfloat xe3 = mesh->EX[id+2], ye3 = mesh->EY[id+2], ze3 = mesh->EZ[id+2];
    dfloat xe4 = mesh->EX[id+3], ye4 = mesh->EY[id+3], ze4 = mesh->EZ[id+3];

    /* Jacobian matrix */
    dfloat xr = 0.5*(xe2-xe1), xs = 0.5*(xe3-xe1), xt = 0.5*(xe4-xe1);
    dfloat yr = 0.5*(ye2-ye1), ys = 0.5*(ye3-ye1), yt = 0.5*(ye4-ye1);
    dfloat zr = 0.5*(ze2-ze1), zs = 0.5*(ze3-ze1), zt = 0.5*(ze4-ze1);

    /* compute geometric factors for affine coordinate transform*/
    dfloat J = xr*(ys*zt-zs*yt) - yr*(xs*zt-zs*xt) + zr*(xs*yt-ys*xt);
    dfloat rx =  (ys*zt - zs*yt)/J, ry = -(xs*zt - zs*xt)/J, rz =  (xs*yt - ys*xt)/J;
    dfloat sx = -(yr*zt - zr*yt)/J, sy =  (xr*zt - zr*xt)/J, sz = -(xr*yt - yr*xt)/J;
    dfloat tx =  (yr*zs - zr*ys)/J, ty = -(xr*zs - zr*xs)/J, tz =  (xr*ys - yr*xs)/J;

    if(J<0) printf("bugger: got negative geofac\n");
    
    /* face 1 */
    dlong base = mesh->Nsgeo*mesh->Nfaces*e;
    dfloat nx1 = -tx;
    dfloat ny1 = -ty;
    dfloat nz1 = -tz;
    dfloat sJ1 = norm3(nx1,ny1,nz1);

    mesh->sgeo[base+NXID] = nx1/sJ1;
    mesh->sgeo[base+NYID] = ny1/sJ1;
    mesh->sgeo[base+NZID] = nz1/sJ1;
    mesh->sgeo[base+SJID] = sJ1*J;
    mesh->sgeo[base+IJID] = 1./J;

    /* face 2 */
    base += mesh->Nsgeo;
    dfloat nx2 = -sx;
    dfloat ny2 = -sy;
    dfloat nz2 = -sz;
    dfloat sJ2 = norm3(nx2,ny2,nz2);

    mesh->sgeo[base+NXID] = nx2/sJ2;
    mesh->sgeo[base+NYID] = ny2/sJ2;
    mesh->sgeo[base+NZID] = nz2/sJ2;
    mesh->sgeo[base+SJID] = sJ2*J;
    mesh->sgeo[base+IJID] = 1./J;

    /* face 3 */
    base += mesh->Nsgeo;
    dfloat nx3 = rx+sx+tx;
    dfloat ny3 = ry+sy+ty;
    dfloat nz3 = rz+sz+tz;
    dfloat sJ3 = norm3(nx3,ny3,nz3);

    mesh->sgeo[base+NXID] = nx3/sJ3;
    mesh->sgeo[base+NYID] = ny3/sJ3;
    mesh->sgeo[base+NZID] = nz3/sJ3;
    mesh->sgeo[base+SJID] = sJ3*J;
    mesh->sgeo[base+IJID] = 1./J;

    /* face 4 */
    base += mesh->Nsgeo;
    dfloat nx4 = -rx;
    dfloat ny4 = -ry;
    dfloat nz4 = -rz;
    dfloat sJ4 = norm3(nx4,ny4,nz4);

    mesh->sgeo[base+NXID] = nx4/sJ4;
    mesh->sgeo[base+NYID] = ny4/sJ4;
    mesh->sgeo[base+NZID] = nz4/sJ4;
    mesh->sgeo[base+SJID] = sJ4*J;
    mesh->sgeo[base+IJID] = 1./J;

  }
  
  for(dlong e=0;e<mesh->Nelements;++e){ /* for each non-halo element */
    for(int f=0;f<mesh->Nfaces;++f){
      dlong baseM = e*mesh->Nfaces + f;
      
      // awkward: (need to find eP,fP relative to bulk+halo)
      dlong idP = mesh->vmapP[e*mesh->Nfp*mesh->Nfaces+f*mesh->Nfp+0];
      dlong eP = (idP>=0) ? (idP/mesh->Np):e;
      
      int fP = mesh->EToF[baseM];
      fP = (fP==-1) ? f:fP;
      
      dlong baseP = eP*mesh->Nfaces + fP;
      
      // rescaling,  V = A*h/3 => (J*4/3) = (sJ*2)*h/3 => h  = 0.5*J/sJ
      dfloat hinvM = 0.5*mesh->sgeo[baseM*mesh->Nsgeo + SJID]*mesh->sgeo[baseM*mesh->Nsgeo + IJID];
      dfloat hinvP = 0.5*mesh->sgeo[baseP*mesh->Nsgeo + SJID]*mesh->sgeo[baseP*mesh->Nsgeo + IJID];
      
      mesh->sgeo[baseM*mesh->Nsgeo+IHID] = mymax(hinvM,hinvP);
      mesh->sgeo[baseP*mesh->Nsgeo+IHID] = mymax(hinvM,hinvP);
    }
  }
}


/* compute outwards facing normals, surface Jacobian, and volume Jacobian for all face nodes */
void meshSurfaceGeometricFactorsHex3D(mesh3D *mesh){

  /* unified storage array for geometric factors */
  mesh->Nsgeo = 17;
  mesh->sgeo = (dfloat*) calloc((mesh->Nelements+mesh->totalHaloPairs)*
                                mesh->Nsgeo*mesh->Nfp*mesh->Nfaces, 
                                sizeof(dfloat));

  mesh->cubsgeo = (dfloat*) calloc((mesh->Nelements+mesh->totalHaloPairs)*
				   mesh->Nsgeo*mesh->cubNfp*mesh->Nfaces, 
				   sizeof(dfloat));
  
  dfloat *xre = (dfloat*) calloc(mesh->Np, sizeof(dfloat));
  dfloat *xse = (dfloat*) calloc(mesh->Np, sizeof(dfloat));
  dfloat *xte = (dfloat*) calloc(mesh->Np, sizeof(dfloat));
  dfloat *yre = (dfloat*) calloc(mesh->Np, sizeof(dfloat));
  dfloat *yse = (dfloat*) calloc(mesh->Np, sizeof(dfloat));
  dfloat *yte = (dfloat*) calloc(mesh->Np, sizeof(dfloat));
  dfloat *zre = (dfloat*) calloc(mesh->Np, sizeof(dfloat));
  dfloat *zse = (dfloat*) calloc(mesh->Np, sizeof(dfloat));
  dfloat *zte = (dfloat*) calloc(mesh->Np, sizeof(dfloat));

  dfloat *cubxre = (dfloat*) calloc(mesh->cubNq*mesh->cubNq, sizeof(dfloat));
  dfloat *cubxse = (dfloat*) calloc(mesh->cubNq*mesh->cubNq, sizeof(dfloat));
  dfloat *cubxte = (dfloat*) calloc(mesh->cubNq*mesh->cubNq, sizeof(dfloat));
  dfloat *cubyre = (dfloat*) calloc(mesh->cubNq*mesh->cubNq, sizeof(dfloat));
  dfloat *cubyse = (dfloat*) calloc(mesh->cubNq*mesh->cubNq, sizeof(dfloat));
  dfloat *cubyte = (dfloat*) calloc(mesh->cubNq*mesh->cubNq, sizeof(dfloat));
  dfloat *cubzre = (dfloat*) calloc(mesh->cubNq*mesh->cubNq, sizeof(dfloat));
  dfloat *cubzse = (dfloat*) calloc(mesh->cubNq*mesh->cubNq, sizeof(dfloat));
  dfloat *cubzte = (dfloat*) calloc(mesh->cubNq*mesh->cubNq, sizeof(dfloat));

  dfloat *cubxe = (dfloat*) calloc(mesh->cubNq*mesh->cubNq, sizeof(dfloat));
  dfloat *cubye = (dfloat*) calloc(mesh->cubNq*mesh->cubNq, sizeof(dfloat));
  dfloat *cubze = (dfloat*) calloc(mesh->cubNq*mesh->cubNq, sizeof(dfloat));
  
  for(dlong e=0;e<mesh->Nelements+mesh->totalHaloPairs;++e){ /* for each element */

    /* find vertex indices and physical coordinates */
    dlong id = e*mesh->Nverts;

    dfloat *xe = mesh->EX + id;
    dfloat *ye = mesh->EY + id;
    dfloat *ze = mesh->EZ + id;

    for(int n=0;n<mesh->Np;++n){
      xre[n] = 0; xse[n] = 0; xte[n] = 0;
      yre[n] = 0; yse[n] = 0; yte[n] = 0; 
      zre[n] = 0; zse[n] = 0; zte[n] = 0; 
    }

    for(int k=0;k<mesh->Nq;++k){
      for(int j=0;j<mesh->Nq;++j){
        for(int i=0;i<mesh->Nq;++i){
	  
          int n = i + j*mesh->Nq + k*mesh->Nq*mesh->Nq;

          /* local node coordinates */
          dfloat rn = mesh->r[n]; 
          dfloat sn = mesh->s[n];
          dfloat tn = mesh->t[n];

	  for(int m=0;m<mesh->Nq;++m){
	    int idr = e*mesh->Np + k*mesh->Nq*mesh->Nq + j*mesh->Nq + m;
	    int ids = e*mesh->Np + k*mesh->Nq*mesh->Nq + m*mesh->Nq + i;
	    int idt = e*mesh->Np + m*mesh->Nq*mesh->Nq + j*mesh->Nq + i;
	    xre[n] += mesh->D[i*mesh->Nq+m]*mesh->x[idr];
	    xse[n] += mesh->D[j*mesh->Nq+m]*mesh->x[ids];
	    xte[n] += mesh->D[k*mesh->Nq+m]*mesh->x[idt];
	    yre[n] += mesh->D[i*mesh->Nq+m]*mesh->y[idr];
	    yse[n] += mesh->D[j*mesh->Nq+m]*mesh->y[ids];
	    yte[n] += mesh->D[k*mesh->Nq+m]*mesh->y[idt];
	    zre[n] += mesh->D[i*mesh->Nq+m]*mesh->z[idr];
	    zse[n] += mesh->D[j*mesh->Nq+m]*mesh->z[ids];
	    zte[n] += mesh->D[k*mesh->Nq+m]*mesh->z[idt];
	  }
	}
      }
    }
	  
    for(int f=0;f<mesh->Nfaces;++f){ // for each face
      
      for(int i=0;i<mesh->Nfp;++i){  // for each node on face

        /* volume index of face node */
        int n = mesh->faceNodes[f*mesh->Nfp+i];

        /* local node coordinates */
        dfloat rn = mesh->r[n]; 
        dfloat sn = mesh->s[n];
        dfloat tn = mesh->t[n];
	
	dfloat xr = xre[n], xs = xse[n], xt = xte[n];
	dfloat yr = yre[n], ys = yse[n], yt = yte[n];
	dfloat zr = zre[n], zs = zse[n], zt = zte[n];
	
        /* determinant of Jacobian matrix */
        dfloat J = xr*(ys*zt-zs*yt) - yr*(xs*zt-zs*xt) + zr*(xs*yt-ys*xt);
        
        dfloat rx =  (ys*zt - zs*yt)/J, ry = -(xs*zt - zs*xt)/J, rz =  (xs*yt - ys*xt)/J;
        dfloat sx = -(yr*zt - zr*yt)/J, sy =  (xr*zt - zr*xt)/J, sz = -(xr*yt - yr*xt)/J;
        dfloat tx =  (yr*zs - zr*ys)/J, ty = -(xr*zs - zr*xs)/J, tz =  (xr*ys - yr*xs)/J;
        
        /* face f normal and length */
        dfloat nx, ny, nz, d;
        switch(f){
        case 0: nx = -tx; ny = -ty; nz = -tz; break;
        case 1: nx = -sx; ny = -sy; nz = -sz; break;
        case 2: nx = +rx; ny = +ry; nz = +rz; break;
        case 3: nx = +sx; ny = +sy; nz = +sz; break;
        case 4: nx = -rx; ny = -ry; nz = -rz; break;
        case 5: nx = +tx; ny = +ty; nz = +tz; break;
        }

        dfloat sJ = sqrt(nx*nx+ny*ny+nz*nz);
        nx /= sJ; ny /= sJ; nz /= sJ;
        sJ *= J;
        
        /* output index */
        dlong base = mesh->Nsgeo*(mesh->Nfaces*mesh->Nfp*e + mesh->Nfp*f + i);

        /* store normal, surface Jacobian, and reciprocal of volume Jacobian */
        mesh->sgeo[base+NXID] = nx;
        mesh->sgeo[base+NYID] = ny;
        mesh->sgeo[base+NZID] = nz;
        mesh->sgeo[base+SJID] = sJ;
        mesh->sgeo[base+IJID] = 1./J;

        mesh->sgeo[base+WIJID] = 1./(J*mesh->gllw[0]);
        mesh->sgeo[base+WSJID] = sJ*mesh->gllw[i%mesh->Nq]*mesh->gllw[i/mesh->Nq];

      }
    
      // now interpolate geofacs to cubature
      interpolateFaceHex3D(mesh->faceNodes+f*mesh->Nfp, mesh->cubInterp, xre, mesh->Nq, cubxre, mesh->cubNq);
      interpolateFaceHex3D(mesh->faceNodes+f*mesh->Nfp, mesh->cubInterp, xse, mesh->Nq, cubxse, mesh->cubNq);
      interpolateFaceHex3D(mesh->faceNodes+f*mesh->Nfp, mesh->cubInterp, xte, mesh->Nq, cubxte, mesh->cubNq);
      interpolateFaceHex3D(mesh->faceNodes+f*mesh->Nfp, mesh->cubInterp, yre, mesh->Nq, cubyre, mesh->cubNq);
      interpolateFaceHex3D(mesh->faceNodes+f*mesh->Nfp, mesh->cubInterp, yse, mesh->Nq, cubyse, mesh->cubNq);
      interpolateFaceHex3D(mesh->faceNodes+f*mesh->Nfp, mesh->cubInterp, yte, mesh->Nq, cubyte, mesh->cubNq);
      interpolateFaceHex3D(mesh->faceNodes+f*mesh->Nfp, mesh->cubInterp, zre, mesh->Nq, cubzre, mesh->cubNq);
      interpolateFaceHex3D(mesh->faceNodes+f*mesh->Nfp, mesh->cubInterp, zse, mesh->Nq, cubzse, mesh->cubNq);
      interpolateFaceHex3D(mesh->faceNodes+f*mesh->Nfp, mesh->cubInterp, zte, mesh->Nq, cubzte, mesh->cubNq);

      interpolateFaceHex3D(mesh->faceNodes+f*mesh->Nfp, mesh->cubInterp, mesh->x+e*mesh->Np, mesh->Nq, cubxe, mesh->cubNq);
      interpolateFaceHex3D(mesh->faceNodes+f*mesh->Nfp, mesh->cubInterp, mesh->y+e*mesh->Np, mesh->Nq, cubye, mesh->cubNq);
      interpolateFaceHex3D(mesh->faceNodes+f*mesh->Nfp, mesh->cubInterp, mesh->z+e*mesh->Np, mesh->Nq, cubze, mesh->cubNq);
      
      //geometric data for quadrature
      for(int i=0;i<mesh->cubNfp;++i){  // for each quadrature node on face

	dfloat xr = cubxre[i], xs = cubxse[i], xt = cubxte[i];
	dfloat yr = cubyre[i], ys = cubyse[i], yt = cubyte[i];
	dfloat zr = cubzre[i], zs = cubzse[i], zt = cubzte[i];

        /* determinant of Jacobian matrix */
        dfloat J = xr*(ys*zt-zs*yt) - yr*(xs*zt-zs*xt) + zr*(xs*yt-ys*xt);
        
        dfloat rx =  (ys*zt - zs*yt)/J, ry = -(xs*zt - zs*xt)/J, rz =  (xs*yt - ys*xt)/J;
        dfloat sx = -(yr*zt - zr*yt)/J, sy =  (xr*zt - zr*xt)/J, sz = -(xr*yt - yr*xt)/J;
        dfloat tx =  (yr*zs - zr*ys)/J, ty = -(xr*zs - zr*xs)/J, tz =  (xr*ys - yr*xs)/J;
        
        /* face f normal and length */
        dfloat nx, ny, nz, d;
        switch(f){
        case 0: nx = -tx; ny = -ty; nz = -tz; break;
        case 1: nx = -sx; ny = -sy; nz = -sz; break;
        case 2: nx = +rx; ny = +ry; nz = +rz; break;
        case 3: nx = +sx; ny = +sy; nz = +sz; break;
        case 4: nx = -rx; ny = -ry; nz = -rz; break;
        case 5: nx = +tx; ny = +ty; nz = +tz; break;
        }

        dfloat sJ = sqrt(nx*nx+ny*ny+nz*nz);
        nx /= sJ; ny /= sJ; nz /= sJ;
        sJ *= J;
        

        /* output index */
        dlong base = mesh->Nsgeo*(mesh->Nfaces*mesh->cubNfp*e + mesh->cubNfp*f + i);

        /* store normal, surface Jacobian, and reciprocal of volume Jacobian */
        mesh->cubsgeo[base+NXID] = nx;
        mesh->cubsgeo[base+NYID] = ny;
        mesh->cubsgeo[base+NZID] = nz;
        mesh->cubsgeo[base+SJID] = sJ;
        mesh->cubsgeo[base+IJID] = 1./J;

        mesh->cubsgeo[base+WIJID] = 1./(J*mesh->cubw[0]);
        mesh->cubsgeo[base+WSJID] = sJ*mesh->cubw[i%mesh->cubNq]*mesh->cubw[i/mesh->cubNq];

        mesh->cubsgeo[base+SURXID] = cubxe[i];
	mesh->cubsgeo[base+SURYID] = cubye[i];
	mesh->cubsgeo[base+SURZID] = cubze[i];
	
      }
    }
  }

  for(dlong e=0;e<mesh->Nelements;++e){ /* for each non-halo element */
    for(int n=0;n<mesh->Nfp*mesh->Nfaces;++n){
      dlong baseM = e*mesh->Nfp*mesh->Nfaces + n;
      dlong baseP = mesh->mapP[baseM];
      // rescaling - missing factor of 2 ? (only impacts penalty and thus stiffness)
      dfloat hinvM = mesh->sgeo[baseM*mesh->Nsgeo + SJID]*mesh->sgeo[baseM*mesh->Nsgeo + IJID];
      dfloat hinvP = mesh->sgeo[baseP*mesh->Nsgeo + SJID]*mesh->sgeo[baseP*mesh->Nsgeo + IJID];
      mesh->sgeo[baseM*mesh->Nsgeo+IHID] = mymax(hinvM,hinvP);
      mesh->sgeo[baseP*mesh->Nsgeo+IHID] = mymax(hinvM,hinvP);
    }
  }


  free(xre); free(xse); free(xte);
  free(yre); free(yse); free(yte);
  free(zre); free(zse); free(zte);

  free(cubxre); free(cubxse); free(cubxte);
  free(cubyre); free(cubyse); free(cubyte);
  free(cubzre); free(cubzse); free(cubzte);
  
}

int isHigher(const void *a, const void *b){

  hlong *pta = (hlong*) a;
  hlong *ptb = (hlong*) b;

  if(*pta < *ptb) return -1;
  if(*pta > *ptb) return +1;

  return 0;
}

int isLower(const void *a, const void *b){

  hlong *pta = (hlong*) a;
  hlong *ptb = (hlong*) b;

  if(*pta > *ptb) return -1;
  if(*pta < *ptb) return +1;

  return 0;
}

void mysort(hlong *data, int N, const char *order){

  if(strstr(order, "ascend")){
    qsort(data, N, sizeof(hlong), isHigher);
  }
  else{
    qsort(data, N, sizeof(hlong), isLower);
  }

}

void occaDeviceConfig(mesh_t *mesh, setupAide &options){

  // OCCA build stuff
  char deviceConfig[BUFSIZ];
  int rank, size;
  rank = mesh->rank;
  size = mesh->size;

  long int hostId = gethostid();

  long int* hostIds = (long int*) calloc(size,sizeof(long int));
  MPI_Allgather(&hostId,1,MPI_LONG,hostIds,1,MPI_LONG,mesh->comm);

  int device_id = 0;
  int totalDevices = 0;
  for (int r=0;r<rank;r++) {
    if (hostIds[r]==hostId) device_id++;
  }
  for (int r=0;r<size;r++) {
    if (hostIds[r]==hostId) totalDevices++;
  }

  if (size==1) options.getArgs("DEVICE NUMBER" ,device_id);

  printf("device_id = %d\n", device_id);
  
  //  device_id = device_id%2;

  occa::properties deviceProps;
  
  // read thread model/device/platform from options
  if(options.compareArgs("THREAD MODEL", "CUDA")){
    //    deviceProps["mode"] = "CUDA";
    //    deviceProps["device_id"] = 0; string(device_id);
    sprintf(deviceConfig, "mode: 'CUDA', device_id: %d", device_id);

#if USE_CUDA_NATIVE==1
    cudaStream_t cuStream;
    // Default: cuStream = 0
    cudaStreamCreate(&cuStream);// , CU_STREAM_DEFAULT);

    //  ---[ Get CUDA Info ]----
    int cuDeviceID = 0;
    CUdevice cuDevice;
    CUcontext cuContext;
    
    cuDeviceGet(&cuDevice, cuDeviceID);
    cuCtxGetCurrent(&cuContext);
    mesh->device = occa::cuda::wrapDevice(cuDevice, cuContext);
#endif
    
  }
  else if(options.compareArgs("THREAD MODEL", "HIP")){
    sprintf(deviceConfig, "mode: 'HIP', device_id: %d",device_id);
  }
  else if(options.compareArgs("THREAD MODEL", "OpenCL")){
    int plat;
    options.getArgs("PLATFORM NUMBER", plat);
    sprintf(deviceConfig, "mode: 'OpenCL', device_id: %d, platform_id: %d", device_id, plat);
  }
  else if(options.compareArgs("THREAD MODEL", "OpenMP")){
    sprintf(deviceConfig, "mode: 'OpenMP' ");
  }
  else{
    sprintf(deviceConfig, "mode: 'Serial' ");
  }

  //set number of omp threads to use
  int Ncores = sysconf(_SC_NPROCESSORS_ONLN);
  int Nthreads = Ncores/totalDevices;

  //  Nthreads = mymax(1,Nthreads/2);
  Nthreads = mymax(1,Nthreads/2);
  omp_set_num_threads(Nthreads);

  if (options.compareArgs("VERBOSE","TRUE"))
    printf("Rank %d: Ncores = %d, Nthreads = %d, device_id = %d \n", rank, Ncores, Nthreads, device_id);

  std::cout << deviceConfig << std::endl;
  
  //  mesh->device.setup( (std::string) deviceConfig); // deviceProps);
#if USE_CUDA_NATIVE!=1
  mesh->device.setup( (std::string)deviceConfig);
#endif

#ifdef USE_OCCA_MEM_BYTE_ALIGN 
  // change OCCA MEM BYTE ALIGNMENT
  occa::env::OCCA_MEM_BYTE_ALIGN = USE_OCCA_MEM_BYTE_ALIGN;
#endif

  #if 0
#if USE_MASTER_NOEL==1

  int foo;
  // check to see if the options specify to use precompiled binaries
  if(options.compareArgs("USE PRECOMPILED BINARIES", "TRUE")){
    mesh->device.UsePreCompiledKernels(1);
    occa::host().UsePreCompiledKernels(1);
  }
  else if(options.compareArgs("USE PRECOMPILED BINARIES", "NONROOT")){
    mesh->device.UsePreCompiledKernels(mesh->rank!=0);
    occa::host().UsePreCompiledKernels(mesh->rank!=0);
  }else{
    mesh->device.UsePreCompiledKernels(0);
    occa::host().UsePreCompiledKernels(0);
  }
    
#endif
#endif
  //  occa::initTimer(mesh->device);

}


void *occaHostMallocPinned(occa::device &device, size_t size, void *source, occa::memory &mem, occa::memory &h_mem){

#if 0

  mem = device.malloc(size, source);
  
  h_mem = device.mappedAlloc(size, source);

  void *ptr = h_mem.getMappedPointer();

#else
  //  mem =  device.malloc(size, source);
  //  void *ptr = device.malloc(size, "mapped: true").ptr();

  occa::properties props;
  props["mapped"] = true;
  
  h_mem =  device.malloc(size, props);
  
  void *ptr = h_mem.ptr();
  
#endif
  
  return ptr;

}

void mergeLists(size_t sz,
		int N1, char *v1,
		int N2, char *v2,
		char *v3,
		int (*compare)(const void *, const void *),
		void (*match)(void *, void *)){
    
  int n1 = 0, n2 = 0, n3 = 0;
    
  // merge two lists from v1 and v2
  for(n3=0;n3<N1+N2;++n3){
    if(n1<N1 && n2<N2){
      int c = compare(v1+n1*sz,v2+n2*sz);
      if(c==-1){
	memcpy(v3+n3*sz, v1+n1*sz, sz);
	++n1;
      }
      else{
	memcpy(v3+n3*sz, v2+n2*sz, sz);
	++n2;
      }
    }
    else if(n1<N1){
      memcpy(v3+n3*sz, v1+n1*sz, sz);
      ++n1;
    }
    else if(n2<N2){
      memcpy(v3+n3*sz, v2+n2*sz, sz);
      ++n2;
    }
  }
  
  // scan for matches
  for(n3=0;n3<N1+N2-1;++n3){
    if(!compare(v3+n3*sz,v3+(n3+1)*sz)){
      match(v3+n3*sz, v3+(n3+1)*sz);
    }
  }
    
  /* copy result back to v1, v2 */
  memcpy(v1, v3,       N1*sz);
  memcpy(v2, v3+sz*N1, N2*sz);
}

// assumes N is even and the same on all ranks
void parallelSort(int size, int rank, MPI_Comm comm,
		  int N, void *vv, size_t sz,
		  int (*compare)(const void *, const void *),
		  void (*match)(void *, void *)
		  ){

#if 0
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
#endif
  
  /* cast void * to char * */
  char *v = (char*) vv;

  /* sort faces by their vertex number pairs */
  qsort(v, N, sz, compare);
    
  /* now do progressive merges */
  int NA=N/2, NB = N/2, NC = N/2;
    
  MPI_Request recvA, recvC;
  MPI_Request sendA, sendC;
  MPI_Status status;
  int tag = 999;
    
  /* temporary buffer for incoming data */
  void *A = (void*) calloc(NA, sz);
  void *B = v;
  void *C = v+NB*sz;
  
  /* temporary space for merge sort */
  void *tmp = (void*) calloc(N, sz);

  /* max and min elements out of place hop one process at each step */
  for(int step=0;step<size-1;++step){
      
    /* send C, receive A */
    if(rank<size-1)
      MPI_Isend(C, NC*sz, MPI_CHAR,  rank+1, tag, comm, &sendC);
    if(rank>0)
      MPI_Irecv(A, NA*sz, MPI_CHAR,  rank-1, tag, comm, &recvA);
      
    if(rank<size-1)
      MPI_Wait(&sendC, &status);
    if(rank>0)
      MPI_Wait(&recvA, &status);
      
    /* merge sort A & B */
    if(rank>0) 
      mergeLists(sz, NA, (char*)A, NB, (char*)B, (char*)tmp, compare, match);
      
    /* send A, receive C */
    if(rank>0)
      MPI_Isend(A, NA*sz, MPI_CHAR, rank-1, tag, comm, &sendA);
    if(rank<size-1)
      MPI_Irecv(C, NC*sz, MPI_CHAR, rank+1, tag, comm, &recvC);
      
    if(rank>0)
      MPI_Wait(&sendA, &status);
    if(rank<size-1)
      MPI_Wait(&recvC, &status);
      
    /* merge sort B & C */
    mergeLists(sz, NB, (char*)B, NC, (char*)C, (char*)tmp, compare, match);
      
  }
    
  free(tmp);
  free(A);
}

void meshParallelGatherScatterSetup(mesh_t *mesh,
                                      dlong N,
                                      dlong *globalIds,
                                      MPI_Comm &comm,
                                      int verbose) { 

  int rank, size;
  MPI_Comm_rank(comm, &rank); 
  MPI_Comm_size(comm, &size); 

  mesh->ogs = ogsSetup(N, globalIds, comm, verbose, mesh->device);

  //use the gs to find what nodes are local to this rank
  int *minRank = (int *) calloc(N,sizeof(int));
  int *maxRank = (int *) calloc(N,sizeof(int));
  for (dlong i=0;i<N;i++) {
    minRank[i] = rank;
    maxRank[i] = rank;
  }

  ogsGatherScatter(minRank, ogsInt, ogsMin, mesh->ogs); //minRank[n] contains the smallest rank taking part in the gather of node n
  ogsGatherScatter(maxRank, ogsInt, ogsMax, mesh->ogs); //maxRank[n] contains the largest rank taking part in the gather of node n

  // count elements that contribute to global C0 gather-scatter
  dlong globalCount = 0;
  dlong localCount = 0;
  for(dlong e=0;e<mesh->Nelements;++e){
    int isHalo = 0;
    for(int n=0;n<mesh->Np;++n){
      dlong id = e*mesh->Np+n;
      if ((minRank[id]!=rank)||(maxRank[id]!=rank)) {
        isHalo = 1;
        break;
      }
    }
    globalCount += isHalo;
    localCount += 1-isHalo;
  }

  mesh->globalGatherElementList = (dlong*) calloc(globalCount, sizeof(dlong));
  mesh->localGatherElementList  = (dlong*) calloc(localCount, sizeof(dlong));

  globalCount = 0;
  localCount = 0;

  for(dlong e=0;e<mesh->Nelements;++e){
    int isHalo = 0;
    for(int n=0;n<mesh->Np;++n){
      dlong id = e*mesh->Np+n;
      if ((minRank[id]!=rank)||(maxRank[id]!=rank)) {
        isHalo = 1;
        break;
      }
    }
    if(isHalo){
      mesh->globalGatherElementList[globalCount++] = e;
    } else{
      mesh->localGatherElementList[localCount++] = e;
    }
  }
  //printf("local = %d, global = %d\n", localCount, globalCount);

  mesh->NglobalGatherElements = globalCount;
  mesh->NlocalGatherElements = localCount;

  if(globalCount)
    mesh->o_globalGatherElementList =
      mesh->device.malloc(globalCount*sizeof(dlong), mesh->globalGatherElementList);

  if(localCount)
    mesh->o_localGatherElementList =
      mesh->device.malloc(localCount*sizeof(dlong), mesh->localGatherElementList);
}


// uniquely label each node with a global index, used for gatherScatter
// - specialized with local numbering for on device GS
void meshLocalizedConnectNodes(mesh_t *mesh){

  int rank, size;
  rank = mesh->rank; 
  size = mesh->size; 

  dlong localNodeCount = mesh->Np*mesh->Nelements;
  dlong *allLocalNodeCounts = (dlong*) calloc(size, sizeof(dlong));

  MPI_Allgather(&localNodeCount,    1, MPI_DLONG,
                allLocalNodeCounts, 1, MPI_DLONG,
                mesh->comm);
  
  hlong gatherNodeStart = 0;
  for(int r=0;r<rank;++r)
    gatherNodeStart += allLocalNodeCounts[r];
  
  free(allLocalNodeCounts);

  // form continuous node numbering (local=>virtual gather)
  parallelNode_t *localNodes =
    (parallelNode_t*) calloc((mesh->totalHaloPairs+mesh->Nelements)*mesh->Np,
                             sizeof(parallelNode_t));

  // use local numbering
  for(dlong e=0;e<mesh->Nelements;++e){
    for(int n=0;n<mesh->Np;++n){
      dlong id = e*mesh->Np+n;

      localNodes[id].baseRank = rank;
      localNodes[id].baseId = 1 + id + gatherNodeStart;

    }
  }

  dlong localChange = 0, gatherChange = 1;

  parallelNode_t *sendBuffer =
    (parallelNode_t*) calloc(mesh->totalHaloPairs*mesh->Np, sizeof(parallelNode_t));

  // keep comparing numbers on positive and negative traces until convergence
  while(gatherChange>0){

    // reset change counter
    localChange = 0;

    // send halo data and recv into extension of buffer
    meshHaloExchange(mesh, mesh->Np*sizeof(parallelNode_t),
                     localNodes, sendBuffer, localNodes+localNodeCount);

    // compare trace nodes
    for(dlong e=0;e<mesh->Nelements;++e){
      for(int n=0;n<mesh->Nfp*mesh->Nfaces;++n){
        dlong id  = e*mesh->Nfp*mesh->Nfaces + n;
        dlong idM = mesh->vmapM[id];
        dlong idP = mesh->vmapP[id];
        hlong gidM = localNodes[idM].baseId;
        hlong gidP = localNodes[idP].baseId;

        int baseRankM = localNodes[idM].baseRank;
        int baseRankP = localNodes[idP].baseRank;
        
        if(gidM<gidP || (gidP==gidM && baseRankM<baseRankP)){
          ++localChange;
          localNodes[idP].baseRank    = localNodes[idM].baseRank;
          localNodes[idP].baseId      = localNodes[idM].baseId;
        }
        
        if(gidP<gidM || (gidP==gidM && baseRankP<baseRankM)){
          ++localChange;
          localNodes[idM].baseRank    = localNodes[idP].baseRank;
          localNodes[idM].baseId      = localNodes[idP].baseId;
        }
      }
    }

    // sum up changes
    MPI_Allreduce(&localChange, &gatherChange, 1, MPI_DLONG, MPI_SUM, mesh->comm);
  }


  // now renumber the nodes that are owned locally and reconnect
  hlong *localizedIds = (hlong*) calloc(localNodeCount, sizeof(hlong));
  for(dlong e=0;e<mesh->Nelements;++e){
    for(int n=0;n<mesh->Np;++n){
      dlong id = e*mesh->Np+n;
      if(localNodes[id].baseRank == rank) { // locally owned node
	dlong gid = localNodes[id].baseId-1-gatherNodeStart; 
	localizedIds[gid] = 1;
      }
    }
  }

  hlong cnt = gatherNodeStart;
  for(dlong e=0;e<mesh->Nelements;++e){
    for(int n=0;n<mesh->Np;++n){
      dlong id = e*mesh->Np+n;
      if(localizedIds[id]){
	++cnt;
	localizedIds[id] = cnt;
      }
    }
  }

  mesh->Nlocalized = cnt-gatherNodeStart;
  mesh->startLocalized = gatherNodeStart;
  
  for(dlong e=0;e<mesh->Nelements;++e){
    for(int n=0;n<mesh->Np;++n){
      dlong id = e*mesh->Np+n;
      if(localNodes[id].baseRank == rank) { // locally owned node
	localNodes[id].localizedId = localizedIds[id];
      }
      else{
	localNodes[id].localizedId = 0;
      }
    }
  }

  gatherChange = 1;
  // keep comparing numbers on positive and negative traces until convergence
  while(gatherChange>0){
    
    // reset change counter
    localChange = 0;
    
    // send halo data and recv into extension of buffer
    meshHaloExchange(mesh, mesh->Np*sizeof(parallelNode_t),
                     localNodes, sendBuffer, localNodes+localNodeCount);

    // look for nodes that have not been numbered yet
    for(dlong e=0;e<mesh->Nelements;++e){
      for(int n=0;n<mesh->Nfp*mesh->Nfaces;++n){
        dlong id  = e*mesh->Nfp*mesh->Nfaces + n;
        dlong idM = mesh->vmapM[id];
        dlong idP = mesh->vmapP[id];

	int localizedIdM = localNodes[idM].localizedId;
	int localizedIdP = localNodes[idP].localizedId;
	
        if(localizedIdM==0){ // not numbered yet
          ++localChange;
	  localNodes[idM].localizedId = localNodes[idP].localizedId;
        }

        if(localizedIdP==0){ // not numbered yet
          ++localChange;
	  localNodes[idP].localizedId = localNodes[idM].localizedId;
        }
      }
    }

    // sum up changes
    MPI_Allreduce(&localChange, &gatherChange, 1, MPI_DLONG, MPI_SUM, mesh->comm);
  }

  //make a locally-ordered version
  mesh->localizedIds = (hlong*) calloc(localNodeCount, sizeof(hlong));
  for(dlong id=0;id<localNodeCount;++id){
    mesh->localizedIds[id] = localNodes[id].localizedId;
  }
  
  printf("Local nodes=%d, Localized nodes=%d\n",
	 localNodeCount, mesh->Nlocalized);
  for(dlong e=0;e<mesh->Nelements;++e){
    for(int n=0;n<mesh->Np;++n){
      if(!mesh->localizedIds[e*mesh->Np+n]){
	printf("!! %d !!", mesh->localizedIds[e*mesh->Np+n]);
      }
    }
  }
  
  free(localNodes);
  free(sendBuffer);
  free(localizedIds);
  
}
