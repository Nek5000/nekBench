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

#include "BP.hpp"

int BPPCG(BP_t* BP, dfloat lambda, dfloat mu,
	  occa::memory &o_r, occa::memory &o_x, 
	  const dfloat tol, const int MAXIT,
	  double *opElapsed){
  
  mesh_t *mesh = BP->mesh;
  setupAide options = BP->options;

  int fixedIterationCountFlag = 0;
  int flexible = options.compareArgs("KRYLOV SOLVER", "FLEXIBLE");
  int verbose = options.compareArgs("VERBOSE", "TRUE");
  
  if(options.compareArgs("FIXED ITERATION COUNT", "TRUE"))
    fixedIterationCountFlag = 1;
  
  // register scalars
  dfloat rdotz1 = 0;
  dfloat rdotz2 = 0;

  // now initialized
  dfloat alpha = 0, beta = 0;

  dlong Ndof = mesh->Nelements*mesh->Np*BP->Nfields;
  
  /*aux variables */
  occa::memory o_p   = BP->o_solveWorkspace + 0*Ndof*sizeof(dfloat);
  occa::memory o_z   = BP->o_solveWorkspace + 1*Ndof*sizeof(dfloat);
  occa::memory o_Ap  = BP->o_solveWorkspace + 2*Ndof*sizeof(dfloat);
  occa::memory o_Ax  = BP->o_solveWorkspace + 3*Ndof*sizeof(dfloat);

  rdotz1 = 1;

  dfloat rdotr0;

  // compute A*x
  dfloat pAp = AxOperator(BP, lambda, mu, o_x, o_Ax, dfloatString); 
  
  // subtract r = b - A*x
  BPScaledAdd(BP, -1.f, o_Ax, 1.f, o_r);

  rdotr0 = BPWeightedNorm2(BP, BP->o_invDegree, o_r);

  dfloat TOL =  mymax(tol*tol*rdotr0,tol*tol);

  double elapsedPreco = 0;
  double elapsedPupdate = 0;
  double elapsedAx = 0;
  double elapsedDot = 0;
  double elapsedUpdate = 0;
  double elapsedOp = 0;
  double elapsedOverall = 0;

  int iter;

//  startOverall = BP->mesh->device.tagStream();
  
  for(iter=1;iter<=MAXIT;++iter){

    // z = Precon^{-1} r 
//    startPreco = BP->mesh->device.tagStream();
    BPPreconditioner(BP, lambda, o_r, o_z);
//    endPreco = BP->mesh->device.tagStream();
     
    rdotz2 = rdotz1;

    // r.z
//    startDot = BP->mesh->device.tagStream();
    rdotz1 = BPWeightedInnerProduct(BP, BP->o_invDegree, o_r, o_z); 
    
    if(flexible){
      dfloat zdotAp = BPWeightedInnerProduct(BP, BP->o_invDegree, o_z, o_Ap);  
      beta = -alpha*zdotAp/rdotz2;
    }
    else{
      beta = (iter==1) ? 0:rdotz1/rdotz2;
    }  
//    endDot = BP->mesh->device.tagStream();
  
    // p = z + beta*p
//    startPupdate = BP->mesh->device.tagStream();
    BPScaledAdd(BP, 1.f, o_z, beta, o_p);
//    endPupdate = BP->mesh->device.tagStream();
	
    // Ap and p.Ap
//    startOp = BP->mesh->device.tagStream();
    pAp = AxOperator(BP, lambda, mu, o_p, o_Ap, dfloatString);
//    endOp = BP->mesh->device.tagStream();
    
    // alpha = r.z/p.Ap
    alpha = rdotz1/pAp;

    //  x <= x + alpha*p
    //  r <= r - alpha*A*p
    //  dot(r,r)

//    startUpdate = BP->mesh->device.tagStream();
    dfloat rdotr = BPUpdatePCG(BP, o_p, o_Ap, alpha, o_x, o_r);
//    endUpdate = BP->mesh->device.tagStream();

    if (verbose&&(mesh->rank==0)) {

      if(rdotr<0)
	printf("WARNING CG: rdotr = %17.15lf\n", rdotr);
      
      printf("CG: it %d r norm %12.12le alpha = %le \n", iter, sqrt(rdotr), alpha);    
    }
    
    if(rdotr<=TOL && !fixedIterationCountFlag) break;
  }

  return iter;
}

void BPZeroMean(BP_t *BP, occa::memory &o_q){

  dfloat qmeanLocal;
  dfloat qmeanGlobal;
  
  dlong Nblock = BP->Nblock;
  dfloat *tmp = BP->tmp;
  mesh_t *mesh = BP->mesh;

  occa::memory &o_tmp = BP->o_tmp;
  
  // this is a C0 thing [ assume GS previously applied to o_q ]
  BP->innerProductKernel(mesh->Nelements*mesh->Np, BP->o_invDegree, o_q, o_tmp);
  
  o_tmp.copyTo(tmp);

  // finish reduction
  qmeanLocal = 0;
  for(dlong n=0;n<Nblock;++n)
    qmeanLocal += tmp[n];

  // globalize reduction
  MPI_Allreduce(&qmeanLocal, &qmeanGlobal, 1, MPI_DFLOAT, MPI_SUM, mesh->comm);

  // normalize
#if USE_WEIGHTED==1
  qmeanGlobal *= BP->nullProjectWeightGlobal;
#else
  qmeanGlobal /= ((dfloat) BP->NelementsGlobal*(dfloat)mesh->Np);
#endif
  
  // q[n] = q[n] - qmeanGlobal
  mesh->addScalarKernel(mesh->Nelements*mesh->Np, -qmeanGlobal, o_q);
}

dfloat BPUpdatePCG(BP_t *BP,
			 occa::memory &o_p, occa::memory &o_Ap, const dfloat alpha,
			 occa::memory &o_x, occa::memory &o_r){

  setupAide &options = BP->options;
  
  int fixedIterationCountFlag = 0;
  int flexible = options.compareArgs("KRYLOV SOLVER", "FLEXIBLE");
  int verbose = options.compareArgs("VERBOSE", "TRUE");
  int serial = options.compareArgs("THREAD MODEL", "SERIAL"); 
 
  mesh_t *mesh = BP->mesh;
  
  // x <= x + alpha*p
  // r <= r - alpha*A*p
  // dot(r,r)

  const dlong offset = mesh->Np*(mesh->Nelements+mesh->totalHaloPairs);

  if(BP->Nfields==1)
    BP->updatePCGKernel(mesh->Nelements, mesh->Np, BP->NblocksUpdatePCG,
			BP->o_invDegree, o_p, o_Ap, alpha, o_x, o_r, BP->o_tmpNormr);
  else
    BP->updateMultiplePCGKernel(mesh->Nelements, mesh->Np, offset, BP->NblocksUpdatePCG,
				BP->o_invDegree, o_p, o_Ap, alpha, o_x, o_r, BP->o_tmpNormr);

  BP->o_tmpNormr.copyTo(BP->tmpNormr);

  dfloat rdotr1 = 0; 
  if(serial) {
    rdotr1 = BP->tmpNormr[0];
  }
  else {
    for(int n=0;n<BP->NblocksUpdatePCG;++n){
      rdotr1 += BP->tmpNormr[n];
    }
  } 

  dfloat globalrdotr1;
  MPI_Allreduce(&rdotr1, &globalrdotr1, 1, MPI_DFLOAT, MPI_SUM, mesh->comm);
  rdotr1 = globalrdotr1;

  return rdotr1;
}

#include "../axhelm/kernelHelper.cpp"

void BPPreconditioner(BP_t *BP, dfloat lambda, occa::memory &o_r, occa::memory &o_z){
  mesh_t *mesh = BP->mesh;
  setupAide &options = BP->options;

  if(options.compareArgs("PRECONDITIONER", "JACOBI")) {
    //dlong Ntotal = mesh->Np*mesh->Nelements; 
    //BP->dotMultiplyKernel(Ntotal, o_r, BP->o_invDiagA, o_z);
    dlong Ndof = mesh->Nelements*mesh->Np*BP->Nfields;
    BP->vecCopyKernel(Ndof, o_r, o_z);
  }
}

dfloat AxOperator(BP_t *BP, dfloat lambda, dfloat mu, occa::memory &o_q, occa::memory &o_Aq,
		  const char *precision){

  mesh_t *mesh = BP->mesh;
  setupAide &options = BP->options;
  ogs_t *ogs = BP->ogs;

  occa::kernel &kernel = BP->BPKernel[BP->BPid];
  
  const dlong offset = mesh->Np*(mesh->Nelements+mesh->totalHaloPairs);

//  if(start)
//    *start = BP->mesh->device.tagStream();

    kernel(mesh->Nelements, offset, mesh->o_ggeo, mesh->o_D, lambda, o_q, o_Aq);

//  if(end)
//    *end = BP->mesh->device.tagStream();
  
  if(BP->Nfields==1)
    ogsGatherScatter(o_Aq, ogsDfloat, ogsAdd, ogs);
  else
    ogsGatherScatterMany(o_Aq, BP->Nfields, offset, ogsDfloat, ogsAdd, ogs);
  
  dfloat pAp = 0;
  pAp = BPWeightedInnerProduct(BP, BP->o_invDegree, o_q, o_Aq);
  
  return pAp;
}

dfloat BPWeightedNorm2(BP_t *BP, occa::memory &o_w, occa::memory &o_a){

  setupAide &options = BP->options;

  mesh_t *mesh = BP->mesh;
  dfloat *tmp = BP->tmp;
  dlong Nblock = BP->Nblock;
  dlong Nblock2 = BP->Nblock2;
  dlong Ntotal = mesh->Nelements*mesh->Np;

  occa::memory &o_tmp = BP->o_tmp;
  occa::memory &o_tmp2 = BP->o_tmp2;

  if(BP->Nfields==1)
    BP->weightedNorm2Kernel(Ntotal, o_w, o_a, o_tmp);
  else
    BP->weightedMultipleNorm2Kernel(Ntotal, Ntotal, o_w, o_a, o_tmp);

  /* add a second sweep if Nblock>Ncutoff */
  dlong Ncutoff = 100;
  dlong Nfinal;
  if(Nblock>Ncutoff){

    mesh->sumKernel(Nblock, o_tmp, o_tmp2);

    o_tmp2.copyTo(tmp);

    Nfinal = Nblock2;
	
  }
  else{
    o_tmp.copyTo(tmp);
    
    Nfinal = Nblock;

  }    

  dfloat wa2 = 0;
  for(dlong n=0;n<Nfinal;++n){
    wa2 += tmp[n];
  }

  dfloat globalwa2 = 0;
  MPI_Allreduce(&wa2, &globalwa2, 1, MPI_DFLOAT, MPI_SUM, mesh->comm);

  return globalwa2;
}

dfloat BPWeightedInnerProduct(BP_t *BP, occa::memory &o_w, occa::memory &o_a, occa::memory &o_b){

  setupAide &options = BP->options;

  mesh_t *mesh = BP->mesh;
  dfloat *tmp = BP->tmp;
  dlong Nblock = BP->Nblock;
  dlong Nblock2 = BP->Nblock2;
  dlong Ntotal = mesh->Nelements*mesh->Np;
  const dlong offset = Ntotal;

  occa::memory &o_tmp = BP->o_tmp;
  occa::memory &o_tmp2 = BP->o_tmp2;


  if(BP->Nfields == 1)
    BP->weightedInnerProduct2Kernel(Ntotal, o_w, o_a, o_b, o_tmp);
  else
    BP->weightedMultipleInnerProduct2Kernel(Ntotal, offset, o_w, o_a, o_b, o_tmp);

  int serial = options.compareArgs("THREAD MODEL", "SERIAL");

  dfloat wab = 0;
  if(serial){
    o_tmp.copyTo(tmp);
    wab = tmp[0];
  } else {
    /* add a second sweep if Nblock>Ncutoff */
    dlong Ncutoff = 100;
    dlong Nfinal;
    if(Nblock>Ncutoff){
  
      mesh->sumKernel(Nblock, o_tmp, o_tmp2);
  
      o_tmp2.copyTo(tmp);
  
      Nfinal = Nblock2;
          
    }
    else{
      o_tmp.copyTo(tmp);
      
      Nfinal = Nblock;
  
    }    
  
    for(dlong n=0;n<Nfinal;++n){
      wab += tmp[n];
    }
  }

  dfloat globalwab = 0;
  MPI_Allreduce(&wab, &globalwab, 1, MPI_DFLOAT, MPI_SUM, mesh->comm);

  return globalwab;
}


// b[n] = alpha*a[n] + beta*b[n] n\in [0,Ntotal)
void BPScaledAdd(BP_t *BP, dfloat alpha, occa::memory &o_a, dfloat beta, occa::memory &o_b){

  mesh_t *mesh = BP->mesh;
  
  dlong Ntotal = mesh->Nelements*mesh->Np*BP->Nfields;
  BP->scaledAddKernel(Ntotal, alpha, o_a, beta, o_b);
}
