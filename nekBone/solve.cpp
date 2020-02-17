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
#include <sys/time.h>

int BPSolve(BP_t *BP, dfloat lambda, dfloat mu, dfloat tol, occa::memory &o_r, occa::memory &o_x, double *opElapsed){
  mesh_t *mesh = BP->mesh;
  setupAide options = BP->options;

  int Niter = 0;
  int maxIter = 1000; 

  options.getArgs("MAXIMUM ITERATIONS", maxIter);
  options.getArgs("SOLVER TOLERANCE", tol);

  // zero mean of RHS
  if(BP->allNeumann) 
    BPZeroMean(BP, o_r);
  
  // solve with preconditioned conjugate gradient (diag precon)
  if(options.compareArgs("KRYLOV SOLVER", "PCG"))
    Niter = BPPCG(BP, lambda, mu, o_r, o_x, tol, maxIter, opElapsed);

  // zero mean of RHS
  if(BP->allNeumann) 
    BPZeroMean(BP, o_x);
  
  return Niter;
}

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
  dlong Nbytes = Ndof*sizeof(dfloat);
  
  /*aux variables */
  occa::memory o_p  = BP->o_solveWorkspace + 0*Ndof*sizeof(dfloat);
  occa::memory o_z  = BP->o_solveWorkspace + 1*Ndof*sizeof(dfloat);
  occa::memory o_Ap = BP->o_solveWorkspace + 2*Ndof*sizeof(dfloat);
  occa::memory o_Ax = BP->o_solveWorkspace + 3*Ndof*sizeof(dfloat);
  occa::memory o_res = BP->o_solveWorkspace + 4*Ndof*sizeof(dfloat);

  occa::streamTag starts[MAXIT+1];
  occa::streamTag ends[MAXIT+1];

  rdotz1 = 1;

  dfloat rdotr0;

  // compute A*x
  dfloat pAp = AxOperator(BP, lambda, mu, o_x, o_Ax, dfloatString, starts, ends); 
  
  // subtract r = b - A*x
  BPScaledAdd(BP, -1.f, o_Ax, 1.f, o_r);

  rdotr0 = BPWeightedNorm2(BP, BP->o_invDegree, o_r);

  dfloat TOL =  mymax(tol*tol*rdotr0,tol*tol);

  double elapsedCopy = 0;
  double elapsedPupdate = 0;
  double elapsedAx = 0;
  double elapsedDot = 0;
  double elapsedUpdate = 0;
  double elapsedOp = 0;
  double elapsedOverall = 0;


  occa::streamTag startCopy;
  occa::streamTag endCopy;

  occa::streamTag startPupdate;
  occa::streamTag endPupdate;

  occa::streamTag startUpdate;
  occa::streamTag endUpdate;

  occa::streamTag startDot;
  occa::streamTag endDot;

  occa::streamTag startOp;
  occa::streamTag endOp;

  occa::streamTag startOverall;
  occa::streamTag endOverall;
  
  int iter;

//  startOverall = BP->mesh->device.tagStream();
  
  for(iter=1;iter<=MAXIT;++iter){

    // z = Precon^{-1} r [ just a copy for this example ]
//    startCopy = BP->mesh->device.tagStream();
    //    o_r.copyTo(o_z, Nbytes);
    BP->vecCopyKernel(Ndof, o_r, o_z);
//    endCopy = BP->mesh->device.tagStream();
    
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
    pAp = AxOperator(BP, lambda, mu, o_p, o_Ap, dfloatString, starts+iter, ends+iter);
//    endOp = BP->mesh->device.tagStream();
    
    // alpha = r.z/p.Ap
    alpha = rdotz1/pAp;

    //  x <= x + alpha*p
    //  r <= r - alpha*A*p
    //  dot(r,r)

//    startUpdate = BP->mesh->device.tagStream();
    dfloat rdotr = BPUpdatePCG(BP, o_p, o_Ap, alpha, o_x, o_r);
//    endUpdate = BP->mesh->device.tagStream();

/* 
    BP->mesh->device.finish();
    elapsedUpdate  += BP->mesh->device.timeBetween(startUpdate,  endUpdate);
    elapsedCopy    += BP->mesh->device.timeBetween(startCopy,    endCopy);
    elapsedPupdate += BP->mesh->device.timeBetween(startPupdate, endPupdate);    
    elapsedDot     += BP->mesh->device.timeBetween(startDot,     endDot);
    elapsedOp      += BP->mesh->device.timeBetween(startOp,      endOp);
*/

    if (verbose&&(mesh->rank==0)) {

      if(rdotr<0)
	printf("WARNING CG: rdotr = %17.15lf\n", rdotr);
      
      printf("CG: it %d r norm %12.12le alpha = %le \n", iter, sqrt(rdotr), alpha);    
    }
    
    if(rdotr<=TOL && !fixedIterationCountFlag) break;
  }

//  endOverall = BP->mesh->device.tagStream();
  
  BP->mesh->device.finish();
  
  //elapsedOverall += BP->mesh->device.timeBetween(startOverall,endOverall);
  
/*
  printf("Elapsed: overall: %g, PCG Update %g, Pupdate: %g, Copy: %g, dot: %g, op: %g\n",
	 elapsedOverall, elapsedUpdate, elapsedPupdate, elapsedCopy, elapsedDot, elapsedOp);
*/

  double gbytesPCG = 7.*mesh->Np*mesh->Nelements*(sizeof(dfloat)/1.e9);
  double gbytesCopy = Nbytes/1.e9;
  double gbytesOp = (7+2*BP->Nfields)*mesh->Np*mesh->Nelements*(sizeof(dfloat)/1.e9);
  double gbytesDot = (2*BP->Nfields+1)*mesh->Np*mesh->Nelements*(sizeof(dfloat)/1.e9);
  double gbytesPupdate =  3*mesh->Np*mesh->Nelements*(sizeof(dfloat)/1.e9);

  int combineDot = 0;
  combineDot = 0; //options.compareArgs("COMBINE DOT PRODUCT", "TRUE");

  if(!combineDot)
    gbytesOp += 3*mesh->Np*mesh->Nelements*(sizeof(dfloat)/1.e9);

#if 0
  if(verbose)
  printf("Bandwidth (GB/s): PCG update: %g, Copy: %g, Op: %g, Dot: %g, Pupdate: %g\n",
	 gbytesPCG*iter/elapsedUpdate,
	 gbytesCopy*iter/elapsedCopy,
	 gbytesOp*iter/elapsedOp,
	 gbytesDot*iter/elapsedDot,
	 gbytesPupdate*iter/elapsedPupdate);
#endif
  
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

dfloat AxOperator(BP_t *BP, dfloat lambda, dfloat mu, occa::memory &o_q, occa::memory &o_Aq,
		  const char *precision, occa::streamTag *start, occa::streamTag *end){

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
  
#if 1
  if(BP->Nfields==1)
    ogsGatherScatter(o_Aq, ogsDfloat, ogsAdd, ogs);
  else
    ogsGatherScatterMany(o_Aq, BP->Nfields, offset, ogsDfloat, ogsAdd, ogs);
#endif
  
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
