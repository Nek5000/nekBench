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
void updateJacobi(BP_t *BP, occa::memory &o_lambda, occa::memory &o_invDiagA);


int BPPCG(BP_t* BP, occa::memory &o_lambda,
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
 
  int iter;
 
  // register scalars
  dfloat rdotz1 = 1;
  dfloat rdotz2 = 0;
  dfloat rdotr0;

  // now initialized
  dfloat alpha = 0, beta = 0;

  /*aux variables */
  occa::memory &o_p   = BP->o_solveWorkspace[0];
  occa::memory &o_z   = BP->o_solveWorkspace[1];
  occa::memory &o_Ap  = BP->o_solveWorkspace[2];
  occa::memory &o_Ax  = BP->o_solveWorkspace[3];

  // compute A*x
  dfloat pAp = AxOperator(BP, o_lambda, o_x, o_Ax, dfloatString); 
  
  // subtract r = b - A*x
  BPScaledAdd(BP, -1.f, o_Ax, 1.f, o_r);

  rdotr0 = BPWeightedNorm2(BP, BP->o_invDegree, o_r);

  const dfloat TOL =  mymax(tol*tol*rdotr0,tol*tol);

  if(BP->profiling) timer::tic("preco");
  if(options.compareArgs("PRECONDITIONER", "JACOBI")) updateJacobi(BP, o_lambda, BP->o_invDiagA); 
  if(BP->profiling) timer::toc("preco");

  for(iter=1;iter<=MAXIT;++iter){

    // z = Precon^{-1} r 
    BPPreconditioner(BP, o_lambda, o_r, o_z);
 
    rdotz2 = rdotz1;

    // r.z
    rdotz1 = BPWeightedInnerProduct(BP, BP->o_invDegree, o_r, o_z); 
 
    if(flexible){
      dfloat zdotAp = BPWeightedInnerProduct(BP, BP->o_invDegree, o_z, o_Ap);  
      beta = -alpha*zdotAp/rdotz2;
    }
    else{
      beta = (iter==1) ? 0:rdotz1/rdotz2;
    }  
 
    // p = z + beta*p
    BPScaledAdd(BP, 1.f, o_z, beta, o_p);
	
    // Ap and p.Ap
    pAp = AxOperator(BP, o_lambda, o_p, o_Ap, dfloatString);
    
    // alpha = r.z/p.Ap
    alpha = rdotz1/pAp;

    //  x <= x + alpha*p
    //  r <= r - alpha*A*p
    //  dot(r,r)
    dfloat rdotr = BPUpdatePCG(BP, o_p, o_Ap, alpha, o_x, o_r);

    if (verbose&&(mesh->rank==0)) {

      if(rdotr<0)
	printf("WARNING CG: rdotr = %17.15lf\n", rdotr);
      
      printf("CG: it %d r norm %12.12le alpha = %le \n", iter, sqrt(rdotr), alpha);    
    }
    
    if(rdotr<=TOL && !fixedIterationCountFlag) break;
  }

  return iter-1;
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

  if(BP->profiling) timer::tic("updatePCG");

  setupAide &options = BP->options;
  
  int fixedIterationCountFlag = 0;
  int flexible = options.compareArgs("KRYLOV SOLVER", "FLEXIBLE");
  int verbose = options.compareArgs("VERBOSE", "TRUE");
  int serial = options.compareArgs("THREAD MODEL", "SERIAL"); 
  int omp = options.compareArgs("THREAD MODEL", "OPENMP"); 
 
  mesh_t *mesh = BP->mesh;
  
  // x <= x + alpha*p
  // r <= r - alpha*A*p
  // dot(r,r)

  const dlong Nlocal = mesh->Nelements*mesh->Np;

  if(BP->Nfields==1)
    BP->updatePCGKernel(Nlocal, BP->NblocksUpdatePCG,
			BP->o_invDegree, o_p, o_Ap, alpha, o_x, o_r, BP->o_tmpNormr);
  else
    BP->updateMultiplePCGKernel(Nlocal, BP->fieldOffset, BP->NblocksUpdatePCG,
				BP->o_invDegree, o_p, o_Ap, alpha, o_x, o_r, BP->o_tmpNormr);

  BP->o_tmpNormr.copyTo(BP->tmpNormr);

  dfloat rdotr1 = 0; 
  if(serial || omp) {
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

  if(BP->profiling) timer::toc("updatePCG");
  return rdotr1;
}

#include "../axhelm/kernelHelper.cpp"

void updateJacobi(BP_t *BP, occa::memory &o_lambda, occa::memory &o_invDiagA){

  mesh_t *mesh = BP->mesh;
  const dlong Nlocal = mesh->Np*mesh->Nelements;

  BP->updateJacobiKernel(mesh->Nelements,
                         BP->fieldOffset,
                         BP->o_mapB,
                         mesh->o_ggeo,
                         mesh->o_D,
                         o_lambda,
                         o_invDiagA);

  const dfloat one = 1.0;
  ogsGatherScatter(o_invDiagA, ogsDfloat, ogsAdd, BP->ogs);
  BP->vecInvKernel(Nlocal, o_invDiagA);
}

void BPPreconditioner(BP_t *BP, occa::memory &o_lambda, occa::memory &o_r, occa::memory &o_z){
  mesh_t *mesh = BP->mesh;
  setupAide &options = BP->options;

  if(BP->profiling) timer::tic("preco");
  if(options.compareArgs("PRECONDITIONER", "JACOBI")) {
    const dlong Nlocal = mesh->Np*mesh->Nelements; 
    BP->dotMultiplyKernel(Nlocal, o_r, BP->o_invDiagA, o_z);
  } else {
    dlong Ndof = mesh->Nelements*mesh->Np*BP->Nfields;
    BP->vecCopyKernel(Ndof, o_r, o_z);
  }
  if(BP->profiling) timer::toc("preco");
}

dfloat AxOperator(BP_t *BP, occa::memory &o_lambda, occa::memory &o_q, occa::memory &o_Aq,
		  const char *precision){

  mesh_t *mesh = BP->mesh;
  setupAide &options = BP->options;
  ogs_t *ogs = BP->ogs;

  occa::kernel &kernel = BP->BPKernel[0];
  
  const dlong fieldOffset = mesh->Np*(mesh->Nelements+mesh->totalHaloPairs);

  if(BP->profiling) timer::tic("Ax");
  kernel(mesh->Nelements, fieldOffset, mesh->o_ggeo, mesh->o_D, o_lambda, o_q, o_Aq);
  if(BP->profiling) timer::toc("Ax");
  
  if(BP->profiling) timer::tic("gs");
  ogsGatherScatterMany(o_Aq, BP->Nfields, fieldOffset, ogsDfloat, ogsAdd, ogs);
  if(BP->profiling) timer::toc("gs");

  dfloat pAp = BPWeightedInnerProduct(BP, BP->o_invDegree, o_q, o_Aq);
 
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
  dlong Ncutoff = 1000;
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

  mesh_t *mesh = BP->mesh;
  mesh->device.finish();
  if(BP->profiling) timer::tic("dot");
  setupAide &options = BP->options;

  dfloat *tmp = BP->tmp;
  dlong Nblock = BP->Nblock;
  dlong Nblock2 = BP->Nblock2;
  dlong Ntotal = mesh->Nelements*mesh->Np;

  occa::memory &o_tmp = BP->o_tmp;
  occa::memory &o_tmp2 = BP->o_tmp2;

  if(BP->Nfields == 1)
    BP->weightedInnerProduct2Kernel(Ntotal, o_w, o_a, o_b, o_tmp);
  else
    BP->weightedMultipleInnerProduct2Kernel(Ntotal, BP->fieldOffset, o_w, o_a, o_b, o_tmp);

  int serial = options.compareArgs("THREAD MODEL", "SERIAL");
  int omp = options.compareArgs("THREAD MODEL", "OPENMP");

  dfloat wab = 0;
  if(serial || omp){
    o_tmp.copyTo(tmp);
    wab = tmp[0];
  } else {
    /* add a second sweep if Nblock>Ncutoff */
    dlong Ncutoff = 1000;
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

  if(BP->profiling) timer::toc("dot");
  return globalwab;
}


// b[n] = alpha*a[n] + beta*b[n] n\in [0,Ntotal)
void BPScaledAdd(BP_t *BP, dfloat alpha, occa::memory &o_a, dfloat beta, occa::memory &o_b){

  mesh_t *mesh = BP->mesh;
  
  dlong Ntotal = mesh->Nelements*mesh->Np*BP->Nfields;
  BP->scaledAddKernel(Ntotal, alpha, o_a, beta, o_b);
}
