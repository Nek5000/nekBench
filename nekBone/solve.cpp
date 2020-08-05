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
void updateJacobi(BP_t* BP, occa::memory &o_lambda, occa::memory &o_invDiagA);

static void BPScaledAdd(BP_t* BP, dfloat alpha, occa::memory &o_a, dfloat beta, occa::memory &o_b);
static void BPPreconditioner(BP_t* BP, occa::memory &o_lambda, occa::memory &o_r, occa::memory &o_z);
static dfloat BPWeightedInnerProduct(BP_t* BP, occa::memory &o_w, occa::memory &o_a, occa::memory &o_b);
static void BPUpdatePCG(BP_t* BP, occa::memory &o_p, occa::memory &o_Ap, dfloat alpha,
                        occa::memory &o_x, occa::memory &o_r);
static void BPWeightedInnerProduct2(BP_t* BP, occa::memory &o_w, occa::memory &o_a, occa::memory &o_b, 
                                    dfloat *rdotz, dfloat *rdotr);
static dfloat BPAxOperator(BP_t* BP,
                  occa::memory &o_lambda,
                  occa::memory &o_q,
                  occa::memory &o_Aq,
                  const char* precision);


int BPPCG(BP_t* BP, occa::memory &o_lambda,
          occa::memory &o_r, occa::memory &o_x,
          const dfloat tol, const int MAXIT,
          double* opElapsed)
{
  mesh_t* mesh = BP->mesh;
  setupAide options = BP->options;

  const int flexible = options.compareArgs("KRYLOV SOLVER", "FLEXIBLE");
  const int verbose = options.compareArgs("VERBOSE", "TRUE");
  int fixedIterationCountFlag = 0;
  if(options.compareArgs("FIXED ITERATION COUNT", "TRUE"))
    fixedIterationCountFlag = 1;

  dfloat rdotz1 = 1;
  dfloat rdotz2 = 0;
  dfloat alpha = 0, beta = 0;
  dfloat TOL;

  occa::memory &o_p   = BP->o_solveWorkspace[0];
  occa::memory &o_z   = BP->o_solveWorkspace[1];
  occa::memory &o_Ap  = BP->o_solveWorkspace[2];
  occa::memory &o_Ax  = BP->o_solveWorkspace[3];

  dfloat pAp = BPAxOperator(BP, o_lambda, o_x, o_Ax, dfloatString);
  BPScaledAdd(BP, -1.f, o_Ax, 1.f, o_r);

  if(BP->profiling) timer::tic("preco");
  if(options.compareArgs("PRECONDITIONER", "JACOBI")) updateJacobi(BP, o_lambda, BP->o_invDiagA);
  if(BP->profiling) timer::toc("preco");

  int iter;
  for(iter = 1; iter <= MAXIT; ++iter) {
    BPPreconditioner(BP, o_lambda, o_r, o_z);
    rdotz2 = rdotz1;

    // dot(r,z) + dot(r,r)
    dfloat rdotr;  
    if(BP->profiling) timer::tic("dot1");
    BPWeightedInnerProduct2(BP, BP->o_invDegree, o_r, o_z, &rdotz1, &rdotr);
    if(BP->profiling) timer::toc("dot1");
    if(iter == 1) TOL = mymax(tol * tol * rdotr,tol * tol);

    // converged?
    if (verbose && mesh->rank == 0)
      printf("CG: it %d r norm %12.12le alpha = %le \n", iter, sqrt(rdotr), alpha);
    if(rdotr <= TOL && !fixedIterationCountFlag) break;

    if(flexible) {
      //TODO: fuse into BPWeightedInnerProduct2
      dfloat zdotAp = BPWeightedInnerProduct(BP, BP->o_invDegree, o_z, o_Ap);
      beta = -alpha * zdotAp / rdotz2;
    }else {
      beta = (iter == 1) ? 0:rdotz1 / rdotz2;
    }

    BPScaledAdd(BP, 1.f, o_z, beta, o_p);

    pAp = BPAxOperator(BP, o_lambda, o_p, o_Ap, dfloatString);

    alpha = rdotz1 / pAp;

    //  x <= x + alpha*p and r <= r - alpha*A*p
    if(BP->profiling) timer::tic("updatePCG"); 
    BPUpdatePCG(BP, o_p, o_Ap, alpha, o_x, o_r);
    if(BP->profiling) timer::toc("updatePCG");
  }

  return iter - 1;
}

void BPZeroMean(BP_t* BP, occa::memory &o_q)
{
  dfloat qmeanGlobal;

  dlong Nblock = BP->Nblock;
  dfloat* tmp = BP->tmp;
  mesh_t* mesh = BP->mesh;

  occa::memory &o_tmp = BP->o_tmp;

  dfloat qmeanLocal = 0;
  mesh->sumKernel(mesh->Nelements*mesh->Np, BP->fieldOffset, o_q, o_tmp);
  o_tmp.copyTo(tmp, Nblock*sizeof(dfloat));
  for(dlong n = 0; n < Nblock; ++n)
    qmeanLocal += tmp[n];

  MPI_Allreduce(&qmeanLocal, &qmeanGlobal, 1, MPI_DFLOAT, MPI_SUM, mesh->comm);
  qmeanGlobal /= ((dfloat) BP->NelementsGlobal * BP->Nfields*mesh->Np);
  mesh->addScalarKernel(BP->Nfields*BP->fieldOffset, -qmeanGlobal, o_q);
}

void BPUpdatePCG(BP_t* BP,
                   occa::memory &o_p, occa::memory &o_Ap, const dfloat alpha,
                   occa::memory &o_x, occa::memory &o_r)
{
  setupAide &options = BP->options;
  BP->updatePCGKernel(BP->Nfields*BP->fieldOffset, alpha, o_p, o_Ap, o_x, o_r); 
}

#include "../axhelm/kernelHelper.cpp"

void updateJacobi(BP_t* BP, occa::memory &o_lambda, occa::memory &o_invDiagA)
{
  mesh_t* mesh = BP->mesh;
  const dlong Nlocal = mesh->Np * mesh->Nelements;

  // todo: support BP->Nfields > 1
  BP->updateJacobiKernel(mesh->Nelements,
                         BP->fieldOffset,
                         BP->o_mapB,
                         mesh->o_ggeo,
                         mesh->o_D,
                         o_lambda,
                         o_invDiagA);

  const dfloat one = 1.0;
  //ogsGatherScatter(o_invDiagA, ogsDfloat, ogsAdd, BP->ogs);
  oogs::start(o_invDiagA, BP->Nfields, BP->fieldOffset, ogsDfloat, ogsAdd, (oogs_t*)BP->ogs);
  oogs::finish(o_invDiagA, BP->Nfields, BP->fieldOffset, ogsDfloat, ogsAdd,(oogs_t*)BP->ogs);
  BP->vecInvKernel(Nlocal, o_invDiagA);
}

void BPPreconditioner(BP_t* BP, occa::memory &o_lambda, occa::memory &o_r, occa::memory &o_z)
{
  mesh_t* mesh = BP->mesh;
  setupAide &options = BP->options;

  if(BP->profiling) timer::tic("preco");
  if(options.compareArgs("PRECONDITIONER", "JACOBI")) {
    BP->dotMultiplyKernel(BP->Nfields*BP->fieldOffset, o_r, BP->o_invDiagA, o_z);
  } else {
    BP->vecCopyKernel(BP->Nfields*BP->fieldOffset, o_r, o_z);
  }
  if(BP->profiling) timer::toc("preco");
}

dfloat BPAxOperator(BP_t* BP, occa::memory &o_lambda, occa::memory &o_q, occa::memory &o_Aq,
                  const char* precision)
{
  mesh_t* mesh = BP->mesh;
  setupAide &options = BP->options;
  oogs_t* ogs = (oogs_t*) BP->ogs;
  occa::kernel &kernel = BP->BPKernel[0];

  if(BP->overlap) {
    if(BP->profiling) timer::tic("Ax1");
    if(mesh->NglobalGatherElements)
      kernel(mesh->NglobalGatherElements,
             BP->fieldOffset,
             mesh->o_globalGatherElementList,
             mesh->o_ggeo,
             mesh->o_D,
             o_lambda,
             o_q,
             o_Aq);
    if(BP->profiling) timer::toc("Ax1");

    if(BP->profiling) timer::tic("AxGs");
    oogs::start(o_Aq, BP->Nfields, BP->fieldOffset, ogsDfloat, ogsAdd, ogs);
    if(BP->profiling) timer::tic("Ax2");
    kernel(mesh->NlocalGatherElements,
           BP->fieldOffset,
           mesh->o_localGatherElementList,
           mesh->o_ggeo,
           mesh->o_D,
           o_lambda,
           o_q,
           o_Aq);
    if(BP->profiling) timer::toc("Ax2");
    oogs::finish(o_Aq, BP->Nfields, BP->fieldOffset, ogsDfloat, ogsAdd, ogs);
    if(BP->profiling) timer::toc("AxGs");
  } else {
    if(BP->profiling) timer::tic("Ax");
    kernel(mesh->Nelements, BP->fieldOffset, mesh->o_ggeo, mesh->o_D, o_lambda, o_q, o_Aq);
    if(BP->profiling) timer::toc("Ax");

    if(BP->profiling) timer::tic("gs");

    //ogs_t *ogs = (ogs_t*)((oogs_t*)BP->ogs)->ogs;
    //ogsGatherScatterMany(o_Aq, BP->Nfields, BP->fieldOffset, ogsDfloat, ogsAdd, ogs);
    oogs::start(o_Aq, BP->Nfields, BP->fieldOffset, ogsDfloat, ogsAdd, ogs);
    oogs::finish(o_Aq, BP->Nfields, BP->fieldOffset, ogsDfloat, ogsAdd, ogs);
    if(BP->profiling) timer::toc("gs");
  }

  if(BP->profiling) timer::tic("dot2");
  dfloat pAp = BPWeightedInnerProduct(BP, BP->o_invDegree, o_q, o_Aq);
  if(BP->profiling) timer::toc("dot2");

  return pAp;
}

dfloat BPWeightedInnerProduct(BP_t* BP, occa::memory &o_w, occa::memory &o_a, occa::memory &o_b)
{
  mesh_t* mesh = BP->mesh;
  setupAide &options = BP->options;
  const int serial = options.compareArgs("THREAD MODEL", "SERIAL");
  const int omp = options.compareArgs("THREAD MODEL", "OPENMP");
  const dlong Ntotal = mesh->Nelements * mesh->Np;

  BP->weightedMultipleInnerProduct2Kernel(Ntotal, BP->fieldOffset, o_w, o_a, o_b, BP->o_tmp);

  dfloat wab = 0;
  if(serial || omp) {
    BP->o_tmp.copyTo(BP->tmp, sizeof(dfloat));
    wab = BP->tmp[0];
  } else {
    BP->o_tmp.copyTo(BP->tmp, BP->Nblock*sizeof(dfloat));
    for(dlong n = 0; n < BP->Nblock; ++n)
      wab += BP->tmp[n];
  }

  MPI_Allreduce(MPI_IN_PLACE, &wab, 1, MPI_DFLOAT, MPI_SUM, mesh->comm);
  return wab;
}

void BPWeightedInnerProduct2(BP_t* BP, occa::memory &o_w, occa::memory &o_r, occa::memory &o_z, dfloat *rdotz, dfloat *rdotr)
{
  mesh_t* mesh = BP->mesh;
  setupAide &options = BP->options;
  const int serial = options.compareArgs("THREAD MODEL", "SERIAL");
  const int omp = options.compareArgs("THREAD MODEL", "OPENMP");
  const dlong Ntotal = mesh->Nelements * mesh->Np;

  BP->weightedInnerProductUpdateKernel(Ntotal, BP->fieldOffset, BP->Nblock, o_w, o_r, o_z, BP->o_tmp);

  dfloat wab[] = {0,0};
  if(serial || omp) {
    BP->o_tmp.copyTo(BP->tmp, 2*sizeof(dfloat));
    wab[0] = BP->tmp[0];
    wab[1] = BP->tmp[1];
  } else {
    BP->o_tmp.copyTo(BP->tmp, 2*BP->Nblock*sizeof(dfloat));
    for(dlong n = 0; n < BP->Nblock; ++n) {
      wab[0] += BP->tmp[n];
      wab[1] += BP->tmp[n + BP->Nblock];
    }
  }

  MPI_Allreduce(MPI_IN_PLACE, wab, 2, MPI_DFLOAT, MPI_SUM, mesh->comm);
  *rdotz = wab[0];
  *rdotr = wab[1];
}

void BPScaledAdd(BP_t* BP, dfloat alpha, occa::memory &o_a, dfloat beta, occa::memory &o_b)
{
  const dlong Ntotal = BP->Nfields*BP->fieldOffset;
  BP->scaledAddKernel(Ntotal, alpha, o_a, beta, o_b);
}
