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

#ifndef BP_H
#define BP_H 1

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "mpi.h"
#include "params.h"
#include "mesh.h"
#include "timer.hpp"

typedef struct
{
  hlong row;
  hlong col;
  int ownerRank;
  dfloat val;
}nonZero_t;

typedef struct
{
  int profiling;
  int BPid;
  int knlId;
  bool overlap;

  int dim;
  int elementType; // number of edges (3=tri, 4=quad, 6=tet, 12=hex)
  int Nfields;

  dlong fieldOffset;

  mesh_t* mesh;

  void* ogs;

  setupAide options;

  char* type;

  dlong Nblock;
  dlong Nblock2; // second reduction

  dfloat tau;

  int* BCType;

  bool allNeumann;

  // field
  dfloat* q;

  dfloat lambda1;

  // HOST shadow copies
  dfloat* x, * r, * lambda;
  dfloat* solveWorkspace;
  dfloat* tmp;

  dfloat* invDegree;

  int* EToB;

  dfloat* sendBuffer, * recvBuffer;
  dfloat* gradSendBuffer, * gradRecvBuffer;

  occa::memory o_sendBuffer, o_recvBuffer;
  occa::memory h_sendBuffer, h_recvBuffer;

  occa::stream defaultStream;
  occa::stream stream1;
  occa::stream dataStream;

  occa::memory o_mapB;
  occa::memory o_q, o_x, o_r, o_lambda;
  occa::memory o_invDiagA;

  // PCG storage
  int NsolveWorkspace;
  dlong offsetSolveWorkspace;

  occa::memory* o_solveWorkspace;
  occa::memory o_tmp; // temporary
  occa::memory o_tmp2; // temporary (second reduction)
  occa::memory o_invDegree;
  occa::memory o_EToB;

  occa::kernel* BPKernel;

  occa::kernel updateJacobiKernel;

  occa::kernel innerProductKernel;

  occa::kernel weightedInnerProduct1Kernel;
  occa::kernel weightedInnerProduct2Kernel;
  occa::kernel weightedMultipleInnerProduct2Kernel;

  occa::kernel innerProduct2Kernel;
  occa::kernel multipleInnerProduct2Kernel;

  occa::kernel scaledAddKernel;
  occa::kernel dotMultiplyKernel;
  occa::kernel dotMultiplyAddKernel;
  occa::kernel dotDivideKernel;

  occa::kernel weightedNorm2Kernel;
  occa::kernel weightedMultipleNorm2Kernel;

  occa::kernel norm2Kernel;
  occa::kernel multipleNorm2Kernel;

  occa::kernel vecInvKernel;
  occa::kernel vecZeroKernel;
  occa::kernel vecScaleKernel;
  occa::kernel vecCopyKernel;

  occa::kernel vecScatterKernel;
  occa::kernel vecMultipleScatterKernel;

  occa::kernel weightedInnerProductUpdateKernel;
  occa::kernel updatePCGKernel;

  occa::memory* o_pcgWork;

  hlong NelementsGlobal;
}BP_t;

BP_t* setup(mesh_t* mesh, occa::properties &kernelInfo, setupAide &options);

void solveSetup(BP_t* BP, occa::properties &kernelInfo);

void BPStartHaloExchange(BP_t* BP,
                         occa::memory &o_q,
                         int Nentries,
                         dfloat* sendBuffer,
                         dfloat* recvBuffer);
void BPInterimHaloExchange(BP_t* BP,
                           occa::memory &o_q,
                           int Nentries,
                           dfloat* sendBuffer,
                           dfloat* recvBuffer);
void BPEndHaloExchange(BP_t* BP, occa::memory &o_q, int Nentries, dfloat* recvBuffer);

//Linear solvers
int BPPCG   (BP_t* BP,
             occa::memory &o_lambda,
             occa::memory &o_r,
             occa::memory &o_x,
             const dfloat tol,
             const int MAXIT,
             double* opElapsed);

void BPBuildContinuous(BP_t* BP, dfloat lambda, nonZero_t** A,
                       dlong* nnz, ogs_t** ogs, hlong* globalStarts);

void BPBuildJacobi(BP_t* BP, dfloat lambda, dfloat** invDiagA);
void BPZeroMean(BP_t* BP, occa::memory &o_q);
occa::properties BPKernelInfo(mesh_t* mesh);

#endif
