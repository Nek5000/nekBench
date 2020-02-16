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

#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "omp.h"
#include <unistd.h>
#include "mpi.h"
#include "occa.hpp"
#include "meshBasis.hpp"

#include "kernelHelper.cpp"

dfloat *drandAlloc(int N){

  dfloat *v = (dfloat*) calloc(N, sizeof(dfloat));

  for(int n=0;n<N;++n){
    v[n] = drand48();
  }

  return v;
}

int main(int argc, char **argv){

  if(argc<6){
    printf("Usage: ./axhelm N Ndim numElements [NATIVE|OKL]+SERIAL|CUDA|OPENCL SERIAL+VOLTA [kernelVersion] [deviceID] [platformID]\n");
    return 1;
  }

  int rank = 0, size = 1;
#ifdef USEMPI
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
#endif

  const int N = atoi(argv[1]);
  const int Ndim = atoi(argv[2]);
  const dlong Nelements = atoi(argv[3]);
  char *threadModel = strdup(argv[4]);

  std::string arch("");
  if(argc>=6)
    arch.assign(argv[5]);

  int kernelVersion = 0;
  if(argc>=7)
    kernelVersion = atoi(argv[6]);

  int deviceId = 0;
  if(argc>=8)
    deviceId = atoi(argv[6]);
  
  int platformId = 0;
  if(argc>=9)
    platformId = atoi(argv[8]);

  const int Nq = N+1;
  const int Np = Nq*Nq*Nq;
  const int Ntests = 100;
  const dfloat lambda = 1.1;
  
  const dlong offset = Nelements*Np;

  const int assembled = 0;

  // build element nodes and operators
  dfloat *rV, *wV, *DrV;
  meshJacobiGQ(0,0,N, &rV, &wV);
  meshDmatrix1D(N, Nq, rV, &DrV);

  // build device
  occa::device device;
  char deviceConfig[BUFSIZ];

  if(strstr(threadModel, "CUDA")){
    sprintf(deviceConfig, "mode: 'CUDA', device_id: %d",deviceId);
  }
  else if(strstr(threadModel,  "HIP")){
    sprintf(deviceConfig, "mode: 'HIP', device_id: %d",deviceId);
  }
  else if(strstr(threadModel,  "OPENCL")){
    sprintf(deviceConfig, "mode: 'OpenCL', device_id: %d, platform_id: %d", deviceId, platformId);
  }
  else if(strstr(threadModel,  "OPENMP")){
    sprintf(deviceConfig, "mode: 'OpenMP' ");
  }
  else{
    sprintf(deviceConfig, "mode: 'Serial' ");
  }

  std::string deviceConfigString(deviceConfig);
  device.setup(deviceConfigString);
  occa::env::OCCA_MEM_BYTE_ALIGN = USE_OCCA_MEM_BYTE_ALIGN;

  if(rank==0) {
   std::cout << "word size: " << sizeof(dfloat) << " bytes\n";
   std::cout << "active occa mode: " << device.mode() << "\n";
  }

  // load kernel
  std::string kernelName = "axhelm";
  if(assembled) kernelName = "axhelm_partial"; 
  if(Ndim > 1) kernelName += "_N" + std::to_string(Ndim);
  kernelName += "_v" + std::to_string(kernelVersion);
  loadAxKernel(device, threadModel, arch, kernelName, N, Nelements);

  // populate device arrays
  dfloat *ggeo = drandAlloc(Np*Nelements*p_Nggeo);
  dfloat *q    = drandAlloc((Ndim*Np)*Nelements);
  dfloat *Aq   = drandAlloc((Ndim*Np)*Nelements);

  occa::memory o_ggeo  = device.malloc(Np*Nelements*p_Nggeo*sizeof(dfloat), ggeo);
  occa::memory o_q     = device.malloc((Ndim*Np)*Nelements*sizeof(dfloat), q);
  occa::memory o_Aq    = device.malloc((Ndim*Np)*Nelements*sizeof(dfloat), Aq);
  occa::memory o_DrV   = device.malloc(Nq*Nq*sizeof(dfloat), DrV);

  occa::streamTag start, end;

  // warm up
  runAxKernel(Nelements, Ndim, offset, o_ggeo, o_DrV, lambda, o_q, o_Aq, assembled);

  // check for correctness
  for(int n=0;n<Ndim;++n){
    dfloat *x = q + n*offset;
    dfloat *Ax = Aq + n*offset; 
    meshReferenceBK5(Nq, Nelements, lambda, ggeo, DrV, x, Ax);
  }
  o_Aq.copyTo(q);
  dfloat maxDiff = 0;
  for(int n=0;n<Ndim*Np*Nelements;++n){
    dfloat diff = fabs(q[n]-Aq[n]);
    maxDiff = (maxDiff<diff) ? diff:maxDiff;
  }
  if (rank==0 && maxDiff > 5e-12) {
    std::cout << "WARNING: Correctness check failed!" << maxDiff << "\n";
  }

  // run kernel
  device.finish();
#ifdef USEMPI
  MPI_Barrier(MPI_COMM_WORLD);
#endif
  start = device.tagStream();

  for(int test=0;test<Ntests;++test)
    runAxKernel(Nelements, Ndim, offset, o_ggeo, o_DrV, lambda, o_q, o_Aq, assembled);

 #ifdef USEMPI
  MPI_Barrier(MPI_COMM_WORLD);
#endif
  end = device.tagStream();

  // print statistics
  const double elapsed = device.timeBetween(start, end)/Ntests;
  const dfloat GnodesPerSecond = (size*Ndim*Np*Nelements/elapsed)/1.e9;
  const long long bytesMoved = (Ndim*2*Np+7*Np)*sizeof(dfloat); // x, Mx, opa
  const double bw = (size*bytesMoved*Nelements/elapsed)/1.e9;
  double flopCount = Ndim*Np*12*Nq;
  if(Ndim == 1) flopCount += 18*Np;
  if(Ndim == 3) flopCount += 57*Np;
  double gflops = (flopCount*size*Nelements/elapsed)/1.e9;
  if(rank==0) {
    std::cout << "MPItasks=" << size
              << " Ndim=" << Ndim
              << " N=" << N
              << " Nelements=" << size*Nelements
              << " Nnodes=" << Ndim*Np*size*Nelements
              << " elapsed time=" << elapsed
              << " Gnodes/s=" << GnodesPerSecond
              << " GB/s=" << bw
              << " GFLOPS/s=" << gflops
              << "\n";
  } 

#ifdef USEMPI
  MPI_Finalize();
#endif 
  exit(0);
}
