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

static int USEMPI = 0;
static int Nelements = 0;
static int N;
static int assembled = 0;

dfloat *drandAlloc(int N){

  dfloat *v = (dfloat*) calloc(N, sizeof(dfloat));

  for(int n=0;n<N;++n){
    v[n] = drand48();
  }

  return v;
}

occa::kernel loadKernel(occa::device device, char *threadModel,
                        std::string arch, std::string kernelName){

  int rank = 1;
  if(USEMPI) MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  const int Nq = N + 1;
  const int Np = Nq*Nq*Nq;
  occa::env::OCCA_MEM_BYTE_ALIGN = USE_OCCA_MEM_BYTE_ALIGN;
 
  occa::properties props;
  props["defines"].asObject();
  props["includes"].asArray();
  props["header"].asArray();
  props["flags"].asObject();

  props["defines/p_Nq"] = Nq;
  props["defines/p_Np"] = Np;

  props["defines/p_Nggeo"] = p_Nggeo;
  props["defines/p_G00ID"] = p_G00ID;
  props["defines/p_G01ID"] = p_G01ID;
  props["defines/p_G02ID"] = p_G02ID;
  props["defines/p_G11ID"] = p_G11ID;
  props["defines/p_G12ID"] = p_G12ID;
  props["defines/p_G22ID"] = p_G22ID;
  props["defines/p_GWJID"] = p_GWJID;

  props["defines/dfloat"] = dfloatString;
  props["defines/dlong"]  = dlongString;

  props["okl"] = false;

  occa::kernel axKernel;

  std::string filename = "BK5" + arch;
  for (int r=0;r<2;r++){
    if ((r==0 && rank==0) || (r==1 && rank>0)) {
      if(strstr(threadModel, "NATIVE+CUDA")){
        axKernel = device.buildKernel(filename + ".cu", kernelName, props);
        axKernel.setRunDims(Nelements, (N+1)*(N+1));
      } else if(strstr(threadModel, "NATIVE+SERIAL")){
        props["defines/USE_OCCA_MEM_BYTE_ALIGN"] = USE_OCCA_MEM_BYTE_ALIGN;
        axKernel = device.buildKernel(filename + ".c", kernelName, props);
      } else { // fallback is okl
        props["okl"] = true;
        axKernel = device.buildKernel(filename + ".okl", kernelName, props);
      }
    }
    if(USEMPI) MPI_Barrier(MPI_COMM_WORLD);
  }
  return axKernel;
}

void runKernel(occa::kernel axKernel,
               dlong Nelements, occa::memory o_ggeo, occa::memory o_DrV, dfloat lambda, 
               occa::memory o_q, occa::memory o_Aq){

    if(assembled) {
/*
      axKernel(NglobalGatherElements, o_globalGatherElementList, o_ggeo, o_DrV, 
               lambda, o_q, o_Aq);
      ogsGatherScatterStart(o_Aq, ogsDfloat, ogsAdd, ogs);
      axKernel(NlocalGatherElements, o_localGatherElementList, o_ggeo, o_DrV, 
               lambda, o_q, o_Aq);
      ogsGatherScatterFinish(o_Aq, ogsDfloat, ogsAdd, ogs);
*/
      std::cout << "ERROR: assembled version not implemented yet!\n";
      exit(1);
    } else {
      axKernel(Nelements, o_ggeo, o_DrV, lambda, o_q, o_Aq);
    }
}

int main(int argc, char **argv){

  if(argc<4){
    printf("Usage: ./axhelm N numElements [MPI]+[NATIVE|OKL]+SERIAL|CUDA|OPENCL|SERIAL+VOLTA [kernelVersion] [deviceID] [platformID]\n");
    return 1;
  }

  N = atoi(argv[1]);
  Nelements = atoi(argv[2]);
  char *threadModel = strdup(argv[3]);

  if(strstr(threadModel, "MPI")) USEMPI = 1;

  int rank = 0, size = 1;
  if(USEMPI) {
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
  }

  std::string arch("");
  if(argc>=5)
    arch.assign(argv[4]);

  int kernelVersion = 0;
  if(argc>=6)
    kernelVersion = atoi(argv[5]);

  int deviceId = 0;
  if(argc>=7)
    deviceId = atoi(argv[6]);
  
  int platformId = 0;
  if(argc>=8)
    platformId = atoi(argv[7]);

  const int Nq = N+1;
  const int Np = Nq*Nq*Nq;
  const int Ndim  = 1;
  const int Ntests = 40;
  const dfloat lambda = 0;
  
  const dlong offset = Nelements*Np;

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

  if(rank==0) {
   std::cout << "word size: " << sizeof(dfloat) << " bytes\n";
   std::cout << "active occa mode: " << device.mode() << "\n";
  }

  // load kernel
  std::string kernelName = "BK5_v";
  if(assembled) kernelName = "BK5partial_v";   
  kernelName += std::to_string(kernelVersion);
  occa::kernel axKernel = loadKernel(device, threadModel, arch, kernelName);

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
  runKernel(axKernel, Nelements, o_ggeo, o_DrV, lambda, o_q, o_Aq);

  // check for correctness
  meshReferenceBK5(Nq, Nelements, lambda, ggeo, DrV, q, Aq);
  o_Aq.copyTo(q);
  dfloat maxDiff = 0;
  for(int n=0;n<Ndim*Np*Nelements;++n){
    dfloat diff = fabs(q[n]-Aq[n]);
    maxDiff = (maxDiff<diff) ? diff:maxDiff;
  }
  if (rank==0 && maxDiff > 1e-12) {
    std::cout << "WARNING: Correctness check failed!" << maxDiff << "\n";
  }

  // run kernel
  device.finish();
  if(USEMPI) MPI_Barrier(MPI_COMM_WORLD); 
  start = device.tagStream();

  for(int test=0;test<Ntests;++test)
    runKernel(axKernel, Nelements, o_ggeo, o_DrV, lambda, o_q, o_Aq);
 
  if(USEMPI) MPI_Barrier(MPI_COMM_WORLD); 
  end = device.tagStream();

  // print statistics
  const double elapsed = device.timeBetween(start, end)/Ntests;
  const dfloat GnodesPerSecond = (size*Np*Nelements/elapsed)/1.e9;
  const int bytesMoved = (2*Np+7*Np)*sizeof(dfloat); // x, Mx, opa
  const double bw = (size*bytesMoved*Nelements/elapsed)/1.e9;
  double flopCount = Np*(6*2*Nq + 17);
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

  if(USEMPI) MPI_Finalize(); 
  return 0;
}
