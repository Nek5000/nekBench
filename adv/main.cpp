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

static occa::kernel kernel;

dfloat *drandAlloc(int N){

  dfloat *v = (dfloat*) calloc(N, sizeof(dfloat));

  for(int n=0;n<N;++n){
    v[n] = drand48();
  }

  return v;
}

int main(int argc, char **argv){

  if(argc<6){
    printf("Usage: ./adv N cubN numElements [NATIVE|OKL]+SERIAL|CUDA|OPENCL CPU|VOLTA [nRepetitions] [kernelVersion]\n");
    return 1;
  }

  int rank = 0, size = 1;
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  const int N = atoi(argv[1]);
  const int cubN = atoi(argv[2]);
  const dlong Nelements = atoi(argv[3]);
  std::string threadModel;
  threadModel.assign(strdup(argv[4]));

  std::string arch("");
  if(argc>=6)
    arch.assign(argv[5]);

  int Ntests = 1;
  if(argc>=7)
    Ntests = atoi(argv[6]);

  int kernelVersion = 0;
  if(argc>=8)
    kernelVersion = atoi(argv[7]);

  const int deviceId = 0;
  const int platformId = 0;

  const int Nq = N+1;
  const int cubNq = cubN+1;
  const int Np = Nq*Nq*Nq;
  const int cubNp = cubNq*cubNq*cubNq;
  
  const dlong offset = Nelements*Np;

  // build device
  occa::device device;
  char deviceConfig[BUFSIZ];

  if(strstr(threadModel.c_str(), "CUDA")){
    sprintf(deviceConfig, "mode: 'CUDA', device_id: %d",deviceId);
  }
  else if(strstr(threadModel.c_str(),  "HIP")){
    sprintf(deviceConfig, "mode: 'HIP', device_id: %d",deviceId);
  }
  else if(strstr(threadModel.c_str(),  "OPENCL")){
    sprintf(deviceConfig, "mode: 'OpenCL', device_id: %d, platform_id: %d", deviceId, platformId);
  }
  else if(strstr(threadModel.c_str(),  "OPENMP")){
    sprintf(deviceConfig, "mode: 'OpenMP' ");
  }
  else{
    sprintf(deviceConfig, "mode: 'Serial' ");
    omp_set_num_threads(1);
  }

  int Nthreads =  omp_get_max_threads();
  std::string deviceConfigString(deviceConfig);
  device.setup(deviceConfigString);
  occa::env::OCCA_MEM_BYTE_ALIGN = USE_OCCA_MEM_BYTE_ALIGN;

  if(rank==0) {
   std::cout << "word size: " << sizeof(dfloat) << " bytes\n";
   std::cout << "active occa mode: " << device.mode() << "\n";
  }

  // load kernel
  std::string kernelName = "advCubatureHex3D";
  kernelName += "_v" + std::to_string(kernelVersion);
  kernel = loadKernel(device, threadModel, arch, kernelName, N, cubN, Nelements);

  // populate device arrays
  dfloat *vgeo           = drandAlloc(Np*Nelements*p_Nvgeo);
  dfloat *cubvgeo        = drandAlloc(cubNp*Nelements*p_Nvgeo);
  dfloat *cubDiffInterpT = drandAlloc(3*cubNp*Nelements);
  dfloat *cubInterpT     = drandAlloc(Np*cubNp);
  dfloat *cubProjectT    = drandAlloc(Np*cubNp);
  dfloat *u              = drandAlloc(3*Np*Nelements);
  dfloat *adv            = drandAlloc(3*Np*Nelements);

  occa::memory o_vgeo           = device.malloc(Np*Nelements*p_Nvgeo*sizeof(dfloat), vgeo);
  occa::memory o_cubvgeo        = device.malloc(cubNp*Nelements*p_Nvgeo*sizeof(dfloat), cubvgeo);
  occa::memory o_cubDiffInterpT = device.malloc(3*cubNp*Nelements*sizeof(dfloat), cubDiffInterpT);
  occa::memory o_cubInterpT     = device.malloc(Np*cubNp*sizeof(dfloat), cubInterpT);
  occa::memory o_cubProjectT    = device.malloc(Np*cubNp*sizeof(dfloat), cubProjectT);
  occa::memory o_u              = device.malloc(3*Np*Nelements*sizeof(dfloat), u);
  occa::memory o_adv             = device.malloc(3*Np*Nelements*sizeof(dfloat), adv);

  // run kernel
  kernel(
       Nelements,
       o_vgeo,
       o_cubvgeo,
       o_cubDiffInterpT,
       o_cubInterpT,
       o_cubProjectT,
       offset,
       o_u,
       o_adv);
  device.finish();
  MPI_Barrier(MPI_COMM_WORLD);
  const double start = MPI_Wtime();
  for(int test=0;test<Ntests;++test) {
    kernel(
         Nelements,
         o_vgeo,
         o_cubvgeo,
         o_cubDiffInterpT,
         o_cubInterpT,
         o_cubProjectT,
         offset,
         o_u,
         o_adv);
  }
  device.finish();
  MPI_Barrier(MPI_COMM_WORLD);
  const double elapsed = (MPI_Wtime() - start)/Ntests;

  // print statistics
  const dfloat GDOFPerSecond = (size*(N*N*N)*Nelements/elapsed)/1.e9;
//  const long long bytesMoved = ?; 
//  const double bw = (size*bytesMoved*Nelements/elapsed)/1.e9;
//  double flopCount = ?;
//  double gflops = (size*flopCount*Nelements/elapsed)/1.e9;
  if(rank==0) {
    std::cout << "MPItasks=" << size
              << " OMPthreads=" << Nthreads
              << " NRepetitions=" << Ntests
              << " N=" << N
              << " cubN=" << cubN
              << " Nelements=" << size*Nelements
              << " elapsed time=" << elapsed
              << " GDOF/s=" << GDOFPerSecond
              //<< " GB/s=" << bw
              //<< " GFLOPS/s=" << gflops
              << "\n";
  } 

  MPI_Finalize();
  exit(0);
}
