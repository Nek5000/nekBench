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
#include  "mpi.h"
#include "occa.hpp"
#include "meshBasis.hpp"

dfloat *drandAlloc(int N){

  dfloat *v = (dfloat*) calloc(N, sizeof(dfloat));

  for(int n=0;n<N;++n){
    v[n] = drand48();
  }

  return v;
}

int main(int argc, char **argv){

  const int Nq = atoi(argv[1]);
  dlong Nelements = atoi(argv[2]);
  char *threadModel = strdup(argv[3]);

  int mpi = 0;
  if(strstr(threadModel, "MPI")) mpi = 1;

  int rank = 0, size = 1;
  if(mpi) {
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    Nelements = Nelements/size;
  }

  int deviceId = 0;

  if(argc>=5)
    deviceId = atoi(argv[4]);
  
  int platformId = 0;
  if(argc>=6)
    platformId = atoi(argv[5]);

  if(rank==0) std::cout << "Running: Nq=" << Nq << " Nelements=" << Nelements << "\n";
  
  const int N = Nq-1;
  const int Np = Nq*Nq*Nq;
  const int Nggeo = 7;
  const int Ndim  = 1;
  const int Ntests = 10;
  const dfloat lambda = 0;
  
  const dlong offset = Nelements*Np;

  // ------------------------------------------------------------------------------
  // build element nodes and operators
  
  dfloat *rV, *wV, *DrV;

  meshJacobiGQ(0,0,N, &rV, &wV);
  meshDmatrix1D(N, Nq, rV, &DrV);

  // ------------------------------------------------------------------------------
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

  if(rank==0) std::cout <<  "active occa mode: " << device.mode() << "\n";

  // ------------------------------------------------------------------------------
  // build kernel defines
 
  occa::env::OCCA_MEM_BYTE_ALIGN = USE_OCCA_MEM_BYTE_ALIGN;
 
  occa::properties props;
  props["defines"].asObject();
  props["includes"].asArray();
  props["header"].asArray();
  props["flags"].asObject();

  props["defines/p_Nq"] = Nq;
  props["defines/p_Np"] = Np;

  props["defines/p_Nggeo"] = Nggeo;
  props["defines/p_G00ID"] = 0;
  props["defines/p_G01ID"] = 1;
  props["defines/p_G02ID"] = 2;
  props["defines/p_G11ID"] = 3;
  props["defines/p_G12ID"] = 4;
  props["defines/p_G22ID"] = 5;
  props["defines/p_GWJID"] = 6;

  props["defines/dfloat"] = dfloatString;
  props["defines/dlong"]  = dlongString;

  // build kernel  
  int BKid = 5;
  if (Ndim > 1) BKid = 6; 
  std::string filename = "BK" + std::to_string(BKid);
  std::string kernelName = "BK" + std::to_string(BKid);
  occa::kernel axKernel;

  if(strstr(threadModel, "NATIVE/CUDA")){
    const std::string filename = filename + ".cu";
    axKernel = device.buildKernel(filename, kernelName, "okl: false");
  } else if(strstr(threadModel, "NATIVE/SERIAL")){
    const std::string filename = filename + ".c";
    axKernel = device.buildKernel(filename, kernelName, "okl: false");
  } else { // fallback is okl
    axKernel = device.buildKernel(filename + ".okl", kernelName, props);
  }

  // populate device arrays
  dfloat *ggeo = drandAlloc(Np*Nelements*Nggeo);
  dfloat *q    = drandAlloc((Ndim*Np)*Nelements);
  dfloat *Aq   = drandAlloc((Ndim*Np)*Nelements);

  dlong *elementList = (dlong*) calloc(Nelements, sizeof(dlong));
  for(dlong e=0;e<Nelements;++e)
    elementList[e] = e;
  
  occa::memory o_ggeo  = device.malloc(Np*Nelements*Nggeo*sizeof(dfloat), ggeo);
  occa::memory o_q     = device.malloc((Ndim*Np)*Nelements*sizeof(dfloat), q);
  occa::memory o_Aq    = device.malloc((Ndim*Np)*Nelements*sizeof(dfloat), Aq);
  occa::memory o_DrV   = device.malloc(Nq*Nq*sizeof(dfloat), DrV);
  occa::memory o_elementList  = device.malloc(Nelements*sizeof(dlong), elementList);

  occa::streamTag start, end;

  // compute reference solution
  meshReferenceBK5(Nq, Nelements, lambda, ggeo, DrV, q, Aq);

  // warm up
  axKernel(Nelements, o_ggeo, o_DrV, lambda, o_q, o_Aq);

  // check for correctness
  o_Aq.copyTo(q);
  dfloat maxDiff = 0;
  for(int n=0;n<Ndim*Np*Nelements;++n){
    dfloat diff = fabs(q[n]-Aq[n]);
    maxDiff = (maxDiff<diff) ? diff:maxDiff;
  }
  if (maxDiff > 1e-14)
    printf("correctness check failed! e_inf = % e\n", maxDiff);

  // run kernel
  device.finish();
  if(mpi) MPI_Barrier(MPI_COMM_WORLD); 
  start = device.tagStream();

  for(int test=0;test<Ntests;++test)
    axKernel(Nelements, o_ggeo, o_DrV, lambda, o_q, o_Aq);
 
  if(mpi) MPI_Barrier(MPI_COMM_WORLD); 
  end = device.tagStream();

  device.finish();

  double elapsed = device.timeBetween(start, end)/Ntests;

  dfloat GnodesPerSecond = (Np*Nelements/elapsed)/1.e9;
  int bytesMoved = (2*Np+7*Np)*sizeof(dfloat); // x, Mx, opa
  double bw = (bytesMoved*Nelements/elapsed)/1.e9;

  if(rank==0) {
    std::cout << "Ndim=" << Ndim
              << " N=" << N
              << " Nelements=" << Nelements
              << " Nnodes=" << Ndim*Np*Nelements
              << " elapsed time=" << elapsed
              << " Gnodes/s=" << GnodesPerSecond
              << " GB/s=" << bw
              << "\n";
  } 

  if(mpi) MPI_Finalize(); 
  return 0;
}

