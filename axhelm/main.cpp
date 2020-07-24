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
#include "axhelmReference.cpp"


static occa::kernel axKernel;

dfloat *drandAlloc(int N){

  dfloat *v = (dfloat*) calloc(N, sizeof(dfloat));

  for(int n=0;n<N;++n){
    v[n] = drand48();
  }

  return v;
}

int main(int argc, char **argv){

  if(argc<6){
    printf("Usage: ./axhelm N Ndim numElements [NATIVE|OKL]+SERIAL|CUDA|HIP|OPENCL CPU|VOLTA [BKmode] [nRepetitions] [kernelVersion]\n");
    return 1;
  }

  int rank = 0, size = 1;
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  const int N = atoi(argv[1]);
  const int Ndim = atoi(argv[2]);
  const dlong Nelements = atoi(argv[3]);
  std::string threadModel;
  threadModel.assign(strdup(argv[4]));

  std::string arch("");
  if(argc>=6)
    arch.assign(argv[5]);

  int BKmode = 0;
  if(argc>=7)
    BKmode = atoi(argv[6]);

  int Ntests = 100;
  if(argc>=8)
    Ntests = atoi(argv[7]);

  int kernelVersion = 0;
  if(argc>=9)
    kernelVersion = atoi(argv[8]);

  const int deviceId = 0;
  const int platformId = 0;

  const int Nq = N+1;
  const int Np = Nq*Nq*Nq;
  
  const dlong offset = Nelements*Np;

  const int assembled = 0;

  // build element nodes and operators
  dfloat *rV, *wV, *DrV;
  meshJacobiGQ(0,0,N, &rV, &wV);
  meshDmatrix1D(N, Nq, rV, &DrV);

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
  std::string kernelName = "axhelm";
  if(assembled) kernelName = "axhelm_partial"; 
  if(BKmode) kernelName = "axhelm_bk";
  if(Ndim > 1) kernelName += "_n" + std::to_string(Ndim);
  kernelName += "_v" + std::to_string(kernelVersion);
  axKernel = loadAxKernel(device, threadModel, arch, kernelName, N, Nelements);

  // populate device arrays
  dfloat *ggeo = drandAlloc(Np*Nelements*p_Nggeo);
  dfloat *q    = drandAlloc((Ndim*Np)*Nelements);
  dfloat *Aq   = drandAlloc((Ndim*Np)*Nelements);

  occa::memory o_ggeo   = device.malloc(Np*Nelements*p_Nggeo*sizeof(dfloat), ggeo);
  occa::memory o_q      = device.malloc((Ndim*Np)*Nelements*sizeof(dfloat), q);
  occa::memory o_Aq     = device.malloc((Ndim*Np)*Nelements*sizeof(dfloat), Aq);
  occa::memory o_DrV    = device.malloc(Nq*Nq*sizeof(dfloat), DrV);

  dfloat lambda1 = 1.1;
  if(BKmode) lambda1 = 0;
  dfloat *lambda = (dfloat*) calloc(2*offset, sizeof(dfloat));
  for(int i=0; i<offset; i++) {
    lambda[i]        = 1.0; // don't change
    lambda[i+offset] = lambda1;
  }
  occa::memory o_lambda = device.malloc(2*offset*sizeof(dfloat), lambda);

  // run kernel
  axKernel(Nelements, offset, o_ggeo, o_DrV, o_lambda, o_q, o_Aq);
  device.finish();
  MPI_Barrier(MPI_COMM_WORLD);
  const double start = MPI_Wtime();

  for(int test=0;test<Ntests;++test)
    axKernel(Nelements, offset, o_ggeo, o_DrV, o_lambda, o_q, o_Aq);

  device.finish();
  MPI_Barrier(MPI_COMM_WORLD);
  const double elapsed = (MPI_Wtime() - start)/Ntests;

  // check for correctness
  for(int n=0;n<Ndim;++n){
    dfloat *x = q + n*offset;
    dfloat *Ax = Aq + n*offset; 
    axhelmReference(Nq, Nelements, lambda1, ggeo, DrV, x, Ax);
  }
  o_Aq.copyTo(q);
  dfloat maxDiff = 0;
  for(int n=0;n<Ndim*Np*Nelements;++n){
    dfloat diff = fabs(q[n]-Aq[n]);
    maxDiff = (maxDiff<diff) ? diff:maxDiff;
  }
  MPI_Allreduce(MPI_IN_PLACE, &maxDiff, 1, MPI_DFLOAT, MPI_SUM, MPI_COMM_WORLD);
  if (rank==0)
    std::cout << "Correctness check: maxError = " << maxDiff << "\n";

  // print statistics
  const dfloat GDOFPerSecond = (size*Ndim*(N*N*N)*Nelements/elapsed)/1.e9;
  long long bytesMoved = (Ndim*2*Np+7*Np)*sizeof(dfloat); // x, Ax, geo
  if(!BKmode) bytesMoved += 2*Np*sizeof(dfloat);
  const double bw = (size*bytesMoved*Nelements/elapsed)/1.e9;
  double flopCount = Ndim*Np*12*Nq;
  flopCount += 15*Np;
  if(!BKmode) flopCount += 5*Np;
  flopCount *= Ndim;
  double gflops = (size*flopCount*Nelements/elapsed)/1.e9;
  if(rank==0) {
    std::cout << "MPItasks=" << size
              << " OMPthreads=" << Nthreads
              << " NRepetitions=" << Ntests
              << " Ndim=" << Ndim
              << " N=" << N
              << " Nelements=" << size*Nelements
              << " elapsed time=" << elapsed
              << " GDOF/s=" << GDOFPerSecond
              << " GB/s=" << bw
              << " GFLOPS/s=" << gflops
              << "\n";
  } 

  MPI_Finalize();
  exit(0);
}
