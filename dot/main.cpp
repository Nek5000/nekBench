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

occa::memory o_tmp;
occa::memory o_tmp2;
dfloat *tmp;
static occa::kernel kernel;

dfloat *drandAlloc(int N){

  dfloat *v = (dfloat*) calloc(N, sizeof(dfloat));

  for(int n=0;n<N;++n){
    v[n] = drand48();
  }

  return v;
}

dfloat weightedInnerProduct(dlong N, dlong Ncutoff, int Nblock, occa::memory &o_w, 
                            occa::memory &o_a, occa::memory &o_b, int global){

  dfloat globalwab = 0;
  kernel(N, o_w, o_a, o_b, o_tmp);

//  if(Nblock>Ncutoff){ /* add a second sweep if Nblock>Ncutoff */
//    sumKernel(Nblock, o_tmp, o_tmp2);
//    o_tmp2.copyTo(tmp);
//  }
//  else{
//    o_tmp.copyTo(tmp);
//  }    

  o_tmp.copyTo(tmp);
  dfloat wab = 0;
  for(dlong n=0;n<Nblock;++n) wab += tmp[n];

  if(global) MPI_Allreduce(&wab, &globalwab, 1, MPI_DFLOAT, MPI_SUM, MPI_COMM_WORLD);
   
  return globalwab;
}


int main(int argc, char **argv){

  if(argc<5){
    printf("Usage: ./dot N numElements [NATIVE|OKL]+SERIAL|CUDA|OPENCL CPU|VOLTA [nRepetitions] [deviceId] [blockSize] [MPI]\n");
    return 1;
  }

  int rank = 0, size = 1;
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  const int N = atoi(argv[1]);
  const dlong Nelements = atoi(argv[2]);
  std::string threadModel;
  threadModel.assign(strdup(argv[3]));

  std::string arch("");
  if(argc>=5)
    arch.assign(argv[4]);

  int Ntests = 100;
  if(argc>=6)
    Ntests = atoi(argv[5]);

  int deviceId = 0;
  if(argc>=7)
    deviceId = atoi(argv[6]);

  int blockSize = 256;
  if(argc>=8)
    blockSize = atoi(argv[7]);

  int global = 0;
  if(argc>=9)
    global = atoi(argv[8]);

  const int platformId = 0;

  const int Nq = N+1;
  const int Np = Nq*Nq*Nq;
  
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
  std::string kernelName = "weightedInnerProduct2";
  const int kernelVersion = 0; // hardwired for now
  kernelName += "_v" + std::to_string(kernelVersion);
  kernel = loadKernel(device, threadModel, arch, kernelName, N, Nelements, blockSize);

  // populate device arrays
  dfloat *a = drandAlloc(Np*Nelements);
  dfloat *b = drandAlloc(Np*Nelements);
  dfloat *c = drandAlloc(Np*Nelements);

  occa::memory o_a = device.malloc(Np*Nelements*sizeof(dfloat), a);
  occa::memory o_b = device.malloc(Np*Nelements*sizeof(dfloat), a);
  occa::memory o_c = device.malloc(Np*Nelements*sizeof(dfloat), a);

  int Nblock  = ((Nelements*Np)+blockSize-1)/blockSize;
  if(Nblock < 1) Nblock = 1;
  if(rank == 0) std::cout << "blockSize: " << Nblock << "\n";

  tmp = drandAlloc(Nblock);
  o_tmp = device.malloc(Nblock*sizeof(dfloat), tmp);
  o_tmp2 = device.malloc(Nblock*sizeof(dfloat), tmp);

  // run kernel
  weightedInnerProduct(Nelements*Np, 0, Nblock, o_c, o_a, o_b, global);   
  device.finish();
  MPI_Barrier(MPI_COMM_WORLD);
  const double start = MPI_Wtime();
  for(int test=0;test<Ntests;++test) {
    weightedInnerProduct(Nelements*Np, 0, Nblock, o_c, o_a, o_b, global);   
  }
  device.finish();
  MPI_Barrier(MPI_COMM_WORLD);
  const double elapsed = (MPI_Wtime() - start)/Ntests;

  // print statistics
  double GDOFPerSecond = size*(Nelements*N*N*N)/elapsed/1.e9;
  const long long bytesMoved = 3*Np; 
  const double bw = (size*bytesMoved*Nelements/elapsed)/1.e9;
  //  double flopCount = ?;
  //  double gflops = (size*flopCount*Nelements/elapsed)/1.e9;
  if(rank==0) {
    std::cout << "MPItasks=" << size
              << " OMPthreads=" << Nthreads
              << " NRepetitions=" << Ntests
              << " N=" << N
              << " Nelements=" << size*Nelements
              << " blockSize=" << blockSize
              << " elapsed time=" << elapsed
              << " GDOF/s=" << GDOFPerSecond
              << " GB/s=" << bw
              << "\n";
  } 

  MPI_Finalize();
  exit(0);
}
