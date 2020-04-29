#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cmath>
#include "omp.h"
#include <unistd.h>
#include "mpi.h"
#include "occa.hpp"
#include "meshBasis.hpp"
#include "timer.hpp"

int main(int argc, char **argv){

  MPI_Init(&argc, &argv);

  if(argc<2){
    printf("Usage: ./bw SERIAL|CUDA|OPENCL\n");
    return 1;
  }

  std::string threadModel;
  threadModel.assign(strdup(argv[1]));

  const int deviceId = 0;
  const int platformId = 0;

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

 std::cout << "active occa mode: " << device.mode() << "\n";

  occa::properties props;
  props["defines"].asObject();
  props["includes"].asArray();
  props["header"].asArray();
  props["flags"].asObject();
  occa::kernel triadKernel = device.buildKernel("kernel/triad.okl", "triad", props);
  double *u = (double*) calloc(1e8,sizeof(double));
  occa::memory o_a = device.malloc(1e8*sizeof(double), u);
  occa::memory o_b = device.malloc(1e8*sizeof(double), u);
  occa::memory o_c = device.malloc(1e8*sizeof(double), u);

  timer::init(MPI_COMM_WORLD, device, 0);

  int N[] = {1, 1000*512, 2000*512, 4000*512};
  int Ntests = 1000;
  for(int test=0;test<Ntests;++test) triadKernel(N[0], 1.0, o_a, o_b, o_c);
  for(int i=0; i<4; ++i) { 
    long long int bytes = 3*N[i]*sizeof(double);
    device.finish();
    timer::reset("triad");
    timer::tic("triad");
    for(int test=0;test<Ntests;++test) triadKernel(N[i], 1.0, o_a, o_b, o_c);
    device.finish();
    timer::toc("triad");
    double elapsed = timer::query("triad", "HOST:MAX")/Ntests;
    std::cout << "triad stream: "
              << N[i] << ", "
              << elapsed << " s, "
              << bytes/elapsed << " bytes/s\n";
  }

  std::cout << "\n";

  Ntests = 100;
  for(int i=0; i<8; ++i) {
    const long long int bytes = pow(10,i)*sizeof(double);
    device.finish();
    timer::reset("memcpyDH");
    timer::tic("memcpyDH");
    for(int test=0;test<Ntests;++test) o_a.copyTo(u, bytes);
    device.finish();
    timer::toc("memcpyDH");
    const double elapsed = timer::query("memcpyDH", "HOST:MAX")/Ntests;
    std::cout << "D->H copy " <<  bytes << " bytes, " 
              << elapsed << " s, "
              << bytes/elapsed << " bytes/s\n"; 
  }

  std::cout << "\n";

  Ntests = 1000;
  for(int i=0; i<9; ++i) {
    const long long int bytes = pow(10,i)*sizeof(double);
    device.finish();
    timer::reset("memcpyDD");
    timer::tic("memcpyDD");
    for(int test=0;test<Ntests;++test) o_a.copyTo(o_b, bytes);
    device.finish();
    timer::toc("memcpyDD");
    const double elapsed = timer::query("memcpyDD", "HOST:MAX")/Ntests;
    std::cout << "D->D copy " <<  bytes << " bytes, " 
              << elapsed << " s, "
              << bytes/elapsed << " bytes/s\n"; 
  }



  MPI_Finalize();
  exit(0);
}
