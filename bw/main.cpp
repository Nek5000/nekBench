#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cmath>
#include "omp.h"
#include <unistd.h>
#include "mpi.h"
#include "occa.hpp"
#include "timer.hpp"
#include "setCompilerFlags.hpp"

int main(int argc, char** argv)
{
  MPI_Init(&argc, &argv);

  if(argc < 2) {
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

  if(strstr(threadModel.c_str(), "CUDA")) {
    sprintf(deviceConfig, "mode: 'CUDA', device_id: %d",deviceId);
  }else if(strstr(threadModel.c_str(),  "HIP")) {
    sprintf(deviceConfig, "mode: 'HIP', device_id: %d",deviceId);
  }else if(strstr(threadModel.c_str(),  "OPENCL")) {
    sprintf(deviceConfig, "mode: 'OpenCL', device_id: %d, platform_id: %d", deviceId, platformId);
  }else if(strstr(threadModel.c_str(),  "OPENMP")) {
    sprintf(deviceConfig, "mode: 'OpenMP' ");
  }else {
    sprintf(deviceConfig, "mode: 'Serial' ");
    omp_set_num_threads(1);
  }

  int Nthreads =  omp_get_max_threads();
  std::string deviceConfigString(deviceConfig);
  device.setup(deviceConfigString);
  occa::env::OCCA_MEM_BYTE_ALIGN = USE_OCCA_MEM_BYTE_ALIGN;

  std::cout << "active occa mode: " << device.mode() << "\n\n";

  occa::properties props;
  props["defines"].asObject();
  props["includes"].asArray();
  props["header"].asArray();
  props["flags"].asObject();
  setCompilerFlags(device, props);

  timer::init(MPI_COMM_WORLD, device, 0);

  {
    const int Ntests = 2000;
    const int N[] = {1, 1024, 8192, 16384, 64000, 256*512, 512 * 512, 1000 * 512, 2000 * 512, 4000 * 512, 8000 * 512, 16000 * 512};
    const int Nsize = sizeof(N) / sizeof(int);
    const int nWords = N[sizeof(N) / sizeof(N[0]) - 1];
    occa::memory o_a = device.malloc(nWords * sizeof(double));
    occa::memory o_b = device.malloc(nWords * sizeof(double));
    occa::memory o_c = device.malloc(nWords * sizeof(double));
    occa::kernel triadKernel = device.buildKernel(DBP "kernel/bwKernels.okl", "triad", props);

    for(int test = 0; test < Ntests; ++test) triadKernel(1000, 1.0, o_a, o_b, o_c);

    for(int i = 0; i < Nsize; ++i) {
      long long int bytes = 3 * N[i] * sizeof(double);
      device.finish();
      timer::reset("triad");
      timer::tic("triad");
      for(int test = 0; test < Ntests; ++test) triadKernel(N[i], 1.0, o_a, o_b, o_c);
      device.finish();
      timer::toc("triad");
      timer::update();
      double elapsed = timer::query("triad", "HOST:MAX") / Ntests;
      std::cout << "triad stream: "
                << N[i] << " words, "
                << elapsed << " s, "
                << bytes / elapsed << " bytes/s\n";
    }
    o_a.free();
    o_b.free();
    o_c.free();
/*
    occa::memory o_wrk = device.malloc(3 * nWords * sizeof(double));
    o_a = o_wrk + 0 * nWords * sizeof(double);
    o_b = o_wrk + 1 * nWords * sizeof(double);
    o_c = o_wrk + 2 * nWords * sizeof(double);
    for(int i = 0; i < Nsize; ++i) {
      long long int bytes = 3 * N[i] * sizeof(double);
      device.finish();
      timer::reset("triad");
      timer::tic("triad");
      for(int test = 0; test < Ntests; ++test) triadKernel(N[i], 1.0, o_a, o_b, o_c);
      device.finish();
      timer::toc("triad");
      timer::update();
      double elapsed = timer::query("triad", "HOST:MAX") / Ntests;
      std::cout << "triad stream subBuffer: "
                << N[i] << " words, "
                << elapsed << " s, "
                << bytes / elapsed << " bytes/s\n";
    }
    o_wrk.free();
*/
  }

  std::cout << "\n";

  {
    const int Ntests = 50;
    const int blockSize = 128;
    const int N = blockSize*1024*1024; 
    const int Nblock = N/blockSize; 
    occa::memory o_a = device.malloc(N * sizeof(double));
    occa::kernel smemKernel = device.buildKernel(DBP "kernel/bwKernels.okl", "smem", props);

    smemKernel(Nblock, N, o_a);

    device.finish();
    timer::reset("smem");
    timer::tic("smem");
    for(int test = 0; test < Ntests; ++test) smemKernel(Nblock, N, o_a);
    device.finish();
    timer::toc("smem");
    timer::update();
    double elapsed = timer::query("smem", "HOST:MAX") / (double)Ntests;
    long bytes = blockSize*2*sizeof(double)*N;
    std::cout << "shared mem bw: "
              << elapsed << " s, "
              << bytes / elapsed << " bytes/s\n";
    o_a.free();
  }

  std::cout << "\n";

  {
    const int N[] =
    {1, 2000, 4000, 8000, 16000, 50000, 100000, 150000, 300000, 2000 * 512, 4000 * 512, 8000 * 512};
    const int Nsize = sizeof(N) / sizeof(int);
    const int nWords = N[sizeof(N) / sizeof(N[0]) - 1];
    props["mapped"] = true;
    occa::memory h_u = device.malloc(nWords * sizeof(double), props);
    occa::memory o_a = device.malloc(nWords * sizeof(double));

    for(int i = 0; i < Nsize; ++i) {
      int Ntests = 5000;
      if(N[i] > 10000) Ntests = 100;
      const long long int bytes = N[i] * sizeof(double);
      void* ptr = h_u.ptr(props);
      device.finish();
      timer::reset("memcpyDH");
      timer::tic("memcpyDH");
      for(int test = 0; test < Ntests; ++test) o_a.copyTo(ptr, bytes);
      device.finish();
      timer::toc("memcpyDH");
      timer::update();
      const double elapsed = timer::query("memcpyDH", "HOST:MAX") / Ntests;
      std::cout << "D->H memcpy " <<  N[i] << " words, "
                << elapsed << " s, "
                << bytes / elapsed << " bytes/s\n";
    }

  std::cout << "\n";

    for(int i = 0; i < Nsize; ++i) {
      int Ntests = 5000;
      if(N[i] > 10000) Ntests = 100;
      const long long int bytes = N[i] * sizeof(double);
      void* ptr = h_u.ptr(props);
      device.finish();
      timer::reset("memcpyHD");
      timer::tic("memcpyHD");
      for(int test = 0; test < Ntests; ++test) o_a.copyFrom(ptr, bytes);
      device.finish();
      timer::toc("memcpyHD");
      timer::update();
      const double elapsed = timer::query("memcpyHD", "HOST:MAX") / Ntests;
      std::cout << "H->D memcpy " <<  N[i] << " words, "
                << elapsed << " s, "
                << bytes / elapsed << " bytes/s\n";
    }

    h_u.free();
    o_a.free();
  }

  std::cout << "\n";

  {
    const int Ntests = 20000;
    const int N[] = {1000 * 512, 2000 * 512, 4000 * 512, 8000 * 512};
    const int Nsize = sizeof(N) / sizeof(int);
    const int nWords = N[sizeof(N) / sizeof(N[0]) - 1];
    occa::memory o_a = device.malloc(nWords * sizeof(double));
    occa::memory o_b = device.malloc(nWords * sizeof(double));

    for(int i = 0; i < Nsize; ++i) {
      const long long int bytes = N[i] * sizeof(double);
      device.finish();
      timer::reset("memcpyDD");
      timer::tic("memcpyDD");
      for(int test = 0; test < Ntests; ++test) o_a.copyTo(o_b, bytes);
      device.finish();
      timer::toc("memcpyDD");
      timer::update();
      const double elapsed = timer::query("memcpyDD", "HOST:MAX") / Ntests;
      std::cout << "D->D memcpy " <<  N[i] << " words, "
                << elapsed << " s, "
                << 2 * bytes / elapsed << " bytes/s\n";
    }
    o_a.free();
    o_b.free();
  }

  MPI_Finalize();
  exit(0);
}
