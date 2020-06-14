#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <omp.h>

#include "mpi.h"
#include "params.h"
#include "mesh.h"
#include "timer.hpp"

#include "osu_multi_lat.h"

#include "gslib.h"
#include "ogs.hpp"
#include "ogsKernels.hpp"
#include "ogsInterface.h"

static void gsStart(occa::memory o_v, const char *type, const char *op, ogs_t *ogs); 
static void gsFinish(occa::memory o_v, const char *type, const char *op, ogs_t *ogs); 
static void printMeshPartitionStatistics(mesh_t *mesh);

#define ASYNC

int main(int argc, char **argv)
{
  // start up MPI
  MPI_Init(&argc, &argv);
  setupAide options;

  if(argc<6){
    printf("usage: ./gs N nelX nelY nelZ SERIAL|CUDA|HIP|OPENCL <nRepetitions> <run dummy kernel> <use FP32> <DEVICE-ID>\n");

    MPI_Finalize();
    exit(1);
  }

  const int N = atoi(argv[1]);
  const int NX = atoi(argv[2]);
  const int NY = atoi(argv[3]);
  const int NZ = atoi(argv[4]);
  std::string threadModel;
  threadModel.assign(strdup(argv[5]));
  options.setArgs("THREAD MODEL", threadModel);

  int Ntests = 100;
  if(argc>6) Ntests = atoi(argv[6]);

  int dummyKernel = 0;
  if(argc>7) dummyKernel = atoi(argv[7]);

  std::string floatType("double");
  if(argc>8 && atoi(argv[8])) floatType = "float";

  options.setArgs("DEVICE NUMBER", "LOCAL-RANK");
  if(argc>9) {
    std::string deviceNumber;
    deviceNumber.assign(strdup(argv[9]));
    options.setArgs("DEVICE NUMBER", deviceNumber);
  }

  options.setArgs("BOX NX", std::to_string(NX));
  options.setArgs("BOX NY", std::to_string(NY));
  options.setArgs("BOX NZ", std::to_string(NZ));

  options.setArgs("BOX XMIN", "-1.0");
  options.setArgs("BOX YMIN", "-1.0");
  options.setArgs("BOX ZMIN", "-1.0");
  options.setArgs("BOX XMAX", "1.0");
  options.setArgs("BOX YMAX", "1.0");
  options.setArgs("BOX ZMAX", "1.0");
  options.setArgs("MESH DIMENSION", "3");
  options.setArgs("BOX DOMAIN", "TRUE");

  options.setArgs("DISCRETIZATION", "CONTINUOUS");
  options.setArgs("ELEMENT MAP", "ISOPARAMETRIC");
  options.setArgs("ELEMENT TYPE", std::to_string(HEXAHEDRA));
  options.setArgs("POLYNOMIAL DEGREE", std::to_string(N));

  mesh_t *mesh = meshSetupBoxHex3D(N, 0, options);
  mesh->elementType = HEXAHEDRA;
  occa::properties kernelInfo;
  meshOccaSetup3D(mesh, options, kernelInfo);

  const int uDim = mesh->Nelements*mesh->Np;
  occa::memory o_U = mesh->device.malloc(uDim*sizeof(double));
  occa::kernel kernel;
  if(dummyKernel) {
    for (int r=0;r<2;r++){
      if ((r==0 && mesh->rank==0) || (r==1 && mesh->rank>0)) {
        kernel = mesh->device.buildKernel("gs_dummy.okl", "dummy", kernelInfo);
      }
      MPI_Barrier(MPI_COMM_WORLD);
    }
  }

  timer::init(mesh->comm, mesh->device, 1);

  if(mesh->rank == 0) cout << "\n";
  if(mesh->size > 1 && (mesh->size)%2 == 0) osu_multi_latency(0,argv);
  if(mesh->rank == 0) cout << "\n";

  const dlong Nlocal = mesh->Nelements*mesh->Np; 
  ogs_t *ogs= ogsSetup(Nlocal, mesh->globalIds, mesh->comm, 1, mesh->device);
  mesh->ogs = ogs; 
  meshPrintPartitionStatistics(mesh);

  occa::memory o_q = mesh->device.malloc(Nlocal*sizeof(dfloat));

  gsStart(o_q, floatType.c_str(), ogsAdd, ogs);
  gsFinish(o_q, floatType.c_str(), ogsAdd, ogs);
  timer::reset();

  if(mesh->rank == 0) cout << "starting measurement ...\n"; fflush(stdout);
  mesh->device.finish();
  MPI_Barrier(mesh->comm);
  const double start = MPI_Wtime();

  for(int test=0;test<Ntests;++test) {
    gsStart(o_q, floatType.c_str(), ogsAdd, ogs);
    if(dummyKernel) { 
      timer::tic("dummyKernel");
      kernel(uDim, o_U);
      timer::toc("dummyKernel");
    }
    gsFinish(o_q, floatType.c_str(), ogsAdd, ogs);
    timer::update();
   }

  mesh->device.finish();
  MPI_Barrier(mesh->comm);
  const double elapsed = (MPI_Wtime() - start)/Ntests;

  double etime[10];
  etime[0] = timer::query("gather_halo", "DEVICE:MAX")/Ntests;
  etime[1] = timer::query("gs_interior", "DEVICE:MAX")/Ntests;
  etime[2] = timer::query("scatter", "DEVICE:MAX")/Ntests;
  etime[3] = timer::query("gs_host", "HOST:MAX")/Ntests;
  etime[4] = timer::query("gs_memcpy", "DEVICE:MAX")/Ntests;
  etime[5] = timer::query("dummyKernel", "DEVICE:MAX")/Ntests;
  if (etime[5] < 0) etime[5] = 0;

  if(mesh->rank==0){
    int Nthreads =  omp_get_max_threads();
    cout << "\nsummary\n"
         << "  MPItasks             : " << mesh->size << "\n";

    if(options.compareArgs("THREAD MODEL", "OPENMP"))
      cout << "  OMPthreads           : " << Nthreads << "\n";

    cout << "  polyN                : " << N << "\n"
         << "  Nelements            : " << NX*NY*NZ << "\n"
         << "  Nrepetitions         : " << Ntests << "\n"
         << "  floatType            : " << floatType << "\n" 
         << "  avg elapsed time     : " << elapsed << " s\n"
         << "    gather halo        : " << etime[0] << " s\n"
         << "    gs interior        : " << etime[1] << " s\n"
         << "    scatter            : " << etime[2] << " s\n"
         << "    gslib (host)       : " << etime[3] << " s\n"
         << "    memcpy halo        : " << etime[4] << " s\n"
         << "    dummy kernel       : " << etime[5] << " s\n"
         << "  throughput           : " << ((dfloat)(NX*NY*NZ)*N*N*N/elapsed)/1.e9 << " GDOF/s\n";
  }

  MPI_Finalize();
  return 0;
}

void gsStart(occa::memory o_v, const char *type, const char *op, ogs_t *ogs) 
{
  size_t Nbytes;
  if (!strcmp(type, "float")) 
    Nbytes = sizeof(float);
  else if (!strcmp(type, "double")) 
    Nbytes = sizeof(double);
  else if (!strcmp(type, "int")) 
    Nbytes = sizeof(int);
  else if (!strcmp(type, "long long int")) 
    Nbytes = sizeof(long long int);

  if (ogs->NhaloGather) {
    if (ogs::o_haloBuf.size() < ogs->NhaloGather*Nbytes) {
      if (ogs::o_haloBuf.size()) ogs::o_haloBuf.free();

      occa::properties props;
      props["mapped"] = true;
      ogs::o_haloBuf = ogs->device.malloc(ogs->NhaloGather*Nbytes, props);
      ogs::haloBuf = ogs::o_haloBuf.ptr();
    }
  }

  if (ogs->NhaloGather) {
    timer::tic("gather_halo");
    occaGather(ogs->NhaloGather, ogs->o_haloGatherOffsets, ogs->o_haloGatherIds, type, op, o_v, ogs::o_haloBuf);
    timer::toc("gather_halo");

#ifdef ASYNC
    ogs->device.finish();
    ogs->device.setStream(ogs::dataStream);
    timer::tic("gs_memcpy");
    ogs::o_haloBuf.copyTo(ogs::haloBuf, ogs->NhaloGather*Nbytes, 0, "async: true");
    timer::toc("gs_memcpy");
    ogs->device.setStream(ogs::defaultStream);
#else    
    timer::tic("gs_memcpy");
    ogs::o_haloBuf.copyTo(ogs::haloBuf, ogs->NhaloGather*Nbytes);
    timer::toc("gs_memcpy");
#endif
  }
}

void gsFinish(occa::memory o_v, const char *type, const char *op, ogs_t *ogs) 
{
  size_t Nbytes;
  if (!strcmp(type, "float")) 
    Nbytes = sizeof(float);
  else if (!strcmp(type, "double")) 
    Nbytes = sizeof(double);
  else if (!strcmp(type, "int")) 
    Nbytes = sizeof(int);
  else if (!strcmp(type, "long long int")) 
    Nbytes = sizeof(long long int);

  if(ogs->NlocalGather) {
    timer::tic("gs_interior");
    occaGatherScatter(ogs->NlocalGather, ogs->o_localGatherOffsets, ogs->o_localGatherIds, type, op, o_v);
    timer::toc("gs_interior");
  }

  if (ogs->NhaloGather) {
#ifdef ASYNC
    ogs->device.setStream(ogs::dataStream);
    //timer::tic("gs_memcpy");
    ogs->device.finish();
    //timer::toc("gs_memcpy");
#endif

    timer::tic("gs_host");
    ogsHostGatherScatter(ogs::haloBuf, type, op, ogs->haloGshSym);
    timer::toc("gs_host");

#ifdef ASYNC
    timer::tic("gs_memcpy");
    ogs::o_haloBuf.copyFrom(ogs::haloBuf, ogs->NhaloGather*Nbytes, 0, "async: true");
    timer::toc("gs_memcpy");
#else    
    timer::tic("gs_memcpy");
    ogs::o_haloBuf.copyFrom(ogs::haloBuf, ogs->NhaloGather*Nbytes);
    timer::toc("gs_memcpy");
#endif

    timer::tic("scatter");
    occaScatter(ogs->NhaloGather, ogs->o_haloGatherOffsets, ogs->o_haloGatherIds, type, op, ogs::o_haloBuf, o_v);
    timer::toc("scatter");

#ifdef ASYNC
    //timer::tic("gs_memcpy");
    ogs->device.finish();
    //timer::toc("gs_memcpy");
    ogs->device.setStream(ogs::defaultStream);
#endif    
  }
}

