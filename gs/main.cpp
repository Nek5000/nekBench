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

#include "ogs.hpp"
#include "ogsKernels.hpp"
#include "ogsInterface.h"

static void gs(occa::memory o_v, const char *type, const char *op, ogs_t *ogs); 
static void printMeshPartitionStatistics(mesh_t *mesh);

int main(int argc, char **argv){

  // start up MPI
  MPI_Init(&argc, &argv);
  setupAide options;

  if(argc<6){
    printf("usage: ./gs N nelX nelY nelZ SERIAL|CUDA|HIP|OPENCL <nRepetitions> <FP32> <DEVICE-ID>\n");

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

  int Ntests = 1;
  if(argc>6) Ntests = atoi(argv[6]);

  std::string floatType("double");
  if(argc>7 && atoi(argv[7])) floatType = "float";

  options.setArgs("DEVICE NUMBER", "LOCAL-RANK");
  if(argc>8) {
    std::string deviceNumber;
    deviceNumber.assign(strdup(argv[8]));
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
  printMeshPartitionStatistics(mesh);
  mesh->elementType = HEXAHEDRA;
  occa::properties kernelInfo;
  meshOccaSetup3D(mesh, options, kernelInfo);

  timer::init(mesh->comm, mesh->device, 1);

  if(mesh->rank == 0) cout << "\n";
  if(mesh->size > 1) osu_multi_latency(0,argv);
  if(mesh->rank == 0) cout << "\n";

  const dlong Nlocal = mesh->Nelements*mesh->Np; 
  ogs_t *ogs= ogsSetup(Nlocal, mesh->globalIds, mesh->comm, 1, mesh->device); 

  occa::memory o_q = mesh->device.malloc(Nlocal*sizeof(dfloat));

  gs(o_q, floatType.c_str(), ogsAdd, ogs);
  timer::reset();

  if(mesh->rank == 0) cout << "starting measurement ...\n"; fflush(stdout);
  mesh->device.finish();
  MPI_Barrier(mesh->comm);
  const double start = MPI_Wtime();
  for(int test=0;test<Ntests;++test)
    gs(o_q, floatType.c_str(), ogsAdd, ogs);
  mesh->device.finish();
  MPI_Barrier(mesh->comm);
  const double elapsed = (MPI_Wtime() - start)/Ntests;

  double etime[10];
  etime[0] = timer::query("gs_local", "DEVICE:MAX")/Ntests;
  etime[1] = timer::query("gs_host", "HOST:MAX")/Ntests;
  etime[2] = timer::query("gs_memcpy", "DEVICE:MAX")/Ntests;

  if(mesh->rank==0){
    int Nthreads =  omp_get_max_threads();
    cout << "\nsummary\n"
         << "  MPItasks         : " << mesh->size << "\n";
    if(options.compareArgs("THREAD MODEL", "OPENMP"))
      cout << "  OMPthreads       : " << Nthreads << "\n";
    cout << "  polyN            : " << N << "\n"
         << "  Nelements        : " << NX*NY*NZ << "\n"
         << "  Nrepetitions     : " << Ntests << "\n"
         << "  avg elapsed time : " << elapsed << " s\n"
         << "    gs_local       : " << etime[0] << " s\n"
         << "    gs_host        : " << etime[1] << " s\n"
         << "    gs_memcpy      : " << etime[2] << " s\n"
         << "  throughput       : " << ((dfloat)(NX*NY*NZ)*N*N*N/elapsed)/1.e9 << " GDOF/s\n";
  }

  MPI_Finalize();
  return 0;
}

void printMeshPartitionStatistics(mesh_t *mesh){

  /* get MPI rank and size */
  int rank, size;
  rank = mesh->rank;
  size = mesh->size;
  
  /* now gather statistics on connectivity between processes */
  int *comms = (int*) calloc(size, sizeof(int));
  int Ncomms = 0;
  
  /* count elements with neighbors on each other rank ranks */
  for(dlong e=0;e<mesh->Nelements;++e){
    for(int f=0;f<mesh->Nfaces;++f){
      if(mesh->EToP[e*mesh->Nfaces+f]!=-1){
        ++comms[mesh->EToP[e*mesh->Nfaces+f]];
        ++Ncomms;
      } 
    } 
  } 
  
  int Nmessages = 0;
  for(int r=0;r<size;++r)
    if(comms[r]>0)
      ++Nmessages;
      
  for(int r=0;r<size;++r){
    MPI_Barrier(mesh->comm);
    if(r==rank){
      fflush(stdout);
      printf("r: %02d [", rank);
      for(int s=0;s<size;++s){
        printf(" %04d", comms[s]);
      }
      printf("] (Nelements=" dlongFormat ", Nmessages=%d, Nfaces=%d)\n", mesh->Nelements,Nmessages, Ncomms);
      fflush(stdout);
    } 
  } 
  
  free(comms);
}

//#define ASYNC
void gs(occa::memory o_v, const char *type, const char *op, ogs_t *ogs) 
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
    timer::tic("gs_local");
    occaGather(ogs->NhaloGather, ogs->o_haloGatherOffsets, ogs->o_haloGatherIds, type, op, o_v, ogs::o_haloBuf);
    timer::toc("gs_local");

#ifdef ASYNC
    ogs->device.finish();
    ogs->device.setStream(ogs::dataStream);
    ogs::o_haloBuf.copyTo(ogs::haloBuf, ogs->NhaloGather*Nbytes, 0, "async: true");
    ogs->device.setStream(ogs::defaultStream);
#else    
    timer::tic("gs_memcpy");
    ogs::o_haloBuf.copyTo(ogs::haloBuf, ogs->NhaloGather*Nbytes);
    timer::toc("gs_memcpy");
#endif
  }

  if(ogs->NlocalGather) {
    timer::tic("gs_local");
    occaGatherScatter(ogs->NlocalGather, ogs->o_localGatherOffsets, ogs->o_localGatherIds, type, op, o_v);
    timer::toc("gs_local");
  }

  if (ogs->NhaloGather) {
#ifdef ASYNC
    ogs->device.setStream(ogs::dataStream);
    ogs->device.finish();
#endif

    timer::tic("gs_host");
    ogsHostGatherScatter(ogs::haloBuf, type, op, ogs->haloGshSym);
    timer::toc("gs_host");

#ifdef ASYNC
    ogs::o_haloBuf.copyFrom(ogs::haloBuf, ogs->NhaloGather*Nbytes, 0, "async: true");
#else    
    timer::tic("gs_memcpy");
    ogs::o_haloBuf.copyFrom(ogs::haloBuf, ogs->NhaloGather*Nbytes);
    timer::toc("gs_memcpy");
#endif

    timer::tic("gs_local");
    occaScatter(ogs->NhaloGather, ogs->o_haloGatherOffsets, ogs->o_haloGatherIds, type, op, ogs::o_haloBuf, o_v);
    timer::toc("gs_local");

#ifdef ASYNC
    ogs->device.finish();
    ogs->device.setStream(ogs::defaultStream);
#endif    
  }
}

