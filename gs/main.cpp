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

static void printMeshPartitionStatistics(mesh_t *mesh);

int main(int argc, char **argv){

  // start up MPI
  MPI_Init(&argc, &argv);
  setupAide options;

  if(argc!=7){
    printf("usage: ./gs N nelX nelY nelZ SERIAL|CUDA|HIP|OPENCL DEVICE-ID\n");

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

  std::string deviceNumber;
  deviceNumber.assign(strdup(argv[6]));
  options.setArgs("DEVICE NUMBER", deviceNumber);

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

  if(mesh->rank == 0) cout << "\n";
  osu_multi_latency(0,argv);
  if(mesh->rank == 0) cout << "\n";

  const dlong Nlocal = mesh->Nelements*mesh->Np; 
  ogs_t *ogs= ogsSetup(Nlocal, mesh->globalIds, mesh->comm, 1, mesh->device); 

  occa::memory o_q = mesh->device.malloc(Nlocal*sizeof(dfloat));
  const int Ntests = 10;

  ogsGatherScatter(o_q, ogsDfloat, ogsAdd, ogs);
  mesh->device.finish();
  MPI_Barrier(MPI_COMM_WORLD);
  const double start = MPI_Wtime();

  for(int test=0;test<Ntests;++test)
    ogsGatherScatter(o_q, ogsDfloat, ogsAdd, ogs);

  mesh->device.finish();
  MPI_Barrier(MPI_COMM_WORLD);
  const double elapsed = MPI_Wtime() - start;

  if(mesh->rank==0){
    int Nthreads =  omp_get_max_threads();
    cout << "\nsummary\n"
         << "  MPItasks         : " << mesh->size << "\n";
    if(options.compareArgs("THREAD MODEL", "OPENMP"))
      cout << "  OMPthreads       : " << Nthreads << "\n";
    cout << "  polyN            : " << N << "\n"
         << "  Nelements        : " << NX*NY*NZ << "\n"
         << "  avg elapsed time : " << elapsed/Ntests << " s\n"
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
      printf("] (Nelements=" dlongFormat ", Nmessages=%d, Ncomms=%d)\n", mesh->Nelements,Nmessages, Ncomms);
      fflush(stdout);
    } 
  } 
  
  free(comms);
}
