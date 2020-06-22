#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <omp.h>

#include "mpi.h"
#include "params.h"
#include "mesh.h"
#include "timer.hpp"

#include "ping_pong.h"

#include "mygs.h"


int main(int argc, char **argv)
{
  MPI_Init(&argc, &argv);
  setupAide options;

  if(argc<6){
    printf("usage: ./gs N nelX nelY nelZ SERIAL|CUDA|HIP|OPENCL <nRepetitions> <run dummy kernel> <use FP32> <GPU aware MPI> <DEVICE-ID>\n");

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

  int enabledGPUMPI = 0;
  if(argc>9) {
    if(argv[9]) enabledGPUMPI = 1;
  }

  options.setArgs("DEVICE NUMBER", "LOCAL-RANK");
  if(argc>10) {
    std::string deviceNumber;
    deviceNumber.assign(strdup(argv[10]));
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

  // setup mesh
  mesh_t *mesh = meshSetupBoxHex3D(N, 0, options);
  mesh->elementType = HEXAHEDRA;
  occa::properties kernelInfo;
  meshOccaSetup3D(mesh, options, kernelInfo);

  // setup dummy kernel
  const int uDim = mesh->Nelements*mesh->Np;
  double *U = (double*) calloc(uDim, sizeof(double));
  occa::memory o_U = mesh->device.malloc(uDim*sizeof(double), U);
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

  // setup gs
  const dlong Nlocal = mesh->Nelements*mesh->Np; 
  ogs_t *ogs= ogsSetup(Nlocal, mesh->globalIds, mesh->comm, 1, mesh->device);
  mesh->ogs = ogs; 

  meshPrintPartitionStatistics(mesh);

  occa::memory o_q = mesh->device.malloc(Nlocal*sizeof(dfloat));
  for(int i=0; i<Nlocal; i++) U[i] = 1;
  o_q.copyFrom(U, Nlocal*sizeof(double));
  gsStart(o_q, floatType.c_str(), ogsAdd, ogs);
  gsFinish(o_q, floatType.c_str(), ogsAdd, ogs);
  o_q.copyTo(U, Nlocal*sizeof(double));
  for(int i=0; i<Nlocal; i++) U[i] = 1/U[i];

  o_q.copyFrom(U, Nlocal*sizeof(double));
  gsStart(o_q, floatType.c_str(), ogsAdd, ogs);
  gsFinish(o_q, floatType.c_str(), ogsAdd, ogs);
  o_q.copyTo(U, Nlocal*sizeof(double));
  double nPts = 0;
  for(int i=0; i<Nlocal; i++) nPts += U[i];
  MPI_Allreduce(MPI_IN_PLACE,&nPts,1,MPI_DOUBLE,MPI_SUM,mesh->comm);
  if(nPts == NX*NY*NZ * (double) mesh->Np) { 
    if(mesh->rank == 0) printf("\nverfication test passed!\n");
  } else {
    if(mesh->rank == 0) printf("\nverfication test failed!\n");
    MPI_Abort(mesh->comm, 1);
  }
 
  if(mesh->rank == 0) cout << "\nstarting measurement ...\n"; fflush(stdout);

  // ping pong
  timer::reset();
  mesh->device.finish();
  MPI_Barrier(mesh->comm);
  {
    const int nPairs = mesh->size/2;
    pingPongMulti(nPairs, 0, mesh->device, mesh->comm);
    if(enabledGPUMPI) pingPongMulti(nPairs, enabledGPUMPI, mesh->device, mesh->comm);
  }

  // gs
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

  // print stats 
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
