#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <omp.h>
#include <list>

#include "mpi.h"
#include "params.h"
#include "mesh.h"
#include "timer.hpp"

#include "ping_pong.h"

#include "mygs.h"


int main(int argc, char **argv)
{
  MPI_Init(&argc, &argv);
  int rank;
  MPI_Comm_rank (MPI_COMM_WORLD, &rank);
  setupAide options;

  if(argc<6){
    if(rank ==0) printf("usage: ./gs N nelX nelY nelZ SERIAL|CUDA|HIP|OPENCL <ogs_mode> <nRepetitions> <run dummy kernel> <use FP32> <GPU aware MPI> <DEVICE-ID>\n");

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

  std::list<ogs_mode> ogs_mode_list;
  if(argc>6 && atoi(argv[6])>0) {
    const int mode = atoi(argv[6]);
    if(mode < 0 || mode > 4){
      if(rank == 0) printf("invalid ogs_mode!\n");
      MPI_Abort(MPI_COMM_WORLD,1);
    }
    if(mode) ogs_mode_list.push_back((ogs_mode)(mode-1)); 
  } else {
    ogs_mode_list.push_back(OGS_DEFAULT); 
    ogs_mode_list.push_back(OGS_HOSTMPI); 
    ogs_mode_list.push_back(OGS_DEVICEMPI); 
  }

  int Ntests = 100;
  if(argc>7) Ntests = atoi(argv[7]);

  int dummyKernel = 0;
  if(argc>8) dummyKernel = atoi(argv[8]);

  int unit_size = 8;
  std::string floatType("double");
  if(argc>9 && atoi(argv[9])) {
    floatType = "float";
    unit_size = 4;
    if(rank == 0) printf("FP32 unsupported!\n");
    MPI_Abort(MPI_COMM_WORLD,1);
  }

  int enabledGPUMPI = 0;
  if(argc>10) {
    if(argv[10]) enabledGPUMPI = 1;
  }

  options.setArgs("DEVICE NUMBER", "LOCAL-RANK");
  if(argc>11) {
    std::string deviceNumber;
    deviceNumber.assign(strdup(argv[11]));
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
  mygsSetup(ogs);
  mesh->ogs = ogs; 

  meshPrintPartitionStatistics(mesh);

  double *U = (double*) calloc(Nlocal, sizeof(double));
  occa::memory o_U = mesh->device.malloc(Nlocal*sizeof(double), U);

  double *Q = (double*) calloc(Nlocal, unit_size);
  occa::memory o_q = mesh->device.malloc(Nlocal*unit_size, Q);

  // warm-up + check for correctness
  for (auto const& ogs_mode_enum : ogs_mode_list) {
    for(int i=0; i<Nlocal; i++) Q[i] = 1;
    o_q.copyFrom(Q, Nlocal*unit_size);
    mygsStart(o_q, floatType.c_str(), ogsAdd, ogs, ogs_mode_enum);
    mygsFinish(o_q, floatType.c_str(), ogsAdd, ogs, ogs_mode_enum);
    o_q.copyTo(Q, Nlocal*unit_size);
    for(int i=0; i<Nlocal; i++) Q[i] = 1/Q[i];
 
    o_q.copyFrom(Q, Nlocal*unit_size);
    mygsStart(o_q, floatType.c_str(), ogsAdd, ogs, ogs_mode_enum);
    mygsFinish(o_q, floatType.c_str(), ogsAdd, ogs, ogs_mode_enum);
    o_q.copyTo(Q, Nlocal*unit_size);
    double nPts = 0;
    for(int i=0; i<Nlocal; i++) nPts += Q[i];
    MPI_Allreduce(MPI_IN_PLACE,&nPts,1,MPI_DOUBLE,MPI_SUM,mesh->comm);
    if(fabs(nPts - NX*NY*NZ*(double)mesh->Np) > 1e-6) { 
      if(mesh->rank == 0) printf("\ncorrectness check failed for mode=%d! %ld\n", ogs_mode_enum, (long long int)nPts);
      fflush(stdout);
      MPI_Abort(mesh->comm, 1);
    }
  }

  if(mesh->rank == 0) cout << "\nstarting measurement ...\n"; fflush(stdout);

  // ping pong
  mesh->device.finish();
  MPI_Barrier(mesh->comm);
  {
    const int nPairs = mesh->size/2;
    pingPongMulti(nPairs, 0, mesh->device, mesh->comm);
    if(enabledGPUMPI) pingPongMulti(nPairs, enabledGPUMPI, mesh->device, mesh->comm);
  }

  for (auto const& ogs_mode_enum : ogs_mode_list) { 

  // gs
  timer::reset();
  mesh->device.finish();
  MPI_Barrier(mesh->comm);
  const double start = MPI_Wtime();
  for(int test=0;test<Ntests;++test) {
    mygsStart(o_q, floatType.c_str(), ogsAdd, ogs, ogs_mode_enum);
    if(dummyKernel) { 
      timer::tic("dummyKernel");
      kernel(mesh->Nelements*mesh->Np, o_U);
      timer::toc("dummyKernel");
    }
    mygsFinish(o_q, floatType.c_str(), ogsAdd, ogs, ogs_mode_enum);
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
  etime[4] = timer::query("gs_memcpy_dh", "DEVICE:MAX")/Ntests;
  etime[5] = timer::query("gs_memcpy_hd", "DEVICE:MAX")/Ntests;
  etime[6] = timer::query("dummyKernel", "DEVICE:MAX")/Ntests;
  etime[7] = timer::query("pw_exec", "HOST:MAX")/Ntests;
  etime[8] = timer::query("pack", "DEVICE:MAX")/Ntests;
  etime[9] = timer::query("unpack", "DEVICE:MAX")/Ntests;
  if(etime[6] < 0) etime[6] = 0;

  if(mesh->rank==0){
    int Nthreads =  omp_get_max_threads();
    cout << "\nsummary\n"
         << "  ogsMode                 : " << ogs_mode_enum << "\n"
         << "  MPItasks                : " << mesh->size << "\n";

    if(options.compareArgs("THREAD MODEL", "OPENMP"))
      cout << "  OMPthreads              : " << Nthreads << "\n";

    cout << "  polyN                   : " << N << "\n"
         << "  Nelements               : " << NX*NY*NZ << "\n"
         << "  Nrepetitions            : " << Ntests << "\n"
         << "  floatType               : " << floatType << "\n" 
         << "  avg elapsed time        : " << elapsed << " s\n"
         << "    gather halo           : " << etime[0] << " s\n"
         << "    gs interior           : " << etime[1] << " s\n"
         << "    scatter halo          : " << etime[2] << " s\n"
         << "    pw exec               : " << etime[7] << " s\n"
         << "    memcpy host<->device  : " << etime[4] + etime[5] << " s\n"
         << "    pack/unpack buf       : " << etime[8] + etime[9] << " s\n"
         << "    dummy kernel          : " << etime[6] << " s\n"
         << "  throughput              : " << ((double)(NX*NY*NZ)*N*N*N/elapsed)/1.e9 << " GDOF/s\n";
  }

  }
  MPI_Finalize();
  return 0;
}
