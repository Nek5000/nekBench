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

#include <unistd.h>
#include "BP.hpp"
#include "../axhelm/kernelHelper.cpp"
#include "timer.hpp"

static occa::memory p_tmp;

void reportMemoryUsage(occa::device &device, const char* mess);

BP_t* setup(mesh_t* mesh, occa::properties &kernelInfo, setupAide &options)
{
  BP_t* BP = new BP_t();

  BP->BPid = 0;
  BP->Nfields = 1;
  if(options.compareArgs("BPMODE", "TRUE")) {
    if(mesh->rank == 0) printf("BP mode enabled\n");
    options.setArgs("PRECONDITIONER", "COPY");
    BP->BPid = 5;
    //options.getArgs("NUMBER OF FIELDS", BP->Nfields);
    //if(BP->Nfields == 3) BP->BPid = 6;
  }

  BP->lambda1 = 1.1;
  options.getArgs("LAMBDA", BP->lambda1);
  if(BP->BPid) BP->lambda1 = 0.0;

  BP->overlap = true;
  if(options.compareArgs("OVERLAP", "FALSE")) BP->overlap = false;
  if(options.compareArgs("THREAD MODEL", "SERIAL")) BP->overlap = false;
  if(options.compareArgs("THREAD MODEL", "OPENMP")) BP->overlap = false;
  if(mesh->size == 1) BP->overlap = false;
  if(BP->overlap) {
    if(mesh->rank == 0) printf("overlap enabled\n");
  } else {
    if(mesh->rank == 0) printf("overlap disabled\n");
  }

  options.getArgs("MESH DIMENSION", BP->dim);
  options.getArgs("ELEMENT TYPE", BP->elementType);
  BP->mesh = mesh;
  BP->options = options;

  solveSetup(BP, kernelInfo);

  const dlong Ndof = mesh->Np * (mesh->Nelements + mesh->totalHaloPairs);
  BP->fieldOffset = Ndof;
  const dlong Nall = BP->Nfields * Ndof;

  BP->r   = (dfloat*) calloc(Nall,   sizeof(dfloat));
  BP->x   = (dfloat*) calloc(Nall,   sizeof(dfloat));
  BP->q   = (dfloat*) calloc(Nall,   sizeof(dfloat));

  // setup RHS
  for(dlong e = 0; e < mesh->Nelements; ++e)
    for(int n = 0; n < mesh->Np; ++n) {
      dfloat JW = mesh->ggeo[mesh->Np * (e * mesh->Nggeo + GWJID) + n];

      dlong id = n + e * mesh->Np;
      dfloat xn = mesh->x[id];
      dfloat yn = mesh->y[id];
      dfloat zn = mesh->z[id];

      dfloat mode = 1;

      for(int fld = 0; fld < BP->Nfields; ++fld) {
        dlong fldid = id + fld * Ndof;

        // mass projection rhs
        BP->r[fldid] =
          (3. * M_PI * M_PI * mode * mode + BP->lambda1) * JW * cos(mode * M_PI * xn) * cos(
            mode * M_PI * yn) * cos(mode * M_PI * zn);

        BP->x[fldid] = 0;
      }
    }

  BP->o_r = mesh->device.malloc(Nall * sizeof(dfloat), BP->r);
  BP->o_x = mesh->device.malloc(Nall * sizeof(dfloat), BP->x);

  BP->lambda = (dfloat*) calloc(2 * Nall, sizeof(dfloat));
  for(int i = 0; i < BP->fieldOffset; i++) {
    BP->lambda[i]        = 1.0; // don't change
    BP->lambda[i + BP->fieldOffset] = BP->lambda1;
  }
  BP->o_lambda = mesh->device.malloc(2 * Nall * sizeof(dfloat), BP->lambda);

  char* suffix = strdup("Hex3D");

  if(options.compareArgs("DISCRETIZATION","CONTINUOUS"))
    ogsGatherScatterMany(BP->o_r, BP->Nfields, Ndof, ogsDfloat, ogsAdd, mesh->ogs);

  if (mesh->rank == 0)
    reportMemoryUsage(mesh->device, "setup done");

  if (options.compareArgs("VERBOSE", "TRUE")) {
    fflush(stdout);
    MPI_Barrier(mesh->comm);
    printf("rank %d has %d internal elements and %d non-internal elements\n",
           mesh->rank,
           mesh->NinternalElements,
           mesh->NnotInternalElements);
  }

  BP->profiling = 0;
  if(options.compareArgs("PROFILING", "TRUE")) BP->profiling = 1;
  int sync = 0;
  if(options.compareArgs("TIMER SYNC", "TRUE")) sync = 1;
  if(BP->profiling) timer::init(MPI_COMM_WORLD, mesh->device, sync);

  return BP;
}

void solveSetup(BP_t* BP, occa::properties &kernelInfo)
{
  mesh_t* mesh = BP->mesh;
  setupAide options = BP->options;

  int knlId = 0;
  options.getArgs("KERNEL ID", knlId);
  BP->knlId = knlId;

  dlong Ntotal = mesh->Np * mesh->Nelements;
  dlong Nhalo  = mesh->Np * mesh->totalHaloPairs;
  dlong Nall   = (Ntotal + Nhalo) * BP->Nfields;

  dlong Nblock  = mymax(1,(Ntotal + blockSize - 1) / blockSize);
  dlong Nblock2 = mymax(1,(Nblock + blockSize - 1) / blockSize);

  dlong NthreadsUpdatePCG = blockSize;
  //dlong NblocksUpdatePCG = mymin((Ntotal+NthreadsUpdatePCG-1)/NthreadsUpdatePCG, 32);
  dlong NblocksUpdatePCG = (Ntotal + NthreadsUpdatePCG - 1) / NthreadsUpdatePCG;

  BP->NthreadsUpdatePCG = NthreadsUpdatePCG;
  BP->NblocksUpdatePCG = NblocksUpdatePCG;

  BP->NsolveWorkspace = 4;
  BP->offsetSolveWorkspace = Nall;

/*
   const int PAGESIZE = 512; // in bytes
   const int pageW = PAGESIZE/sizeof(dfloat);
   if (BP->offsetSolveWorkspace%pageW) BP->offsetSolveWorkspace = (BP->offsetSolveWorkspace + 1)*pageW;
   dfloat *scratch = (dfloat*) calloc(BP->offsetSolveWorkspace*BP->NsolveWorkspace, sizeof(dfloat));
   occa::memory o_scratch = mesh->device.malloc(BP->offsetSolveWorkspace*BP->NsolveWorkspace*sizeof(dfloat), scratch);
 */

  BP->o_solveWorkspace = new occa::memory[BP->NsolveWorkspace];
  for(int wk = 0; wk < BP->NsolveWorkspace; ++wk) {
    BP->o_solveWorkspace[wk] = mesh->device.malloc(Nall * sizeof(dfloat), BP->solveWorkspace);
    //BP->o_solveWorkspace[wk] = o_scratch.slice(wk*BP->offsetSolveWorkspace*sizeof(dfloat));
  }

  if(options.compareArgs("PRECONDITIONER", "JACOBI")) {
    BP->o_invDiagA = mesh->device.malloc(Nall * sizeof(dfloat));
    int* mapB = (int*) calloc(Nall, sizeof(int));
    BP->o_mapB = mesh->device.malloc(Nall * sizeof(int), mapB);
    free(mapB);
  }

  occa::properties props = kernelInfo;
  props["mapped"] = true;
  p_tmp = mesh->device.malloc(Nblock * sizeof(dfloat), props);
  BP->tmp  = (dfloat*)p_tmp.ptr(props);
  BP->o_tmp = mesh->device.malloc(Nblock * sizeof(dfloat), BP->tmp);
  BP->o_tmp2 = mesh->device.malloc(Nblock2 * sizeof(dfloat), BP->tmp);

  BP->tmpNormr = (dfloat*) calloc(BP->NblocksUpdatePCG,sizeof(dfloat));
  BP->o_tmpNormr = mesh->device.malloc(BP->NblocksUpdatePCG * sizeof(dfloat), BP->tmpNormr);

  BP->tmpAtomic = (dfloat*) calloc(1,sizeof(dfloat));
  BP->o_tmpAtomic = mesh->device.malloc(1 * sizeof(dfloat), BP->tmpAtomic);
  BP->o_zeroAtomic = mesh->device.malloc(1 * sizeof(dfloat), BP->tmpAtomic);

  //setup async halo stream
  BP->defaultStream = mesh->defaultStream;
  BP->dataStream = mesh->dataStream;

  dlong Nbytes = BP->Nfields * mesh->totalHaloPairs * mesh->Np * sizeof(dfloat);
  if(Nbytes > 0) {
    BP->sendBuffer =
      (dfloat*) occaHostMallocPinned(mesh->device, Nbytes, NULL, BP->o_sendBuffer,
                                     BP->h_sendBuffer);
    BP->recvBuffer =
      (dfloat*) occaHostMallocPinned(mesh->device, Nbytes, NULL, BP->o_recvBuffer,
                                     BP->h_recvBuffer);
  }else{
    BP->sendBuffer = NULL;
    BP->recvBuffer = NULL;
  }
  mesh->device.setStream(BP->defaultStream);

  BP->type = strdup(dfloatString);

  BP->Nblock = Nblock;
  BP->Nblock2 = Nblock2;

  // count total number of elements
  hlong NelementsLocal = mesh->Nelements;
  hlong NelementsGlobal = 0;

  MPI_Allreduce(&NelementsLocal, &NelementsGlobal, 1, MPI_HLONG, MPI_SUM, mesh->comm);

  BP->NelementsGlobal = NelementsGlobal;

  //check all the bounaries for a Dirichlet
  bool allNeumann = (BP->lambda1 == 0) ? true :false;
  BP->allNeumannPenalty = 1.;
  hlong localElements = (hlong) mesh->Nelements;
  hlong totalElements = 0;
  MPI_Allreduce(&localElements, &totalElements, 1, MPI_HLONG, MPI_SUM, mesh->comm);
  BP->allNeumannScale = 1. / sqrt((dfloat)mesh->Np * totalElements);

  BP->EToB = (int*) calloc(mesh->Nelements * mesh->Nfaces,sizeof(int));

  int lallNeumann, gallNeumann;
  lallNeumann = allNeumann ? 0:1;
  MPI_Allreduce(&lallNeumann, &gallNeumann, 1, MPI_INT, MPI_SUM, mesh->comm);
  BP->allNeumann = (gallNeumann > 0) ? false: true;
  //printf("allNeumann = %d \n", BP->allNeumann);

  //copy boundary flags
  BP->o_EToB = mesh->device.malloc(mesh->Nelements * mesh->Nfaces * sizeof(int), BP->EToB);

  //setup an unmasked gs handle
  meshParallelGatherScatterSetup(mesh, Ntotal, mesh->globalIds, mesh->comm, 0);

  //make a masked version of the global id numbering
  mesh->maskedGlobalIds = (hlong*) calloc(Ntotal,sizeof(hlong));
  memcpy(mesh->maskedGlobalIds, mesh->globalIds, Ntotal * sizeof(hlong));

  kernelInfo["defines/p_Nalign"] = USE_OCCA_MEM_BYTE_ALIGN;
  kernelInfo["defines/" "p_blockSize"] = blockSize;
  kernelInfo["defines/" "p_Nfields"] = BP->Nfields;

  string threadModel;
  options.getArgs("THREAD MODEL", threadModel);
  string arch = "VOLTA";
  options.getArgs("ARCH", arch);
  for (int r = 0; r < 2; r++) {
    if ((r == 0 && mesh->rank == 0) || (r == 1 && mesh->rank > 0)) {
      mesh->addScalarKernel =
        mesh->device.buildKernel(DBP "/kernel/utils.okl",
                                 "addScalar",
                                 kernelInfo);

      mesh->maskKernel =
        mesh->device.buildKernel(DBP "/kernel/utils.okl",
                                 "mask",
                                 kernelInfo);

      mesh->sumKernel =
        mesh->device.buildKernel(DBP "/kernel/utils.okl",
                                 "sum",
                                 kernelInfo);

      BP->innerProductKernel =
        mesh->device.buildKernel(DBP "/kernel/utils.okl", "innerProduct", kernelInfo);

      BP->weightedNorm2Kernel =
        mesh->device.buildKernel(DBP "/kernel/utils.okl", "weightedNorm2", kernelInfo);

      BP->weightedMultipleNorm2Kernel =
        mesh->device.buildKernel(DBP "/kernel/utils.okl", "weightedMultipleNorm2", kernelInfo);

      BP->scaledAddKernel =
        mesh->device.buildKernel(DBP "/kernel/utils.okl", "scaledAdd", kernelInfo);

      BP->dotMultiplyKernel =
        mesh->device.buildKernel(DBP "/kernel/utils.okl", "dotMultiply", kernelInfo);

      BP->vecCopyKernel =
        mesh->device.buildKernel(DBP "/kernel/utils.okl", "vecCopy", kernelInfo);

      BP->vecInvKernel =
        mesh->device.buildKernel(DBP "/kernel/utils.okl", "vecInv", kernelInfo);

      BP->vecScaleKernel =
        mesh->device.buildKernel(DBP "/kernel/utils.okl", "vecScale", kernelInfo);

      BP->updateJacobiKernel =
        mesh->device.buildKernel(DBP "/kernel/updateJacobi.okl", "updateJacobi", kernelInfo);

/*
      BP->vecAtomicGatherKernel =
          mesh->device.buildKernel(DBP "/kernel/utils.okl", "vecAtomicGather", kernelInfo);

      BP->vecAtomicMultipleGatherKernel =
          mesh->device.buildKernel(DBP "/kernel/utils.okl", "vecAtomicMultipleGather", kernelInfo);

      BP->vecAtomicInnerProductKernel =
   mesh->device.buildKernel(DBP "/kernel/utils.okl", "vecAtomicInnerProduct", kernelInfo);
 */

      // add custom defines
      kernelInfo["defines/" "p_NpTet"] = mesh->Np;

      kernelInfo["defines/" "p_NpP"] = (mesh->Np + mesh->Nfp * mesh->Nfaces);
      kernelInfo["defines/" "p_Nverts"] = mesh->Nverts;

      int Nmax = mymax(mesh->Np, mesh->Nfaces * mesh->Nfp);
      int maxNodes = mymax(mesh->Np, (mesh->Nfp * mesh->Nfaces));
      int NblockV = mymax(1,maxNthreads / mesh->Np);
      int NnodesV = 1; //hard coded for now
      int NblockS = mymax(1,maxNthreads / maxNodes);
      int NblockP = mymax(1,maxNthreads / (4 * mesh->Np));
      int NblockG;
      if(mesh->Np <= 32) NblockG = ( 32 / mesh->Np );
      else NblockG = maxNthreads / mesh->Np;

      kernelInfo["defines/" "p_Nmax"] = Nmax;
      kernelInfo["defines/" "p_maxNodes"] = maxNodes;
      kernelInfo["defines/" "p_NblockV"] = NblockV;
      kernelInfo["defines/" "p_NnodesV"] = NnodesV;
      kernelInfo["defines/" "p_NblockS"] = NblockS;
      kernelInfo["defines/" "p_NblockP"] = NblockP;
      kernelInfo["defines/" "p_NblockG"] = NblockG;

      kernelInfo["defines/" "p_halfC"] = (int)((mesh->cubNq + 1) / 2);
      kernelInfo["defines/" "p_halfN"] = (int)((mesh->Nq + 1) / 2);

      kernelInfo["defines/" "p_NthreadsUpdatePCG"] = (int) NthreadsUpdatePCG; // WARNING SHOULD BE MULTIPLE OF 32
      kernelInfo["defines/" "p_NwarpsUpdatePCG"] = (int) (NthreadsUpdatePCG / 32); // WARNING: CUDA SPECIFIC

      BP->BPKernel = (occa::kernel*) new occa::kernel[1];

      int combineDot = 0;
      combineDot = 0; //options.compareArgs("COMBINE DOT PRODUCT", "TRUE");

      occa::properties props = kernelInfo;
      if(strstr(threadModel.c_str(), "NATIVE")) props["okl/enabled"] = false;

      string fileName = DBP "/kernel/" + arch + "/updatePCG.okl";
      if(strstr(threadModel.c_str(),
                "NATIVE+SERIAL") || strstr(threadModel.c_str(), "NATIVE+OPENMP"))
        fileName = "kernel/" + arch + "/updatePCG.c";

      BP->updatePCGKernel =
        mesh->device.buildKernel(fileName.c_str(), "BPUpdatePCG", props);
      BP->updateMultiplePCGKernel =
        mesh->device.buildKernel(fileName.c_str(), "BPMultipleUpdatePCG", props);

      fileName = DBP "/kernel/" + arch + "/updateOverlapPCG.okl"; 
      BP->updateOverlapPCGKernel =
        mesh->device.buildKernel(fileName.c_str(), "BPUpdateOverlapPCG", props);
      BP->updateOverlapMultiplePCGKernel =
        mesh->device.buildKernel(fileName.c_str(), "BPMultipleUpdateOverlapPCG", props);

      fileName = DBP "/kernel/utils.okl";
      if(strstr(threadModel.c_str(),
                "NATIVE+SERIAL") || strstr(threadModel.c_str(), "NATIVE+OPENMP"))
        fileName = "kernel/" + arch + "/weightedInnerProduct.c";
      BP->weightedInnerProduct2Kernel =
        mesh->device.buildKernel(fileName.c_str(), "weightedInnerProduct2", props);
      BP->weightedMultipleInnerProduct2Kernel =
        mesh->device.buildKernel(fileName.c_str(), "weightedMultipleInnerProduct2", props);

      occa::kernel nothingKernel = mesh->device.buildKernel(DBP "/kernel/utils.okl",
                                                            "nothingKernel",
                                                            kernelInfo);
      nothingKernel();
    }
    MPI_Barrier(mesh->comm);
  }

  string kernelName = "axhelm";
  if(BP->overlap)
    kernelName += "Partial";
  if(BP->BPid)
    kernelName += "_bk";
  if(BP->Nfields > 1) kernelName += "_n" + std::to_string(BP->Nfields);
  kernelName += "_v" + std::to_string(knlId);

  BP->BPKernel[0] = loadAxKernel(mesh->device,
                                 threadModel,
                                 arch,
                                 kernelName,
                                 mesh->N,
                                 mesh->Nelements);

  dfloat nullProjectWeightLocal = 0;
  dfloat nullProjectWeightGlobal = 0;
  for(dlong n = 0; n < BP->Nblock; ++n)
    nullProjectWeightLocal += BP->tmp[n];

  MPI_Allreduce(&nullProjectWeightLocal,
                &nullProjectWeightGlobal,
                1,
                MPI_DFLOAT,
                MPI_SUM,
                mesh->comm);

  BP->nullProjectWeightGlobal = 1. / nullProjectWeightGlobal;

  //use the masked ids to make another gs handle
  //BP->ogs = ogsSetup(Ntotal, mesh->maskedGlobalIds, mesh->comm, 1, mesh->device);
  auto callback = [&]() {
                    if(!BP->overlap || !mesh->NlocalGatherElements) return;

                    mesh_t* mesh = BP->mesh;
                    const dlong fieldOffset = BP->fieldOffset;
                    occa::kernel &kernel = BP->BPKernel[0];
                    occa::memory &o_lambda = BP->o_solveWorkspace[1];
                    occa::memory &o_q  = BP->o_solveWorkspace[2];
                    occa::memory &o_Aq = BP->o_solveWorkspace[3];
                    kernel(mesh->NlocalGatherElements,
                           fieldOffset,
                           mesh->o_localGatherElementList,
                           mesh->o_ggeo,
                           mesh->o_D,
                           o_lambda,
                           o_q,
                           o_Aq);
                  };
  BP->ogs = (void*) oogs::setup(Ntotal,
                                mesh->maskedGlobalIds,
                                ogsDfloat,
                                mesh->comm,
                                1,
                                mesh->device,
                                callback,
                                OOGS_AUTO);
  BP->o_invDegree = ((oogs_t*)BP->ogs)->ogs->o_invDegree;

/*
  if(options.compareArgs("DISABLE HALOGS", "TRUE")) {
    ((oogs_t*)BP->ogs)->ogs->NhaloGather = 0;
  }
*/

  BP->streamDefault = mesh->device.getStream();
  BP->stream1 = mesh->device.createStream();
}
