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

#include "omp.h"
#include "BP.hpp"

int solve(BP_t *BP, occa::memory &o_lambda, dfloat tol, occa::memory &o_r, occa::memory &o_x, double *opElapsed){
  mesh_t *mesh = BP->mesh;
  setupAide options = BP->options;

  int Niter = 0;
  int maxIter = 1000; 

  options.getArgs("MAXIMUM ITERATIONS", maxIter);
  options.getArgs("SOLVER TOLERANCE", tol);

  if(BP->allNeumann) 
    BPZeroMean(BP, o_r);
  
  if(options.compareArgs("KRYLOV SOLVER", "PCG"))
    Niter = BPPCG(BP, o_lambda, o_r, o_x, tol, maxIter, opElapsed);

  if(BP->allNeumann) 
    BPZeroMean(BP, o_x);
  
  return Niter;
}

int main(int argc, char **argv){

  // start up MPI
  MPI_Init(&argc, &argv);

  if(argc!=2){
    printf("usage: ./nekBone setupfile\n");

    MPI_Finalize();
    exit(1);
  }

  // if argv > 2 then should load input data from argv
  setupAide options(argv[1]);

  // set up mesh stuff
  string fileName;
  int N, dim, elementType, kernelId;

  options.getArgs("POLYNOMIAL DEGREE", N);
  int cubN = 0;

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
  elementType=HEXAHEDRA;
  options.setArgs("ELLIPTIC INTEGRATION", "NODAL");
  options.setArgs("BASIS", "NODAL");
  options.getArgs("KERNEL ID", kernelId);

  int sync = 0;
  if(options.compareArgs("TIMER SYNC", "TRUE")) sync = 1;

  int combineDot = 0;
  combineDot = 0; //options.compareArgs("COMBINE DOT PRODUCT", "TRUE");

  mesh_t *mesh;
  
  // set up mesh
  mesh = meshSetupBoxHex3D(N, cubN, options);
  mesh->elementType = elementType;
 
  // set up
  occa::properties kernelInfo;
  //kernelInfo["defines"].asObject();
  //kernelInfo["includes"].asArray();
  //kernelInfo["header"].asArray();
  //kernelInfo["flags"].asObject();

  meshOccaSetup3D(mesh, options, kernelInfo);
  timer::init(MPI_COMM_WORLD, mesh->device, sync); 

  dfloat lambda1 = 1;
  options.getArgs("LAMBDA", lambda1);
  
  BP_t *BP = setup(mesh, lambda1, kernelInfo, options);

  dlong Ndofs = BP->Nfields*mesh->Np*mesh->Nelements;
 
  // convergence tolerance
  dfloat tol = 1e-8;
  
  int it;
  int bpstart = BP->BPid;
  int bpid = BP->BPid;
  {
    BP->BPid = bpid;
    double opElapsed = 0;
    int Ntests = 1;
    it = 0;

    if(mesh->rank==0) cout << "\nrunning ...";
    mesh->device.finish();  
    MPI_Barrier(mesh->comm);
    double elapsed = MPI_Wtime();
    for(int test=0;test<Ntests;++test){
      it += solve(BP, BP->o_lambda, tol, BP->o_r, BP->o_x, &opElapsed);
    }
    mesh->device.finish();  
    MPI_Barrier(mesh->comm);
    if(mesh->rank==0) cout << " done\n";
    elapsed = MPI_Wtime() - elapsed; 

    hlong globalNelements, localNelements=mesh->Nelements;
    MPI_Reduce(&localNelements, &globalNelements, 1, MPI_HLONG, MPI_SUM, 0, mesh->comm);
  
    hlong globalNdofs = pow(mesh->N,3)*mesh->Nelements; // mesh->Nlocalized;
    MPI_Allreduce(MPI_IN_PLACE, &globalNdofs, 1, MPI_HLONG, MPI_SUM, mesh->comm);
 
    // copy solution from DEVICE to HOST
    BP->o_x.copyTo(BP->q);
    const dlong offset = mesh->Np*(mesh->Nelements+mesh->totalHaloPairs); 
    dfloat maxError = 0;
    for(dlong fld=0;fld<BP->Nfields;++fld){
    for(dlong e=0;e<mesh->Nelements;++e){
      for(int n=0;n<mesh->Np;++n){
	dlong  id = e*mesh->Np+n;
	dfloat xn = mesh->x[id];
	dfloat yn = mesh->y[id];
	dfloat zn = mesh->z[id];
      
	dfloat exact;
	double mode = 1.0;
	// hard coded to match the RHS used in BPSetup
	exact = (3.*M_PI*M_PI*mode*mode+lambda1)*cos(mode*M_PI*xn)*cos(mode*M_PI*yn)*cos(mode*M_PI*zn);
	exact /= (3.*mode*mode*M_PI*M_PI+lambda1);
	dfloat error = fabs(exact - BP->q[id+fld*offset]);
	maxError = mymax(maxError, error);
      }
    }
    }
    dfloat globalMaxError = 0;
    MPI_Allreduce(&maxError, &globalMaxError, 1, MPI_DFLOAT, MPI_MAX, mesh->comm);

    // print statistics 
    double NGbytes;
    int combineDot = 0; //options.compareArgs("COMBINE DOT PRODUCT", "TRUE");
    int useInvDeg = 1;

    double Nbytes = Ndofs*sizeof(dfloat);
    double gbytesPCG = 7.*mesh->Np*mesh->Nelements*(sizeof(dfloat)/1.e9);
    double gbytesCopy = Nbytes/1.e9;
    double gbytesOp = (7+2*BP->Nfields)*mesh->Np*mesh->Nelements*(sizeof(dfloat)/1.e9);
    double gbytesDot = (2*BP->Nfields+1)*mesh->Np*mesh->Nelements*(sizeof(dfloat)/1.e9);
    double gbytesPupdate =  3*mesh->Np*mesh->Nelements*(sizeof(dfloat)/1.e9);
    if(!combineDot)
      gbytesOp += 3*mesh->Np*mesh->Nelements*(sizeof(dfloat)/1.e9);
 
    if(options.compareArgs("KRYLOV SOLVER", "PCG"))
      // z=r, z.r/deg, p=z+beta*p, A*p (p in/Ap out), [x=x+alpha*p, r=r-alpha*Ap, r.r./deg]
      NGbytes = mesh->Nlocalized*((BP->Nfields*(2+2+3+2+3+3+1)+2*useInvDeg)/1.e9);  
            
    if(!combineDot) NGbytes += (BP->Nfields*2+useInvDeg)*(mesh->Nlocalized/1.e9);  // z.Az/deg

    NGbytes += mesh->Nelements*(mesh->Nggeo*mesh->Np/1.e9);

    NGbytes *= sizeof(dfloat);
    double bw = (it*(NGbytes/(elapsed)));
    MPI_Allreduce(MPI_IN_PLACE, &bw, 1, MPI_DFLOAT, MPI_SUM, mesh->comm);

    double etime[10];
    etime[0] = timer::query("Ax", "DEVICE:MAX");
    etime[1] = timer::query("gs", "HOST:MAX");
    etime[2] = timer::query("updatePCG", "HOST:MAX");
    etime[3] = timer::query("dotp", "HOST:MAX");

    if(mesh->rank==0){
      printf("correctness check: maxError = %g\n", globalMaxError);
 
      int knlId = 0;
      options.getArgs("KERNEL ID", knlId);

      int Nthreads =  omp_get_max_threads();
      cout << "\nsummary\n" 
           << "  MPItasks   : " << mesh->size << "\n"
           << "  OMPthreads : " << Nthreads << "\n"
           << "  polyN      : " << N << "\n"
           << "  Nelements  : " << globalNelements << "\n"
           << "  iterations : " << it << "\n"
           << "  walltime   : " << elapsed << " s\n"
           << "  throughput : " << BP->Nfields*(it*(globalNdofs/elapsed))/1.e9 << " GDOF/s/iter\n"
           << "  bandwidth  : " << bw << " GB/s\n";

      cout << "\ntimings\n" 
           << "  Ax         : " << etime[0] << " s\n"
           << "  gs         : " << etime[1] << " s\n"
           << "  updatePCG  : " << etime[2] << " s\n"
           << "  dotp       : " << etime[3] << " s\n"
           << endl;
    }

    if (options.compareArgs("VERBOSE", "TRUE")){
      fflush(stdout);
      MPI_Barrier(mesh->comm);
      printf("rank %d has %d internal elements and %d non-internal elements\n",
	     mesh->rank,
	     mesh->NinternalElements,
	     mesh->NnotInternalElements);
      MPI_Barrier(mesh->comm);
    }
  
  }  
  MPI_Finalize();
  return 0;
}
