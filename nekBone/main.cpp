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

#include "BP.hpp"

int main(int argc, char **argv){

  // start up MPI
  MPI_Init(&argc, &argv);

  if(argc!=2){
    printf("usage: ./BP setupfile\n");

    MPI_Finalize();
    exit(-1);
  }

  // if argv > 2 then should load input data from argv
  setupAide options(argv[1]);

  // set up mesh stuff
  string fileName;
  int N, dim, elementType, kernelId;

  options.getArgs("POLYNOMIAL DEGREE", N);
  int cubN = 0;

  options.setArgs("MESH DIMENSION", "3");
  options.setArgs("BOX DOMAIN", "TRUE");
  options.setArgs("DISCRETIZATION", "CONTINUOUS");
  options.setArgs("ELEMENT MAP", "ISOPARAMETRIC");

  options.setArgs("ELEMENT TYPE", std::to_string(HEXAHEDRA));
  elementType=HEXAHEDRA;
  options.setArgs("ELLIPTIC INTEGRATION", "NODAL");
  options.setArgs("BASIS", "NODAL");
  options.getArgs("KERNEL ID", kernelId);

  int combineDot = 0;
  combineDot = 0; //options.compareArgs("COMBINE DOT PRODUCT", "TRUE");

  mesh_t *mesh;
  
  // set up mesh
  mesh = meshSetupBoxHex3D(N, cubN, options);
  mesh->elementType = elementType;
  
  dfloat lambda = 1, mu = 1;
  options.getArgs("LAMBDA", lambda);
  //options.getArgs("VISCOSITY",  mu);
  
  // set up
  occa::properties kernelInfo;
  kernelInfo["defines"].asObject();
  kernelInfo["includes"].asArray();
  kernelInfo["header"].asArray();
  kernelInfo["flags"].asObject();

  meshOccaSetup3D(mesh, options, kernelInfo);
  
  BP_t *BP = setup(mesh, lambda, mu, kernelInfo, options);

  occa::memory o_r, o_x;

  dlong Ndofs = mesh->Np*mesh->Nelements;
  Ndofs *= BP->Nfields;
  o_r = mesh->device.malloc(Ndofs*sizeof(dfloat), BP->o_r);
  o_x = mesh->device.malloc(Ndofs*sizeof(dfloat), BP->o_x);    
  
  // convergence tolerance
  dfloat tol = 1e-8;
  
  int it;
  int bpstart = BP->BPid;
  int bpid = BP->BPid;
  {
    BP->BPid = bpid;
    
    MPI_Barrier(mesh->comm);

    // warm up
    double opElapsed = 0;

    //BPSolve(BP, lambda, mu, tol, BP->o_r, BP->o_x, &opElapsed);
    
    opElapsed = 0;
    
    int Ntests = 1;
    occa::streamTag *startTags = new occa::streamTag[Ntests];
    occa::streamTag *stopTags  = new occa::streamTag[Ntests];

    it = 0;
    double globalElapsed;
    double elapsed = MPI_Wtime();
    for(int test=0;test<Ntests;++test){

      o_r.copyTo(BP->o_r);
      o_x.copyTo(BP->o_x);
      
//      startTags[test] = mesh->device.tagStream();

      it += BPSolve(BP, lambda, mu, tol, BP->o_r, BP->o_x, &opElapsed);

//      stopTags[test] = mesh->device.tagStream();
    }
    mesh->device.finish();  
    MPI_Barrier(mesh->comm);
  
#if 1 
    elapsed = MPI_Wtime() - elapsed; 
    globalElapsed = elapsed;
#else
    for(int test=0;test<Ntests;++test){
      elapsed += mesh->device.timeBetween(startTags[test], stopTags[test]);
    }
    MPI_Reduce(&elapsed, &globalElapsed, 1, MPI_DOUBLE, MPI_MAX, 0, mesh->comm);
#endif

    hlong globalNelements, localNelements=mesh->Nelements;
    MPI_Reduce(&localNelements, &globalNelements, 1, MPI_HLONG, MPI_SUM, 0, mesh->comm);
   
    // copy solution from DEVICE to HOST
    BP->o_x.copyTo(BP->q);
    dfloat maxError = 0;
    for(dlong e=0;e<mesh->Nelements;++e){
      for(int n=0;n<mesh->Np;++n){
	dlong   id = e*mesh->Np+n;
	dfloat xn = mesh->x[id];
	dfloat yn = mesh->y[id];
	dfloat zn = mesh->z[id];
      
	dfloat exact;
	double mode = 1.0;
	// hard coded to match the RHS used in BPSetup
	exact = (3.*M_PI*M_PI*mode*mode+lambda)*cos(mode*M_PI*xn)*cos(mode*M_PI*yn)*cos(mode*M_PI*zn);
	exact /= (3.*mode*mode*M_PI*M_PI+lambda);
	dfloat error = fabs(exact-BP->q[id]);
	maxError = mymax(maxError, error);
      }
    }
    dfloat globalMaxError = 0;
    MPI_Allreduce(&maxError, &globalMaxError, 1, MPI_DFLOAT, MPI_MAX, mesh->comm);

    // print statistics 
    double NGbytes;
    int combineDot = 0; //options.compareArgs("COMBINE DOT PRODUCT", "TRUE");
    int useInvDeg = 1;
 
    if(options.compareArgs("KRYLOV SOLVER", "PCG"))
      // z=r, z.r/deg, p=z+beta*p, A*p (p in/Ap out), [x=x+alpha*p, r=r-alpha*Ap, r.r./deg]
      NGbytes = mesh->Nlocalized*((BP->Nfields*(2+2+3+2+3+3+1)+2*useInvDeg)/1.e9);  
            
    if(!combineDot) NGbytes += (BP->Nfields*2+useInvDeg)*(mesh->Nlocalized/1.e9);  // z.Az/deg

    NGbytes += mesh->Nelements*(mesh->Nggeo*mesh->Np/1.e9);

    NGbytes *= sizeof(dfloat);
    double bw = (it*(NGbytes/(globalElapsed)));
    MPI_Allreduce(MPI_IN_PLACE, &bw, 1, MPI_DFLOAT, MPI_SUM, mesh->comm);

    if(mesh->rank==0){
      printf("globalMaxError = %g\n", globalMaxError);
 
      int knlId = 0;
      options.getArgs("KERNEL ID", knlId);
  
      printf("global: N, Nfields, Nelements, elapsed, iterations, GNodes/s/iter, BW GB/s, kernel Id\n"); 
      printf("%d, %d, %d, %g, %d, %g, %g, %d\n",
	     mesh->N,
             BP->Nfields,
	     globalNelements,
	     globalElapsed,
	     it,
	     BP->Nfields*(it*(mesh->Np*globalNelements/globalElapsed))/1.e9,
	     bw,
	     knlId);
    }

#if 0   
    if (options.compareArgs("VERBOSE", "TRUE")){
      fflush(stdout);
      MPI_Barrier(mesh->comm);
      printf("rank %d has %d internal elements and %d non-internal elements\n",
	     mesh->rank,
	     mesh->NinternalElements,
	     mesh->NnotInternalElements);
      MPI_Barrier(mesh->comm);
    }
#endif
  
  }  
  MPI_Finalize();
  return 0;
}
