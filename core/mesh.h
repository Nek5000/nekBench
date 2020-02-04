
#ifndef MESH_H
#define MESH_H 1

#include <unistd.h>

#include "mpi.h"
#include <math.h>
#include <stdlib.h>
#include <occa.hpp>

#include "types.h"
#include "ogs.hpp"

//#include "timer.h"

#include "setupAide.hpp"
#include "meshBasis.hpp"

typedef struct {

  MPI_Comm comm;
  int rank, size; // MPI rank and size (process count)

  int elementType;
  
  int dim;
  int Nverts, Nfaces, NfaceVertices;

  hlong Nnodes;
  dfloat *EX; // coordinates of vertices for each element
  dfloat *EY;
  dfloat *EZ;

  dlong Nelements;
  hlong *EToV; // element-to-vertex connectivity
  dlong *EToE; // element-to-element connectivity
  int   *EToF; // element-to-(local)face connectivity
  int   *EToP; // element-to-partition/process connectivity

  dlong NboundaryFaces;
  int   *EToB; // element-to-boundary condition type
  

  hlong *elementInfo; //type of element

  // MPI halo exchange info
  dlong  totalHaloPairs;  // number of elements to be sent in halo exchange
  dlong *haloElementList; // sorted list of elements to be sent in halo exchange
  int *NhaloPairs;      // number of elements worth of data to send/recv
  int  NhaloMessages;     // number of messages to send

  dlong *haloGetNodeIds; // volume node ids of outgoing halo nodes
  dlong *haloPutNodeIds; // volume node ids of incoming halo nodes

  void *haloSendRequests;
  void *haloRecvRequests;

  dlong NinternalElements; // number of elements that can update without halo exchange
  dlong NnotInternalElements; // number of elements that cannot update without halo exchange

  // CG gather-scatter info
  hlong *globalIds;
  hlong *maskedGlobalIds;
  void *gsh, *hostGsh; // gslib struct pointer
  ogs_t *ogs; //occa gs pointer

  dlong Nlocalized;
  hlong *localizedIds;
  hlong startLocalized;
  occa::memory o_localizedIds;

  // list of elements that are needed for global gather-scatter
  dlong NglobalGatherElements;
  dlong *globalGatherElementList;
  occa::memory o_globalGatherElementList;

  // list of elements that are not needed for global gather-scatter
  dlong NlocalGatherElements;
  dlong *localGatherElementList;
  occa::memory o_localGatherElementList;

  //list of fair pairs
  dlong NfacePairs;
  dlong *EToFPairs;
  dlong *FPairsToE;
  int *FPairsToF;

  // NBN: streams / command queues
  occa::stream stream0, stream1;

  // volumeGeometricFactors;
  dlong Nvgeo;
  dfloat *vgeo;

  // second order volume geometric factors
  dlong Nggeo;
  dfloat *ggeo;

  // volume node info
  int N, Np;
  dfloat *r, *s, *t;    // coordinates of local nodes
  dfloat *Dr, *Ds, *Dt; // collocation differentiation matrices
  dfloat *Dmatrices;
  dfloat *filterMatrix; // C0 basis filter matrix
  
  dfloat *MM, *invMM;           // reference mass matrix
  dfloat *Smatrices;
  int maxNnzPerRow;
  dfloat *x, *y, *z;    // coordinates of physical nodes

  // indices of vertex nodes
  int *vertexNodes;

  // quad specific quantity
  int Nq;
  
  dfloat *D; // 1D differentiation matrix (for tensor-product)
  dfloat *gllz; // 1D GLL quadrature nodes
  dfloat *gllw; // 1D GLL quadrature weights

  int gjNq;
  dfloat *gjr,*gjw; // 1D nodes and weights for Gauss Jacobi quadature
  dfloat *gjI,*gjD; // 1D GLL to Gauss node interpolation and differentiation matrices
  dfloat *gjD2;     // 1D GJ to GJ node differentiation

  // face node info
  int Nfp;        // number of nodes per face
  int *faceNodes; // list of element reference interpolation nodes on element faces
  dlong *vmapM;     // list of volume nodes that are face nodes
  dlong *vmapP;     // list of volume nodes that are paired with face nodes
  dlong *mapP;     // list of surface nodes that are paired with -ve surface  nodes
  int *faceVertices; // list of mesh vertices on each face

  dlong   Nsgeo;
  dfloat *sgeo;

  // cubature
  int cubNp, cubNfp, cubNq;
  dfloat *cubr, *cubs, *cubt, *cubw; // coordinates and weights of local cubature nodes
  dfloat *cubx, *cuby, *cubz;    // coordinates of physical nodes
  dfloat *cubInterp;  // interpolate from W&B to cubature nodes
  dfloat *cubInterp3D; // interpolate from W&B to cubature nodes
  dfloat *cubD;       // 1D differentiation matrix

  dfloat *cubvgeo;  //volume geometric data at cubature points
  dfloat *cubsgeo;  //surface geometric data at cubature points
  dfloat *cubggeo;  //second type volume geometric data at cubature points
  
  // occa stuff
  occa::device device;

  occa::stream defaultStream;
  occa::stream dataStream;
  occa::stream computeStream;

  occa::memory o_D; // tensor product differentiation matrix (for Hexes)
  occa::memory o_cubD; // tensor product differentiation matrix (for Hexes)
  occa::memory o_cubInterp;
  occa::memory o_cubInterp3D;

  occa::memory o_filterMatrix; // tensor product filter matrix (for hexes)
  
  occa::memory o_Smatrices;
  occa::memory o_vgeo, o_sgeo;
  occa::memory o_vmapM, o_vmapP, o_mapP;
  occa::memory o_EToE, o_EToF, o_EToB, o_x, o_y, o_z;
  occa::memory o_EToFPairs, o_FPairsToE, o_FPairsToF;

  occa::memory o_cubvgeo, o_cubsgeo, o_cubggeo;

  // DG halo exchange info
  occa::memory o_haloElementList;
  occa::memory o_haloBuffer;
  occa::memory o_haloGetNodeIds;
  occa::memory o_haloPutNodeIds;
  
  occa::memory o_internalElementIds;
  occa::memory o_notInternalElementIds;

  occa::memory o_ggeo; // second order geometric factors

  occa::kernel haloExtractKernel;
  occa::kernel haloGetKernel;
  occa::kernel haloPutKernel;

  occa::kernel gatherKernel;
  occa::kernel scatterKernel;
  occa::kernel gatherScatterKernel;

  occa::kernel getKernel;
  occa::kernel putKernel;

  occa::kernel sumKernel;
  occa::kernel addScalarKernel;

  occa::kernel innerProductKernel;
  occa::kernel weightedInnerProduct1Kernel;
  occa::kernel weightedInnerProduct2Kernel;
  occa::kernel scaledAddKernel;
  occa::kernel dotMultiplyKernel;
  occa::kernel dotDivideKernel;

  occa::kernel gradientKernel;
  occa::kernel ipdgKernel;

  occa::kernel maskKernel;

}mesh_t;

// serial sort
void mysort(hlong *data, int N, const char *order);

// sort entries in an array in parallel
void parallelSort(int size, int rank, MPI_Comm comm,
		  int N, void *vv, size_t sz,
		  int (*compare)(const void *, const void *),
		  void (*match)(void *, void *)
		  );

#define mymax(a,b) (((a)>(b))?(a):(b))
#define mymin(a,b) (((a)<(b))?(a):(b))

/* hash function */
unsigned int hash(const unsigned int value) ;

/* dimension independent mesh operations */
void meshConnect(mesh_t *mesh);

/* build parallel face connectivity */
void meshParallelConnect(mesh_t *mesh);

/* build global connectivity in parallel */
void meshParallelConnectNodes(mesh_t *mesh);

/* build global connectivity in terms of localized global nodes */
void meshLocalizedConnectNodes(mesh_t *mesh);

void meshHaloSetup(mesh_t *mesh);

/* extract whole elements for the halo exchange */
void meshHaloExtract(mesh_t *mesh, size_t Nbytes, void *sourceBuffer, void *haloBuffer);

void meshHaloExchange(mesh_t *mesh,
    size_t Nbytes,         // message size per element
    void *sourceBuffer,
    void *sendBuffer,    // temporary buffer
    void *recvBuffer);

void meshHaloExchangeStart(mesh_t *mesh,
    size_t Nbytes,       // message size per element
    void *sendBuffer,    // temporary buffer
    void *recvBuffer);


void meshHaloExchangeFinish(mesh_t *mesh);

void meshHaloExchangeBlocking(mesh_t *mesh,
			     size_t Nbytes,       // message size per element
			     void *sendBuffer,    // temporary buffer
			      void *recvBuffer);

// print out parallel partition i
void meshPartitionStatistics(mesh_t *mesh);

// build element-boundary connectivity
void meshConnectBoundary(mesh_t *mesh);

void meshParallelGatherScatterSetup(mesh_t *mesh,
                                      dlong N,
                                      hlong *globalIds,
                                      MPI_Comm &comm,
                                      int verbose);

// generic mesh setup
mesh_t *meshSetup(char *filename, int N, setupAide &options);

extern "C"
{
  void dgesv_ ( int     *N, int     *NRHS, double  *A,
                int     *LDA,
                int     *IPIV, 
                double  *B,
                int     *LDB,
                int     *INFO );

   void dgemm_ (char *, char *, int *, int *, int *,
         const dfloat *, const dfloat * __restrict, int *,
         const dfloat * __restrict, int *,
         const dfloat *, dfloat * __restrict, int *);

  void sgesv_(int *N, int *NRHS,float  *A, int *LDA, int *IPIV, float  *B, int *LDB,int *INFO);

  void dgetrf_(int* M, int *N, double* A, int* lda, int* IPIV, int* INFO);
  void dgetri_(int* N, double* A, int* lda, int* IPIV, double* WORK, int* lwork, int* INFO);
  void dgeev_(char *JOBVL, char *JOBVR, int *N, double *A, int *LDA, double *WR, double *WI,
              double *VL, int *LDVL, double *VR, int *LDVR, double *WORK, int *LWORK, int *INFO );
  
  double dlange_(char *NORM, int *M, int *N, double *A, int *LDA, double *WORK);
  void dgecon_(char *NORM, int *N, double *A, int *LDA, double *ANORM,
                double *RCOND, double *WORK, int *IWORK, int *INFO );
}

void readDfloatArray(FILE *fp, const char *label, dfloat **A, int *Nrows, int* Ncols);
void readIntArray   (FILE *fp, const char *label, int **A   , int *Nrows, int* Ncols);

void meshApplyElementMatrix(mesh_t *mesh, dfloat *A, dfloat *q, dfloat *Aq);

void matrixInverse(int N, dfloat *A);

dfloat matrixConditionNumber(int N, dfloat *A);

void occaDeviceConfig(mesh_t *mesh, setupAide &newOptions);

void *occaHostMallocPinned(occa::device &device, size_t size, void *source, occa::memory &mem, occa::memory &h_mem);

#define mesh3D mesh_t

// mesh readers
mesh3D* meshParallelReaderHex3D(char *fileName);

// build connectivity in serial
void meshConnect3D(mesh3D *mesh);

// build element-boundary connectivity
void meshConnectBoundary3D(mesh3D *mesh);

// build connectivity in parallel
void meshParallelConnect3D(mesh3D *mesh);

// repartition elements in parallel
void meshGeometricPartition3D(mesh3D *mesh);

// print out mesh 
void meshPrint3D(mesh3D *mesh);

// print out mesh in parallel from the root process
void meshParallelPrint3D(mesh3D *mesh);

// print out mesh partition in parallel
void meshVTU3D(mesh3D *mesh, char *fileName);

// print out mesh field
void meshPlotVTU3D(mesh3D *mesh, char *fileNameBase, int fld);
void meshPlotContour3D(mesh_t *mesh, char *fname, dfloat *u, int Nlevels, dfloat *levels);
void meshPlotAdaptiveContour3D(mesh_t *mesh, char *fname, dfloat *u, int Nlevels, dfloat *levels, dfloat tol);

// compute geometric factors for local to physical map
void meshGeometricFactorsHex3D(mesh3D *mesh);

void meshSurfaceGeometricFactorsHex3D(mesh3D *mesh);
void meshSurfaceGeometricFactorsTet3D(mesh3D *mesh);

void meshPhysicalNodesHex3D(mesh3D *mesh);
void meshPhysicalNodesTet3D(mesh3D *mesh);

void meshLoadReferenceNodesHex3D(mesh3D *mesh, int N, int cubN);
void meshLoadReferenceNodesTet3D(mesh3D *mesh, int N, int cubN);

void meshGradientHex3D(mesh3D *mesh, dfloat *q, dfloat *dqdx, dfloat *dqdy, dfloat *dqdz);

// print out parallel partition i
void meshPartitionStatistics3D(mesh3D *mesh);

// default occa set up
void meshOccaSetup3D(mesh3D *mesh, setupAide &newOptions, occa::properties &kernelInfo);

void meshOccaPopulateDevice3D(mesh3D *mesh, setupAide &newOptions, occa::properties &kernelInfo);
void meshOccaCloneDevice(mesh_t *donorMesh, mesh_t *mesh);

// functions that call OCCA kernels
void occaTest3D(mesh3D *mesh, dfloat *q, dfloat *dqdx, dfloat *dqdy, dfloat *dqdz);

// 
void occaOptimizeGradientHex3D(mesh3D *mesh, dfloat *q, dfloat *dqdx, dfloat *dqdy, dfloat *dqdz);

// serial face-node to face-node connection
void meshConnectFaceNodes3D(mesh3D *mesh);

//
mesh3D *meshSetupHex3D(char *filename, int N);

void meshParallelConnectNodesHex3D(mesh3D *mesh);

// halo connectivity information
void meshHaloSetup3D(mesh3D *mesh);

// perform halo exchange
void meshHaloExchange3D(mesh3D *mesh,
			size_t Nbytes,  // number of bytes per element
			void *sourceBuffer, 
			void *sendBuffer, 
			void *recvBuffer);

void meshHaloExchangeStart3D(mesh3D *mesh,
			     size_t Nbytes,       // message size per element
			     void *sendBuffer,    // temporary buffer
			     void *recvBuffer);

void meshHaloExchangeFinish3D(mesh3D *mesh);

// build list of nodes on each face of the reference element
void meshBuildFaceNodes3D(mesh3D *mesh);
void meshBuildFaceNodesHex3D(mesh3D *mesh);



dfloat meshMRABSetup3D(mesh3D *mesh, dfloat *EToDT, int maxLevels, dfloat finalTime); 

//MRAB weighted mesh partitioning
void meshMRABWeightedPartition3D(mesh3D *mesh, dfloat *weights,
                                      int numLevels, int *levels);

void meshInterpolateHex3D(dfloat *Inter, dfloat *x, int N, dfloat *Ix, int M);
void meshInterpolateTet3D(dfloat *I, dfloat *x, int N, dfloat *Ix, int M);

#define norm3(a,b,c) ( sqrt((a)*(a)+(b)*(b)+(c)*(c)) )


mesh3D *meshSetupBoxHex3D(int N, int cubN, setupAide &options);
mesh3D *meshSetupBoxTet3D(int N, int cubN, setupAide &options);

void meshConnectPeriodicFaceNodes3D(mesh3D *mesh, dfloat xper, dfloat yper, dfloat zper);


int meshWarpBlendNodesTet3D(int N, dfloat **r, dfloat **s, dfloat **t);

#define TRIANGLES 3
#define QUADRILATERALS 4
#define TETRAHEDRA 6
#define HEXAHEDRA 12

#endif

