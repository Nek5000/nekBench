/*
 * TODO:
 *  - Add device support for gs_init
 *  - Use separate device MPI buffers for each message (to avoid alignment issues)?
 *  - Add support for different data types + gs ops
 *  - Add gs input array to gathered array
 *  - Copy MPI buffer async
 */

#include <omp.h>
#include <occa.hpp>
#include "ogs.hpp"
#include "ogsKernels.hpp"
#include "ogsInterface.h"
#include "timer.hpp"

/* pack/unpack MPI buffer on device but stage communication throuh HOST */
//#define USE_GPU_GSBUF

/* includes USE_GPU_GSBUF but bypasses HOST also for communication */
//#define USE_GPU_AWARE_MPI


#ifdef __cplusplus
extern "C" {
#endif

#include "gslib.h"

typedef void gs_init_fun(
  void *out, const unsigned vn,
  const uint *map, gs_dom dom, gs_op op);
extern gs_init_fun gs_init;

typedef void gs_scatter_fun(
  void *out, const void *in, const unsigned vn,
  const uint *map, gs_dom dom);
extern gs_scatter_fun gs_scatter;

typedef void gs_gather_fun(
  void *out, const void *in, const unsigned vn,
  const uint *map, gs_dom dom, gs_op op);
extern gs_gather_fun gs_gather;

typedef enum { mode_plain, mode_vec, mode_many,
               mode_dry_run } gs_mode;

struct pw_comm_data {
  uint n;      /* number of messages */
  uint *p;     /* message source/dest proc */
  uint *size;  /* size of message */
  uint total;  /* sum of message sizes */
};

struct pw_data {
  struct pw_comm_data comm[2];
  const uint *map[2];
  comm_req *req;
  uint buffer_size;
};

typedef void exec_fun(
  void *data, gs_mode mode, unsigned vn, gs_dom dom, gs_op op,
  unsigned transpose, const void *execdata, const struct comm *comm, char *buf);
typedef void fin_fun(void *data);

struct gs_remote {
  uint buffer_size, mem_size;
  void *data;
  exec_fun *exec;
  exec_fun *exec_irecv;
  exec_fun *exec_isend;
  exec_fun *exec_wait;
  fin_fun *fin;
};

struct gs_data {
  struct comm comm;
  const uint *map_local[2]; /* 0=unflagged, 1=all */
  const uint *flagged_primaries;
  struct gs_remote r;
  uint handle_size;
};

#ifdef USE_GPU_AWARE_MPI
#define USE_GPU_GSBUF
#endif

// GLOBALS
static occa::memory h_buff, o_buff;
void *buff;

static int firstTime = 1;

static int *scatterOffsets, *gatherOffsets;
static int *scatterIds, *gatherIds;
static occa::memory o_scatterOffsets, o_gatherOffsets;
static occa::memory o_scatterIds, o_gatherIds;

static const double gs_identity_double[] = { 0, 1, 1.7976931348623157e+308, -1.7976931348623157e+308, 0 };

static void init_double(double *restrict out, 
                        const unsigned int *restrict map, 
                        gs_op op){ 
   unsigned int i; 
   const double e = gs_identity_double[op];
   while((i=*map++)!=(UINT_MAX)) out[i]=e;
}

static void myscatter_double(const int Nscatter,
                             const int *restrict starts,
                             const int *restrict ids,
                             const double *restrict q,
                             double *restrict scatterq){

//#pragma omp parallel for
  for(int s=0;s<Nscatter;++s){

    const double qs = q[s];

    const int start = starts[s];
    const int end   = starts[s+1];

    for(int n=start;n<end;++n){
      const int id = ids[n];
      scatterq[id] = qs;
    }
  }
}

void mygather_doubleAdd(const int Ngather,
                        const int *restrict starts,
                        const int *restrict ids,
                        const double *restrict q,
                        double *restrict gatherq){

//#pragma omp parallel for
  for(int g=0;g<Ngather;++g){

    const int start = starts[g];
    const int end = starts[g+1];

    double gq = 0.f;
    for(int n=start;n<end;++n){
      const int id = ids[n];
      gq += q[id];
    }

    gatherq[g] = gq;
  }
}


static void convertPwMap(const uint *restrict map,
                         int *restrict starts,
                         int *restrict ids){

  uint i,j; 
  int n=0, s=0;
  while((i=*map++)!=UINT_MAX) {
    starts[s] = n;
    j=*map++; 
    do {
      ids[n] = j;
      n++;
    } while((j=*map++)!=UINT_MAX);
    starts[s+1] = n;
    s++;
  }
}

static void myHostGatherScatter(
#ifdef USE_GPU_GSBUF
                                occa::memory o_u,
#else
                                void *u,
#endif
                                gs_dom dom, gs_op op,
                                unsigned transpose, ogs_t *ogs)
{
  struct gs_data *gsh = (gs_data*) ogs->haloGshSym;
  const unsigned recv = 0^transpose, send = 1^transpose;
  const void* execdata = gsh->r.data; 
  const struct pw_data *pwd = (pw_data*) execdata; 
  const struct comm *comm = &gsh->comm;

  // hardwired for now
  const unsigned vn = 1;
  const unsigned unit_size = vn*sizeof(double); //gs_dom_size[dom];
  const char type[] = "double";

  double eTime_pw = 0;

  if(firstTime) {
    occa::properties props;
    props["mapped"] = true;
    h_buff = ogs->device.malloc(gsh->r.buffer_size*unit_size, props);
    buff = h_buff.ptr(props);  
 
    scatterOffsets = (int*) calloc(2*ogs->NhaloGather,sizeof(int));
    gatherOffsets  = (int*) calloc(2*ogs->NhaloGather,sizeof(int));

    scatterIds = (int*) calloc(pwd->comm[send].total,sizeof(int));
    gatherIds  = (int*) calloc(pwd->comm[recv].total,sizeof(int));

    convertPwMap(pwd->map[send], scatterOffsets, scatterIds);
    convertPwMap(pwd->map[recv], gatherOffsets, gatherIds);

#ifdef USE_GPU_GSBUF
    o_buff = ogs->device.malloc(gsh->r.buffer_size*unit_size);

    o_scatterOffsets = ogs->device.malloc(2*ogs->NhaloGather*sizeof(int), scatterOffsets);
    o_gatherOffsets  = ogs->device.malloc(2*ogs->NhaloGather*sizeof(int), gatherOffsets);

    o_scatterIds = ogs->device.malloc(pwd->comm[send].total*sizeof(int), scatterIds);
    o_gatherIds  = ogs->device.malloc(pwd->comm[recv].total*sizeof(int), gatherIds);
#endif
    firstTime = 0;
  }

#ifdef USE_GPU_AWARE_MPI
  // todo
#else
  //if(transpose==0) gs_init(u,vn,gsh->flagged_primaries,dom,op);
  //if(transpose==0) init_double((double*) u,gsh->flagged_primaries,gs_add);
#endif

  // prepost recv
  {
    MPI_Barrier(comm->c);
    double t0 = MPI_Wtime(); 
    comm_req *req = pwd->req; 
    const struct pw_comm_data *c = &pwd->comm[recv];
    const uint *p, *pe, *size=c->size;
    uint bufOffset = 0;
    for(p=c->p,pe=p+c->n;p!=pe;++p) {
      size_t len = *(size++)*unit_size;
#ifdef USE_GPU_AWARE_MPI
      char *recvbuf = (char*) o_buff.ptr() + bufOffset;
#else
      char *recvbuf = (char *) buff + bufOffset;
#endif
      MPI_Irecv((void*) recvbuf,len,MPI_UNSIGNED_CHAR,*p,*p,comm->c,req++);
      bufOffset += len;
    }
    eTime_pw += MPI_Wtime() - t0;
  }

  // scatter
  {
    char *buf = (char *) buff + pwd->comm[recv].total*unit_size;
#ifdef USE_GPU_GSBUF
    occa::memory o_buf = o_buff + pwd->comm[recv].total*unit_size;
    occaScatter(ogs->NhaloGather, o_scatterOffsets, o_scatterIds, type, ogsAdd, o_u, o_buf);
#ifndef USE_GPU_AWARE_MPI
    o_buf.copyTo(buf, pwd->comm[send].total*unit_size);
#endif 

#else
  //gs_scatter(buf,u,vn,pwd->map[send],dom); 
  myscatter_double(ogs->NhaloGather, scatterOffsets, scatterIds, (double *) u, (double *) buf); 
#endif
  }

  // pw-exchange
  {
    MPI_Barrier(comm->c);
    double t0 = MPI_Wtime(); 
    comm_req *req = &pwd->req[pwd->comm[recv].n]; 
    const struct pw_comm_data *c = &pwd->comm[send];
    const uint *p, *pe, *size=c->size;
    uint bufOffset = pwd->comm[recv].total*unit_size;
    for(p=c->p,pe=p+c->n;p!=pe;++p) {
      size_t len = *(size++)*unit_size;
#ifdef USE_GPU_AWARE_MPI
      char *sendbuf = (char*) o_buff.ptr() + bufOffset;
#else
      char *sendbuf = (char*) buff + bufOffset;
#endif
      MPI_Isend((void*) sendbuf,len,MPI_UNSIGNED_CHAR,*p,comm->id,comm->c,req++);
      bufOffset += len;
    }
    MPI_Waitall(pwd->comm[send].n + pwd->comm[recv].n,pwd->req,MPI_STATUSES_IGNORE);
    MPI_Barrier(comm->c);
    eTime_pw += MPI_Wtime() - t0;
    //if(comm->id ==0) printf("time pw_exchange: %g s\n", eTime_pw);
  }

  // gahter
  {
    char *buf = (char *) buff;
#ifdef USE_GPU_GSBUF

    occa::memory o_buf = o_buff;
#ifndef USE_GPU_AWARE_MPI
    o_buf.copyFrom(buf,pwd->comm[recv].total*unit_size); 
#endif
    occaGather(ogs->NhaloGather, o_gatherOffsets, o_gatherIds, type, ogsAdd, o_buf, o_u);

#else
    //gs_gather(u,(char*) buff,vn,pwd->map[recv],dom,op);
    mygather_doubleAdd(ogs->NhaloGather, gatherOffsets, gatherIds, (double *) buf, (double *) u);
#endif
  }

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

#ifndef USE_GPU_GSBUF
    ogs->device.finish();
    ogs->device.setStream(ogs::dataStream);
    timer::tic("gs_memcpy");
    ogs::o_haloBuf.copyTo(ogs::haloBuf, ogs->NhaloGather*Nbytes, 0, "async: true");
    timer::toc("gs_memcpy");
    ogs->device.setStream(ogs::defaultStream);
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
    ogs->device.setStream(ogs::dataStream);
#ifndef USE_GPU_GSBUF
    ogs->device.finish();
#endif

    timer::tic("gs_host");
#ifdef USE_GPU_GSBUF
    ogs->device.finish(); // just for timing 
    myHostGatherScatter(ogs::o_haloBuf, gs_double, gs_add, 0, ogs);
    ogs->device.finish(); // just for timing
#else
    //ogsHostGatherScatter(ogs::haloBuf, type, op, ogs->haloGshSym);
    myHostGatherScatter(ogs::haloBuf, gs_double, gs_add, 0, ogs);
#endif
    timer::toc("gs_host");

#ifndef USE_GPU_GSBUF
    timer::tic("gs_memcpy");
    ogs::o_haloBuf.copyFrom(ogs::haloBuf, ogs->NhaloGather*Nbytes, 0, "async: true");
    timer::toc("gs_memcpy");
#endif

    timer::tic("scatter");
    occaScatter(ogs->NhaloGather, ogs->o_haloGatherOffsets, ogs->o_haloGatherIds, type, op, ogs::o_haloBuf, o_v);
    timer::toc("scatter");

    ogs->device.finish();
    ogs->device.setStream(ogs::defaultStream);
  }
}


#ifdef __cplusplus
}
#endif
