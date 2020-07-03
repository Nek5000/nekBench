/*
 * TODO:
 *  - Add device support for gs_init
 *  - Fix alignments issues of device MPI buffers
 *  - Add occaGather kernel initializing result with original value not zero 
 */

#include <omp.h>
#include <occa.hpp>
#include "ogs.hpp"
#include "ogsKernels.hpp"
#include "ogsInterface.h"
#include "timer.hpp"

#ifdef __cplusplus
extern "C" {
#endif

enum ogs_mode { OGS_DEFAULT, OGS_HOSTMPI, OGS_DEVICEMPI };

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

// GLOBALS
static occa::memory h_buffSend, h_buffRecv;
static unsigned char *bufSend, *bufRecv;
static occa::memory o_bufSend, o_bufRecv;

static int *scatterOffsets, *gatherOffsets;
static int *scatterIds, *gatherIds;
static occa::memory o_scatterOffsets, o_gatherOffsets;
static occa::memory o_scatterIds, o_gatherIds;

static int enabledTimer = 0;

static const double gs_identity_double[] = { 0, 1, 1.7976931348623157e+308, -1.7976931348623157e+308, 0 };

#ifdef __cplusplus
}
#endif

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

    gatherq[g] += gq;
    //gatherq[g] = gq;
  }
}


static void convertMap(const uint *restrict map,
                       int *restrict starts,
                       int *restrict ids){

  int i, j, n=0, s=0;
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

void mygsSetup(ogs_t *ogs, int timer)
{
  const unsigned transpose = 0;
  struct gs_data *gsh = (gs_data*) ogs->haloGshSym;
  const unsigned recv = 0^transpose, send = 1^transpose;
  const void* execdata = gsh->r.data;
  const struct pw_data *pwd = (pw_data*) execdata;
  const unsigned Nhalo = ogs->NhaloGather;
  const unsigned vn = 1;
  const unsigned unit_size = vn*sizeof(double);

  enabledTimer = timer;
  if(Nhalo == 0) return;
  occa::properties props;
  props["mapped"] = true;

  h_buffSend = ogs->device.malloc(pwd->comm[send].total*unit_size, props);
  bufSend = (unsigned char*)h_buffSend.ptr(props); 
  scatterOffsets = (int*) calloc(2*Nhalo,sizeof(int));
  scatterIds = (int*) calloc(pwd->comm[send].total,sizeof(int));
  convertMap(pwd->map[send], scatterOffsets, scatterIds);

  o_bufSend = ogs->device.malloc(pwd->comm[send].total*unit_size);
  o_scatterOffsets = ogs->device.malloc(2*Nhalo*sizeof(int), scatterOffsets);
  o_scatterIds = ogs->device.malloc(pwd->comm[send].total*sizeof(int), scatterIds);

  h_buffRecv = ogs->device.malloc(pwd->comm[recv].total*unit_size, props);
  bufRecv = (unsigned char*)h_buffRecv.ptr(props);
  gatherOffsets  = (int*) calloc(2*Nhalo,sizeof(int));
  gatherIds  = (int*) calloc(pwd->comm[recv].total,sizeof(int));
  convertMap(pwd->map[recv], gatherOffsets, gatherIds);

  o_bufRecv = ogs->device.malloc(pwd->comm[recv].total*unit_size);
  o_gatherOffsets  = ogs->device.malloc(2*Nhalo*sizeof(int), gatherOffsets);
  o_gatherIds  = ogs->device.malloc(pwd->comm[recv].total*sizeof(int), gatherIds);
}

static void myHostGatherScatter(occa::memory o_u,
                                const char *type, const char *op, 
                                ogs_t *ogs, ogs_mode ogs_mode)
{
  struct gs_data *gsh = (gs_data*) ogs->haloGshSym;
  const void* execdata = gsh->r.data; 
  const struct pw_data *pwd = (pw_data*) execdata; 
  const struct comm *comm = &gsh->comm;
  const unsigned Nhalo = ogs->NhaloGather;

  // hardwired for now
  const unsigned transpose = 0;
  const unsigned recv = 0^transpose, send = 1^transpose;

  size_t unit_size;
  if (!strcmp(type, "float"))
    unit_size = sizeof(float);
  else if (!strcmp(type, "double"))
    unit_size  = sizeof(double);
  else if (!strcmp(type, "int"))
    unit_size  = sizeof(int);
  else if (!strcmp(type, "long long int"))
    unit_size  = sizeof(long long int);

  { // prepost recv
    if(enabledTimer) {
      MPI_Barrier(comm->c);
      timer::hostTic("pw_exec");
    }

    comm_req *req = pwd->req; 
    const struct pw_comm_data *c = &pwd->comm[recv];
    const uint *p, *pe, *size=c->size;
    uint bufOffset = 0;
    for(p=c->p,pe=p+c->n;p!=pe;++p) {
      size_t len = *(size++)*unit_size;
      unsigned char *recvbuf = (unsigned char *)bufRecv + bufOffset;
      if(ogs_mode == OGS_DEVICEMPI) recvbuf = (unsigned char*)o_bufRecv.ptr() + bufOffset;
      MPI_Irecv((void*)recvbuf,len,MPI_UNSIGNED_CHAR,*p,*p,comm->c,req++);
      bufOffset += len;
    }

    if(enabledTimer)  timer::hostToc("pw_exec");
  }

  { // scatter
    if(enabledTimer) timer::deviceTic("pack");
    occaScatter(Nhalo, o_scatterOffsets, o_scatterIds, type, op, o_u, o_bufSend);
    if(enabledTimer) timer::deviceToc("pack");

    if(ogs_mode == OGS_HOSTMPI) {
      if(enabledTimer) timer::deviceTic("gs_memcpy_dh");
      o_bufSend.copyTo(bufSend, pwd->comm[send].total*unit_size, 0, "async: true");
      if(enabledTimer) timer::deviceToc("gs_memcpy_dh");
    }
  }

  { // pw exchange
    ogs->device.finish(); // waiting for buffers to be ready
    MPI_Barrier(comm->c);
    if(enabledTimer) {
      timer::hostUpdate("pw_exec");
      timer::hostTic("pw_exec");
    }

    comm_req *req = &pwd->req[pwd->comm[recv].n]; 
    const struct pw_comm_data *c = &pwd->comm[send];
    const uint *p, *pe, *size=c->size;
    uint bufOffset = 0;
    for(p=c->p,pe=p+c->n;p!=pe;++p) {
      size_t len = *(size++)*unit_size;
      unsigned char *sendbuf = (unsigned char*)bufSend + bufOffset;
      if(ogs_mode == OGS_DEVICEMPI) sendbuf = (unsigned char*)o_bufSend.ptr() + bufOffset;
      MPI_Isend((void*)sendbuf,len,MPI_UNSIGNED_CHAR,*p,comm->id,comm->c,req++);
      bufOffset += len;
    }
    MPI_Waitall(pwd->comm[send].n + pwd->comm[recv].n,pwd->req,MPI_STATUSES_IGNORE);

    if(enabledTimer) timer::hostToc("pw_exec");
  }

  { // gather
    if(ogs_mode == OGS_HOSTMPI){
      if(enabledTimer) timer::deviceTic("gs_memcpy_hd");
      o_bufRecv.copyFrom(bufRecv,pwd->comm[recv].total*unit_size, 0, "async: true");
      if(enabledTimer) timer::deviceToc("gs_memcpy_hd");
    }

    if(enabledTimer) timer::deviceTic("unpack");
    occaGather(Nhalo, o_gatherOffsets, o_gatherIds, type, op, o_bufRecv, o_u);
    if(enabledTimer) timer::deviceToc("unpack");
  }

}

static void myHostGatherScatter(void *u, ogs_t *ogs)
{
  struct gs_data *gsh = (gs_data*) ogs->haloGshSym;
  const void* execdata = gsh->r.data; 
  const struct pw_data *pwd = (pw_data*) execdata; 
  const struct comm *comm = &gsh->comm;
  const unsigned Nhalo = ogs->NhaloGather;

  // hardwired for now
  const unsigned transpose = 0;
  const unsigned recv = 0^transpose, send = 1^transpose;
  const unsigned vn = 1;
  const unsigned unit_size = vn*sizeof(double);
  const char type[] = "double";
  const char op[]   = "add";

  { // prepost recv
    MPI_Barrier(comm->c);
    if(enabledTimer) timer::tic("pw_exec");

    comm_req *req = pwd->req; 
    const struct pw_comm_data *c = &pwd->comm[recv];
    const uint *p, *pe, *size=c->size;
    uint bufOffset = 0;
    for(p=c->p,pe=p+c->n;p!=pe;++p) {
      size_t len = *(size++)*unit_size;
      unsigned char *recvbuf = (unsigned char *)bufRecv + bufOffset;
      MPI_Irecv((void*)recvbuf,len,MPI_UNSIGNED_CHAR,*p,*p,comm->c,req++);
      bufOffset += len;
    }

    if(enabledTimer) timer::toc("pw_exec");
  }

  { // scatter
    if(enabledTimer) timer::tic("pack");
    myscatter_double(Nhalo, scatterOffsets, scatterIds, (double *) u, (double *) bufSend); 
    if(enabledTimer) timer::toc("pack");
  }

  { // pw exchange
    MPI_Barrier(comm->c);
    if(enabledTimer) timer::update("pw_exec");
    if(enabledTimer) timer::tic("pw_exec");

    comm_req *req = &pwd->req[pwd->comm[recv].n]; 
    const struct pw_comm_data *c = &pwd->comm[send];
    const uint *p, *pe, *size=c->size;
    uint bufOffset = 0;
    for(p=c->p,pe=p+c->n;p!=pe;++p) {
      size_t len = *(size++)*unit_size;
      unsigned char *sendbuf = (unsigned char*)bufSend + bufOffset;
      MPI_Isend((void*)sendbuf,len,MPI_UNSIGNED_CHAR,*p,comm->id,comm->c,req++);
      bufOffset += len;
    }
    MPI_Waitall(pwd->comm[send].n + pwd->comm[recv].n,pwd->req,MPI_STATUSES_IGNORE);

    if(enabledTimer) timer::toc("pw_exec");
  }

  { // gather
    if(enabledTimer) timer::tic("unpack");
    mygather_doubleAdd(Nhalo, gatherOffsets, gatherIds, (double *) bufRecv, (double *) u);
    if(enabledTimer) timer::toc("unpack");
  }

}

void mygsStart(occa::memory o_v, const char *type, const char *op, ogs_t *ogs, ogs_mode ogs_mode) 
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
    if(enabledTimer) timer::deviceTic("gather_halo");
    occaGather(ogs->NhaloGather, ogs->o_haloGatherOffsets, ogs->o_haloGatherIds, type, op, o_v, ogs::o_haloBuf);
    if(enabledTimer) timer::deviceToc("gather_halo");
    ogs->device.finish(); // just in case dataStream is non-blocking 

    if(ogs_mode == OGS_DEFAULT) {
      ogs->device.setStream(ogs::dataStream);
      if(enabledTimer) timer::deviceTic("gs_memcpy_dh");
      ogs::o_haloBuf.copyTo(ogs::haloBuf, ogs->NhaloGather*Nbytes, 0, "async: true");
      if(enabledTimer) timer::deviceToc("gs_memcpy_dh");
      ogs->device.setStream(ogs::defaultStream);
    }
  }
}

void mygsFinish(occa::memory o_v, const char *type, const char *op, ogs_t *ogs, ogs_mode ogs_mode) 
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
    if(enabledTimer) timer::deviceTic("gs_interior");
    occaGatherScatter(ogs->NlocalGather, ogs->o_localGatherOffsets, ogs->o_localGatherIds, type, op, o_v);
    if(enabledTimer) timer::deviceToc("gs_interior");
  }

  if (ogs->NhaloGather) {
    ogs->device.setStream(ogs::dataStream);
    if(ogs_mode == OGS_DEFAULT) {
      ogs->device.finish(); // waiting for ogs::haloBuf copy to finish 
      if(enabledTimer) timer::hostTic("gs_host");
      ogsHostGatherScatter(ogs::haloBuf, type, op, ogs->haloGshSym);
      //myHostGatherScatter(ogs::haloBuf, ogs);
      if(enabledTimer) timer::hostToc("gs_host");
    } else {
      myHostGatherScatter(ogs::o_haloBuf, type, op, ogs, ogs_mode);
    }   
 
    if(ogs_mode == OGS_DEFAULT) { 
      if(enabledTimer) timer::deviceTic("gs_memcpy_hd");
      ogs::o_haloBuf.copyFrom(ogs::haloBuf, ogs->NhaloGather*Nbytes, 0, "async: true");
      if(enabledTimer) timer::deviceToc("gs_memcpy_hd");
    }

    ogs->device.finish();
    ogs->device.setStream(ogs::defaultStream);

    if(enabledTimer) timer::deviceTic("scatter");
    occaScatter(ogs->NhaloGather, ogs->o_haloGatherOffsets, ogs->o_haloGatherIds, type, op, ogs::o_haloBuf, o_v);
    if(enabledTimer) timer::deviceToc("scatter");
  }
}
