#include <occa.hpp>
#include "ogs.hpp"
#include "ogsKernels.hpp"
#include "ogsInterface.h"
#include "timer.hpp"



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

static void my_scatter_to_buf(
  double *restrict out, const unsigned out_stride,
  const double *restrict in, const unsigned in_stride,
  const uint *restrict map)
{
  uint i,j;
  while((i=*map++)!=UINT_MAX) {

    double t=in[i*in_stride];
    j=*map++; 

    do {
      out[j*out_stride]=t;
    } while((j=*map++)!=UINT_MAX);

  }
}

static buffer buff = null_buffer;
static occa::memory o_buff;
static int firstTime = 1;

//#define MPI_CPU_BUFFER
static void myHostGatherScatter(void *u, gs_dom dom, gs_op op,
                                unsigned transpose, ogs_t *ogs)
{
  struct gs_data *gsh = (gs_data*) ogs->haloGshSym;
  const unsigned recv = 0^transpose, send = 1^transpose;
  const void* execdata = gsh->r.data; 
  const struct pw_data *pwd = (pw_data*) execdata; 
  const struct comm *comm = &gsh->comm;
  const unsigned vn = 1;
  const unsigned unit_size = vn*sizeof(double); //gs_dom_size[dom];
  const gs_mode mode = mode_plain;
  double eTime_pw = 0;

  if(firstTime) {
    buffer_reserve(&buff, gsh->r.buffer_size*sizeof(double)); // just something large
#ifdef MPI_GPU_BUFFER
    o_buff = ogs->device->malloc(gsh->r.buffer_size*sizeof(double));
#endif
    firstTime = 0;
  }

  char *buf = (char *) buff.ptr;
#ifdef MPI_GPU_BUFFER
  occa::memory o_buf = o_buff; 
#endif
  
  if(transpose==0) gs_init(u,vn,gsh->flagged_primaries,dom,op);

  {
    MPI_Barrier(comm->c);
    double t0 = MPI_Wtime(); 
    comm_req *req = pwd->req; 
    const struct pw_comm_data *c = &pwd->comm[recv];
    const uint *p, *pe, *size=c->size;
    for(p=c->p,pe=p+c->n;p!=pe;++p) {
      size_t len = *(size++)*unit_size;
#ifdef MPI_GPU_BUFFER
      void *recvbuf = o_buf->ptr();
#else
      void *recvbuf = buf;
#endif
      MPI_Irecv(recvbuf,len,MPI_UNSIGNED_CHAR,*p,*p,comm->c,req++); 
      buf += len;
#ifdef MPI_GPU_BUFFER
      o_buf += len;
#endif
    }
    eTime_pw += MPI_Wtime() - t0;
  }

  gs_scatter(buf,u,vn,pwd->map[send],dom); 
  //my_scatter_to_buf((double *) sendbuf, 1, (double *) u, 1, pwd->map[send]); 
#ifdef MPI_GPU_BUFFER
  o_buff.copyFrom(buff.ptr);  
#endif
 
  {
    MPI_Barrier(comm->c);
    double t0 = MPI_Wtime(); 
    comm_req *req = &pwd->req[pwd->comm[recv].n]; 
    const struct pw_comm_data *c = &pwd->comm[send];
    const uint *p, *pe, *size=c->size;
    for(p=c->p,pe=p+c->n;p!=pe;++p) {
      size_t len = *(size++)*unit_size;
#ifdef MPI_GPU_BUFFER
      void *recvbuf = o_buf->ptr();
#else
      void *sendbuf = buf;
#endif
      MPI_Isend(sendbuf,len,MPI_UNSIGNED_CHAR,*p,comm->id,comm->c,req++);
      buf += len;
#ifdef MPI_GPU_BUFFER
      o_buf += len;
#endif
    }
    MPI_Waitall(pwd->comm[send].n + pwd->comm[recv].n,pwd->req,MPI_STATUSES_IGNORE);
    MPI_Barrier(comm->c);
    eTime_pw += MPI_Wtime() - t0;
  }

  if(comm->id ==0) printf("time pw_exchange: %g s\n", eTime_pw);

#ifdef MPI_GPU_BUFFER
  o_buff.copyTo(buff.ptr);  
#endif
 
  gs_gather(u,(char*) buff.ptr,vn,pwd->map[recv],dom,op);
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

#ifdef ASYNC
    ogs->device.finish();
    ogs->device.setStream(ogs::dataStream);
    timer::tic("gs_memcpy");
    ogs::o_haloBuf.copyTo(ogs::haloBuf, ogs->NhaloGather*Nbytes, 0, "async: true");
    timer::toc("gs_memcpy");
    ogs->device.setStream(ogs::defaultStream);
#else    
    timer::tic("gs_memcpy");
    ogs::o_haloBuf.copyTo(ogs::haloBuf, ogs->NhaloGather*Nbytes);
    timer::toc("gs_memcpy");
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
#ifdef ASYNC
    ogs->device.setStream(ogs::dataStream);
    //timer::tic("gs_memcpy");
    ogs->device.finish();
    //timer::toc("gs_memcpy");
#endif

    timer::tic("gs_host");
    //ogsHostGatherScatter(ogs::haloBuf, type, op, ogs->haloGshSym);
    myHostGatherScatter(ogs::haloBuf, gs_double, gs_add, 0, ogs);
    timer::toc("gs_host");

#ifdef ASYNC
    timer::tic("gs_memcpy");
    ogs::o_haloBuf.copyFrom(ogs::haloBuf, ogs->NhaloGather*Nbytes, 0, "async: true");
    timer::toc("gs_memcpy");
#else    
    timer::tic("gs_memcpy");
    ogs::o_haloBuf.copyFrom(ogs::haloBuf, ogs->NhaloGather*Nbytes);
    timer::toc("gs_memcpy");
#endif

    timer::tic("scatter");
    occaScatter(ogs->NhaloGather, ogs->o_haloGatherOffsets, ogs->o_haloGatherIds, type, op, ogs::o_haloBuf, o_v);
    timer::toc("scatter");

#ifdef ASYNC
    //timer::tic("gs_memcpy");
    ogs->device.finish();
    //timer::toc("gs_memcpy");
    ogs->device.setStream(ogs::defaultStream);
#endif    
  }
}


#ifdef __cplusplus
}
#endif
