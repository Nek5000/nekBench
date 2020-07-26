#include <mpi.h>
#include <assert.h>
#include <unistd.h>
#include <occa.hpp>

#define FIELD_WIDTH 20
#define FLOAT_PRECISION 2
#define LARGE_MESSAGE_SIZE 8192
#define LAT_LOOP_SMALL 10000
#define LAT_SKIP_SMALL 100
#define LAT_LOOP_LARGE 1000
#define LAT_SKIP_LARGE 10

#define MPI_CHECK(stmt)                                          \
  do {                                                             \
    int mpi_errno = (stmt);                                       \
    if (MPI_SUCCESS != mpi_errno) {                               \
      fprintf(stderr, "[%s:%d] MPI call failed with %d \n",     \
              __FILE__, __LINE__,mpi_errno);                           \
      exit(EXIT_FAILURE);                                       \
    }                                                             \
    assert(MPI_SUCCESS == mpi_errno);                             \
  } while (0)

struct options_t
{
  size_t min_message_size;
  size_t max_message_size;
  size_t iterations;
  size_t iterations_large;
  size_t skip;
  size_t skip_large;
  size_t pairs;
};

static void multi_latency(MPI_Comm comm);

// GLOBALS
struct options_t options;
char* s_buf, * r_buf;

extern "C" { // Begin C Linkage
int pingPongMulti(int pairs, int useDevice, occa::device device, MPI_Comm comm)
{
  int size, rank;
  MPI_Comm_size(comm,&size);
  MPI_Comm_rank(comm,&rank);

  if(size == 0 || size % 2 != 0) return 1;

  options.min_message_size = 0;
  options.max_message_size = 1 << 20;
  options.pairs = pairs;

  options.iterations = LAT_LOOP_SMALL;
  options.skip = LAT_SKIP_SMALL;
  options.iterations_large = LAT_LOOP_LARGE;
  options.skip_large = LAT_SKIP_LARGE;

  occa::memory o_s_buf;
  occa::memory o_r_buf;

  if(useDevice) {
    o_s_buf = device.malloc(options.max_message_size);
    o_r_buf = device.malloc(options.max_message_size);
    s_buf = (char*) o_s_buf.ptr();
    r_buf = (char*) o_s_buf.ptr();
  } else {
    unsigned long align_size = sysconf(_SC_PAGESIZE);
    posix_memalign((void**)&s_buf, align_size, options.max_message_size);
    posix_memalign((void**)&r_buf, align_size, options.max_message_size);
  }

  if(rank == 0) {
    printf("\nping pong multi - pairs: %d useDevice: %d\n", pairs, useDevice);
    fflush(stdout);
  }

  MPI_CHECK(MPI_Barrier(comm));
  multi_latency(comm);
  MPI_CHECK(MPI_Barrier(comm));

  if(useDevice) {
    o_s_buf.free();
    o_r_buf.free();
  } else {
    free(s_buf);
    free(r_buf);
  }

  return EXIT_SUCCESS;
}
} // end C Linkage

static void multi_latency(MPI_Comm comm)
{
  int size, partner;
  int i;
  double t_start = 0.0, t_end = 0.0,
         latency = 0.0, total_lat = 0.0,
         avg_lat = 0.0;

  int rank;
  MPI_Comm_rank(comm,&rank);
  MPI_Status reqstat;

  int pairs = options.pairs;

  for(size = options.min_message_size; size <= options.max_message_size;
      size  = (size ? size * 2 : 1)) {
    MPI_CHECK(MPI_Barrier(comm));

    if(size > LARGE_MESSAGE_SIZE) {
      options.iterations = options.iterations_large;
      options.skip = options.skip_large;
    } else {
      options.iterations = options.iterations;
      options.skip = options.skip;
    }

    if (rank < pairs) {
      partner = rank + pairs;

      for (i = 0; i < options.iterations + options.skip; i++) {
        if (i == options.skip) {
          t_start = MPI_Wtime();
          MPI_CHECK(MPI_Barrier(comm));
        }

        MPI_CHECK(MPI_Send(s_buf, size, MPI_CHAR, partner, 1, comm));
        MPI_CHECK(MPI_Recv(r_buf, size, MPI_CHAR, partner, 1, comm,
                           &reqstat));
      }

      t_end = MPI_Wtime();
    } else {
      partner = rank - pairs;

      for (i = 0; i < options.iterations + options.skip; i++) {
        if (i == options.skip) {
          t_start = MPI_Wtime();
          MPI_CHECK(MPI_Barrier(comm));
        }

        MPI_CHECK(MPI_Recv(r_buf, size, MPI_CHAR, partner, 1, comm,
                           &reqstat));
        MPI_CHECK(MPI_Send(s_buf, size, MPI_CHAR, partner, 1, comm));
      }

      t_end = MPI_Wtime();
    }

    latency = (t_end - t_start) * 1.0e6 / (2.0 * options.iterations);

    MPI_CHECK(MPI_Reduce(&latency, &total_lat, 1, MPI_DOUBLE, MPI_SUM, 0,
                         comm));

    avg_lat = total_lat / (double) (pairs * 2);

    if(0 == rank) {
      fprintf(stdout, "%-*d%*.*f\n", 10, size, FIELD_WIDTH,
              FLOAT_PRECISION, avg_lat);
      fflush(stdout);
    }
  }
}
