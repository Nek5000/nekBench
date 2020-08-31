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

struct options_t {
    size_t min_message_size;
    size_t max_message_size;
    size_t iterations;
    size_t iterations_large;
    size_t skip;
    size_t skip_large;
    size_t pairs;
};

static void multi_latency(MPI_Comm comm);
static void single_latency(int nmessages, MPI_Comm comm);

// GLOBALS
struct options_t options;
char *s_buf, *r_buf;

extern "C" { // Begin C Linkage
int pingPongMulti(int pairs, int useDevice, occa::device device, MPI_Comm comm)
{

    int size, rank;
    MPI_Comm_size(comm,&size);
    MPI_Comm_rank(comm,&rank);

    if(size == 0 || size%2 != 0) return 1;

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

int pingPongSingle(int nmessages, int useDevice, occa::device device, MPI_Comm comm)
{

    int size, rank;
    MPI_Comm_size(comm,&size);
    MPI_Comm_rank(comm,&rank);

    options.min_message_size = 0;
    options.max_message_size = 1 << 20;

    options.iterations = LAT_LOOP_SMALL;
    options.skip = LAT_SKIP_SMALL;
    options.iterations_large = LAT_LOOP_LARGE;
    options.skip_large = LAT_SKIP_LARGE;

    occa::memory o_s_buf;
    occa::memory o_r_buf;

    if(useDevice) {
      o_s_buf = device.malloc(options.max_message_size*nmessages);
      o_r_buf = device.malloc(options.max_message_size*nmessages);
      s_buf = (char*) o_s_buf.ptr();
      r_buf = (char*) o_s_buf.ptr();
    } else {
      unsigned long align_size = sysconf(_SC_PAGESIZE);
      posix_memalign((void**)&s_buf, align_size, options.max_message_size*nmessages);
      posix_memalign((void**)&r_buf, align_size, options.max_message_size*nmessages);
    }

    if(rank == 0) {
      if(nmessages == 1) {
          if(size > 2)
              printf("\n\nping pong single - ranks 1-%d to rank 0, useDevice: %d\n\n", size-1, useDevice);
          else
              printf("\n\nping pong single - ranks 1 to rank 0, useDevice: %d\n\n", useDevice);
      } else {
        if(size > 2)
              printf("\n\nping pong pairwise exchange - ranks 1-%d to rank 0, useDevice: %d\n\n", size-1, useDevice);
          else
              printf("\n\nping pong pairwise exchange - ranks 1 to rank 0, useDevice: %d\n\n", useDevice);
      }
      fflush(stdout);
    }

    MPI_CHECK(MPI_Barrier(comm));
    single_latency(nmessages, comm);
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

    for(size = options.min_message_size; size <= options.max_message_size; size  = (size ? size * 2 : 1)) {

        MPI_CHECK(MPI_Barrier(comm));

        if(size > LARGE_MESSAGE_SIZE) {
            options.iterations = options.iterations_large;
            options.skip = options.skip_large;
        } else {
            options.iterations = options.iterations;
            options.skip = options.skip;
        }

        MPI_Request rReq, sReq;

        if (rank < pairs) {
            partner = rank + pairs;

            for (i = 0; i < options.iterations + options.skip; i++) {

                if (i == options.skip) {
                    t_start = MPI_Wtime();
                    MPI_CHECK(MPI_Barrier(comm));
                }

                MPI_CHECK(MPI_Isend(s_buf, size, MPI_CHAR, partner, 1, comm, &sReq));
                MPI_CHECK(MPI_Irecv(r_buf, size, MPI_CHAR, partner, 1, comm, &rReq));

                MPI_CHECK(MPI_Wait(&rReq, MPI_STATUS_IGNORE));
                MPI_CHECK(MPI_Wait(&sReq, MPI_STATUS_IGNORE));

            }

            t_end = MPI_Wtime();

        } else {
            partner = rank - pairs;

            for (i = 0; i < options.iterations + options.skip; i++) {

                if (i == options.skip) {
                    t_start = MPI_Wtime();
                    MPI_CHECK(MPI_Barrier(comm));
                }

                MPI_CHECK(MPI_Irecv(r_buf, size, MPI_CHAR, partner, 1, comm, &rReq));
                MPI_CHECK(MPI_Isend(s_buf, size, MPI_CHAR, partner, 1, comm, &sReq));

                MPI_CHECK(MPI_Wait(&rReq, MPI_STATUS_IGNORE));
                MPI_CHECK(MPI_Wait(&sReq, MPI_STATUS_IGNORE));

            }

            t_end = MPI_Wtime();
        }

        latency = (t_end - t_start) * 1.0e6 / (2.0 * options.iterations);

        MPI_CHECK(MPI_Reduce(&latency, &total_lat, 1, MPI_DOUBLE, MPI_SUM, 0,
                   comm));

        avg_lat = total_lat/(double) (pairs * 2);

        if(0 == rank) {
            fprintf(stdout, "%-*d%*.*f\n", 10, size, FIELD_WIDTH,
                    FLOAT_PRECISION, avg_lat);
            fflush(stdout);
        }
    }
}

static void single_latency(int nmessages, MPI_Comm comm) {

  int myRank, mpiSize;
  MPI_Comm_rank(comm, &myRank);
  MPI_Comm_size(comm, &mpiSize);

  int loopCounter = 0;
  for(int size = options.min_message_size; size <= options.max_message_size; size=((int)(size*1.1) > size ? (int)(size*1.1) : size+1))
    loopCounter += 1;

  int *all_sizes;
  double *all_min, *all_max, *all_avg;

  all_sizes = (int*)calloc(loopCounter, sizeof(int));
  all_min = (double*)calloc(loopCounter, sizeof(double));
  all_max = (double*)calloc(loopCounter, sizeof(double));
  all_avg = (double*)calloc(loopCounter, sizeof(double));

  FILE *fp;
  if(0 == myRank) {

    time_t rawtime;
    struct tm *timeinfo;
    time(&rawtime);
    timeinfo = localtime(&rawtime);
    char timebuffer[80];
    char buffer[80];
    strftime(timebuffer,80,"%Y_%m_%d_%R.txt", timeinfo);
    sprintf(buffer, "pingpong_nmessages_%i_%s", nmessages, timebuffer);
    fp = fopen(buffer, "w");
    fprintf(fp, "%-10s %-10s %-10s %-10s %-10s %-15s %-15s\n", "sender", "total", "receiver", "loopcount", "n messages", "bytes", "timing");

  }

  int maxRank = std::min(512, mpiSize);
  for(int iRank = 1; iRank < maxRank; iRank = (iRank<32||maxRank<128 ? iRank+1 : (maxRank-1-iRank < 5 ? maxRank-1 : iRank+5))) {

    int iSize = 0;

    for(int size = options.min_message_size; size <= options.max_message_size; size=((int)(size*1.1) > size ? (int)(size*1.1) : size+1)) {

      MPI_CHECK(MPI_Barrier(comm));

      if(size > LARGE_MESSAGE_SIZE) {
        options.iterations = options.iterations_large;
        options.skip = options.skip_large;
      } else {
        options.iterations = options.iterations;
        options.skip = options.skip;
      }

      double latency = 0;

      MPI_Request rReq[nmessages], sReq[nmessages];

      if(myRank == iRank) {

        double t_start = 0;

        for(int i = 0; i < options.iterations + options.skip; i++) {

          if(i == options.skip)
            t_start = MPI_Wtime();

          for(int iMessage = 0; iMessage < nmessages; ++iMessage) {
            MPI_CHECK(MPI_Isend(&s_buf[iMessage*options.max_message_size], size, MPI_CHAR, 0, iMessage, comm, &rReq[iMessage]));
            MPI_CHECK(MPI_Irecv(&r_buf[iMessage*options.max_message_size], size, MPI_CHAR, 0, iMessage, comm, &sReq[iMessage]));
          }

          for(int iMessage = 0; iMessage < nmessages; ++iMessage) {
            MPI_CHECK(MPI_Wait(&rReq[iMessage], MPI_STATUS_IGNORE));
            MPI_CHECK(MPI_Wait(&sReq[iMessage], MPI_STATUS_IGNORE));
          }

        }

        double t_end = MPI_Wtime();

        latency = (t_end - t_start) * 1.0e6 / (2.0 * options.iterations);
        MPI_CHECK(MPI_Send(&latency, 1, MPI_DOUBLE, 0, 2, comm));

      } else if(myRank == 0) {

        double t_start = 0;

        for(int i = 0; i < options.iterations + options.skip; i++) {

          if(i == options.skip)
            t_start = MPI_Wtime();

          for(int iMessage = 0; iMessage < nmessages; ++iMessage) {
            MPI_CHECK(MPI_Irecv(&r_buf[iMessage*options.max_message_size], size, MPI_CHAR, iRank, iMessage, comm, &rReq[iMessage]));
            MPI_CHECK(MPI_Isend(&s_buf[iMessage*options.max_message_size], size, MPI_CHAR, iRank, iMessage, comm, &sReq[iMessage]));
          }

          for(int iMessage = 0; iMessage < nmessages; ++iMessage) {
            MPI_CHECK(MPI_Wait(&rReq[iMessage], MPI_STATUS_IGNORE));
            MPI_CHECK(MPI_Wait(&sReq[iMessage], MPI_STATUS_IGNORE));
          }

        }

        double t_end = MPI_Wtime();

        latency = (t_end - t_start) * 1.0e6 / (2.0 * options.iterations);

      } // all other ranks are idle

      MPI_CHECK(MPI_Barrier(comm));

      if(iRank == 1)
        all_sizes[iSize] = size;

      if(0 == myRank) {

        double peer_latency;
        MPI_CHECK(MPI_Recv(&peer_latency, 1, MPI_DOUBLE, iRank, 2, comm, MPI_STATUS_IGNORE));
        double avg_lat = (latency+peer_latency)/2.0;

        if(all_min[iSize] > avg_lat || all_min[iSize] == 0)
          all_min[iSize] = avg_lat;
        if(all_max[iSize] < avg_lat)
          all_max[iSize] = avg_lat;
        all_avg[iSize] += avg_lat;

        fprintf(fp, "%-10d %-10d %-10d %-10d %-10d %-15d %-15f\n", iRank, mpiSize, 0, options.iterations, nmessages, size, avg_lat);

      }

      ++iSize;

    }

    if(iRank == maxRank-1)
      break;

  }

  if(myRank == 0) {
    printf("%-10s %-13s %-13s %-13s\n", "bytes", "average", "minimum", "maximum");
    for(int i = 0; i < loopCounter; ++i) {
      printf("%-10d %-13f %-13f %-13f\n", all_sizes[i], all_avg[i]/((double)(mpiSize-1)), all_min[i], all_max[i]);
    }
    double avg_sum = 0;
    for(int i = 0; i < loopCounter; ++i)
      avg_sum += all_avg[i];
    printf("\nGlobal average: %f\n\n", avg_sum/(double)loopCounter);
    fflush(stdout);

    fclose(fp);

  }

  free(all_sizes);
  free(all_min);
  free(all_max);
  free(all_avg);

}
