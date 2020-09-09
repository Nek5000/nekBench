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

static void pingpong(MPI_Comm comm);
static void pairexchange(int nmessages, MPI_Comm comm);

// GLOBALS
struct options_t options;
int *s_buf, *r_buf;

extern "C" { // Begin C Linkage

int pingPongSinglePair(int useDevice, occa::device device, MPI_Comm comm) {

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
    o_s_buf = device.malloc(options.max_message_size*sizeof(int));
    o_r_buf = device.malloc(options.max_message_size*sizeof(int));
    s_buf = (int*) o_s_buf.ptr();
    r_buf = (int*) o_s_buf.ptr();
  } else {
    unsigned long align_size = sysconf(_SC_PAGESIZE);
    posix_memalign((void**)&s_buf, align_size, options.max_message_size*sizeof(int));
    posix_memalign((void**)&r_buf, align_size, options.max_message_size*sizeof(int));
  }

  if(rank == 0) {
      printf("\npingpong - useDevice: %d\n", useDevice);
      fflush(stdout);
  }

  MPI_CHECK(MPI_Barrier(comm));
  pingpong(comm);
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

int multiPairExchange(int nmessages, int useDevice, occa::device device, MPI_Comm comm) {

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
    o_s_buf = device.malloc(nmessages*options.max_message_size*sizeof(int));
    o_r_buf = device.malloc(nmessages*options.max_message_size*sizeof(int));
    s_buf = (int*) o_s_buf.ptr();
    r_buf = (int*) o_s_buf.ptr();
  } else {
    unsigned long align_size = sysconf(_SC_PAGESIZE);
    posix_memalign((void**)&s_buf, align_size, nmessages*options.max_message_size*sizeof(int));
    posix_memalign((void**)&r_buf, align_size, nmessages*options.max_message_size*sizeof(int));
  }

  if(rank == 0) {
      printf("\npairwise exchange - n messages: %d,  useDevice: %d\n", nmessages, useDevice);
      fflush(stdout);
  }

  MPI_CHECK(MPI_Barrier(comm));
  pairexchange(nmessages, comm);
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

static void pingpong(MPI_Comm comm) {

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
    sprintf(buffer, "pingpong_%s", timebuffer);
    fp = fopen(buffer, "w");
    fprintf(fp, "%-10s %-10s %-10s %-10s %-15s %-15s\n", "sender", "total", "receiver", "loopcount", "bytes", "timing");

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

      if(myRank == iRank) {

        double t_start = 0;

        for(int i = 0; i < options.iterations + options.skip; i++) {

          if(i == options.skip)
            t_start = MPI_Wtime();

          MPI_CHECK(MPI_Recv(r_buf, size, MPI_INT, 0, 1, comm, MPI_STATUS_IGNORE));
          MPI_CHECK(MPI_Send(s_buf, size, MPI_INT, 0, 1, comm));

        }

        double t_end = MPI_Wtime();

        latency = (t_end - t_start) * 1.0e6 / (2.0 * options.iterations);
        MPI_CHECK(MPI_Send(&latency, 1, MPI_DOUBLE, 0, 2, comm));

      } else if(myRank == 0) {

        double t_start = 0;

        for(int i = 0; i < options.iterations + options.skip; i++) {

          if(i == options.skip)
            t_start = MPI_Wtime();

          MPI_CHECK(MPI_Send(s_buf, size, MPI_INT, iRank, 1, comm));
          MPI_CHECK(MPI_Recv(r_buf, size, MPI_INT, iRank, 1, comm, MPI_STATUS_IGNORE));

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

        fprintf(fp, "%-10d %-10d %-10d %-10d %-15d %-15f\n", iRank, mpiSize, 0, options.iterations, size*4, avg_lat);

      }

      ++iSize;

    }

    if(iRank == maxRank-1)
      break;

  }

  if(myRank == 0) {
    printf("%-10s %-13s %-13s %-13s\n", "bytes", "average", "minimum", "maximum");
    for(int i = 0; i < loopCounter; ++i) {
      printf("%-10d %-13f %-13f %-13f\n", all_sizes[i]*4, all_avg[i]/((double)(mpiSize-1)), all_min[i], all_max[i]);
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

static void pairexchange(int nmessages, MPI_Comm comm) {

  int myRank, mpiSize;
  MPI_Comm_rank(comm, &myRank);
  MPI_Comm_size(comm, &mpiSize);

  // create communicator for shared memory
  MPI_Comm commdup;
  MPI_Comm_dup(comm, &commdup);
  MPI_Comm sharedcomm;
  MPI_Comm_split_type(commdup, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &sharedcomm);
  int sharedsize;
  MPI_Comm_size(sharedcomm, &sharedsize);

  int mypartner;
  if((int(myRank/sharedsize))%2 == 0) {
    mypartner = (myRank + sharedsize)%mpiSize;
    if(mypartner == myRank && mpiSize > 1)
      mypartner = (mypartner+1)%mpiSize;
  } else {
    mypartner = (mpiSize + myRank - sharedsize)%mpiSize;
    if(mypartner == myRank && mpiSize > 1)
      mypartner = (mypartner-1)%mpiSize;
  }

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
    sprintf(buffer, "pairwiseexchange_nmessages_%i_%s", nmessages, timebuffer);
    fp = fopen(buffer, "w");
    fprintf(fp, "%-10s %-10s %-10s %-15s %-15s\n", "total", "loopcount", "n messages", "bytes", "timing");

  }

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

    double t_start = 0;

    MPI_Request allReqS[nmessages];
    MPI_Request allReqR[nmessages];

    for(int i = 0; i < options.iterations + options.skip; i++) {

      if(i == options.skip)
        t_start = MPI_Wtime();

      for(int iMessage = 0; iMessage < nmessages; ++iMessage)
        MPI_CHECK(MPI_Irecv(&r_buf[iMessage*options.max_message_size], size, MPI_INT, mypartner, iMessage, MPI_COMM_WORLD, &allReqR[iMessage]));
      for(int iMessage = 0; iMessage < nmessages; ++iMessage)
        MPI_CHECK(MPI_Isend(&s_buf[iMessage*options.max_message_size], size, MPI_INT, mypartner, iMessage, MPI_COMM_WORLD, &allReqS[iMessage]));

      MPI_CHECK(MPI_Waitall(nmessages, allReqS, MPI_STATUSES_IGNORE));
      MPI_CHECK(MPI_Waitall(nmessages, allReqR, MPI_STATUSES_IGNORE));

    }

    double t_end = MPI_Wtime();

    double latency = (t_end - t_start) * 1.0e6 / options.iterations;
    MPI_CHECK(MPI_Send(&latency, 1, MPI_DOUBLE, 0, 2, comm));

    MPI_CHECK(MPI_Barrier(comm));

    if(myRank == 0)
      all_sizes[iSize] = size;

    double avglatency;
    MPI_Reduce(&latency, &avglatency, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    avglatency /= mpiSize;


    if(0 == myRank) {

      if(all_min[iSize] > avglatency || all_min[iSize] == 0)
        all_min[iSize] = avglatency;
      if(all_max[iSize] < avglatency)
        all_max[iSize] = avglatency;
      all_avg[iSize] += avglatency;

      fprintf(fp, "%-10d %-10d %-10d %-15d %-15f\n", mpiSize, options.iterations, nmessages, size*4, avglatency);

    }

    ++iSize;

  }

  if(myRank == 0) {
    printf("%-10s %-13s %-13s %-13s\n", "bytes", "average", "minimum", "maximum");
    for(int i = 0; i < loopCounter; ++i) {
      printf("%-10d %-13f %-13f %-13f\n", all_sizes[i]*4, all_avg[i]/((double)(mpiSize-1)), all_min[i], all_max[i]);
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
