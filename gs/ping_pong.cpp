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
static void multi_latency_paul(int writeToFile, MPI_Comm comm);

// GLOBALS
struct options_t options;
char *s_buf, *r_buf;

extern "C" { // Begin C Linkage
int pingPongMulti(int pairs, int useDevice, int createDetailedPingPongFile, occa::device device, MPI_Comm comm)
{

    int size, rank;
    MPI_Comm_size(comm,&size);
    MPI_Comm_rank(comm,&rank);

    if(size == 0 || size%2 != 0) return 1;

    options.min_message_size = 1;
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
        printf("\nping pong multi - pairs: %d useDevice: %d\n\n", pairs, useDevice);
        fflush(stdout);
    }

    MPI_CHECK(MPI_Barrier(comm));
    multi_latency(comm);
    MPI_CHECK(MPI_Barrier(comm));

    if(rank == 0) {
        if(size > 2)
            printf("\n\nping pong - ranks 1-%d to rank 0, useDevice: %d\n\n", size-1, useDevice);
        else
            printf("\n\nping pong - ranks 1 to rank 0, useDevice: %d\n\n", useDevice);
        fflush(stdout);
    }

    MPI_CHECK(MPI_Barrier(comm));
    multi_latency_paul(createDetailedPingPongFile, comm);
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

        avg_lat = total_lat/(double) (pairs * 2);

        if(0 == rank) {
            fprintf(stdout, "%-*d%*.*f\n", 10, size, FIELD_WIDTH,
                    FLOAT_PRECISION, avg_lat);
            fflush(stdout);
        }
    }
}

static void multi_latency_paul(int writeToFile, MPI_Comm comm) {
    
    int myRank, mpiSize;
    MPI_Comm_rank(comm, &myRank);
    MPI_Comm_size(comm, &mpiSize);
    
    int all_sizes[21];
    double all_min[21];
    double all_max[21];
    double all_avg[21];
    
    // initialize stats
    int i = 0;
    for(int size = options.min_message_size; size <= options.max_message_size; size  = (size ? size * 2 : 1)) {
        all_sizes[i] = size;
        all_min[i] = 99999;
        all_max[i] = 0;
        all_avg[i] = 0;
        ++i;
    }
    
    FILE *fp;
    if(0 == myRank && writeToFile) {
        fp = fopen("pingpong.txt", "w");
        fprintf(fp, "%-10s %-10s %-15s %-15s\n", "sender", "receiver", "bytes", "latency");
    }
    
    
    
    for(int iRank = 1; iRank < mpiSize; ++iRank) {
        
        int iSize = 0;
        
        for(int size = options.min_message_size; size <= options.max_message_size; size  = (size ? size * 2 : 1)) {

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

                    MPI_CHECK(MPI_Send(s_buf, size, MPI_CHAR, 0, 1, comm));
                    MPI_CHECK(MPI_Recv(r_buf, size, MPI_CHAR, 0, 1, comm, MPI_STATUS_IGNORE));
                }

                double t_end = MPI_Wtime();
                
                latency = (t_end - t_start) * 1.0e6 / (2.0 * options.iterations);
                MPI_CHECK(MPI_Send(&latency, 1, MPI_DOUBLE, 0, 2, comm));
                
            } else if(myRank == 0) {
                
                double t_start = 0;
                
                for(int i = 0; i < options.iterations + options.skip; i++) {

                    if(i == options.skip)
                        t_start = MPI_Wtime();

                    MPI_CHECK(MPI_Recv(r_buf, size, MPI_CHAR, iRank, 1, comm, MPI_STATUS_IGNORE));
                    MPI_CHECK(MPI_Send(s_buf, size, MPI_CHAR, iRank, 1, comm));
                }

                double t_end = MPI_Wtime();
                
                latency = (t_end - t_start) * 1.0e6 / (2.0 * options.iterations);
                MPI_CHECK(MPI_Send(&latency, 1, MPI_DOUBLE, 0, 3, comm));
                
            } // all other ranks are idle
            
            MPI_CHECK(MPI_Barrier(comm));

            if(0 == myRank) {
                
                double peer_latency;
                MPI_CHECK(MPI_Recv(&peer_latency, 1, MPI_DOUBLE, iRank, 2, comm, MPI_STATUS_IGNORE));
                double avg_lat = (latency+peer_latency)/2.0;
                
                if(all_min[iSize] > avg_lat)
                    all_min[iSize] = avg_lat;
                if(all_max[iSize] < avg_lat)
                    all_max[iSize] = avg_lat;
                all_avg[iSize] += avg_lat;

                if(writeToFile) {
                    fprintf(fp, "%-10d %-10d %-15d %-15f\n", iRank, 0, size, avg_lat);
                    fflush(fp);
                }

            }
            
            ++iSize;
            
        }
    }
    
    if(myRank == 0) {
        printf("%-10s %-13s %-13s %-13s\n", "bytes", "average", "minimum", "maximum");
        for(int i = 0; i < 21; ++i) {
            printf("%-10d %-13f %-13f %-13f\n", all_sizes[i], all_avg[i]/((double)(mpiSize-1)), all_min[i], all_max[i]);
        }
        double avg_sum = 0;
        for(int i = 0; i < 21; ++i)
            avg_sum += all_avg[i];
        printf("\nGlobal average: %f\n", avg_sum/21.0);

        if(writeToFile)
            fclose(fp);
    }
    
}
