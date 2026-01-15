#include <sys/stat.h>
#include <assert.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <mpi.h>

#define CHECK_ERROR(cond, errstr)               \
    do {                                        \
        if (cond) {                             \
            perror(errstr);                     \
            MPI_Abort(MPI_COMM_WORLD, 1);       \
        }                                       \
    } while (0)

#define BUFFER_SIZE (1L << 30) // 1GB Chunks

static double get_elapsed(struct timespec t1, struct timespec t2);

int main(int argc, char **argv) {
    struct timespec start, end;
    const char *destdir;
    char command[4096];
    int rank;

    // Fix: Declare and initialize the variable tracking total size
    unsigned long long total_bytes = 0;

    clock_gettime(CLOCK_MONOTONIC, &start);

    MPI_Init(NULL, NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (argc < 2) {
        if (rank == 0) fprintf(stderr, "Usage: cptotmp <src> [dest]\n");
        MPI_Finalize();
        return 1;
    }

    destdir = (argc < 3) ? "/tmp" : argv[2];

    FILE *archive = NULL;

    // --- RANK 0: OPEN READ STREAM ---
    if (rank == 0) {
        // Strip trailing slash to safely handle directories
        int last_idx = strlen(argv[1]) - 1;
        if (argv[1][last_idx] == '/') argv[1][last_idx] = '\0';

        char *dup = strdup(argv[1]);
        char *slash = strrchr(dup, '/');
        char *left, *right;

        if (slash != NULL) {
            *slash = '\0';
            left = dup;
            right = slash + 1;
        } else {
            left = ".";
            right = dup;
            // Use cwd as base if no path provided
        }

        // REMOVED: The pre-calculation "tar --totals" step that was causing the hang.
        // We now start the real read immediately.
        snprintf(command, sizeof(command), "tar -C %s -cf - %s", left, right);
        archive = popen(command, "r");
        CHECK_ERROR(!archive, "popen (read)");
        free(dup);

        printf("cptotmp: Streaming %s to %s (Immediate Start)...\n", argv[1], destdir);
    }

    // --- ALL RANKS: OPEN WRITE STREAM ---
    snprintf(command, sizeof(command), "mkdir -p %s", destdir);
    system(command);

    snprintf(command, sizeof(command), "tar -xf - -C %s", destdir);
    FILE *dest = popen(command, "w");
    CHECK_ERROR(!dest, "popen (write)");

    // --- STREAMING LOOP ---
    void *buf = malloc(BUFFER_SIZE);
    assert(buf);

    int active = 1;

    while (active) {
        long chunk_size = 0;

        // Rank 0: Read until BUFFER_SIZE is full or EOF
        if (rank == 0) {
            size_t bytes_read = 0;
            while (bytes_read < BUFFER_SIZE) {
                // Try to read the remaining space in the buffer
                size_t n = fread((char*)buf + bytes_read, 1, BUFFER_SIZE - bytes_read, archive);

                if (n == 0) { // EOF or Error
                    if (ferror(archive)) {
                         perror("fread error");
                         MPI_Abort(MPI_COMM_WORLD, 1);
                    }
                    break;
                }
                bytes_read += n;
            }
            chunk_size = bytes_read;
        }

        // 1. Broadcast how much data is in this chunk (0 means Done)
        MPI_Bcast(&chunk_size, 1, MPI_LONG, 0, MPI_COMM_WORLD);

        if (chunk_size == 0) {
            break;
        }

        // 2. Broadcast the actual data
        MPI_Bcast(buf, chunk_size, MPI_BYTE, 0, MPI_COMM_WORLD);

        // 3. Write to local SSD (Handle partial writes if necessary, though uncommon on local disk)
        size_t total_written = 0;
        while (total_written < chunk_size) {
            size_t n = fwrite((char*)buf + total_written, 1, chunk_size - total_written, dest);
            if (n == 0) {
                 fprintf(stderr, "Rank %d: Write error (Disk full?)\n", rank);
                 MPI_Abort(MPI_COMM_WORLD, 1);
            }
            total_written += n;
        }

        total_bytes += chunk_size;
    }

    if (rank == 0) pclose(archive);
    pclose(dest);
    free(buf);

    // --- TIMING ---
    clock_gettime(CLOCK_MONOTONIC, &end);
    double elapsed = get_elapsed(start, end);
    double max_time;
    MPI_Reduce(&elapsed, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        double gb = (double)total_bytes / (1024.0 * 1024.0 * 1024.0);
        printf("cptotmp: Transferred %.2f GiB in %.2f seconds (%.2f GiB/s)\n",
            gb, max_time, gb/max_time);
    }

    MPI_Finalize();
    return 0;
}

static double get_elapsed(struct timespec t1, struct timespec t2)
{
    time_t sec = t2.tv_sec - t1.tv_sec;
    long nsec = t2.tv_nsec - t1.tv_nsec;
    if (nsec < 0) { sec--; nsec += 1000000000L; }
    return sec + nsec * 1e-9;
}
