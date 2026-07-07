#pragma once

#include <infinicore/dtype.hpp>

#include <cstddef>

namespace infinilm::engine::distributed::infiniccl_adapter {

// Adapter for the standalone InfiniCCL library (github.com/InfiniTensor/InfiniCCL).
//
// The default mode follows InfiniLM's normal execution model: one process hosts
// every tensor-parallel rank, with one worker thread per local rank. Standalone
// InfiniCCL communicators are created through `infinicclGetUniqueId` +
// `infinicclCommInitRank`, mirroring InfiniCCL's native CCL example.
//
// The library is loaded with dlopen(RTLD_LOCAL) because it exports the same
// `infiniccl*` symbol names as InfiniCore's libinfiniccl.so (already linked into
// this binary) but with different signatures; dlsym keeps the two strictly apart.
//
// Activation: set INFINILM_USE_INFINICCL=1 and INFINICCL_LIB=/abs/path/to/the
// standalone libinfiniccl.so. Set INFINILM_INFINICCL_COMM_MODE=mpi only for the
// legacy one-process-per-rank experiment.

enum class CommMode {
    CclSingleProcess,
    Mpi,
};

struct UniqueId {
    char internal[128];
};

// True when INFINILM_USE_INFINICCL=1 (checked once per process).
bool enabled();

CommMode mode();
bool ccl_single_process_mode();
bool mpi_mode();
const char *mode_name();

// Loads the standalone InfiniCCL library (idempotent).
void load();

// Loads the library and initializes MPI (idempotent).
void init();

// Finalizes MPI.
void finalize();

// Rank / world size of this process in MPI_COMM_WORLD.
int rank();
int size();

// Creates the communicator for this process. `devlist[local_rank]` selects the
// device ordinal each rank binds to (the library reads OMPI_COMM_WORLD_LOCAL_RANK).
void *comm_init_all(int ndev, const int *devlist);

UniqueId get_unique_id();
void *comm_init_rank(int nranks, UniqueId id, int rank);

void comm_destroy(void *comm);

// In-place capable sum-allreduce over `count` elements of `dtype`.
void allreduce_sum(const void *sendbuf, void *recvbuf, size_t count,
                   infinicore::DataType dtype, void *comm, void *stream);

// Broadcast `count` elements of `dtype` from `root` (send/recv in-place).
void broadcast(void *buf, size_t count, infinicore::DataType dtype, int root, void *comm);

} // namespace infinilm::engine::distributed::infiniccl_adapter
