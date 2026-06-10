#pragma once

#include <infinicore/dtype.hpp>

#include <cstddef>

namespace infinilm::engine::distributed::infiniccl_adapter {

// Adapter for the standalone InfiniCCL library (github.com/InfiniTensor/InfiniCCL).
//
// The standalone library follows an MPI-style process model: every tensor-parallel
// rank is a separate process (launched with `mpirun -np <tp_size>`), and each
// process owns exactly one communicator. This differs from InfiniCore's built-in
// infiniccl, which creates all communicators in a single process.
//
// The library is loaded with dlopen(RTLD_LOCAL) because it exports the same
// `infiniccl*` symbol names as InfiniCore's libinfiniccl.so (already linked into
// this binary) but with different signatures; dlsym keeps the two strictly apart.
//
// Activation: set INFINILM_USE_INFINICCL=1 and INFINICCL_LIB=/abs/path/to/the
// standalone libinfiniccl.so, then launch one process per rank under mpirun.

// True when INFINILM_USE_INFINICCL=1 (checked once per process).
bool enabled();

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

void comm_destroy(void *comm);

// In-place capable sum-allreduce over `count` elements of `dtype`.
void allreduce_sum(const void *sendbuf, void *recvbuf, size_t count,
                   infinicore::DataType dtype, void *comm);

// Broadcast `count` elements of `dtype` from `root` (send/recv in-place).
void broadcast(void *buf, size_t count, infinicore::DataType dtype, int root, void *comm);

} // namespace infinilm::engine::distributed::infiniccl_adapter
