#include "infiniccl_adapter.hpp"

#include <dlfcn.h>

#include <cstdlib>
#include <mutex>
#include <stdexcept>
#include <string>

namespace infinilm::engine::distributed::infiniccl_adapter {

namespace {

// Mirrors infinicclDataType_t from the standalone InfiniCCL's data_type.h.
enum IcclDataType : int {
    ICCL_INT8 = 0,
    ICCL_INT16 = 1,
    ICCL_INT32 = 2,
    ICCL_INT64 = 3,
    ICCL_UINT8 = 4,
    ICCL_UINT16 = 5,
    ICCL_UINT32 = 6,
    ICCL_UINT64 = 7,
    ICCL_FLOAT16 = 8,
    ICCL_BFLOAT16 = 9,
    ICCL_FLOAT32 = 10,
    ICCL_FLOAT64 = 11,
};

// Mirrors infinicclRedOp_t.
constexpr int ICCL_SUM = 0;

struct Api {
    void *handle = nullptr;
    int (*init)(int *, char ***) = nullptr;
    int (*finalize)() = nullptr;
    int (*get_rank)(int *) = nullptr;
    int (*get_size)(int *) = nullptr;
    int (*get_unique_id)(UniqueId *) = nullptr;
    int (*comm_init_all)(void **, int, const int *) = nullptr;
    int (*comm_init_rank)(void **, int, UniqueId, int) = nullptr;
    int (*comm_destroy)(void *) = nullptr;
    int (*all_reduce)(const void *, void *, size_t, int, int, void *, void *) = nullptr;
    int (*broadcast)(const void *, void *, size_t, int, int, void *, void *) = nullptr;
};

Api &api() {
    static Api instance;
    return instance;
}

template <typename Fn>
void resolve(void *handle, Fn &fn, const char *name) {
    fn = reinterpret_cast<Fn>(dlsym(handle, name));
    if (fn == nullptr) {
        throw std::runtime_error(std::string("infiniccl_adapter: failed to resolve symbol `") + name + "`: " + dlerror());
    }
}

CommMode parse_mode() {
    const char *mode = std::getenv("INFINILM_INFINICCL_COMM_MODE");
    if (mode == nullptr || mode[0] == '\0' || std::string(mode) == "ccl_single_process") {
        return CommMode::CclSingleProcess;
    }
    if (std::string(mode) == "mpi") {
        return CommMode::Mpi;
    }
    throw std::runtime_error(
        std::string("infiniccl_adapter: unsupported `INFINILM_INFINICCL_COMM_MODE` value `") +
        mode + "`; expected `ccl_single_process` or `mpi`");
}

void load_library_for_mode(CommMode comm_mode) {
    const char *path = std::getenv("INFINICCL_LIB");
    if (path == nullptr || path[0] == '\0') {
        throw std::runtime_error(
            "infiniccl_adapter: INFINILM_USE_INFINICCL=1 requires INFINICCL_LIB to point "
            "to the standalone InfiniCCL libinfiniccl.so (an absolute path, to avoid "
            "accidentally loading InfiniCore's libinfiniccl.so of the same name)");
    }
    // RTLD_LOCAL keeps the standalone library's `infiniccl*` symbols out of the
    // global namespace; InfiniCore's libinfiniccl.so exports the same names with
    // different signatures.
    void *handle = dlopen(path, RTLD_NOW | RTLD_LOCAL);
    if (handle == nullptr) {
        throw std::runtime_error(std::string("infiniccl_adapter: dlopen failed: ") + dlerror());
    }
    Api &a = api();
    a.handle = handle;
    resolve(handle, a.comm_destroy, "infinicclCommDestroy");
    resolve(handle, a.all_reduce, "infinicclAllReduce");
    if (comm_mode == CommMode::CclSingleProcess) {
        resolve(handle, a.get_unique_id, "infinicclGetUniqueId");
        resolve(handle, a.comm_init_rank, "infinicclCommInitRank");
    } else {
        resolve(handle, a.init, "infinicclInit");
        resolve(handle, a.finalize, "infinicclFinalize");
        resolve(handle, a.get_rank, "infinicclGetRank");
        resolve(handle, a.get_size, "infinicclGetSize");
        resolve(handle, a.comm_init_all, "infinicclCommInitAll");
        resolve(handle, a.broadcast, "infinicclBroadcast");
    }
}

void check(int status, const char *what) {
    if (status != 0) {
        throw std::runtime_error(std::string("infiniccl_adapter: ") + what + " failed with status " + std::to_string(status));
    }
}

int to_iccl_dtype(infinicore::DataType dtype) {
    switch (dtype) {
    case infinicore::DataType::I8:
        return ICCL_INT8;
    case infinicore::DataType::I16:
        return ICCL_INT16;
    case infinicore::DataType::I32:
        return ICCL_INT32;
    case infinicore::DataType::I64:
        return ICCL_INT64;
    case infinicore::DataType::U8:
        return ICCL_UINT8;
    case infinicore::DataType::U16:
        return ICCL_UINT16;
    case infinicore::DataType::U32:
        return ICCL_UINT32;
    case infinicore::DataType::U64:
        return ICCL_UINT64;
    case infinicore::DataType::F16:
        return ICCL_FLOAT16;
    case infinicore::DataType::BF16:
        return ICCL_BFLOAT16;
    case infinicore::DataType::F32:
        return ICCL_FLOAT32;
    case infinicore::DataType::F64:
        return ICCL_FLOAT64;
    default:
        throw std::runtime_error("infiniccl_adapter: unsupported dtype " + infinicore::toString(dtype));
    }
}

std::once_flag init_flag;
std::once_flag load_flag;

} // namespace

bool enabled() {
    static const bool value = [] {
        const char *v = std::getenv("INFINILM_USE_INFINICCL");
        return v != nullptr && v[0] == '1';
    }();
    return value;
}

CommMode mode() {
    static const CommMode value = parse_mode();
    return value;
}

bool ccl_single_process_mode() {
    return mode() == CommMode::CclSingleProcess;
}

bool mpi_mode() {
    return mode() == CommMode::Mpi;
}

const char *mode_name() {
    return ccl_single_process_mode() ? "ccl_single_process" : "mpi";
}

void load() {
    std::call_once(load_flag, [] {
        load_library_for_mode(mode());
    });
}

void init() {
    std::call_once(init_flag, [] {
        load();
        if (!mpi_mode()) {
            return;
        }
        check(api().init(nullptr, nullptr), "infinicclInit");
    });
}

void finalize() {
    if (api().finalize != nullptr) {
        api().finalize();
    }
}

int rank() {
    int r = -1;
    check(api().get_rank(&r), "infinicclGetRank");
    return r;
}

int size() {
    int s = 0;
    check(api().get_size(&s), "infinicclGetSize");
    return s;
}

void *comm_init_all(int ndev, const int *devlist) {
    load();
    void *comm = nullptr;
    check(api().comm_init_all(&comm, ndev, devlist), "infinicclCommInitAll");
    return comm;
}

UniqueId get_unique_id() {
    load();
    UniqueId id{};
    check(api().get_unique_id(&id), "infinicclGetUniqueId");
    return id;
}

void *comm_init_rank(int nranks, UniqueId id, int rank) {
    load();
    void *comm = nullptr;
    check(api().comm_init_rank(&comm, nranks, id, rank), "infinicclCommInitRank");
    return comm;
}

void comm_destroy(void *comm) {
    if (comm != nullptr) {
        load();
        check(api().comm_destroy(comm), "infinicclCommDestroy");
    }
}

void allreduce_sum(const void *sendbuf, void *recvbuf, size_t count,
                   infinicore::DataType dtype, void *comm, void *stream) {
    load();
    check(api().all_reduce(sendbuf, recvbuf, count, to_iccl_dtype(dtype), ICCL_SUM, comm,
                           stream),
          "infinicclAllReduce");
}

void broadcast(void *buf, size_t count, infinicore::DataType dtype, int root, void *comm) {
    load();
    check(api().broadcast(buf, buf, count, to_iccl_dtype(dtype), root, comm,
                          /*stream=*/nullptr),
          "infinicclBroadcast");
}

} // namespace infinilm::engine::distributed::infiniccl_adapter
