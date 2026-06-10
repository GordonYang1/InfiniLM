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
    int (*comm_init_all)(void **, int, const int *) = nullptr;
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

void load_library() {
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
    resolve(handle, a.init, "infinicclInit");
    resolve(handle, a.finalize, "infinicclFinalize");
    resolve(handle, a.get_rank, "infinicclGetRank");
    resolve(handle, a.get_size, "infinicclGetSize");
    resolve(handle, a.comm_init_all, "infinicclCommInitAll");
    resolve(handle, a.comm_destroy, "infinicclCommDestroy");
    resolve(handle, a.all_reduce, "infinicclAllReduce");
    resolve(handle, a.broadcast, "infinicclBroadcast");
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

} // namespace

bool enabled() {
    static const bool value = [] {
        const char *v = std::getenv("INFINILM_USE_INFINICCL");
        return v != nullptr && v[0] == '1';
    }();
    return value;
}

void init() {
    std::call_once(init_flag, [] {
        load_library();
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
    void *comm = nullptr;
    check(api().comm_init_all(&comm, ndev, devlist), "infinicclCommInitAll");
    return comm;
}

void comm_destroy(void *comm) {
    if (comm != nullptr) {
        check(api().comm_destroy(comm), "infinicclCommDestroy");
    }
}

void allreduce_sum(const void *sendbuf, void *recvbuf, size_t count,
                   infinicore::DataType dtype, void *comm) {
    check(api().all_reduce(sendbuf, recvbuf, count, to_iccl_dtype(dtype), ICCL_SUM, comm,
                           /*stream=*/nullptr),
          "infinicclAllReduce");
}

void broadcast(void *buf, size_t count, infinicore::DataType dtype, int root, void *comm) {
    check(api().broadcast(buf, buf, count, to_iccl_dtype(dtype), root, comm,
                          /*stream=*/nullptr),
          "infinicclBroadcast");
}

} // namespace infinilm::engine::distributed::infiniccl_adapter
