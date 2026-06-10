#include "communication_group.hpp"
#include "../../utils.hpp"
#include "infiniccl_adapter.hpp"

#include <spdlog/spdlog.h>

namespace infinilm::engine::distributed {

CommunicationGroup::CommunicationGroup(const DistConfig &dist_config, infinicore::Device::Type device_type)
    : dist_config_(dist_config), device_type_(device_type),
      communicators_(std::vector<infinicclComm_t>(dist_config.tp_device_ids.size(), nullptr)) {

    size_t world_size = dist_config_.tp_device_ids.size();

    if (infiniccl_adapter::enabled()) {
        // Multi-process mode: this process is one tensor-parallel rank; the
        // communicator comes from the standalone InfiniCCL library (MPI).
        use_infiniccl_adapter_ = true;
        infiniccl_adapter::init();
        if ((size_t)infiniccl_adapter::size() != world_size) {
            throw std::runtime_error("infiniccl_adapter: MPI world size (" + std::to_string(infiniccl_adapter::size()) + ") must equal the tensor parallel size (" + std::to_string(world_size) + "); launch with mpirun -np <tp_size>");
        }
        local_rank_ = infiniccl_adapter::rank();

        // Allow more ranks than physical devices (e.g. 2 ranks sharing one
        // GPU): wrap each requested device id onto an existing ordinal.
        size_t device_count = infinicore::context::getDeviceCount(device_type);
        if (device_count == 0) {
            throw std::runtime_error("infiniccl_adapter: no device of the requested type is available");
        }
        for (auto &device_id : dist_config_.tp_device_ids) {
            device_id = device_id % (int)device_count;
        }

        infinicore::context::setDevice(infinicore::Device(device_type_, dist_config_.tp_device_ids[local_rank_]));
        communicators_[local_rank_] = (infinicclComm_t)infiniccl_adapter::comm_init_all(
            (int)world_size, dist_config_.tp_device_ids.data());
        spdlog::info("[infiniccl_adapter] rank {}/{} bound to device {} (standalone InfiniCCL over MPI)",
                     local_rank_, world_size, dist_config_.tp_device_ids[local_rank_]);
        return;
    }

    size_t device_count = infinicore::context::getDeviceCount(device_type);
    if (device_count < world_size) {
        throw std::runtime_error("infinilm::engine::distributed::CommunicationGroup error, world size is larger than the number of available GPUs. world size: " + std::to_string(world_size) + ", device count: " + std::to_string(device_count));
    }

    if (infinicore::context::getDevice().getType() != device_type_) {
        infinicore::context::setDevice(infinicore::Device(device_type_, 0));
    }
    if (world_size > 1) {
        RUN_INFINI(infinicclCommInitAll(
            (infiniDevice_t)infinicore::context::getDevice().getType(),
            communicators_.data(),
            dist_config.tp_device_ids.size(),
            dist_config.tp_device_ids.data()));
    }
}

const DistConfig &CommunicationGroup::get_dist_config() const {
    return dist_config_;
}

RankInfo CommunicationGroup::get_rank_info(int rank) const {
    RankInfo info;
    info.tp_size = dist_config_.tp_device_ids.size();
    info.tp_rank = rank;
    info.device = infinicore::Device(device_type_, dist_config_.tp_device_ids[rank]);
    info.comm = communicators_[rank];
    return info;
}

int CommunicationGroup::get_world_size() const {
    return dist_config_.tp_device_ids.size();
}

std::vector<int> CommunicationGroup::get_local_ranks() const {
    if (use_infiniccl_adapter_) {
        return {local_rank_};
    }
    std::vector<int> ranks(dist_config_.tp_device_ids.size());
    for (size_t r = 0; r < ranks.size(); ++r) {
        ranks[r] = (int)r;
    }
    return ranks;
}

CommunicationGroup::~CommunicationGroup() {
    if (use_infiniccl_adapter_) {
        infiniccl_adapter::comm_destroy(communicators_[local_rank_]);
        infiniccl_adapter::finalize();
        return;
    }
    if (communicators_.size() > 1) {
        for (auto &comm : communicators_) {
            infinicclCommDestroy(comm);
        }
    }
}

} // namespace infinilm::engine::distributed
