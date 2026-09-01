#pragma once

#include "bcc32_checkpoint.hpp"
#include "bcc32_cuda_executor.cuh"

#include <cstdint>
#include <filesystem>
#include <string>

namespace substrate::bcc32 {

struct PageSchedule {
    // Zero means use every aperture slot left after exact halo closure.
    std::uint32_t maximum_core_chunks = 0u;
    bool reverse_core_order = false;
};

struct TransitionReceipt {
    ContentAddress input_identity{};
    ContentAddress output_identity{};
    std::uint64_t input_chunks = 0u;
    std::uint64_t output_chunks = 0u;
    std::uint64_t pages = 0u;
    bool inverse = false;
    ContentAddress execution_profile{};
};

// Executes the one frozen law against immutable chunk objects. The CUDA field
// is a cache aperture: global material identity remains in the commit index,
// and page ordering cannot alter the resulting root.
class PagedWorldExecutor {
  public:
    [[nodiscard]] static PagedWorldExecutor production();
    // A direct sparse-world pager with an explicitly bounded reusable CUDA
    // aperture. Unlike testing(), this is a production execution shape: total
    // world size is carried by the sparse store, not by aperture width.
    [[nodiscard]] static PagedWorldExecutor windowed(
        std::uint32_t aperture_chunks);
    [[nodiscard]] static PagedWorldExecutor testing(std::uint32_t aperture_chunks);

    PagedWorldExecutor(const PagedWorldExecutor&) = delete;
    PagedWorldExecutor& operator=(const PagedWorldExecutor&) = delete;
    PagedWorldExecutor(PagedWorldExecutor&& other) noexcept;
    PagedWorldExecutor& operator=(PagedWorldExecutor&& other) noexcept;
    ~PagedWorldExecutor();

    bool advance(const std::filesystem::path& repository,
                 ArtifactKind expected_kind,
                 bool inverse,
                 PageSchedule schedule,
                 TransitionReceipt* receipt,
                 std::string* error,
                 PublicationFailurePoint failure = PublicationFailurePoint::none);

    // Advances one immutable task head without consulting or mutating the
    // repository's convenience lineage root. Independent population slots can
    // therefore execute concurrently against the shared object store.
    bool advance_object(
        const std::filesystem::path& repository,
        const ContentAddress& input_identity,
        ArtifactKind expected_kind,
        bool inverse,
        PageSchedule schedule,
        TransitionReceipt* receipt,
        std::string* error,
        PublicationFailurePoint failure = PublicationFailurePoint::none);

    // Applies one complete F/F^-1 directly to an in-memory sparse store. The
    // input store is replaced only after every factor and DeltaN_Q audit
    // succeeds, so callers can use it as a failure-atomic pager transaction
    // without filesystem publication.
    bool advance_store(WorldStore* world, bool inverse, PageSchedule schedule,
                       std::uint64_t* page_count, std::string* error);

    // Advances several complete F/F^-1 steps while one bounded sparse
    // neighborhood remains resident in the CUDA aperture. The exact halo is
    // derived from step count; if it cannot fit, the call rejects without
    // mutating the input so the caller may use the general pager instead.
    bool advance_store_resident(WorldStore* world, std::uint32_t steps,
                                bool inverse, std::string* error,
                                PageSchedule schedule = {});
    [[nodiscard]] static std::uint32_t required_resident_chunks(
        const WorldStore& world, std::uint32_t steps);

    [[nodiscard]] std::uint32_t aperture_chunks() const {
        return executor_.chunk_count();
    }
    [[nodiscard]] std::uint64_t aperture_bytes() const {
        return executor_.aperture_bytes();
    }
    void verify_quiescent_aperture();
    [[nodiscard]] std::uint32_t verify_nontrivial_forward_inverse_aperture();

  private:
    explicit PagedWorldExecutor(CudaBcc32Executor executor);
    bool advance_loaded(const std::filesystem::path& repository,
                        WorldCommit input,
                        ArtifactKind expected_kind,
                        bool inverse,
                        PageSchedule schedule,
                        bool publish_mutable_root,
                        TransitionReceipt* receipt,
                        std::string* error,
                        PublicationFailurePoint failure);
    void release() noexcept;

    CudaBcc32Executor executor_;
    DeviceChunkSlot* device_slots_ = nullptr;
};

}  // namespace substrate::bcc32
