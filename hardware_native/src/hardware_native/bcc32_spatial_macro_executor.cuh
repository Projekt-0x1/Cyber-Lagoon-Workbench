#pragma once

#include <cstdint>

#include "bcc32_cuda_executor.cuh"

namespace substrate::bcc32 {

// No-sync launchers shared by the canonical executor and the realtime driver.
// Submission ownership, topology validation, and synchronization remain with
// the caller; each launcher preserves immutable match->collision->apply order.
void launch_carrier_pair_splitter_cuda(
    SiteWord* words, std::uint8_t* scratch, std::uint64_t site_count,
    const DeviceChunkMap& chunks, cudaStream_t stream);
void launch_processive_rearm_cuda(
    SiteWord* words, std::uint8_t* scratch, std::uint64_t site_count,
    const DeviceChunkMap& chunks, cudaStream_t stream);
void launch_processive_release_cuda(
    SiteWord* words, std::uint8_t* scratch, std::uint64_t site_count,
    const DeviceChunkMap& chunks, bool inverse, cudaStream_t stream);
void launch_active_processive_release_cuda(
    SiteWord* words, std::uint8_t* scratch, std::uint8_t* resolved,
    std::uint64_t site_count,
    const DeviceChunkMap& chunks, const std::uint64_t* active_slots,
    std::uint64_t active_count, bool inverse, cudaStream_t stream);
void launch_carrier_corner_cuda(
    SiteWord* words, std::uint8_t* scratch, std::uint64_t site_count,
    const DeviceChunkMap& chunks, bool inverse, cudaStream_t stream);
void launch_eligibility_residual_junction_cuda(
    SiteWord* words, std::uint8_t* scratch, std::uint64_t site_count,
    const DeviceChunkMap& chunks, const std::uint64_t* active_slots,
    std::uint64_t active_count, cudaStream_t stream);
void launch_eligibility_residual_junction_graph_safe_cuda(
    SiteWord* words, std::uint8_t* scratch, std::uint64_t site_count,
    const DeviceChunkMap& chunks, const std::uint64_t* active_slots,
    const std::uint32_t* device_active_count, std::uint32_t capacity,
    cudaStream_t stream);
void launch_active_spatial_macros_cuda(
    SiteWord* words, std::uint8_t* scratch, std::uint8_t* resolved,
    std::uint64_t site_count,
    const DeviceChunkMap& chunks, bool inverse,
    const std::uint64_t* active_slots, std::uint64_t active_count,
    cudaStream_t stream);
void launch_active_spatial_macros_graph_safe_cuda(
    SiteWord* words, std::uint8_t* scratch, std::uint8_t* resolved,
    std::uint64_t site_count,
    const DeviceChunkMap& chunks, bool inverse,
    const std::uint64_t* active_slots,
    const std::uint32_t* device_active_count, std::uint32_t capacity,
    cudaStream_t stream);
void launch_prediction_residual_route_toggle_cuda(
    SiteWord* words, std::uint8_t* scratch, std::uint8_t* resolved,
    std::uint64_t site_count,
    const DeviceChunkMap& chunks, bool inverse,
    const std::uint64_t* active_slots, std::uint64_t active_count,
    cudaStream_t stream);
void launch_prediction_residual_route_toggle_graph_safe_cuda(
    SiteWord* words, std::uint8_t* scratch, std::uint8_t* resolved,
    std::uint64_t site_count,
    const DeviceChunkMap& chunks, bool inverse,
    const std::uint64_t* active_slots,
    const std::uint32_t* device_active_count, std::uint32_t capacity,
    cudaStream_t stream);
void launch_developmental_append_cuda(
    SiteWord* words, std::uint8_t* scratch, std::uint64_t site_count,
    const DeviceChunkMap& chunks, bool inverse,
    const std::uint64_t* active_slots, std::uint64_t active_count,
    cudaStream_t stream);
void launch_developmental_append_graph_safe_cuda(
    SiteWord* words, std::uint8_t* scratch, std::uint64_t site_count,
    const DeviceChunkMap& chunks, bool inverse,
    const std::uint64_t* active_slots,
    const std::uint32_t* device_active_count, std::uint32_t capacity,
    cudaStream_t stream);
void launch_developmental_learned_receptor_cuda(
    SiteWord* words, std::uint8_t* scratch, std::uint64_t site_count,
    const DeviceChunkMap& chunks, bool inverse,
    const std::uint64_t* active_slots, std::uint64_t active_count,
    cudaStream_t stream);
void launch_developmental_learned_receptor_graph_safe_cuda(
    SiteWord* words, std::uint8_t* scratch, std::uint64_t site_count,
    const DeviceChunkMap& chunks, bool inverse,
    const std::uint64_t* active_slots,
    const std::uint32_t* device_active_count, std::uint32_t capacity,
    cudaStream_t stream);
void launch_developmental_credit_service_cuda(
    SiteWord* words, std::uint8_t* scratch, std::uint64_t site_count,
    const DeviceChunkMap& chunks, bool inverse,
    const std::uint64_t* active_slots, std::uint64_t active_count,
    cudaStream_t stream);
void launch_developmental_credit_service_graph_safe_cuda(
    SiteWord* words, std::uint8_t* scratch, std::uint64_t site_count,
    const DeviceChunkMap& chunks, bool inverse,
    const std::uint64_t* active_slots,
    const std::uint32_t* device_active_count, std::uint32_t capacity,
    cudaStream_t stream);

}  // namespace substrate::bcc32
