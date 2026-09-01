#pragma once

#include "bcc32_cuda_adult_v1.cuh"
#include "bcc32_cuda_incremental_association_merge.cuh"

#include <thrust/execution_policy.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>
#include <thrust/sort.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace bcc32_cuda_incremental_adult {

namespace adult = bcc32_cuda_adult_v1;
namespace incremental = bcc32_cuda_incremental_association_merge;

using Clock = std::chrono::steady_clock;

inline double milliseconds(Clock::time_point begin, Clock::time_point end) {
  return std::chrono::duration<double, std::milli>(end - begin).count();
}

__global__ void append_online_learning_delta_kernel(
    std::uint32_t* unit_lengths, std::uint32_t* unit_content,
    std::uint32_t* vitality, adult::BigramKey* online_bigrams,
    std::uint32_t* online_bigram_counts, adult::TrigramKey* online_trigrams,
    std::uint32_t* online_trigram_counts, adult::AssociationKey* associations,
    std::uint32_t* association_counts, std::uint32_t* episode_units,
    std::uint32_t* episode_breaks, std::uint32_t* mutable_sizes,
    const std::uint32_t* sequence, std::uint32_t sequence_count,
    const std::uint32_t* segment_ids, const std::uint32_t* closure_bytes,
    std::uint32_t* ledger, std::uint32_t* status) {
  if (blockIdx.x != 0u || status[0] != 0u) return;
  __shared__ std::uint32_t bigram_base;
  __shared__ std::uint32_t trigram_base;
  __shared__ std::uint32_t association_base;
  __shared__ std::uint32_t episode_base;
  __shared__ std::uint32_t association_events;
  __shared__ std::uint32_t association_appended;
  __shared__ std::uint32_t update_allowed;
  __shared__ std::uint32_t episode_segments;
  // Same pure integer tally as the raw-bytes kernel: block-distributed with an
  // unchanged pair set, so the fold stays bit-identical.
  __shared__ unsigned long long event_partials[adult::kBlock];
  __shared__ unsigned long long weighted_partials[adult::kBlock];
  {
    unsigned long long events = 0u;
    unsigned long long weighted = 0u;
    const std::uint32_t distances = min(
        adult::kAssociationRadius, sequence_count > 0u ? sequence_count - 1u : 0u);
    for (std::uint32_t distance = 1u; distance <= distances; ++distance) {
      const std::uint32_t count = sequence_count - distance;
      for (std::uint32_t index = threadIdx.x; index < count; index += blockDim.x) {
        if (segment_ids[index] != segment_ids[index + distance]) continue;
        ++events;
        weighted += adult::kAssociationRadius + 1u - distance;
      }
    }
    event_partials[threadIdx.x] = events;
    weighted_partials[threadIdx.x] = weighted;
  }
  __syncthreads();
  if (threadIdx.x == 0u) {
    unsigned long long events = event_partials[0];
    unsigned long long weighted = weighted_partials[0];
    for (std::uint32_t lane = 1u; lane < blockDim.x; ++lane) {
      events += event_partials[lane];
      weighted += weighted_partials[lane];
    }
    association_events = static_cast<std::uint32_t>(events);
    bigram_base = mutable_sizes[1];
    trigram_base = mutable_sizes[2];
    association_base = mutable_sizes[3];
    episode_base = mutable_sizes[4];
    association_appended = 0u;
    episode_segments = 0u;
    std::uint32_t segment_begin = 0u;
    for (std::uint32_t index = 0u; index < sequence_count; ++index) {
      const std::uint32_t extent = index + 1u - segment_begin;
      const bool learned_close = adult::resident_unit_contains_any(
          unit_lengths, unit_content, sequence[index], closure_bytes,
          adult::kClosureCount);
      if (index + 1u == sequence_count ||
          extent >= 4u * adult::kCompositionUnits ||
          (extent >= 2u * adult::kCompositionUnits && learned_close)) {
        ++episode_segments;
        segment_begin = index + 1u;
      }
    }
    const std::uint32_t bigram_added = sequence_count > 1u ? sequence_count - 1u : 0u;
    const std::uint32_t trigram_added = sequence_count > 2u ? sequence_count - 2u : 0u;
    update_allowed = 1u;
    if (bigram_base + bigram_added > adult::kOnlineNgramCapacity) status[0] = 3u;
    else if (trigram_base + trigram_added > adult::kOnlineNgramCapacity) status[0] = 4u;
    else if (association_base + association_events > adult::kOnlineAssociationCapacity) {
      status[0] = 5u;
    } else if (episode_base + sequence_count > adult::kOnlineEpisodeCapacity ||
               mutable_sizes[5] + episode_segments >
                   adult::kOnlineEpisodeBreakCapacity) {
      status[0] = 6u;
    }
    const unsigned long long added_mass =
        static_cast<unsigned long long>(sequence_count) * 2u + bigram_added +
        trigram_added + weighted;
    if (status[0] == 0u && added_mass > ledger[1]) status[0] = 2u;
    if (status[0] != 0u) update_allowed = 0u;
    else {
      ledger[1] -= static_cast<std::uint32_t>(added_mass);
      ledger[2] += static_cast<std::uint32_t>(added_mass);
    }
  }
  __syncthreads();
  if (update_allowed == 0u) return;

  for (std::uint32_t index = threadIdx.x; index < sequence_count;
       index += blockDim.x) {
    atomicAdd(vitality + sequence[index], 1u);
    episode_units[episode_base + index] = sequence[index];
    if (index + 1u < sequence_count) {
      online_bigrams[bigram_base + index] =
          adult::BigramKey{sequence[index], sequence[index + 1u]};
      online_bigram_counts[bigram_base + index] = 1u;
    }
    if (index + 2u < sequence_count) {
      online_trigrams[trigram_base + index] = adult::TrigramKey{
          sequence[index], sequence[index + 1u], sequence[index + 2u]};
      online_trigram_counts[trigram_base + index] = 1u;
    }
  }
  const std::uint32_t distances = min(
      adult::kAssociationRadius, sequence_count > 0u ? sequence_count - 1u : 0u);
  for (std::uint32_t distance = 1u; distance <= distances; ++distance) {
    const std::uint32_t count = sequence_count - distance;
    for (std::uint32_t index = threadIdx.x; index < count; index += blockDim.x) {
      if (segment_ids[index] != segment_ids[index + distance]) continue;
      const std::uint32_t output =
          association_base + atomicAdd(&association_appended, 1u);
      associations[output] =
          adult::AssociationKey{sequence[index], sequence[index + distance]};
      association_counts[output] = adult::kAssociationRadius + 1u - distance;
    }
  }
  __syncthreads();
  if (threadIdx.x == 0u) {
    std::uint32_t segment_begin = 0u;
    for (std::uint32_t index = 0u; index < sequence_count; ++index) {
      const std::uint32_t extent = index + 1u - segment_begin;
      const bool learned_close = adult::resident_unit_contains_any(
          unit_lengths, unit_content, sequence[index], closure_bytes,
          adult::kClosureCount);
      if (index + 1u == sequence_count ||
          extent >= 4u * adult::kCompositionUnits ||
          (extent >= 2u * adult::kCompositionUnits && learned_close)) {
        episode_breaks[mutable_sizes[5]++] = episode_base + index + 1u;
        segment_begin = index + 1u;
      }
    }
    mutable_sizes[1] = bigram_base + (sequence_count > 1u ? sequence_count - 1u : 0u);
    mutable_sizes[2] = trigram_base + (sequence_count > 2u ? sequence_count - 2u : 0u);
    mutable_sizes[3] = association_base + association_appended;
    mutable_sizes[4] = episode_base + sequence_count;
  }
}

__global__ void compute_policy_candidates_kernel(
    const std::uint32_t* sequence, std::uint32_t sequence_count,
    const adult::BigramKey* base_bigrams, const std::uint32_t* base_bigram_counts,
    std::uint32_t base_bigram_count, const adult::TrigramKey* base_trigrams,
    const std::uint32_t* base_trigram_counts, std::uint32_t base_trigram_count,
    const adult::BigramKey* online_bigrams,
    const std::uint32_t* online_bigram_counts, std::uint32_t online_bigram_count,
    const adult::TrigramKey* online_trigrams,
    const std::uint32_t* online_trigram_counts, std::uint32_t online_trigram_count,
    std::uint32_t event_count, std::uint32_t* candidates) {
  namespace policy = bcc32_cuda_resident_synthesis;
  __shared__ unsigned long long scores[adult::kBlock];
  __shared__ std::uint32_t units[adult::kBlock];
  const std::uint32_t task = blockIdx.x;
  const std::uint32_t event = task / policy::kResidentSynthesisPolicyVariants;
  const std::uint32_t variant = task % policy::kResidentSynthesisPolicyVariants;
  if (event >= event_count) return;
  const std::uint32_t index = sequence_count - event_count + event;
  const std::uint32_t first = sequence[index - 2u];
  const std::uint32_t second = sequence[index - 1u];
  const std::uint32_t weight = policy::resident_synthesis_policy_weight(variant);
  const std::uint32_t base_begin = adult::lower_bigram(
      base_bigrams, base_bigram_count, second);
  const std::uint32_t base_end = adult::upper_bigram(
      base_bigrams, base_bigram_count, second);
  const std::uint32_t online_begin = adult::lower_bigram(
      online_bigrams, online_bigram_count, second);
  const std::uint32_t online_end = adult::upper_bigram(
      online_bigrams, online_bigram_count, second);
  const std::uint32_t base_extent = base_end - base_begin;
  const std::uint32_t online_extent = online_end - online_begin;

  unsigned long long best_score = 0ull;
  std::uint32_t best_unit = 0xffffffffu;
  for (std::uint32_t edge = threadIdx.x; edge < base_extent + online_extent;
       edge += blockDim.x) {
    const std::uint32_t candidate = edge < base_extent
        ? base_bigrams[base_begin + edge].next
        : online_bigrams[online_begin + edge - base_extent].next;
    const unsigned long long bigram = adult::resident_bigram_count(
        base_bigrams, base_bigram_counts, base_bigram_count, second, candidate) +
        adult::resident_bigram_count(online_bigrams, online_bigram_counts,
                                     online_bigram_count, second, candidate);
    const unsigned long long trigram = adult::resident_trigram_count(
        base_trigrams, base_trigram_counts, base_trigram_count, first, second,
        candidate) + adult::resident_trigram_count(
        online_trigrams, online_trigram_counts, online_trigram_count, first, second,
        candidate);
    const unsigned long long score = bigram + weight * trigram;
    if (score > best_score ||
        (score == best_score && score != 0ull && candidate < best_unit)) {
      best_score = score;
      best_unit = candidate;
    }
  }
  scores[threadIdx.x] = best_score;
  units[threadIdx.x] = best_unit;
  __syncthreads();
  for (std::uint32_t offset = blockDim.x / 2u; offset != 0u; offset >>= 1u) {
    if (threadIdx.x < offset) {
      const unsigned long long other_score = scores[threadIdx.x + offset];
      const std::uint32_t other_unit = units[threadIdx.x + offset];
      if (other_score > scores[threadIdx.x] ||
          (other_score == scores[threadIdx.x] && other_score != 0ull &&
           other_unit < units[threadIdx.x])) {
        scores[threadIdx.x] = other_score;
        units[threadIdx.x] = other_unit;
      }
    }
    __syncthreads();
  }
  if (threadIdx.x == 0u) {
    candidates[event * policy::kResidentSynthesisPolicyVariants + variant] = units[0];
  }
}

__global__ void apply_policy_observations_kernel(
    const std::uint32_t* sequence, std::uint32_t sequence_count,
    std::uint32_t event_count, const std::uint32_t* candidates,
    bcc32_cuda_resident_synthesis::ResidentSynthesisPolicyState* state) {
  namespace policy = bcc32_cuda_resident_synthesis;
  if (blockIdx.x != 0u || threadIdx.x != 0u) return;
  const std::uint32_t begin = sequence_count - event_count;
  for (std::uint32_t event = 0u; event < event_count; ++event) {
    (void)policy::resident_synthesis_policy_predict(
        state, candidates + event * policy::kResidentSynthesisPolicyVariants);
    (void)policy::resident_synthesis_policy_observe(state, sequence[begin + event]);
  }
}

void advance_synthesis_policy_incrementally(
    adult::AdultState& state, const std::uint32_t* sequence,
    std::uint32_t sequence_count) {
  namespace policy = bcc32_cuda_resident_synthesis;
  if (sequence_count < 3u) return;
  const std::uint32_t event_count = std::min(
      sequence_count - 2u, policy::kResidentSynthesisPolicyWindow);
  adult::DeviceArray<std::uint32_t> candidates(
      event_count * policy::kResidentSynthesisPolicyVariants);
  compute_policy_candidates_kernel<<<
      event_count * policy::kResidentSynthesisPolicyVariants, adult::kBlock>>>(
      sequence, sequence_count, state.bigrams.get(), state.bigram_counts.get(),
      state.bigram_count, state.trigrams.get(), state.trigram_counts.get(),
      state.trigram_count, state.online_bigrams.get(), state.online_bigram_counts.get(),
      state.online_bigram_count, state.online_trigrams.get(),
      state.online_trigram_counts.get(), state.online_trigram_count, event_count,
      candidates.get());
  const std::uint32_t* observation_sequence = sequence;
  std::uint32_t observation_sequence_count = sequence_count;
  std::uint32_t observation_event_count = event_count;
  const std::uint32_t* observation_candidates = candidates.get();
  auto* observation_state = state.synthesis_policy.get();
  void* observation_arguments[] = {
      &observation_sequence, &observation_sequence_count, &observation_event_count,
      &observation_candidates, &observation_state};
  adult::cuda_require(
      cudaLaunchKernel(reinterpret_cast<const void*>(apply_policy_observations_kernel),
                       dim3{1u, 1u, 1u}, dim3{2u, 1u, 1u}, observation_arguments, 0u,
                       nullptr),
      "launch parallel resident synthesis policy");
}

struct CandidateAssimilationReport {
  incremental::MergeReport bigram_merge;
  incremental::MergeReport trigram_merge;
  incremental::MergeReport association_merge;
  incremental::MergeReport conditioned_transition_merge;
  std::uint32_t sequence_count = 0u;
  bool exact_replay = false;
  double prepare_ms = 0.0;
  double policy_ms = 0.0;
  double append_ms = 0.0;
  double merge_ms = 0.0;
  double unigram_ms = 0.0;
  double audit_ms = 0.0;
};

CandidateAssimilationReport assimilate_incrementally(
    adult::AdultState& state, const std::uint8_t* host_bytes, std::uint32_t byte_count,
    incremental::Workspace<adult::BigramKey>& bigram_workspace,
    incremental::Workspace<adult::TrigramKey>& trigram_workspace,
    incremental::Workspace<adult::AssociationKey>& association_workspace,
    incremental::Workspace<adult::ConditionedTransitionKey>&
        conditioned_transition_workspace) {
  if (byte_count == 0u) return {};
  CandidateAssimilationReport report{};
  auto stage_begin = Clock::now();
  const std::uint32_t previous_unit_count = state.unit_count;
  const std::uint32_t previous_bigram_count = state.online_bigram_count;
  const std::uint32_t previous_trigram_count = state.online_trigram_count;
  const std::uint32_t previous_association_count = state.online_association_count;
  const std::uint32_t previous_conditioned_transition_count =
      state.online_conditioned_transition_count;
  adult::DeviceArray<std::uint8_t> device_bytes(byte_count);
  adult::DeviceArray<std::uint32_t> sequence(byte_count);
  adult::DeviceArray<std::uint32_t> status(1u);
  adult::DeviceArray<std::uint32_t> exact_replay(1u);
  adult::cuda_require(cudaMemcpy(device_bytes.get(), host_bytes, byte_count,
                                 cudaMemcpyHostToDevice),
                      "upload incremental online raw bytes");
  // Incremental and full post-action assimilation must observe the same
  // physical byte surface. Control bytes are residently normalized before
  // replay detection, boundary learning, and unit formation on both paths.
  adult::normalize_surface_whitespace_kernel<<<
      adult::blocks_for(byte_count), adult::kBlock>>>(device_bytes.get(),
                                                       byte_count);
  adult::cuda_require(cudaMemset(exact_replay.get(), 0, exact_replay.bytes()),
                      "clear incremental replay gate");
  if (state.online_episode_count != 0u) {
    adult::detect_exact_resident_bytes_replay_kernel<<<
        adult::blocks_for(state.online_episode_count), adult::kBlock>>>(
        device_bytes.get(), byte_count, state.online_episode_units.get(),
        state.online_episode_count, state.online_episode_breaks.get(),
        state.online_episode_break_count, state.unit_lengths.get(),
        state.unit_content.get(), state.unit_count, 1u, 0u, exact_replay.get());
  }
  if (!state.base_episode_lesioned && state.unit_occurrences != 0u) {
    adult::detect_exact_resident_bytes_replay_kernel<<<
        adult::blocks_for(state.unit_occurrences), adult::kBlock>>>(
        device_bytes.get(), byte_count, state.base_episode_units.get(),
        state.unit_occurrences, nullptr, 0u, state.unit_lengths.get(),
        state.unit_content.get(), state.unit_count, 0u, 0u, exact_replay.get());
  }
  std::uint32_t host_exact_replay = 0u;
  adult::cuda_require(cudaMemcpy(&host_exact_replay, exact_replay.get(),
                                 sizeof(host_exact_replay), cudaMemcpyDeviceToHost),
                      "read incremental exact-replay gate");
  if (host_exact_replay != 0u) {
    report.exact_replay = true;
    report.prepare_ms = milliseconds(stage_begin, Clock::now());
    return report;
  }
  const std::uint32_t boundary_mass = byte_count * 2u - 1u;
  adult::reserve_fixed_mass_kernel<<<1u, 32u>>>(boundary_mass, state.ledger.get(),
                                                status.get());
  adult::cuda_require(cudaDeviceSynchronize(), "reserve incremental boundary evidence mass");
  std::uint32_t boundary_status = 0u;
  adult::cuda_require(cudaMemcpy(&boundary_status, status.get(), sizeof(boundary_status),
                                 cudaMemcpyDeviceToHost),
                      "read incremental boundary evidence status");
  if (boundary_status != 0u) {
    throw std::runtime_error("incremental boundary evidence exceeds fixed resident mass");
  }

  adult::byte_statistics_kernel<<<std::min(4096u, adult::blocks_for(byte_count)),
                                  adult::kBlock>>>(
      device_bytes.get(), byte_count, state.boundary_histogram.get(),
      state.boundary_pairs.get());
  adult::discover_boundary_kernel<<<1u, 256u>>>(
      state.boundary_histogram.get(), state.boundary_pairs.get(),
      state.boundary_mask.get(), state.boundary_bytes.get());
  adult::discover_closure_kernel<<<1u, 256u>>>(
      state.boundary_histogram.get(), state.boundary_pairs.get(),
      state.boundary_bytes.get(), state.closure_bytes.get());

  adult::DeviceArray<std::uint32_t> flags(byte_count);
  adult::DeviceArray<std::uint32_t> anchors(byte_count);
  adult::DeviceArray<std::uint32_t> scanned_ids(byte_count);
  adult::mark_base_boundaries_kernel<<<adult::blocks_for(byte_count), adult::kBlock>>>(
      device_bytes.get(), byte_count, state.boundary_mask.get(), flags.get(), anchors.get());
  thrust::inclusive_scan(thrust::device, anchors.get(), anchors.get() + byte_count,
                         anchors.get(), thrust::maximum<std::uint32_t>());
  adult::mark_bounded_units_kernel<<<adult::blocks_for(byte_count), adult::kBlock>>>(
      byte_count, anchors.get(), flags.get());
  thrust::inclusive_scan(thrust::device, flags.get(), flags.get() + byte_count,
                         scanned_ids.get());
  std::uint32_t sequence_count = 0u;
  adult::cuda_require(cudaMemcpy(&sequence_count, scanned_ids.get() + byte_count - 1u,
                                 sizeof(sequence_count), cudaMemcpyDeviceToHost),
                      "read incremental online unit extent");
  adult::DeviceArray<std::uint32_t> starts(sequence_count);
  adult::scatter_unit_starts_kernel<<<adult::blocks_for(byte_count), adult::kBlock>>>(
      byte_count, flags.get(), scanned_ids.get(), starts.get());
  adult::DeviceArray<adult::UnitKey> sorted_keys(sequence_count);
  adult::DeviceArray<std::uint32_t> sorted_occurrences(sequence_count);
  adult::hash_units_kernel<<<adult::blocks_for(sequence_count), adult::kBlock>>>(
      device_bytes.get(), byte_count, starts.get(), sequence_count,
      sorted_keys.get(), sorted_occurrences.get());
  thrust::stable_sort_by_key(thrust::device, sorted_keys.get(),
                             sorted_keys.get() + sequence_count,
                             sorted_occurrences.get());
  adult::DeviceArray<std::uint32_t> unique_flags(sequence_count);
  adult::DeviceArray<std::uint32_t> group_ids(sequence_count);
  adult::mark_unique_units_kernel<<<adult::blocks_for(sequence_count), adult::kBlock>>>(
      device_bytes.get(), byte_count, starts.get(), sequence_count,
      sorted_keys.get(), sorted_occurrences.get(), unique_flags.get());
  thrust::inclusive_scan(thrust::device, unique_flags.get(),
                         unique_flags.get() + sequence_count, group_ids.get());
  std::uint32_t unique_count = 0u;
  adult::cuda_require(cudaMemcpy(&unique_count, group_ids.get() + sequence_count - 1u,
                                 sizeof(unique_count), cudaMemcpyDeviceToHost),
                      "read incremental unique unit extent");
  adult::DeviceArray<std::uint32_t> representatives(unique_count);
  adult::DeviceArray<std::uint32_t> group_units(unique_count);
  adult::DeviceArray<std::uint32_t> novel_flags(unique_count);
  adult::DeviceArray<std::uint32_t> novel_ids(unique_count);
  adult::cuda_require(cudaMemset(group_units.get(), 0xff, group_units.bytes()),
                      "clear incremental deterministic unit groups");
  adult::cuda_require(cudaMemset(novel_flags.get(), 0, novel_flags.bytes()),
                      "clear incremental deterministic novelty flags");
  adult::scatter_unique_assimilation_representatives_kernel<<<
      adult::blocks_for(sequence_count), adult::kBlock>>>(
      sorted_occurrences.get(), unique_flags.get(), group_ids.get(),
      sequence_count, representatives.get());
  adult::resolve_unique_assimilation_units_kernel<<<
      adult::blocks_for(unique_count), adult::kBlock>>>(
      device_bytes.get(), byte_count, starts.get(), sequence_count,
      representatives.get(), unique_count,
      state.unit_lengths.get(), state.unit_content.get(), state.unit_hash_slots.get(),
      state.unit_hash_capacity, group_units.get(), novel_flags.get(), status.get(),
      exact_replay.get());
  thrust::inclusive_scan(thrust::device, novel_flags.get(),
                         novel_flags.get() + unique_count, novel_ids.get());
  std::uint32_t novel_count = 0u;
  std::uint32_t materialize_status = 0u;
  adult::cuda_require(cudaMemcpy(&novel_count, novel_ids.get() + unique_count - 1u,
                                 sizeof(novel_count), cudaMemcpyDeviceToHost),
                      "read incremental deterministic novelty extent");
  adult::cuda_require(cudaMemcpy(&materialize_status, status.get(),
                                 sizeof(materialize_status), cudaMemcpyDeviceToHost),
                      "read incremental deterministic lookup status");
  if (materialize_status != 0u) {
    throw std::runtime_error("incremental resident unit fingerprint index is full");
  }
  if (novel_count > state.unit_capacity - state.unit_count) {
    throw std::runtime_error("incremental online unit reserve exhausted");
  }
  adult::materialize_novel_assimilation_units_kernel<<<
      adult::blocks_for(unique_count), adult::kBlock>>>(
      device_bytes.get(), byte_count, starts.get(), sequence_count,
      representatives.get(), unique_count, novel_flags.get(), novel_ids.get(),
      state.unit_count, state.unit_lengths.get(), state.unit_content.get(),
      state.unit_vitality.get(), group_units.get(), exact_replay.get());
  if (novel_count != 0u) {
    adult::populate_unit_hash_range_kernel<<<
        adult::blocks_for(novel_count), adult::kBlock>>>(
        state.unit_lengths.get(), state.unit_content.get(), state.unit_count,
        novel_count, state.unit_hash_slots.get(), state.unit_hash_capacity);
    const std::uint32_t updated_unit_count = state.unit_count + novel_count;
    adult::cuda_require(cudaMemcpy(state.mutable_sizes.get(), &updated_unit_count,
                                   sizeof(updated_unit_count),
                                   cudaMemcpyHostToDevice),
                        "publish incremental deterministic unit extent");
    // Keep the host extent coherent with the resident mirror before the
    // remaining online passes consume it.
    state.unit_count = updated_unit_count;
  }
  adult::map_assimilation_occurrence_groups_kernel<<<adult::blocks_for(sequence_count),
                                                     adult::kBlock>>>(
      sorted_occurrences.get(), group_ids.get(), sequence_count, group_units.get(), sequence.get(),
      exact_replay.get());
  adult::cuda_require(cudaDeviceSynchronize(), "complete incremental unit materialization");
  adult::DeviceArray<std::uint32_t> segment_starts(sequence_count);
  adult::DeviceArray<std::uint32_t> segment_ids(sequence_count);
  adult::mark_sequence_segment_starts_kernel<<<adult::blocks_for(sequence_count), adult::kBlock>>>(
      sequence.get(), sequence_count, state.unit_lengths.get(), state.unit_content.get(),
      state.closure_bytes.get(), state.boundary_bytes.get(), segment_starts.get());
  thrust::inclusive_scan(thrust::device, segment_starts.get(),
                         segment_starts.get() + sequence_count, segment_ids.get());
  report.prepare_ms = milliseconds(stage_begin, Clock::now());
  stage_begin = Clock::now();
  advance_synthesis_policy_incrementally(state, sequence.get(), sequence_count);
  adult::cuda_require(cudaDeviceSynchronize(), "complete incremental synthesis policy");
  report.policy_ms = milliseconds(stage_begin, Clock::now());
  stage_begin = Clock::now();
  append_online_learning_delta_kernel<<<1u, 32u>>>(
      state.unit_lengths.get(), state.unit_content.get(), state.unit_vitality.get(),
      state.online_bigrams.get(), state.online_bigram_counts.get(),
      state.online_trigrams.get(), state.online_trigram_counts.get(),
      state.online_associations.get(), state.online_association_counts.get(),
      state.online_episode_units.get(), state.online_episode_breaks.get(),
      state.mutable_sizes.get(), sequence.get(), sequence_count,
      segment_ids.get(), state.closure_bytes.get(), state.ledger.get(),
      status.get());
  adult::cuda_require(cudaGetLastError(), "launch incremental raw-byte assimilation");
  adult::cuda_require(cudaDeviceSynchronize(), "complete incremental raw-byte assimilation");
  report.append_ms = milliseconds(stage_begin, Clock::now());

  std::uint32_t host_status = 0u;
  std::uint32_t sizes[7] = {};
  adult::cuda_require(cudaMemcpy(&host_status, status.get(), sizeof(host_status),
                                 cudaMemcpyDeviceToHost),
                      "read incremental assimilation status");
  adult::cuda_require(cudaMemcpy(sizes, state.mutable_sizes.get(), sizeof(sizes),
                                 cudaMemcpyDeviceToHost),
                      "read incremental mutable state extents");
  if (host_status != 0u) {
    throw std::runtime_error("incremental online assimilation capacity or mass failure " +
                             std::to_string(host_status));
  }
  const std::uint32_t conditioned_transition_events =
      adult::conditioned_transition_event_count(sequence_count);
  if (conditioned_transition_events >
      adult::kOnlineConditionedTransitionCapacity -
          previous_conditioned_transition_count) {
    throw std::runtime_error(
        "incremental conditioned transition reserve exhausted");
  }
  if (conditioned_transition_events != 0u) {
    adult::reserve_fixed_mass_kernel<<<1u, 32u>>>(
        conditioned_transition_events, state.ledger.get(), status.get());
    adult::cuda_require(cudaDeviceSynchronize(),
                        "reserve incremental conditioned transition mass");
    adult::cuda_require(cudaMemcpy(&host_status, status.get(), sizeof(host_status),
                                   cudaMemcpyDeviceToHost),
                        "read incremental conditioned transition reserve status");
    if (host_status != 0u) {
      throw std::runtime_error(
          "incremental conditioned transitions exceed fixed resident mass");
    }
    adult::DeviceArray<std::uint32_t> appended_count(1u);
    adult::cuda_require(cudaMemset(appended_count.get(), 0,
                                   appended_count.bytes()),
                        "clear incremental conditioned append extent");
    adult::append_conditioned_transitions_kernel<<<
        std::min(4096u, adult::blocks_for(conditioned_transition_events)),
        adult::kBlock>>>(
        sequence.get(), sequence_count, 0u, segment_ids.get(),
        state.online_conditioned_transitions.get(),
        state.online_conditioned_transition_counts.get(),
        previous_conditioned_transition_count, conditioned_transition_events,
        appended_count.get(), state.ledger.get());
    adult::cuda_require(cudaGetLastError(),
                        "append incremental resident conditioned transitions");
    adult::cuda_require(cudaDeviceSynchronize(),
                        "complete incremental conditioned transition append");
    std::uint32_t appended = 0u;
    adult::cuda_require(cudaMemcpy(&appended, appended_count.get(),
                                   sizeof(appended), cudaMemcpyDeviceToHost),
                        "read incremental boundary-aware conditioned extent");
    sizes[6] = previous_conditioned_transition_count + appended;
  } else {
    sizes[6] = previous_conditioned_transition_count;
  }
  if (sizes[3] < previous_association_count) {
    throw std::runtime_error("incremental association extent regressed");
  }
  state.unit_count = sizes[0];
  state.online_bigram_count = sizes[1];
  state.online_trigram_count = sizes[2];
  state.online_episode_count = sizes[4];
  state.online_episode_break_count = sizes[5];

  report.sequence_count = sequence_count;
  stage_begin = Clock::now();
  report.bigram_merge = incremental::merge_sorted_delta(
      state.online_bigrams.get(), state.online_bigram_counts.get(),
      previous_bigram_count, adult::kOnlineNgramCapacity,
      state.online_bigrams.get() + previous_bigram_count,
      state.online_bigram_counts.get() + previous_bigram_count,
      sizes[1] - previous_bigram_count, bigram_workspace,
      state.ledger.get() + 1u, state.ledger.get() + 2u);
  report.trigram_merge = incremental::merge_sorted_delta(
      state.online_trigrams.get(), state.online_trigram_counts.get(),
      previous_trigram_count, adult::kOnlineNgramCapacity,
      state.online_trigrams.get() + previous_trigram_count,
      state.online_trigram_counts.get() + previous_trigram_count,
      sizes[2] - previous_trigram_count, trigram_workspace,
      state.ledger.get() + 1u, state.ledger.get() + 2u);
  report.association_merge = incremental::merge_sorted_delta(
      state.online_associations.get(), state.online_association_counts.get(),
      previous_association_count, adult::kOnlineAssociationCapacity,
      state.online_associations.get() + previous_association_count,
      state.online_association_counts.get() + previous_association_count,
      sizes[3] - previous_association_count, association_workspace,
      state.ledger.get() + 1u, state.ledger.get() + 2u);
  report.conditioned_transition_merge = incremental::merge_sorted_delta(
      state.online_conditioned_transitions.get(),
      state.online_conditioned_transition_counts.get(),
      previous_conditioned_transition_count,
      adult::kOnlineConditionedTransitionCapacity,
      state.online_conditioned_transitions.get() +
          previous_conditioned_transition_count,
      state.online_conditioned_transition_counts.get() +
          previous_conditioned_transition_count,
      sizes[6] - previous_conditioned_transition_count,
      conditioned_transition_workspace, state.ledger.get() + 1u,
      state.ledger.get() + 2u);
  state.online_bigram_count = report.bigram_merge.output_resident_count;
  state.online_trigram_count = report.trigram_merge.output_resident_count;
  state.online_association_count = report.association_merge.output_resident_count;
  state.online_conditioned_transition_count =
      report.conditioned_transition_merge.output_resident_count;
  sizes[1] = state.online_bigram_count;
  sizes[2] = state.online_trigram_count;
  sizes[3] = state.online_association_count;
  sizes[6] = state.online_conditioned_transition_count;
  adult::cuda_require(cudaMemcpy(state.mutable_sizes.get(), sizes, sizeof(sizes),
                                 cudaMemcpyHostToDevice),
                      "publish incrementally merged resident extents");
  adult::cuda_require(cudaDeviceSynchronize(), "complete incremental resident merges");
  report.merge_ms = milliseconds(stage_begin, Clock::now());
  stage_begin = Clock::now();
  adult::build_unigram_top_kernel<<<1u, adult::kBlock>>>(
      state.unit_vitality.get(), state.unit_count, state.unigram_top_ids.get());
  adult::cuda_require(cudaDeviceSynchronize(), "complete incremental unigram update");
  report.unigram_ms = milliseconds(stage_begin, Clock::now());
  stage_begin = Clock::now();
  adult::audit_ledger_kernel<<<1u, adult::kBlock>>>(
      state.unit_vitality.get(), state.unit_count, state.bigram_counts.get(),
      state.bigram_count, state.trigram_counts.get(), state.trigram_count,
      state.online_bigram_counts.get(), state.online_bigram_count,
      state.online_trigram_counts.get(), state.online_trigram_count,
      state.online_association_counts.get(), state.online_association_count,
      state.online_conditioned_transition_counts.get(),
      state.online_conditioned_transition_count,
      (state.base_episode_lesioned ? 0u : state.unit_occurrences) +
          state.online_episode_count,
      state.boundary_histogram.get(), state.boundary_pairs.get(), state.ledger.get());
  adult::cuda_require(cudaGetLastError(), "audit incrementally learned mass");
  adult::cuda_require(cudaDeviceSynchronize(), "complete incremental learned mass audit");
  adult::learn_incremental_surface_episodes(
      state, sequence.get(), sequence_count, segment_ids.get(), previous_unit_count);
  report.audit_ms = milliseconds(stage_begin, Clock::now());
  return report;
}


struct Workspace {
  static std::uint32_t conditioned_capacity(
      std::uint32_t association_delta_capacity) {
    const std::uint32_t raw_byte_capacity =
        (association_delta_capacity + adult::kAssociationRadius - 1u) /
        adult::kAssociationRadius;
    return adult::conditioned_transition_event_count(raw_byte_capacity);
  }

  explicit Workspace(std::uint32_t delta_capacity)
      : bigrams(adult::kOnlineNgramCapacity, delta_capacity, 4096u),
        trigrams(adult::kOnlineNgramCapacity, delta_capacity, 4096u),
        associations(adult::kOnlineAssociationCapacity, delta_capacity, 4096u),
        conditioned_transitions(adult::kOnlineConditionedTransitionCapacity,
                                conditioned_capacity(delta_capacity), 4096u) {}

  incremental::Workspace<adult::BigramKey> bigrams;
  incremental::Workspace<adult::TrigramKey> trigrams;
  incremental::Workspace<adult::AssociationKey> associations;
  incremental::Workspace<adult::ConditionedTransitionKey>
      conditioned_transitions;
};

inline CandidateAssimilationReport assimilate(
    adult::AdultState& state, const std::uint8_t* bytes,
    std::uint32_t byte_count, Workspace& workspace) {
  if (static_cast<unsigned long long>(byte_count) * adult::kAssociationRadius >
      workspace.associations.delta_capacity) {
    throw std::runtime_error("incremental adult contact exceeds workspace capacity");
  }
  if (adult::conditioned_transition_event_count(byte_count) >
      workspace.conditioned_transitions.delta_capacity) {
    throw std::runtime_error(
        "incremental adult conditioned contact exceeds workspace capacity");
  }
  return assimilate_incrementally(
      state, bytes, byte_count, workspace.bigrams, workspace.trigrams,
      workspace.associations, workspace.conditioned_transitions);
}

}  // namespace bcc32_cuda_incremental_adult
