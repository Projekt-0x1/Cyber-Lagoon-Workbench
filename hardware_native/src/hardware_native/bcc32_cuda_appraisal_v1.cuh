#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace bcc32_cuda_appraisal_v1 {

constexpr std::uint32_t kGenomeParameters = 8u;
constexpr std::uint32_t kCandidateCount = 256u;
constexpr std::uint32_t kBlacklistCapacity = 512u;
constexpr std::uint32_t kProposalAttempts = 32u;
constexpr std::uint32_t kOperationalHoldoutBytes = 4096u;
constexpr std::int32_t kParameterLimit = 1024;
constexpr std::int32_t kParameterMassOffset = kParameterLimit;
constexpr std::uint32_t kInitialFreeQuanta = 4096u;

struct Genome {
  std::int32_t parameter[kGenomeParameters];
  std::uint32_t free_quanta;
  std::uint32_t revision;
};

struct FailedProposal {
  Genome parent;
  std::uint32_t parameter_index;
  std::int32_t replacement;
};

struct FailureMemory {
  FailedProposal entries[kBlacklistCapacity];
  std::uint32_t count;
  std::uint32_t head;
};

struct AppraisalControl {
  std::uint32_t appraisal_intact;
  std::uint32_t lesion_events;
  std::uint32_t generations;
  std::uint32_t reserved;
};

struct ResidentMassLedger {
  std::uint64_t adult_bytes;
  std::uint64_t failure_bytes;
  std::uint64_t control_bytes;
  std::uint64_t ledger_bytes;
  std::uint64_t receipt_bytes;
  std::uint64_t padding_bytes;
  std::uint64_t workspace_bytes;
  std::uint64_t resident_bytes;
  std::uint64_t total_bytes;
  std::uint64_t initial_parameter_quanta;
};

struct AppraisalReceipt {
  Genome before_genome;
  Genome after_genome;
  std::uint64_t before_genome_hash;
  std::uint64_t after_genome_hash;
  std::uint64_t baseline_loss;
  std::uint64_t best_shadow_loss;
  std::uint64_t after_loss;
  std::uint64_t predictor_hash_before;
  std::uint64_t predictor_hash_after;
  std::uint64_t parameter_quanta_before;
  std::uint64_t parameter_quanta_after;
  std::uint64_t resident_bytes_before;
  std::uint64_t resident_bytes_after;
  std::uint64_t proposal_fingerprint;
  std::uint64_t first_proposal_signature;
  std::uint64_t stream_byte_begin;
  std::uint64_t stream_byte_end;
  std::uint32_t raw_byte_count;
  std::uint32_t heldout_begin;
  std::uint32_t heldout_count;
  std::uint32_t generation;
  std::uint32_t best_candidate;
  std::uint32_t best_parameter_index;
  std::int32_t best_delta;
  std::int32_t best_replacement;
  std::uint32_t first_parameter_index;
  std::int32_t first_replacement;
  std::uint32_t accepted;
  std::uint32_t strict_improvement;
  std::uint32_t appraisal_intact;
  std::uint32_t lesion_blocked_improvement;
  std::uint32_t rollback_exact;
  std::uint32_t blacklist_count_before;
  std::uint32_t blacklist_count_after;
  std::uint32_t blacklist_skips;
  std::uint32_t rejected_proposals;
  std::uint32_t valid_proposals;
  std::uint32_t accounting_ok;
  std::uint32_t parameter_mass_ok;
  std::uint32_t source_reafferent;
  std::uint32_t parallel_summary;
  std::uint32_t reserved;
};

struct ResidentAppraisal {
  Genome adult;
  FailureMemory failure_memory;
  AppraisalControl control;
  ResidentMassLedger mass;
  AppraisalReceipt receipt;
};

struct ShadowCandidate {
  Genome genome;
  std::uint64_t signature;
  std::uint64_t loss;
  std::uint64_t prediction_hash;
  std::uint32_t parameter_index;
  std::int32_t delta;
  std::int32_t replacement;
  std::uint32_t valid;
};

struct AppraisalWorkspace {
  ShadowCandidate candidates[kCandidateCount];
  std::uint64_t baseline_loss;
  std::uint64_t baseline_prediction_hash;
};

constexpr std::size_t kResidentComponentBytes =
    sizeof(Genome) + sizeof(FailureMemory) + sizeof(AppraisalControl) +
    sizeof(ResidentMassLedger) + sizeof(AppraisalReceipt);
static_assert(sizeof(ResidentAppraisal) >= kResidentComponentBytes,
              "resident appraisal component accounting underflow");
static_assert(kCandidateCount == 256u,
              "the proposal kernel requires one full CUDA block");

__device__ __forceinline__ std::uint64_t mix64(std::uint64_t value) {
  value ^= value >> 30u;
  value *= 0xbf58476d1ce4e5b9ull;
  value ^= value >> 27u;
  value *= 0x94d049bb133111ebull;
  return value ^ (value >> 31u);
}

__device__ __forceinline__ bool same_genome(const Genome& left,
                                            const Genome& right) {
  if (left.free_quanta != right.free_quanta || left.revision != right.revision)
    return false;
  for (std::uint32_t i = 0u; i < kGenomeParameters; ++i)
    if (left.parameter[i] != right.parameter[i]) return false;
  return true;
}

__device__ __forceinline__ std::uint64_t genome_hash(const Genome& genome) {
  std::uint64_t hash = 0x243f6a8885a308d3ull;
  for (std::uint32_t i = 0u; i < kGenomeParameters; ++i) {
    const std::uint64_t word = static_cast<std::uint32_t>(genome.parameter[i]);
    hash = mix64(hash ^ (word + (static_cast<std::uint64_t>(i) << 32u)));
  }
  hash = mix64(hash ^ genome.free_quanta);
  return mix64(hash ^ (static_cast<std::uint64_t>(genome.revision) << 32u));
}

__device__ __forceinline__ std::uint64_t parameter_quanta(
    const Genome& genome) {
  std::uint64_t total = genome.free_quanta;
  for (std::uint32_t i = 0u; i < kGenomeParameters; ++i)
    total += static_cast<std::uint64_t>(
        static_cast<std::int64_t>(genome.parameter[i]) + kParameterMassOffset);
  return total;
}

__device__ __forceinline__ std::uint8_t predict_raw_byte(
    const Genome& genome, const std::uint8_t* raw_bytes, std::uint32_t index) {
  std::int64_t accumulator = genome.parameter[0];
  for (std::uint32_t lag = 1u; lag < kGenomeParameters; ++lag) {
    const std::uint8_t prior = index >= lag ? raw_bytes[index - lag] : 0u;
    accumulator += static_cast<std::int64_t>(genome.parameter[lag]) * prior;
  }
  return static_cast<std::uint8_t>(static_cast<std::uint64_t>(accumulator) & 0xffu);
}

__device__ __forceinline__ std::uint64_t chronological_predictive_loss(
    const Genome& genome, const std::uint8_t* raw_bytes,
    std::uint32_t raw_byte_count, std::uint32_t heldout_begin) {
  if (raw_bytes == nullptr || heldout_begin >= raw_byte_count) return ~0ull;
  std::uint64_t loss = 0u;
  for (std::uint32_t i = heldout_begin; i < raw_byte_count; ++i) {
    const std::int32_t predicted = predict_raw_byte(genome, raw_bytes, i);
    const std::int32_t observed = raw_bytes[i];
    loss += static_cast<std::uint64_t>(predicted >= observed
                                          ? predicted - observed
                                          : observed - predicted);
  }
  return loss;
}

__device__ __forceinline__ std::uint64_t prediction_hash(
    const Genome& genome, const std::uint8_t* raw_bytes,
    std::uint32_t raw_byte_count, std::uint32_t heldout_begin) {
  std::uint64_t hash = 0x6a09e667f3bcc909ull;
  for (std::uint32_t i = heldout_begin; i < raw_byte_count; ++i) {
    hash = mix64(hash ^ (static_cast<std::uint64_t>(predict_raw_byte(
                             genome, raw_bytes, i)) +
                         (static_cast<std::uint64_t>(i) << 8u)));
  }
  return hash;
}

__device__ __forceinline__ std::uint64_t proposal_signature(
    const Genome& parent, std::uint32_t parameter_index,
    std::int32_t replacement) {
  const std::uint64_t edit =
      (static_cast<std::uint64_t>(parameter_index) << 32u) |
      static_cast<std::uint32_t>(replacement);
  return mix64(genome_hash(parent) ^ edit ^ 0xa4093822299f31d0ull);
}

__device__ __forceinline__ bool is_blacklisted(
    const FailureMemory& memory, const Genome& parent,
    std::uint32_t parameter_index, std::int32_t replacement) {
  for (std::uint32_t i = 0u; i < memory.count; ++i) {
    const FailedProposal& failed = memory.entries[i];
    if (failed.parameter_index == parameter_index &&
        failed.replacement == replacement && same_genome(failed.parent, parent))
      return true;
  }
  return false;
}

__device__ __forceinline__ void remember_failure(
    FailureMemory& memory, const Genome& parent,
    std::uint32_t parameter_index, std::int32_t replacement) {
  FailedProposal& destination = memory.entries[memory.head];
  destination.parent = parent;
  destination.parameter_index = parameter_index;
  destination.replacement = replacement;
  memory.head = (memory.head + 1u) % kBlacklistCapacity;
  if (memory.count < kBlacklistCapacity) ++memory.count;
}

__device__ __forceinline__ bool exact_accounting(
    const ResidentAppraisal& resident) {
  const ResidentMassLedger& mass = resident.mass;
  const std::uint64_t component_sum =
      mass.adult_bytes + mass.failure_bytes + mass.control_bytes +
      mass.ledger_bytes + mass.receipt_bytes + mass.padding_bytes;
  return mass.adult_bytes == sizeof(Genome) &&
      mass.failure_bytes == sizeof(FailureMemory) &&
      mass.control_bytes == sizeof(AppraisalControl) &&
      mass.ledger_bytes == sizeof(ResidentMassLedger) &&
      mass.receipt_bytes == sizeof(AppraisalReceipt) &&
      mass.padding_bytes == sizeof(ResidentAppraisal) - kResidentComponentBytes &&
      mass.workspace_bytes == sizeof(AppraisalWorkspace) &&
      mass.resident_bytes == sizeof(ResidentAppraisal) &&
      component_sum == mass.resident_bytes &&
      mass.total_bytes == mass.resident_bytes + mass.workspace_bytes;
}

__global__ void initialize_appraisal_kernel(ResidentAppraisal* resident) {
  if (blockIdx.x != 0u || threadIdx.x != 0u) return;
  for (std::uint32_t i = 0u; i < kGenomeParameters; ++i)
    resident->adult.parameter[i] = 0;
  resident->adult.parameter[1] = 1;
  resident->adult.free_quanta = kInitialFreeQuanta;
  resident->adult.revision = 0u;
  resident->failure_memory.count = 0u;
  resident->failure_memory.head = 0u;
  resident->control.appraisal_intact = 1u;
  resident->control.lesion_events = 0u;
  resident->control.generations = 0u;
  resident->control.reserved = 0u;
  resident->mass.adult_bytes = sizeof(Genome);
  resident->mass.failure_bytes = sizeof(FailureMemory);
  resident->mass.control_bytes = sizeof(AppraisalControl);
  resident->mass.ledger_bytes = sizeof(ResidentMassLedger);
  resident->mass.receipt_bytes = sizeof(AppraisalReceipt);
  resident->mass.padding_bytes = sizeof(ResidentAppraisal) - kResidentComponentBytes;
  resident->mass.workspace_bytes = sizeof(AppraisalWorkspace);
  resident->mass.resident_bytes = sizeof(ResidentAppraisal);
  resident->mass.total_bytes = sizeof(ResidentAppraisal) + sizeof(AppraisalWorkspace);
  resident->mass.initial_parameter_quanta = parameter_quanta(resident->adult);
  auto* receipt_bytes = reinterpret_cast<unsigned char*>(&resident->receipt);
  for (std::size_t i = 0u; i < sizeof(AppraisalReceipt); ++i) receipt_bytes[i] = 0u;
}

__global__ void lesion_appraisal_kernel(ResidentAppraisal* resident) {
  if (blockIdx.x != 0u || threadIdx.x != 0u) return;
  resident->control.appraisal_intact = 0u;
  ++resident->control.lesion_events;
}

__global__ void propose_mutations_kernel(ResidentAppraisal* resident,
                                         AppraisalWorkspace* workspace) {
  const std::uint32_t candidate_index = blockIdx.x * blockDim.x + threadIdx.x;
  if (candidate_index >= kCandidateCount) return;

  if (candidate_index == 0u) {
    auto* receipt_bytes = reinterpret_cast<unsigned char*>(&resident->receipt);
    for (std::size_t i = 0u; i < sizeof(AppraisalReceipt); ++i)
      receipt_bytes[i] = 0u;
    resident->receipt.before_genome = resident->adult;
    resident->receipt.before_genome_hash = genome_hash(resident->adult);
    resident->receipt.parameter_quanta_before = parameter_quanta(resident->adult);
    resident->receipt.resident_bytes_before = resident->mass.total_bytes;
    resident->receipt.generation = resident->control.generations;
    resident->receipt.blacklist_count_before = resident->failure_memory.count;
    resident->receipt.appraisal_intact = resident->control.appraisal_intact;
  }
  __syncthreads();

  ShadowCandidate proposal{};
  const Genome parent = resident->adult;
  for (std::uint32_t attempt = 0u; attempt < kProposalAttempts; ++attempt) {
    const std::uint32_t ordinal = candidate_index + attempt * kCandidateCount;
    const std::uint32_t parameter_index = ordinal % kGenomeParameters;
    const std::uint32_t step_code = ordinal / kGenomeParameters;
    const std::int32_t magnitude = static_cast<std::int32_t>(step_code / 2u + 1u);
    const std::int32_t delta = (step_code & 1u) == 0u ? -magnitude : magnitude;
    const std::int64_t replacement =
        static_cast<std::int64_t>(parent.parameter[parameter_index]) + delta;
    const std::int64_t free_quanta =
        static_cast<std::int64_t>(parent.free_quanta) - delta;
    if (replacement < -kParameterLimit || replacement > kParameterLimit ||
        free_quanta < 0 || free_quanta > 0xffffffffll)
      continue;
    if (is_blacklisted(resident->failure_memory, parent, parameter_index,
                       static_cast<std::int32_t>(replacement))) {
      atomicAdd(&resident->receipt.blacklist_skips, 1u);
      continue;
    }

    proposal.genome = parent;
    proposal.genome.parameter[parameter_index] =
        static_cast<std::int32_t>(replacement);
    proposal.genome.free_quanta = static_cast<std::uint32_t>(free_quanta);
    proposal.parameter_index = parameter_index;
    proposal.delta = delta;
    proposal.replacement = static_cast<std::int32_t>(replacement);
    proposal.signature = proposal_signature(parent, parameter_index,
                                            proposal.replacement);
    proposal.valid = 1u;
    break;
  }
  proposal.loss = ~0ull;
  proposal.prediction_hash = 0u;
  workspace->candidates[candidate_index] = proposal;
  if (proposal.valid != 0u) {
    atomicAdd(&resident->receipt.valid_proposals, 1u);
    atomicXor(reinterpret_cast<unsigned long long*>(
                  &resident->receipt.proposal_fingerprint),
              static_cast<unsigned long long>(proposal.signature));
  }
  if (candidate_index == 0u) {
    resident->receipt.first_proposal_signature = proposal.signature;
    resident->receipt.first_parameter_index = proposal.parameter_index;
    resident->receipt.first_replacement = proposal.replacement;
  }
}

__global__ void shadow_evaluate_kernel(
    const std::uint8_t* raw_bytes, std::uint32_t raw_byte_count,
    std::uint32_t heldout_begin, AppraisalWorkspace* workspace) {
  const std::uint32_t candidate_index = blockIdx.x * blockDim.x + threadIdx.x;
  if (candidate_index >= kCandidateCount) return;
  ShadowCandidate& proposal = workspace->candidates[candidate_index];
  if (proposal.valid == 0u) return;
  proposal.loss = chronological_predictive_loss(
      proposal.genome, raw_bytes, raw_byte_count, heldout_begin);
}

// The operational stream launches one block for the resident baseline and one
// block for every shadow genome. Each block reduces its chronological holdout
// in parallel; no candidate or contact-length scan is serialized through one
// thread.
__global__ void shadow_evaluate_parallel_kernel(
    const std::uint8_t* raw_bytes, std::uint32_t raw_byte_count,
    std::uint32_t heldout_begin, const ResidentAppraisal* resident,
    AppraisalWorkspace* workspace) {
  const std::uint32_t evaluation = blockIdx.x;
  if (evaluation > kCandidateCount || blockDim.x != kCandidateCount) return;

  Genome genome{};
  bool valid = true;
  if (evaluation == 0u) {
    genome = resident->adult;
  } else {
    const ShadowCandidate& candidate = workspace->candidates[evaluation - 1u];
    genome = candidate.genome;
    valid = candidate.valid != 0u;
  }

  std::uint64_t local_loss = 0u;
  std::uint64_t local_hash = 0u;
  if (valid) {
    for (std::uint32_t i = heldout_begin + threadIdx.x;
         i < raw_byte_count; i += blockDim.x) {
      const std::int32_t predicted = predict_raw_byte(genome, raw_bytes, i);
      const std::int32_t observed = raw_bytes[i];
      local_loss += static_cast<std::uint64_t>(
          predicted >= observed ? predicted - observed : observed - predicted);
      local_hash ^= mix64(static_cast<std::uint64_t>(predicted) |
                          (static_cast<std::uint64_t>(i) << 8u));
    }
  } else {
    local_loss = ~0ull;
  }

  __shared__ std::uint64_t losses[kCandidateCount];
  __shared__ std::uint64_t hashes[kCandidateCount];
  losses[threadIdx.x] = local_loss;
  hashes[threadIdx.x] = local_hash;
  __syncthreads();
  for (std::uint32_t offset = blockDim.x / 2u; offset != 0u; offset >>= 1u) {
    if (threadIdx.x < offset) {
      if (losses[threadIdx.x] == ~0ull ||
          losses[threadIdx.x + offset] == ~0ull) {
        losses[threadIdx.x] = ~0ull;
      } else {
        losses[threadIdx.x] += losses[threadIdx.x + offset];
      }
      hashes[threadIdx.x] ^= hashes[threadIdx.x + offset];
    }
    __syncthreads();
  }

  if (threadIdx.x == 0u) {
    if (evaluation == 0u) {
      workspace->baseline_loss = losses[0];
      workspace->baseline_prediction_hash = hashes[0];
    } else {
      ShadowCandidate& candidate = workspace->candidates[evaluation - 1u];
      candidate.loss = losses[0];
      candidate.prediction_hash = hashes[0];
    }
  }
}

__global__ void resident_select_commit_parallel_kernel(
    std::uint32_t raw_byte_count, std::uint32_t heldout_begin,
    std::uint64_t stream_byte_begin, std::uint32_t source_reafferent,
    ResidentAppraisal* resident, const AppraisalWorkspace* workspace) {
  if (blockIdx.x != 0u || blockDim.x != kCandidateCount) return;

  const std::uint32_t candidate_index = threadIdx.x;
  const ShadowCandidate& local_candidate = workspace->candidates[candidate_index];
  __shared__ std::uint64_t losses[kCandidateCount];
  __shared__ std::uint32_t indices[kCandidateCount];
  losses[candidate_index] = local_candidate.valid != 0u
                                ? local_candidate.loss
                                : ~0ull;
  indices[candidate_index] = local_candidate.valid != 0u
                                 ? candidate_index
                                 : kCandidateCount;
  __syncthreads();
  for (std::uint32_t offset = blockDim.x / 2u; offset != 0u; offset >>= 1u) {
    if (threadIdx.x < offset) {
      const std::uint64_t other_loss = losses[threadIdx.x + offset];
      const std::uint32_t other_index = indices[threadIdx.x + offset];
      if (other_loss < losses[threadIdx.x] ||
          (other_loss == losses[threadIdx.x] &&
           other_index < indices[threadIdx.x])) {
        losses[threadIdx.x] = other_loss;
        indices[threadIdx.x] = other_index;
      }
    }
    __syncthreads();
  }
  if (threadIdx.x != 0u) return;

  AppraisalReceipt& receipt = resident->receipt;
  const Genome parent = resident->adult;
  receipt.raw_byte_count = raw_byte_count;
  receipt.heldout_begin = heldout_begin;
  receipt.heldout_count =
      heldout_begin < raw_byte_count ? raw_byte_count - heldout_begin : 0u;
  receipt.stream_byte_begin = stream_byte_begin;
  receipt.stream_byte_end = stream_byte_begin + raw_byte_count;
  receipt.source_reafferent = source_reafferent != 0u ? 1u : 0u;
  receipt.parallel_summary = 1u;
  receipt.baseline_loss = workspace->baseline_loss;
  receipt.predictor_hash_before = workspace->baseline_prediction_hash;

  const std::uint64_t best_loss = losses[0];
  const std::uint32_t best_index = indices[0];
  receipt.best_shadow_loss = best_loss;
  receipt.best_candidate = best_index;
  const bool available_improvement =
      best_index < kCandidateCount && best_loss < receipt.baseline_loss;
  bool accepted = false;
  if (available_improvement && resident->control.appraisal_intact != 0u) {
    const ShadowCandidate& best = workspace->candidates[best_index];
    if (parameter_quanta(best.genome) == parameter_quanta(parent)) {
      resident->adult = best.genome;
      ++resident->adult.revision;
      accepted = true;
    }
  }

  if (best_index < kCandidateCount) {
    const ShadowCandidate& best = workspace->candidates[best_index];
    receipt.best_parameter_index = best.parameter_index;
    receipt.best_delta = best.delta;
    receipt.best_replacement = best.replacement;
  }
  for (std::uint32_t i = 0u; i < kCandidateCount; ++i) {
    const ShadowCandidate& candidate = workspace->candidates[i];
    if (candidate.valid == 0u || (accepted && i == best_index)) continue;
    remember_failure(resident->failure_memory, parent,
                     candidate.parameter_index, candidate.replacement);
    ++receipt.rejected_proposals;
  }

  receipt.accepted = accepted ? 1u : 0u;
  receipt.lesion_blocked_improvement =
      available_improvement && resident->control.appraisal_intact == 0u ? 1u : 0u;
  receipt.after_genome = resident->adult;
  receipt.after_genome_hash = genome_hash(resident->adult);
  receipt.after_loss = accepted ? best_loss : receipt.baseline_loss;
  receipt.predictor_hash_after = accepted
      ? workspace->candidates[best_index].prediction_hash
      : receipt.predictor_hash_before;
  receipt.parameter_quanta_after = parameter_quanta(resident->adult);
  receipt.resident_bytes_after = resident->mass.total_bytes;
  receipt.strict_improvement =
      accepted && receipt.after_loss < receipt.baseline_loss ? 1u : 0u;
  receipt.rollback_exact =
      !accepted && same_genome(receipt.before_genome, receipt.after_genome) ? 1u : 0u;
  receipt.blacklist_count_after = resident->failure_memory.count;
  receipt.accounting_ok = exact_accounting(*resident) ? 1u : 0u;
  receipt.parameter_mass_ok =
      receipt.parameter_quanta_before == receipt.parameter_quanta_after &&
              receipt.parameter_quanta_after == resident->mass.initial_parameter_quanta
          ? 1u
          : 0u;
  ++resident->control.generations;
}

__global__ void resident_select_commit_kernel(
    const std::uint8_t* raw_bytes, std::uint32_t raw_byte_count,
    std::uint32_t heldout_begin, ResidentAppraisal* resident,
    const AppraisalWorkspace* workspace) {
  if (blockIdx.x != 0u || threadIdx.x != 0u) return;

  AppraisalReceipt& receipt = resident->receipt;
  const Genome parent = resident->adult;
  receipt.raw_byte_count = raw_byte_count;
  receipt.heldout_begin = heldout_begin;
  receipt.heldout_count =
      heldout_begin < raw_byte_count ? raw_byte_count - heldout_begin : 0u;
  receipt.baseline_loss = chronological_predictive_loss(
      parent, raw_bytes, raw_byte_count, heldout_begin);
  receipt.predictor_hash_before = prediction_hash(
      parent, raw_bytes, raw_byte_count, heldout_begin);

  std::uint64_t best_loss = ~0ull;
  std::uint32_t best_index = kCandidateCount;
  for (std::uint32_t i = 0u; i < kCandidateCount; ++i) {
    const ShadowCandidate& candidate = workspace->candidates[i];
    if (candidate.valid != 0u &&
        (candidate.loss < best_loss ||
         (candidate.loss == best_loss && i < best_index))) {
      best_loss = candidate.loss;
      best_index = i;
    }
  }
  receipt.best_shadow_loss = best_loss;
  receipt.best_candidate = best_index;
  const bool available_improvement =
      best_index < kCandidateCount && best_loss < receipt.baseline_loss;
  bool accepted = false;
  if (available_improvement && resident->control.appraisal_intact != 0u) {
    const ShadowCandidate& best = workspace->candidates[best_index];
    if (parameter_quanta(best.genome) == parameter_quanta(parent)) {
      resident->adult = best.genome;
      ++resident->adult.revision;
      accepted = true;
    }
  }

  if (best_index < kCandidateCount) {
    const ShadowCandidate& best = workspace->candidates[best_index];
    receipt.best_parameter_index = best.parameter_index;
    receipt.best_delta = best.delta;
    receipt.best_replacement = best.replacement;
  }
  for (std::uint32_t i = 0u; i < kCandidateCount; ++i) {
    const ShadowCandidate& candidate = workspace->candidates[i];
    if (candidate.valid == 0u || (accepted && i == best_index)) continue;
    remember_failure(resident->failure_memory, parent,
                     candidate.parameter_index, candidate.replacement);
    ++receipt.rejected_proposals;
  }

  receipt.accepted = accepted ? 1u : 0u;
  receipt.lesion_blocked_improvement =
      available_improvement && resident->control.appraisal_intact == 0u ? 1u : 0u;
  receipt.after_genome = resident->adult;
  receipt.after_genome_hash = genome_hash(resident->adult);
  receipt.after_loss = chronological_predictive_loss(
      resident->adult, raw_bytes, raw_byte_count, heldout_begin);
  receipt.predictor_hash_after = prediction_hash(
      resident->adult, raw_bytes, raw_byte_count, heldout_begin);
  receipt.parameter_quanta_after = parameter_quanta(resident->adult);
  receipt.resident_bytes_after = resident->mass.total_bytes;
  receipt.strict_improvement =
      accepted && receipt.after_loss < receipt.baseline_loss ? 1u : 0u;
  receipt.rollback_exact =
      !accepted && same_genome(receipt.before_genome, receipt.after_genome) ? 1u : 0u;
  receipt.blacklist_count_after = resident->failure_memory.count;
  receipt.accounting_ok = exact_accounting(*resident) ? 1u : 0u;
  receipt.parameter_mass_ok =
      receipt.parameter_quanta_before == receipt.parameter_quanta_after &&
              receipt.parameter_quanta_after == resident->mass.initial_parameter_quanta
          ? 1u
          : 0u;
  ++resident->control.generations;
}

__global__ void copy_receipt_kernel(const ResidentAppraisal* resident,
                                    AppraisalReceipt* observer_receipt) {
  if (blockIdx.x == 0u && threadIdx.x == 0u)
    *observer_receipt = resident->receipt;
}

}  // namespace bcc32_cuda_appraisal_v1
