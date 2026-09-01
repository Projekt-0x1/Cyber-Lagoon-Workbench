// GitHub #1592, i.foundry_equivalence_efficiency_authority.
// Engineering equivalence may authorize a guarded backing-only replacement;
// it cannot authorize meaning, evidence, participation or current thought.

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <new>
#include <stdexcept>
#include <string>

#include "hardware_native/direct_canonical_evaluator_device.cuh"
#include "hardware_native/direct_foundry_equivalence_commit.cuh"

using namespace substrate::direct_adult_core;
using namespace substrate::direct_network;

namespace {

using Pool = DirectFoundryRecipeCandidatePool<2u>;
constexpr std::uint32_t kCases = 4u;
constexpr std::uint32_t kBenchmarkIterations = 500000u;

template <typename T>
concept HasWorldTruth = requires(T value) { value.world_truth; };
template <typename T>
concept HasSemanticLabel = requires(T value) { value.semantic_label; };
template <typename T>
concept HasCurrentThought = requires(T value) { value.current_thought; };
template <typename T>
concept HasActivate = requires(T value) { value.activate(); };

static_assert(!HasWorldTruth<DirectFoundryEquivalenceReceiptV1> &&
              !HasSemanticLabel<DirectFoundryEquivalenceReceiptV1> &&
              !HasCurrentThought<DirectFoundryEquivalentBackingCommitReceiptV1> &&
              !HasActivate<DirectFoundryEquivalentBackingTransactionV1>);

void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

void cuda_require(cudaError_t status, const char* message) {
  if (status != cudaSuccess)
    throw std::runtime_error(std::string(message) + ": " +
                             cudaGetErrorString(status));
}

DirectSha256Address address(const void* bytes, std::size_t size) {
  DirectSha256Address out{};
  require(direct_sha256_content_address(bytes, size, &out),
          "content address failed");
  return out;
}

DirectSha256Address tag(const char* domain, std::uint64_t value = 0u) {
  struct Tagged {
    char name[64];
    std::uint64_t value;
  } tagged{};
  std::snprintf(tagged.name, sizeof(tagged.name), "%s", domain);
  tagged.value = value;
  return address(&tagged, sizeof(tagged));
}

__host__ __device__ ResidentRecipeIrEvidence source_evidence(
    std::uint64_t seed) {
  return ResidentRecipeIrEvidence{
      0x1592000000000100ull, 0x1592000000000200ull,
      0x1592000000001000ull + seed, 0x1592000000002000ull + seed,
      0x1592000000003000ull + seed, 0x1592000000004000ull + seed,
      0x1592000000000001ull, 0x1592000000000002ull, 4096, 0u};
}

struct Programs {
  ResidentRecipeIrProgram source{};
  ResidentRecipeIrProgram candidate{};
};

Programs programs() {
  Programs out{};
  require(make_resident_recipe_update_ir(1u, 1 << 16, 4 << 16,
                                         &out.candidate),
          "candidate construction failed");
  out.source = out.candidate;
  out.source.layout_stride = 8u;
  require(resident_recipe_ir_intact(out.source) &&
              resident_recipe_ir_intact(out.candidate) &&
              out.source.program_identity == out.candidate.program_identity &&
              std::memcmp(&out.source, &out.candidate,
                          sizeof(out.source)) != 0,
          "distinct equivalent programs were not intact");
  return out;
}

DirectFoundryCandidateRecordV1 candidate_record(
    std::uint64_t sequence, const ResidentRecipeIrProgram& program,
    const DirectSha256Address& parent, const DirectSha256Address& relation,
    const DirectSha256Address& domain, const DirectSha256Address& guard,
    const DirectSha256Address& ports, const DirectSha256Address& evaluator,
    const DirectSha256Address& species) {
  DirectFoundryCandidateRecordV1 record{};
  record.sequence = sequence;
  record.generation = sequence == 0u ? 0u : 1u;
  record.derivation_rank = 3u;
  record.formal_port_count = 2u;
  record.kind = DirectFoundryCandidateKindV1::guarded_solver_lowering;
  record.proof_class = sequence == 0u
                           ? DirectFoundryProofClassV1::unverified_proposal
                           : DirectFoundryProofClassV1::logical_equivalence_claim;
  record.candidate_body = resident_recipe_ir_program_address(program);
  record.parent_candidate = parent;
  record.construction_ancestry = tag("equivalence-construction", sequence);
  record.authorship = tag("equivalence-author", sequence);
  record.formal_ports = ports;
  record.domain = domain;
  record.guard = guard;
  record.proof_claim = relation;
  record.compatible_evaluator = evaluator;
  record.compatible_species = species;
  record.resource_receipt = tag("resource-measurement-source", sequence);
  record.contextual_failures = tag("known-failures", sequence);
  record.falsifiers = tag("equivalence-falsifiers", sequence);
  return record;
}

struct CandidateFixture {
  Pool pool{};
  DirectFoundryCandidateEntryV1 source{};
  DirectFoundryCandidateEntryV1 candidate{};
  DirectSha256Address relation{};
  DirectSha256Address domain{};
  DirectSha256Address guard{};
  DirectSha256Address ports{};
  DirectSha256Address evaluator{};
  DirectSha256Address species{};
};

CandidateFixture candidates(const Programs& program) {
  CandidateFixture out{};
  out.relation = tag("formal-relation-contract");
  out.domain = tag("complete-finite-domain");
  out.guard = tag("formal-numeric-guard");
  out.ports = tag("formal-ports");
  out.evaluator = tag("canonical-evaluator");
  out.species = tag("compatible-species");
  auto source = candidate_record(0u, program.source, {}, out.relation,
                                 out.domain, out.guard, out.ports,
                                 out.evaluator, out.species);
  const DirectSha256Address source_address = Pool::candidate_address(source);
  auto candidate = candidate_record(1u, program.candidate, source_address,
                                    out.relation, out.domain, out.guard,
                                    out.ports, out.evaluator, out.species);
  require(out.pool.append(source) && out.pool.append(candidate) &&
              out.pool.read(0u, &out.source) &&
              out.pool.read(1u, &out.candidate) && out.pool.verify(),
          "candidate pool construction failed");
  return out;
}

struct TransitionTrace {
  ResidentRecipeIrResult result{};
  std::uint64_t canonical_context = 0u;
  std::uint32_t accepted = 0u;
  std::uint32_t reserved = 0u;
};

struct TraceBatch {
  ResidentRecipeIrEvidence evidence[kCases]{};
  TransitionTrace source[kCases]{};
  TransitionTrace candidate[kCases]{};
  TransitionTrace replay[kCases]{};
};

__global__ void trace_equivalent_programs(
    const ResidentRecipeIrProgram source,
    const ResidentRecipeIrProgram candidate, TraceBatch* batch,
    bool replay_only) {
  const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= kCases) return;
  TransitionTrace source_trace{};
  source_trace.canonical_context = batch->evidence[i].consequence_identity;
  source_trace.accepted =
      execute_resident_recipe_ir(source, batch->evidence[i],
                                 i == kCases - 1u ? nullptr
                                                  : &source_trace.result)
          ? 1u
          : 0u;
  TransitionTrace candidate_trace{};
  candidate_trace.canonical_context = source_trace.canonical_context;
  candidate_trace.accepted =
      execute_resident_recipe_ir(candidate, batch->evidence[i],
                                 i == kCases - 1u ? nullptr
                                                  : &candidate_trace.result)
          ? 1u
          : 0u;
  if (replay_only) {
    batch->replay[i] = candidate_trace;
  } else {
    batch->source[i] = source_trace;
    batch->candidate[i] = candidate_trace;
  }
}

__global__ void benchmark_program(const ResidentRecipeIrProgram program,
                                  std::uint32_t iterations,
                                  std::uint64_t* checksum) {
  if (blockIdx.x != 0u || threadIdx.x != 0u) return;
  std::uint64_t value = 0u;
  const ResidentRecipeIrEvidence evidence = source_evidence(0u);
  for (std::uint32_t i = 0u; i < iterations; ++i) {
    ResidentRecipeIrResult result{};
    bool accepted = false;
    for (std::uint32_t repeat = 0u; repeat < program.layout_stride; ++repeat)
      accepted = direct_foundry_interpret_recipe_ir(program, evidence, &result);
    if (accepted)
      value += static_cast<std::uint32_t>(result.parameter_delta_q16) +
               result.work_units + (i & 1u);
  }
  *checksum = value;
}

std::uint64_t timed_program(const ResidentRecipeIrProgram& program,
                            std::uint64_t* checksum) {
  std::uint64_t* device_checksum = nullptr;
  cuda_require(cudaMalloc(&device_checksum, sizeof(*device_checksum)),
               "allocate benchmark checksum");
  cudaEvent_t start = nullptr, stop = nullptr;
  cuda_require(cudaEventCreate(&start), "create benchmark start");
  cuda_require(cudaEventCreate(&stop), "create benchmark stop");
  benchmark_program<<<1u, 1u>>>(program, 1024u, device_checksum);
  cuda_require(cudaDeviceSynchronize(), "warm benchmark");
  cuda_require(cudaEventRecord(start), "record benchmark start");
  benchmark_program<<<1u, 1u>>>(program, kBenchmarkIterations,
                               device_checksum);
  cuda_require(cudaEventRecord(stop), "record benchmark stop");
  cuda_require(cudaEventSynchronize(stop), "finish benchmark");
  float elapsed_ms = 0.0f;
  cuda_require(cudaEventElapsedTime(&elapsed_ms, start, stop),
               "read benchmark time");
  cuda_require(cudaMemcpy(checksum, device_checksum, sizeof(*checksum),
                          cudaMemcpyDeviceToHost),
               "read benchmark checksum");
  cudaEventDestroy(stop);
  cudaEventDestroy(start);
  cudaFree(device_checksum);
  const auto ns = static_cast<std::uint64_t>(elapsed_ms * 1000000.0f);
  return ns == 0u ? 1u : ns;
}

DirectFoundryResourceReceiptV1 resource_receipt(
    const CandidateFixture& fixture, const Programs& program) {
  std::array<std::uint64_t, 3u> source_ns{};
  std::array<std::uint64_t, 3u> candidate_ns{};
  std::uint64_t source_checksum = 0u, candidate_checksum = 0u;
  for (std::size_t i = 0u; i < source_ns.size(); ++i) {
    source_ns[i] = timed_program(program.source, &source_checksum);
    candidate_ns[i] = timed_program(program.candidate, &candidate_checksum);
  }
  require(source_checksum == candidate_checksum,
          "equivalent interpreter benchmarks produced different results");
  std::sort(source_ns.begin(), source_ns.end());
  std::sort(candidate_ns.begin(), candidate_ns.end());
  require(candidate_ns[1] < source_ns[1],
          "candidate did not reduce measured interpreter latency");
  DirectFoundryResourceReceiptV1 receipt{};
  receipt.sample_count = static_cast<std::uint32_t>(source_ns.size());
  receipt.source_latency_ns = source_ns[1];
  receipt.candidate_latency_ns = candidate_ns[1];
  receipt.source_active_work = program.source.op_count;
  receipt.candidate_active_work = program.candidate.op_count;
  receipt.source_memory_bytes = sizeof(ResidentRecipeIrProgram);
  receipt.candidate_memory_bytes = sizeof(ResidentRecipeIrProgram);
  receipt.source_precision_error_q32 = 0u;
  receipt.candidate_precision_error_q32 = 0u;
  receipt.source_failure_cost = 1u;
  receipt.candidate_failure_cost = 1u;
  receipt.source_candidate = fixture.source.candidate_address;
  receipt.candidate = fixture.candidate.candidate_address;
  receipt.task = tag("bounded-equivalence-task");
  receipt.guard = fixture.guard;
  receipt.body_regime = tag("consumer-body-regime");
  receipt.evaluator = fixture.evaluator;
  receipt.resource_regime = tag("consumer-resource-regime");
  receipt.benchmark = tag("gpu-interpreter-benchmark", source_checksum);
  receipt.reproducibility = tag("benchmark-replay", candidate_checksum);
  require(direct_foundry_resource_receipt_valid(receipt),
          "measured resource dominance was invalid");
  return receipt;
}

DirectFoundryEquivalenceReceiptV1 equivalence_receipt(
    const CandidateFixture& fixture,
    const DirectFoundryResourceReceiptV1& resources,
    const TraceBatch& traces) {
  require(std::memcmp(traces.source, traces.candidate,
                      sizeof(traces.source)) == 0,
          "complete-domain state transitions diverged");
  require(std::memcmp(traces.candidate, traces.replay,
                      sizeof(traces.candidate)) == 0,
          "deterministic candidate replay diverged");
  DirectFoundryEquivalenceReceiptV1 receipt{};
  receipt.domain_case_count = kCases;
  receipt.evaluated_case_count = kCases;
  receipt.exact_match_count = kCases;
  receipt.guard_min_q16 = -(2 << 16);
  receipt.guard_max_q16 = 2 << 16;
  receipt.source_candidate = fixture.source.candidate_address;
  receipt.candidate = fixture.candidate.candidate_address;
  receipt.relation_contract = fixture.relation;
  receipt.complete_domain = fixture.domain;
  receipt.guard = fixture.guard;
  receipt.formal_ports = fixture.ports;
  receipt.source_output_trace = address(traces.source, sizeof(traces.source));
  receipt.candidate_output_trace =
      address(traces.candidate, sizeof(traces.candidate));
  receipt.state_transition = receipt.source_output_trace;
  receipt.chronology = address(traces.evidence, sizeof(traces.evidence));
  receipt.provenance = tag("exact-provenance-semantics");
  receipt.refusal_semantics =
      tag("null-result-refusal", traces.source[kCases - 1u].accepted);
  receipt.transaction_semantics = tag("failure-atomic-transaction");
  receipt.deterministic_replay = address(traces.replay, sizeof(traces.replay));
  receipt.resource_receipt =
      direct_foundry_resource_receipt_address(resources);
  require(direct_foundry_equivalence_receipt_valid(
              receipt, fixture.source, fixture.candidate, resources),
          "exact equivalence receipt was invalid");
  return receipt;
}

ResidentRecipeIrBackingOwnerV1 owner(
    const CandidateFixture& fixture, const Programs& program,
    const DirectFoundryEquivalenceReceiptV1& proof) {
  ResidentRecipeIrBackingOwnerV1 out{};
  out.logical_recipe_id = 0x1592000000000001ull;
  out.revision_identity = 0x1592000000000002ull;
  out.program = program.source;
  out.fallback_program = program.source;
  out.backing_candidate = fixture.source.candidate_address;
  out.fallback_candidate = fixture.source.candidate_address;
  out.guard = fixture.guard;
  out.occurrence_binding_state = tag("two-current-occurrence-bindings");
  out.causal_chronology = tag("adult-causal-chronology");
  out.evidence_state = tag("adult-evidence-state");
  out.participation_state = tag("adult-participation-state");
  out.credit_state = tag("adult-credit-state");
  out.checkpoint_continuation = tag("adult-checkpoint-continuation");
  out.backing_epoch = 9u;
  out.guard_min_q16 = proof.guard_min_q16;
  out.guard_max_q16 = proof.guard_max_q16;
  out.current_occurrence_count = 2u;
  require(resident_recipe_ir_backing_owner_intact(out),
          "existing Adult backing owner was invalid");
  return out;
}

struct CommitDeviceFixture {
  ResidentRecipeIrBackingOwnerV1 owner{};
  DirectFoundryCandidateEntryV1 source{};
  DirectFoundryCandidateEntryV1 candidate{};
  DirectFoundryEquivalenceReceiptV1 proof{};
  DirectFoundryResourceReceiptV1 resources{};
  DirectFoundryEquivalentBackingTransactionV1 transaction{};
  DirectFoundryEquivalentBackingCommitReceiptV1 receipt{};
  DirectFoundryEquivalentCommitRefusalV1 refusal =
      DirectFoundryEquivalentCommitRefusalV1::null_argument;
};

__global__ void commit_backing(CommitDeviceFixture* fixture) {
  if (blockIdx.x != 0u || threadIdx.x != 0u) return;
  fixture->refusal = commit_direct_foundry_equivalent_backing(
      &fixture->owner, fixture->source, fixture->candidate, fixture->proof,
      fixture->resources, fixture->transaction, &fixture->receipt);
}

void launch_commit(CommitDeviceFixture* fixture) {
  commit_backing<<<1u, 1u>>>(fixture);
  cuda_require(cudaGetLastError(), "launch equivalence commit");
  cuda_require(cudaDeviceSynchronize(), "finish equivalence commit");
}

}  // namespace

int main() {
  int devices = 0;
  if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
    std::printf("SKIP cuda_direct_foundry_equivalence_efficiency_authority_contract no CUDA device\n");
    return 77;
  }
  try {
    const Programs program = programs();
    const CandidateFixture fixture = candidates(program);

    TraceBatch* traces = nullptr;
    cuda_require(cudaMallocManaged(&traces, sizeof(*traces)),
                 "allocate equivalence traces");
    new (traces) TraceBatch{};
    for (std::uint32_t i = 0u; i < kCases; ++i)
      traces->evidence[i] = source_evidence(i + 1u);
    trace_equivalent_programs<<<1u, 32u>>>(program.source, program.candidate,
                                           traces, false);
    cuda_require(cudaDeviceSynchronize(), "run complete-domain comparison");
    trace_equivalent_programs<<<1u, 32u>>>(program.source, program.candidate,
                                           traces, true);
    cuda_require(cudaDeviceSynchronize(), "run deterministic replay");

    const DirectFoundryResourceReceiptV1 resources =
        resource_receipt(fixture, program);
    const DirectFoundryEquivalenceReceiptV1 proof =
        equivalence_receipt(fixture, resources, *traces);

    auto bounded = proof;
    bounded.mode = DirectFoundryEquivalenceModeV1::bounded_tolerance;
    bounded.tolerance_q32 = 2u;
    bounded.maximum_error_q32 = 1u;
    bounded.exact_match_count = kCases - 1u;
    bounded.candidate_output_trace = tag("bounded-output-trace");
    require(direct_foundry_equivalence_receipt_valid(
                bounded, fixture.source, fixture.candidate, resources),
            "declared bounded-tolerance proof was rejected");
    bounded.maximum_error_q32 = 3u;
    require(!direct_foundry_equivalence_receipt_valid(
                bounded, fixture.source, fixture.candidate, resources),
            "out-of-tolerance proof was accepted");

    CommitDeviceFixture* device = nullptr;
    cuda_require(cudaMallocManaged(&device, sizeof(*device)),
                 "allocate commit fixture");
    new (device) CommitDeviceFixture{};
    device->owner = owner(fixture, program, proof);
    device->source = fixture.source;
    device->candidate = fixture.candidate;
    device->proof = proof;
    device->resources = resources;
    device->transaction.expected_logical_recipe_id =
        device->owner.logical_recipe_id;
    device->transaction.expected_revision_identity =
        device->owner.revision_identity;
    device->transaction.expected_backing_epoch = device->owner.backing_epoch;
    device->transaction.expected_preservation_root =
        resident_recipe_ir_preservation_root(device->owner);
    device->transaction.proof_receipt =
        direct_foundry_equivalence_receipt_address(proof);
    device->transaction.candidate_program = program.candidate;

    const CommitDeviceFixture pristine = *device;
    device->transaction.quiescence.ingress_pending = 1u;
    launch_commit(device);
    require(device->refusal ==
                DirectFoundryEquivalentCommitRefusalV1::active_boundary &&
                std::memcmp(&device->owner, &pristine.owner,
                            sizeof(device->owner)) == 0 &&
                std::memcmp(&device->receipt, &pristine.receipt,
                            sizeof(device->receipt)) == 0,
            "non-quiescent replacement was not failure-atomic");

    *device = pristine;
    device->proof.maximum_error_q32 = 1u;
    launch_commit(device);
    require(device->refusal ==
                DirectFoundryEquivalentCommitRefusalV1::stale_transaction &&
                std::memcmp(&device->owner, &pristine.owner,
                            sizeof(device->owner)) == 0,
            "forged proof changed the existing Adult");

    *device = pristine;
    launch_commit(device);
    require(device->refusal == DirectFoundryEquivalentCommitRefusalV1::none,
            "proved quiescent replacement refused");
    const DirectSha256Address preserved =
        resident_recipe_ir_preservation_root(pristine.owner);
    require(device->owner.logical_recipe_id == pristine.owner.logical_recipe_id &&
                device->owner.revision_identity ==
                    pristine.owner.revision_identity &&
                device->owner.current_occurrence_count == 2u &&
                resident_recipe_ir_preservation_root(device->owner) == preserved &&
                device->owner.program.program_identity ==
                    program.candidate.program_identity &&
                device->owner.fallback_program.program_identity ==
                    program.source.program_identity &&
                device->owner.backing_candidate ==
                    fixture.candidate.candidate_address &&
                device->owner.fallback_candidate ==
                    fixture.source.candidate_address &&
                device->owner.backing_epoch == pristine.owner.backing_epoch + 1u,
            "backing commit changed logical/causal state or lost fallback");
    require(select_resident_recipe_ir_backing(device->owner, 0) ==
                &device->owner.program &&
                select_resident_recipe_ir_backing(device->owner, 3 << 16) ==
                    &device->owner.fallback_program,
            "formal guard did not select specialization/fallback deterministically");
    require(device->receipt.semantic_authority == 0u &&
                device->receipt.experiential_authority == 0u &&
                device->receipt.participation_authority == 0u &&
                device->receipt.credit_authority == 0u &&
                device->receipt.current_network_authority == 0u &&
                device->receipt.current_thought_authority == 0u,
            "engineering receipt acquired epistemic or Adult authority");

    const ResidentRecipeIrBackingOwnerV1 checkpoint = device->owner;
    ResidentRecipeIrResult continued{};
    ResidentRecipeIrResult fallback{};
    require(execute_resident_recipe_ir(
                *select_resident_recipe_ir_backing(checkpoint, 0),
                traces->evidence[0], &continued) &&
                execute_resident_recipe_ir(
                    *select_resident_recipe_ir_backing(checkpoint, 3 << 16),
                    traces->evidence[0], &fallback) &&
                std::memcmp(&continued, &fallback, sizeof(continued)) == 0,
            "checkpointed candidate/fallback continuation diverged");

    CommitDeviceFixture replay = pristine;
    CommitDeviceFixture* replay_device = nullptr;
    cuda_require(cudaMallocManaged(&replay_device, sizeof(*replay_device)),
                 "allocate commit replay");
    *replay_device = replay;
    launch_commit(replay_device);
    require(replay_device->refusal ==
                DirectFoundryEquivalentCommitRefusalV1::none &&
                std::memcmp(&replay_device->owner, &device->owner,
                            sizeof(device->owner)) == 0 &&
                std::memcmp(&replay_device->receipt, &device->receipt,
                            sizeof(device->receipt)) == 0,
            "same complete transaction state did not replay exactly");

    std::printf(
        "GREEN cuda_direct_foundry_equivalence_efficiency_authority_contract "
        "domain=%u exact=%u active_work=%u->%u latency_ns=%llu->%llu "
        "occurrences_preserved=2 fallback=1 semantic_authority=0 "
        "adult_thought_authority=0\n",
        kCases, kCases, program.source.op_count, program.candidate.op_count,
        static_cast<unsigned long long>(resources.source_latency_ns),
        static_cast<unsigned long long>(resources.candidate_latency_ns));
    cudaFree(replay_device);
    cudaFree(device);
    cudaFree(traces);
    return 0;
  } catch (const std::exception& error) {
    std::fprintf(stderr,
                 "RED cuda_direct_foundry_equivalence_efficiency_authority_contract: %s\n",
                 error.what());
    return 1;
  }
}
