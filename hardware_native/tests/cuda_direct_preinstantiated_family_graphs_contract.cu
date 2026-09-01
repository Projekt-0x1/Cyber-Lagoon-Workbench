// GitHub #1570 -- g.preinstantiated_family_graphs.
// One generic Recipe IR CUDA graph is instantiated once.  Distinct resident
// programs and immutable evidence subsequently pass through the same
// executable by pointer/count updates only and land exactly like direct
// invocation.

#include <cuda_runtime.h>

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>

#include "hardware_native/direct_adult_preinstantiated_family.cuh"

using namespace substrate::direct_adult_core;
using namespace substrate::direct_network;

namespace {

constexpr std::uint32_t kJobs = 4u;
constexpr std::int32_t kDeltas[kJobs] = {
    1 << 12, -(1 << 13), 1 << 14, 1 << 15};

void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

void cuda_require(cudaError_t status, const char* operation) {
  if (status != cudaSuccess)
    throw std::runtime_error(std::string(operation) + ": " +
                             cudaGetErrorString(status));
}

struct Workload {
  ResidentRecipeIrProgram* programs = nullptr;
  ResidentRecipeIrEvidence* evidence = nullptr;
  ResidentRecipeIrResult* results = nullptr;
  DirectPreinstantiatedFamilyOutcome* outcomes = nullptr;
  DirectPreinstantiatedFamilyMetrics* metrics = nullptr;

  Workload() {
    cuda_require(cudaMallocManaged(&programs, sizeof(*programs) * kJobs),
                 "allocate family programs");
    cuda_require(cudaMallocManaged(&evidence, sizeof(*evidence) * kJobs),
                 "allocate family evidence");
    cuda_require(cudaMallocManaged(&results, sizeof(*results) * kJobs),
                 "allocate family results");
    cuda_require(cudaMallocManaged(&outcomes, sizeof(*outcomes) * kJobs),
                 "allocate family outcomes");
    cuda_require(cudaMallocManaged(&metrics, sizeof(*metrics)),
                 "allocate family metrics");
    for (std::uint32_t i = 0u; i < kJobs; ++i) {
      require(make_resident_recipe_update_ir(
                  i + 1u, static_cast<std::int32_t>((i + 1u) << 16u),
                  4 << 16,
                                             &programs[i]),
              "prepare resident update program");
      evidence[i] = ResidentRecipeIrEvidence{
          0x1570000000000100ull, 0x1570000000000200ull + i,
          0x1570000000000300ull + i, 0x1570000000000400ull + i,
          0x1570000000000500ull + i, 0x1570000000000600ull + i,
          0x1570000000000700ull + i, 0x1570000000000800ull + i,
          kDeltas[i], i};
      results[i] = {};
      outcomes[i] = {};
    }
    *metrics = {};
  }
  Workload(const Workload&) = delete;
  Workload& operator=(const Workload&) = delete;
  ~Workload() {
    cudaFree(metrics);
    cudaFree(outcomes);
    cudaFree(results);
    cudaFree(evidence);
    cudaFree(programs);
  }
};

struct Snapshot {
  std::array<ResidentRecipeIrResult, kJobs> results{};
  std::array<DirectPreinstantiatedFamilyOutcome, kJobs> outcomes{};
  DirectPreinstantiatedFamilyMetrics metrics{};
};

Snapshot snapshot(const Workload& workload) {
  Snapshot result{};
  for (std::uint32_t i = 0u; i < kJobs; ++i) {
    result.results[i] = workload.results[i];
    result.outcomes[i] = workload.outcomes[i];
  }
  result.metrics = *workload.metrics;
  return result;
}

struct PairedRun {
  Snapshot graph{};
  Snapshot direct{};
};

PairedRun run_pair(DirectPreinstantiatedRecipeFamily& family,
                   std::uint32_t offset, std::uint32_t count) {
  require(offset + count <= kJobs && count != 0u,
          "invalid mechanical workload range");
  Workload graph;
  Workload direct;
  cuda_require(family.launch(graph.programs + offset,
                             graph.evidence + offset,
                             graph.results + offset,
                             graph.outcomes + offset, graph.metrics, count),
               "launch preinstantiated family");
  direct_preinstantiated_recipe_family_kernel<0u><<<1, 128>>>(
      direct.programs + offset, direct.evidence + offset,
      direct.results + offset, direct.outcomes + offset, direct.metrics, count);
  cuda_require(cudaGetLastError(), "launch direct family reference");
  cuda_require(cudaDeviceSynchronize(), "finish paired family executions");
  return {snapshot(graph), snapshot(direct)};
}

void require_parity(const PairedRun& run, const char* message) {
  require(std::memcmp(&run.graph, &run.direct, sizeof(run.graph)) == 0,
          message);
}

}  // namespace

int main() {
  int devices = 0;
  if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
    std::printf("SKIP cuda_direct_preinstantiated_family_graphs_contract no CUDA device\n");
    return 77;
  }

  try {
    DirectPreinstantiatedRecipeFamily family;
    cuda_require(family.bootstrap(), "bootstrap family graph");
    const std::uintptr_t executable = family.executable_identity();
    require(executable != 0u && family.instantiate_count() == 1u,
            "bootstrap did not instantiate exactly one executable");
    cuda_require(family.bootstrap(), "repeat bootstrap");
    require(family.instantiate_count() == 1u &&
                family.executable_identity() == executable,
            "repeat bootstrap instantiated or replaced the executable");

    const PairedRun one = run_pair(family, 0u, 1u);
    const PairedRun four = run_pair(family, 0u, 4u);
    const PairedRun shifted = run_pair(family, 2u, 2u);
    require_parity(one, "single-program graph/direct parity failed");
    require_parity(four, "diverse-program graph/direct parity failed");
    require_parity(shifted, "shifted-cell graph/direct parity failed");
    require(family.instantiate_count() == 1u &&
                family.parameter_update_count() == 3u &&
                family.launch_count() == 3u &&
                family.executable_identity() == executable,
            "diverse compositions recompiled or replaced the graph");

    for (std::uint32_t i = 0u; i < kJobs; ++i) {
      require(four.graph.outcomes[i].accepted != 0u,
              "resident IR composition refused");
      for (std::uint32_t j = i + 1u; j < kJobs; ++j)
        require(four.graph.outcomes[i].program_identity !=
                    four.graph.outcomes[j].program_identity,
                "distinct resident compositions collapsed to one program");
    }
    require(four.graph.metrics.resident_ir_executions == kJobs &&
                four.graph.metrics.resident_ir_refusals == 0u &&
                four.graph.metrics.host_semantic_dispatches == 0u,
            "semantic execution escaped resident IR authority");

    DirectPreinstantiatedRecipeFamily twin;
    cuda_require(twin.bootstrap(), "bootstrap determinism twin");
    const PairedRun twin_run = run_pair(twin, 0u, 4u);
    require_parity(twin_run, "twin direct parity failed");
    require(std::memcmp(&twin_run.graph, &four.graph,
                        sizeof(four.graph)) == 0 &&
                twin.instantiate_count() == 1u,
            "same-state determinism twin diverged");

    std::printf(
        "PREINSTANTIATED_FAMILY_GRAPHS_RECEIPT status=GREEN "
        "instantiate_count=%u launches=%u parameter_updates=%u "
        "distinct_programs=%u resident_executions=%llu "
        "host_semantic_dispatches=%llu executable_stable=PASS "
        "direct_parity=BYTE_IDENTICAL twins=BYTE_IDENTICAL\n",
        family.instantiate_count(), family.launch_count(),
        family.parameter_update_count(), kJobs,
        static_cast<unsigned long long>(
            four.graph.metrics.resident_ir_executions),
        static_cast<unsigned long long>(
            four.graph.metrics.host_semantic_dispatches));
    return 0;
  } catch (const std::exception& error) {
    std::printf("PREINSTANTIATED_FAMILY_GRAPHS_RECEIPT status=RED error=%s\n",
                error.what());
    return 1;
  }
}
