// GitHub #1584 -- g.bootstrap_once_device_recurrence.
//
// One birth bootstrap ignites bounded resident execution: every device cycle
// executes resident Recipe IR over immutable evidence, the tail node decides
// continuation via tail launch, and post-birth the host cannot regain
// semantic or output authority.

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>

#include "hardware_native/direct_adult_bootstrap_once_recurrence.cuh"

using namespace substrate::direct_adult_core;
using namespace substrate::direct_network;

namespace {

constexpr std::uint32_t kCells = 64u;
constexpr std::uint32_t kCyclesCap = 6u;
constexpr std::int64_t kStructuralDeltaQ16 = 4096;
constexpr std::int64_t kExperienceDeltaQ16 = 2048;

void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}
void cuda_require(cudaError_t status, const char* message) {
  if (status != cudaSuccess)
    throw std::runtime_error(std::string(message) + ": " +
                             cudaGetErrorString(status));
}

// Managed-memory organism fixture; host preparation is transport only.
struct Organism {
  ResidentRecipeIrProgram* programs = nullptr;
  ResidentRecipeIrEvidence* evidence = nullptr;
  ResidentRecipeIrResult* results = nullptr;
  RecurrenceReceipt* receipt = nullptr;
  RecurrenceWorkload workload{};

  explicit Organism(std::uint64_t base) {
    cuda_require(cudaMallocManaged(&programs, sizeof(*programs) * kCells),
                 "allocate programs");
    cuda_require(cudaMallocManaged(&evidence, sizeof(*evidence) * kCells),
                 "allocate evidence");
    cuda_require(cudaMallocManaged(&results, sizeof(*results) * kCells),
                 "allocate results");
    cuda_require(cudaMallocManaged(&receipt, sizeof(*receipt)),
                 "allocate receipt");
    for (std::uint32_t i = 0u; i < kCells; ++i) {
      require(make_resident_recipe_update_ir(1u + (i % 3u), 1 << 16,
                                             4 << 16, &programs[i]),
              "birth preparation program refused");
      const std::int32_t delta = (i % 2u) == 0u
                                     ? kStructuralDeltaQ16
                                     : kExperienceDeltaQ16;
      evidence[i] = ResidentRecipeIrEvidence{
          0x1584000000000100ull + base, base * 3ull + i + 1u,
          base * 5ull + i + 1u, base * 7ull + i + 1u,
          base * 9ull + i + 1u, base * 11ull + i + 1u, base + i,
          base * 13ull + i + 1u, delta, i};
      results[i] = ResidentRecipeIrResult{};
    }
    *receipt = RecurrenceReceipt{};
    workload = RecurrenceWorkload{programs, evidence, results, receipt, nullptr,
                                  kCells};
  }
  Organism(const Organism&) = delete;
  Organism& operator=(const Organism&) = delete;
  ~Organism() {
    cudaFree(receipt);
    cudaFree(results);
    cudaFree(evidence);
    cudaFree(programs);
  }
};

}  // namespace

int main() {
  int devices = 0;
  if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) return 77;
  std::printf("cuda_direct_bootstrap_once_recurrence_contract\n");
  bool green = true;
  auto verify = [&](bool condition, const char* message) {
    std::printf("  %s %s\n", condition ? "GREEN" : "RED", message);
    green = green && condition;
  };
  try {
    // ---- A1: one ignition drives kCyclesCap fully grown cycles -------------
    Organism grown{0x15a40000ull};
    BootstrapOnceDeviceRecurrence chain;
    const std::uint64_t ledger_pre_birth = g_recurrence_host_graph_launches;
    cuda_require(chain.birth(&grown.workload, kCyclesCap), "birth recurrence");
    require(g_recurrence_host_graph_launches - ledger_pre_birth == 1ull,
            "exactly one host ignition launch at birth");
    cuda_require(chain.settle(), "settle growing recurrence");

    const RecurrenceControl* control = chain.control();
    verify(control->completed_cycles == kCyclesCap && control->quiescent == 0u &&
               control->chain_violations == 0u &&
               control->device_tail_launches == kCyclesCap - 1ull,
           "device self-tailed exactly kCycles-1 relays to the birth cap");
    verify(g_recurrence_host_graph_launches - ledger_pre_birth == 1ull,
           "zero host launch traffic after ignition");
    verify(grown.receipt->accepted_executions ==
                   static_cast<std::uint64_t>(kCells) * kCyclesCap &&
               grown.receipt->refused_executions == 0u,
           "every resident program executed once per cycle");
    bool all_exact = true;
    for (std::uint32_t i = 0u; i < kCells; ++i) {
      const std::int32_t expected = (i % 2u) == 0u
                                        ? kStructuralDeltaQ16
                                        : kExperienceDeltaQ16;
      if (grown.results[i].parameter_delta_q16 != expected ||
          grown.results[i].execution_identity == 0u)
        all_exact = false;
    }
    verify(all_exact, "every bounded update result remained exact");

    // ---- A2: quiescent dynamics self-stop on device before the cap ---------
    Organism stalled{0x15a50000ull};
    for (std::uint32_t i = 0u; i < kCells; ++i)
      stalled.programs[i].op_count = 0u;  // corrupted: refuses fail-closed
    BootstrapOnceDeviceRecurrence stalled_chain;
    const std::uint64_t ledger_pre_stall = g_recurrence_host_graph_launches;
    cuda_require(stalled_chain.birth(&stalled.workload, kCyclesCap),
                 "birth stalling recurrence");
    cuda_require(stalled_chain.settle(), "settle stalling recurrence");
    const RecurrenceControl* stall_control = stalled_chain.control();
    verify(stall_control->quiescent == 1u &&
               stall_control->completed_cycles == 1u &&
               stall_control->completed_cycles < kCyclesCap &&
               stall_control->device_tail_launches == 0u &&
               stall_control->chain_violations == 0u,
           "device declared quiescence and stopped before the birth cap");
    verify(stalled.receipt->accepted_executions == 0u &&
               stalled.receipt->refused_executions == kCells,
           "stalled execution refused without a partial result");
    verify(g_recurrence_host_graph_launches - ledger_pre_stall == 1ull,
           "stalling organism also needed only the one birth ignition");

    // ---- A3: post-birth the host cannot regain authority --------------------
    const std::uintptr_t identity_at_birth =
        chain.topology().executable_identity;
    require(chain.birth(&grown.workload, kCyclesCap * 2u) != cudaSuccess,
            "second birth was not refused");
    require(chain.birth_refusals() >= 1u, "re-birth refusal not counted");
    verify(chain.topology().instantiate_count == 1u &&
               chain.topology().executable_identity == identity_at_birth,
           "post-birth re-instantiation never happened");
    unsigned char before[sizeof(ResidentRecipeIrResult) * kCells];
    std::memcpy(before, grown.results, sizeof(before));
    ResidentRecipeIrProgram foreign{};
    verify(chain.inject_semantic_program(foreign, 3u) == cudaErrorNotSupported &&
               chain.injection_refusals() >= 1u,
           "semantic injection refused fail-closed");
    verify(std::memcmp(before, grown.results, sizeof(before)) == 0 &&
               control->max_cycles == kCyclesCap,
           "resident state and birth budget untouched by refused injection");

    // ---- A4: twins replay byte-identically; effective evidence diverges -----
    Organism twin_a{0x15a60000ull};
    Organism twin_b{0x15a60000ull};
    BootstrapOnceDeviceRecurrence twin_chain_a;
    BootstrapOnceDeviceRecurrence twin_chain_b;
    cuda_require(twin_chain_a.birth(&twin_a.workload, kCyclesCap),
                 "birth twin a");
    cuda_require(twin_chain_b.birth(&twin_b.workload, kCyclesCap),
                 "birth twin b");
    cuda_require(twin_chain_a.settle(), "settle twin a");
    cuda_require(twin_chain_b.settle(), "settle twin b");
    verify(std::memcmp(twin_a.results, twin_b.results,
                       sizeof(ResidentRecipeIrResult) * kCells) == 0 &&
               std::memcmp(twin_a.receipt, twin_b.receipt,
                           sizeof(RecurrenceReceipt)) == 0,
           "identical-state twins replayed byte-identically");

    // One externally attributable variation: program 7's credit evidence differs.
    Organism variant{0x15a60000ull};
    const std::uint32_t changed = 7u;
    variant.evidence[changed].exact_credit_delta_q16 = 8192;
    BootstrapOnceDeviceRecurrence variant_chain;
    cuda_require(variant_chain.birth(&variant.workload, kCyclesCap),
                 "birth evidence variant");
    cuda_require(variant_chain.settle(), "settle evidence variant");
    bool diverged = false;
    for (std::uint32_t i = 0u; i < kCells; ++i)
      if (std::memcmp(&variant.results[i], &twin_a.results[i],
                      sizeof(ResidentRecipeIrResult)) != 0)
        diverged = true;
    verify(diverged &&
               variant.receipt->cycle_digest != twin_a.receipt->cycle_digest,
           "effective authenticated evidence changed the execution trace");

    std::printf("%s\n",
                green ? "GREEN cuda_direct_bootstrap_once_recurrence_contract"
                      : "RED   cuda_direct_bootstrap_once_recurrence_contract");
    return green ? 0 : 1;
  } catch (const std::exception& error) {
    std::fprintf(stderr,
                 "cuda_direct_bootstrap_once_recurrence_contract RED error=%s\n",
                 error.what());
    return 1;
  }
}
