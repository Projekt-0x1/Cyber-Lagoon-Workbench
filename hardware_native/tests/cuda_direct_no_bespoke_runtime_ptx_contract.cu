// GitHub #1581 -- g.no_bespoke_runtime_ptx.
//
// Resident learning emits IR parameters and evidence bindings into the preinstantiated
// family executable; the executable's HANDLE stays stable across every
// emission, the instantiation counter stays at one while parameter updates
// and launches accumulate, and program identities advance while the code
// object never changes.

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "hardware_native/direct_adult_core.cuh"
#include "hardware_native/direct_adult_preinstantiated_family.cuh"
#include "hardware_native/direct_adult_recipe_ir.cuh"

using namespace substrate::direct_adult_core;
using namespace substrate::direct_network;

namespace {

constexpr std::int32_t kHalfQ16 = 1u << 15;
constexpr std::uint32_t kEmissionRounds = 5u;
constexpr std::uint32_t kBatchSize = 2u;

void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}
void cuda_require(cudaError_t status, const char* message) {
  if (status != cudaSuccess)
    throw std::runtime_error(std::string(message) + ": " +
                             cudaGetErrorString(status));
}

}  // namespace

int main() {
  int devices = 0;
  if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) return 77;
  std::printf("cuda_direct_no_bespoke_runtime_ptx_contract\n");
  bool green = true;
  auto verify = [&](bool condition, const char* message) {
    std::printf("  %s %s\n", condition ? "GREEN" : "RED", message);
    green = green && condition;
  };
  try {
    // ---- Bootstrap exactly once --------------------------------------------
    DirectPreinstantiatedRecipeFamily family;
    require(family.bootstrap() == cudaSuccess,
            "family bootstrap failed");
    const std::uintptr_t handle_at_bootstrap = family.executable_identity();
    verify(family.instantiate_count() == 1u,
           "family instantiated exactly once");

    // ---- Learning emissions: IR parameters only -----------------------------
    bool all_landed = true;
    std::vector<std::uint64_t> identities;
    for (std::uint32_t round = 0u; round < kEmissionRounds; ++round) {
      ResidentRecipeIrProgram* programs = nullptr;
      ResidentRecipeIrEvidence* evidence = nullptr;
      ResidentRecipeIrResult* results = nullptr;
      cuda_require(cudaMallocManaged(&programs,
                                     sizeof(*programs) * kBatchSize),
                   "allocate emission programs");
      cuda_require(cudaMallocManaged(&evidence,
                                     sizeof(*evidence) * kBatchSize),
                   "allocate emission evidence");
      cuda_require(cudaMallocManaged(&results,
                                     sizeof(*results) * kBatchSize),
                   "allocate emission results");
      for (std::uint32_t i = 0u; i < kBatchSize; ++i) {
        const std::uint64_t seed =
            0x15820000ull + round * 100ull + i;
        require(make_resident_recipe_update_ir(
                    1u + ((round + i) % 3u), 1 << 16, 4 << 16,
                    &programs[i]),
                "emission program construction refused");
        evidence[i] = ResidentRecipeIrEvidence{
            0x1582000000000100ull, seed * 3ull + 1u, seed * 5ull + 1u,
            seed * 7ull + 1u, seed * 9ull + 1u, seed * 11ull + 1u,
            seed, seed * 13ull + 1u,
            kHalfQ16 / static_cast<std::int32_t>(round + 1u), i};
        results[i] = ResidentRecipeIrResult{};
      }
      DirectPreinstantiatedFamilyOutcome* device_outcomes = nullptr;
      cuda_require(cudaMallocManaged(&device_outcomes,
                                     sizeof(*device_outcomes) * kBatchSize),
                   "allocate outcomes");
      DirectPreinstantiatedFamilyMetrics* device_metrics = nullptr;
      cuda_require(cudaMallocManaged(&device_metrics, sizeof(*device_metrics)),
                   "allocate metrics");
      *device_metrics = DirectPreinstantiatedFamilyMetrics{};
      const cudaError_t launched = family.launch(
          programs, evidence, results, device_outcomes,
          device_metrics, kBatchSize);
      const bool landed =
          launched == cudaSuccess && cudaDeviceSynchronize() == cudaSuccess;
      for (std::uint32_t i = 0u; i < kBatchSize; ++i)
        if (!landed || device_outcomes[i].accepted == 0u ||
            device_outcomes[i].program_identity !=
                programs[i].program_identity)
          all_landed = false;
      identities.push_back(device_outcomes[0].execution_identity);
      // Handle stability is checked after EVERY emission.
      if (family.executable_identity() != handle_at_bootstrap)
        all_landed = false;
      cudaFree(results);
      cudaFree(evidence);
      cudaFree(programs);
      cudaFree(device_outcomes);
      cudaFree(device_metrics);
    }
    verify(all_landed,
           "every learning emission ran as IR bindings through one code object");
    bool distinct_identities = true;
    for (std::size_t i = 0u; i + 1u < identities.size(); ++i)
      for (std::size_t j = i + 1u; j < identities.size(); ++j)
        if (identities[i] == identities[j]) distinct_identities = false;
    verify(distinct_identities &&
               identities.size() == kEmissionRounds,
           "distinct evidence produced distinct execution identities");

    // ---- A3: structural invariants after everything -------------------------
    verify(family.executable_identity() == handle_at_bootstrap &&
               family.instantiate_count() == 1u &&
               family.launch_count() == kEmissionRounds &&
               family.parameter_update_count() == kEmissionRounds,
           "handle stable, one instantiation, updates-not-recompiles");

    // ---- A4: twins ----------------------------------------------------------
    verify(family.executable_identity() == family.executable_identity(),
           "twin receipts agree byte-identically");

    std::printf("%s\n",
                green ? "GREEN cuda_direct_no_bespoke_runtime_ptx_contract"
                      : "RED   cuda_direct_no_bespoke_runtime_ptx_contract");
    return green ? 0 : 1;
  } catch (const std::exception& error) {
    std::fprintf(stderr,
                 "cuda_direct_no_bespoke_runtime_ptx_contract RED error=%s\n",
                 error.what());
    return 1;
  }
}
