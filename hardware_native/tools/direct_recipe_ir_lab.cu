// Compile-once public Workbench lane for the production Resident Recipe IR.
//
// Candidate files change bounded numeric program/evidence DATA only. This tool
// has zero authority over a living Adult's current thought, evidence, credit, or
// participation. It is a physical evaluator/shadow lane using the same fixed
// interpreter and pre-instantiated CUDA Graph mechanism as production contracts.

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <stdexcept>
#include <string>
#include <unordered_map>

#include "hardware_native/direct_adult_preinstantiated_family.cuh"

using namespace substrate::direct_adult_core;
using namespace substrate::direct_network;

namespace {

std::unordered_map<std::string, std::int64_t> read_config(const char* path) {
  std::ifstream input(path);
  if (!input) throw std::runtime_error("cannot open candidate file");
  std::unordered_map<std::string, std::int64_t> values;
  std::string line;
  while (std::getline(input, line)) {
    const auto comment = line.find('#');
    if (comment != std::string::npos) line.resize(comment);
    const auto split = line.find('=');
    if (split == std::string::npos) continue;
    const std::string key = line.substr(0, split);
    const std::string raw = line.substr(split + 1);
    if (key.empty() || raw.empty()) continue;
    char* end = nullptr;
    const long long value = std::strtoll(raw.c_str(), &end, 0);
    if (end == raw.c_str() || *end != '\0')
      throw std::runtime_error("candidate value is not an integer: " + key);
    values[key] = static_cast<std::int64_t>(value);
  }
  return values;
}

std::int64_t value(const std::unordered_map<std::string, std::int64_t>& values,
                   const char* key, std::int64_t fallback) {
  const auto found = values.find(key);
  return found == values.end() ? fallback : found->second;
}

void cuda_require(cudaError_t status, const char* operation) {
  if (status != cudaSuccess)
    throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
}

std::uint64_t derived_identity(std::uint64_t domain, std::uint64_t seed,
                               std::uint64_t ordinal) {
  std::uint64_t identity = exact_history_fold_word(domain, seed);
  identity = exact_history_fold_word(identity, ordinal);
  return identity == 0u ? ordinal + 1u : identity;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 2) {
    std::fprintf(stderr, "usage: %s CANDIDATE.ir\n", argv[0]);
    return 2;
  }
  int devices = 0;
  if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
    std::printf("DIRECT_RECIPE_IR_LAB status=SKIP reason=no_cuda_device\n");
    return 77;
  }

  try {
    const auto config = read_config(argv[1]);
    const auto layout_stride = static_cast<std::uint32_t>(value(config, "layout_stride", 1));
    const auto scale_q16 = static_cast<std::int32_t>(value(config, "scale_q16", 1 << 16));
    const auto clamp_q16 = static_cast<std::int32_t>(value(config, "clamp_q16", 4 << 16));
    const auto exact_credit = static_cast<std::int32_t>(
        value(config, "exact_credit_delta_q16", 1 << 15));
    const auto seed = static_cast<std::uint64_t>(value(config, "evidence_seed", 1));
    const auto recipe_cell = static_cast<std::uint32_t>(value(config, "recipe_cell", 0));

    ResidentRecipeIrProgram* program = nullptr;
    ResidentRecipeIrEvidence* evidence = nullptr;
    ResidentRecipeIrResult* result = nullptr;
    DirectPreinstantiatedFamilyOutcome* outcome = nullptr;
    DirectPreinstantiatedFamilyMetrics* metrics = nullptr;
    cuda_require(cudaMallocManaged(&program, sizeof(*program)), "allocate program");
    cuda_require(cudaMallocManaged(&evidence, sizeof(*evidence)), "allocate evidence");
    cuda_require(cudaMallocManaged(&result, sizeof(*result)), "allocate result");
    cuda_require(cudaMallocManaged(&outcome, sizeof(*outcome)), "allocate outcome");
    cuda_require(cudaMallocManaged(&metrics, sizeof(*metrics)), "allocate metrics");

    if (!make_resident_recipe_update_ir(layout_stride, scale_q16, clamp_q16, program))
      throw std::runtime_error("candidate program refused by production IR validator");

    *evidence = ResidentRecipeIrEvidence{
        derived_identity(0x7075626c69637375ull, seed, 1u),
        derived_identity(0x7075626c69636269ull, seed, 2u),
        derived_identity(0x7075626c6963636full, seed, 3u),
        derived_identity(0x7075626c69636f63ull, seed, 4u),
        derived_identity(0x7075626c69637061ull, seed, 5u),
        derived_identity(0x7075626c6963776full, seed, 6u),
        derived_identity(0x7075626c69637265ull, seed, 7u),
        derived_identity(0x7075626c69637276ull, seed, 8u),
        exact_credit,
        recipe_cell};
    *result = {};
    *outcome = {};
    *metrics = {};

    DirectPreinstantiatedRecipeFamily family;
    cuda_require(family.bootstrap(), "bootstrap fixed evaluator");
    const auto executable = family.executable_identity();
    cuda_require(family.launch(program, evidence, result, outcome, metrics, 1u),
                 "launch candidate data");
    cuda_require(cudaDeviceSynchronize(), "settle candidate data");

    const bool stable = family.instantiate_count() == 1u &&
                        family.executable_identity() == executable &&
                        family.parameter_update_count() == 1u &&
                        family.launch_count() == 1u;
    const bool accepted = outcome->accepted != 0u;
    std::printf(
        "DIRECT_RECIPE_IR_LAB status=%s claim_scope=experiment_only "
        "semantic_authority=0 current_thought_authority=0 host_semantic_dispatches=%llu "
        "instantiate_count=%u executable_stable=%u program_identity=%016llx "
        "execution_identity=%016llx parameter_delta_q16=%d work_units=%u\n",
        accepted && stable ? "EXPERIMENT_PASS" : "RED",
        static_cast<unsigned long long>(metrics->host_semantic_dispatches),
        family.instantiate_count(), stable ? 1u : 0u,
        static_cast<unsigned long long>(program->program_identity),
        static_cast<unsigned long long>(result->execution_identity),
        result->parameter_delta_q16, result->work_units);

    cudaFree(metrics);
    cudaFree(outcome);
    cudaFree(result);
    cudaFree(evidence);
    cudaFree(program);
    return accepted && stable ? 0 : 1;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "DIRECT_RECIPE_IR_LAB status=RED error=%s\n", error.what());
    return 1;
  }
}
