// GitHub #1569 -- g.device_resident_recipe_ir.
//
// A resident-owned bounded update program executes immutable authenticated
// evidence through the fixed device interpreter. Alternate physical layout
// preserves the trace and corrupted programs refuse fail-closed.

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "hardware_native/direct_adult_recipe_ir.cuh"

using namespace substrate::direct_adult_core;
using namespace substrate::direct_network;

namespace {

constexpr std::uint32_t kHalfQ16 = 1u << 15;

void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}
void cuda_require(cudaError_t status, const char* message) {
  if (status != cudaSuccess)
    throw std::runtime_error(std::string(message) + ": " +
                             cudaGetErrorString(status));
}

struct IrOutcome {
  ResidentRecipeIrProgram program{};
  ResidentRecipeIrResult result{};
  std::uint32_t executed = 0u;
};

__global__ void execute_kernel(ResidentRecipeIrProgram program,
                                      ResidentRecipeIrEvidence evidence,
                                      IrOutcome* out) {
  if (blockIdx.x != 0u || threadIdx.x != 0u) return;
  out->program = program;
  out->executed = execute_resident_recipe_ir(
                      program, evidence, &out->result)
                      ? 1u
                      : 0u;
}

__global__ void corrupt_kernel(ResidentRecipeIrProgram program,
                               ResidentRecipeIrEvidence evidence,
                               IrOutcome* out) {
  if (blockIdx.x != 0u || threadIdx.x != 0u) return;
  out->program = program;
  out->executed =
      execute_resident_recipe_ir(program, evidence, &out->result) ? 1u : 0u;
}

IrOutcome run_execute(const ResidentRecipeIrProgram& program,
                      const ResidentRecipeIrEvidence& evidence) {
  IrOutcome* outcome = nullptr;
  cuda_require(cudaMallocManaged(&outcome, sizeof(*outcome)),
               "allocate ir outcome");
  new (outcome) IrOutcome{};
  execute_kernel<<<1, 1>>>(program, evidence, outcome);
  cuda_require(cudaGetLastError(), "launch recipe ir");
  cuda_require(cudaDeviceSynchronize(), "finish recipe ir");
  const IrOutcome result = *outcome;
  cudaFree(outcome);
  return result;
}

}  // namespace

int main() {
  int devices = 0;
  if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) return 77;
  std::printf("cuda_direct_device_resident_recipe_ir_contract\n");
  bool green = true;
  auto verify = [&](bool condition, const char* message) {
    std::printf("  %s %s\n", condition ? "GREEN" : "RED", message);
    green = green && condition;
  };
  try {
    constexpr std::int32_t kDelta = kHalfQ16;
    ResidentRecipeIrProgram program{};
    require(make_resident_recipe_update_ir(1u, 1 << 16, 4 << 16,
                                           &program),
            "resident update program construction refused");
    ResidentRecipeIrEvidence evidence{};
    evidence.subject_identity = 0x1569000000000001ull;
    evidence.binding_identity = 0x1569000000000002ull;
    evidence.consequence_identity = 0x1569000000000003ull;
    evidence.occurrence_identity = 0x1569000000000004ull;
    evidence.participation_identity = 0x1569000000000005ull;
    evidence.active_work_identity = 0x1569000000000006ull;
    evidence.logical_recipe_id = 0x1569000000000007ull;
    evidence.prior_revision_identity = 0x1569000000000008ull;
    evidence.exact_credit_delta_q16 = kDelta;

    // ---- ARM 1: deterministic bounded program ------------------------------
    const IrOutcome encoded = run_execute(program, evidence);
    verify(encoded.executed != 0u && encoded.program.op_count == 5u &&
               encoded.program.program_identity != 0u,
           "resident update program executed through fixed IR");

    // ---- ARM 2: typed numeric result ---------------------------------------
    verify(encoded.result.parameter_delta_q16 == kDelta &&
               encoded.result.execution_identity != 0u,
           "typed interpreter produced the exact bounded update proposal");

    // ---- ARM 3: alternate backing layout preserves the trace ----------------
    ResidentRecipeIrProgram alternate = program;
    alternate.layout_stride = 7u;
    const IrOutcome restride = run_execute(alternate, evidence);
    verify(restride.program.program_identity ==
                   encoded.program.program_identity &&
               std::memcmp(&restride.result, &encoded.result,
                           sizeof(ResidentRecipeIrResult)) == 0,
           "alternate backing layout preserved the execution trace");

    // ---- ARM 4: corrupted programs refuse fail-closed -----------------------
    ResidentRecipeIrProgram corrupted = encoded.program;
    corrupted.instructions[corrupted.op_count - 1u].op =
        ResidentRecipeIrOp::emit_parameter_delta;  // no halt terminator
    corrupted.program_identity =
        resident_recipe_ir_identity(corrupted);
    IrOutcome* corrupt_outcome = nullptr;
    cuda_require(cudaMallocManaged(&corrupt_outcome, sizeof(*corrupt_outcome)),
                 "allocate corrupt outcome");
    new (corrupt_outcome) IrOutcome{};
    corrupt_kernel<<<1, 1>>>(corrupted, evidence, corrupt_outcome);
    cuda_require(cudaGetLastError(), "launch corrupt");
    cuda_require(cudaDeviceSynchronize(), "finish corrupt");
    const IrOutcome corrupt_result = *corrupt_outcome;
    verify(corrupt_result.executed == 0u &&
               corrupt_result.result.execution_identity == 0u,
           "corrupted program refused with no partial result");
    cudaFree(corrupt_outcome);

    // ---- ARM 5: twins -------------------------------------------------------
    const IrOutcome twin_a = run_execute(program, evidence);
    const IrOutcome twin_b = run_execute(program, evidence);
    verify(std::memcmp(&twin_a, &twin_b, sizeof(IrOutcome)) == 0,
           "twin organisms executed byte-identically");

    std::printf("%s\n",
                green ? "GREEN cuda_direct_device_resident_recipe_ir_contract"
                      : "RED   cuda_direct_device_resident_recipe_ir_contract");
    return green ? 0 : 1;
  } catch (const std::exception& error) {
    std::fprintf(stderr,
                 "cuda_direct_device_resident_recipe_ir_contract RED error=%s\n",
                 error.what());
    return 1;
  }
}
