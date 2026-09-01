#include <cstdint>
#include "hardware_native/direct_relational_sequence_composition.cuh"
using namespace substrate::direct_network;
__global__ void direct_relseq_device_compile_kernel(DirectRelSeqRecipe* recipes,
                                                     DirectRelSeqOccurrence* occurrences,
                                                     DirectRelSeqOutput* out) {
  if (blockIdx.x || threadIdx.x) return;
  direct_relseq_evaluate_occurrence(recipes, 1u, occurrences, 1u,
                                    occurrences[0].occurrence_identity, out);
}
int main() { return 0; }
