#ifndef HARDWARE_NATIVE_DIRECT_ADULT_PHENOTYPE_ANATOMY_GATE_CUH
#define HARDWARE_NATIVE_DIRECT_ADULT_PHENOTYPE_ANATOMY_GATE_CUH

// h.phenotype_before_anatomy (#1587).
// Certification-order discipline as a fail-closed state machine: behavioral
// competence measured from boundary evidence alone is receipted first; the
// receipt -- and only the receipt -- releases internal anatomical analysis,
// which is stamped downstream of the phenotype evidence it explains.

#include <cstdint>
#include <type_traits>

namespace substrate::direct_adult_core {

struct DirectPhenotypeEvidence {
  std::uint64_t assimilated_consequences;
  std::uint64_t emitted_actions;
  std::uint32_t activation_covered_nodes;
  std::uint32_t lived_epochs;
};

inline constexpr std::uint32_t kPhenotypeMinAssimilated = 8u;
inline constexpr std::uint32_t kPhenotypeMinActions = 4u;
inline constexpr std::uint32_t kPhenotypeMinCoveredNodes = 8u;

inline std::uint64_t direct_phenotype_evidence_digest(
    const DirectPhenotypeEvidence& evidence) {
  std::uint64_t identity = 0x7068656e6f676174ull;
  identity ^= evidence.assimilated_consequences + 0x9e3779b97f4a7c15ull +
              (identity << 6) + (identity >> 2);
  identity ^= evidence.emitted_actions + 0x9e3779b97f4a7c15ull +
              (identity << 6) + (identity >> 2);
  identity ^= evidence.activation_covered_nodes + 0x9e3779b97f4a7c15ull +
              (identity << 6) + (identity >> 2);
  identity ^= evidence.lived_epochs + 0x9e3779b97f4a7c15ull +
              (identity << 6) + (identity >> 2);
  return identity == 0u ? 1u : identity | (1ull << 63);
}

struct DirectPhenotypeAnatomyGate {
  enum class Stage : std::uint32_t {
    unproven = 0u,
    phenotype_proven = 1u,
    anatomy_released = 2u,
  };

  Stage stage;                        // unproven
  std::uint64_t phenotype_receipt_identity;
  std::uint64_t anatomy_downstream_stamp;
};
static_assert(std::is_trivial_v<DirectPhenotypeAnatomyGate> &&
              std::is_standard_layout_v<DirectPhenotypeAnatomyGate>);

// Records a competence receipt only when boundary-measured evidence meets
// every threshold.  Sub-threshold evidence cannot mint a receipt.
inline bool record_phenotype_proof(DirectPhenotypeAnatomyGate* gate,
                                   const DirectPhenotypeEvidence& evidence) {
  if (gate == nullptr || gate->stage != DirectPhenotypeAnatomyGate::Stage::unproven)
    return false;
  if (evidence.assimilated_consequences < kPhenotypeMinAssimilated ||
      evidence.emitted_actions < kPhenotypeMinActions ||
      evidence.activation_covered_nodes < kPhenotypeMinCoveredNodes ||
      evidence.lived_epochs == 0u)
    return false;
  gate->phenotype_receipt_identity =
      direct_phenotype_evidence_digest(evidence);
  gate->stage = DirectPhenotypeAnatomyGate::Stage::phenotype_proven;
  return true;
}

// Releases anatomical analysis only downstream of a recorded phenotype
// proof, stamping every released artifact with that dependency.
inline bool release_phenotype_gated_anatomy(
    DirectPhenotypeAnatomyGate* gate) {
  if (gate == nullptr ||
      gate->stage != DirectPhenotypeAnatomyGate::Stage::phenotype_proven ||
      gate->phenotype_receipt_identity == 0u)
    return false;
  gate->anatomy_downstream_stamp = gate->phenotype_receipt_identity ^
                                   0x616e61746f6d7931ull;
  gate->stage = DirectPhenotypeAnatomyGate::Stage::anatomy_released;
  return true;
}

}  // namespace substrate::direct_adult_core

#endif
