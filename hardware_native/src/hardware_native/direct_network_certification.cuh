#ifndef HARDWARE_NATIVE_DIRECT_NETWORK_CERTIFICATION_CUH
#define HARDWARE_NATIVE_DIRECT_NETWORK_CERTIFICATION_CUH

#include "hardware_native/direct_network_life_function.cuh"

// Patch 0006 t0->t6 observer receipt. A valid receipt is necessary for adult
// test eligibility but cannot cause behavior. Every stage can refuse; the
// whole receipt certifies only when all seven measured stages certify.
namespace substrate::direct_network::certification {

// `not_evaluated` keeps "nothing asked" distinct from pass or refusal.
enum class StageVerdict : std::uint32_t {
  not_evaluated = 0,
  refused = 1,
  certified = 2,
};

inline constexpr std::uint32_t kStageCount = 7u;   // t0..t6
// Raw counts the verdicts were derived from. Wider than the requirement count
// on purpose: when two independent observations share ONE spec requirement,
// only the separate counts can show that both halves of the guard are alive.
inline constexpr std::uint32_t kMaxMeasured = 11u;

struct StageReceipt {
  std::uint32_t stage;
  StageVerdict verdict;
  // How many requirements the SPEC names for this stage.
  std::uint32_t requirement_count;
  // How many of them this build can actually measure. The gap between these
  // two numbers is the honest hole, carried in the receipt instead of a
  // comment.
  std::uint32_t requirement_assayed;
  std::uint32_t requirement_met;
  std::uint32_t unmet_mask;      // bit i: requirement i was assayed and FAILED
  std::uint32_t unassayed_mask;  // bit i: no assay exists for requirement i
  std::uint32_t measured[kMaxMeasured];
};
static_assert(std::is_standard_layout_v<StageReceipt> && std::is_trivial_v<StageReceipt>);

struct NetworkFoundationReceipt {
  Root256 genome_root;
  Root256 body_root;
  Root256 environment_root;
  Root256 birth_root;
  Root256 juvenile_morphology_root;
  std::uint64_t matter_bytes_paid;
  std::uint32_t replica_count;
  StageReceipt stage[kStageCount];
  std::uint32_t stages_certified;
  std::uint32_t stages_refused;
  // 1 only when every one of the seven stages is `certified`.
  std::uint32_t certified;
};
static_assert(std::is_standard_layout_v<NetworkFoundationReceipt> &&
              std::is_trivial_v<NetworkFoundationReceipt>);

// Observer-only copy of the resident admission law.  `earned` is useful only
// as evidence that resident development already changed the organism; this
// receipt has no execution, scheduling, learning or credit authority.
struct ResidentAdultAdmissionReceipt {
  ResidentAdultAdmissionState state;
  Root256 birth_root;
};
static_assert(std::is_standard_layout_v<ResidentAdultAdmissionReceipt> &&
              std::is_trivial_v<ResidentAdultAdmissionReceipt>);

ResidentAdultAdmissionReceipt observe_resident_adult_admission(
    const DirectBrain& brain);

// C0 certifies one construction event, never later learning or capability.
// It is intentionally independent of the wider t0->t6 receipt: standing
// unassayed later stages cannot turn a measured lawful birth into a language
// or adult-capability claim, and an unrelated later-stage RED cannot hide the
// exact C0 construction result. Donor shape and copied root strings remain
// explicit refusal controls.
// `lawful` is exactly the conjunction of the eight named requirement bits.
inline constexpr std::uint32_t kC0CanonicalSpecies = 1u << 0, kC0ExactConstructionInputs = 1u << 1;
inline constexpr std::uint32_t kC0ExactBirthAuthority = 1u << 2, kC0GrownMorphology = 1u << 3;
inline constexpr std::uint32_t kC0DetachedResidentHandoff = 1u << 4, kC0SemanticallyBlank = 1u << 5;
inline constexpr std::uint32_t kC0FiniteOwnedMatter = 1u << 6, kC0PhysicalBody = 1u << 7;
inline constexpr std::uint32_t kC0AllRequirements =
    kC0CanonicalSpecies | kC0ExactConstructionInputs | kC0ExactBirthAuthority |
    kC0GrownMorphology | kC0DetachedResidentHandoff | kC0SemanticallyBlank |
    kC0FiniteOwnedMatter | kC0PhysicalBody;

struct LawfulConstructionReceipt {
  Root256 genome_root, territory_layout_root, body_root, environment_root;
  Root256 birth_root, observed_state_root;
  std::uint64_t born_owned_bytes, current_charged_bytes;
  std::uint32_t node_count, active_route_count, route_capacity, territory_count;
  std::uint32_t boundary_port_count, lineage_count_lower_bound;
  std::uint32_t lived_node_count, credited_route_count, exact_history_records;
  std::uint32_t raw_contact_binding_count, recipe_incidence_count;
  std::uint32_t assayed_mask, unmet_mask, lawful;
};
static_assert(std::is_standard_layout_v<LawfulConstructionReceipt> &&
              std::is_trivial_v<LawfulConstructionReceipt>);

// ---------------------------------------------------------------------------
// Requirement bits, named so a falsifier can assert WHICH requirement refused
// rather than only that the stage did. A stage-level pass/fail cannot tell a
// guard that fired for the right reason from one that fires on everything.
// ---------------------------------------------------------------------------

// t0 -- construction fronts.
inline constexpr std::uint32_t kT0TwoGenesisLineages = 1u << 0;
inline constexpr std::uint32_t kT0FrontGeometryExact = 1u << 1;
inline constexpr std::uint32_t kT0NoHostWrittenMatureEdges = 1u << 2;
inline constexpr std::uint32_t kT0MatterClosure = 1u << 3;
inline constexpr std::uint32_t kT0AllocationPermutationInvariance = 1u << 4;
inline constexpr std::uint32_t kT0RequirementCount = 5u;

// t1 -- branch / fuse / retract.
inline constexpr std::uint32_t kT1RouteExtension = 1u << 0;
inline constexpr std::uint32_t kT1BranchEvents = 1u << 1;
inline constexpr std::uint32_t kT1CrossLineageFusion = 1u << 2;
inline constexpr std::uint32_t kT1RetractionUnderPressure = 1u << 3;
inline constexpr std::uint32_t kT1NoDenseCollapse = 1u << 4;
inline constexpr std::uint32_t kT1NoSilentOverflow = 1u << 5;
inline constexpr std::uint32_t kT1ReplayIdentical = 1u << 6;
inline constexpr std::uint32_t kT1RequirementCount = 7u;

// t6 -- certified language-naive juvenile.
inline constexpr std::uint32_t kT6DeterministicRegrowth = 1u << 0;
inline constexpr std::uint32_t kT6ExactMatterClosure = 1u << 1;
inline constexpr std::uint32_t kT6RequirementCount = 13u;

// t2 is assayed only by the organism's local law in direct_network_basin_probe.
inline constexpr std::uint32_t kT2ProbeSeedsPostHocFromMatureTissue = 1u << 0;
inline constexpr std::uint32_t kT2MultipleHorizonsThroughLocalLaw = 1u << 1;
inline constexpr std::uint32_t kT2SeveralRecurrentReturnBasins = 1u << 2;
inline constexpr std::uint32_t kT2DisjointAndOverlappingPopulations = 1u << 3;

inline constexpr std::uint32_t kT2RequirementCount = 4u;

// t3 asks different Gamma dispositions to grow different recurrence dynamics;
// the basin probe groups returns by seed territory against a uniform sham.
inline constexpr std::uint32_t kT3DistinctRecurrenceAcrossTerritories = 1u << 0;
inline constexpr std::uint32_t kT3RequirementCount = 6u;

// t4..t5 have no assay in this build at all; their counts are the spec's, so
// the receipt reports the size of the hole rather than hiding it.
inline constexpr std::uint32_t kT4RequirementCount = 5u;
inline constexpr std::uint32_t kT5RequirementCount = 6u;

// One grown organism plus the receipt gestation returned for it.
struct JuvenileReplica {
  const DirectBrain* brain;
  const DirectBirthReceiptV1* birth;
};

// Certification takes REPLICAS, not one organism, because two of the spec's
// requirements -- allocation permutation invariance (t0) and deterministic
// fresh regrowth from the same Gamma (t6) -- are not properties of a single
// juvenile. A single-brain entry point would have to score them from evidence
// that does not exist, so the signature refuses to offer one: with
// replica_count == 1 those requirements come back UNASSAYED and their stages
// cannot certify.
//
// Observer-only. This reads device memory and computes hashes; it writes
// nothing back into any organism and never appears on an execution hot path.
NetworkFoundationReceipt certify_direct_juvenile(const JuvenileReplica* replicas,
                                                 std::uint32_t replica_count,
                                                 std::uint32_t block_size = 256u);

// Source objects authorize the observer-only C0 assay through exact roots.
LawfulConstructionReceipt certify_lawful_direct_construction(
    const DirectGenomeV1& genome, const DirectBodyManifestV1& body,
    const DirectDevelopmentEnvironmentV1& environment,
    const JuvenileReplica& juvenile, std::uint32_t block_size = 256u);

// Score one stage from an assayed/unmet mask pair. Exposed so that a module
// which measures a stage this translation unit cannot -- t2 needs the adult
// runtime, which sits above this library -- fills the receipt through the SAME
// promotion rule rather than inventing its own. Nothing outside this rule may
// set a verdict: a stage certifies only when every requirement the spec names
// was assayed AND met.
void score_stage(StageReceipt& stage, std::uint32_t index, std::uint32_t requirement_count,
                 std::uint32_t assayed_mask, std::uint32_t unmet_mask);

// Recompute the whole-receipt conjunction after a stage has been scored.
void recompute_certification(NetworkFoundationReceipt& receipt);

// The eligibility predicate patch 0010 consumes. Necessary, never sufficient,
// and never causal: nothing downstream of this reads organism state.
bool adult_test_eligible(const NetworkFoundationReceipt& receipt);

}  // namespace substrate::direct_network::certification

#endif  // HARDWARE_NATIVE_DIRECT_NETWORK_CERTIFICATION_CUH
