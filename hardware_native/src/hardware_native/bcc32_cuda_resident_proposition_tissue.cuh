#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace bcc32_cuda_resident_proposition_tissue {

constexpr std::uint32_t kMaximumPopulationCells = 16u;
constexpr std::uint32_t kDefaultCompletionCells = 8u;
constexpr std::uint32_t kObservationQuantum = 1u;
constexpr std::uint32_t kInterventionQuantum = 4u;
constexpr std::uint32_t kMinimumIndependentContexts = 3u;
constexpr std::uint32_t kMinimumObservationMass = 3u * kObservationQuantum;
constexpr std::uint32_t kMinimumInterventionMass = 2u * kInterventionQuantum;
// Candidate acquisition runs in the adult's causal tick path.  A saturated
// tissue must therefore exert bounded physical pressure rather than turning
// one novel contact into a scan of the entire resident table.  Claim and
// lookup deliberately share this bound: no relation can be inserted where it
// later becomes invisible to lookup.
constexpr std::uint32_t kMaximumSynapseProbe = 32u;
// A cell-pair can be shared by several sparse source populations.  Those
// populations must remain exact resident cohorts: a digest may locate a
// synapse, but no digest or cell overlap is allowed to stand in for cohort
// identity when causal evidence is selected.
constexpr std::uint32_t kMaximumCohortsPerSynapse = 4u;
constexpr std::uint32_t kOrderedBindingRoleCount = 3u;
constexpr std::uint32_t kMaximumOrderedRoleUnits = 2u;
constexpr std::uint32_t kMaximumOrderedBindingProbe = 32u;

enum class CompletionPolicy : std::uint32_t {
  causal = 0u,
  discourse = 1u,
};

enum class OrderedBindingQualification : std::uint32_t {
  exact_episode = 0u,
  discourse = 1u,
  causal = 2u,
};

enum class OrderedBindingAssimilationStatus : std::uint32_t {
  accepted = 0u,
  replay = 1u,
  invalid = 2u,
  capacity_overflow = 3u,
  insufficient_mass = 4u,
};

// Cell numbers are opaque addresses in resident matter. No value names an
// entity, argument role, relation, word, or authored semantic category.
struct SparsePopulationView {
  const std::uint32_t* cells = nullptr;
  std::uint32_t count = 0u;
};

// Exact learned unit identities preserve the physical event that produced a
// role population. Counts, rather than any numeric unit value, mark presence.
// Population geometry remains the generalized representation.
struct OrderedRoleUnitIdentity {
  std::uint32_t units[kOrderedBindingRoleCount][kMaximumOrderedRoleUnits]{};
  std::uint32_t counts[kOrderedBindingRoleCount]{};
};

struct PopulationCohortEvidence {
  std::uint32_t cells[kMaximumPopulationCells]{};
  std::uint32_t cell_count = 0u;
  std::uint32_t claimed = 0u;
  std::uint64_t observational_support = 0u;
  std::uint64_t intervention_support = 0u;
  std::uint64_t counterevidence = 0u;
  // Only the exact contexts needed to establish the independent-context
  // threshold are retained. Later contexts may add evidence, but cannot forge
  // diversity or satisfy context-conditioned recall unless their complete
  // anatomy is one of these resident records.
  std::uint32_t qualifying_context_cells[kMinimumIndependentContexts]
                                                [kMaximumPopulationCells]{};
  std::uint32_t qualifying_context_counts[kMinimumIndependentContexts]{};
  std::uint32_t qualifying_context_count = 0u;
  // One unit claims the cohort and one unit retains each exact member.
  std::uint64_t structure_mass = 0u;
  std::uint64_t context_mass = 0u;
};

// A proposition is retained as one ordered, indivisible population record.
// The array offsets identify physical roles inside this record only; no cell
// number or hash value means agent, predicate, patient, or a relation kind.
// Candidate sensory encoders may propose these three populations, but only
// full resident equality below can authorize later use.
struct OrderedRoleBindingEvidence {
  std::uint32_t role_cells[kOrderedBindingRoleCount]
                          [kMaximumPopulationCells]{};
  std::uint32_t role_counts[kOrderedBindingRoleCount]{};
  std::uint32_t role_units[kOrderedBindingRoleCount]
                          [kMaximumOrderedRoleUnits]{};
  std::uint32_t role_unit_counts[kOrderedBindingRoleCount]{};
  std::uint32_t claimed = 0u;
  std::uint32_t reserved = 0u;
  // One-shot episode evidence is deliberately distinct from evidence that
  // generalized across varied exact contexts. The former can recall only this
  // complete record; it can never license a generalized or causal assertion.
  std::uint64_t episodic_observation_mass = 0u;
  std::uint64_t generalized_observation_mass = 0u;
  std::uint64_t intervention_support = 0u;
  std::uint64_t counterevidence = 0u;
  std::uint32_t qualifying_context_cells[kMinimumIndependentContexts]
                                                [kMaximumPopulationCells]{};
  std::uint32_t qualifying_context_counts[kMinimumIndependentContexts]{};
  std::uint64_t qualifying_episode_revisions[kMinimumIndependentContexts]{};
  std::uint64_t last_evidence_revision = 0u;
  std::uint32_t qualifying_context_count = 0u;
  std::uint32_t reserved_context = 0u;
  // Every retained cell and the record header own conserved structure mass.
  // Role-specific mass makes a focal predicate or target cut observable and
  // prevents a partially represented proposition from remaining authorized.
  std::uint64_t role_structure_mass[kOrderedBindingRoleCount]{};
  std::uint64_t structure_mass = 0u;
  std::uint64_t context_mass = 0u;
};

struct OrderedBindingAdmissionState {
  std::uint64_t accepted = 0u;
  std::uint64_t replays = 0u;
  std::uint64_t invalid_attempts = 0u;
  std::uint64_t capacity_rejections = 0u;
  std::uint64_t mass_rejections = 0u;
  // Capacity pressure owns conserved matter instead of disappearing as a
  // silent failed insertion. This residue never authorizes a proposition.
  std::uint64_t overflow_mass = 0u;
  std::uint64_t projected_bytes = 0u;
};

struct SparseBindingSynapse {
  std::uint32_t source_cell = 0u;
  std::uint32_t target_cell = 0u;
  std::uint32_t claimed = 0u;
  // Conserved residue of relation-formation attempts rejected because this
  // synapse's home collision window was saturated.  Keeping it in resident
  // matter exposes capacity pressure without inventing an off-lattice table
  // or silently discarding the attempted structural work.
  std::uint32_t overflow_mass = 0u;
  // A full cohort table makes this synapse semantically ambiguous.  The
  // residue is conserved and forces completion to abstain rather than treat a
  // lossy subset as the whole learned population.
  std::uint32_t cohort_overflow_mass = 0u;
  // The tissue's OWN record of "was I contacted since the last drive" -- no
  // clock, no age field, no host-supplied list. assimilate_experience sets
  // this the instant it credits this synapse with any contact (observation,
  // concordant intervention, or discordant counterevidence), at the same
  // place it credits support. apply_unreinforced_support_drain clears it
  // every time it inspects this synapse: set means spared for THIS drive
  // only, never permanent and never inherited from a prior drive. A synapse
  // that is claimed but never contacted (e.g. a bare find_or_claim_synapse
  // with no cohort ever written) starts and stays at 0, so it decays on the
  // very next drive.
  std::uint32_t contacted_since_drain = 0u;
  // Exact identity of the relation whose rejected attempts own overflow_mass.
  // A locator digest is insufficient here: unrelated challengers may collide
  // in the same bounded neighborhood but may not pool eviction pressure.
  std::uint64_t overflow_binding = 0u;
  std::uint64_t structure_mass = 0u;
};

struct TissueScalars {
  std::uint64_t initial_mass = 0u;
  std::uint64_t free_mass = 0u;
  std::uint64_t revision = 0u;
  std::uint64_t accepted_observations = 0u;
  std::uint64_t accepted_interventions = 0u;
  std::uint64_t accepted_counterevidence = 0u;
  std::uint32_t occupied_synapses = 0u;
  std::uint32_t lesion_revision = 0u;
};

struct TissueView {
  SparseBindingSynapse* synapses = nullptr;
  std::uint32_t synapse_capacity = 0u;
  std::uint32_t cell_capacity = 0u;
  TissueScalars* scalars = nullptr;
  PopulationCohortEvidence* cohorts = nullptr;
  std::uint32_t cohort_capacity = 0u;
  OrderedRoleBindingEvidence* ordered_bindings = nullptr;
  std::uint32_t ordered_binding_capacity = 0u;
  OrderedBindingAdmissionState* ordered_binding_admission = nullptr;
};

struct CompletionWorkspaceView {
  std::uint64_t* cell_scores = nullptr;
  std::uint32_t cell_capacity = 0u;
  std::uint32_t* output_cells = nullptr;
  std::uint64_t* output_scores = nullptr;
  std::uint32_t output_capacity = 0u;
};

struct CompletionResult {
  std::uint32_t ready = 0u;
  std::uint32_t output_count = 0u;
  std::uint32_t cue_cells_matched = 0u;
  std::uint32_t qualified_synapses = 0u;
  std::uint64_t strongest_score = 0u;
  std::uint64_t uncertain_mass = 0u;
  std::uint64_t tissue_revision = 0u;
};

struct OrderedBindingLookupResult {
  std::uint32_t ready = 0u;
  std::uint32_t output_count = 0u;
  std::uint32_t overflow = 0u;
  std::uint32_t exact_topic_matches = 0u;
  std::uint64_t tissue_revision = 0u;
};

struct OrderedBindingAssimilationResult {
  OrderedBindingAssimilationStatus status =
      OrderedBindingAssimilationStatus::invalid;
  std::uint32_t binding_index = 0xffffffffu;
  std::uint64_t required_mass = 0u;
  std::uint64_t projected_bytes = 0u;
  std::uint64_t tissue_revision = 0u;
};

static_assert(sizeof(SparseBindingSynapse) == 40u);
static_assert(sizeof(PopulationCohortEvidence) == 320u);

__host__ __device__ inline std::uint32_t mix32(std::uint32_t value) {
  value ^= value >> 16u;
  value *= 0x7feb352du;
  value ^= value >> 15u;
  value *= 0x846ca68bu;
  value ^= value >> 16u;
  return value;
}

__device__ inline bool valid_population(SparsePopulationView population,
                                        std::uint32_t cell_capacity) {
  if (population.cells == nullptr || population.count == 0u ||
      population.count > kMaximumPopulationCells)
    return false;
  for (std::uint32_t index = 0u; index < population.count; ++index) {
    if (population.cells[index] >= cell_capacity)
      return false;
    for (std::uint32_t previous = 0u; previous < index; ++previous)
      if (population.cells[previous] == population.cells[index])
        return false;
  }
  return true;
}

__device__ inline bool population_contains(SparsePopulationView population, std::uint32_t cell) {
  for (std::uint32_t index = 0u; index < population.count; ++index)
    if (population.cells[index] == cell)
      return true;
  return false;
}

__device__ inline bool cohort_contains(const PopulationCohortEvidence& cohort,
                                       std::uint32_t cell) {
  for (std::uint32_t index = 0u; index < cohort.cell_count; ++index)
    if (cohort.cells[index] == cell)
      return true;
  return false;
}

__device__ inline bool same_cohort(const PopulationCohortEvidence& cohort,
                                   SparsePopulationView population) {
  if (cohort.claimed == 0u || cohort.cell_count != population.count)
    return false;
  for (std::uint32_t index = 0u; index < population.count; ++index)
    if (!cohort_contains(cohort, population.cells[index]))
      return false;
  return true;
}

__device__ inline std::uint32_t cohort_overlap(const PopulationCohortEvidence& cohort,
                                               SparsePopulationView population) {
  std::uint32_t overlap = 0u;
  for (std::uint32_t index = 0u; index < population.count; ++index)
    overlap += cohort_contains(cohort, population.cells[index]);
  return overlap;
}

__device__ inline PopulationCohortEvidence* synapse_cohorts(
    TissueView tissue, SparseBindingSynapse* synapse) {
  if (tissue.cohorts == nullptr || synapse == nullptr)
    return nullptr;
  const std::uint32_t index = static_cast<std::uint32_t>(synapse - tissue.synapses);
  const std::uint64_t begin =
      static_cast<std::uint64_t>(index) * kMaximumCohortsPerSynapse;
  return begin + kMaximumCohortsPerSynapse <= tissue.cohort_capacity
             ? tissue.cohorts + begin
             : nullptr;
}

__device__ inline const PopulationCohortEvidence* synapse_cohorts(
    TissueView tissue, const SparseBindingSynapse* synapse) {
  return synapse_cohorts(tissue, const_cast<SparseBindingSynapse*>(synapse));
}

__device__ inline void return_synapse_mass(TissueView tissue,
                                           SparseBindingSynapse* synapse) {
  TissueScalars* scalars = tissue.scalars;
  PopulationCohortEvidence* cohorts = synapse_cohorts(tissue, synapse);
  scalars->free_mass += synapse->structure_mass + synapse->overflow_mass +
                        synapse->cohort_overflow_mass;
  if (cohorts != nullptr) {
    for (std::uint32_t index = 0u; index < kMaximumCohortsPerSynapse; ++index) {
      const PopulationCohortEvidence& cohort = cohorts[index];
      scalars->free_mass += cohort.observational_support + cohort.intervention_support +
                            cohort.counterevidence + cohort.structure_mass +
                            cohort.context_mass;
      cohorts[index] = PopulationCohortEvidence{};
    }
  }
  *synapse = SparseBindingSynapse{};
  --scalars->occupied_synapses;
}

__device__ inline bool exact_context_equals(const std::uint32_t* cells,
                                            std::uint32_t count,
                                            SparsePopulationView context) {
  if (count != context.count)
    return false;
  for (std::uint32_t index = 0u; index < context.count; ++index) {
    bool present = false;
    for (std::uint32_t resident = 0u; resident < count; ++resident)
      present = present || cells[resident] == context.cells[index];
    if (!present)
      return false;
  }
  return true;
}

__device__ inline bool valid_distinct_population(SparsePopulationView population,
                                                 std::uint32_t cell_capacity) {
  if (!valid_population(population, cell_capacity))
    return false;
  for (std::uint32_t left = 0u; left < population.count; ++left)
    for (std::uint32_t right = left + 1u; right < population.count; ++right)
      if (population.cells[left] == population.cells[right])
        return false;
  return true;
}

__device__ inline bool exact_population_equals(const std::uint32_t* cells,
                                               std::uint32_t count,
                                               SparsePopulationView population) {
  return exact_context_equals(cells, count, population);
}

__host__ __device__ inline std::uint32_t ordered_population_hash(
    SparsePopulationView population) {
  std::uint32_t sum = mix32(population.count * 0x9e3779b9u);
  std::uint32_t xors = 0u;
  for (std::uint32_t index = 0u; index < population.count; ++index) {
    const std::uint32_t mixed = mix32(population.cells[index] + 0x85ebca6bu);
    sum += mixed;
    xors ^= mixed;
  }
  return mix32(sum ^ xors);
}

__host__ __device__ inline std::uint32_t ordered_binding_hash(
    SparsePopulationView agent, SparsePopulationView predicate,
    SparsePopulationView patient) {
  return mix32(ordered_population_hash(agent) * 0x9e3779b9u ^
               ordered_population_hash(predicate) * 0x85ebca6bu ^
               ordered_population_hash(patient) * 0xc2b2ae35u);
}

__host__ __device__ inline std::uint32_t ordered_identity_hash(
    const OrderedRoleUnitIdentity& identity) {
  std::uint32_t hash = 0x9e3779b9u;
  for (std::uint32_t role = 0u; role < kOrderedBindingRoleCount; ++role) {
    hash = mix32(hash ^ identity.counts[role] * (0x85ebca6bu + role));
    for (std::uint32_t index = 0u; index < identity.counts[role]; ++index)
      hash = mix32(hash ^ mix32(identity.units[role][index] + role));
  }
  return hash;
}

__host__ __device__ inline bool ordered_identity_present(
    const OrderedRoleUnitIdentity& identity) {
  for (std::uint32_t role = 0u; role < kOrderedBindingRoleCount; ++role)
    if (identity.counts[role] != 0u)
      return true;
  return false;
}

__host__ __device__ inline bool ordered_identity_valid(
    const OrderedRoleUnitIdentity& identity) {
  for (std::uint32_t role = 0u; role < kOrderedBindingRoleCount; ++role)
    if (identity.counts[role] > kMaximumOrderedRoleUnits)
      return false;
  return true;
}

__device__ inline bool same_ordered_identity(
    const OrderedRoleBindingEvidence& binding,
    const OrderedRoleUnitIdentity& identity) {
  if (!ordered_identity_present(identity))
    return true;
  for (std::uint32_t role = 0u; role < kOrderedBindingRoleCount; ++role) {
    if (identity.counts[role] == 0u ||
        identity.counts[role] > kMaximumOrderedRoleUnits ||
        binding.role_unit_counts[role] != identity.counts[role])
      return false;
    for (std::uint32_t index = 0u; index < identity.counts[role]; ++index)
      if (binding.role_units[role][index] != identity.units[role][index])
        return false;
  }
  return true;
}

__device__ inline SparsePopulationView ordered_binding_role(
    const OrderedRoleBindingEvidence& binding, std::uint32_t role) {
  return role < kOrderedBindingRoleCount
             ? SparsePopulationView{binding.role_cells[role],
                                    binding.role_counts[role]}
             : SparsePopulationView{};
}

__device__ inline bool same_ordered_binding(
    const OrderedRoleBindingEvidence& binding, SparsePopulationView agent,
    SparsePopulationView predicate, SparsePopulationView patient) {
  return binding.claimed != 0u &&
         exact_population_equals(binding.role_cells[0], binding.role_counts[0], agent) &&
         exact_population_equals(binding.role_cells[1], binding.role_counts[1], predicate) &&
         exact_population_equals(binding.role_cells[2], binding.role_counts[2], patient);
}

__device__ inline bool ordered_binding_structurally_intact(
    const OrderedRoleBindingEvidence& binding, std::uint32_t cell_capacity) {
  if (binding.claimed == 0u)
    return false;
  std::uint64_t expected_structure_mass = 1u;
  for (std::uint32_t role = 0u; role < kOrderedBindingRoleCount; ++role) {
    const SparsePopulationView population = ordered_binding_role(binding, role);
    if (!valid_distinct_population(population, cell_capacity) ||
        binding.role_unit_counts[role] > kMaximumOrderedRoleUnits ||
        binding.role_structure_mass[role] !=
            population.count + binding.role_unit_counts[role])
      return false;
    expected_structure_mass +=
        population.count + binding.role_unit_counts[role];
  }
  return binding.structure_mass == expected_structure_mass;
}

__device__ inline std::uint64_t ordered_binding_positive_mass(
    const OrderedRoleBindingEvidence& binding) {
  return binding.episodic_observation_mass +
         binding.generalized_observation_mass + binding.intervention_support;
}

__device__ inline bool ordered_binding_qualified(
    const OrderedRoleBindingEvidence& binding,
    OrderedBindingQualification qualification, std::uint32_t cell_capacity) {
  if (!ordered_binding_structurally_intact(binding, cell_capacity) ||
      ordered_binding_positive_mass(binding) <= binding.counterevidence)
    return false;
  if (qualification == OrderedBindingQualification::exact_episode)
    return binding.episodic_observation_mass > binding.counterevidence;
  if (binding.qualifying_context_count < kMinimumIndependentContexts)
    return false;
  if (qualification == OrderedBindingQualification::discourse)
    return binding.generalized_observation_mass + binding.intervention_support >
           binding.counterevidence;
  return binding.intervention_support >= kMinimumInterventionMass &&
         binding.intervention_support > binding.counterevidence;
}

__device__ inline std::int32_t ordered_binding_context_slot(
    const OrderedRoleBindingEvidence& binding, SparsePopulationView context) {
  for (std::uint32_t index = 0u; index < binding.qualifying_context_count; ++index)
    if (exact_context_equals(binding.qualifying_context_cells[index],
                             binding.qualifying_context_counts[index], context))
      return static_cast<std::int32_t>(index);
  return -1;
}

__device__ inline bool ordered_binding_episode_seen(
    const OrderedRoleBindingEvidence& binding, std::uint64_t evidence_revision) {
  if (evidence_revision == 0u)
    return false;
  for (std::uint32_t index = 0u; index < binding.qualifying_context_count; ++index)
    if (binding.qualifying_episode_revisions[index] == evidence_revision)
      return true;
  return false;
}

__device__ inline bool cohort_has_context(const PopulationCohortEvidence& cohort,
                                          SparsePopulationView context) {
  for (std::uint32_t index = 0u; index < cohort.qualifying_context_count; ++index)
    if (exact_context_equals(cohort.qualifying_context_cells[index],
                             cohort.qualifying_context_counts[index], context))
      return true;
  return false;
}

__device__ inline void retain_qualifying_context(TissueScalars* scalars,
                                                 PopulationCohortEvidence* cohort,
                                                 SparsePopulationView context) {
  if (cohort_has_context(*cohort, context) ||
      cohort->qualifying_context_count >= kMinimumIndependentContexts)
    return;
  const std::uint64_t required_mass = 1u + context.count;
  if (scalars->free_mass < required_mass)
    return;
  const std::uint32_t slot = cohort->qualifying_context_count;
  for (std::uint32_t index = 0u; index < context.count; ++index)
    cohort->qualifying_context_cells[slot][index] = context.cells[index];
  cohort->qualifying_context_counts[slot] = context.count;
  ++cohort->qualifying_context_count;
  cohort->context_mass += required_mass;
  scalars->free_mass -= required_mass;
}

__host__ __device__ inline std::uint32_t binding_hash(std::uint32_t source,
                                                      std::uint32_t target) {
  return mix32(source * 0x9e3779b9u ^ target * 0x85ebca6bu);
}

// Retention is learned evidence, not age or an authored semantic priority.
// Counterevidence performs local LTD by cancelling positive support before a
// relation competes for scarce tissue.
__device__ inline std::uint64_t synapse_retained_support(
    TissueView tissue, const SparseBindingSynapse* synapse) {
  const PopulationCohortEvidence* cohorts = synapse_cohorts(tissue, synapse);
  if (cohorts == nullptr)
    return 0u;
  std::uint64_t positive = 0u;
  std::uint64_t counter = 0u;
  for (std::uint32_t index = 0u; index < kMaximumCohortsPerSynapse; ++index) {
    positive += cohorts[index].observational_support +
                cohorts[index].intervention_support;
    counter += cohorts[index].counterevidence;
  }
  return positive > counter ? positive - counter : 0u;
}

__host__ __device__ inline std::uint64_t exact_binding_identity(
    std::uint32_t source, std::uint32_t target) {
  return (static_cast<std::uint64_t>(source) << 32u) | target;
}

__device__ inline std::uint64_t synapse_represented_mass(
    TissueView tissue, const SparseBindingSynapse* synapse) {
  std::uint64_t mass = synapse->structure_mass + synapse->overflow_mass +
                       synapse->cohort_overflow_mass;
  const PopulationCohortEvidence* cohorts = synapse_cohorts(tissue, synapse);
  if (cohorts == nullptr)
    return mass;
  for (std::uint32_t index = 0u; index < kMaximumCohortsPerSynapse; ++index) {
    mass += cohorts[index].observational_support + cohorts[index].intervention_support +
            cohorts[index].counterevidence + cohorts[index].structure_mass +
            cohorts[index].context_mass;
  }
  return mass;
}

// A full local collision window turns rejected structural matter into
// homeostatic pressure. Each recurrent challenger owns one available pressure
// reservoir in that bounded window; unrelated challengers never reset or pool
// one another's acquisition history. A challenger can reclaim only an
// unpressured incumbent or its own reservoir. The strict mass comparison means
// one-off novelty cannot evict a one-observation relation; recurrent demand
// can. No clock, random victim, host score, or semantic category participates,
// and all displaced matter returns through the same conservation ledger before
// the challenger is claimed.
__device__ inline SparseBindingSynapse* apply_synapse_turnover_pressure(
    TissueView tissue, std::uint32_t begin, std::uint32_t probe_limit,
    std::uint32_t source, std::uint32_t target,
    std::uint64_t required_followup_mass) {
  const std::uint64_t challenger_identity = exact_binding_identity(source, target);
  SparseBindingSynapse* pressure_owner = nullptr;
  SparseBindingSynapse* empty_reservoir = nullptr;
  for (std::uint32_t probe = 0u; probe < probe_limit; ++probe) {
    SparseBindingSynapse* incumbent =
        tissue.synapses + (begin + probe) % tissue.synapse_capacity;
    if (incumbent->claimed == 0u)
      continue;
    if (incumbent->overflow_mass != 0u &&
        incumbent->overflow_binding == UINT64_MAX) {
      tissue.scalars->free_mass += incumbent->overflow_mass;
      incumbent->overflow_mass = 0u;
      incumbent->overflow_binding = 0u;
    }
    if (incumbent->overflow_mass != 0u &&
        incumbent->overflow_binding == challenger_identity)
      pressure_owner = incumbent;
    if (incumbent->overflow_mass == 0u && empty_reservoir == nullptr)
      empty_reservoir = incumbent;
  }

  // This challenger's own live accumulated pressure. Read once, before the
  // weakest scan below, because that scan now needs it to decide whether a
  // slot parked with a DIFFERENT identity's overflow may be considered at
  // all -- an unfundable bid must not be able to squat a slot forever simply
  // because nobody else's pressure ever happens to clear its stake.
  const std::uint64_t pressure =
      pressure_owner != nullptr ? pressure_owner->overflow_mass : 0u;

  // Two pools, scored separately. `weakest_plain` is exactly the original
  // scan: any claimed incumbent that is not currently hosting a STRANGER's
  // parked overflow (this challenger's own reservoir, pressure_owner, is
  // plain here -- it competes on its own resident evidence like anything
  // else). `weakest_foreign` is new: an incumbent parked with another
  // identity's overflow is only even a candidate once this challenger's own
  // pressure strictly clears that stranger's stake, and even then it only
  // ever WINS over the plain pool when it is genuinely, strictly the weaker
  // resident matter -- a tie always goes to the plain incumbent. That
  // second guard is what stops two still-live, still-competing challengers
  // from ever reaching into each other's reservoir merely by tying and
  // arriving first in probe order:
  // bcc32_cuda_proposition_tissue_turnover_contract's alternating-challenger
  // arm depends on exactly this, and every incumbent it ever parks foreign
  // overflow on is support-tied with the plain pool, never strictly weaker.
  SparseBindingSynapse* weakest_plain = nullptr;
  std::uint64_t weakest_plain_support = UINT64_MAX;
  SparseBindingSynapse* weakest_foreign = nullptr;
  std::uint64_t weakest_foreign_support = UINT64_MAX;
  for (std::uint32_t probe = 0u; probe < probe_limit; ++probe) {
    SparseBindingSynapse* incumbent =
        tissue.synapses + (begin + probe) % tissue.synapse_capacity;
    if (incumbent->claimed == 0u)
      continue;
    const bool foreign_overflow =
        incumbent->overflow_mass != 0u && incumbent != pressure_owner;
    if (foreign_overflow && pressure <= incumbent->overflow_mass)
      continue;
    const std::uint64_t support = synapse_retained_support(tissue, incumbent);
    if (foreign_overflow) {
      if (weakest_foreign == nullptr || support < weakest_foreign_support) {
        weakest_foreign = incumbent;
        weakest_foreign_support = support;
      }
    } else if (weakest_plain == nullptr || support < weakest_plain_support) {
      weakest_plain = incumbent;
      weakest_plain_support = support;
    }
  }
  SparseBindingSynapse* weakest = weakest_plain;
  std::uint64_t weakest_support = weakest_plain_support;
  if (weakest_foreign != nullptr &&
      (weakest_plain == nullptr || weakest_foreign_support < weakest_plain_support)) {
    weakest = weakest_foreign;
    weakest_support = weakest_foreign_support;
  }

  const std::uint64_t victim_mass =
      weakest != nullptr ? synapse_represented_mass(tissue, weakest) : 0u;
  const std::uint64_t reclaimed_mass =
      victim_mass + ((pressure_owner != nullptr && weakest != pressure_owner)
                         ? pressure
                         : 0u);
  const bool transaction_funded =
      weakest != nullptr && reclaimed_mass != 0u &&
      tissue.scalars->free_mass + reclaimed_mass - 1u >= required_followup_mass;
  if (weakest != nullptr && pressure > weakest_support && transaction_funded) {
    if (pressure_owner != weakest) {
      pressure_owner->overflow_mass = 0u;
      pressure_owner->overflow_binding = 0u;
      tissue.scalars->free_mass += pressure;
    }
    // If `weakest` is itself a foreign-held incumbent (weakest_foreign above),
    // its parked overflow_mass is folded into free_mass by return_synapse_mass
    // below, exactly like its structure and cohort mass -- refunded to the
    // shared pool, never transferred to this challenger. The winner still
    // pays its own structure_mass out of that same pool a few lines down.
    return_synapse_mass(tissue, weakest);
    weakest->source_cell = source;
    weakest->target_cell = target;
    weakest->claimed = 1u;
    weakest->structure_mass = 1u;
    --tissue.scalars->free_mass;
    ++tissue.scalars->occupied_synapses;
    return weakest;
  }

  SparseBindingSynapse* reservoir =
      pressure_owner != nullptr ? pressure_owner : empty_reservoir;
  if (reservoir != nullptr && tissue.scalars->free_mass != 0u &&
      reservoir->overflow_mass != UINT32_MAX) {
    if (reservoir->overflow_mass == 0u)
      reservoir->overflow_binding = challenger_identity;
    --tissue.scalars->free_mass;
    ++reservoir->overflow_mass;
  }
  return nullptr;
}

__device__ inline SparseBindingSynapse* find_or_claim_synapse(TissueView tissue,
                                                              std::uint32_t source,
                                                              std::uint32_t target,
                                                              std::uint64_t required_followup_mass = 0u) {
  const std::uint32_t begin = binding_hash(source, target) % tissue.synapse_capacity;
  const std::uint32_t probe_limit = tissue.synapse_capacity < kMaximumSynapseProbe
                                        ? tissue.synapse_capacity
                                        : kMaximumSynapseProbe;
  // Recognition first, over the WHOLE bounded window: a vacancy ahead of an
  // already-resident relation must not be mistaken for that relation's absence.
  // Claiming at the first vacancy without looking further would lay a SECOND
  // record for a key that is still resident behind the hole -- two records for
  // one relation, its evidence split across both.
  SparseBindingSynapse* vacancy = nullptr;
  for (std::uint32_t probe = 0u; probe < probe_limit; ++probe) {
    SparseBindingSynapse* synapse = tissue.synapses + (begin + probe) % tissue.synapse_capacity;
    if (synapse->claimed == 0u) {
      if (vacancy == nullptr)
        vacancy = synapse;
      continue;
    }
    if (synapse->source_cell == source && synapse->target_cell == target)
      return synapse;
  }
  // Not resident. The earliest vacancy in probe order takes the claim, so a
  // vacated slot is reused exactly where the old early-exit would have claimed.
  if (vacancy != nullptr) {
    if (tissue.scalars->free_mass < 1u + required_followup_mass)
      return nullptr;
    vacancy->source_cell = source;
    vacancy->target_cell = target;
    vacancy->claimed = 1u;
    vacancy->structure_mass = 1u;
    --tissue.scalars->free_mass;
    ++tissue.scalars->occupied_synapses;
    return vacancy;
  }
  return apply_synapse_turnover_pressure(
      tissue, begin, probe_limit, source, target, required_followup_mass);
}

// Cooperative equivalent of find_or_claim_synapse. Exactly one warp calls
// this function in lockstep. Every lane inspects one slot of the bounded probe
// window; lane zero alone performs the mutation selected by the lowest
// candidate lane. Like the scalar routine it recognises across the WHOLE
// bounded window before it claims, so a vacancy ahead of an already-resident
// relation cannot be mistaken for that relation's absence and lay a duplicate
// record. With no vacancy anywhere in the chain the two paths take the same
// slot in the same order, and the table anatomy, overflow residue and mass
// movement remain byte for byte identical.
__device__ inline SparseBindingSynapse* find_or_claim_synapse_warp(
    TissueView tissue, std::uint32_t source, std::uint32_t target,
    std::uint64_t required_followup_mass = 0u) {
  const std::uint32_t lane = threadIdx.x & 31u;
  const unsigned active = __activemask();
  const std::uint32_t begin = binding_hash(source, target) % tissue.synapse_capacity;
  const std::uint32_t probe_limit = tissue.synapse_capacity < kMaximumSynapseProbe
                                        ? tissue.synapse_capacity
                                        : kMaximumSynapseProbe;
  std::uint32_t selected_index = UINT32_MAX;
  bool resident = false;
  bool vacant = false;
  if (lane < probe_limit) {
    const SparseBindingSynapse& synapse =
        tissue.synapses[(begin + lane) % tissue.synapse_capacity];
    resident = synapse.claimed != 0u && synapse.source_cell == source &&
               synapse.target_cell == target;
    vacant = synapse.claimed == 0u;
  }
  const unsigned residents = __ballot_sync(active, resident);
  if (residents != 0u) {
    const std::uint32_t resident_probe =
        static_cast<std::uint32_t>(__ffs(static_cast<int>(residents))) - 1u;
    return tissue.synapses + (begin + resident_probe) % tissue.synapse_capacity;
  }
  const unsigned vacancies = __ballot_sync(active, vacant);
  if (lane == 0u) {
    if (vacancies == 0u) {
      SparseBindingSynapse* selected = apply_synapse_turnover_pressure(
          tissue, begin, probe_limit, source, target, required_followup_mass);
      if (selected != nullptr)
        selected_index = static_cast<std::uint32_t>(selected - tissue.synapses);
    } else {
      const std::uint32_t selected_probe =
          static_cast<std::uint32_t>(__ffs(static_cast<int>(vacancies))) - 1u;
      selected_index = (begin + selected_probe) % tissue.synapse_capacity;
      SparseBindingSynapse* synapse = tissue.synapses + selected_index;
      if (tissue.scalars->free_mass < 1u + required_followup_mass) {
        selected_index = UINT32_MAX;
      } else {
        synapse->source_cell = source;
        synapse->target_cell = target;
        synapse->claimed = 1u;
        synapse->structure_mass = 1u;
        --tissue.scalars->free_mass;
        ++tissue.scalars->occupied_synapses;
      }
    }
  }
  __syncwarp(active);
  selected_index = __shfl_sync(active, selected_index, 0);
  return selected_index == UINT32_MAX ? nullptr : tissue.synapses + selected_index;
}

__device__ inline PopulationCohortEvidence* find_or_claim_cohort(
    TissueView tissue, SparseBindingSynapse* synapse,
    SparsePopulationView population) {
  TissueScalars* scalars = tissue.scalars;
  PopulationCohortEvidence* cohorts = synapse_cohorts(tissue, synapse);
  if (cohorts == nullptr)
    return nullptr;
  PopulationCohortEvidence* empty = nullptr;
  for (std::uint32_t index = 0u; index < kMaximumCohortsPerSynapse; ++index) {
    PopulationCohortEvidence* cohort = cohorts + index;
    if (same_cohort(*cohort, population))
      return cohort;
    if (cohort->claimed == 0u && empty == nullptr)
      empty = cohort;
  }
  const std::uint64_t required_mass = 1u + population.count;
  if (empty == nullptr || scalars->free_mass < required_mass) {
    if (scalars->free_mass != 0u && synapse->cohort_overflow_mass != UINT32_MAX) {
      --scalars->free_mass;
      ++synapse->cohort_overflow_mass;
    }
    return nullptr;
  }
  empty->claimed = 1u;
  empty->cell_count = population.count;
  for (std::uint32_t index = 0u; index < population.count; ++index)
    empty->cells[index] = population.cells[index];
  empty->structure_mass = required_mass;
  scalars->free_mass -= required_mass;
  return empty;
}

__device__ inline void move_support_mass(TissueScalars* scalars, PopulationCohortEvidence* cohort,
                                         bool intervention, std::uint32_t quantum) {
  std::uint64_t* destination =
      intervention ? &cohort->intervention_support : &cohort->observational_support;
  for (std::uint32_t unit = 0u; unit < quantum; ++unit) {
    if (cohort->counterevidence != 0u)
      --cohort->counterevidence;
    else if (scalars->free_mass != 0u)
      --scalars->free_mass;
    else
      return;
    ++*destination;
  }
}

[[nodiscard]] __device__ inline std::uint32_t move_support_mass_repeated(
    TissueScalars* scalars, PopulationCohortEvidence* cohort, bool intervention,
    std::uint32_t quantum, std::uint32_t repetitions) {
  std::uint64_t* destination =
      intervention ? &cohort->intervention_support : &cohort->observational_support;
  const std::uint64_t requested =
      static_cast<std::uint64_t>(quantum) * repetitions;
  const std::uint64_t available = cohort->counterevidence + scalars->free_mass;
  const std::uint64_t moved = requested < available ? requested : available;
  const std::uint64_t cancelled =
      cohort->counterevidence < moved ? cohort->counterevidence : moved;
  cohort->counterevidence -= cancelled;
  scalars->free_mass -= moved - cancelled;
  *destination += moved;
  return moved == 0u
             ? 0u
             : static_cast<std::uint32_t>((moved + quantum - 1u) / quantum);
}

__device__ inline void move_counter_mass(TissueScalars* scalars, PopulationCohortEvidence* cohort,
                                         std::uint32_t quantum) {
  for (std::uint32_t unit = 0u; unit < quantum; ++unit) {
    if (cohort->intervention_support != 0u)
      --cohort->intervention_support;
    else if (cohort->observational_support != 0u)
      --cohort->observational_support;
    else if (scalars->free_mass != 0u)
      --scalars->free_mass;
    else
      return;
    ++cohort->counterevidence;
  }
}

[[nodiscard]] __device__ inline std::uint32_t move_counter_mass_repeated(
    TissueScalars* scalars, PopulationCohortEvidence* cohort,
    std::uint32_t quantum, std::uint32_t repetitions) {
  const std::uint64_t requested =
      static_cast<std::uint64_t>(quantum) * repetitions;
  const std::uint64_t available = cohort->intervention_support +
                                  cohort->observational_support +
                                  scalars->free_mass;
  const std::uint64_t moved = requested < available ? requested : available;
  std::uint64_t remaining = moved;
  const std::uint64_t intervention_cancelled =
      cohort->intervention_support < remaining ? cohort->intervention_support
                                               : remaining;
  cohort->intervention_support -= intervention_cancelled;
  remaining -= intervention_cancelled;
  const std::uint64_t observation_cancelled =
      cohort->observational_support < remaining ? cohort->observational_support
                                                : remaining;
  cohort->observational_support -= observation_cancelled;
  remaining -= observation_cancelled;
  scalars->free_mass -= remaining;
  cohort->counterevidence += moved;
  return moved == 0u
             ? 0u
             : static_cast<std::uint32_t>((moved + quantum - 1u) / quantum);
}

// A resident relation that stops earning contact must stop costing capacity
// forever, but this tissue has no clock and no age field: the only time it
// knows is a CONTACT EVENT. A "drive" is one such event. Sparing is decided
// by the tissue's OWN mark -- SparseBindingSynapse::contacted_since_drain,
// set by assimilate_experience the instant it credits a synapse and cleared
// right here -- never by a caller-supplied list. That mark being cleared
// every time this drain inspects the synapse is the whole mechanism: a
// synapse contacted during THIS drive is spared for THIS drive only: the
// next drive it goes uncontacted, it decays like anything else. No host
// authority names who is spared; only having been contacted does.
//
// Every claimed synapse whose mark is clear forfeits exactly one
// kObservationQuantum of positive support -- the same unit
// assimilate_experience itself grants for one accepted observation -- back to
// the free pool that funds new admissions. This makes forgetting paid for by
// traffic that bypassed a relation, not by elapsed wall time.

// One evidence quantum of positive support returned to the free pool.
// Observational support -- the weaker, more generalized kind -- drains first;
// intervention support (concordant efferent evidence) drains only once
// observational is already exhausted. This is an exact swap, never a
// destruction: whatever leaves the cohort lands in scalars->free_mass in the
// same call. A cohort already at zero positive support is left untouched and
// reported as not drained -- there is no floor to cross below zero.
__device__ inline bool drain_one_evidence_quantum(TissueScalars* scalars,
                                                   PopulationCohortEvidence* cohort) {
  if (cohort->observational_support >= kObservationQuantum) {
    cohort->observational_support -= kObservationQuantum;
  } else if (cohort->intervention_support >= kObservationQuantum) {
    cohort->intervention_support -= kObservationQuantum;
  } else {
    return false;
  }
  scalars->free_mass += kObservationQuantum;
  return true;
}

// The maintenance kernels assign each synapse to exactly one worker. The
// cohort mutation is therefore private to that worker; only conserved tissue
// mass and aggregate receipts need atomics. Keeping this path separate from
// the scalar helper preserves the serial reference drive.
__device__ inline bool drain_one_evidence_quantum_parallel(
    TissueScalars* scalars, PopulationCohortEvidence* cohort) {
  if (cohort->observational_support >= kObservationQuantum) {
    cohort->observational_support -= kObservationQuantum;
  } else if (cohort->intervention_support >= kObservationQuantum) {
    cohort->intervention_support -= kObservationQuantum;
  } else {
    return false;
  }
  atomicAdd(reinterpret_cast<unsigned long long*>(&scalars->free_mass),
            static_cast<unsigned long long>(kObservationQuantum));
  return true;
}

// The drain itself: every claimed synapse is inspected exactly once. If its
// own contacted_since_drain mark is set, it is spared -- the mark is cleared
// and nothing is drained, because sparing is good for THIS drive only. If the
// mark is clear, each of its claimed cohorts is drained by one evidence
// quantum. No capacity ceiling or magic threshold is introduced: the scan
// bound is tissue.synapse_capacity, resident state the tissue already
// carries, and the quantum drained is kObservationQuantum, the same constant
// assimilate_experience already spends per accepted observation.
__device__ inline std::uint32_t apply_unreinforced_support_drain(TissueView tissue) {
  if (tissue.synapses == nullptr || tissue.scalars == nullptr || tissue.cohorts == nullptr)
    return 0u;
  std::uint32_t drained_relations = 0u;
  for (std::uint32_t index = 0u; index < tissue.synapse_capacity; ++index) {
    SparseBindingSynapse* synapse = tissue.synapses + index;
    if (synapse->claimed == 0u)
      continue;
    if (synapse->contacted_since_drain != 0u) {
      synapse->contacted_since_drain = 0u;
      continue;
    }
    PopulationCohortEvidence* cohorts = synapse_cohorts(tissue, synapse);
    if (cohorts == nullptr)
      continue;
    for (std::uint32_t cohort_index = 0u; cohort_index < kMaximumCohortsPerSynapse;
         ++cohort_index) {
      PopulationCohortEvidence* cohort = cohorts + cohort_index;
      if (cohort->claimed == 0u)
        continue;
      if (drain_one_evidence_quantum(tissue.scalars, cohort))
        ++drained_relations;
    }
  }
  return drained_relations;
}

__global__ void apply_unreinforced_support_drain_kernel(TissueView tissue,
                                                        std::uint32_t* drained_relations) {
  // One cooperative block strip-mines the resident synapse capacity. The
  // caller must launch one block; each worker owns disjoint synapses.
  __shared__ std::uint32_t drained;
  if (threadIdx.x == 0u) {
    drained = 0u;
    if (drained_relations != nullptr)
      *drained_relations = 0u;
  }
  __syncthreads();
  if (tissue.synapses == nullptr || tissue.scalars == nullptr || tissue.cohorts == nullptr)
    return;
  for (std::uint32_t index = threadIdx.x; index < tissue.synapse_capacity;
       index += blockDim.x) {
    SparseBindingSynapse* synapse = tissue.synapses + index;
    if (synapse->claimed == 0u)
      continue;
    if (synapse->contacted_since_drain != 0u) {
      synapse->contacted_since_drain = 0u;
      continue;
    }
    PopulationCohortEvidence* cohorts = synapse_cohorts(tissue, synapse);
    if (cohorts == nullptr)
      continue;
    for (std::uint32_t cohort_index = 0u; cohort_index < kMaximumCohortsPerSynapse;
         ++cohort_index) {
      PopulationCohortEvidence* cohort = cohorts + cohort_index;
      if (cohort->claimed != 0u && drain_one_evidence_quantum_parallel(tissue.scalars, cohort))
        atomicAdd(&drained, 1u);
    }
  }
  __syncthreads();
  if (threadIdx.x == 0u && drained_relations != nullptr)
    *drained_relations = drained;
}

// One reversible fact this drive changed: either a synapse's contacted_since_drain mark was
// cleared (cohort_index == UINT32_MAX, sparing this drive only) or one cohort's positive support
// forfeited exactly kObservationQuantum (cohort_index names it, was_intervention names which
// field). Recording BOTH kinds of change is what makes the undo below exact: a mark clear moves
// no mass but is still a real mutation of tissue state, and skipping it would leave undo unable to
// reproduce the complete pre-drive tissue hash even though every quantum of mass came back.
struct DrainJournalEntry {
  std::uint32_t synapse_index = UINT32_MAX;
  std::uint32_t cohort_index = UINT32_MAX;
  bool was_intervention = false;
};

__device__ inline void append_drain_journal_entry(
    DrainJournalEntry* journal, std::uint32_t journal_capacity,
    std::uint32_t* journal_count, std::uint32_t* journal_overflow,
    DrainJournalEntry entry) {
  if (journal == nullptr)
    return;
  while (true) {
    const std::uint32_t current = *journal_count;
    if (current >= journal_capacity) {
      atomicExch(journal_overflow, 1u);
      return;
    }
    if (atomicCAS(journal_count, current, current + 1u) == current) {
      journal[current] = entry;
      return;
    }
  }
}

// Same scan and same mutations as apply_unreinforced_support_drain (that function is untouched;
// every one of its other callers is unaffected), but appends one DrainJournalEntry per change to a
// caller-owned buffer. journal may be nullptr to opt out (then this behaves exactly like the
// unjournaled function). A bounded buffer is a normal CUDA-kernel pattern; running out never
// silently drops an entry -- *journal_overflow is set and the caller must not trust replay to be
// exact against an overflowed run.
__device__ inline std::uint32_t apply_unreinforced_support_drain_with_journal(
    TissueView tissue, DrainJournalEntry* journal, std::uint32_t journal_capacity,
    std::uint32_t* journal_count_out, std::uint32_t* journal_overflow_out) {
  std::uint32_t journal_count = 0u;
  bool overflow = false;
  if (tissue.synapses == nullptr || tissue.scalars == nullptr || tissue.cohorts == nullptr) {
    if (journal_count_out != nullptr)
      *journal_count_out = 0u;
    if (journal_overflow_out != nullptr)
      *journal_overflow_out = 0u;
    return 0u;
  }
  std::uint32_t drained_relations = 0u;
  for (std::uint32_t index = 0u; index < tissue.synapse_capacity; ++index) {
    SparseBindingSynapse* synapse = tissue.synapses + index;
    if (synapse->claimed == 0u)
      continue;
    if (synapse->contacted_since_drain != 0u) {
      synapse->contacted_since_drain = 0u;
      if (journal != nullptr) {
        if (journal_count < journal_capacity)
          journal[journal_count++] = DrainJournalEntry{index, UINT32_MAX, false};
        else
          overflow = true;
      }
      continue;
    }
    PopulationCohortEvidence* cohorts = synapse_cohorts(tissue, synapse);
    if (cohorts == nullptr)
      continue;
    for (std::uint32_t cohort_index = 0u; cohort_index < kMaximumCohortsPerSynapse;
         ++cohort_index) {
      PopulationCohortEvidence* cohort = cohorts + cohort_index;
      if (cohort->claimed == 0u)
        continue;
      // Read the SAME condition drain_one_evidence_quantum itself branches on, before calling it,
      // so the journal records which field it actually took -- the function's return value alone
      // does not say which.
      const bool drains_observational = cohort->observational_support >= kObservationQuantum;
      if (drain_one_evidence_quantum(tissue.scalars, cohort)) {
        ++drained_relations;
        if (journal != nullptr) {
          if (journal_count < journal_capacity)
            journal[journal_count++] =
                DrainJournalEntry{index, cohort_index, !drains_observational};
          else
            overflow = true;
        }
      }
    }
  }
  if (journal_count_out != nullptr)
    *journal_count_out = journal_count;
  if (journal_overflow_out != nullptr)
    *journal_overflow_out = overflow ? 1u : 0u;
  return drained_relations;
}

// The exact reverse of one apply_unreinforced_support_drain_with_journal call: every entry is
// undone (mark restored, or the drained quantum credited back and free_mass debited), replayed in
// reverse order for LIFO symmetry (the individual mutations are disjoint per synapse/cohort, so
// order does not change the RESULT, only the convention). Fails closed -- returns false and stops
// -- rather than silently apply a partial undo against a stale or mismatched journal.
__device__ inline bool undo_unreinforced_support_drain(TissueView tissue,
                                                        const DrainJournalEntry* journal,
                                                        std::uint32_t journal_count) {
  if (tissue.synapses == nullptr || tissue.scalars == nullptr || journal == nullptr)
    return false;
  for (std::uint32_t remaining = journal_count; remaining-- > 0;) {
    const DrainJournalEntry& entry = journal[remaining];
    if (entry.synapse_index >= tissue.synapse_capacity)
      return false;
    SparseBindingSynapse* synapse = tissue.synapses + entry.synapse_index;
    if (synapse->claimed == 0u)
      return false;
    if (entry.cohort_index == UINT32_MAX) {
      synapse->contacted_since_drain = 1u;
      continue;
    }
    if (entry.cohort_index >= kMaximumCohortsPerSynapse)
      return false;
    PopulationCohortEvidence* cohort = synapse_cohorts(tissue, synapse) + entry.cohort_index;
    if (cohort->claimed == 0u || tissue.scalars->free_mass < kObservationQuantum)
      return false;
    if (entry.was_intervention)
      cohort->intervention_support += kObservationQuantum;
    else
      cohort->observational_support += kObservationQuantum;
    tissue.scalars->free_mass -= kObservationQuantum;
  }
  return true;
}

__global__ void apply_unreinforced_support_drain_with_journal_kernel(
    TissueView tissue, DrainJournalEntry* journal, std::uint32_t journal_capacity,
    std::uint32_t* drained_relations, std::uint32_t* journal_count_out,
    std::uint32_t* journal_overflow_out) {
  __shared__ std::uint32_t drained;
  __shared__ std::uint32_t journal_count;
  __shared__ std::uint32_t journal_overflow;
  if (threadIdx.x == 0u) {
    drained = 0u;
    journal_count = 0u;
    journal_overflow = 0u;
    if (drained_relations != nullptr)
      *drained_relations = 0u;
    if (journal_count_out != nullptr)
      *journal_count_out = 0u;
    if (journal_overflow_out != nullptr)
      *journal_overflow_out = 0u;
  }
  __syncthreads();
  if (tissue.synapses == nullptr || tissue.scalars == nullptr || tissue.cohorts == nullptr)
    return;
  for (std::uint32_t index = threadIdx.x; index < tissue.synapse_capacity;
       index += blockDim.x) {
    SparseBindingSynapse* synapse = tissue.synapses + index;
    if (synapse->claimed == 0u)
      continue;
    if (synapse->contacted_since_drain != 0u) {
      synapse->contacted_since_drain = 0u;
      append_drain_journal_entry(journal, journal_capacity, &journal_count, &journal_overflow,
                                 DrainJournalEntry{index, UINT32_MAX, false});
      continue;
    }
    PopulationCohortEvidence* cohorts = synapse_cohorts(tissue, synapse);
    if (cohorts == nullptr)
      continue;
    for (std::uint32_t cohort_index = 0u; cohort_index < kMaximumCohortsPerSynapse;
         ++cohort_index) {
      PopulationCohortEvidence* cohort = cohorts + cohort_index;
      if (cohort->claimed == 0u)
        continue;
      const bool drains_observational = cohort->observational_support >= kObservationQuantum;
      if (drain_one_evidence_quantum_parallel(tissue.scalars, cohort)) {
        atomicAdd(&drained, 1u);
        append_drain_journal_entry(
            journal, journal_capacity, &journal_count, &journal_overflow,
            DrainJournalEntry{index, cohort_index, !drains_observational});
      }
    }
  }
  __syncthreads();
  if (threadIdx.x == 0u) {
    if (drained_relations != nullptr)
      *drained_relations = drained;
    if (journal_count_out != nullptr)
      *journal_count_out = journal_count;
    if (journal_overflow_out != nullptr)
      *journal_overflow_out = journal_overflow;
  }
}

__global__ void undo_unreinforced_support_drain_kernel(TissueView tissue,
                                                        const DrainJournalEntry* journal,
                                                        std::uint32_t journal_count,
                                                        std::uint32_t* undone) {
  __shared__ std::uint32_t valid;
  __shared__ std::uint64_t required_mass;
  if (threadIdx.x == 0u) {
    valid = 1u;
    required_mass = 0u;
    if (undone != nullptr)
      *undone = 0u;
  }
  __syncthreads();

  const bool globally_usable = tissue.synapses != nullptr && tissue.scalars != nullptr &&
                               journal != nullptr && undone != nullptr;
  if (!globally_usable)
    atomicExch(&valid, 0u);
  if (globally_usable) {
    for (std::uint32_t index = threadIdx.x; index < journal_count; index += blockDim.x) {
      const DrainJournalEntry& entry = journal[index];
      if (entry.synapse_index >= tissue.synapse_capacity) {
        atomicExch(&valid, 0u);
        continue;
      }
      SparseBindingSynapse* synapse = tissue.synapses + entry.synapse_index;
      if (synapse->claimed == 0u) {
        atomicExch(&valid, 0u);
        continue;
      }
      if (entry.cohort_index == UINT32_MAX)
        continue;
      if (entry.cohort_index >= kMaximumCohortsPerSynapse || tissue.cohorts == nullptr) {
        atomicExch(&valid, 0u);
        continue;
      }
      PopulationCohortEvidence* cohorts = synapse_cohorts(tissue, synapse);
      if (cohorts == nullptr || cohorts[entry.cohort_index].claimed == 0u) {
        atomicExch(&valid, 0u);
        continue;
      }
      for (std::uint32_t prior = 0u; prior < index; ++prior) {
        const DrainJournalEntry& previous = journal[prior];
        if (previous.synapse_index == entry.synapse_index &&
            previous.cohort_index == entry.cohort_index) {
          atomicExch(&valid, 0u);
          break;
        }
      }
      atomicAdd(reinterpret_cast<unsigned long long*>(&required_mass),
                static_cast<unsigned long long>(kObservationQuantum));
    }
  }
  __syncthreads();
  if (threadIdx.x == 0u && valid != 0u &&
      tissue.scalars->free_mass < required_mass)
    valid = 0u;
  __syncthreads();
  if (valid == 0u)
    return;

  for (std::uint32_t index = threadIdx.x; index < journal_count; index += blockDim.x) {
    const DrainJournalEntry& entry = journal[index];
    SparseBindingSynapse* synapse = tissue.synapses + entry.synapse_index;
    if (entry.cohort_index == UINT32_MAX) {
      synapse->contacted_since_drain = 1u;
      continue;
    }
    PopulationCohortEvidence* cohort = synapse_cohorts(tissue, synapse) + entry.cohort_index;
    if (entry.was_intervention)
      cohort->intervention_support += kObservationQuantum;
    else
      cohort->observational_support += kObservationQuantum;
    atomicAdd(reinterpret_cast<unsigned long long*>(&tissue.scalars->free_mass),
              0ull - static_cast<unsigned long long>(kObservationQuantum));
  }
  __syncthreads();
  if (threadIdx.x == 0u)
    *undone = 1u;
}

__global__ void initialize_tissue_kernel(TissueView tissue, std::uint64_t initial_mass) {
  if (tissue.synapses == nullptr || tissue.scalars == nullptr || tissue.cohorts == nullptr ||
      tissue.synapse_capacity == 0u || tissue.cell_capacity == 0u ||
      tissue.cohort_capacity < tissue.synapse_capacity * kMaximumCohortsPerSynapse)
    return;
  const std::uint32_t thread_index = blockIdx.x * blockDim.x + threadIdx.x;
  const std::uint32_t thread_stride = blockDim.x * gridDim.x;
  for (std::uint32_t index = thread_index; index < tissue.synapse_capacity;
       index += thread_stride)
    tissue.synapses[index] = SparseBindingSynapse{};
  for (std::uint32_t index = thread_index; index < tissue.cohort_capacity;
       index += thread_stride)
    tissue.cohorts[index] = PopulationCohortEvidence{};
  if (tissue.ordered_bindings != nullptr)
    for (std::uint32_t index = thread_index; index < tissue.ordered_binding_capacity;
         index += thread_stride)
      tissue.ordered_bindings[index] = OrderedRoleBindingEvidence{};
  if (thread_index == 0u) {
    if (tissue.ordered_binding_admission != nullptr) {
      *tissue.ordered_binding_admission = OrderedBindingAdmissionState{};
      tissue.ordered_binding_admission->projected_bytes =
          static_cast<std::uint64_t>(tissue.ordered_binding_capacity) *
          sizeof(OrderedRoleBindingEvidence);
    }
    *tissue.scalars = TissueScalars{};
    tissue.scalars->initial_mass = initial_mass;
    tissue.scalars->free_mass = initial_mass;
  }
}

#include "bcc32_cuda_resident_proposition_tissue_tail.inl"
