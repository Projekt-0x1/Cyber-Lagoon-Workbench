#ifndef HARDWARE_NATIVE_DIRECT_NETWORK_BASIN_PROBE_CUH
#define HARDWARE_NATIVE_DIRECT_NETWORK_BASIN_PROBE_CUH

#include "hardware_native/direct_adult_core.cuh"
#include "hardware_native/direct_network_certification.cuh"

// Patch 0006 rung 2 -- t2, processor basins.
//
// The spec: "Use CUDA probe masks over live morphology. Select probe seeds post
// hoc from mature tissue. Run multiple horizons through the exact local-reaction
// law. Require several recurrent return basins and both disjoint and overlapping
// activation populations."
//
// Two words in that paragraph decide the whole design.
//
// EXACT. The probe does not carry its own propagation kernel. A private
// reaction law would measure basins of a dynamics the organism never runs, and
// it would pass whether or not the organism has any. This module drives
// `step_direct_adult_epochs` -- the same executor the adult uses -- so what it
// finds is a property of the organism rather than of the instrument. That is
// why this lives in its own library above both direct_network_life_function and
// direct_adult_core instead of inside the certification translation unit.
//
// POST HOC. Seeds are derived on device from grown state (maturation and the
// node's own live out-degree), never from an authored index list. The
// accompanying contract asserts the difference is real by measuring an
// index-authored sham seed set against the derived one: if maturity were
// uniform the two would coincide and "post hoc" would be a word rather than a
// method.
//
// STRATIFIED, and still post hoc. #1280: ranking the whole organism once put
// five of eight seeds in territory 0, so ten of the 28 pairs the probe compared
// were same-territory by construction and could never have tested territory
// separation -- yet the receipt printed `disjoint_pairs=0 / compared_pairs=28`
// as if all 28 were fair tests. Selection now draws each rank from one observed
// territory in turn, and the pair census is reported split by territory
// relation. Both are instrument repairs: the candidate SET is narrowed, the
// ranking quantity, the tie-break, the activation threshold and the
// `shared == 0` disjointness test are all untouched. The territories are read
// out of grown tissue rather than from the brain header or from index
// arithmetic, because `territory_index` is stamped per node by the genesis seed
// and only the nodes know what was actually grown.
//
// The probe runs on a DEVICE CLONE of the juvenile. Injecting activity and
// stepping epochs is an intervention; performing it on the organism would make
// certification causal, which the receipt forbids of itself. t4 already
// requires cloning for its lesions, so this is the same mechanism one stage
// early. The contract asserts the original's morphology root is unchanged.
namespace substrate::direct_network::basin_probe {

using substrate::direct_adult_core::AdultExecutionConfig;
using certification::NetworkFoundationReceipt;
using certification::StageReceipt;

inline constexpr std::uint32_t kMaxProbeSeeds = 12u;
inline constexpr std::uint32_t kMaxHorizons = 8u;

// "Several" is stated once, here, so an arm cannot quietly redefine it.
inline constexpr std::uint32_t kSeveralBasins = 3u;

// No territory restriction -- never a valid `territory_index` (a uint16), so
// this cannot collide with any grown value. Public so a caller can name a
// specific territory via `BasinProbeConfig::required_territory` below.
inline constexpr std::uint32_t kAnyTerritory = 0xffffffffu;

// WHICH CANDIDATE SET EACH SEED RANK DRAWS FROM.
//
// #1280: the probe compared 28 seed pairs and reported `disjoint_pairs=0` as
// though all 28 were fair tests of territory separation. Five of the eight
// seeds were in territory 0, so ten of those pairs were same-territory BY
// CONSTRUCTION and could never have been disjoint. The stratification mode is
// an explicit measured axis rather than a silent change so the old selection
// stays runnable as the control it now is.
//
// Every mode ranks by the SAME grown quantity and picks by the SAME atomicMax.
// Stratification restricts the candidate set; it never authors an index.
enum class SeedStratification : std::uint32_t {
  // Rank the whole organism once and take the top `seed_count`. What t2
  // measured up to 2026-08-19.
  global = 0u,
  // One seed per observed territory first -- the highest-scoring node WITHIN
  // each -- then fill the remaining slots from the whole organism. This is the
  // literal repair #1280 proposed.
  per_territory_then_global = 1u,
  // Round-robin over the observed territories at every rank, so seed k draws
  // from territory k % territory_count. The only mode whose per-territory seed
  // count cannot be dominated by one territory once the ranks run past the
  // territory count.
  round_robin = 2u,
  // Two seeds per observed territory: seed k draws from territory
  // (k / 2) % territory_count.
  //
  // WHY THIS MODE HAS TO EXIST (gh #1359). Every mode above draws AT MOST one
  // seed per territory while `seed_count <= observed_territory_count`, so on
  // an organism with more territories than seeds `same_territory_pairs` is
  // identically zero -- by arithmetic, not by outcome. The canonical atlas
  // species has 42 observed territories, so at the usual 8 or 12 seeds the
  // within-territory question loses its ENTIRE sample and `disjoint_pairs`
  // silently answers only "does the organism separate territories". That is
  // #1280 in mirror image: there a stratifier that was too weak filled the
  // denominator with pairs that could not separate; here a stratifier that is
  // too strong empties it of the pairs that ask the other question.
  //
  // The pair census already splits the two questions and reports both. This
  // mode is what gives the same-territory half a non-empty sample, so a run
  // can carry BOTH questions at once: seed_count/2 same-territory pairs and
  // the rest cross-territory, from one selection over one organism.
  //
  // It authors no index and changes no ranking. Like every mode above it only
  // narrows the CANDIDATE SET handed to the same atomicMax over the same grown
  // quantity, so the two co-located seeds are the two highest-scoring nodes of
  // that territory rather than two positions chosen by the caller.
  paired_within_territory = 3u,
};

struct BasinProbeConfig {
  std::uint32_t seed_count = 8u;
  std::uint32_t horizon_ticks[kMaxHorizons] = {1u, 2u, 3u, 4u, 6u, 8u, 12u, 16u};
  std::uint32_t horizon_count = 8u;
  // A node counts as in the activation population above this drive. Reported in
  // the result so a reader can see what "active" meant.
  std::int32_t activation_threshold_q16 = (1 << 16) / 8;
  std::uint32_t block_size = 256u;
  // Selects seeds by node index instead of by grown maturity. This is the
  // dose-matched sham for the post-hoc requirement, never a production path.
  bool authored_index_seeds = false;
  // Stratification is ORTHOGONAL to `authored_index_seeds`: the sham keeps
  // whatever stratification the real run used, so the only thing that differs
  // between them is the ranking QUANTITY. A sham that also lost stratification
  // would differ for two reasons at once and could no longer isolate "post hoc
  // from mature tissue".
  SeedStratification stratification = SeedStratification::round_robin;

  // gh #1359: a rare declaring family (NET09's three inhibitory territories,
  // 23 of 42) can sit past every `stratification` mode's reach at any
  // ordinary seed count -- round_robin never revisits a territory once it has
  // stepped past it, so a family that starts at rank 23 is unreachable at
  // seed_count 8 or 12 by arithmetic, not by chance. Raising `kMaxProbeSeeds`
  // to force a collision would grow `pair_shared_matrix`/
  // `reachable_per_territory` quadratically for no reason. This is the
  // cheaper, orthogonal fix: when set to a real territory index, EVERY seed
  // is drawn from exactly that territory instead of whatever `stratification`
  // would otherwise pick -- the two fields are mutually exclusive per run,
  // not layered. Falls back to the whole organism (same rule every other
  // restriction in this file already follows) if the named territory is
  // exhausted before `seed_count` is reached.
  std::uint32_t required_territory = kAnyTerritory;

  // ── BRAKE ABLATIONS ────────────────────────────────────────────────────────
  //
  // Wiring the refractory window and the inhibitory sign together cut the
  // reachable set from 2046 nodes to ~990, but nothing separated their
  // contributions. If either does nothing measurable it is wired but INERT --
  // the same defect class the wiring was meant to close, and it would be
  // invisible behind the other one's effect.
  //
  // The lesions act on the CLONE's grown tissue, not on the reaction law, so
  // they are ordinary dose-matchable interventions rather than a production
  // config that carries debug switches. Refractoriness has no tissue to lesion,
  // so it is ablated through the executor's own `refractory_period_ticks`.
  //
  // `sham_ablation` applies the SAME NUMBER of writes to nodes that do not carry
  // the flag, so a change in outcome cannot be credited to having touched device
  // memory at all.
  // Contacts delivered to each seed, one per tick from tick 1. The default of 1
  // cannot exercise `node_slow_context_q16`: that accumulator only grows on
  // sensory ingestion, one contact adds at most ~4095, and the ceiling that
  // bounds it sits at kQ16One -- so a single-contact probe reports the bound as
  // inert when it has simply never been asked. Raising this is how the assay
  // asks.
  std::uint32_t contacts_per_seed = 1u;

  bool ablate_inhibitory_sign = false;
  bool ablate_inhibition_threshold = false;
  bool sham_ablation = false;
};

// No node in the grown tissue carries kNodeFlagInhibitory.
inline constexpr std::uint32_t kNoInhibitoryTissue = 0xffffffffu;

struct BasinProbeResult {
  std::uint32_t seed_count;
  std::uint32_t horizon_count;
  std::uint32_t seed_node[kMaxProbeSeeds];
  std::uint32_t seed_maturation_q16[kMaxProbeSeeds];
  std::uint32_t horizon_ticks[kMaxHorizons];
  // Population size at each (seed, horizon).
  std::uint32_t population[kMaxProbeSeeds][kMaxHorizons];
  std::uint32_t seed_territory[kMaxProbeSeeds];
  // One bit per horizon: was the seed's OWN node above threshold there.
  std::uint32_t seed_active_mask[kMaxProbeSeeds];
  // A return, not persistence. Set only when the seed node was active at some
  // horizon, INACTIVE at a later one, and active again at a later one still.
  // Drive that merely stays high is forward propagation that never left, and
  // scoring it as recurrence would let a feedforward organism pass t2.
  std::uint32_t recurrent_return[kMaxProbeSeeds];
  std::uint32_t recurrent_basin_count;
  std::uint32_t disjoint_pairs;
  std::uint32_t overlapping_pairs;
  std::uint32_t compared_pairs;
  // ── THE PAIR CENSUS, SPLIT BY TERRITORY RELATION (#1280 A) ────────────────
  //
  // `compared_pairs` alone is a misreadable denominator. A pair of seeds inside
  // ONE territory is not a test of whether the organism separates territories --
  // it is a test of whether one territory splits into two basins, which is a
  // different question with a different answer. Reporting the two together let
  // `disjoint_pairs=0 / compared_pairs=28` read as twenty-eight failed
  // separation tests when ten of them never asked about separation at all.
  //
  // The split changes NOTHING about how a pair is judged disjoint: shared == 0,
  // same threshold, same populations. It only says which question each pair was
  // asking. The requirement below still scores on the TOTALS.
  std::uint32_t same_territory_pairs;
  std::uint32_t same_territory_disjoint;
  std::uint32_t same_territory_overlapping;
  std::uint32_t cross_territory_pairs;
  std::uint32_t cross_territory_disjoint;
  std::uint32_t cross_territory_overlapping;
  // Overlap magnitude restricted to the pairs that actually span two
  // territories. If cross-territory separation is ever going to be attacked in
  // the reaction law, this is the number that has to move -- the all-pairs
  // range is contaminated by within-territory pairs that should overlap.
  std::uint32_t cross_min_pair_shared;
  std::uint32_t cross_max_pair_shared;
  std::uint32_t cross_min_pair_jaccard_q16;
  std::uint32_t cross_max_pair_jaccard_q16;
  std::uint32_t same_min_pair_shared;
  std::uint32_t same_max_pair_shared;
  // The full symmetric shared-node matrix. A range hides whether "341 shared"
  // was one unusual pair or the whole picture; the matrix cannot.
  std::uint32_t pair_shared_matrix[kMaxProbeSeeds][kMaxProbeSeeds];
  // Overlap MAGNITUDE, not just the boolean. "No disjoint pairs" is a very
  // different finding when the least-overlapping pair shares 2 nodes than when
  // it shares 400, and only the magnitude says which repair the next rung needs.
  // Jaccard is reported in Q16 (65536 == identical populations).
  std::uint32_t min_pair_shared;
  std::uint32_t max_pair_shared;
  std::uint32_t min_pair_jaccard_q16;
  std::uint32_t max_pair_jaccard_q16;
  std::uint32_t min_final_population;
  std::uint32_t max_final_population;

  // ── REACHABILITY CENSUS ────────────────────────────────────────────────────
  //
  // The activation population answers "what did fire". This answers "what COULD
  // have", by breadth-first expansion over the active route graph from the same
  // seeds, one hop per tick -- which is the executor's own step, since the
  // transfer kernel applies every active route's contribution immediately and
  // never reads `route.delay`.
  //
  // The comparison is the point. If population == reachable, the reaction law
  // selects nothing and the reachable set is pure topology, so no parameter
  // change can ever alter which nodes participate. If population < reachable,
  // something prunes and the pruning is worth finding.
  std::uint32_t reachable[kMaxProbeSeeds][kMaxHorizons];
  // |union over all horizons of the attributable active sets|. `population` is
  // the INSTANTANEOUS active set at one horizon and `reachable` is cumulative,
  // so comparing those two directly would confound sampling with pruning. This
  // is the cumulative quantity, and it is what `reachable` must be compared to.
  std::uint32_t ever_active[kMaxProbeSeeds];
  // Pairwise structure of the REACHABLE sets at the final horizon. If no pair of
  // reachable sets is disjoint, t2's disjointness requirement cannot be met by
  // any change to the reaction law, and the subject is the route graph.
  // ── WHY THE REACHABLE SETS FALL INTO CLASSES ───────────────────────────────
  //
  // Measured 2026-08-19: reachable sets take exactly three sizes (1837, 2744,
  // 2713) and the grouping does NOT follow territory -- nodes 990/991 reach 1837
  // while 992/993 reach 2744/2713, four adjacent indices in one territory. Long
  // tracts are the only edges that cross territories in this genome, so these
  // fields ask whether owning one is what separates the classes.
  static constexpr std::uint32_t kMaxCensusTerritories = 8u;
  std::uint32_t reachable_per_territory[kMaxProbeSeeds][kMaxCensusTerritories];
  std::uint32_t reachable_territory_count[kMaxProbeSeeds];
  // Long tracts leaving the seed node itself, and long tracts leaving ANY node
  // in its reachable set. The first says whether the seed personally owns a
  // cross-territory edge; the second says whether its reachable set contains one
  // at all.
  std::uint32_t seed_long_tracts[kMaxProbeSeeds];
  std::uint32_t reachable_long_tracts[kMaxProbeSeeds];
  // ── THE DENOMINATORS (#1293) ───────────────────────────────────────────────
  //
  // `reachable_long_tracts` on its own is a bare count. "This seed's reachable
  // set contains 74 long tracts" is unreadable without knowing whether those
  // same nodes carry 200 active routes or 20000 -- one says cross-territory
  // edges dominate the set, the other says they are a rounding error, and the
  // count is identical in both. These are the MATCHED denominators: counted
  // over exactly the same node set, in exactly the same route slice, in the
  // same kernel pass, so the ratio is a fraction of a measured whole rather
  // than of an assumed one.
  //
  // They are also the falsifier for gh #1243. substrate::direct_adult keeps a
  // separate DirectRoute whose flag NAMES match this file's and whose flag BITS
  // do not: its kRouteFlagLongTract is 1u << 0, which is this file's
  // kRouteFlagActive. Gestation sets Active on every route it grows, so a
  // census that reached for the adult constant would report every active route
  // as a long tract -- a clean-looking 100% cross-territory graph, with no
  // symptom anywhere except that the two numbers coincide. Equality here is
  // that bug's SIGNATURE, not a finding, and the contract refuses it.
  std::uint32_t seed_active_routes[kMaxProbeSeeds];
  std::uint32_t reachable_active_routes[kMaxProbeSeeds];
  // Tree-wide over every node's own route slice: what exists to be exposed to.
  // Without this a per-seed exposure cannot be read as large or small, only as
  // larger or smaller than another seed's.
  std::uint32_t measured_active_routes;
  std::uint32_t measured_long_tract_routes;
  // What the grown brain header SAYS it holds, recorded BESIDE the measurement
  // and never in place of it. A header field with no producer reads exactly
  // like a measured one until something counts the tissue and compares.
  std::uint32_t declared_active_route_count;
  std::uint32_t declared_long_tract_count;
  // ── THE INHIBITORY DENOMINATOR (#1300) ─────────────────────────────────────
  //
  // `inhibitory_nodes`/`inhibitory_routes` below are a TREE-WIDE census with no
  // denominator anywhere -- #1293's own shape, unmeasured for this quantity.
  // "The inhibitory sign moves 2.5% of the reachable set" is unreadable without
  // knowing whether the routes it touches are 5% of what a reachable set
  // contains or 50%; the two demand opposite readings of that 2.5%. These reuse
  // `measured_active_routes`/`reachable_active_routes` above as the matched
  // denominator -- the SAME quantity, not a second one gathered by a second
  // pass that could disagree about which routes were examined -- and add only
  // the inhibitory numerator, counted in the same kernel passes off the same
  // route load.
  //
  // No `inhibitory_route_count` field exists on `DirectBrain` (checked),
  // so unlike the long-tract case there is no declared-header leg to cross-
  // check the tree-wide total against; `measured_inhibitory_routes` stands on
  // its own.
  std::uint32_t seed_inhibitory_routes[kMaxProbeSeeds];
  std::uint32_t reachable_inhibitory_routes[kMaxProbeSeeds];
  std::uint32_t measured_inhibitory_routes;
  std::uint32_t territory_node_count[kMaxCensusTerritories];
  std::uint32_t reachable_disjoint_pairs;
  std::uint32_t reachable_overlapping_pairs;
  std::uint32_t reachable_compared_pairs;
  std::uint32_t min_reachable_pair_shared;
  std::int32_t activation_threshold_q16;
  // The SAME horizons on an unstimulated clone. Without this the population
  // numbers say nothing: measured first run, every seed reported 4096 of 4096
  // nodes active at every horizon, which is indistinguishable between "the
  // injection lit the whole organism" and "the organism was already lit". Every
  // population above is the injected set MINUS this one, so what is reported is
  // drive attributable to the seed rather than the organism's resting state.
  std::uint32_t baseline_population[kMaxHorizons];
  // Population of the injected clone before the baseline is subtracted, kept so
  // a reader can see how much of the raw number the control removed.
  std::uint32_t raw_population[kMaxProbeSeeds][kMaxHorizons];
  // A thresholded population is a LOSSY readout. If every node sits above the
  // threshold at rest, an injection that genuinely perturbs the organism
  // produces an identical population and the probe reports nothing -- blind at
  // the world level, not the bit level. These digests are the positive control
  // that separates the two cases: an exact fold over every node's activation
  // and attractor support, so any state difference at all is visible even when
  // the population is not.
  std::uint64_t baseline_state_digest[kMaxHorizons];
  std::uint64_t seed_state_digest[kMaxProbeSeeds][kMaxHorizons];
  // Seeds whose final-horizon digest differs from the unstimulated baseline.
  std::uint32_t seeds_that_changed_state;
  // Instrument controls. `events_injected` counts accepted ingress calls;
  // `events_ingested_by_law` is what the executor's own metric says it consumed.
  // The baseline pass injects nothing, so its count must be zero -- a non-zero
  // one would mean the two passes were not dose-matched on contact.
  // State as GESTATION left it, before a single epoch is stepped. This is what
  // decides whether a saturated resting readout is something the compiler built
  // or something the executor drives it into, and the two have different owners.
  std::uint32_t birth_population;
  std::int32_t birth_activation_min_q16;
  std::int32_t birth_activation_max_q16;
  std::int32_t birth_attractor_support_min_q16;
  std::int32_t birth_attractor_support_max_q16;
  // Resting census at the final horizon. Names WHY the population readout
  // saturates instead of leaving the next rung to rediscover it.
  std::int32_t baseline_activation_min_q16;
  std::int32_t baseline_activation_max_q16;
  std::int32_t baseline_attractor_support_min_q16;
  std::int32_t baseline_attractor_support_max_q16;
  // Peak slow-context value reached at any node, so an arm can show the ceiling
  // actually binds instead of assuming it.
  std::int32_t max_slow_context_q16;
  std::uint32_t events_injected;
  std::uint64_t events_ingested_by_law;
  std::uint64_t baseline_events_ingested_by_law;
  // Node count, so a reader can tell a saturated baseline from a quiet one.
  std::uint32_t node_count;
  // TERRITORIES AS THE TISSUE HAS THEM. `territory_index` is stamped on a node
  // by its genesis seed (direct_network_life_function.cu:
  // `node.territory_index = plan.seed_index`), so the only authority on how
  // many territories an organism HAS is the grown nodes. `declared_` is the
  // brain header's count, recorded beside it purely as a cross-check: a
  // stratifier that trusted the header would silently stratify over territories
  // no node was ever grown into, and index arithmetic (`index >> 10`) would
  // invent a partition the Life Function never made.
  std::uint32_t observed_territory_count;
  std::uint32_t declared_territory_count;
  // How many distinct territories the chosen seeds actually span. This is the
  // number that makes the pair split legible: with 8 seeds over 4 territories,
  // span 4 is the most a stratifier can buy.
  std::uint32_t seed_territory_span;
  // Largest number of seeds landing in any single territory. Five of eight was
  // the #1280 finding; a stratified selection must bound this at
  // ceil(seed_count / observed_territory_count).
  std::uint32_t max_seeds_in_one_territory;
  // Which territory filter / SeedStratification actually ran.
  std::uint32_t required_territory, stratification;
  // Census of the tissue an ablation has to work with. An ablation of nothing
  // is indistinguishable from an ablation with no effect, so these are reported
  // before any lesion result is read.
  std::uint32_t inhibitory_nodes;
  std::uint32_t inhibitory_routes;
  // gh #1310: the lowest index that grew inhibitory, `kNoInhibitoryTissue` when
  // none did. Node indices are territory-ordered, so this says WHERE the lesion
  // is -- and the sham is drawn from here rather than from index 0, so that a
  // control for "does touching matter matter" is placed in the same
  // neighbourhood as the matter it controls for. Reported, because a sham whose
  // location nobody can read is a control nobody can check.
  std::uint32_t first_inhibitory_node;
  std::uint32_t lesioned_nodes;
  std::uint32_t lesioned_routes;
  // The lowest node index the lesion touched, `kNoInhibitoryTissue` when it
  // touched nothing. For a real ablation this equals `first_inhibitory_node`;
  // for a sham it is what proves the sham was drawn from the same neighbourhood
  // rather than from index 0. A sham that never relocated and a sham that
  // relocated and proved harmless both report "moved nothing" without it.
  std::uint32_t lesion_first_node;
  std::uint32_t clone_pointer_repairs;
  // Pools outside the arena that were deep-copied instead of offset-repaired.
  std::uint32_t clone_pools_deep_copied;
  // Pools the clone could neither offset-repair nor deep-copy, so it aliases
  // the original. Non-zero means the clone is not faithful and the result must
  // not be read as a measurement about an isolated organism.
  std::uint32_t clone_pointers_outside_arena;
  std::uint32_t probe_ran;
};

// Measure t2 on a device clone. Returns probe_ran = 0 if the clone or the
// runtime could not be built; the caller must not read a failed probe as a
// negative result about the organism.
BasinProbeResult probe_basins(const DirectBrain& brain, const BasinProbeConfig& config,
                              const AdultExecutionConfig& execution = {});

// certify_direct_juvenile() plus a real t2 measurement. The t2 stage is scored
// through certification::score_stage, so this module cannot promote a stage by
// a rule of its own; it can only supply the evidence.
NetworkFoundationReceipt certify_direct_juvenile_with_basins(
    const certification::JuvenileReplica* replicas, std::uint32_t replica_count,
    const BasinProbeConfig& config, const AdultExecutionConfig& execution = {},
    BasinProbeResult* out_probe = nullptr);

}  // namespace substrate::direct_network::basin_probe

#endif  // HARDWARE_NATIVE_DIRECT_NETWORK_BASIN_PROBE_CUH
