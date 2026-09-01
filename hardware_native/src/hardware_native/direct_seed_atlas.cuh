#ifndef HARDWARE_NATIVE_DIRECT_SEED_ATLAS_CUH
#define HARDWARE_NATIVE_DIRECT_SEED_ATLAS_CUH

#include "hardware_native/direct_network_genome.cuh"

namespace substrate::direct_network {

// The atlas -> Gamma translation layer (github #1353).
//
// docs/architecture/human_connectome_to_gamma_seed_atlas_v0.md specifies
// NET00..NET18, each with a `gamma_seed:` block naming its territories and its
// growth terms. Until this file, NOTHING in the tree translated any of it:
// `grep -rn '\.opcode = DirectRuleOpcodeV1::' hardware_native/src/` returned
// zero, so every genome in existence was hand-authored inside a test fixture
// for that test's own purposes and no network specification authored anything.
//
// This is the first producer. It seeds a network because the atlas says what
// that network is, not because a contract needed a fixture -- which is the
// whole distinction, since the atlas is the documented prerequisite for
// language and had no executable path at all.
//
// Deliberately ONE network per function and no registry: the atlas is nineteen
// networks long and scaffolding all of them before one grows is the exact
// drift this exists to stop. An enum arrives when there is a third.

// NET00_TERMINAL_NATIVE_SURFACE_INTERFACE.
//
// Its three atlas territories, in the atlas's own order:
//   0  fast_surface_temporal_binders
//   1  short_horizon_successor_predecessor_ecology
//   2  expression_surface_motor_binders
//
// Nothing here is derived from a byte, a character, an alphabet size or a
// symbol table, which is what the atlas's `kill:` block forbids:
//   - semantic_language_requires_host_tokenizer
//   - symbol_identity_preloaded_as_meaning
// The genome's size is a function of the motif (3 territories), never of an
// alphabet, and `cuda_direct_seed_atlas_net00_contract` asserts that rather
// than trusting this comment.
DirectGenomeV1 seed_atlas_net00_terminal_surface();

// NET01_DORSAL_PERISYLVIAN_STREAM (github #1268).
//
// Its three atlas territories, in the atlas's own order:
//   0  posterior_temporal_sequence_territory
//   1  inferior_parietal_interface
//   2  inferior_frontal_premotor_sequence_territory
//
// The atlas requires the DIRECT route (posterior_temporal <-> inferior_frontal)
// and the INDIRECT route (posterior_temporal <-> inferior_parietal <-> inferior
// frontal) to be "intentionally parallel" -- alternative paths between the SAME
// endpoints, not a fork that leaves posterior_temporal for two destinations
// that never rejoin (github #1268/82bef40571) -- and names
// `recurrent_bidirectional_coupling` as its own motif term, writing every
// growth term with `<->`, not `->`. Six corridor families are authored to
// match, all six directed legs of the three `<->` pairs: posterior_temporal
// owns two (direct, to inferior_frontal; indirect first hop, to
// inferior_parietal), inferior_parietal owns two (the indirect route's second
// hop, to inferior_frontal, `51ba5cea71`; its own back-projection, to
// posterior_temporal), and inferior_frontal owns two back-projections of its
// own (to posterior_temporal and to inferior_parietal). Each is an ordinary
// long_tract rule carrying its own field (`3ddf739421`'s "a long_tract rule
// is a corridor family"); without an authored field a territory's long tracts
// are steered by geometry against wherever Xi happened to place its
// neighbours, an accident of one individual rather than a species corridor --
// `cuda_direct_net01_route_convergence_contract` measures the distinction
// directly (an authored corridor's size is a genome constant across Xi, an
// accidental one is not).
//
// The three back-projections are left at the pure geometric delay rather than
// the forward indirect hops' authored transport cost -- nothing in the atlas
// or in prior measurement yet distinguishes a forward leg's delay from a
// backward one's; that asymmetry question stays open for a future contract.
//
// One asymmetry IS now authored: inferior_parietal's own back-projection to
// posterior_temporal recruits only nodes born after a developmental tick
// (`tract_maturation`, `627e280ce5`) -- the first family in the tree to
// author a real maturation window, on the reasoning that feedback/recurrent
// corridors typically stabilize later than the feedforward pathways they
// trail. The other two back-projections author no window.
DirectGenomeV1 seed_atlas_net01_dorsal_stream();

// NET02_LANGUAGE_VENTRAL_TEMPOROFRONTAL_STREAM (github #1268).
//
// Its three atlas territories, read off the motif in the atlas's own order:
//   0  temporal_integration (posterior_middle_anterior_temporal_integration)
//   1  multimodal_grounding_convergence
//   2  inferior_frontal_context_control
//
// The atlas's growth term `multiple_parallel_ventral_tract_corridors` wants
// temporal_integration to carry two independent authored affinities at once:
// the direct leg (`crossmodal_territory_affinity`'s primary leg, toward
// inferior_frontal_context_control) and a parallel leg toward multimodal_
// grounding_convergence. Both are authored here as two ordinary long_tract
// rules, each carrying its own field -- `3ddf739421`'s "a long_tract rule is a
// corridor family" landing, applied to NET02 the same way it retrofitted
// NET01's own two dorsal legs.
DirectGenomeV1 seed_atlas_net02_ventral_stream();

// NET12_DIFFUSE_NEUROMODULATORY_MULTI_CHANNEL_ECOLOGY (github #1276).
//
// Two independently-lesionable "modulatory relation families", the atlas's
// `multiple_modulatory_relation_families` term, each authored with its own
// `distinct_spatial_reach` and its own `local_receptor_like_recipe_
// compatibility` -- a long_tract field whose require_value matches only ONE
// target territory's chemistry, so channel_a's corridor cannot land on
// channel_b's target and vice versa. Four territories: channel_a, channel_b
// (the two modulatory sources) and target_x, target_y (the two territories
// each channel's corridor is compatible with).
//
// `distinct_decay` (github #1276 rung 2): channel_a's corridor field decays
// toward zero over gestation ticks, channel_b's does not, via a decay CLASS
// packed into the same legacy `polarity` word `field_kind` already uses
// (`direct_network_life_function.cu`'s `field_decay_class`/
// `field_decayed_strength_q16`) rather than widening `FieldBlock`, the
// BCC-shared POD -- see that function's own comment. `channel_a_decays`
// selects between the real (fast decay) and reference (no decay, matching
// this seed's pre-rung-2 behavior exactly) configurations so a caller can
// measure the causal difference rather than assume it; the public no-arg
// caller always gets the real, decaying configuration.
//
// `effect_vector` over GAIN (github #1276 rung 3): both channels' corridor
// fields now opt into `field_gain_participates`
// (`direct_network_life_function.cu`), a bit packed above the decay-class
// digit in the same `polarity` word. When set, the field's own influence
// score at the tick a long-tract corridor forms (which already carries that
// field's decay, via `field_decayed_strength_q16`) biases the INITIAL
// `route.conductance_q16` of the corridors it wins, instead of every
// long-tract route landing on the flat kInitialConductanceQ16 regardless of
// which field chose it -- the same score-derived-conductance formula the
// sparse-degree path already used, now also reachable from the long-tract
// path for opted-in fields only. `conductance_q16` is not a cosmetic
// field: it is the exact per-route GAIN the executor's own activation
// kernel reads (`mul_q16(route.conductance_q16, src_node.activation_q16)`
// in `direct_adult_core.cu`). Every field authored before this rung has
// polarity < kDevelopmentFieldKindCount * kFieldDecayClassCount, so
// `field_gain_participates` is false for all of them and this is a strict
// no-op for every existing NET00-NET11/NET17 long-tract corridor.
//
// WHAT THIS STILL DOES NOT AUTHOR. No effect vector over PLASTICITY, search
// or retention exists anywhere in the executor (grep for plasticity_threshold,
// search_pressure, retention_bias all still return zero) -- that remains open
// for a future rung. What IS authored and measured here: two channels, grown
// on a real device, whose corridors are causally separable by
// `direct_network_tract_lesion`'s channel_specific_lesions -- lesioning
// channel_a's corridor removes target_x's incoming connectivity and leaves
// target_y's untouched, and symmetrically for channel_b/target_y -- whose
// decay timescales are causally separable by comparing the decaying and
// reference configurations' final corridor counts, AND whose GAIN is now
// causally separable the same way: the decaying configuration's surviving
// target_x corridors measure a lower mean conductance_q16 than the
// non-decaying reference's, while target_y's stays unchanged between the two.
// See cuda_direct_seed_atlas_net12_device_contract.cu.
DirectGenomeV1 seed_atlas_net12_modulatory_channels(bool channel_a_decays = true);

// NET04_RAPID_BINDING_INDEX_ANTERIOR_POSTERIOR (github #1269).
//
// The atlas's `anterior_posterior_connectivity_differentiation` term: the
// medial-temporal index is not one uniform structure that binds everything to
// everything.  Its two segments carry DIFFERENT cortical connectivity, and
// that difference is anatomy, so it is the half of NET04 this ABI can actually
// author.  Four territories: `anterior` and `posterior` (the two index
// segments) plus `transmodal` and `sensory` (the two cortical partners each
// segment is receptor-compatible with).  Same DirectFieldSpecV1 chemistry-
// matched corridor mechanism NET01/NET02/NET12 already proved -- anterior's
// long_tract field require_value names ONLY transmodal's chemistry, posterior's
// names ONLY sensory's, so neither segment's corridor can land on the other's
// partner.  Anterior carries the broader reach and posterior the narrower one,
// the conservative reading of the biological gradient.
//
// `rapid episode index distinct from slower cortical consolidation` (github
// #1269, timescale rung): posterior's corridor field now decays toward zero
// over gestation ticks, anterior's does not, via the same decay CLASS packed
// into the legacy `polarity` word that NET12 rung 2 proved on a real device
// (`direct_network_life_function.cu`'s `field_decay_class`/
// `field_decayed_strength_q16`, 08c38a9803) -- no new ABI field, no new
// opcode.  Posterior carries the narrower, finer-grained reach into sensory
// cortex, the detail-preserving half of this seed's own anterior/posterior
// gradient, so it is cast as the fast, rapidly-fading index; anterior
// carries the broad reach into transmodal/associative cortex, the partner
// most plausibly doing the slower consolidation this index feeds, so it
// stays persistent (decay class 0).  `posterior_decays` selects between the
// real (fast decay) and reference (no decay, matching this seed's
// pre-timescale-rung behavior exactly) configurations so a caller can
// measure the causal difference rather than assume it; the public default
// always grows the real, decaying configuration.
//
// WHAT THIS STILL DOES NOT AUTHOR, and why each is a gap rather than an
// omission.  `fast one/few-contact binding`, `partial-cue reinstatement /
// completion` and `high interference separation` are DYNAMICS, not structure
// or timescale: they are claims about how quickly and how separably NEW
// relations FORM under contact -- a plasticity rate or binding threshold --
// and decay alone does not author formation.  Nothing in DirectRuleSpecV1 or
// DirectFieldSpecV1 carries either.  `recurrence: high` has no carrier
// either -- measured, not assumed: DirectRuleOpcodeV1 is exactly {extend,
// branch, fuse, retract, mature, repair, long_tract, endogenous_source},
// with no recurrent or closed-loop opcode, and choose_long_tract_target
// skips p == source_plan so a territory cannot address itself.
//
// The named ports (`surface_form`, `source`, `place/body_state`,
// `object/relation`, `temporal_context`, `cortical_recipe_reference`) are
// deliberately NOT authored as named things.  Authoring a port called
// `surface_form` would be exactly the semantic endpoint the architectural rule
// forbids; the seed reserves port capacity and lets contact decide what
// occupies it.  `shared_global_state: forbidden` is likewise not an authored
// term -- it is a property of the substrate, and asserting it here would
// restate the executor rather than test it.
//
// What IS authored and measured: two index segments, grown on a real device,
// each reaching only its own named cortical partner, each independently
// cuttable without touching the other, AND now each with its own
// independently measurable decay timescale -- posterior's decaying corridor
// count strictly below its non-decaying reference's, anterior's unchanged
// between the two.  See cuda_direct_seed_atlas_net04_device_contract.cu.
DirectGenomeV1 seed_atlas_net04_binding_index(bool posterior_decays = true);

// NET08_THALAMOCORTICAL_RELAYS_AND_CONNECTOR (github #1270).
//
// The atlas asks that thalamus be neither one passive relay nor one executive
// switch: `network_specific_relays` and `connector_subdivisions` must coexist,
// with `preserve_parallel_subdivisions` holding between them.  That is a claim
// about projection topology, so it is authorable, and it is a real test rather
// than a restatement -- an integrative hub is exactly where specificity is
// most likely to collapse into a universal sink.
//
// Five territories: `relay_a`, `relay_b` and `connector` (three thalamic
// subdivisions) plus `cortex_x` and `cortex_y` (the two cortical partner
// families).  The two relays are network-specific -- each one's long_tract
// corridor field carries a FULL require_mask naming exactly one cortical
// chemistry.  The connector is the same mechanism with a RELAXED mask: since
// chemistry_matches() is `(chemistry & require_mask) == require_value`, a mask
// that ignores the low bit of the two adjacent cortical chemistries matches
// BOTH of them, while still matching neither relay nor the connector itself
// (so field_addresses_partner() still reads it as addressing a partner).  The
// breadth of a subdivision's partner set is therefore authored by how much of
// the chemistry it insists on -- no new ABI field, no new opcode, and no
// per-subdivision code path.
//
// `explicit_delay_phase`'s DELAY half IS now authored: each subdivision's
// `long_tract.extent` carries its own transport cost -- the two relays FAST
// (point-to-point, network-specific), the connector SLOWER (broader,
// integrative) -- the same carrier b0a747d01a proved. See arm 6/7.
//
// WHAT THIS DOES NOT AUTHOR. The PHASE half of `explicit_delay_phase` still
// has no carrier: `extent` is one scalar per subdivision, not a phase, and
// nothing anywhere carries an occurrence-local phase. `plasticity: medium`
// has no rate field -- the same gap #1276 and #1269 both
// stop at.  `fanout: high_but_bounded` is authored only as far as
// branch_count reaches; the BOUND is a resource-ecology question owned by
// #1178, not a term this seed can express.  `partner_sets:
// developmentally_biased_not_fixed_slots` is honoured in the sense that
// nothing here names a slot -- the partner set falls out of chemistry
// matching -- but this seed does not demonstrate that the bias can be revised
// during a lifetime, which is the Delta-rho half of the ticket.
//
// No subdivision contains a semantic route name, and none is authored: the
// territories are named for their topology, never for what might travel them.
// See cuda_direct_seed_atlas_net08_device_contract.cu.
DirectGenomeV1 seed_atlas_net08_thalamic_relays();

// NET10_REPLICATED_MICROZONES (github #1272).
//
// The ticket's worker packet is explicit: "Do not author 688 semantic patches.
// Implement a replicated developmental generator", and "First decisive
// contract: use TWO domains with the same generic replicated family."  So the
// authored object here is a GENERATOR, not a patch list: kNet10MicrozoneCount
// microzone territories are emitted by ONE loop that differs across units in
// exactly one place -- which partner domain each unit's corridor field names.
// Everything else (reach, degree, population, port reserve, tract count) is
// the shared generic local motif, written once.
//
// Two partner domains, `domain_a` and `domain_b`, and the microzones alternate
// between them by index.  That alternation is the atlas's
// `fractured_noncontiguous_partner_topography_allowed`: index-adjacent
// microzones serve DIFFERENT domains, and the units serving one domain are
// noncontiguous in the generator's own ordering, so the partner map is not a
// contiguous strip that could be mistaken for one block with a shared edge.
// `duplicated_maps_allowed` follows from the same construction -- several
// distinct units carry the same domain.
//
// `precise_delay_timing` IS NOW AUTHORED.  Each unit already grows its OWN
// long_tract rule -- its own corridor family, one per unit -- so each one's
// `extent` (the transport-cost carrier b0a747d01a proved, generalized past a
// two-family binary by NET03/d792ebead0 and NET05/b8a5a0b30b) carries a value
// distinct from every other unit's: `kMicrozoneDelayBase + index *
// kMicrozoneDelayStep`, a monotonic per-unit sequence rather than one shared
// scalar. See cuda_direct_seed_atlas_net10_device_contract.cu arm 6.
//
// WHAT THIS DOES NOT AUTHOR. `local_error_correction`, `cheap_predictive_mapping`
// and the expected-residual/precision work belong to #1291/#1289 and are
// dynamics with no field in this ABI.  `resource_scaled_count` is not authored:
// the unit count is a seed constant, and scaling it by available matter is
// #1178's question.  `recursive_branching_allowed` is not exercised -- the
// generator emits a flat replicated set, not a recursively branched one.
//
// No unit carries a patch type, a map name or a semantic function.  Units are
// distinguished only by index and by which domain their corridor names.  See
// cuda_direct_seed_atlas_net10_device_contract.cu.
DirectGenomeV1 seed_atlas_net10_replicated_microzones();

// How many microzones the generator above emits.  Exposed so a probe can
// assert the replication is COMPLETE rather than assuming it, without
// hard-coding the count in a second place.
std::uint32_t seed_atlas_net10_microzone_count();

// Which partner domain territory ordinal microzone `index` is authored to
// reach.  Exposed for the same reason: a probe must be able to ask the seed
// what it authored rather than re-deriving the alternation and testing its own
// arithmetic against itself.
std::uint32_t seed_atlas_net10_partner_ordinal(std::uint32_t index);

// NET05_ASSOCIATION_GRADIENT and its FLATTER SIBLING (github #1274).
//
// The ticket's worker packet asks for a "first decisive sibling-Gamma test":
// a graded body->association->transmodal seed against a RESOURCE-MATCHED
// flatter homogeneous one, same body, and a measurement of what the grading
// buys.  The behavioural half of that comparison (crossmodal grounding, long-
// context disambiguation, language grounding) is dynamics with no carrier
// here.  The STRUCTURAL half is decidable now, and it is the half that has to
// hold first: does the graded seed actually author a different connectivity
// than the flat one under an identical budget?
//
// Three territories in both siblings: `sensory` (body-tethered edge),
// `heteromodal` (the association middle) and `transmodal` (the far edge).
// Chemistries are chosen so heteromodal and transmodal are ADJACENT, which
// lets one bit of one mask be the entire difference between the two genomes:
//
//   graded  sensory corridor require_mask = 0xffffffff  -> heteromodal ONLY
//   flat    sensory corridor require_mask = 0xfffffffe  -> heteromodal AND
//                                                          transmodal
//
// Everything else -- territory count, reach, degree, population, long-tract
// budget, port reserve, development seed, the heteromodal->transmodal corridor
// -- is identical between the two, so the siblings are resource-matched by
// construction rather than by a claim.  The graded genome therefore grows a
// CHAIN (sensory reaches the middle, the middle reaches the far edge, and the
// far edge is not reachable from the body edge in one hop) while the flat one
// grows the shortcut as well.
//
// ⚠ WHY THE SIBLING IS LOAD-BEARING.  On its own, "sensory -> transmodal is
// empty" in the graded seed is worthless: an empty corridor is exactly what a
// lattice that never put the two territories in reach would also produce.  The
// flat sibling grows that corridor from the same budget and the same
// development seed, so the absence in the graded organism is AUTHORED rather
// than merely unobserved.  No sham can supply this control -- only a genome
// that differs in the one property under test.
//
// WHAT THIS DOES NOT AUTHOR.  The atlas is explicit that the gradient is a
// "connectivity/timescale/convergence" gradient; connectivity plus the
// `delays: short -> long` third are now expressible: each authored hop's
// `long_tract` rule carries its own transport-cost `extent` (the carrier
// proven by NET03/b0a747d01a), so sensory->heteromodal costs strictly less
// than heteromodal->transmodal in BOTH siblings (identical on both, since
// both are emitted by net05_seed() -- resource-matched by construction, same
// as the connectivity mask).  `timescales: medium` (a per-rank node
// persistence/state timescale, not a corridor transport cost) and `long
// recurrence` still have no carrier: the delay is one scalar per corridor
// FAMILY, not a per-rank node property, and `recurrence` additionally needs a
// recurrent opcode, which DirectRuleOpcodeV1 has none of.  `endogenous
// reconstruction/prediction`, `revision_readiness` and `weak dependence on
// immediate surface state` are dynamics.  The atlas also notes the gradient
// "coexists with direct cross-rank connections"; this seed authors none,
// which is a choice about THIS seed and not a claim that the substrate
// forbids them -- the flat sibling demonstrates exactly that they are
// available.
//
// Neither sibling maps a rank to a derivation depth, which the packet
// explicitly forbids: nothing here mentions p(n), and the three territories
// differ only in chemistry and in which partner their corridor names.  See
// cuda_direct_seed_atlas_net05_device_contract.cu.
DirectGenomeV1 seed_atlas_net05_association_gradient();

// The resource-matched flatter sibling.  Byte-identical to the graded genome
// except for the one mask that widens the sensory corridor's partner set.
DirectGenomeV1 seed_atlas_net05_flat_sibling();

// NET13_14_18_BACKBONE_TRACT_CLASSES (github #1275).
//
// The ticket's worker packet names a "first structural proof" for the
// backbone: grow a language long tract, a cortico-subcortical loop tract, a
// cerebro-cerebellar return tract and a commissural/homologous connection
// "through the same generic construction machinery", and treat human tract
// names as observer metadata rather than as runtime dispatch keys.
//
// So the authored object is ONE TABLE of corridor priors and one loop over it.
// It started at four tract classes and eight territories; a thalamocortical
// radiation and a memory-context (cingulum-like) corridor -- the remaining two
// classes NET18 (WHITE_MATTER_TRACT_GROWTH_PALETTE) names -- are now the
// table's fifth and sixth row and its ninth through twelfth territory, added
// with no other edit anywhere.  Not a single branch anywhere on which class a
// row belongs to -- the rows differ only in which chemistry each side names
// and in whether the partner names the source back.  The tract names live in
// a `provenance` string that no rule, field or kernel reads; deleting every
// one of them would change nothing the organism grows, which is the packet's
// requirement stated as a property rather than as an intention.
//
// ⭐ THE COMMISSURAL ROW IS THE DISCRIMINATOR.  Three of the four classes are
// one-way: the source's corridor field names the partner and the partner names
// nothing back.  The commissural row is reciprocal -- BOTH homologues carry a
// corridor field naming the other -- and reciprocity is therefore an authored
// property of a row rather than a property of the machinery.  A generic
// mechanism that could only produce one of these two shapes would not be
// generic, and the contract measures both shapes in one organism: the reverse
// direction of each one-way class is asserted EMPTY while the commissural
// reverse is asserted LIVE.
//
// `delay/bandwidth distribution` IS now authored: every row carries its own
// `long_tract.extent` (one per class, the commissural pair's two legs sharing
// its row's value), the same transport-cost carrier NET03/NET05/NET10 already
// prove -- see arm 7.  It remains a per-corridor-FAMILY scalar, not a
// per-route resource penalty.
//
// WHAT THIS DOES NOT AUTHOR.  The packet's corridor ABI wish-list is longer
// than what exists.  `route length/resource penalty` still has no term -- a
// long tract's BYTE/MATTER cost is whatever geometry charges regardless of
// class, and `local_edge_cost: low` versus `long_tract_cost: high` is not
// expressible, so the rest of the `construction_economy` block remains
// unauthored; that is #1178's ledger, not a seed constant.  `maturation window` per corridor
// is not authored here (begin/end ticks exist on the rule but are not varied
// across classes, because varying them without a way to MEASURE a maturation
// order would be decoration).  `branch/fusion permissions` per corridor have
// no field.  `asymmetry_seed: weak_bias_only` is not authored: the two
// homologues are exact mirrors, so this seed demonstrates the coupling and not
// the specialization.  `construction ancestry/source refs` is #1180's.
//
// The obstruction-and-lawful-rerouting half of the packet's proof is NOT
// attempted -- it needs an environment that can forbid a region, and
// [the environment can make a region expensive, never unavailable] is a
// standing gap owned elsewhere.  See
// cuda_direct_seed_atlas_net13_device_contract.cu.
DirectGenomeV1 seed_atlas_net13_backbone_tracts();

// How many corridor rows the table above holds, and the two ordinals of row
// `index`.  Exposed so the probe enumerates the seed's own table instead of
// restating it -- a contract that hard-codes the pairs would pass by agreeing
// with itself.
std::uint32_t seed_atlas_net13_tract_count();
std::uint32_t seed_atlas_net13_tract_source(std::uint32_t index);
std::uint32_t seed_atlas_net13_tract_partner(std::uint32_t index);
// Whether row `index` was authored as a reciprocal (commissural) pair.
bool seed_atlas_net13_tract_is_reciprocal(std::uint32_t index);
// Row `index`'s own authored construction-economy transport cost.
std::uint32_t seed_atlas_net13_tract_delay(std::uint32_t index);

// NET06_ADAPTIVE_AND_NET07_STABLE_CONTROL (github #1273).
//
// The ticket's worker packet asks for two control phenotypes with "materially
// distinct parameter distributions" -- NET06 frontoparietal-like with faster
// reconfiguration, high cross-network port diversity and high revision
// readiness, NET07 cingulo-opercular-like with slower recurrent persistence
// and stable set maintenance -- and forbids `ExecutiveMode::FAST/SLOW` in the
// hot path.
//
// 🔴->✅ THIS WAS THE SEED THE TIMESCALE GAP ACTUALLY BLOCKED, and it should
// still be said plainly even though the gap has since closed.  The
// distinguishing property of NET06 versus NET07 in the packet is SPEED and
// PERSISTENCE -- "faster reconfiguration" against "slower recurrent
// persistence" -- and when this seed was first authored (`5278aea4ed`) there
// was no timescale, rate, decay or persistence carrier anywhere in
// DirectRuleSpecV1 or DirectFieldSpecV1; six seeds had recorded that absence.
// `field_decay_class` (#1276/NET12, `08c38a9803`) closed it 49 minutes later:
// a per-field decay-timescale digit packed into `DirectFieldSpecV1.polarity`,
// orthogonal to `DevelopmentFieldKind`.  This seed now authors it directly --
// the adaptive controller's field decays (class 2 -- measured, not guessed:
// the fastest class collapses arm 1's own breadth measurement to a tie, the
// slowest leaves arm 7's decay measurement unmoved), the stable controller's
// stays PERMANENT (class 0, never decays) -- so the field itself
// now outlives or fades exactly as the packet's language describes, not just
// its reach.  `revision_readiness` and `resistance to transient disturbance`
// as full corridor-formation-rate effects remain unauthored; this closes the
// decay/persistence half of the axis, not the whole packet.
//
// What IS authorable is the packet's other axis, and it is not a consolation
// prize: `high cross-network port diversity` versus `stable context/set
// maintenance` is a claim about PARTNER BREADTH, which chemistry masks carry
// exactly.  The adaptive controller's corridor field relaxes two bits and so
// names a whole block of network partners; the stable controller's insists on
// the full chemistry and names exactly one.  Same mechanism, same budget, and
// no mode enum anywhere -- breadth is a property of a mask, not of a branch.
//
// ⭐ AND THE PACKET NAMES ITS OWN CONTROL: a "flattened sibling Γ".  The
// flattened sibling here gives BOTH controllers the stable controller's narrow
// mask, so the two phenotypes become indistinguishable in breadth while every
// other constant, the development seed and the body binding stay identical.
// Without it, "the adaptive controller reaches four partners and the stable
// one reaches one" could be a fact about where the lattice happened to put
// them; with it, the difference is attributable to the mask alone.
// ⚠ This is the BREADTH analogue of the packet's flattened sibling, NOT a
// flattened-TIMESCALE sibling -- the two controllers' decay classes stay
// distinct (fast vs. permanent) even in the flattened sibling, since
// flattening here is specifically about the mask, not the decay digit.
//
// See cuda_direct_seed_atlas_net06_device_contract.cu.
DirectGenomeV1 seed_atlas_net06_net07_control();

// The flattened sibling: both controllers narrow, everything else identical.
DirectGenomeV1 seed_atlas_net06_net07_flattened_sibling();

// The seed's own view of its layout, so the probe enumerates rather than
// restates it: how many network partners exist, and the two controller
// ordinals.
std::uint32_t seed_atlas_net06_partner_count();
std::uint32_t seed_atlas_net06_partner_ordinal(std::uint32_t index);
std::uint32_t seed_atlas_net06_adaptive_ordinal();
std::uint32_t seed_atlas_net06_stable_ordinal();

// NET03_TEMPORAL_BOUNDARY_AND_MULTI_TIMESCALE_SEQUENCE_ECOLOGY (github #1268).
//
// Its three territories, read off the motif's own gradient in the atlas's own
// order: local_fast_sequence_state, medium_phrase_state,
// slow_discourse_recurrence.
//
// `recurrence_time_constant_gradient`: reach and degree move MONOTONICALLY
// across the three -- local is tight and dense (fast, high-fanout local
// recurrence, `fuse`-authored), discourse is broad and sparse (slow, wide
// integration, no dense fuse at all).
//
// `broad_delay_distribution`: a two-hop chain, local -> medium -> discourse,
// each hop a distinct corridor family (`3ddf739421`) carrying its OWN
// transport cost (`b0a747d01a`'s `long_tract` rule extent -- the same
// mechanism NET01's `fast_medium` uses). The medium->discourse cost is set
// strictly larger than local->medium's, so the distribution is graded across
// three regimes rather than NET01's binary fast/medium.
//
// `boundary_sensitive_but_not_boundary_dependent_plasticity` is NOT authored.
// It names a LEARNING dynamic -- state that responds to a boundary without a
// boundary becoming a hard terminal -- and this ABI has no plasticity-rate or
// boundary-sensitivity field on either DirectRuleSpecV1 or DirectFieldSpecV1.
// `DirectRuleSpecV1::minimum_age`/`maximum_age` are the nearest candidate
// carriers and are declared but read by nothing in the Life Function today
// (a pre-existing gap, not one this seed introduces). Naming the gap here
// rather than a rule that would silently claim it.
DirectGenomeV1 seed_atlas_net03_temporal_boundary_ecology();

// NET15_SENSORY_MOTOR_GROUNDING_STREAMS (github #1267).
//
// Three territories:
//   0  sensor_entry          modality_specific_entry, the port body events
//                            arrive through
//   1  object_identity       the "what" stream -- object_identity_relations
//   2  action_affordance     the "how/where" stream -- action_affordance_
//                            relations, body_reafference
//
// crossmodal_binding_corridors is authored as four ordinary corridor fields,
// bidirectional per the atlas's own recurrent_sensorimotor_loops human_basis
// term: sensor_entry forks a named corridor to each stream, and each stream
// loops one back -- the same construction NET01's back-projections used, one
// hop further down a different motif.
//
// separate_what_vs_how_where_processing_biases is authored as a difference in
// GROWTH BIAS rather than in corridor topology, the same precedent NET03's
// three-regime timescale gradient set: object_identity gets a broad,
// low-fanout reach (integrative, convergent, the shape NET02's
// temporal_integration and NET05's heteromodal territories already use);
// action_affordance gets NET00's own strong_local_recurrence shape (tight
// reach, high fanout), realizing action_adjacent_recurrent_territories
// literally rather than by name only. Population, dense width and long-tract
// count match the species table's own uniform NET15 row so a reader comparing
// the generic and the dedicated genome finds them disagreeing only on the
// per-territory bias the table's one row per family cannot express.
//
// NOT authored: sensor_adjacent_topographic_territories_if_body_has_geometry.
// The atlas's own gamma_seed names this term conditionally -- the current
// test body (a BoundaryPortBinding pair, no spatial extent) has no geometry
// for a topographic territory to be adjacent to, so this term does not apply
// to this body rather than being an authoring gap.
//
// Deliberately not derived from a byte, symbol or text co-occurrence, per the
// atlas's own kill: all_grounding_is_text_cooccurrence.
DirectGenomeV1 seed_atlas_net15_grounding_streams();

// NET14_BILATERAL_COMMISSURAL_DUPLICATION_AND_LATERALIZATION (github #1267).
//
// Two territories, both grown from the SAME symmetric baseline reach/degree
// (32/8, the species table's own uniform NET14 row): homologue_a (ordinal 0)
// and homologue_b (ordinal 1) -- generic ordinals, not "left"/"right", per
// the atlas's own kill: hardcode_left_equals_language_truth.
//
// strong_commissural_growth_corridors is authored as an ordinary bidirectional
// corridor pair, the same construction NET15's crossmodal binding uses.
//
// asymmetry_parameter_seed_is_weak_bias_not_semantic_assignment is authored
// as a ONE-TENTH-of-baseline degree bias on whichever homologue the caller
// names favored, via `homologue_a_favored`. Calling with the argument flipped
// realizes the atlas's own mirrored_Gamma_asymmetry_swap probe directly: the
// favored side moves, proving the bias is a parameter of the call rather than
// an identity baked into either territory -- `force_perfect_symmetry` is
// avoided (the bias is real and measurable) without ever hardcoding which
// ordinal wins (`redundancy_without_exact_duplication`: both homologues still
// grow real, comparable capacity either way).
DirectGenomeV1 seed_atlas_net14_bilateral_homologues(bool homologue_a_favored = true);

// The canonical species (github #1357).
//
//   NO LANGUAGE HACKS.  GROW THE SPECIES.  THEN TEACH THE ORGANISM.
//
// One content-blank Gamma covering every atlas family, compiled to GENERIC
// DEVELOPMENTAL CAUSES. There is no per-family code path, no enum of network
// types and no field anywhere that names a network: each family is a ROW of
// scalars the Life Function already consumes -- reach, degree, population,
// dense width, long-tract count, developmental window, coordinate -- and one
// loop compiles every row identically. The family name appears only in a
// provenance comment beside its row.
//
// That distinction is the whole ticket. A species whose networks were structs
// would be nineteen authored outcomes; a species whose networks are rows in a
// growth-parameter table is a seeded developmental landscape, and which of them
// ends up mattering for anything is the organism's to settle.
//
// CORRECTION (github #1357/#1268, 2026-08-19): this table's rows author no
// field, so THIS TABLE cannot express corridor/affinity terms -- but that is a
// fact about the table, not about the ABI. `DirectFieldSpecV1` (a territory-
// relative attract field, target-chemistry-matched, landed 007ac93107) is a
// real, declarative authoring path for `A<->C long_tract_affinity` and
// `crossmodal_territory_affinity`, and `3ddf739421` closed the remaining gap:
// a `long_tract` RULE is a corridor family, so one territory can author
// several parallel corridors, each to its own named partner -- proven end to
// end by seed_atlas_net01_dorsal_stream() and seed_atlas_net02_ventral_stream()
// above, both of which author two parallel legs from one source territory.
// `reach` remains the only lever over territory PLACEMENT, since
// DirectTerritorySpecV1 has no coordinate field and every origin is
// lattice-derived -- but placement was never the mechanism affinity needs.
// `b0a747d01a` closed the remaining delay gap this note used to name: a
// `long_tract` rule's `extent` (previously unread by that opcode) is now that
// corridor family's authored transport cost, added to the geometric delay --
// so two parallel corridors from the same source CAN carry different delays,
// proven by NET01's own `fast_medium` (`49622fd6c1`) and NET03's three-regime
// gradient above; `627e280ce5` closed tract_maturation, whose `minimum_age` is
// now a corridor family's maturation window. What remains genuinely
// unauthorable (gh #1367 made this list executable): route_priority, bandwidth.
//
// This seeds a species. It does not claim the species behaves: C0/C1/C2, C3
// motif signatures and C4 interactions all need a real device.
DirectGenomeV1 seed_atlas_canonical_species();

// Host-side atlas authoring variant used by the magnitude contract. The three
// fields are deliberately orthogonal: disposition, authored Q16 strength, and
// strength presence. In particular, {true, 0, true} means an explicit zero,
// while {true, 0, false} preserves the unauthored 32768 fallback.
struct DirectAtlasCompetitionSpecV1 {
  bool competition_enabled;
  std::uint32_t competition;
  bool competition_magnitude_authored;
};

DirectGenomeV1 seed_atlas_canonical_species(DirectAtlasCompetitionSpecV1 competition);

// NET09's three parallel cortico-striato-pallido-thalamo-cortical selection
// loops: twelve territories, four stations per channel, each station's own
// corridor family addressed to the next station's chemistry and the last
// wrapping back to the first. The atlas's `output_returns_to_cortex_via_
// thalamus` is that wrap; no loop opcode or cycle descriptor exists or was
// added. Channels share no chemistry, which is the structural form of
// `no_single_global_winner`. Anatomy only -- nothing here selects an action.
DirectGenomeV1 seed_atlas_net09_selection_loops();

// NET17_HOMEOSTATIC_BRAINSTEM_HYPOTHALAMIC_BODY_CONTROL (github #1276).
//
// Two independently-lesionable body-tethered loops: `short_latency_
// protective_loops` (narrow reach, pure geometric delay) and `slow_
// allostatic_state` (broad reach, authored transport cost on top) -- the
// atlas's own fast/slow timescale split, the same `DirectFieldSpecV1`
// chemistry-matched corridor mechanism NET01/NET02/NET09/NET12 already
// proved. Structural anatomy only: `dedicated_body_vitality_damage_resource_
// channels`, modulatory links, and the delta_r learning terms are NOT
// authored -- no such state exists anywhere in the executor yet (see the
// seed function's own comment in direct_seed_atlas.cu).
DirectGenomeV1 seed_atlas_net17_homeostatic_loops();

// NET11_SALIENCE_INTEROCEPTIVE_SWITCH_ECOLOGY (github #1276).
//
// The atlas's `distinct_from_world_truth` term, made concrete: TWO
// independently-lesionable corridor families rather than one global
// attention scalar. `salience_source` (chemistry-gated on the atlas's
// `high_access_to_body_consequence_channels` / `high_access_to_prediction_
// mismatch` terms) reaches only `salience_target` (the atlas's
// `recruit_reconfiguration` partner); `world_source` (ordinary perceptual
// evidence) reaches only `world_target`, an ordinary world-evidence
// territory, on DISTINCT chemistry -- the same `DirectFieldSpecV1`
// chemistry-matched corridor mechanism NET01/NET02/NET09/NET12/NET17
// already proved. Lesioning the salience family leaves the world-evidence
// family's corridor untouched and vice versa: salience cannot become truth
// merely by being adjacent to it, because there is no shared route it could
// borrow. `broad_but_bounded_cross_network_ports` is authored as the
// salience family's reach: wider than the ordinary world-evidence family's,
// but a finite territory reach, not the near-unbounded affinity radius a
// `global_attention_scalar` would need.
//
// WHAT THIS DOES NOT AUTHOR. Every `delta_r` term
// (`context_specific_salience_relation`, `threat_or_novelty_
// overgeneralization_split`, `source_conditioned_relevance`) has no ABI
// carrier -- no relation-revision mechanism reachable from seed authoring
// exists yet. `repeated_consequence_pattern -> compact_trigger_recipe`
// (n_plus_1) needs a condensation mechanism this seed layer cannot reach.
// `lesion_salience_ecology_vs_frontoparietal` needs the control family
// #1273 owns (cross-issue, out of scope here); `spoofed_endogenous_
// surprise_cannot_gain_world_evidence` needs a learning/evidence-weighting
// mechanism that does not exist. This seed and its contract prove the
// anatomy only: two named corridor families, grown on a real device, each
// reaching only its own named partner, independently cuttable without
// touching the other. See cuda_direct_seed_atlas_net11_device_contract.cu.
DirectGenomeV1 seed_atlas_net11_salience_switch_ecology();

// NET16_AMYGDALA_LIMBIC_SOCIAL_AFFECTIVE_LOOPS (github #1267).
//
// Three territories: source (the body-consequence hub), fast_target
// (NARROW/immediate) and slow_target (BROAD/integrative, the atlas's
// transmodal_slow_path). Unlike NET17's two wholly separate loops, NET16's
// atlas language centers on a SINGLE hub, so source forks two independently-
// authored, independently-lesionable corridor fields -- the same fork-from-
// one-territory construction NET15's crossmodal binding uses.
// high_source_context_binding is the hub's own higher degree. The fast leg
// stays at the pure geometric delay; the slow leg's long_tract extent adds
// an authored transport cost, realizing the atlas's own amygdala_like_fast_
// path_lesion_vs_transmodal_slow_path probe directly (independently
// cuttable, measurably different delay). source_partner_conditioning,
// hippocampal/prefrontal interaction, extinction_and_reversal and self_vs_
// other_agency are dynamics, not anatomy, and are NOT authored -- see the
// seed function's own comment in direct_seed_atlas.cu.
DirectGenomeV1 seed_atlas_net16_limbic_loops();

// How many families the canonical species declares, and how many territories
// each contributes. Exposed so a contract can assert coverage against the table
// rather than against a number retyped into the test.
std::uint32_t seed_atlas_family_count();

}  // namespace substrate::direct_network

#endif  // HARDWARE_NATIVE_DIRECT_SEED_ATLAS_CUH
