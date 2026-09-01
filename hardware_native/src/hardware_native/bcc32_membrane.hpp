#pragma once

// Legacy boundary provenance used by the current adult. A descended boundary
// must move raw represented matter without exposing a coordinate, fixed
// receptor, semantic port, or alternate update law; this receipt does not meet
// that requirement and is not its interface.

#include <cstddef>
#include <cstdint>

#include "bcc32_law.cuh"

namespace substrate::bcc32 {

// Transitional receipt used by the current adult transaction sites. `port`
// remains only until the device-owned exterior face replaces these sites. It
// must not be carried into the descended boundary or used by new production
// code.
struct MembraneReceipt {
  std::uint64_t epoch = 0u;
  std::uint32_t port = 0u;
  SiteWord before = 0u;
  SiteWord after = 0u;
  bool conserved = true;
  std::uint64_t transaction = 0u;

  // ⛔ A BATCHED EXCHANGE HAS NO SINGLE PORT, AND `port` MUST NOT PRETEND
  // OTHERWISE. The fourth boundary transaction, GrownAdult::apply_contact_stage,
  // exchanges MANY world/tape bit pairs in one launch. Writing any one index
  // into `port` there would be an observable whose name asserts what its
  // definition does not -- and writing 0 would be indistinguishable from a
  // genuine transaction at port 0. So a batch sets `port = kNoSinglePort` and
  // carries what `port` cannot in these fields.
  //
  //   pairs == 1  -> `port` names the declared boundary port, and the
  //                  fingerprint repeats it.
  //   pairs >  1  -> `port` is kNoSinglePort and MEANINGLESS.
  //
  // 🔴 RENAMED FROM `declaration`, WHICH OVERCLAIMED. It was documented as
  // naming WHICH batch ran. It does not, and the counterexample is systematic
  // rather than a rare accident: the fold was a SUM of a per-descriptor mix that
  // is LINEAR in the slot, so any two batches whose slot sums agree collide.
  // Measured directly -- descriptor slots {0,9} and {1,8} both fold to
  // 9895606726579. An order-independent sum is never injective; the fix is a
  // per-descriptor avalanche before summing, which destroys the linear
  // degeneracy, plus a name that does not assert identity.
  //
  // ⇒ EQUAL FINGERPRINTS DO NOT PROVE EQUAL BATCHES. It detects a changed
  // declaration; it does not certify one.
  std::uint32_t pairs = 1u;
  std::uint64_t declaration_fingerprint = 0u;

  // ⭐ THE FIELD THAT ACTUALLY PROTECTS THE EXCHANGE, as opposed to reporting on
  // it afterwards. Two declared pairs naming the same world word and the same
  // world bit make the exchange kernel's two atomicXors cancel: the world keeps
  // a bit it should have given away and the tape gives one to nobody. A batch
  // with `aliased > 0` is REFUSED -- the kernel is never launched and the world
  // is not touched -- rather than executed and reported as unconserved.
  //
  // ⚠ AND THAT MAKES `conserved` ENTAILED FOR A BATCH, WHICH MUST BE SAID
  // RATHER THAN QUIETLY ENJOYED. The exchange swaps only when the two sides
  // disagree, so every non-aliased batch conserves the pair sum by construction.
  // Once aliasing is refused, `conserved` on a launched batch can no longer read
  // 0 -- it is a consequence of `aliased == 0`, not an independent check. The
  // arm that can fail is the alias arm.
  std::uint32_t aliased = 0u;
};

inline constexpr std::uint32_t kNoSinglePort = 0xFFFFFFFFu;

// ⭐ THE THIRD THING THAT WRITES THE WORLD, NAMED INSTEAD OF SMUGGLED.
//
// tools/audit_mutation_classes.sh measures 89 device-side world-write entry
// points against the two classes below: 4 boundary transactions, 62 canonical
// factor writes, and 23 that are neither -- 14 of them lesion/restore kernels.
// So requirement 2 is FALSE in the tree, and the honest response is not to widen
// the enum until it becomes true.
//
// ⛔ A LESION IS NOT SOMETHING THE ORGANISM DOES. It is an experimenter's
// manipulation, and modelling it as a boundary transaction would let an assay
// tool write the world through the organism's own ingress. It is a PRIVILEGED,
// EPOCH-FORMING INTERVENTION: it is allowed, it is counted, it stamps an epoch
// so a claim can distinguish the unintervened trajectory from the post-lesion
// one, and it is excluded from the two runtime classes rather than added to
// them.
//
// ⚠ WHAT THIS DOES NOT DO, stated so the receipt is not read as more than it is.
// It does not validate a declared write set -- that is requirement 5 and needs
// the device-side write primitive, which is unbuilt. It records WHO intervened,
// WHEN, and HOW MANY TIMES. The kernels keep their own matter-before/after
// receipts; this does not duplicate them.
enum class InterventionReason : std::uint32_t {
  lesion_relation = 0u,
  matched_remote_perturbation = 1u,
  route_ablation = 2u,
  cloud_lesion = 3u,
  situation_lesion = 4u,
  readout_route_lesion = 5u,
  genesis_scatter = 6u,
  // Swapping two grown regions, or a situation's outcome, is a manipulation
  // that asks "is this region the carrier?". Both return a LesionReceipt in
  // their own tissues, which is the tissues agreeing.
  region_exchange = 7u,
  situation_outcome_exchange = 8u,
  // ⚠ ADDED FOR ACCURACY, NOT CONVENIENCE. Converting the test sites turned up
  // manipulations the existing reasons did not describe, and reusing a
  // near-miss would put a false label on the thing requirement 2 counts. The
  // CLASS list is the taxonomy claim and is unchanged; the REASON list just
  // enumerates what actually happens.
  stimulus_presentation = 9u,   // a scene/candidate set written into world sites
  credit_set = 10u,             // a credit value the experimenter chose
  // Deliberate corruption of resident state, used as a control arm: the
  // experimenter breaks something on purpose to show a check notices. It is not
  // a lesion (nothing is removed) and not a stimulus (nothing is presented).
  corrupt_state = 11u,
};

struct InterventionReceipt {
  std::uint64_t epoch = 0u;
  std::uint64_t intervention = 0u;
  InterventionReason reason = InterventionReason::lesion_relation;
};

// ⭐ THE THIRD CLASS, NAMED — AND IT IS NOT AN INTERVENTION.
//
// tools/audit_mutation_classes.sh counts 23 world-write entry points that are
// neither canonical factor writes nor boundary transactions. Some are the
// experimenter's (above). The REST are the organism's own machinery running
// OUTSIDE the canonical tick: a contact arriving through no declared port, a
// trajectory being learned from presented experience, resident context being
// bound to a grounded relation, a motor byte being staged.
//
// ⛔ THESE MUST NOT BE BROKERED AS INTERVENTIONS. Calling the organism's own
// learning an "experimenter manipulation" would put a false label on the thing
// requirement 2 is trying to count, and a mislabelled census is worse than a
// large one. They get their own authority so the 23 can be DECOMPOSED rather
// than merely reported.
//
// ⚠ AND NAMING IS NOT LEGITIMISING. A resident stage is still outside the two
// permitted classes. Every one of these is a route that should eventually be
// either a canonical factor write inside a tick or a boundary transaction
// through a declared port. Until then it is counted, under the name that is
// true of it.
enum class ResidentStageReason : std::uint32_t {
  situation_contact = 0u,
  cloud_contact = 1u,
  learn_trajectory = 2u,
  bind_grounded_context = 3u,
  stage_motor_byte = 4u,
  stage_signature = 5u,
  form_edge = 6u,               // edge_bank::form_edge_device -- production
                                // mechanism, test-side launcher
};

struct ResidentStageReceipt {
  std::uint64_t epoch = 0u;
  std::uint64_t stage = 0u;
  ResidentStageReason reason = ResidentStageReason::situation_contact;
};

// The two permitted mutation classes, and nothing else. §0.12: "every
// post-genesis world change must be one of: canonical factor write during
// advance_one_tick; explicit reciprocal boundary transaction. There must be no
// third instrumentation, repair, or contact-helper route."
enum class MutationClass : std::uint32_t {
  canonical_factor_write = 0u,
  boundary_transaction = 1u,
};
inline constexpr std::uint32_t kPermittedMutationClasses = 2u;

// ⭐ THE INVARIANT, AS A NUMBER THE AUDIT CAN READ.
//
// Every type that can mutate organism memory must be counted here. The target is
// ONE -- the membrane itself. It is not 1 today and this header does not pretend
// otherwise: `device_words()` remains public on two classes, so the honest
// current value is what the ratchet measures. Lowering it is the migration.
[[nodiscard]] constexpr std::uint32_t membrane_target_world_writers() { return 1u; }

}  // namespace substrate::bcc32
