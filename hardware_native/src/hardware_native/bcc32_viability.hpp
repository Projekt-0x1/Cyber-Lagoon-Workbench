#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "bcc32_coordinate.hpp"
#include "bcc32_g1_blueprint.hpp"
#include "bcc32_reference.hpp"
#include "bcc32_types.cuh"

namespace substrate::bcc32::viability {

// A declared or transcribed observable: at `coordinate`, the bits under
// `observed_mask` are the witness, read at superstep `tick`.
struct Port {
  Z3Coordinate coordinate{};
  SiteWord observed_mask = 0u;
  std::uint32_t tick = 0u;

  friend bool operator==(const Port&, const Port&) = default;
};

// Mechanical restatement of the committed bond-restore assertions in
// bcc32_g1_blueprint_contract.cpp:62-91. One port per seed site carrying a
// nonzero owned-bond field, masked to kOwnedBondMask, read at `horizon`.
// This is transcription, not authorship: it selects exactly the coordinates
// the committed contract's bond field selects.
[[nodiscard]] std::vector<Port> transcribe_bond_restore_ports(
    const std::vector<ReferenceSite>& seed, std::uint32_t horizon);

// Masked readings from an unperturbed run: baseline[i][t-1] is the reading at
// ports[i].coordinate under ports[i].observed_mask after t supersteps.
using BaselineTrace = std::vector<std::vector<SiteWord>>;

[[nodiscard]] BaselineTrace baseline_trace(const std::vector<ReferenceSite>& arrangement,
                                           const std::vector<Port>& ports);

// viable(A') iff for every port p and every tick t in 1..p.tick, the masked reading
// of A' at (p.coordinate, t) equals the baseline's masked reading there.
//
// The trajectory clause is load-bearing: matching the BASELINE (not the seed) at
// intermediate ticks encodes the committed turnover assertion and excludes the
// frozen-lump degeneracy. Do not relax this to t == p.tick.
[[nodiscard]] bool viable(const std::vector<ReferenceSite>& perturbed,
                          const std::vector<Port>& ports, const BaselineTrace& baseline);

// One draw of the matched null.
struct NullDraw {
  std::vector<ReferenceSite> arrangement;
  std::uint32_t redraws = 0u;
  // True when the support has fewer than two distinct words, so no non-trivial
  // shuffle exists and `arrangement` is necessarily identical to the seed. A caller
  // MUST record this: a degenerate null IS the real motif, so its density equals the
  // real density, and a receipt that did not say so would read as "real == null" --
  // a false kill signal under the pre-registered outcome table.
  bool degenerate = false;
};

// The matched null: identical coordinates, identical word multiset, assignment
// permuted by a seeded PRNG. Occupancy, delta_n_q, and the entire port apparatus
// are identical to the real motif by construction; only the ORGANIZATION is destroyed.
//
// A shuffle that reproduces the real motif exactly (possible for symmetric motifs)
// is discarded and redrawn; `redraws` receives the count.
//
// A coordinate-scatter null is INVALID here -- relocating matter leaves ports reading
// empty Q space, driving the null density toward 1. Do not add one. See spec section 6.1.
//
// DEGENERACY: if the support has fewer than two distinct words, no non-trivial shuffle
// exists at all -- the only possible "shuffle" reproduces the seed exactly. The returned
// NullDraw flags this via `degenerate`; a caller MUST check `draw.degenerate` before
// using the arrangement as a null, because a degenerate null IS the real motif.
[[nodiscard]] NullDraw word_shuffle_null(const std::vector<ReferenceSite>& seed,
                                         std::uint64_t rng_seed);

// Seals a single-motif blueprint (role: viability) and returns its authored
// support and its declared observed_output ports, as placed. Returns false if
// the motif fails to place, seal, or reopen -- callers must treat that as a
// hard failure, never a silently empty/missing rung. Shared by the rung
// catalog and by tests, so the seal-open-compile sequence has one definition.
//
// If `error` is non-null and the seal fails, it receives a human-readable reason
// (including the underlying G1BlueprintBuilder's error() text, when it set one) so
// callers -- notably build_rungs() -- can record WHY an instance was skipped instead
// of dropping it silently.
[[nodiscard]] bool seal_motif(const G1Motif& motif, const std::string& instance,
                              std::vector<ReferenceSite>& support, std::vector<Port>& ports,
                              std::string* error = nullptr);

// One measured arrangement: its authored seed, its witness, and its horizon.
struct Rung {
  std::string id;
  std::vector<ReferenceSite> seed;
  std::vector<Port> ports;
  std::uint32_t horizon = 0u;
};

// One skipped rung instance, with the reason. The receipt MUST record these:
// a measurement that quietly ran on nine rungs and reported as if it ran on ten
// is the silent truncation the pre-registration forbids (spec sections 8 and 9).
struct SkippedRung {
  std::string id;      // e.g. "cpair_port_relay_2"
  std::string reason;  // e.g. "motif factory returned nullopt", "blueprint failed to seal: <err>"
};

struct RungCatalog {
  std::vector<Rung> rungs;
  std::vector<SkippedRung> skipped;
};

// The four pre-registered rungs in ascending organization, as ten instances
// (1 + 1 + 4 payloads + 4 payloads). Rung 4 (cpair_receiver_transaction) CONTAINS
// rung 3 (cpair_port_relay), which gives the composition test. cpair_receiver_scaffold
// is excluded: it has no standalone witness.
//
// Any instance that cannot be built is recorded in `skipped` with its reason -- never
// dropped silently. The caller MUST write `skipped` into the receipt.
[[nodiscard]] RungCatalog build_rungs();

// Sorted, deduplicated set of every distinct word appearing in any rung's seed,
// union {Q}. Fixed before any perturbation is scored; written into the receipt.
[[nodiscard]] std::vector<SiteWord> catalog_alphabet(const std::vector<Rung>& rungs);

// Which alternative words count as a mutation of one site.
//   hamming1 -- the 32 single-bit neighbours of that site's word (encoding-relative)
//   catalog  -- the catalog alphabet (chemistry-like: one real material for another)
//   full     -- all 2^32 - 1 alternatives (definition-free; GPU only, see Task 7)
enum class Alphabet { hamming1, catalog, full };

struct DensityCount {
  std::uint64_t viable = 0u;
  std::uint64_t total = 0u;
  // Every baseline and perturbed trajectory is stepped through the reference
  // law with DeltaN_Q checked after each complete F. A violation is an executor
  // defect and throws instead of yielding a density.
  DeltaNQ baseline_delta_n_q = 0;
  std::uint64_t delta_n_q_trajectories_checked = 0u;
  bool delta_n_q_invariant = true;
};

// The rung catalog obtains its support by sealing a G1 capsule. Re-seal the
// exact compiled support deterministically so an offline receipt can bind that
// capsule content address without adding audit metadata to runtime matter.
[[nodiscard]] std::optional<ContentAddress> sealed_capsule_identity(
    const std::vector<ReferenceSite>& support);

// Exhaustively perturbs one site at a time and scores the predicate. The identity
// perturbation (replacing a word with itself) is never counted.
//
// Alphabet::full is NOT supported here -- 2^32 per site is a GPU job. Calling with
// full aborts rather than silently sampling.
[[nodiscard]] DensityCount enumerate_cpu(const std::vector<ReferenceSite>& arrangement,
                                         const std::vector<Port>& ports, Alphabet alphabet,
                                         const std::vector<SiteWord>& catalog);

}  // namespace substrate::bcc32::viability
