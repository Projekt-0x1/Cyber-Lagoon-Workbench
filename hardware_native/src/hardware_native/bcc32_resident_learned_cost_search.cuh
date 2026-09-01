#pragma once

#include <cstdint>

#include "causal_rewrite_universe.cuh"  // MUST stay first; see gh #1215
#include "bcc32_resident_turn_world_consequence_binding.cuh"

namespace substrate::bcc32::causal_rewrite::learned_cost_search {

#if defined(__CUDACC__)
#define BCC32_LEARNED_COST_SEARCH_DISPATCH \
  [[maybe_unused]] static __host__ __device__ __noinline__
#else
#define BCC32_LEARNED_COST_SEARCH_DISPATCH [[maybe_unused]] inline
#endif

inline constexpr std::uint64_t kSupportCostScale = 4096u;
// Search capacity is the resident physical ecology, not an authored route
// horizon. One label is retained per distinct reverse-reachable word.
inline constexpr std::uint32_t kMaximumLabels = kRecordCapacity;
// Each live Record can contribute at most one two-term Program edge. This is
// a physical table bound, not a semantic limit on trajectory extent.
inline constexpr std::uint32_t kMaximumEdges = kRecordCapacity;
inline constexpr std::uint32_t kLabelSettled = 1u << 31u;
inline constexpr std::uint32_t kLabelDepthMask = ~kLabelSettled;

struct Route {
  std::uint32_t action_word = 0u;
  std::uint32_t action_program = kInvalid;
  std::uint64_t total_cost = ~std::uint64_t{0};
  std::uint32_t depth = 0u;
};

enum class SelectionStatus : std::uint32_t {
  unique = 0u,
  absent = 1u,
  fail_closed = 2u,
};

__host__ __device__ inline bool checked_add(std::uint64_t left, std::uint64_t right,
                                            std::uint64_t* out) {
  if (left > ~std::uint64_t{0} - right)
    return false;
  *out = left + right;
  return true;
}

// Low resident support is expensive without becoming another weight store.
// Structural distance is the number of Program terms crossed, not numerical
// distance between opaque raw words. When a physically grounded resident
// consequence is available for this edge's Program locus, its unsigned
// quantity is accumulated exactly as represented: no sign, target, threshold,
// grade, normalization or learned coefficient is introduced here. Absence
// (has_consequence == false) preserves the historical intrinsic cost
// bit-for-bit -- this is the fail-closed default, not an opt-in.
__host__ __device__ inline bool learned_edge_cost(std::uint32_t support,
                                                  std::uint32_t term_distance,
                                                  bool has_consequence,
                                                  std::uint32_t consequence_value,
                                                  std::uint64_t* out) {
  if (support == 0u || term_distance == 0u || out == nullptr)
    return false;
  const std::uint64_t inverse_support = (kSupportCostScale + support - 1u) / support;
  std::uint64_t intrinsic = 0u;
  if (!checked_add(inverse_support, term_distance, &intrinsic))
    return false;
  if (!has_consequence) {
    *out = intrinsic;
    return true;
  }
  return checked_add(intrinsic, consequence_value, out);
}

__host__ __device__ inline bool canonical_program_shape(const Record& record) {
  // ProgramTerm/page lookup is resident-native and fail-closed.  A fixed
  // trajectory-length ceiling here silently excluded otherwise authoritative
  // continuation-page Programs before that lookup could validate their
  // terms.  Keep only the structural lower bound; the page directory and
  // resident authority predicates remain the actual physical extent/shape
  // checks.
  return record.matter_q8 != 0u && record.lane[0] == kFormProgram &&
         (record.lane[7] & kProgramFlagEnabled) != 0u && record.lane[2] >= 2u;
}

// Stable logical ordering for equal-cost producers. Slot address is deliberately
// absent: physical Record permutation may move a Program without changing the
// route selected from its resident contents.
__host__ __device__ inline bool program_key_less(
    const ResidentRewriteState* state, std::uint32_t left_slot,
    std::uint32_t right_slot) {
  const Record& left = state->records[left_slot];
  const Record& right = state->records[right_slot];
  const std::uint64_t left_digest = logical_record_digest(left);
  const std::uint64_t right_digest = logical_record_digest(right);
  if (left_digest != right_digest)
    return left_digest < right_digest;
  for (std::uint32_t lane = 0u; lane < kLaneCount; ++lane) {
    if (left.lane[lane] != right.lane[lane])
      return left.lane[lane] < right.lane[lane];
  }
  if (left.revision != right.revision)
    return left.revision < right.revision;
  if (left.matter_q8 != right.matter_q8)
    return left.matter_q8 < right.matter_q8;
  for (std::uint32_t word = 0u; word < 2u; ++word) {
    if (left.reserved[word] != right.reserved[word])
      return left.reserved[word] < right.reserved[word];
  }
  return false;
}

struct Label {
  std::uint64_t cost = ~std::uint64_t{0};
  std::uint32_t word = 0u;
  std::uint32_t depth_and_flags = 0u;
};

static_assert(sizeof(Label) == 16u,
              "resident learned-cost labels must remain compact");

__host__ __device__ inline std::uint32_t label_depth(const Label& label) {
  return label.depth_and_flags & kLabelDepthMask;
}

__host__ __device__ inline bool label_settled(const Label& label) {
  return (label.depth_and_flags & kLabelSettled) != 0u;
}

__host__ __device__ inline void settle_label(Label* label) {
  label->depth_and_flags |= kLabelSettled;
}

// A Program's resident edge is immutable for the duration of one selection
// query.  Cache only the page-native facts needed by the reverse search; the
// physical Record remains the authority for stable tie ordering.  The table
// is bounded by the live Record ecology, not by trajectory length or a
// language-shaped route count.
inline constexpr std::uint32_t kEdgeSourceHasMeta = 1u << 0u;
inline constexpr std::uint32_t kEdgeDestinationHasMeta = 1u << 1u;
inline constexpr std::uint32_t kEdgeVersionSpace = 1u << 2u;
// Set only when a resolved world-consequence value was live for this edge's
// Program locus at cache-build time. A resolved value of zero is still a
// real consequence, which is why presence is its own flag rather than
// consequence_value != 0.
inline constexpr std::uint32_t kEdgeHasWorldConsequence = 1u << 3u;

struct ProgramEdge {
  std::uint32_t program_slot = kInvalid;
  std::uint32_t source = 0u;
  std::uint32_t destination = 0u;
  std::uint32_t support = 0u;
  std::uint32_t flags = 0u;
  std::uint32_t consequence_value = 0u;
};

static_assert(sizeof(ProgramEdge) == 24u,
              "resident learned-cost edges must remain compact");

// This working set is deliberately NOT a stack local.  Measured on the linked
// sm_89 cubin, `Search search{}` alone gave select_resident_learned_cost_route
// a 50,360-byte frame -- 16,384 for labels, 24,576 for edges, 9,216 for the
// cycle-check arrays below, plus scalars -- which was 49% of the whole
// device-stack requirement of the persistent kernel (103,120 bytes at the time
// of this change).  cudaLimitStackSize is reserved per *resident thread slot*,
// not per launched thread: 80 SMs x 1536 slots means every KiB of that limit
// costs 120 MiB of the card before any work runs, so 50 KiB of stack in one
// function was reserving ~6 GiB of a 16 GiB device.  Keeping the working set
// in one addressable object lets the single caller own its storage instead.
//
// The cycle-check scratch (color/stack/next_slot) lives here for the same
// reason: as locals of reachable_cycle they were a separate 9,216-byte frame
// on the same call chain.
struct Search {
  const ResidentRewriteState* state = nullptr;
  std::uint32_t producer = kInvalid;
  Label labels[kMaximumLabels]{};
  ProgramEdge edges[kMaximumEdges]{};
  // reachable_cycle's bounded iterative DFS.  `color` must be zero on entry --
  // it reads color[root] before writing it -- so reset_search clears it.
  // `stack` and `next_slot` are written before every read (stack[0]/next_slot[0]
  // at each root, deeper entries on descend), so they need no initialization.
  std::uint8_t color[kMaximumLabels]{};
  std::uint32_t stack[kMaximumLabels]{};
  std::uint32_t next_slot[kMaximumLabels]{};
  std::uint32_t label_count = 0u;
  std::uint32_t edge_count = 0u;
  Route best{};
  bool found = false;
  bool ambiguous = false;
  bool saw_capacity = false;
  bool saw_cost_overflow = false;
  bool saw_malformed = false;
};

// Exactly equivalent to assigning `Search{}`, but without materializing a
// 50 KiB temporary -- which would put the very frame back that this struct
// exists to remove.  Every field the search reads before writing is cleared
// here; labels[] and edges[] are not, because both insertion paths assign
// every member of the element they append (see relax_label and
// build_edge_cache) and every reader is bounded by label_count/edge_count.
__host__ __device__ inline void reset_search(Search* search) {
  search->state = nullptr;
  search->producer = kInvalid;
  for (std::uint32_t index = 0u; index < kMaximumLabels; ++index)
    search->color[index] = 0u;
  search->label_count = 0u;
  search->edge_count = 0u;
  search->best = Route{};
  search->found = false;
  search->ambiguous = false;
  search->saw_capacity = false;
  search->saw_cost_overflow = false;
  search->saw_malformed = false;
}

__host__ __device__ inline std::uint32_t find_label(
    const Search* search, std::uint32_t word) {
  for (std::uint32_t index = 0u; index < search->label_count; ++index)
    if (search->labels[index].word == word)
      return index;
  return kInvalid;
}

__host__ __device__ inline bool relax_label(
    Search* search, std::uint32_t word, std::uint64_t cost,
    std::uint32_t depth) {
  const std::uint32_t existing = find_label(search, word);
  if (existing != kInvalid) {
    Label& label = search->labels[existing];
    if (label_settled(label)) {
      // Ordinary Dijkstra convergence into an already settled word is not a
      // cycle and cannot improve its positive-cost shortest label.
      return true;
    }
    if (cost < label.cost ||
        (cost == label.cost && depth < label_depth(label))) {
      label.cost = cost;
      label.depth_and_flags = depth;
    }
    return true;
  }
  if (search->label_count == kMaximumLabels) {
    search->saw_capacity = true;
    return false;
  }
  Label& label = search->labels[search->label_count++];
  label.cost = cost;
  label.word = word;
  label.depth_and_flags = depth;
  return true;
}

__host__ __device__ inline std::uint32_t least_unsettled_label(
    const Search* search) {
  std::uint32_t selected = kInvalid;
  for (std::uint32_t index = 0u; index < search->label_count; ++index) {
    const Label& candidate = search->labels[index];
    if (label_settled(candidate))
      continue;
    if (selected == kInvalid ||
        candidate.cost < search->labels[selected].cost ||
        (candidate.cost == search->labels[selected].cost &&
         candidate.word < search->labels[selected].word))
      selected = index;
  }
  return selected;
}

// Resolve each authoritative page-native Program exactly once per query.
// Previously explore() and reachable_cycle() each rediscovered the same
// terms, page locations and authority predicates for every reached label.
// Keeping the validated physical edge facts resident for this one read-only
// query removes that repeated lookup/recurrence amplifier without changing
// which Records are eligible or adding a semantic search ceiling.
__host__ __device__ inline bool build_edge_cache(Search* search) {
  if (search == nullptr || search->state == nullptr)
    return false;
  search->edge_count = 0u;
  for (std::uint32_t slot = 0u; slot < live_record_capacity(search->state); ++slot) {
    const Record& program = search->state->records[slot];
    if (slot == search->producer || program.matter_q8 == 0u ||
        program.lane[0] != kFormProgram)
      continue;
    if (!canonical_program_shape(program) ||
        !resident_program_authoritative(search->state, slot))
      continue;
    const std::uint32_t predecessor_index = program.lane[2] - 2u;
    const std::uint32_t outcome_index = program.lane[2] - 1u;
    std::uint32_t source = 0u;
    std::uint32_t source_meta = 0u;
    std::uint32_t destination = 0u;
    std::uint32_t destination_meta = 0u;
    if (!resident_program_term_at(search->state, slot, predecessor_index, &source,
                                  &source_meta) ||
        !resident_program_term_at(search->state, slot, outcome_index, &destination,
                                  &destination_meta)) {
      search->saw_malformed = true;
      continue;
    }
    if (search->edge_count == kMaximumEdges) {
      search->saw_capacity = true;
      return false;
    }
    ProgramEdge& edge = search->edges[search->edge_count++];
    edge.program_slot = slot;
    edge.source = source;
    edge.destination = destination;
    edge.support = program.lane[3];
    edge.flags = 0u;
    if (source_meta != 0u)
      edge.flags |= kEdgeSourceHasMeta;
    if (destination_meta != 0u)
      edge.flags |= kEdgeDestinationHasMeta;
    if ((program.lane[7] & kProgramFlagVersionSpace) != 0u)
      edge.flags |= kEdgeVersionSpace;

    // A resolved world consequence is another current resident fact about
    // this edge's Program locus, read beside source/destination/support so
    // Dijkstra operates on one internally coherent, frozen edge table. This
    // is deliberately find_claim_for_subject + resolve_claim_current_value
    // only -- both const/read-only -- never ensure_subject_claim, which
    // would allocate resident matter as a side effect of merely inspecting
    // an edge during selection. Establishing the claim is the writer/binding
    // seam's job, not this read-only surface's.
    bool claim_ambiguous = false;
    const std::uint32_t claim_slot =
        turn_world_consequence_binding::find_claim_for_subject(
            search->state, slot, &claim_ambiguous);
    if (claim_ambiguous) {
      // More than one claim scoped to the same producer locus is resident
      // ambiguity, not mere absence: refuse to guess which consequence
      // applies, the same discipline the malformed-term branch above uses.
      search->saw_malformed = true;
      continue;
    }
    if (claim_slot != kInvalid) {
      const std::uint32_t claim_owner = search->state->records[claim_slot].lane[1];
      std::uint32_t current_value = kInvalid;
      if (turn_world_consequence_binding::resolve_claim_current_value(
              search->state, claim_owner, &current_value)) {
        edge.consequence_value = current_value;
        edge.flags |= kEdgeHasWorldConsequence;
      }
      // No usable resolution (unbound claim, severed binding, removed world
      // cell) leaves flags/consequence_value at their intrinsic-only default
      // rather than inventing a numeric value.
    }
  }
  return true;
}

__host__ __device__ inline void retain_candidate(Search* search, const Route& candidate) {
  if (!search->found || candidate.total_cost < search->best.total_cost) {
    search->best = candidate;
    search->found = true;
    search->ambiguous = false;
    return;
  }
  if (candidate.total_cost != search->best.total_cost)
    return;
  if (candidate.action_word != search->best.action_word) {
    search->ambiguous = true;
    return;
  }
  // Two equally costly routes to the same outward action are not semantic
  // ambiguity. Retain one stable resident producer without using slot order.
  if (program_key_less(search->state, candidate.action_program,
                       search->best.action_program))
    search->best = candidate;
}

__host__ __device__ inline void explore(Search* search,
                                        std::uint32_t target_word) {
  if (!relax_label(search, target_word, 0u, 0u))
    return;
  for (;;) {
    const std::uint32_t label_index = least_unsettled_label(search);
    if (label_index == kInvalid)
      return;
    Label& current = search->labels[label_index];

    // learned_edge_cost() is strictly positive. Once the cheapest unresolved
    // label already costs as much as the incumbent action, no descendant can
    // tie or beat it. This is the search stop law.
    if (search->found && current.cost >= search->best.total_cost)
      return;

    settle_label(&current);
    const std::uint32_t target = current.word;
    const std::uint32_t depth = label_depth(current);
    for (std::uint32_t edge_index = 0u; edge_index < search->edge_count; ++edge_index) {
      const ProgramEdge& edge = search->edges[edge_index];
      // A VersionSpace predecessor is already a concrete, residently settled
      // binding.  Its non-zero meta preserves the variable ordinal for other
      // generic consumers; it is not an unresolved route variable.  Ordinary
      // Programs still require a literal predecessor at this one-edge view.
      const bool version_space = (edge.flags & kEdgeVersionSpace) != 0u;
      if ((!version_space && (edge.flags & kEdgeSourceHasMeta) != 0u) ||
          (edge.flags & kEdgeDestinationHasMeta) != 0u || edge.destination != target)
        continue;
      std::uint64_t edge_cost = 0u;
      std::uint64_t total_cost = 0u;
      const bool has_consequence = (edge.flags & kEdgeHasWorldConsequence) != 0u;
      if (!learned_edge_cost(edge.support, 1u, has_consequence, edge.consequence_value,
                             &edge_cost) ||
          !checked_add(current.cost, edge_cost, &total_cost)) {
        search->saw_cost_overflow = true;
        continue;
      }
      if ((edge.source & kRawChannelMask) == (1u << 24u)) {
        // Channel-1 predecessors normally terminate the reverse search.  The
        // initial target may itself be channel 1, however; returning to any
        // already reached channel-1 word is a cycle, not a newly discovered
        // public action.  The old path search performed this ancestry check
        // before classifying terminal actions.
        if (find_label(search, edge.source) != kInvalid) {
          continue;
        }
        Route candidate{};
        candidate.action_word = edge.source;
        candidate.action_program = edge.program_slot;
        candidate.total_cost = total_cost;
        candidate.depth = depth + 1u;
        retain_candidate(search, candidate);
        continue;
      }
      if (!relax_label(search, edge.source, total_cost, depth + 1u))
        return;
    }
  }
}

// Search labels deliberately collapse convergent paths.  Cycle classification
// therefore cannot be inferred from a second relaxation into a settled label:
// that is normal in a DAG.  This bounded iterative DFS examines only the
// reverse-reachable resident word graph after selection and distinguishes an
// actual back edge without restoring path enumeration.
// Takes a mutable Search only because its scratch now lives there; the
// resident world it reads through search->state stays read-only.
__host__ __device__ inline bool reachable_cycle(Search* search) {
  if (search == nullptr || search->state == nullptr)
    return false;
  std::uint8_t* const color = search->color;
  std::uint32_t* const stack = search->stack;
  std::uint32_t* const next_slot = search->next_slot;
  // reset_search cleared color once; clear again here so a second call within
  // one search still starts from an unvisited graph, exactly as the previous
  // per-call locals did.
  for (std::uint32_t index = 0u; index < search->label_count; ++index)
    color[index] = 0u;
  for (std::uint32_t root = 0u; root < search->label_count; ++root) {
    if (color[root] != 0u)
      continue;
    std::uint32_t depth = 0u;
    stack[0] = root;
    next_slot[0] = 0u;
    color[root] = 1u;
    for (;;) {
      const std::uint32_t current_index = stack[depth];
      const std::uint32_t target = search->labels[current_index].word;
      bool descended = false;
      for (std::uint32_t edge_index = next_slot[depth];
           edge_index < search->edge_count; ++edge_index) {
        next_slot[depth] = edge_index + 1u;
        const ProgramEdge& edge = search->edges[edge_index];
        const bool version_space = (edge.flags & kEdgeVersionSpace) != 0u;
        if ((!version_space && (edge.flags & kEdgeSourceHasMeta) != 0u) ||
            (edge.flags & kEdgeDestinationHasMeta) != 0u ||
            edge.destination != target)
          continue;
        const std::uint32_t neighbor = find_label(search, edge.source);
        if (neighbor == kInvalid)
          continue;
        if (color[neighbor] == 1u)
          return true;
        if (color[neighbor] == 0u) {
          ++depth;
          stack[depth] = neighbor;
          next_slot[depth] = 0u;
          color[neighbor] = 1u;
          descended = true;
          break;
        }
      }
      if (descended)
        continue;
      color[current_index] = 2u;
      if (depth == 0u)
        break;
      --depth;
    }
  }
  return false;
}

// Declared under __CUDACC__, not __CUDA_ARCH__: nvcc compiles this header in
// both a host and a device pass, and the host pass still emits a registration
// stub (__nv_cudaEntityRegisterCallback) naming every __device__ variable. A
// declaration visible only in the device pass makes that stub reference a
// symbol it cannot see. Only the *use* below is __CUDA_ARCH__-guarded.
#if defined(__CUDACC__)
// One device-resident working set instead of a per-call stack frame.
//
// Why a single shared instance is sound rather than a race: this search is
// reachable from exactly one kernel, resident_rewrite_epoch_kernel, whose very
// first statement is `if (blockIdx.x != 0u || threadIdx.x != 0u) return;`.
// That was verified against the linked cubin by walking the call graph
// backwards from this symbol over the .rel.text relocation sections -- one
// kernel, and that kernel admits one thread. So at most one thread can ever be
// inside this function.
//
// The guard below enforces that rather than assuming it. If a future caller
// runs this from a multi-threaded kernel it fails closed instead of silently
// sharing the working set, which is the same failure discipline the search
// already applies to capacity, overflow and malformed matter.
__device__ inline Search g_resident_learned_cost_search;
#endif

// Exhaustively searches only canonical, mature resident Program matter.  This
// selection surface is deliberately read-only so retention may ask the same
// question at a later physical END without inventing another route graph.
BCC32_LEARNED_COST_SEARCH_DISPATCH SelectionStatus
select_resident_learned_cost_route(
    const ResidentRewriteState* state, std::uint32_t producer_locus,
    std::uint32_t target_word, Route* selected) {
  if (selected == nullptr || state == nullptr || state->fault != 0u ||
      producer_locus == kInvalid || producer_locus >= live_record_capacity(state))
    return SelectionStatus::fail_closed;
  const Record& producer = state->records[producer_locus];
  if (!canonical_program_shape(producer) ||
      !resident_program_authoritative(state, producer_locus))
    return SelectionStatus::fail_closed;

#if defined(__CUDA_ARCH__)
  if (blockIdx.x != 0u || threadIdx.x != 0u)
    return SelectionStatus::fail_closed;
  Search& search = g_resident_learned_cost_search;
#else
  // Host builds keep the ordinary automatic object: there is no resident
  // thread-slot reservation to pay for, and host contracts may run searches
  // concurrently.
  Search search;
#endif
  reset_search(&search);
  search.state = state;
  search.producer = producer_locus;
  if (!build_edge_cache(&search))
    return SelectionStatus::fail_closed;
  explore(&search, target_word);
  if (search.ambiguous || search.saw_capacity || search.saw_cost_overflow ||
      search.saw_malformed) {
    return SelectionStatus::fail_closed;
  }
  if (!search.found) {
    if (reachable_cycle(&search))
      return SelectionStatus::fail_closed;
    return SelectionStatus::absent;
  }
  *selected = search.best;
  return SelectionStatus::unique;
}

// The RWR19 mutating path is a thin consumer of the read-only selection API.
// It preserves its prior fail-closed clearing behavior exactly.
__host__ __device__ inline bool advance_resident_learned_cost_search_once(
    ResidentRewriteEngine engine) {
  ResidentRewriteState* state = engine.state;
  if (state == nullptr || state->fault != 0u || state->generated_word_valid == 0u ||
      state->generated_locus == kInvalid || state->generated_locus >= live_record_capacity(state))
    return false;
  const Record& producer = state->records[state->generated_locus];
  if (!canonical_program_shape(producer) ||
      !resident_program_authoritative(state, state->generated_locus))
    return false;
  Route selected{};
  const SelectionStatus status = select_resident_learned_cost_route(
      state, state->generated_locus, state->generated_word, &selected);
  if (status == SelectionStatus::fail_closed) {
    clear_generated_word(state);
    return false;
  }
  if (status != SelectionStatus::unique)
    return false;

  state->generated_word = selected.action_word;
  state->generated_word_valid = 1u;
  state->generated_locus = selected.action_program;
  state->active_locus = state->generated_locus;
  return true;
}

}  // namespace substrate::bcc32::causal_rewrite::learned_cost_search

#undef BCC32_LEARNED_COST_SEARCH_DISPATCH
