#pragma once

// #1184: asynchronous causal action-return bridge.
//
// Authority boundary (GitHub issue #1184): this module may know transport
// protocol only. It may not decide why the organism acted, which tool is
// cognitively appropriate, whether a returned payload is true, which
// resident route should learn, or what returned bytes mean. It exists
// purely to:
//
//   (a) capture the ticket ancestry of a motor action the adult already
//       chose to emit (MotorEvent::causal_root + the #1184-added
//       MotorEvent::cue_node);
//   (b) hand the raw request bytes to a pluggable external transport
//       adapter -- the adapter, not this bridge, decides how to execute a
//       shell/API/LLM/robot/etc. transaction;
//   (c) once that adapter later reports a raw result under the same
//       ticket, translate it back into a CausalOrigin::motor_reafference
//       ActivityEvent and inject it into the membrane frontier.
//
// It never blocks the adult step loop. Draining new tickets is bounded by
// the step's own motor event count (never a scan of unrelated device
// state); flushing completions only touches whatever the transport has
// already finished, never waits on it. No `ASK_LLM`/`SEARCH_WEB`/
// `RUN_PYTHON` cognitive opcode exists here or anywhere downstream of this
// header -- an adapter is testimony/world contact, never a resident
// semantic decision-maker.
//
// The adult's own exact-eligibility ledger (#1176,
// direct_adult_legacy_oracle.cu apply_return_credit_kernel) remains the sole
// causal authority that decides whether a returned ActivityEvent actually
// closes a live episode (exact ticket match, not expired, not ambiguous).
// This bridge only gets ticket ancestry to the membrane honestly; it never
// adjudicates causal credit itself.
//
// Spoofed raw-contact rejection (#1184 falsifier 3): `drain_pending_actions`
// arms each newly bound ticket's live #1176 eligibility record(s)
// (`mark_bridge_tickets_strict`, direct_adult_legacy_oracle.cu), and
// `apply_return_credit_kernel` then refuses every ordinary event against an
// armed record -- including an exact ticket and a forged reafferent origin.
// Only `flush_completions`, after `ingest_return`, can append the parallel
// bridge-authenticated frontier authority sidecar through its private grant.
//
// The first version of this guard tested `event.origin ==
// CausalOrigin::external_contact` and called `motor_reafference` "the origin
// only flush_completions stamps". That was a convention no code enforced, and
// in this tree it was not even true -- `cuda_direct_reafferent_byte_stream_
// contract`, `cuda_direct_multicontext_eligibility_contract` and
// `cuda_direct_hybrid_implicit_mesh_contract` all stamp `motor_reafference`
// themselves through the public `inject_raw_event`, legitimately, from body
// loops that are not this bridge. Nor could it have been enforced: origin is a
// field of a public aggregate passed to a public entry point. Both consequences
// were measured before the fix -- a host-stamped `motor_reafference` settled a
// bridge-claimed episode, and `world_return` (never named by the old condition)
// settled one through the coarse fallback without even knowing the ticket. Both
// are now sub-cases of falsifier 3. Provenance therefore lives in device state
// this bridge alone can advance, never in a field its caller supplies.
//
// Episodes the bridge never claims (every non-bridge MotorEvent, and every
// #1176 episode that never reaches a motor node at all) are completely
// unaffected -- ordinary chronological world/language learning keeps using the
// same coarse fallback it always has, including when it injects its own motor
// reafference.
//
// Duplicate-return non-double-credit (#1184 falsifier 4): no bridge-side
// mechanism change was needed here. `flush_completions`'s pre-existing
// single-consumption invariant -- a ticket is erased from `outstanding_`
// the instant its return is translated into an ActivityEvent -- already
// makes a second `ingest_return` for the same ticket (queued in the same
// batch or replayed after the first already settled) find no ancestry to
// bind to, whether or not a checkpoint/restart ever happened; it is counted
// via `dropped_unknown_return_count()`, not silently absorbed.
//
// Contradictory adapter returns remain separate raw testimony.  This bridge
// carries opaque source_id and ticket provenance, injects both, and never
// compares words, prefers a source, or assigns credit/reward.
//
// Checkpoint-with-pending-transport-I/O policy (#1184 falsifier 6): this
// bridge's outstanding-ticket table is host RAM the GPU checkpoint never
// touches. The chosen policy is PERSIST, not cancel: `capture_pending_state`
// snapshots every still-outstanding ticket's ancestry (cue_node/context/
// node/channel/word -- everything `flush_completions` needs to address a
// later return at the right membrane location) plus any adapter return
// already ingested but not yet flushed, and `restore_pending_state`
// rehydrates a freshly constructed bridge from that snapshot. This is the
// issue's own stated semantics -- "[while pending] the adult may
// checkpoint/restart" -- and it costs nothing extra to the real external
// adapter, which is never told a checkpoint happened and keeps working
// toward the same ticket id. `direct_adult_checkpoint.cu` wires this into
// `capture_direct_adult_checkpoint`/`restore_direct_adult_checkpoint` via an
// optional bridge argument so existing non-bridge callers are unaffected.
// No-duplication follows from the existing single-consumption invariant
// `flush_completions` already had: a ticket is erased from `outstanding_`
// the moment its return is translated into an ActivityEvent, whether that
// happens before or after a checkpoint/restore cycle, so a replayed or
// late-arriving return for an already-settled ticket still finds nothing to
// bind to. No-silent-disappearance follows from persisting the full table
// across restore; as a defense-in-depth trace even for a caller that
// forgets to pass a bridge to checkpoint capture, `flush_completions` also
// counts every return for an unknown ticket in `dropped_unknown_return_count_`
// so that loss is at minimum a resident-observable counter, never invisible.

#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <mutex>
#include <stdexcept>
#include <unordered_map>
#include <vector>

#include "hardware_native/direct_adult_legacy_oracle.cuh"

namespace substrate::direct_adult {

// One outstanding causal action, bound the moment the adult's membrane
// emitted it. Audit/ancestry state only -- transport protocol and content
// interpretation live entirely outside this struct.
struct PendingActionTicket {
  std::uint64_t ticket = 0u;               // == MotorEvent::causal_root
  std::uint32_t cue_node = kInvalidIndex;  // episode-opening node (#1184)
  std::uint32_t node = kInvalidIndex;      // physical motor node reached
  std::uint32_t channel = 0u;
  Word word = 0u;
  std::uint32_t context = 0u;
  std::uint32_t issue_tick = 0u;  // adult tick this ticket was drained at
};

// A raw, transport-level result an adapter reports back for one ticket. The
// bridge never interprets `word`; it is opaque adapter payload. `source_id`
// is transport provenance, not a semantic trust or selection signal.
// `history_signature` is optional exact-history disambiguation an adapter
// may thread through (0 means "not used").
struct BridgeReturn {
  std::uint64_t ticket = 0u;
  std::uint32_t node = kInvalidIndex;
  Word word = 0u;
  std::uint64_t history_signature = 0u;
  std::uint32_t source_id = 0u;
};

// Host-side, non-blocking causal action-return bridge. One instance owns
// the outstanding-ticket table for one DirectAdultRuntime. Safe to call
// `ingest_return` from an adapter's own thread while the owning thread
// continues driving the adult step loop.
class DirectCausalActionBridge {
 public:
  // Drains every MotorEvent the most recent adult step produced (per
  // `receipt.motor_count`, clamped to `runtime.motor_capacity`) and binds
  // each nonzero-ticket motor event to a PendingActionTicket. Must be
  // called after `observe_adult_step` and before the next
  // `launch_direct_adult_step`, which overwrites the device motor buffer.
  // Performs exactly one device->host copy sized to this step's own motor
  // event count -- never a poll of unrelated/outstanding device state --
  // and never waits on external transport. Returns the newly bound
  // tickets so the caller can hand raw request bytes to its adapter.
  //
  // #1184 authority boundary: every newly bound ticket is also armed on the
  // #1176 eligibility ledger itself (one small batched kernel launch,
  // `mark_bridge_tickets_strict` -- not a poll, proportional only to this
  // step's own new tickets). This is what makes the spoofed-raw-contact
  // falsifier hold: while armed, no injected event of any origin may settle
  // these specific episodes, so a return must actually traverse
  // `ingest_return`/`flush_completions` -- the only path to authenticated
  // event ingress -- to close one. `runtime` is taken by non-const reference
  // because of this device-side write.
  std::vector<PendingActionTicket> drain_pending_actions(DirectAdultRuntime& runtime,
                                                         const AdultStepReceipt& receipt) {
    std::vector<PendingActionTicket> issued;
    const std::uint32_t count = std::min(receipt.motor_count, runtime.motor_capacity);
    if (count == 0u)
      return issued;
    std::vector<MotorEvent> events(count);
    const cudaError_t status = cudaMemcpy(events.data(), runtime.motor_events,
                                          count * sizeof(MotorEvent), cudaMemcpyDeviceToHost);
    if (status != cudaSuccess)
      return issued;
    issued.reserve(count);
    std::vector<BridgeTicketMark> marks;
    marks.reserve(count);
    {
      std::lock_guard<std::mutex> lock(mutex_);
      for (const MotorEvent& motor : events) {
        if (motor.causal_root == 0u)
          continue;  // no ticket: not an asynchronous external action
        PendingActionTicket entry{};
        entry.ticket = motor.causal_root;
        entry.cue_node = motor.cue_node;
        entry.node = motor.node;
        entry.channel = motor.channel;
        entry.word = motor.word;
        entry.context = motor.context;
        entry.issue_tick = runtime.tick;
        outstanding_[entry.ticket] = entry;
        issued.push_back(entry);
        marks.push_back(BridgeTicketMark{entry.ticket, entry.cue_node, entry.context});
      }
    }
    mark_bridge_tickets_strict(&runtime, marks.data(), static_cast<std::uint32_t>(marks.size()));
    return issued;
  }

  // Called by an external transport adapter -- possibly from another
  // thread -- once a raw result is available for `value.ticket`.
  // Thread-safe. Never touches the GPU and never blocks on it: it only
  // appends to a completion queue this bridge later drains from the
  // caller's own thread via `flush_completions`.
  void ingest_return(const BridgeReturn& value) {
    std::lock_guard<std::mutex> lock(mutex_);
    completed_.push_back(value);
  }

  // Non-blocking: injects every already-completed return as a
  // CausalOrigin::motor_reafference ActivityEvent addressed at the
  // ticket's bound ancestry (cue_node, context), then forgets the ticket.
  // Whether the return actually closes a live episode -- exact ticket
  // match, not expired, not ambiguous -- is decided entirely by the
  // adult's own eligibility ledger the next time the step runs; this call
  // only gets provenance to the membrane, it never adjudicates causal
  // credit. Returns the number of ActivityEvents injected.
  //
  // CausalOrigin::motor_reafference describes this event but is not a
  // credential: public ingress may assert it too.  The credential is the
  // private BridgeReturnInjectionGrant passed only to inject_bridge_return_event,
  // which writes one authenticated sidecar entry without bypassing ordinary
  // chronology/context processing.
  std::uint32_t flush_completions(DirectAdultRuntime* runtime) {
    std::vector<BridgeReturn> ready;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      ready.swap(completed_);
    }
    std::vector<ActivityEvent> events;
    events.reserve(ready.size());
    for (const BridgeReturn& value : ready) {
      PendingActionTicket ticket{};
      {
        std::lock_guard<std::mutex> lock(mutex_);
        const auto it = outstanding_.find(value.ticket);
        if (it == outstanding_.end()) {
          // Unknown/already-settled ticket: no ancestry to bind. This is
          // exactly the "silent disappearance" shape #1184's checkpoint
          // falsifier names -- a return with nowhere to go must still leave
          // a resident-observable trace, so it is counted here even though
          // it cannot be injected without ancestry.
          ++dropped_unknown_return_count_;
          continue;
        }
        ticket = it->second;
        outstanding_.erase(it);
      }
      events.push_back(ActivityEvent{value.node, value.word, CausalOrigin::motor_reafference,
                                     ticket.context, ticket.cue_node, value.source_id, value.ticket,
                                     0u, value.history_signature});
    }
    if (events.empty())
      return 0u;
    for (const ActivityEvent& event : events)
      inject_bridge_return_event(runtime, event, BridgeReturnInjectionGrant{});
    return static_cast<std::uint32_t>(events.size());
  }

  std::size_t outstanding_count() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return outstanding_.size();
  }

  // Resident-observable trace for returns that could not be bound to any
  // ancestry (unknown ticket, already-settled ticket, or a pending ticket
  // that was silently dropped across a checkpoint/restore that did not use
  // `capture_pending_state`/`restore_pending_state`). Never resets itself;
  // a nonzero value is itself the "transport consequence" #1184's
  // checkpoint falsifier requires in place of silent loss.
  std::uint64_t dropped_unknown_return_count() const {
    return dropped_unknown_return_count_.load(std::memory_order_relaxed);
  }

  // Checkpoint-time snapshot of every ticket this bridge has not yet
  // resolved (#1184 "checkpoint pending I/O" falsifier). See the file
  // header for the chosen persist-not-cancel policy.
  struct BridgeCheckpointState {
    std::vector<PendingActionTicket> outstanding;
    std::vector<BridgeReturn> completed;
  };

  BridgeCheckpointState capture_pending_state() const {
    std::lock_guard<std::mutex> lock(mutex_);
    BridgeCheckpointState state;
    state.outstanding.reserve(outstanding_.size());
    for (const auto& entry : outstanding_)
      state.outstanding.push_back(entry.second);
    state.completed = completed_;
    return state;
  }

  // Rehydrates a freshly constructed (empty) bridge with pending-transport
  // state an earlier `capture_pending_state` snapshotted before a
  // checkpoint. Restoring into a bridge that already has state would
  // silently merge two distinct ticket ancestries -- refuse instead of
  // guessing, matching the rest of this codebase's checkpoint-restore
  // shape-mismatch behavior (see direct_adult_checkpoint.cu).
  void restore_pending_state(const BridgeCheckpointState& state) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!outstanding_.empty() || !completed_.empty()) {
      throw std::logic_error(
          "direct causal action bridge: refusing to restore pending state "
          "into a non-empty bridge");
    }
    outstanding_.reserve(state.outstanding.size());
    for (const PendingActionTicket& entry : state.outstanding)
      outstanding_[entry.ticket] = entry;
    completed_ = state.completed;
  }

 private:
  mutable std::mutex mutex_;
  std::unordered_map<std::uint64_t, PendingActionTicket> outstanding_;
  std::vector<BridgeReturn> completed_;
  std::atomic<std::uint64_t> dropped_unknown_return_count_{0u};
};

}  // namespace substrate::direct_adult
