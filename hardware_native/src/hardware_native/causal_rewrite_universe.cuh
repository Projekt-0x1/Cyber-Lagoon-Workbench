#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <type_traits>

#if defined(__CUDACC__)
#define BCC32_REWRITE_HD __host__ __device__
#else
#define BCC32_REWRITE_HD
#endif

namespace substrate::bcc32::causal_rewrite {

// RWR0 is deliberately small. Capacity is infrastructure, not a semantic
// address space or a prescribed neural graph: every persistent object has the
// same uniform substrate representation and all logical lookups scan by
// resident form and content. Program, factor, trajectory, and page forms are
// observer-visible physical witnesses, never one-site-per-concept cells or a
// fixed map of the adult. Writable recurrent organization must be grown by
// resident rewrites and may later distribute, replace, or bypass these forms.
inline constexpr std::uint32_t kRecordCapacity = 1024u;
inline constexpr std::uint32_t kLaneCount = 8u;
inline constexpr std::uint32_t kMatureSupport = 4u;
inline constexpr std::uint32_t kProgramMatureSupport = 3u;
inline constexpr std::uint32_t kSpanProgramMatureSupport = 2u;
inline constexpr std::uint32_t kPartialLifetime = 8u;
inline constexpr std::uint32_t kRecordMatterQ8 = 256u;
inline constexpr std::uint32_t kInvalid = 0xffffffffu;
inline constexpr std::uint32_t kCausalGermlineReflectionConflict =
    0xfffffffeu;
inline constexpr std::uint32_t kConstructionEpisodePairInduction = 1u << 1u;
inline constexpr std::uint32_t kConstructionEpisodeSpanInduction = 1u << 2u;
inline constexpr std::uint32_t kCausalGermlineExternal = 1u;
inline constexpr std::uint32_t kCausalGermlineEnabled = 1u;
inline constexpr std::uint32_t kCausalGermlineMinimumContributors = 3u;
// A trajectory is physically continued through resident page records. The
// first page remains byte-compatible with the original two-word term records;
// later pages carry page-local term owners plus a monotonic base offset. The
// per-page aperture remains 512 so the current CUDA workspace can execute the
// substrate without silently changing its physical ecology. Page rollover is
// representation only: it does not
// yield, close, summarize, or select a semantic boundary. END/PAUSE remain the
// only physical boundaries. There is no semantic page-count or transcript
// length policy: admission continues until the resident Record ecology is
// exhausted, and that exhaustion is a typed runtime RED.
inline constexpr std::uint32_t kTrajectoryPageEvents = 512u;
inline constexpr std::uint32_t kMaximumTrajectoryEvents =
    kTrajectoryPageEvents;
inline constexpr std::uint32_t kMaximumProgramVariables = 32u;
inline constexpr std::uint32_t kMaximumVariableSpanEvents = 64u;
inline constexpr std::uint32_t kMaximumSpanProgramTerms = 256u;
inline constexpr std::uint32_t kVersionSpaceMaximumAtoms = 4u;
inline constexpr std::uint32_t kVersionSpaceMatureSupport = 2u;
// resident_program_authoritative and the causal-germline revision-transfer
// authority chain (resident_program_shadowed_by_revision,
// resident_revision_transfer_product_authoritative,
// resident_revision_participation_reader_authoritative,
// ticketed_revision_source_authoritative,
// ticketed_return_witness_authoritative,
// revision_intervention_lineage_authoritative) form one mutual-recursion
// cycle with a single back-edge (revision_intervention_lineage_authoritative
// calling back into resident_program_authoritative). Live Record graphs are
// finite and acyclic in practice, but nothing in the Record wire format
// forbids a malformed/adversarial graph from making the cycle traverse
// indefinitely. This bound stops that traversal with a typed false rather
// than a device stack fault; it does not change behavior for any graph a
// legitimate resident construction could produce.
inline constexpr std::uint32_t kMaxCausalGermlineRecursionDepth = 12u;

// These are generic physical framing distinctions, not cognitive opcodes. A
// producer may divide the surrounding contact over any number of host packets.
// Pause says that the physical source currently yields the surface; end says
// that the preceding trajectory is complete evidence. Neither names a
// question, answer, language role, task, or desired consequence.
inline constexpr std::uint32_t kBoundaryPause = 0xff000001u;
inline constexpr std::uint32_t kBoundaryEnd = 0xff000002u;
inline constexpr std::uint32_t kEventFrameNone = 0u;
inline constexpr std::uint32_t kEventFramePause = 1u;
inline constexpr std::uint32_t kEventFrameEnd = 2u;
// Physical END is a source boundary, not an instruction to execute every
// resident transformation in the ingress call frame. The production CUDA
// adapter advances these phases one at a time from the same adult clock.
inline constexpr std::uint32_t kCloseWorkIdle = 0u;
inline constexpr std::uint32_t kCloseWorkCaptureSurface = 1u;
inline constexpr std::uint32_t kCloseWorkBindReply = 2u;
inline constexpr std::uint32_t kCloseWorkSettleInquiry = 3u;
inline constexpr std::uint32_t kCloseWorkQualifyHistory = 4u;
inline constexpr std::uint32_t kCloseWorkCrossContact = 5u;
inline constexpr std::uint32_t kCloseWorkClearProvenance = 6u;
inline constexpr std::uint32_t kCloseWorkReactivateInquiry = 7u;
inline constexpr std::uint32_t kCloseWorkComplete = 8u;
// External relation formation is a resumable physical close phase. It is
// deliberately outside the older phase numbering so checkpoints containing
// the bootstrap close values remain physically readable; the value names
// scheduling state only and never a semantic operation.
inline constexpr std::uint32_t kCloseWorkExternalRelation = 9u;
inline constexpr std::uint32_t kCloseWorkInvalidPhaseFault = 0xc10e0001u;
inline constexpr std::uint32_t kCloseWorkTransactionFault = 0xc10e0002u;
inline constexpr std::uint32_t kResidentPageCloneFault = 0xc10e0003u;
inline constexpr std::uint32_t kRawChannelMask = 0x0f000000u;
inline constexpr std::uint32_t kRawPayloadMask = 0x00ffffffu;

// These are the bootstrap interpreter's opaque material forms. They carry no
// task values, vocabulary, actor/object roles, route, answer, or operation.
// The fixed interpreter only recognizes free matter, an admitted sequence,
// an executable description, a staged match, a boundary motor record, and
// resident construction matter.
inline constexpr std::uint32_t kFormEmpty = 0x17d8b3a1u;
inline constexpr std::uint32_t kFormSequence = 0x8a62d4e9u;
inline constexpr std::uint32_t kFormDescription = 0x3f94c271u;
inline constexpr std::uint32_t kFormPartial = 0xc5e137abu;
inline constexpr std::uint32_t kFormMotor = 0x6b28fa43u;
inline constexpr std::uint32_t kFormConstructor = 0xe4139d07u;
// Genesis-authored material-gradient seed (0X1-189). Marks a record slot as
// non-quiescent purely because of its address-space distance from the two
// genesis anchor loci (constructor, motor), not because any host code
// branched on that slot's identity or role. No existing consumer recognizes
// this form, so it is fully inert to every current call site; it exists so a
// future developmental/rewrite law has real non-uniform genesis matter to
// react to instead of two isolated points in an otherwise uniform quiescent
// sea. See initialize_rewrite_state_with_material_gradient below.
inline constexpr std::uint32_t kFormGenesisSeed = 0xd8e29c47u;
inline constexpr std::uint32_t kFormTrajectory = 0x49ac623du;
inline constexpr std::uint32_t kFormTrajectoryTerm = 0x92f64bc1u;
inline constexpr std::uint32_t kFormTrajectoryPage = 0x4a72d9e1u;
inline constexpr std::uint32_t kTrajectoryPageOwnerSalt = 0x70616765u;
// A causal-constraint participation record reserves its source revision as
// an owner too. The participation seam is ordinary resident matter; this
// central form keeps owner generation from colliding with a live source.
inline constexpr std::uint32_t kFormConstraintParticipation = 0x76c4a219u;
// Generic recurrent ecology. Neither form may publish output or pay teaching
// authority. Route fragments retain only an opaque transition fragment;
// carriers are transient zero-authority activation moving through those
// fragments on the adult's existing epoch clock.
inline constexpr std::uint32_t kFormRecurrentRouteFragment = 0x0f3a6c91u;
inline constexpr std::uint32_t kFormRecurrentCarrier = 0x61b7d2e4u;
inline constexpr std::uint32_t kFormProgram = 0x715de4a9u;
inline constexpr std::uint32_t kFormProgramTerm = 0xb6c80f53u;
inline constexpr std::uint32_t kFormSpanProgram = 0x4c19e7a2u;
inline constexpr std::uint32_t kFormSpanProgramTerm = 0xd37a51e8u;
// A bounded derived execution cursor. It stores no new vocabulary or rule;
// it only preserves the already-proven variable extents and physical program
// position so autonomous byte-by-byte continuation does not repeatedly solve
// the same prefix partition.
inline constexpr std::uint32_t kFormSpanExecutionCursor = 0x8e24b6c7u;
// Transient span bindings are ordinary owner-bound matter, one Record per
// variable. They are execution residue, not vocabulary or a packed cursor
// encoding, and are reclaimed with the trajectory that created them.
inline constexpr std::uint32_t kFormSpanExecutionBinding = 0x97d2c5e1u;
inline constexpr std::uint32_t kFormTransformationWitness = 0x2c7a91d3u;
inline constexpr std::uint32_t kFormProgramFactor = 0x1a7e4c92u;
inline constexpr std::uint32_t kFormProgramEvidence = 0x2e8b5d71u;
inline constexpr std::uint32_t kFormProgramAlternativeTerm = 0x3c9f6a84u;
inline constexpr std::uint32_t kFormProgramWitness = 0x4d1a7b63u;
inline constexpr std::uint32_t kFormConstructionEpisode = 0x5e2c8a71u;
inline constexpr std::uint32_t kFormConstructionEpisodeTerm = 0x7b9146d3u;
inline constexpr std::uint32_t kFormCausalConstructor = 0x31cf8e65u;
inline constexpr std::uint32_t kFormCausalConstructorTerm = 0x68a3d19bu;
inline constexpr std::uint32_t kFormCausalConstructorWitness = 0x9c42e7b5u;
inline constexpr std::uint32_t kFormCausalConstructorDelta = 0xc73e5a29u;
inline constexpr std::uint32_t kFormCausalProductWitness = 0xae7513c9u;
inline constexpr std::uint32_t kFormCausalCounterevidence = 0xf19b46d2u;
inline constexpr std::uint32_t kFormRevisionTransferWitness = 0x6ec3a947u;
inline constexpr std::uint32_t kFormRevisionTransferSourceUse = 0x8ad15f03u;
inline constexpr std::uint32_t kFormRevisionTransferPriorTerm = 0x9be26014u;
inline constexpr std::uint32_t kFormRevisionParticipationReader = 0xac037125u;
inline constexpr std::uint32_t kFormRevisionParticipationReaderTerm =
    0xbd148236u;
inline constexpr std::uint32_t kFormRevisionAmbiguityTrace = 0xce259347u;
inline constexpr std::uint32_t kFormRevisionAmbiguityParticipant = 0xdf36a458u;
inline constexpr std::uint32_t kFormRevisionParticipationDelta = 0xe047b569u;
inline constexpr std::uint32_t kFormRevisionEgressWitness = 0xf158c67au;
inline constexpr std::uint32_t kRevisionEgressRetainedTrajectory = 0x45524731u;
// Persistent resident evidence that a channel-1 word was generated by one
// executable locus downstream of one exact completed inquiry episode before
// the public ticket transport was exposed. The ticket never supplies either
// identity.
inline constexpr std::uint32_t kFormRevisionActionIssuance = 0xc64e28b1u;
// P4 is resident return evidence, not a digest of return evidence. Its header
// names the exact staged trajectory and the resident action issuance; one
// owned term Record retains every word observed before END converts that same
// trajectory in place. No ticket field or caller-selected semantic enters it.
inline constexpr std::uint32_t kFormTicketedReturnWitness = 0x53bd926fu;
inline constexpr std::uint32_t kFormTicketedReturnWitnessTerm = 0x64cea380u;
// The sole PersistentKernel writer creates this after the accepted terminal
// transaction converts that transaction's still-current raw trajectory in
// place. It carries no semantic target, answer, or host-selected revision.
inline constexpr std::uint32_t kFormRevisionTransferReturn = 0xb7e46c21u;
// External-contact provenance is ordinary owner-bound Record matter. Keep its
// framing constants in the core ABI so exact conversion can validate and
// retire provenance without including the ingress helper.
inline constexpr std::uint32_t kFormTrajectoryProvenance = 0x2d83a6f1u;
// Parent-route lineage is a second ordinary owner-bound matter form. It keeps
// the upstream physical ancestry separate from the immediate ingress route;
// copied/replayed parent tokens can therefore be detected without treating a
// route token as a source name, trust score, or semantic authority.
inline constexpr std::uint32_t kFormTrajectoryParentRoute = 0x3e94b702u;
inline constexpr std::uint32_t kProvenanceValidityLane = 7u;
inline constexpr std::uint32_t kProvenanceExternalOrigin = 0u;
// A generated event whose closure was rederived from distributed
// participation. This is provenance only; it is never a producer locus.
inline constexpr std::uint32_t kProvenanceDistributedOrigin = 2u;
inline constexpr std::uint32_t kProvenanceGeneratedOrigin = 1u;
inline constexpr std::uint32_t kProgramFlagEnabled = 1u;
// Programs bearing this bit earn support only from resident contributor
// Programs.  Raw trajectory matches may execute them once mature, but may not
// pay their authority directly.
inline constexpr std::uint32_t kProgramFlagResidentEvidenceOnly = 1u << 1u;
// Reserved for the literal-chain port: an exact Program physically converted
// from one wholly external completed episode.
inline constexpr std::uint32_t kProgramFlagPureExternalExact = 1u << 2u;
// Version-space alternatives are Enabled Program subtypes: the canonical
// executor admits them through its consensus collector while the older
// cross-context transformer leaves their shared factors untouched.
inline constexpr std::uint32_t kProgramFlagVersionSpace = 1u << 3u;
inline constexpr std::uint32_t kProgramFlagVersionSpaceLesioned = 1u << 4u;
// The product was constructed from one fresh exact episode by resident causal
// constructor matter. Raw replay cannot pay this authority; its live witness
// chain is revalidated by the root before authority reopens.
inline constexpr std::uint32_t kProgramFlagCausalGermlineProduct = 1u << 5u;
// One fresh wholly external correction extended a mature resident Program
// through the Program's already-earned variable topology.  Authority remains
// physical in the exact source, donor, and one RevisionTransferWitness.
inline constexpr std::uint32_t kProgramFlagRevisionTransferProduct = 1u << 6u;
inline constexpr std::uint32_t kSpanTermLiteral = 0u;
inline constexpr std::uint32_t kSpanTermVariable = 1u;
inline constexpr std::uint32_t kSpanMatchNone = 0u;
inline constexpr std::uint32_t kSpanMatchPrefix = 1u;
inline constexpr std::uint32_t kSpanMatchAmbiguous = 2u;
inline constexpr std::uint32_t kTrajectoryHasGenerated = 1u;
inline constexpr std::uint32_t kTrajectoryWasYielded = 2u;
// Carry is zero-authority execution context.  Support and induction continue
// to require a pristine trajectory, so it cannot become teaching matter.
inline constexpr std::uint32_t kTrajectoryHasCarry = 1u << 4u;
// One freshly grounded ambiguity reply may select a live VersionSpace
// consequence for exactly the next resident execution step. The marker is
// cleared as soon as that consequence is observed; it is neither evidence nor
// a general preference among unresolved alternatives.
inline constexpr std::uint32_t kTrajectoryVersionSpaceSelected = 1u << 6u;

struct alignas(16) Record {
  std::uint32_t lane[kLaneCount]{};
  std::uint32_t revision = 0u;
  std::uint32_t matter_q8 = kRecordMatterQ8;
  // Keep the complete aligned representation deterministic across CPU and
  // CUDA. These words occupy the tail that would otherwise be implicit,
  // indeterminate padding and therefore break byte-exact replay receipts.
  std::uint32_t reserved[2]{};
};

struct RawRewriteEvent {
  std::uint32_t value = 0u;
  std::uint32_t valid = 0u;
  std::uint32_t reserved = 0u;
  // Physical ingress route, not a source name or semantic class. Existing
  // aggregate initializers may omit it; canonical raw ingress supplies the
  // high-byte body channel when stamping external ancestry.
  std::uint32_t physical_route = kInvalid;
  // Opaque upstream physical ancestry. It is optional because legacy callers
  // do not yet have an authenticated parent adapter. When present, it is
  // retained as resident lineage and repeated tokens never count as distinct
  // external histories.
  std::uint32_t parent_route = kInvalid;
};

struct RewriteBatchReceipt {
  std::uint32_t requested = 0u;
  std::uint32_t consumed = 0u;
  std::uint32_t completed = 0u;
  std::uint32_t observer_settled = 0u;
  std::uint32_t fault = 0u;
  std::uint32_t trajectory_pages = 0u;
  std::uint32_t trajectory_continued = 0u;
};

struct ProgramEvidenceView {
  const std::uint32_t* atoms = nullptr;
  std::uint32_t length = 0u;
  std::uint32_t owner = 0u;
  std::uint32_t consequence = 0u;
};

struct ProgramEvidenceReceipt {
  std::uint32_t committed = 0u;
  std::uint32_t duplicate = 0u;
  std::uint32_t rejected = 0u;
  std::uint32_t factor_count = 0u;
  std::uint32_t alternative_count = 0u;
  std::uint32_t mature_alternative_count = 0u;
  std::uint64_t digest_before = 0u;
  std::uint64_t digest_after = 0u;
};

struct LesionEscrow {
  Record displaced[8]{};
  std::uint32_t original_slot[8]{};
  std::uint32_t count = 0u;
  std::uint32_t removed_matter_q8 = 0u;
};

// RWR0 resident matter beyond the 1,024-Record ceiling (Linear 0X1-147). This
// lands only the storage/growth half of the design in
// docs/audits/2026-08-14-rwr0-resident-matter-page-scaling-plan.md
// (Sections 3.1 and 3.3): storage is a two-tier page directory instead of one
// flat array, and allocate_record grows a new page on demand instead of
// failing closed at 1,024. Page 0 stays inline inside ResidentRewriteState,
// first member of the first member, so it remains byte-identical to today's
// `Record records[kRecordCapacity]` for the common case that never exceeds
// kRecordsPerPage live Records -- several existing contracts (e.g.
// bcc32_cuda_resident_causal_constraint_participation_contract.cu) cudaMemcpy
// exactly `kRecordCapacity * sizeof(Record)` bytes straight off the front of
// a device-resident ResidentRewriteState and depend on that layout. Pages
// 1..N-1 are grown on demand as separate device/host pointers (whichever
// address space `grow_resident_pages` is called from -- see its comment).
//
// The staged-transaction copy-on-write half (§3.4) borrows unchanged grown
// pages and clones them on first staged mutation. Core resident enumeration in
// this header is
// page-aware through live_record_capacity(); fixed kRecordCapacity uses that
// remain elsewhere are deliberately treated as separate semantic apertures,
// bounded scratch sizes, or collision-attempt limits until each call site has
// its own measured page-growth contract. A grown population must never be
// silently treated as a complete adult-language capability merely because the
// storage proxy can address it.
inline constexpr std::uint32_t kRecordsPerPage = kRecordCapacity;
inline constexpr std::uint32_t kMaxResidentPages = 8u;

struct RecordPage {
  Record slots[kRecordsPerPage]{};
};

struct ResidentPageDirectory {
  RecordPage page0{};
  RecordPage* pages[kMaxResidentPages - 1u]{};
  std::uint32_t live_page_count = 1u;
  // A staged transaction may borrow grown pages from its canonical source.
  // Bits 0..6 correspond to pages 1..7. A set bit means the page is borrowed
  // and must be copied before a non-const record access; it is never freed by
  // the staging world's release path.
  std::uint32_t shared_page_mask = 0u;
};

struct ResidentRewriteState;

// `state->records[i]` must keep compiling unchanged at every existing call
// site once Record storage moves into `directory`. A stored "owner" pointer
// inside this proxy would go stale the instant the surrounding
// ResidentRewriteState is copied -- and it is copied constantly: ordinary
// struct assignment throughout the contract tests (`ResidentRewriteState
// clone = state;`), `*state = ResidentRewriteState{}` in
// initialize_rewrite_state, and the raw byte copy in
// bcc32_resident_rewrite_runtime.cu's copy_rewrite_state, none of which know
// to fix up an embedded self-pointer. So this proxy stores nothing at all
// and recovers the owning state's address from its own address via
// offsetof, which stays correct across any such copy because it is computed
// from `this` at the point of use, never cached.
struct RecordsProxy {
  BCC32_REWRITE_HD Record& operator[](std::uint32_t index);
  BCC32_REWRITE_HD const Record& operator[](std::uint32_t index) const;

  // Forward-only, index-driven iterators: `record_at` already dispatches
  // page0 vs. a grown page per lookup, so an iterator that carries only
  // (state, index) stays correct across a page boundary for free -- no
  // separate cross-page-transition logic needed. Deliberately not random-
  // access (no operator+, operator-): callers that need arithmetic on a
  // records range (std::reverse(records, records + N) and friends) index
  // through operator[] instead, so this proxy never needs to hand out a
  // pointer that could alias across two non-contiguous RecordPage
  // allocations.
  struct iterator {
    ResidentRewriteState* state;
    std::uint32_t index;
    BCC32_REWRITE_HD Record& operator*() const;
    BCC32_REWRITE_HD iterator& operator++();
    BCC32_REWRITE_HD bool operator==(const iterator& other) const {
      return index == other.index;
    }
    BCC32_REWRITE_HD bool operator!=(const iterator& other) const {
      return index != other.index;
    }
  };
  struct const_iterator {
    const ResidentRewriteState* state;
    std::uint32_t index;
    BCC32_REWRITE_HD const Record& operator*() const;
    BCC32_REWRITE_HD const_iterator& operator++();
    BCC32_REWRITE_HD bool operator==(const const_iterator& other) const {
      return index == other.index;
    }
    BCC32_REWRITE_HD bool operator!=(const const_iterator& other) const {
      return index != other.index;
    }
  };

  BCC32_REWRITE_HD iterator begin();
  BCC32_REWRITE_HD iterator end();
  BCC32_REWRITE_HD const_iterator begin() const;
  BCC32_REWRITE_HD const_iterator end() const;
};

struct ResidentRewriteState {
  ResidentPageDirectory directory{};
  RecordsProxy records{};
  LesionEscrow lesion{};

  std::uint64_t revision = 0u;
  std::uint64_t admitted_events = 0u;
  std::uint32_t allocation_cursor = 0u;
  std::uint32_t fault = 0u;
  // END schedules recursive Program factorization. The root quiet-epoch path
  // settles it after nested ingress frames have unwound on CUDA.
  std::uint32_t cross_context_factor_pending = 0u;
  // CG0 uses the same bounded root-phase discipline. Deep construction and
  // lineage validation never share the CUDA ingress call stack.
  std::uint32_t causal_germline_reflection_program = kInvalid;
  std::uint32_t causal_germline_reflection_source = kInvalid;
  std::uint32_t causal_germline_application_program = kInvalid;
  std::uint32_t causal_germline_construction_pending = 0u;
  std::uint32_t causal_germline_validation_pending = 0u;
  // A queued physical close is resident scheduling state, not a semantic
  // boundary or a context-window counter. While pending, ordered raw ingress
  // waits behind the current adult's close phases; passive observation and the
  // resident clock continue, so one expensive phase cannot monopolize ingress.
  std::uint32_t close_work_pending = 0u;
  std::uint32_t close_work_phase = kCloseWorkIdle;
  // Disposable close-induction cursors. They are resident scheduling state,
  // not semantic cells: a close may scan and reduce one live candidate tile
  // per epoch without turning Record order into output authority.
  std::uint32_t close_induction_cursor = 0u;
  std::uint32_t close_induction_fixed_identity = kInvalid;
  std::uint32_t close_induction_fixed_left = kInvalid;
  std::uint32_t close_induction_span_identity = kInvalid;
  std::uint32_t close_induction_span_left = kInvalid;
  std::uint32_t close_induction_conflict = 0u;

  // Resumable external-contact formation state. These fields are resident
  // transaction cursors and provenance receipts, not a vocabulary, relation,
  // answer, or semantic route. They let one END consume a long raw source
  // over bounded resident epochs while retaining exact source order.
  std::uint32_t external_relation_stage_active = 0u;
  std::uint32_t external_relation_stage_owner = kInvalid;
  // Physical source locus captured when an END stage opens.  The close
  // cursor revalidates this slot on every later epoch instead of rediscovering
  // the source by scanning the complete resident aperture.
  std::uint32_t external_relation_stage_source_slot = kInvalid;
  std::uint32_t external_relation_stage_source_revision = 0u;
  std::uint32_t external_relation_stage_event_cursor = 0u;
  std::uint32_t external_relation_stage_window_cursor = 0u;
  // Read-only lookup cursors for the resumable external relation validator.
  // They cache physical record slots only while the source-validation cursor
  // is active; no semantic category or answer is stored here.
  std::uint32_t external_relation_stage_provenance_ordinal = kInvalid;
  std::uint32_t external_relation_stage_provenance_slot = kInvalid;
  std::uint32_t external_relation_stage_page = kInvalid;
  std::uint32_t external_relation_stage_page_slot = kInvalid;
  std::uint32_t external_relation_stage_term_page = kInvalid;
  std::uint32_t external_relation_stage_term_ordinal = kInvalid;
  std::uint32_t external_relation_stage_term_slot = kInvalid;
  std::uint32_t external_relation_stage_term_alt_page = kInvalid;
  std::uint32_t external_relation_stage_term_alt_ordinal = kInvalid;
  std::uint32_t external_relation_stage_term_alt_slot = kInvalid;
  std::uint32_t external_relation_stage_external_leaves = 0u;
  std::uint32_t external_relation_stage_admitted = 0u;
  std::uint32_t external_relation_stage_rejected = 0u;
  std::uint32_t external_relation_stage_completed = 0u;
  std::uint64_t external_relation_stage_digest = 0u;

  // Passive raw outward matter. The root runtime is the only component that
  // may publish these fields as a public action.
  std::uint32_t raw_motor_value = 0u;
  std::uint32_t raw_motor_valid = 0u;
  std::uint32_t active_locus = kInvalid;
  std::uint32_t constructor_locus = kInvalid;
  std::uint32_t generated_word = 0u;
  std::uint32_t generated_word_valid = 0u;
  std::uint32_t generated_locus = kInvalid;
  // Once a completed public writer has been bound, ordinary teacher-contact
  // binding is closed until the exact external return gate authorizes and
  // settles it. This is resident provenance/aperture state, not a ticket,
  // answer, or semantic cell.
  std::uint32_t open_inquiry_public_return_pending = 0u;
  // Sticky only until the next action-return attempt. This keeps the
  // accepted external-return receipt observable across quiet epochs without
  // turning a replay or forged attempt into a second acceptance.
  std::uint32_t open_inquiry_public_return_receipt = 0u;
  // A distributed closure has no producer Record locus. This transaction
  // receipt is an opaque lineage identity for the public membrane only; its
  // component closure is rederived before append, and neither the receipt nor
  // its owner is a semantic cell or future authority.
  std::uint32_t generated_receipt_owner = kInvalid;
  std::uint32_t generated_receipt_valid = 0u;
  std::uint32_t generated_receipt_participant_records = 0u;
  std::uint32_t generated_receipt_external_leaves = 0u;
  std::uint32_t generated_receipt_independent_sources = 0u;
  std::uint32_t generated_receipt_source_contributions = 0u;
  std::uint64_t generated_receipt_topology_digest = 0u;
  std::uint64_t generated_receipt_revision_digest = 0u;
  std::uint64_t generated_receipt_provenance_digest = 0u;
  std::uint64_t generated_receipt_participation_digest = 0u;
  std::uint64_t generated_receipt_epoch = 0u;

  // Observer receipts. They do not drive matching or choose a description.
  std::uint32_t concrete_descriptions = 0u;
  std::uint32_t mature_descriptions = 0u;
  std::uint32_t partial_matches = 0u;
  std::uint32_t direct_fires = 0u;
  std::uint32_t staged_fires = 0u;
  // How staged hypotheses actually end. A partial is an in-flight hypothesis,
  // and there are two independent ways it can die before its confirming
  // evidence arrives: it can age out under kPartialLifetime, or it can be
  // retired unconditionally at the next value presentation whether or not it
  // matched. Without separating them, "no construction formed" cannot be told
  // apart from "the hypothesis expired first", and the two want opposite
  // repairs. Observer receipts only: nothing below reads them.
  std::uint32_t partials_aged_out = 0u;
  std::uint32_t partials_retired_matched = 0u;
  std::uint32_t partials_retired_unmatched = 0u;
  std::uint32_t conflict_abstentions = 0u;
  std::uint32_t constructor_rewrites = 0u;
  std::uint32_t relearned_descriptions = 0u;
  std::uint32_t program_rules = 0u;
  std::uint32_t mature_program_rules = 0u;
  std::uint32_t trajectory_records = 0u;
  std::uint32_t retained_exemplars = 0u;
  std::uint32_t program_generated_events = 0u;
  std::uint32_t program_conflict_abstentions = 0u;
  std::uint32_t rejected_unbound_variables = 0u;
  std::uint32_t completed_inductions = 0u;
  std::uint32_t inspected_records = 0u;
  std::uint32_t span_program_rules = 0u;
  std::uint32_t mature_span_program_rules = 0u;
  std::uint32_t span_generated_events = 0u;
  std::uint32_t span_conflict_abstentions = 0u;
  std::uint32_t span_rejected_unbound_variables = 0u;
  std::uint32_t span_ambiguous_abstentions = 0u;
  std::uint32_t span_completed_inductions = 0u;
  // Latest successful generic causal-relation publication. These passive
  // receipts are committed only by the common Program emitter after its
  // authoritative const rederivation and raw trajectory append succeed.
  std::uint32_t causal_relation_generated_events = 0u;
  std::uint32_t causal_relation_probe_steps = 0u;
  std::uint32_t causal_relation_participating_records = 0u;
  std::uint32_t causal_relation_independent_sources = 0u;
  std::uint32_t causal_relation_source_contributions = 0u;
  std::uint32_t causal_relation_max_source_contribution = 0u;
  std::uint32_t causal_relation_contribution_concentration_q16 = 0u;
  std::uint32_t causal_relation_singleton_supported_steps = 0u;
  std::uint32_t causal_relation_minimum_probe_support = 0u;
  std::uint64_t causal_relation_component_digest = 0u;
  std::uint64_t causal_relation_component_revision_digest = 0u;
  std::uint64_t causal_relation_external_provenance_digest = 0u;
  std::uint32_t causal_relation_external_leaves = 0u;
  // Revision of the query trajectory at the exact distributed emission. The
  // live trajectory advances when a returned external consequence is
  // appended, so reafferent readout uses this chronology receipt while still
  // revalidating every current word, witness, component and revision digest.
  // The first generated offset in trajectory.lane[5] is retained for legacy
  // readers.  This pair records the exact later participation-closure term
  // when several generated terms share one yielded trajectory, so a return
  // cannot accidentally qualify an earlier ordinary producer instead.
  std::uint32_t causal_relation_trajectory_owner = kInvalid;
  std::uint32_t causal_relation_generated_index = kInvalid;
  std::uint32_t causal_relation_trajectory_revision = 0u;
  std::uint32_t version_space_factors = 0u;
  std::uint32_t version_space_alternatives = 0u;
  std::uint32_t mature_version_space_alternatives = 0u;
  std::uint32_t version_space_witnesses = 0u;
  std::uint32_t version_space_conflict_abstentions = 0u;
  std::uint32_t causal_germline_episodes = 0u;
  std::uint32_t causal_germline_constructors = 0u;
  std::uint32_t causal_germline_applications = 0u;
  std::uint32_t causal_germline_reconstructions = 0u;
  // Live counterevidence is recomputed from resident matter. The suppression
  // fields are cumulative successful authority-transition receipts.
  std::uint32_t causal_germline_counterevidence = 0u;
  std::uint32_t causal_germline_product_suppressions = 0u;
  std::uint32_t causal_germline_constructor_suppressions = 0u;
  std::uint32_t causal_germline_conflict_abstentions = 0u;
  std::uint32_t causal_germline_constructor_locus = kInvalid;
  std::uint32_t causal_germline_product_locus = kInvalid;
  // RWR24 open-inquiry diagnostics (0X1-163/0X1-206); extracted (flat members, see that file) for the 2,000-line ceiling.
#include "bcc32_resident_open_inquiry_diagnostic_counters.inl"
  std::uint64_t organization_digest = 0u;
  // Device-owned live-frontier receipt. It is derived by the parallel root
  // frontier pass from currently active trajectory matter, not from a host
  // scheduler or a semantic category. The ordered lane may consume the
  // receipt as a cheap sparse-frontier gate while the observer census remains
  // responsible for organization counters.
  std::uint32_t live_frontier_records = 0u;
  std::uint64_t live_frontier_digest = 0u;
  // Canonical CUDA epochs defer this passive whole-population receipt until
  // the graph's parallel census node. Direct/host callers leave the flag at
  // zero and retain the synchronous refresh contract.
  std::uint64_t organization_receipt_revision = 0u;
  std::uint32_t organization_receipt_deferred = 0u;
};

static_assert(std::is_trivially_copyable_v<Record>);
static_assert(std::is_trivially_copyable_v<ResidentRewriteState>);
static_assert(std::is_trivially_copyable_v<RewriteBatchReceipt>);
static_assert(std::is_trivially_copyable_v<ProgramEvidenceView>);
static_assert(std::is_trivially_copyable_v<ProgramEvidenceReceipt>);
static_assert(sizeof(ResidentRewriteState) <= 8u * 1024u * 1024u,
              "resident rewrite matter must remain bounded");

BCC32_REWRITE_HD inline Record& record_at(ResidentRewriteState* state,
                                          std::uint32_t index) {
  const std::uint32_t page = index / kRecordsPerPage;
  const std::uint32_t slot = index % kRecordsPerPage;
  if (page == 0u)
    return state->directory.page0.slots[slot];
  const std::uint32_t page_bit = 1u << (page - 1u);
  if ((state->directory.shared_page_mask & page_bit) != 0u) {
    RecordPage* original = state->directory.pages[page - 1u];
    RecordPage* private_page =
        static_cast<RecordPage*>(malloc(sizeof(RecordPage)));
    if (private_page == nullptr || original == nullptr) {
      if (private_page != nullptr) free(private_page);
      // A failed COW allocation must fail closed rather than let a staged
      // mutation write through into canonical resident matter. The page-zero
      // sink is already private to this state and the transaction fault makes
      // its contents unpublishable.
      state->fault = 1u;
      return state->directory.page0.slots[0];
    }
    for (std::uint32_t copy_slot = 0u; copy_slot < kRecordsPerPage;
         ++copy_slot)
      private_page->slots[copy_slot] = original->slots[copy_slot];
    state->directory.pages[page - 1u] = private_page;
    state->directory.shared_page_mask &= ~page_bit;
  }
  return state->directory.pages[page - 1u]->slots[slot];
}

BCC32_REWRITE_HD inline const Record& record_at(
    const ResidentRewriteState* state, std::uint32_t index) {
  const std::uint32_t page = index / kRecordsPerPage;
  const std::uint32_t slot = index % kRecordsPerPage;
  return page == 0u ? state->directory.page0.slots[slot]
                    : state->directory.pages[page - 1u]->slots[slot];
}

BCC32_REWRITE_HD inline Record& RecordsProxy::operator[](std::uint32_t index) {
  auto* state = reinterpret_cast<ResidentRewriteState*>(
      reinterpret_cast<unsigned char*>(this) -
      offsetof(ResidentRewriteState, records));
  return record_at(state, index);
}

BCC32_REWRITE_HD inline const Record& RecordsProxy::operator[](
    std::uint32_t index) const {
  const auto* state = reinterpret_cast<const ResidentRewriteState*>(
      reinterpret_cast<const unsigned char*>(this) -
      offsetof(ResidentRewriteState, records));
  return record_at(state, index);
}

BCC32_REWRITE_HD inline std::uint32_t live_record_capacity(
    const ResidentRewriteState* state) {
  return state->directory.live_page_count * kRecordsPerPage;
}

BCC32_REWRITE_HD inline Record& RecordsProxy::iterator::operator*() const {
  return record_at(state, index);
}

BCC32_REWRITE_HD inline RecordsProxy::iterator&
RecordsProxy::iterator::operator++() {
  ++index;
  return *this;
}

BCC32_REWRITE_HD inline const Record&
RecordsProxy::const_iterator::operator*() const {
  return record_at(state, index);
}

BCC32_REWRITE_HD inline RecordsProxy::const_iterator&
RecordsProxy::const_iterator::operator++() {
  ++index;
  return *this;
}

BCC32_REWRITE_HD inline RecordsProxy::iterator RecordsProxy::begin() {
  auto* state = reinterpret_cast<ResidentRewriteState*>(
      reinterpret_cast<unsigned char*>(this) -
      offsetof(ResidentRewriteState, records));
  return iterator{state, 0u};
}

BCC32_REWRITE_HD inline RecordsProxy::iterator RecordsProxy::end() {
  auto* state = reinterpret_cast<ResidentRewriteState*>(
      reinterpret_cast<unsigned char*>(this) -
      offsetof(ResidentRewriteState, records));
  return iterator{state, live_record_capacity(state)};
}

BCC32_REWRITE_HD inline RecordsProxy::const_iterator RecordsProxy::begin()
    const {
  const auto* state = reinterpret_cast<const ResidentRewriteState*>(
      reinterpret_cast<const unsigned char*>(this) -
      offsetof(ResidentRewriteState, records));
  return const_iterator{state, 0u};
}

BCC32_REWRITE_HD inline RecordsProxy::const_iterator RecordsProxy::end()
    const {
  const auto* state = reinterpret_cast<const ResidentRewriteState*>(
      reinterpret_cast<const unsigned char*>(this) -
      offsetof(ResidentRewriteState, records));
  return const_iterator{state, live_record_capacity(state)};
}

// This predicate remains a useful observer/test distinction for callers that
// specifically require the old inline-only representation. The production
// close staging path now clones grown pages instead of using this predicate as
// an authority gate.
BCC32_REWRITE_HD inline bool resident_population_is_single_page(
    const ResidentRewriteState* state) {
  return state->directory.live_page_count <= 1u;
}

// Draws one more RecordPage on demand, up to kMaxResidentPages, and zero-
// initializes it exactly as initialize_rewrite_state does for page 0 (every
// slot cleared, then marked kFormEmpty). `malloc` resolves to CUDA's device
// heap when this function is instantiated for the device compilation pass
// and to the ordinary host allocator for the host pass -- the same
// `__host__ __device__` dual-context idiom already used elsewhere in this
// header (see e.g. bcc32_grown_cloud_factor.cuh's `similarity`). A
// host-grown page and a device-grown page are never valid in the other
// context, exactly like every other pointer this engine already treats as
// context-local; nothing crosses that boundary through this call.
BCC32_REWRITE_HD inline bool grow_resident_pages(ResidentRewriteState* state) {
  if (state->directory.live_page_count >= kMaxResidentPages) return false;
  auto* page = static_cast<RecordPage*>(malloc(sizeof(RecordPage)));
  if (page == nullptr) return false;
  for (std::uint32_t slot = 0u; slot < kRecordsPerPage; ++slot) {
    page->slots[slot] = Record{};
    page->slots[slot].lane[0] = kFormEmpty;
  }
  state->directory.pages[state->directory.live_page_count - 1u] = page;
  ++state->directory.live_page_count;
  return true;
}

struct ResidentRewriteEngine {
  ResidentRewriteState* state = nullptr;

  BCC32_REWRITE_HD explicit ResidentRewriteEngine(
      ResidentRewriteState* resident)
      : state(resident) {}
};

// Device ingress may discover more than one newly mature construction in one
// settlement aperture. The root must never inherit whichever candidate was
// visited first. Agreement keeps one opaque locus; disagreement clears both
// loci and records one fail-closed conflict for root-phase settlement.
BCC32_REWRITE_HD inline void schedule_causal_germline_reflection(
    ResidentRewriteState* state, std::uint32_t program,
    std::uint32_t source) {
  if (state == nullptr || program == kInvalid || source == kInvalid ||
      state->causal_germline_reflection_program ==
          kCausalGermlineReflectionConflict)
    return;
  if (state->causal_germline_reflection_program == kInvalid) {
    state->causal_germline_reflection_program = program;
    state->causal_germline_reflection_source = source;
    return;
  }
  if (state->causal_germline_reflection_program == program &&
      state->causal_germline_reflection_source == source)
    return;
  state->causal_germline_reflection_program =
      kCausalGermlineReflectionConflict;
  state->causal_germline_reflection_source = kInvalid;
}

BCC32_REWRITE_HD inline bool causal_germline_reflection_conflicted(
    const ResidentRewriteState* state) {
  return state != nullptr &&
         state->causal_germline_reflection_program ==
             kCausalGermlineReflectionConflict;
}

BCC32_REWRITE_HD inline std::uint32_t rewrite_mix(std::uint32_t a,
                                                  std::uint32_t b,
                                                  std::uint32_t c) {
  std::uint32_t x = a ^ (b + 0x9e3779b9u + (a << 6u) + (a >> 2u));
  x ^= c + 0x85ebca6bu + (x << 13u) + (x >> 7u);
  x ^= x >> 16u;
  x *= 0x7feb352du;
  x ^= x >> 15u;
  x *= 0x846ca68bu;
  return x ^ (x >> 16u);
}

BCC32_REWRITE_HD inline void clear_record(Record* record) {
  const std::uint32_t matter = record->matter_q8;
  const std::uint32_t revision = record->revision + 1u;
  *record = Record{};
  record->lane[0] = kFormEmpty;
  record->matter_q8 = matter;
  record->revision = revision;
}

BCC32_REWRITE_HD inline std::uint32_t find_form(
    const ResidentRewriteState* state, std::uint32_t form) {
  for (std::uint32_t i = 0u; i < live_record_capacity(state); ++i)
    if (state->records[i].matter_q8 != 0u &&
        state->records[i].lane[0] == form)
      return i;
  return kInvalid;
}

BCC32_REWRITE_HD inline bool expired_zero_authority_partial(
    const Record& record) {
  // A partial is transient match state, never an executable Program, a
  // Constructor, trajectory/provenance, or learned authority.  Keep that
  // boundary explicit: only an already-expired partial with its reserved
  // authority tail still clear may be recycled.
  return record.matter_q8 != 0u && record.lane[0] == kFormPartial &&
         record.lane[3] > kPartialLifetime && record.lane[5] == 0u &&
         record.lane[6] == 0u && record.lane[7] == 0u;
}

BCC32_REWRITE_HD inline std::uint32_t unique_expired_partial(
    const ResidentRewriteState* state) {
  std::uint32_t candidate = kInvalid;
  for (std::uint32_t slot = 0u; slot < live_record_capacity(state); ++slot) {
    if (!expired_zero_authority_partial(state->records[slot])) continue;
    // Reclamation must not use physical position as an eviction rank.  One
    // unique transient record is safe; competing candidates fail closed.
    if (candidate != kInvalid) return kInvalid;
    candidate = slot;
  }
  return candidate;
}

BCC32_REWRITE_HD inline std::uint32_t allocate_record(
    ResidentRewriteState* state) {
  // Growth-on-demand (RWR0 §3.3). Each outer iteration is exactly today's
  // scan-then-reclaim over the *current* live population; only once both are
  // exhausted does this reach for one more RecordPage, up to
  // kMaxResidentPages, before finally failing closed exactly as before. When
  // live_page_count stays at 1 this is byte-for-byte the original algorithm
  // (capacity == kRecordCapacity on the first and only iteration).
  for (;;) {
    const std::uint32_t capacity = live_record_capacity(state);
    for (std::uint32_t offset = 0u; offset < capacity; ++offset) {
      const std::uint32_t slot = (state->allocation_cursor + offset) % capacity;
      Record& record = state->records[slot];
      if (record.matter_q8 == 0u || record.lane[0] != kFormEmpty) continue;
      state->allocation_cursor = (slot + 1u) % capacity;
      return slot;
    }
    const std::uint32_t reclaimed = unique_expired_partial(state);
    if (reclaimed != kInvalid) {
      clear_record(&state->records[reclaimed]);
      state->allocation_cursor = (reclaimed + 1u) % capacity;
      return reclaimed;
    }
    if (!grow_resident_pages(state)) {
      state->fault = 1u;
      return kInvalid;
    }
  }
}

BCC32_REWRITE_HD inline std::uint64_t logical_record_digest(
    const Record& record) {
  std::uint32_t left = 0x243f6a88u;
  std::uint32_t right = 0x85a308d3u;
  for (std::uint32_t lane = 0u; lane < kLaneCount; ++lane) {
    left = rewrite_mix(left, record.lane[lane], lane);
    right = rewrite_mix(right, record.lane[lane], left);
  }
  return (static_cast<std::uint64_t>(left) << 32u) | right;
}

// Bootstrap motor exploration is a generic resident disposition, not a body
// decoder. Given two already-admitted raw distinctions and the unique live
// motor matter, derive one raw channel-1 actuator word entirely from resident
// state. The body mapping and its returned consequence are deliberately absent.
// Reafference uses the same law to prove that an accepted action was resident-
// derived rather than trusting a provenance locus by itself.
BCC32_REWRITE_HD inline bool resident_motor_babble_action_word(
    const ResidentRewriteState* state, std::uint32_t motor_locus,
    std::uint32_t first, std::uint32_t second, std::uint32_t* word) {
  if (state == nullptr || word == nullptr || first == 0u || second == 0u ||
      first == kInvalid || second == kInvalid ||
      motor_locus >= live_record_capacity(state))
    return false;
  std::uint32_t motors = 0u;
  for (std::uint32_t slot = 0u; slot < live_record_capacity(state); ++slot) {
    const Record& candidate = state->records[slot];
    if (candidate.matter_q8 != 0u && candidate.lane[0] == kFormMotor)
      ++motors;
  }
  const Record& motor = state->records[motor_locus];
  if (motors != 1u || motor.matter_q8 == 0u ||
      motor.lane[0] != kFormMotor)
    return false;
  const std::uint64_t digest = logical_record_digest(motor);
  const std::uint32_t physical_seed =
      static_cast<std::uint32_t>(digest) ^
      static_cast<std::uint32_t>(digest >> 32u);
  const std::uint32_t payload =
      rewrite_mix(first, second, physical_seed) & kRawPayloadMask;
  *word = (1u << 24u) | payload;
  return true;
}

BCC32_REWRITE_HD inline void refresh_receipt(ResidentRewriteState* state) {
  if (state->organization_receipt_deferred != 0u) return;
  // END plus its scheduled Program factorization is one atomic observer step.
  // Do not publish the pre-factor intermediate organization.
  if (state->cross_context_factor_pending != 0u) return;
  state->concrete_descriptions = 0u;
  state->mature_descriptions = 0u;
  state->partial_matches = 0u;
  state->program_rules = 0u;
  state->mature_program_rules = 0u;
  state->trajectory_records = 0u;
  state->retained_exemplars = 0u;
  state->span_program_rules = 0u;
  state->mature_span_program_rules = 0u;
  state->version_space_factors = 0u;
  state->version_space_alternatives = 0u;
  state->mature_version_space_alternatives = 0u;
  state->version_space_witnesses = 0u;
  state->causal_germline_episodes = 0u;
  state->causal_germline_constructors = 0u;
  state->causal_germline_counterevidence = 0u;
  state->causal_germline_constructor_locus = kInvalid;
  state->causal_germline_product_locus = kInvalid;
  state->organization_digest = 0u;
  for (std::uint32_t i = 0u; i < live_record_capacity(state); ++i) {
    const Record& record = state->records[i];
    if (record.matter_q8 == 0u || record.lane[0] == kFormEmpty) continue;
    // XOR is intentionally slot-order independent. Physical loci may move
    // under a record permutation without changing logical organization.
    state->organization_digest ^= logical_record_digest(record);
    if (record.lane[0] == kFormDescription) {
      ++state->concrete_descriptions;
      if (record.lane[5] >= kMatureSupport)
        ++state->mature_descriptions;
    } else if (record.lane[0] == kFormPartial) {
      ++state->partial_matches;
    } else if (record.lane[0] == kFormProgram) {
      if ((record.lane[7] & kProgramFlagVersionSpace) != 0u) {
        ++state->version_space_alternatives;
        if ((record.lane[7] & kProgramFlagVersionSpaceLesioned) == 0u &&
            record.lane[3] >= kVersionSpaceMatureSupport)
          ++state->mature_version_space_alternatives;
      } else {
        ++state->program_rules;
        if (record.lane[3] >= kProgramMatureSupport)
          ++state->mature_program_rules;
      }
    } else if (record.lane[0] == kFormTrajectory ||
               record.lane[0] == kFormTrajectoryTerm ||
               record.lane[0] == kFormTrajectoryPage) {
      ++state->trajectory_records;
      if (record.lane[0] == kFormTrajectory && record.lane[3] != 0u)
        ++state->retained_exemplars;
    } else if (record.lane[0] == kFormSpanProgram) {
      ++state->span_program_rules;
      if (record.lane[3] >= kSpanProgramMatureSupport)
        ++state->mature_span_program_rules;
    } else if (record.lane[0] == kFormProgramFactor) {
      ++state->version_space_factors;
    } else if (record.lane[0] == kFormProgramWitness) {
      ++state->version_space_witnesses;
    } else if (record.lane[0] == kFormConstructionEpisode) {
      ++state->causal_germline_episodes;
    } else if (record.lane[0] == kFormCausalConstructor) {
      ++state->causal_germline_constructors;
      if (state->causal_germline_constructor_locus == kInvalid)
        state->causal_germline_constructor_locus = i;
    } else if (record.lane[0] == kFormCausalCounterevidence) {
      ++state->causal_germline_counterevidence;
    }
    if ((record.lane[0] == kFormProgram ||
         record.lane[0] == kFormSpanProgram) &&
        (record.lane[7] & kProgramFlagCausalGermlineProduct) != 0u &&
        state->causal_germline_product_locus == kInvalid)
      state->causal_germline_product_locus = i;
  }
  state->organization_receipt_revision = state->revision;
}

BCC32_REWRITE_HD inline void initialize_rewrite_state(
    ResidentRewriteState* state, std::uint32_t permutation = 0u) {
  *state = ResidentRewriteState{};
  for (std::uint32_t i = 0u; i < kRecordCapacity; ++i) {
    state->records[i] = Record{};
    state->records[i].lane[0] = kFormEmpty;
  }
  const std::uint32_t constructor =
      rewrite_mix(0x13u, permutation, 0x71u) % kRecordCapacity;
  std::uint32_t motor =
      rewrite_mix(0x29u, permutation, 0x43u) % kRecordCapacity;
  if (motor == constructor) motor = (motor + 1u) % kRecordCapacity;

  state->records[constructor].lane[0] = kFormConstructor;
  state->records[constructor].lane[1] = 2u;  // maximum antecedent count
  state->records[constructor].lane[2] = kMatureSupport;
  state->records[motor].lane[0] = kFormMotor;
  state->constructor_locus = constructor;
  state->allocation_cursor =
      rewrite_mix(permutation, 0xa5u, 0x5au) % kRecordCapacity;
  refresh_receipt(state);
}

// 0X1-189 first slice: a genesis-authored spatial material gradient, additive
// and opt-in. `initialize_rewrite_state` above places exactly two isolated
// non-quiescent points (one constructor locus, one motor locus) in an
// otherwise uniformly empty sea, with no other structure. This function
// starts from that same state, then deterministically marks every record
// slot within a bounded address-space radius of either anchor as
// kFormGenesisSeed, with an intensity (lane[1]) that decays linearly with
// distance from the nearest anchor -- an explicit, inspectable, ordinary-
// arithmetic spatial distribution of bit-native matter computed once here,
// never a host branch on a slot's semantic identity and never GPU hardware
// interpolation (Invariant III, already-decided and rejected; see
// docs/0x1_handbook_complete.md around line 25150).
//
// The record's flat address (its slot index) is the only "position" concept
// this substrate currently has; there is no literal (x, y, z) lattice. Using
// address-distance from the constructor/motor loci as a stand-in for
// distance from body/sensor/motor apertures is a first, honest step, not a
// claim that this is the eventual full lattice-geometry mechanism.
//
// Every produced slot keeps matter_q8 at its untouched Record{} default
// (kRecordMatterQ8), so the kRecordCapacity * kRecordMatterQ8 matter-
// conservation invariant used elsewhere (matter_account_is_closed and its
// callers) is unaffected -- only which form a slot starts in changes, never
// how much matter it carries. kFormGenesisSeed is a brand-new form tag no
// existing call site recognizes, so this is inert to every current consumer
// of ResidentRewriteState; only future logic that explicitly looks for it
// can react.
BCC32_REWRITE_HD inline void
initialize_rewrite_state_with_material_gradient(
    ResidentRewriteState* state, std::uint32_t permutation = 0u,
    std::uint32_t gradient_reach = kRecordCapacity / 8u) {
  initialize_rewrite_state(state, permutation);
  const std::uint32_t constructor = state->constructor_locus;
  const std::uint32_t motor = find_form(state, kFormMotor);
  for (std::uint32_t i = 0u; i < kRecordCapacity; ++i) {
    if (state->records[i].lane[0] != kFormEmpty) continue;
    const std::uint32_t distance_to_constructor =
        i > constructor ? i - constructor : constructor - i;
    const std::uint32_t distance_to_motor =
        i > motor ? i - motor : motor - i;
    const std::uint32_t distance =
        distance_to_constructor < distance_to_motor ? distance_to_constructor
                                                      : distance_to_motor;
    if (distance >= gradient_reach) continue;
    state->records[i].lane[0] = kFormGenesisSeed;
    state->records[i].lane[1] = gradient_reach - distance;
  }
  refresh_receipt(state);
}

BCC32_REWRITE_HD inline std::uint32_t description_identity(
    std::uint32_t arity, std::uint32_t first, std::uint32_t second,
    std::uint32_t consequence) {
  return rewrite_mix(arity ^ (first << 8u), second, consequence);
}

BCC32_REWRITE_HD inline bool same_antecedent(const Record& record,
                                             std::uint32_t arity,
                                             std::uint32_t first,
                                             std::uint32_t second) {
  return record.matter_q8 != 0u &&
         record.lane[0] == kFormDescription && record.lane[1] == arity &&
         record.lane[2] == first && record.lane[3] == second;
}

BCC32_REWRITE_HD inline std::uint32_t find_description(
    const ResidentRewriteState* state, std::uint32_t arity,
    std::uint32_t first, std::uint32_t second, std::uint32_t consequence) {
  for (std::uint32_t i = 0u; i < live_record_capacity(state); ++i) {
    const Record& record = state->records[i];
    if (same_antecedent(record, arity, first, second) &&
        record.lane[4] == consequence)
      return i;
  }
  return kInvalid;
}

BCC32_REWRITE_HD inline void install_or_support_description(
    ResidentRewriteState* state, std::uint32_t arity,
    std::uint32_t first, std::uint32_t second, std::uint32_t consequence) {
  const std::uint32_t existing =
      find_description(state, arity, first, second, consequence);
  if (existing != kInvalid) {
    Record& description = state->records[existing];
    if (description.lane[5] != 0xffffffffu) ++description.lane[5];
    ++description.revision;
    ++state->revision;
    ++state->constructor_rewrites;
    return;
  }

  const std::uint32_t slot = allocate_record(state);
  if (slot == kInvalid) return;
  Record& description = state->records[slot];
  description.lane[0] = kFormDescription;
  description.lane[1] = arity;
  description.lane[2] = first;
  description.lane[3] = second;
  description.lane[4] = consequence;
  description.lane[5] = 1u;
  description.lane[6] =
      description_identity(arity, first, second, consequence);
  description.lane[7] = 0u;
  ++description.revision;
  ++state->revision;
  ++state->constructor_rewrites;
}

BCC32_REWRITE_HD inline void clear_motor(ResidentRewriteState* state) {
  state->raw_motor_value = 0u;
  state->raw_motor_valid = 0u;
  state->active_locus = kInvalid;
  const std::uint32_t motor = find_form(state, kFormMotor);
  if (motor == kInvalid) return;
  Record& record = state->records[motor];
  record.lane[1] = 0u;
  record.lane[2] = 0u;
  record.lane[3] = kInvalid;
}

BCC32_REWRITE_HD inline void clear_generated_word(
    ResidentRewriteState* state) {
  state->generated_word = 0u;
  state->generated_word_valid = 0u;
  state->generated_locus = kInvalid;
  state->generated_receipt_owner = kInvalid;
  state->generated_receipt_valid = 0u;
  state->generated_receipt_participant_records = 0u;
  state->generated_receipt_external_leaves = 0u;
  state->generated_receipt_independent_sources = 0u;
  state->generated_receipt_source_contributions = 0u;
  state->generated_receipt_topology_digest = 0u;
  state->generated_receipt_revision_digest = 0u;
  state->generated_receipt_provenance_digest = 0u;
  state->generated_receipt_participation_digest = 0u;
  state->generated_receipt_epoch = 0u;
}

BCC32_REWRITE_HD inline bool schedule_physical_end(
    ResidentRewriteState* state) {
  if (state == nullptr || state->fault != 0u) return false;
  if (state->close_work_pending != 0u) return true;
  state->close_work_pending = 1u;
  // The external relation stage must finish against the still-live raw
  // trajectory before the ordinary close helpers can retire its pages.
  // stage_current_before_end() already ran synchronously on canonical state
  // before this call (see bcc32_resident_rewrite_runtime.cu's kBoundaryEnd
  // handling) and populated external_relation_stage_* on the copy this
  // function now runs on; do not re-zero it here, or the very staging this
  // phase exists to consume is wiped before advance_current_before_end()
  // ever reads it. Whenever staging did not occur, those fields are already
  // clean because advance_current_before_end() clears
  // external_relation_stage_active on every prior completion/rejection.
  state->close_work_phase = kCloseWorkExternalRelation;
  state->close_induction_cursor = 0u;
  state->close_induction_fixed_identity = kInvalid;
  state->close_induction_fixed_left = kInvalid;
  state->close_induction_span_identity = kInvalid;
  state->close_induction_span_left = kInvalid;
  state->close_induction_conflict = 0u;
  // `stage_current_before_end()` has already installed a resumable cursor on
  // the canonical source before this close shadow is copied. Do not erase it
  // here: the close worker must carry that resident transaction into the
  // private shadow and advance it before retiring the raw trajectory. A
  // caller that has no staged source is already represented by
  // external_relation_stage_active == 0 and needs no reset.
  state->cross_context_factor_pending = 1u;
  return true;
}

BCC32_REWRITE_HD inline void emit_candidate(std::uint32_t value,
                                            std::uint32_t locus,
                                            std::uint32_t* candidate_count,
                                            std::uint32_t* candidate_value,
                                            std::uint32_t* candidate_locus,
                                            bool* conflict) {
  if (*candidate_count == 0u) {
    *candidate_count = 1u;
    *candidate_value = value;
    *candidate_locus = locus;
    return;
  }
  if (*candidate_value != value) {
    *conflict = true;
  } else if (locus < *candidate_locus) {
    // Equivalent consequences agree. Keep deterministic minimum physical
    // provenance for receipts and lesions; it never ranks the output value.
    *candidate_locus = locus;
  }
}

BCC32_REWRITE_HD inline void publish_candidate(
    ResidentRewriteState* state, std::uint32_t candidate_count,
    std::uint32_t candidate_value, std::uint32_t candidate_locus,
    bool conflict) {
  if (candidate_count == 0u || conflict) {
    if (conflict) ++state->conflict_abstentions;
    return;
  }
  const std::uint32_t motor = find_form(state, kFormMotor);
  if (motor == kInvalid) {
    state->fault = 2u;
    return;
  }
  Record& record = state->records[motor];
  record.lane[1] = candidate_value;
  record.lane[2] = 1u;
  record.lane[3] = candidate_locus;
  ++record.revision;
  state->raw_motor_value = candidate_value;
  state->raw_motor_valid = 1u;
  state->active_locus = candidate_locus;
}

BCC32_REWRITE_HD inline void retire_partial(Record* partial) {
  clear_record(partial);
}

BCC32_REWRITE_HD inline void age_partials(ResidentRewriteState* state) {
  for (std::uint32_t i = 0u; i < live_record_capacity(state); ++i) {
    Record& record = state->records[i];
    if (record.matter_q8 == 0u || record.lane[0] != kFormPartial) continue;
    if (record.lane[3] != 0xffffffffu) ++record.lane[3];
    if (record.lane[3] > kPartialLifetime) {
      ++state->partials_aged_out;
      retire_partial(&record);
    }
  }
}

BCC32_REWRITE_HD inline bool partial_exists(
    const ResidentRewriteState* state, std::uint32_t identity) {
  for (std::uint32_t i = 0u; i < live_record_capacity(state); ++i)
    if (state->records[i].matter_q8 != 0u &&
        state->records[i].lane[0] == kFormPartial &&
        state->records[i].lane[1] == identity)
      return true;
  return false;
}

BCC32_REWRITE_HD inline void create_partial(ResidentRewriteState* state,
                                            const Record& description) {
  const std::uint32_t identity = description.lane[6];
  if (partial_exists(state, identity)) return;
  const std::uint32_t slot = allocate_record(state);
  if (slot == kInvalid) return;
  Record& partial = state->records[slot];
  partial.lane[0] = kFormPartial;
  partial.lane[1] = identity;
  partial.lane[2] = description.lane[3];
  partial.lane[3] = 0u;
  partial.lane[4] = description.lane[4];
  partial.lane[5] = 0u;
  ++partial.revision;
}

BCC32_REWRITE_HD inline void apply_event_to_descriptions(
    ResidentRewriteState* state, std::uint32_t value) {
  std::uint32_t candidate_count = 0u;
  std::uint32_t candidate_value = 0u;
  std::uint32_t candidate_locus = kInvalid;
  bool conflict = false;

  // Complete resident staged matches before creating new ones. Every partial
  // and description is ordinary Record matter and participates in the same
  // deterministic transaction.
  for (std::uint32_t i = 0u; i < live_record_capacity(state); ++i) {
    Record& partial = state->records[i];
    if (partial.matter_q8 == 0u || partial.lane[0] != kFormPartial) continue;
    ++state->inspected_records;
    // Retirement below is unconditional, so a partial survives at most until
    // the next value presentation through this path -- far shorter than
    // kPartialLifetime. Separating matched from unmatched is what says whether
    // hypotheses are being answered or merely swept.
    if (partial.lane[2] == value)
      ++state->partials_retired_matched;
    else
      ++state->partials_retired_unmatched;
    if (partial.lane[2] == value) {
      const std::uint32_t identity = partial.lane[1];
      for (std::uint32_t d = 0u; d < live_record_capacity(state); ++d) {
        const Record& description = state->records[d];
        if (description.matter_q8 == 0u ||
            description.lane[0] != kFormDescription ||
            description.lane[6] != identity ||
            description.lane[5] < kMatureSupport)
          continue;
        emit_candidate(description.lane[4], d, &candidate_count,
                       &candidate_value, &candidate_locus, &conflict);
        ++state->staged_fires;
      }
    }
    retire_partial(&partial);
  }

  for (std::uint32_t i = 0u; i < live_record_capacity(state); ++i) {
    const Record& description = state->records[i];
    ++state->inspected_records;
    if (description.matter_q8 == 0u ||
        description.lane[0] != kFormDescription ||
        description.lane[5] < kMatureSupport ||
        description.lane[2] != value)
      continue;
    if (description.lane[1] == 1u) {
      emit_candidate(description.lane[4], i, &candidate_count,
                     &candidate_value, &candidate_locus, &conflict);
      ++state->direct_fires;
    } else {
      create_partial(state, description);
    }
  }
  publish_candidate(state, candidate_count, candidate_value, candidate_locus,
                    conflict);
}

BCC32_REWRITE_HD inline std::uint32_t find_sequence(
    const ResidentRewriteState* state) {
  return find_form(state, kFormSequence);
}

BCC32_REWRITE_HD inline Record* ensure_sequence(
    ResidentRewriteState* state) {
  const std::uint32_t existing = find_sequence(state);
  if (existing != kInvalid) return &state->records[existing];
  const std::uint32_t slot = allocate_record(state);
  if (slot == kInvalid) return nullptr;
  Record& sequence = state->records[slot];
  sequence.lane[0] = kFormSequence;
  sequence.lane[1] = 0u;  // event count
  sequence.lane[5] = 0u;  // admitted quiet count
  ++sequence.revision;
  return &sequence;
}

BCC32_REWRITE_HD inline void append_sequence_value(
    ResidentRewriteState* state, std::uint32_t value) {
  Record* sequence = ensure_sequence(state);
  if (sequence == nullptr) return;
  sequence->lane[5] = 0u;
  const std::uint32_t count = sequence->lane[1];
  if (count < 3u) {
    sequence->lane[2u + count] = value;
    sequence->lane[1] = count + 1u;
  } else {
    // A bounded sliding physical history avoids a hidden unbounded episode.
    sequence->lane[2] = sequence->lane[3];
    sequence->lane[3] = sequence->lane[4];
    sequence->lane[4] = value;
  }
  ++sequence->revision;
}

BCC32_REWRITE_HD inline void close_sequence_if_ready(
    ResidentRewriteState* state) {
  const std::uint32_t slot = find_sequence(state);
  if (slot == kInvalid) return;
  Record& sequence = state->records[slot];
  if (sequence.lane[5] < 2u) return;

  const std::uint32_t count = sequence.lane[1];
  const std::uint32_t constructor = find_form(state, kFormConstructor);
  if (constructor != kInvalid && (count == 2u || count == 3u)) {
    const std::uint32_t arity = count - 1u;
    const std::uint32_t first = sequence.lane[2];
    const std::uint32_t second = arity == 2u ? sequence.lane[3] : 0u;
    const std::uint32_t consequence = sequence.lane[1u + count];
    install_or_support_description(state, arity, first, second, consequence);
  }
  clear_record(&sequence);
}

// Variable-bearing programs live in the same Record population as RWR0's
// scalar descriptions. No parallel atom table, rule array, source document, or
// training state exists. A trajectory and a program are only differently
// organized chains of the same fixed-size physical record.
BCC32_REWRITE_HD inline bool record_owner_exists(
    const ResidentRewriteState* state, std::uint32_t owner) {
  for (std::uint32_t i = 0u; i < live_record_capacity(state); ++i) {
    const Record& record = state->records[i];
    if (record.matter_q8 == 0u) continue;
    if (record.lane[0] == kFormConstraintParticipation &&
        record.lane[4] == owner)
      return true;
    if (record.lane[1] == owner &&
        (record.lane[0] == kFormTrajectory ||
         record.lane[0] == kFormTrajectoryTerm ||
         record.lane[0] == kFormTrajectoryPage ||
         record.lane[0] == kFormTrajectoryProvenance ||
         record.lane[0] == kFormTrajectoryParentRoute ||
         record.lane[0] == kFormProgram ||
         record.lane[0] == kFormProgramTerm ||
         record.lane[0] == kFormSpanProgram ||
         record.lane[0] == kFormSpanProgramTerm ||
         record.lane[0] == kFormTransformationWitness ||
         record.lane[0] == kFormProgramFactor ||
         record.lane[0] == kFormProgramEvidence ||
         record.lane[0] == kFormProgramAlternativeTerm ||
         record.lane[0] == kFormProgramWitness ||
         record.lane[0] == kFormConstructionEpisode ||
         record.lane[0] == kFormConstructionEpisodeTerm ||
         record.lane[0] == kFormCausalConstructor ||
         record.lane[0] == kFormCausalConstructorTerm ||
         record.lane[0] == kFormCausalConstructorWitness ||
         record.lane[0] == kFormCausalConstructorDelta ||
         record.lane[0] == kFormCausalProductWitness ||
         record.lane[0] == kFormRevisionTransferWitness ||
         record.lane[0] == kFormRevisionTransferSourceUse ||
         record.lane[0] == kFormRevisionParticipationReader ||
         record.lane[0] == kFormRevisionParticipationReaderTerm ||
         record.lane[0] == kFormRevisionAmbiguityTrace ||
         record.lane[0] == kFormRevisionAmbiguityParticipant ||
         record.lane[0] == kFormRevisionParticipationDelta ||
         record.lane[0] == kFormRevisionEgressWitness ||
         record.lane[0] == kFormRevisionActionIssuance ||
         record.lane[0] == kFormTicketedReturnWitness ||
         record.lane[0] == kFormTicketedReturnWitnessTerm ||
         record.lane[0] == kFormRevisionTransferReturn))
      return true;
  }
  return false;
}

BCC32_REWRITE_HD inline std::uint32_t make_record_owner(
    const ResidentRewriteState* state, std::uint32_t salt) {
  std::uint32_t owner = rewrite_mix(
      static_cast<std::uint32_t>(state->revision),
      static_cast<std::uint32_t>(state->admitted_events), salt);
  if (owner == 0u || owner == kInvalid) owner ^= 0x6d2b79f5u;
  for (std::uint32_t attempt = 0u; attempt < live_record_capacity(state);
       ++attempt) {
    if (!record_owner_exists(state, owner)) return owner;
    owner = rewrite_mix(owner, salt, attempt + 1u);
  }
  return kInvalid;
}

BCC32_REWRITE_HD inline std::uint32_t find_header(
    const ResidentRewriteState* state, std::uint32_t form,
    std::uint32_t owner = kInvalid) {
  for (std::uint32_t i = 0u; i < live_record_capacity(state); ++i) {
    const Record& record = state->records[i];
    if (record.matter_q8 == 0u || record.lane[0] != form) continue;
    if (owner == kInvalid || record.lane[1] == owner) return i;
  }
  return kInvalid;
}

BCC32_REWRITE_HD inline std::uint32_t find_current_trajectory(
    const ResidentRewriteState* state) {
  for (std::uint32_t i = 0u; i < live_record_capacity(state); ++i) {
    const Record& record = state->records[i];
    if (record.matter_q8 != 0u && record.lane[0] == kFormTrajectory &&
        record.lane[3] == 0u)
      return i;
  }
  return kInvalid;
}

BCC32_REWRITE_HD inline std::uint32_t find_owned_block(
    const ResidentRewriteState* state, std::uint32_t form,
    std::uint32_t owner, std::uint32_t ordinal) {
  for (std::uint32_t i = 0u; i < live_record_capacity(state); ++i) {
    const Record& record = state->records[i];
    if (record.matter_q8 != 0u && record.lane[0] == form &&
        record.lane[1] == owner && record.lane[2] == ordinal)
      return i;
  }
  return kInvalid;
}

BCC32_REWRITE_HD inline std::uint32_t trajectory_page_slot(
    const ResidentRewriteState* state, std::uint32_t owner,
    std::uint32_t page) {
  if (page == 0u) {
    std::uint32_t found = kInvalid;
    for (std::uint32_t i = 0u; i < live_record_capacity(state); ++i) {
      const Record& record = state->records[i];
      if (record.matter_q8 != 0u && record.lane[0] == kFormTrajectory &&
          record.lane[1] == owner) {
        if (found != kInvalid) return kInvalid;
        found = i;
      }
    }
    return found;
  }
  return find_owned_block(state, kFormTrajectoryPage, owner, page);
}

BCC32_REWRITE_HD inline std::uint32_t trajectory_page_owner(
    const ResidentRewriteState* state, std::uint32_t owner,
    std::uint32_t page) {
  if (page == 0u) return owner;
  const std::uint32_t slot = trajectory_page_slot(state, owner, page);
  if (slot != kInvalid) return state->records[slot].lane[6];
  std::uint32_t page_owner = rewrite_mix(owner, page, kTrajectoryPageOwnerSalt);
  if (page_owner == 0u || page_owner == kInvalid)
    page_owner ^= 0x6d2b79f5u;
  for (std::uint32_t attempt = 0u; attempt < live_record_capacity(state);
       ++attempt) {
    if (!record_owner_exists(state, page_owner)) return page_owner;
    page_owner = rewrite_mix(page_owner, kTrajectoryPageOwnerSalt,
                             attempt + 1u);
  }
  return kInvalid;
}

BCC32_REWRITE_HD inline std::uint32_t trajectory_page_count(
    const ResidentRewriteState* state, std::uint32_t owner) {
  const std::uint32_t header_slot = trajectory_page_slot(state, owner, 0u);
  if (header_slot == kInvalid) return 0u;
  const std::uint32_t event_count = state->records[header_slot].lane[2];
  const std::uint32_t page_limit =
      event_count == 0u
          ? 1u
          : 1u + (event_count - 1u) / kTrajectoryPageEvents;
  std::uint32_t count = 1u;
  for (std::uint32_t page = 1u;
       page < page_limit && page < live_record_capacity(state); ++page)
    if (trajectory_page_slot(state, owner, page) != kInvalid) ++count;
  return count;
}

BCC32_REWRITE_HD inline std::uint32_t free_record_count(
    const ResidentRewriteState* state) {
  std::uint32_t count = 0u;
  for (std::uint32_t i = 0u; i < live_record_capacity(state); ++i)
    if (state->records[i].matter_q8 != 0u &&
        state->records[i].lane[0] == kFormEmpty)
      ++count;
  return count;
}

BCC32_REWRITE_HD inline bool trajectory_word_at(
    const ResidentRewriteState* state, std::uint32_t owner,
    std::uint32_t index, std::uint32_t* word) {
  const std::uint32_t page = index / kTrajectoryPageEvents;
  if (page >= live_record_capacity(state)) return false;
  const std::uint32_t page_slot = trajectory_page_slot(state, owner, page);
  if (page_slot == kInvalid) return false;
  if (page != 0u) {
    const Record& continuation = state->records[page_slot];
    const std::uint32_t local = index % kTrajectoryPageEvents;
    if (continuation.lane[3] != page * kTrajectoryPageEvents ||
        continuation.lane[4] <= local ||
        continuation.lane[4] > kTrajectoryPageEvents)
      return false;
  }
  const std::uint32_t local = index % kTrajectoryPageEvents;
  const std::uint32_t block = find_owned_block(
      state, kFormTrajectoryTerm, trajectory_page_owner(state, owner, page),
      local / 2u);
  if (block == kInvalid) return false;
  *word = state->records[block].lane[4u + (index % 2u)];
  return true;
}

BCC32_REWRITE_HD inline bool program_term_at(
    const ResidentRewriteState* state, std::uint32_t owner,
    std::uint32_t index, std::uint32_t* word, std::uint32_t* meta) {
  std::uint32_t ordinary_program_count = 0u;
  for (std::uint32_t slot = 0u; slot < live_record_capacity(state); ++slot) {
    const Record& program = state->records[slot];
    if (program.matter_q8 != 0u && program.lane[0] == kFormProgram &&
        (program.lane[7] & kProgramFlagVersionSpace) == 0u &&
        program.lane[1] == owner)
      ++ordinary_program_count;
  }
  if (ordinary_program_count != 1u) return false;
  // A literal Program translated in place from a continued trajectory keeps
  // the resident page directory. Ordinary induced Programs still use the
  // historical root-owned ordinals, so only an actually present and
  // well-formed continuation page changes the physical lookup below.
  std::uint32_t term_owner = owner;
  std::uint32_t ordinal = index / 2u;
  const std::uint32_t page = index / kTrajectoryPageEvents;
  if (page != 0u) {
    const std::uint32_t page_slot =
        find_owned_block(state, kFormTrajectoryPage, owner, page);
    if (page_slot != kInvalid) {
      const Record& continuation = state->records[page_slot];
      const std::uint32_t local = index % kTrajectoryPageEvents;
      if (continuation.lane[3] != page * kTrajectoryPageEvents ||
          continuation.lane[4] <= local ||
          continuation.lane[4] > kTrajectoryPageEvents ||
          continuation.lane[6] == 0u || continuation.lane[6] == kInvalid)
        return false;
      term_owner = continuation.lane[6];
      ordinal = local / 2u;
    }
  }
  std::uint32_t block = kInvalid;
  std::uint32_t block_count = 0u;
  for (std::uint32_t slot = 0u; slot < live_record_capacity(state); ++slot) {
    const Record& term = state->records[slot];
    if (term.matter_q8 == 0u || term.lane[0] != kFormProgramTerm ||
        term.lane[1] != term_owner || term.lane[2] != ordinal ||
        (term.lane[7] & kProgramFlagVersionSpace) != 0u)
      continue;
    block = slot;
    ++block_count;
  }
  if (block_count != 1u) return false;
  const std::uint32_t offset = (index % 2u) * 2u;
  *meta = state->records[block].lane[3u + offset];
  *word = state->records[block].lane[4u + offset];
  return true;
}

BCC32_REWRITE_HD inline void clear_owned_records(
    ResidentRewriteState* state, std::uint32_t form,
    std::uint32_t owner) {
  for (std::uint32_t i = 0u; i < live_record_capacity(state); ++i) {
    Record& record = state->records[i];
    if (record.matter_q8 != 0u && record.lane[0] == form &&
        record.lane[1] == owner)
      clear_record(&record);
  }
}

BCC32_REWRITE_HD inline void clear_trajectory(
    ResidentRewriteState* state, std::uint32_t header_slot) {
  if (header_slot == kInvalid) return;
  Record& header = state->records[header_slot];
  if (header.matter_q8 == 0u || header.lane[0] != kFormTrajectory) return;
  const std::uint32_t owner = header.lane[1];
  const std::uint32_t event_count = header.lane[2];
  const std::uint32_t page_limit =
      event_count == 0u
          ? 1u
          : 1u + (event_count - 1u) / kTrajectoryPageEvents;
  clear_owned_records(state, kFormTrajectoryTerm, owner);
  for (std::uint32_t page = 1u;
       page < page_limit && page < live_record_capacity(state); ++page) {
    const std::uint32_t page_slot = trajectory_page_slot(state, owner, page);
    if (page_slot == kInvalid) continue;
    clear_owned_records(state, kFormTrajectoryTerm,
                        state->records[page_slot].lane[6]);
  }
  clear_owned_records(state, kFormTrajectoryPage, owner);
  clear_owned_records(state, kFormSpanExecutionCursor, owner);
  clear_owned_records(state, kFormSpanExecutionBinding, owner);
  if (state->causal_relation_trajectory_owner == owner) {
    state->causal_relation_trajectory_owner = kInvalid;
    state->causal_relation_generated_index = kInvalid;
    state->causal_relation_trajectory_revision = 0u;
  }
  clear_record(&header);
}

BCC32_REWRITE_HD inline std::uint32_t ensure_current_trajectory(
    ResidentRewriteState* state) {
  const std::uint32_t existing = find_current_trajectory(state);
  if (existing != kInvalid) return existing;
  const std::uint32_t slot = allocate_record(state);
  if (slot == kInvalid) return kInvalid;
  const std::uint32_t owner = make_record_owner(state, 0x7472616au);
  if (owner == kInvalid) {
    state->fault = 3u;
    return kInvalid;
  }
  Record& header = state->records[slot];
  header.lane[0] = kFormTrajectory;
  header.lane[1] = owner;
  header.lane[2] = 0u;  // raw event count
  header.lane[3] = 0u;  // current, not retained exemplar
  header.lane[4] = 0u;  // physical source has not yielded
  // First generated event offset. External prefixes always begin at zero;
  // downstream programs only need to search resident generated matter, not
  // every suffix of the external transcript.
  header.lane[5] = kInvalid;
  header.lane[7] = 0u;  // sticky generated-evidence provenance
  ++header.revision;
  return slot;
}

BCC32_REWRITE_HD inline std::uint32_t ensure_trajectory_page(
    ResidentRewriteState* state, std::uint32_t owner,
    std::uint32_t page) {
  if (page == 0u) return trajectory_page_slot(state, owner, 0u);
  // live_record_capacity(state) is only a physical scan bound. The actual
  // admission limit is the first missing free Record, not this arithmetic
  // bound.
  if (page >= live_record_capacity(state)) {
    state->fault = 4u;
    return kInvalid;
  }
  const std::uint32_t existing = trajectory_page_slot(state, owner, page);
  if (existing != kInvalid) return existing;
  const std::uint32_t page_owner = trajectory_page_owner(state, owner, page);
  if (page_owner == kInvalid) {
    state->fault = 4u;
    return kInvalid;
  }
  const std::uint32_t slot = allocate_record(state);
  if (slot == kInvalid) return kInvalid;
  Record& continuation = state->records[slot];
  continuation.lane[0] = kFormTrajectoryPage;
  continuation.lane[1] = owner;
  continuation.lane[2] = page;
  continuation.lane[3] = page * kTrajectoryPageEvents;
  continuation.lane[4] = 0u;  // words admitted on this physical page
  continuation.lane[5] = 0u;  // page-local rolling digest
  continuation.lane[6] = page_owner;
  continuation.lane[7] = 0u;
  ++continuation.revision;
  return slot;
}

BCC32_REWRITE_HD inline bool append_trajectory_word(
    ResidentRewriteState* state, std::uint32_t word, bool generated) {
  if (!generated) {
    const std::uint32_t previous = find_current_trajectory(state);
    if (previous != kInvalid &&
        (state->records[previous].lane[7] & kTrajectoryWasYielded) != 0u &&
        (state->records[previous].lane[7] & kTrajectoryHasGenerated) == 0u) {
      // Pause is a physical episode boundary even when a quiet observer later
      // lowers the transient yielded latch. New external contact starts a new
      // owner instead of appending to the unanswered prefix.
      clear_trajectory(state, previous);
    } else if (previous != kInvalid &&
               (state->records[previous].lane[7] &
                kTrajectoryHasGenerated) != 0u &&
               state->records[previous].lane[4] != 0u) {
      // A new raw contact starts a fresh owner. The completed generated
      // trajectory has already driven every downstream consumer reachable
      // during autonomous continuation. Retire it instead of retaining
      // generated output as source-like exemplar matter.
      clear_trajectory(state, previous);
    }
  }
  const std::uint32_t header_slot = ensure_current_trajectory(state);
  if (header_slot == kInvalid) return false;
  Record& header = state->records[header_slot];
  const std::uint32_t index = header.lane[2];
  const std::uint32_t page = index / kTrajectoryPageEvents;
  if (page >= live_record_capacity(state)) {
    state->fault = 4u;
    return false;
  }
  const std::uint32_t page_slot =
      ensure_trajectory_page(state, header.lane[1], page);
  if (page_slot == kInvalid) return false;
  const std::uint32_t local = index % kTrajectoryPageEvents;
  const std::uint32_t term_owner =
      trajectory_page_owner(state, header.lane[1], page);
  if (term_owner == kInvalid) return false;
  const std::uint32_t ordinal = local / 2u;
  std::uint32_t block = find_owned_block(
      state, kFormTrajectoryTerm, term_owner, ordinal);
  if (block == kInvalid) {
    block = allocate_record(state);
    if (block == kInvalid) return false;
    Record& term = state->records[block];
    term.lane[0] = kFormTrajectoryTerm;
    term.lane[1] = term_owner;
    term.lane[2] = ordinal;
    ++term.revision;
  }
  state->records[block].lane[4u + (local % 2u)] = word;
  ++state->records[block].revision;
  header.lane[2] = index + 1u;
  if (!generated) header.lane[4] = 0u;
  if (generated &&
      (header.lane[7] & kTrajectoryHasGenerated) == 0u)
    header.lane[5] = index;
  if (generated) header.lane[7] |= kTrajectoryHasGenerated;
  header.lane[6] = rewrite_mix(header.lane[6], word, index);
  ++header.revision;
  if (page != 0u) {
    Record& continuation = state->records[page_slot];
    continuation.lane[4] = local + 1u;
    continuation.lane[5] = rewrite_mix(continuation.lane[5], word, index);
    ++continuation.revision;
  }
  return true;
}

namespace cross_context {
#if defined(__CUDACC__)
static BCC32_REWRITE_HD __noinline__ bool
cross_context_program_is_derived_exact(
    const ResidentRewriteState* state, std::uint32_t program_slot);
#else
BCC32_REWRITE_HD inline bool cross_context_program_is_derived_exact(
    const ResidentRewriteState* state, std::uint32_t program_slot);
#endif
#if defined(__CUDACC__)
static BCC32_REWRITE_HD __noinline__ bool
cross_context_factor_all_mature_programs(ResidentRewriteState* state);
#else
BCC32_REWRITE_HD inline bool cross_context_factor_all_mature_programs(
    ResidentRewriteState* state);
#endif
}
#if defined(__CUDACC__)
static BCC32_REWRITE_HD __noinline__ bool resident_program_authoritative(
    const ResidentRewriteState* state, std::uint32_t program_slot,
    std::uint32_t recursion_depth = 0u);
#else
BCC32_REWRITE_HD inline bool resident_program_authoritative(
    const ResidentRewriteState* state, std::uint32_t program_slot,
    std::uint32_t recursion_depth = 0u);
#endif
#if defined(__CUDACC__)
#define BCC32_CAUSAL_GERMLINE_DISPATCH [[maybe_unused]] static BCC32_REWRITE_HD __noinline__
#else
#define BCC32_CAUSAL_GERMLINE_DISPATCH BCC32_REWRITE_HD inline
#endif
enum class GroundedPairReflectionStatus : std::uint32_t {
  kNotApplicable = 0u,
  kReady = 1u,
  kBlocked = 2u,
};

struct GroundedPairReflectionPlan {
  std::uint32_t episode_owner = kInvalid;
  std::uint32_t program_owner = kInvalid;
  std::uint32_t left_owner = kInvalid;
  std::uint32_t right_owner = kInvalid;
  std::uint32_t extent = 0u;
  std::uint32_t program_records = 0u;
  std::uint32_t episode_records = 0u;
};

BCC32_CAUSAL_GERMLINE_DISPATCH GroundedPairReflectionStatus
preflight_grounded_pair_reflection(
    const ResidentRewriteState* state, std::uint32_t left_slot,
    std::uint32_t right_slot, std::uint32_t program_digest,
    std::uint32_t program_records, GroundedPairReflectionPlan* output);
BCC32_CAUSAL_GERMLINE_DISPATCH void commit_grounded_pair_reflection(
    ResidentRewriteState* state, std::uint32_t left_slot,
    std::uint32_t right_slot, std::uint32_t program_slot,
    const GroundedPairReflectionPlan& plan);
BCC32_CAUSAL_GERMLINE_DISPATCH GroundedPairReflectionStatus
preflight_grounded_span_reflection(
    const ResidentRewriteState* state, std::uint32_t left_slot,
    std::uint32_t right_slot, std::uint32_t program_digest,
    std::uint32_t program_records, std::uint32_t product_terms,
    GroundedPairReflectionPlan* output);
BCC32_CAUSAL_GERMLINE_DISPATCH void commit_grounded_span_reflection(
    ResidentRewriteState* state, std::uint32_t program_slot,
    const GroundedPairReflectionPlan& plan);
BCC32_REWRITE_HD inline std::uint32_t grounded_construction_topology_digest(
    const ResidentRewriteState* state, const Record& program);
BCC32_CAUSAL_GERMLINE_DISPATCH bool reflect_grounded_program_construction(
    ResidentRewriteState* state, std::uint32_t program_slot,
    const Record& source);
BCC32_CAUSAL_GERMLINE_DISPATCH bool settle_causal_germline_constructor(
    ResidentRewriteState* state);
BCC32_CAUSAL_GERMLINE_DISPATCH bool causal_germline_product_authoritative(
    const ResidentRewriteState* state, std::uint32_t program_slot);
BCC32_CAUSAL_GERMLINE_DISPATCH bool
causal_germline_span_product_authoritative(
    const ResidentRewriteState* state, std::uint32_t program_slot);
BCC32_CAUSAL_GERMLINE_DISPATCH bool settle_causal_germline_pending(
    ResidentRewriteState* state);
BCC32_CAUSAL_GERMLINE_DISPATCH bool settle_resident_revision_transfer(
    ResidentRewriteState* state);
BCC32_CAUSAL_GERMLINE_DISPATCH bool
resident_revision_transfer_product_authoritative(
    const ResidentRewriteState* state, std::uint32_t program_slot,
    std::uint32_t recursion_depth = 0u);
BCC32_CAUSAL_GERMLINE_DISPATCH bool resident_program_shadowed_by_revision(
    const ResidentRewriteState* state, std::uint32_t program_slot,
    std::uint32_t recursion_depth = 0u);
BCC32_REWRITE_HD inline bool match_program_prefix(
    const ResidentRewriteState* state, const Record& program,
    const Record& trajectory, std::uint32_t* next_word,
    bool* next_is_unbound);

// Grounded construction preflight/commit helpers are defined in the closure
// unit below, but induction is included first and needs their declarations.
BCC32_REWRITE_HD inline std::uint64_t grounded_record_headroom(
    const ResidentRewriteState* state);
BCC32_REWRITE_HD inline bool reserve_grounded_record_matter(
    ResidentRewriteState* state, std::uint32_t required);

#include "bcc32_resident_causal_relation_source_witness.cuh"
#include "causal_rewrite_program_induction.inl"
#include "bcc32_resident_grounded_construction_closure.cuh"
#include "bcc32_resident_grounded_difference_executor.cuh"
#include "bcc32_resident_causal_germline_runtime.cuh"
#include "bcc32_resident_revision_transfer.inl"
#include "causal_rewrite_program_execution.inl"
#include "causal_rewrite_event_admission.inl"

#include "causal_rewrite_universe_surface.inl"

}  // namespace substrate::bcc32::causal_rewrite

#include "bcc32_resident_cross_context_anti_unification.cuh"

#undef BCC32_REWRITE_HD
