# Graph-neutral organism construction workbench

This directory contains fast experiments for altering, replacing, deleting, and
falsifying candidate mechanisms for one causal organism. Most are reference-only;
small C++ contracts may compile production mathematical headers directly. No
Workbench mechanism is architecture authority merely because it exists here or in
Target V5. The organism runners use no prompt or LLM runtime, and the fast host
contracts use no CUDA or per-Recipe compilation unless a particular falsifier
requires them.

## Construction doctrine: capability is the invariant, architecture is disposable

Target V5, the current Direct runtime, Gamma growth phases, Recipes, eligibility
traces, route transport, staged curricula, and every other named mechanism are
working hypotheses. They may be changed, simplified, fused, replaced, or deleted
when a cheaper or better mechanism improves the end organism while preserving the
required evidence boundaries. Do not preserve an abstraction because a plan,
paper, architecture volume, prior experiment, or existing implementation names it.

The Workbench is the construction side of the project. Its job is to ask, rapidly
and repeatedly: does this mechanism improve the organism; is there a simpler or
more general mechanism; is the mechanism needed at all; and what falsifier would
kill it? A curriculum may be staged, reordered, expanded, compressed, or replaced.
Developmental growth may be altered. A biological analogy is evidence for a
hypothesis, not a mandate to reproduce anatomy. Even a currently successful
mechanism remains deletable if another design produces a stronger, cheaper, more
general continuing Adult.

Apply the first-principles engineering order before optimization: challenge the
requirement, delete the part or process if possible, simplify the surviving path,
then optimize and automate only what remains. Do not optimize a mechanism merely
because Target V5 currently contains it. A replacement wins only if the complete
organism evidence is stronger; when it wins, remove the displaced internal path
instead of preserving architectural legacy for its own sake.

For cognition work, and especially language work, internal elegance is not an
acceptance criterion. Every production mechanism change must have a named
language-facing or cognition-facing assay that measures an actual behavioral delta
such as held-out composition, recursive reuse, answer structure/extent, correction,
clarification, source-sensitive belief revision, initiative, transfer, or another
usable Adult capability. A biology-grounded change that leaves externally usable
capability unchanged is research input, not sufficient production progress.

The retained invariants are behavioral and epistemic rather than architectural:

- one continuing causal organism must acquire and use capability through lawful
  experience rather than host answer selection;
- world evidence must come from authenticated external contact/consequence, not
  from expectations, replay labels, self-generated output, or an expected answer;
- public action must remain ordinary bodily action rather than a privileged host
  semantic channel;
- held-out, withdrawal, lesion, replay, stale/forged, and no-teach controls must
  continue to distinguish genuine learned capability from leakage;
- the Life-Function objective is evaluated by whether a change makes the resulting
  AI more capable, adaptive, efficient, robust, and general, not by fidelity to a
  frozen architecture document.

Current implementation candidates, not commitments:

- `DirectGenomeV1` is the current species/development blueprint.
- `ReferenceRecipeIrV1` is an experimental numeric Recipe/state IR.
- `ResidentRecipeIrProgram.vcurrent` is a later production IR.
- Translation and physical Direct parity are currently undefined/not run.
- `interchange.py` accepts bounded numeric scenario operations only. It cannot
  declare Recipes, enqueue goals, provide expected output, or emit surface bytes.
- `reference_contract_1610.py` currently provides a deterministic endogenous
  executor, authenticated contacts and consequences, resources, causal-difference
  credit, checkpointing, withdrawal, condensation, and per-byte ancestry.

Any item in the candidate list may disappear if a replacement wins the relevant
behavioral/evidence tournament. Do not build compatibility layers for deleted
ideas unless a live external interface actually requires compatibility.

The former symbolic Foundry/VM/starting-organism family was deleted as one closed
legacy dependency set. It is not a donor authority or regression requirement.

Run the authoritative contract:

```bash
./run_reference_contract_1610.sh
```

Freeze an externally authored numeric starting state, then run authenticated
contacts against it. The freezer rejects active queries, actions, occurrences,
continuations, trace, credit, or other current/future cognition:

```bash
python3 starting_state_freezer.py state.json \
  --checkpoint-out state.checkpoint \
  --manifest-out state.manifest.json
python3 interchange.py scenario.json --checkpoint state.checkpoint --out run.json
```

The freezer, manifest, and runtime are separate boundaries. The manifest contains only
hashes and bounded counts; it never republishes state, learned surface, or output. A
complete live checkpoint is created only by resident execution and is restored through
the versioned checkpoint path, never through the starting-state authoring interface.

Run the complete retained graph-neutral workbench:

```bash
./run_foundry_workbench.sh
```

Run the architecture-construction language factory first:

```bash
./run_language_mastery_factory_fast.sh
```

This is the primary mechanism-iteration lane. It uses one unified simulated Adult
behind an opaque contact membrane and currently checks developmental emergence,
held-out composition, hierarchical Causal Programs, consequence/state-dependent
answer selection, source-withdrawal consolidation, fixed-capacity interference,
and the deletion tournament. It is intentionally sub-second and has no optimizer,
grammar-rule API, transformer, token objective, or expected answer. A candidate
that fails here does not earn a Direct/CUDA build.

Run the selected broader Adult regression next when a candidate wins the factory:

```bash
./run_language_mastery_fast.sh
```

Run the complete language/organism workshop only as the wider synchronization
regression:

```bash
./run_language_workbench.sh
```

The language runner launches its read-only assays concurrently, captures each
result independently, then replays output in a stable order and fails if any
child failed. Expensive Direct builds, Adult runtime, sanitizer, and hardware
economics remain separate receipts so ordinary mechanism iteration stays in the
few-second lane.

`reference_network_minimal_closure_verify.cpp` is a fast production-math
falsifier, not a parallel cognitive stack. It binds exact live Occurrences and
formal-port couplings through the Direct relational-Network header. Its current
control proves that one active frontier can contain two disconnected coactive
Networks plus one uncoupled Occurrence, invariant to frontier order and atomic
under stale input. Eligibility is derived as distributed transient Network
geometry; it changes the exact active-N identity while the reusable structural
condensation witness remains eligibility-independent, and creates neither
membership nor credit. The strict portability audit compiles and runs this host contract. GPU,
continuing-Adult attachment, causal settlement and language capability remain
separate RED evidence.

The standalone lowering assay compiles without configuring the full project. It
compares the current dense scalar and half-WMMA mechanics and charges BVH build
cost before nominating a spatial/incidence workload. No GPU returns exit 77;
that is a skip, not hardware evidence:

```bash
python3 hardware_lowering_economics_verify.py
```

The reference runners are CPU-only and bounded by a hard 60-second timeout. A passing named
contract is logical reference evidence only. It is not Adult language capability,
human-level language, production parity, or a constitutional graph transition.
