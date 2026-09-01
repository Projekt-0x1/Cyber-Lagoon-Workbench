# Decision Grounding Gate — Foundry Workbench

Status: normative for substantive architecture changes under `hardware_native/tools/foundry_workbench/`.

The purpose of this gate is not citation ceremony. It prevents architecture from acquiring accidental authority through attractive abstractions, test fixtures, host semantics, or one-off heuristics. Every substantive mechanism change must upgrade the grounded causal model of the Adult before code lands.

## 1. Trigger

Run this gate before changing any mechanism that can alter persistent organism state, action selection/inhibition, learning/credit/consolidation, language grounding/composition, self-model/metacognition/replay, source/reason credibility, allostasis/controllability/resource allocation, cultural transmission, experiment selection/causal inference, contextual memory routing, partner/common-ground state, checkpoint semantics, or the canonical Life Function curriculum.

Pure refactors that provably preserve state transitions may cite the prior grounded decision.

## 2. Pass A — mechanism-specific primary grounding

Before implementation, write or update a dated research decision note stating:

1. **Functional RED** — the exact missing causal capability, never “more intelligence” or phenotype-only language.
2. **Primary grounding** — mechanism-specific empirical/formal work constraining the proposal.
3. **Incumbent causal owner** — which resident mechanism already owns world truth, action credit, language meaning, source history, body regulation, etc.
4. **Novel synthesis boundary** — literature/repo support versus engineering synthesis.
5. **Alternative / ablation** — at least one simpler mechanism that could mimic the phenotype and must be falsified.

A decision note that cannot identify the incumbent owner is not ready to implement.

## 3. Pass B — Sapolsky-style nested causation

Inspect the same event across immediate body/resource/interoceptive state, current controllability, recent consequence/stress/recovery history, relationship/source history, developmental learning history, current social/cultural context, and longer-lived consolidated competence.

State explicitly which variables may change **control policy**, which may change **memory routing/expression**, and which may change **evidence**. Fatigue, controllability, contextual state and relationship state must not silently rewrite semantic/world evidence unless a separately grounded mechanism owns that update.

## 4. Authority ledger

Every new state variable/computation must have one authority class:

- `LIVED_EVIDENCE` — may update one resident causal relation only after authenticated qualifying contact;
- `CONTROL_PRIORITY` — may rank/defer/ask/observe but cannot establish truth;
- `SIMULATION_0` — replay/counterfactual/cultural hypothesis with epistemic authority zero;
- `PROVENANCE` — identifies source/history/context without changing truth;
- `MEMORY_ROUTING` — may decide which existing competence memory is expressed/updated but cannot itself add world evidence;
- `TRANSIENT_COMPUTATION` — rebuildable and non-checkpointed unless it changes future causal interpretation.

If a value serves more than one class, split it. No global scalar may stand simultaneously for truth, utility, confidence, fatigue, credibility, context and control.

## 5. Intervention / replay firewall

Any imagined, replayed, counterfactual, culturally transmitted or host-proposed object starts at authority zero. It may nominate an already-earned action/observation, allocate private computation, create an information need, or select a low-risk intervention after current affordance/control/resource gates permit it.

It may not by itself create world-transition evidence, settle a reason/source as true, calibrate self-reliability, confirm a cultural procedure, create eligibility for an unexperienced event, change action credit, split external world context, or overwrite body state. Only the appropriate authenticated lived consequence can cross those boundaries.

## 6. Falsifiers before code

Every decision note must include a positive capability case, authority-leak negative case, source-withdrawal case when provenance is involved, non-independent/reafferent case when consequence is involved, checkpoint/restore case, body/resource/controllability contrast when control is involved, replay/counterfactual no-evidence case when simulation is involved, and a matched simpler shortcut/fixture ablation when relevant.

### Category-separation firewall

Before introducing a shared latent variable, score, context, confidence, trust value, social model, or memory key, state exactly **which causal relation it denotes**. The following objects are not interchangeable:

- external state-transition prediction;
- consequence/value/effect;
- self-performance/metacognitive confidence;
- body/resource/controllability state;
- source competence or procedural predictive history;
- proposition/message truth;
- consequence-certified shared access/common ground;
- inferred belief content;
- shared goal/intention;
- memory-routing context;
- action/control priority.

Mandatory rules:

1. **World prediction ≠ self confidence.** External WORLD_CONTEXT split pressure must come from an incumbent world-transition prediction captured before the outcome. Metacognitive confidence cannot substitute for it.
2. **Transition truth ≠ outcome value.** A destination mismatch may revise external transition structure; `effect`, reward, valence or success may alter utility/self competence but cannot stand in for external state truth.
3. **WORLD_CONTEXT ≠ CONTROL_CONTEXT.** Body, target, resource, controllability or partner participation may route performance/control memory while leaving external procedural-reason evidence in the same world context.
4. **Source competence ≠ message truth.** A source/reason/action history may alter testimony applicability or procedural predictiveness but does not establish the proposition itself as true.
5. **Shared access ≠ belief.** Consequence-certified common ground may establish what a partner had access to. Missing later access may establish staleness/applicability uncertainty. It does not mint `partner believes X` or `false belief X`.
6. **Shared access ≠ shared intention.** Jointly witnessing an event does not by itself establish a shared goal, commitment or role structure.
7. **Memory partition ≠ discovered ontology.** A context identity is a memory-routing hypothesis unless a separately grounded owner has evidence for an external latent cause.
8. **One consequence, one write per causal relation.** If one lived return legitimately updates world transition, self competence and procedural reason history, each owner is written once. Wrapper composition must not double-credit the same relation.
9. **Pre-outcome receipt.** Any mechanism that evaluates prediction error after a learning contact must preserve the incumbent prediction before that contact can mutate the predictor.

Any architecture or verifier that collapses one of these distinctions is RED even if downstream phenotype metrics improve.

### Causal-generalization ablation

Whenever a learned key/category/strategy/concept/compression class transfers across events, include a matched case preserving superficial similarity while changing a causally relevant distinction, such as:

- same action/risk geometry with versus without a live testable prediction;
- same surface overlap with different hierarchical role;
- same source identity with different action/procedure relation;
- same stressor magnitude with different controllability;
- same replay content with versus without prior lived eligibility.

If the mechanism generalizes across the causal distinction, the decision is RED even when phenotype metrics improve. Learned abstraction must compress invariants, not erase variables that determine what evidence the event can supply.

### Search-policy mixture ablation

Do not assume one authored search/experiment objective is universally rational when primary grounding supports adaptive strategy mixtures. If two zero-authority search policies can trade diagnosticity against cognitive effort or confirmation cost, test them in matched contexts and require context-conditional learned policy competence.

A policy architecture is RED if:

- one fixed objective wins by construction rather than lived diagnostic history;
- learned policy preference transfers across a resource/control/ambiguity distinction not supported by evidence;
- past policy success relaxes a present safety veto;
- a policy choice itself writes world/self/reason evidence;
- suppressed policy state falls through to arbitrary action rather than ASK/OBSERVE/DEFER.

### Contextual-inference merge/split ablation

Any mechanism that makes competence, memory expression, source/reason weighting, action confidence or experiment skill context-dependent must prove both **transfer** and **separation**.

Required matched cases:

- same external world under different body/control state: WORLD_CONTEXT must remain stable while CONTROL_CONTEXT may differ;
- non-identical external episodes with sufficient world-cue overlap may provisionally reuse a WORLD_CONTEXT;
- only a supported **pre-outcome transition destination mismatch** plus world-cue novelty and independent return may pressure a WORLD_CONTEXT split;
- contradiction under effectively identical world cues must update the same WORLD_CONTEXT rather than spawning an explanatory escape memory;
- non-independent/replay mismatch cannot create a WORLD_CONTEXT split;
- returning to prior world cues must recover prior world-context evidence without retraining;
- self/strategy/policy competence may differ across CONTROL_CONTEXT without changing procedural reason truth when WORLD_CONTEXT is unchanged;
- global compatibility summaries must not override current typed contextual authority.

Exact-context lookup is RED because it prevents transfer. Universal pooling is RED because it causes interference. Unlimited surprise-triggered splitting is RED because it turns context into an excuse generator.

### Partner epistemic-access ablation

Any partner model derived from common ground must prove:

- partner presence alone creates zero shared-access evidence;
- only consequence-certified positive independent shared episodes create access evidence;
- P cannot borrow Q's access history;
- a world update not jointly witnessed can make P's prior access stale without asserting a proposition-level belief;
- genuinely sharing the updated episode can close staleness;
- negative/non-independent public outcomes create no shared-access evidence;
- source withdrawal can deactivate current applicability while preserving historical access provenance;
- access state has authority zero over world truth.

A test or API using `belief`, `false_belief`, or `perspective` terminology is RED unless a separate grounded mechanism has actually earned proposition-level mental-state evidence.

A test that only checks an expected output string is insufficient for architecture authority.

## 7. Shared-life curriculum law

Developmental capability changes must extend the canonical append-only Life Function chronology using existing ordinary life lanes whenever possible. Do not add observer opcodes such as `INSTRUCTION`, `TRUST`, `REASON_TRUE`, `METACOGNITION`, `LANGUAGE_ID`, `POLICY`, `REGIME`, `CONTEXT`, `BELIEF`, `FALSE_BELIEF`, or `CULTURE` merely to make a test pass.

Held-out means not experienced before the chronological transfer point. No reset/rebirth/train-test switch.

## 8. End-to-end receipt

Every substantive mechanism needs an executable receipt reaching the real organism lifecycle, e.g.:

`authenticated contact -> incumbent owner -> new control/memory-routing mechanism -> real motor/inquiry -> independent consequence -> separated owner updates -> checkpoint continuity`

Where one consequence updates several relations, verify each owner separately and verify **one write per relation**. Where context changes memory routing, verify the world-transition record survives regardless of merge/split. Do not infer success from an aggregate score.

Canonical union command:

```bash
hardware_native/tools/foundry_workbench/run_foundry_frontier_union.sh
```

Do not call a change runtime GREEN unless the relevant receipts actually ran and passed.

## 9. Shared-main concurrency law

Before landing: re-read current `main`; preserve peer changes and donor authority; prefer exact-path strangler/donor preservation for risky authority changes; never force-push over peer history; re-run the head guard after the final write. Unexpected peer mechanisms are integration inputs, not cleanup targets.

## 10. Grounding must ratchet

If a later mechanism reveals an earlier grounded decision was too broad, conflated owners, or admitted a shortcut: update/supersede the mechanism and grounding note, and add the newly discovered general falsifier here.

Grounding is cumulative architecture state. It must become stricter as the Adult becomes more capable.