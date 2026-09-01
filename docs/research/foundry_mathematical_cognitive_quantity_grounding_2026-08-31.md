# Foundry grounding: mathematical cognitive quantity without magic emergence

Status: research/design grounding only. This is not a capability claim and does not promote or freeze a substrate ontology.

## Project question

If the Adult is represented as replaceable mathematics rather than literal artificial cells, what should **quantity** mean, and how do we preserve organismal/human-like cognition without assuming that an underspecified substrate will invent useful mechanisms?

The working answer is:

> Offer explicit deterministic cognitive/developmental mechanisms; measure quantity as the scale and dimensionality of their causal composition; call a macroscopic property emergent only when it appears from executing those fully specified mechanisms together and is not tractable by inspection.

Emergence is an observation about the composed deterministic system, not an implementation primitive and not a substitute for mechanism.

## 1. Sapolsky: deterministic complexity does not create missing causes

Sapolsky's Stanford Human Behavioral Biology lectures 21 and 22 use chaos, reductionism, fractal magnification and emergence to show why complex biological systems can be deterministic yet difficult or impossible to predict from casual inspection of their parts. Small state differences can be magnified through nonlinear interacting systems. The relevant engineering lesson is **not** to omit mechanisms. It is that explicit mechanisms can compose into macroscopic behavior whose trajectory must be executed/experimented on rather than mentally inferred.

- Robert Sapolsky, Stanford Human Behavioral Biology, Lecture 21, *Chaos and Reductionism*: https://www.dnatube.com/courses/stanfordhumanbio
- Robert Sapolsky, Stanford Human Behavioral Biology, Lecture 22, *Emergence and Complexity*: https://www.dnatube.com/courses/stanfordhumanbio
- Transcript mirror for Lecture 22, including deterministic/aperiodic chaotic systems and fractal magnification: https://www.iomindfulness.org/video-directory-r/emergence-and-complexity

Project translation:

```text
explicit mechanism A
+ explicit mechanism B
+ explicit mechanism C
+ recurrence
+ developmental history
+ large interacting quantity
= potentially surprising whole-system behavior
```

Never:

```text
generic substrate
+ hope
= cognition
```

## 2. Human cognition is population/composition-level, not one-neuron-one-function

### Mixed selectivity and useful representational dimensionality

Rigotti et al. found that heterogeneous mixed selectivity in prefrontal populations creates high-dimensional representations and increases the repertoire of input-output functions that downstream readouts can implement. Behavioral errors were associated with reduced dimensionality.

- Rigotti et al., *The importance of mixed selectivity in complex cognitive tasks*, Nature 2013: https://www.nature.com/articles/nature12160

Engineering constraint: raw element count is insufficient. One quantity axis must capture how many independently useful distinctions/combinations the current population-level computation can realize.

### Neural manifolds constrain huge populations into usable collective state spaces

Langdon, Genkin & Engel and later Perich, Narain & Gallego review how large neural populations often occupy structured lower-dimensional manifolds shaped by connectivity, task and behavior. The manifold view is complementary to circuit mechanisms rather than a replacement for causal circuit explanations.

- Langdon, Genkin & Engel, *A unifying perspective on neural manifolds and circuits for cognition*, Nature Reviews Neuroscience 2023: https://www.nature.com/articles/s41583-023-00693-x
- Perich, Narain & Gallego, *A neural manifold view of the brain*, Nature Neuroscience 2025: https://www.nature.com/articles/s41593-025-02031-z

Engineering constraint: we should not pay physical cost for every potential degree of freedom if cognition actually occupies a much smaller structured active subspace. Workbench and hardware may represent inactive potential procedurally and materialize only the currently reachable causal subspace.

### Flexible cognition reuses and composes learned components

Tafazoli et al. showed in monkeys that shared neural subspaces representing sensory and motor information can be flexibly reused and compositionally combined across tasks. Driscoll et al. found reusable dynamical motifs in recurrent networks trained for flexible multitask computation.

- Tafazoli et al., *Building compositional tasks with shared neural subspaces*, Nature 2025/2026: https://www.nature.com/articles/s41586-025-09805-2
- Driscoll, Shenoy & Sussillo, *Flexible multitask computation in recurrent networks utilizes shared dynamical motifs*, Nature Neuroscience 2024: https://www.nature.com/articles/s41593-024-01668-6

Engineering constraint: reusable offered building blocks are scientifically plausible and computationally valuable. Recipes/Networks remain current hypotheses for such reusable composition. If their storage representation is replaced, their earned compositional function must survive.

## 3. Recurrence, memory and internally generated trajectories are functional constraints

Strong recurrent connectivity supports complex computation but requires stabilizing mechanisms; memory and planning research repeatedly treats cognition as dynamics over population states rather than static lookup.

- Sadeh & Clopath, *Inhibitory stabilization and cortical computation*, Nature Reviews Neuroscience 2021: https://www.nature.com/articles/s41583-020-00390-z
- Chaudhuri & Fiete, *Computational principles of memory*, Nature Neuroscience 2016: https://www.nature.com/articles/nn.4237
- Kaefer et al., *Replay, the default mode network and the cascaded memory systems model*, Nature Reviews Neuroscience 2022: https://www.nature.com/articles/s41583-022-00620-6
- Jensen, Hennequin & Mattar, *A recurrent network model of planning explains hippocampal replay and human behavior*, Nature Neuroscience 2024: https://www.nature.com/articles/s41593-024-01675-7
- Buckner, *The Role of the Hippocampus in Prediction and Imagination*, Annual Review of Psychology 2010: https://www.annualreviews.org/content/journals/10.1146/annurev.psych.60.110707.163508

Engineering constraint: a mathematical Adult needs explicit mechanisms capable of persistence across time, recurrent interaction, internally generated/replayed trajectories, prediction, competition/stabilization and compositional reuse. Whether these remain named Recipes, Networks, replay structures, or become better mathematics is an empirical Foundry question.

## 4. Development supplies machinery; development is not magic

Developmental gene-regulatory networks are explicit information-processing systems. Ben-Tabou de-Leon & Davidson describe the regulatory genome as a large parallel computational device whose interconnected cis-regulatory modules implement developmental specification.

- Ben-Tabou de-Leon & Davidson, *Gene Regulation: Gene Control Network in Development*, Annual Review of Biophysics 2007: https://www.annualreviews.org/content/journals/10.1146/annurev.biophys.35.040405.102002

Engineering constraint: the Genome/Life Function analogy should mean **compact offered developmental programs produce larger phenotype through deterministic execution under lived inputs**. It does not mean a minimal seed is expected to invent missing cognition.

## 5. Quantity in the mathematical paradigm

Do not define cognitive quantity by one scalar such as neuron count, cell count, graph nodes or parameter count. Use independent axes that measure what the offered mechanisms can actually do.

Candidate Workbench quantity vector `Q`:

```text
Q = (
  D_active,       # effective active representational dimension
  N_distinct,     # independently addressable learned distinctions/relations
  I_tick,         # causally effective interaction events per organism tick
  R_depth,        # recurrent/iterative causal depth before settlement
  C_width,        # simultaneously composable reusable mechanisms/subspaces
  C_depth,        # hierarchical compositional depth
  H_span,         # causally reachable lived-history span
  P_branch,       # resident counterfactual/predictive alternatives concurrently supported
  W_work,         # exact touched mathematical work
  S_persist,      # persistent learned description size
  S_active,       # currently materialized computational state size
  X_transfer,     # held-out reuse/generalization across contexts/tasks
  K_interference  # competence retained under accumulating unrelated learning
)
```

These are candidate measurement dimensions, not permanent architecture nouns. Delete, merge or replace them if better falsifiable measurements are found.

### Why this vector is preferable to raw cell count

A system can have a gigantic nominal state space but little useful dimensionality or compositionality. Conversely, compact reusable mathematics can generate large active state spaces and many interactions only when demanded. Neuroscience demonstrates both high-dimensional mixed selectivity and lower-dimensional collective manifolds; this makes raw population size an inadequate proxy for useful cognition.

The 80-billion-neuron human scale remains a **pressure/reference point**, not a mapping requirement. Foundry should ask how much effective cognitive quantity a candidate mathematics provides per byte, joule and unit of touched work.

## 6. A quantitative emergence test

Do not label a phenomenon emergent merely because it is complicated. A Foundry result qualifies as an emergent whole-system effect only when all of the following hold:

1. Every participating mechanism and external input path is explicitly represented and deterministic under complete state.
2. No host solver, semantic oracle, hidden goal, expected answer or unmodeled stochastic authority supplies the behavior.
3. The phenomenon is absent or materially altered under one or more causal interventions on the participating mechanisms.
4. It appears or changes as composition/quantity/history is scaled while mechanism definitions remain fixed.
5. Replaying the exact complete state/input schedule reproduces the trajectory exactly under the same reference semantics.
6. The macroscopic property is not directly installed as a privileged state variable or output rule.

Useful scale sweeps should vary axes independently, for example:

```text
interaction quantity:       1x, 4x, 16x, 64x
active dimensionality:      low → high
recurrent depth:            shallow → deep
compositional width:        1 → many mechanisms
history length:             short → long curriculum
resource pressure:          abundant → constrained
persistent description:     direct → compressed/generative
```

The objective is to find phase changes, interference, unexpected capabilities and failure regimes produced by **known mechanics**.

## 7. Workbench development law

The Workbench should behave like an engine laboratory:

```text
research phenotype/function
        ↓
choose/derive explicit mathematical mechanism
        ↓
bench-test mechanism
        ↓
compose with existing mechanisms
        ↓
sweep quantity axes
        ↓
observe expected + unexpected behavior
        ↓
causal interventions / hostile controls
        ↓
simplify, replace, or retain mechanism
```

Most work should remain in deterministic fast reference semantics. GPU compilation is not part of the hypothesis inner loop. Hardware compilation occurs only after a mathematical mechanism/composition earns its place.

## 8. Hardware-neutral mathematics

The source mathematical mechanism should not inherit a cellular-automaton or neuron-object implementation unless that representation wins a measured tournament.

Possible lowerings include, depending on mathematical shape:

- dense/tiled algebra -> Tensor Core style operations;
- sparse incidence -> indexed sparse gather/scatter;
- spatial/incidence search -> BVH/RT traversal if measured beneficial;
- compact binary/integer state -> bit/integer execution;
- sequential dynamical state -> state-space/scan execution;
- irregular small causal fronts -> ordinary scalar/CUDA execution.

The backend is free to change while the organism's causal/phenotype contract remains fixed.

## 9. Immediate architecture implications

1. **Keep Recipes and Networks as offered donor mechanisms.** Do not delete them because a generic mathematical substrate sounds elegant.
2. **Question their representation.** Recipe-specific cells, permanent graph storage, route identity and other storage classes may be deleted if equivalent causal function survives.
3. **Do not build a CA replacement.** Rule 110 is an irreducibility analogy only.
4. **Add quantity instrumentation to the fast Workbench before production refactor.** We need to know whether a representation actually increases useful causal computation per resource unit.
5. **Prefer qualitative mechanism gains before brute quantity.** A mechanism that creates more reusable/composable computation at lower cost should replace a worse one even if it uses fewer nominal elements.
6. **Unexpected behavior is a research result.** Reproduce it, intervene on it and determine which composition/scale causes it; do not call it magic and do not immediately hard-code it.
7. **Whole-Adult runs remain integration evidence, not daily development.** Most candidate mechanisms should be killed or improved in partial assemblies; full Life Function development is periodic integration/qualification.

## 10. Falsifiers for the mathematical-compression direction

Reject or weaken the current compression hypothesis if any of these repeatedly occur:

- compact mathematical representation destroys online plasticity or held-out composition;
- compression reduces effective active dimensionality needed for cognition;
- rematerialization costs exceed keeping useful structures resident;
- state migration requires global semantic understanding/host logic;
- supposedly generic math recreates current Recipes/Networks as hidden metadata with no real resource gain;
- hardware lowerings cannot preserve deterministic causal traces or resource laws;
- useful cognition scales mainly with permanently materialized state rather than reusable/generative structure.

No percentage compression target (90%, 99%, 99.9%) is constitutional. The winner is the smallest/cheapest mathematics that preserves or improves the organism phenotype and continual development.

## 11. Whole Adult as a stateful mathematical operator

The strongest current abstraction is not "a graph made mathematical" and not "a universal rewrite substrate."  It is one continuing stateful system:

```text
(z_(t+1), output_t) = Phi_G(z_t, authenticated_input_t)
```

`G` is the content-free Genome/mechanism specification. `z_t` is the complete individual-specific state required to determine a lawful future: learned mechanism parameters/program deltas, unresolved causal commitments, current body/resource state and whatever compact causal certificates remain future-relevant.  A learned change to a Recipe, causal program, predictor or source relation is therefore a change inside `z_t`, not a second "model update" system.

This operator view is mathematically ordinary.  Rutten's coalgebraic treatment covers deterministic transition/dynamical systems via state-transition maps and behavioural equivalence; the behavioral-systems tradition explicitly treats the system independently of a preferred input/output/state-space representation.  Kalman's minimal-realization work, predictive-state representations and computational mechanics independently show that state representation can be reduced by preserving future behaviour rather than historical implementation detail.

- Rutten, *Universal coalgebra: a theory of systems*: https://doi.org/10.1016/S0304-3975(00)00056-6
- Kalman, *Mathematical Description of Linear Dynamical Systems*: https://doi.org/10.1137/0301010
- Littman & Sutton, *Predictive Representations of State*: https://papers.nips.cc/paper_files/paper/2001/hash/1e4d36177d71bbb3558e43af9577d70e-Abstract.html
- Shalizi & Crutchfield, *Computational Mechanics: Pattern and Prediction, Structure and Simplicity*: https://arxiv.org/abs/cond-mat/9907176
- Markovsky, *Behavioral systems theory in data-driven analysis, signal processing, and control*: https://doi.org/10.1016/j.arcontrol.2021.09.005

The project-specific equivalence is stricter than observable output equivalence.  Two stored histories may share one compact state only if all admissible future authenticated input schedules and interventions preserve both the future causal trace **and future update authority**.  Same answer now is insufficient if delayed consequence, source withdrawal, lesion, repair or later credit can distinguish the histories.

This is consistent with Maizels & Briscoe's warning that increasingly detailed inferred regulatory networks can become causal "hairballs": mechanistic abstraction is useful only when it continues to explain and survive perturbation. https://www.nature.com/articles/s41576-026-00939-1

It also prevents a false "one giant matrix" conclusion. Koopman theory provides an operator view of nonlinear dynamics, but the exact Koopman operator is generally infinite-dimensional and finite-dimensional invariant representations containing the original state are strongly restricted for general nonlinear systems. The Adult should therefore be **one semantic operator with a heterogeneous factorization**, not one forced linear latent representation. https://doi.org/10.1371/journal.pone.0150171

Hybrid-system semantics reinforce the same engineering point: mixed discrete/continuous mathematical systems can be given compositional behavioural semantics without pretending every component has the same internal mathematics. https://doi.org/10.1016/j.ic.2024.105189

## 12. First executable state-equivalence receipt

`reference_mathematical_adult_state_equivalence_verify.py` turns the operator/minimal-state doctrine into a hostile Workbench test.

The current source-epistemics checkpoint stores a repetition counter on one source assertion.  A donor history with twenty repetitions and a candidate state with the same assertion but repetition count reduced to one are different stored representations.  The candidate is treated as compressible only after both branches remain identical through future source conflict, independent support, source withdrawal and another repetition.  The future projection includes source/causal state, not merely the immediate public action.

The negative control is more important: two Adults can emit the **same motor action** from two different source lineages, yet their later consequence updates calibrate different sources.  Those histories therefore cannot be merged even though current outward behaviour is equal.

Observed fast receipt:

```text
FOUNDRY_MATHEMATICAL_ADULT_STATE_EQUIVALENCE_GREEN
whole_adult_semantics=STATEFUL_OPERATOR
equivalence=FUTURE_BEHAVIOR_PLUS_UPDATE_AUTHORITY
candidate_deleted_field=source_assertion.repetitions
interventions=conflict,independent_support,withdraw_original,repeat_then_conflict
```

This does not yet authorize deleting that field globally; it proves the **form of the deletion test**.  A real representation deletion must expand the intervention battery until every downstream phenotype/authority seam owning that distinction is covered.

## 13. Language gain from recursive operator factorization

The same Workbench pass found that the current causal-program language winner already behaves like a recursively factored operator more strongly than the previous depth-2 assay exposed.  Without adding a new language mechanism, the Adult now builds eight distinct clauses from only two demonstrated construction exemplars, groups them into four learned pair programs, then two pair-of-pair programs, then one higher program whose members are the two opaque higher programs rather than the eight leaves.

Observed continuing-Adult phenotype:

```text
pair surface:             79 bytes
depth-2 surface:         159 bytes
depth-3 surface:         319 bytes
distinct held-out clauses: 8
causal-program depth:      3
current decision width:    1
incremental public bytes: 319
```

The 319-byte program remains consequence-selected, resource/urgency-sensitive, authenticated-relief-sensitive, source-withdrawal robust and socially selectable as the more explicit answer; ambiguous social history still routes to clarification.  This matters because it demonstrates the desired direction directly: **factored learned mathematics can increase language depth and surface complexity without flattening current controller width.**

The fast factory now gates developmental curve, language mastery, interference, the causal-program deletion tournament and mathematical-state equivalence together.  The broader language-mastery gate remains GREEN after this depth increase.

## 14. Local learning / write-amplification receipt

`reference_mathematical_adult_learning_locality_verify.py` measures the other half of the operator hypothesis: the mathematical individual must be locally writable during life rather than only globally compressible offline.

The Adult is first given one provisional exposure to a novel lexical surface.  That is insufficient for outward use.  The one update under measurement is the next independent confirmation of the same lexical binding.  Before and after that single event, the assay snapshots active lexical state, hierarchy state, causal-program structure and program-selection credit separately from the exact history append.

Observed receipt:

```text
FOUNDRY_MATHEMATICAL_ADULT_LEARNING_LOCALITY_GREEN
changed_active_lexeme_rows=1
archive_events_appended=1
active_lexical_json_growth_bytes=5
heldout_language_fanout=4
```

The one confirmation changes exactly one active lexeme row and appends one history event.  Hierarchy state, causal-program structure and program-selection credit remain unchanged by the lexical confirmation itself.  That local learned delta immediately enables four distinct held-out construction surfaces using the newly confirmed word.

This is not a claim that JSON byte count is the final state metric.  It establishes the quantity we should protect during representation replacement: **small logically local persistent change, small implementation write amplification, and large future phenotype leverage**.  A compact algebra that requires global rewriting after ordinary conversation loses even if its static checkpoint is smaller.

## 15. Recursive representation deletion inside the operator

`reference_mathematical_adult_operator_factorization_verify.py` began by attacking a real duplicated representation boundary: recursive language structure existed both as hierarchical composite closures for surface realization and as the causal-program membership tree for learned execution/selection. The integrated winner now goes further: recursive public composition is ephemeral, learned program membership is the sole recursive structural owner, and the generic causal-program bank owns the opaque program->factor relation used by outward rematerialization. The language Adult no longer owns a second `surface` map or a persistent hierarchy/composite-closure tree.

The representation does **not** flatten the recursive program and does not store the generated answer. It incrementally traverses the learned factorization, applies learned surface/template mathematics as required, emits one surface byte, and waits for matching reafference before advancing.

Current reference receipts:

```text
FOUNDRY_MATHEMATICAL_ADULT_OPERATOR_FACTORIZATION_GREEN
integrated branch: 16 distinct held-out clauses / 639 public bytes / depth 4
scale branch:      64 distinct held-out clauses / 2495 public bytes / depth 6
current_decision_width=1
persisted_composite_closures=0
program_tree_is_only_recursive_surface_owner=true

FOUNDRY_MATHEMATICAL_ADULT_CAUSAL_PROGRAM_FACTOR_OWNER_GREEN
adult_surface_shadow_namespace=false
factor_checkpoint_owned_by_generic_program_bank=true
factor_checkpoint_replays_exact_surface=true
focal_factor_lesion_preserves_program_and_credit_but_blocks_transduction=true
```

Only two complete clause constructions seed the productive relation in the scale fixture; the same learned relation expands to sixty-four distinct held-out clause combinations at depth six. Earned factors survive withdrawal of an original teaching source for already-earned expression, unsupported new structure refuses, and focal factor/template lesions remain causally visible. Thus the reduction is not an output cache.

The earlier `9011 -> 4219` reference serialization reduction was useful evidence during the donor/challenger stage, but the stronger architectural result is now qualitative and integrated: **the same recursive structure has one persistent owner, and even that owner remains a replaceable representation rather than constitutional ontology**. A future state-space, sparse algebraic or other factorization may replace causal programs if it preserves or improves the same phenotype, update-authority, locality and intervention controls.

The factory gate now combines this factorization with developmental language, interference, causal-program deletion, multiple state-retirement/factorization assays, state-equivalence and learning locality. The full language mastery gate remains the broader phenotype union.

## Working conclusion

Cyber Lagoon should seek **mechanistic mathematical compression**: one continuing AGI operator, factored into explicitly engineered but replaceable mathematics, acting on the smallest causally sufficient learned individual state we can experimentally earn. Current cognitive phenotypes should be unfolded only when reached, and every algebraic simplification must survive future-behaviour plus update-authority interventions.

The architecture "cheat code" is therefore not that one equation eliminates cognition. It is that **the whole organism can have one mathematical semantics while its factorization, state realization and hardware lowering remain aggressively replaceable**. That gives Foundry a principled way to simplify, compose and accelerate AGI without turning it into either a giant graph or an underspecified emergent soup.

That is the intended meaning of emergence and mathematical replaceability in Foundry.
