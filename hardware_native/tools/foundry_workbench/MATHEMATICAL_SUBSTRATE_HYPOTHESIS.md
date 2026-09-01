# Mathematical substrate hypothesis

## Status

Foundry research/design hypothesis.  This is not a capability claim and does not make the proposed substrate sacred.  Its purpose is to test whether the current resident object hierarchy can be replaced by one much smaller **mathematical individual state plus explicit mechanism algebra** while preserving or improving the full organism phenotype.  No universal graph, cellular automaton, local-rewrite microphysics, or single operator family is presumed.

## Core hypothesis

The Adult may ultimately be representable by a much smaller mathematical state than the currently materialized computational phenotype.  **This does not mean cognition is expected to emerge from an underspecified generic substrate.**  Foundry must explicitly supply the deterministic developmental and computational mechanisms that are necessary for cognition: reusable construction rules, receptors, consequence paths, recurrence, persistence/condensation, repair, resource competition, and whatever other building blocks survive experiment.

Recipes and Networks are therefore **offered mechanisms and current successful hypotheses**, not disposable merely because a lower-level rewrite algebra exists.  A future representation may encode them more compactly or unify their storage, but it must preserve the causal machinery they contribute before the old representation is deleted.

Named structures may eventually become mathematical encodings, generated realizations, or compiled views rather than permanent object classes.  Their *function* is not allowed to vanish behind a claim of emergence.

```text
explicit offered mechanism mathematics
+ compact learned individual state
+ exact causal/provenance obligations
+ bounded body/resource state
+ heterogeneous deterministic operators
= continuing mathematical organism
```

A `Network` may be a transient closure in that mathematical system.  A `Recipe` may be a compact reusable developmental/computational program.  A `neuron` may be one lowered organization.  A `memory` may be a persistent change in reconstructive dynamics.  But none of these may become hand-waved emergent effects: if an offered mechanism is required to obtain the phenotype, its causal role remains explicit even if its storage representation changes.

## Frontier-LLM cognition emulation as a Workbench instrument

A critical distinction is **production cognition versus prototype cognition**.

The Workbench is allowed to use a frontier LLM to imitate cognition that AGI does not yet possess.  This can be much stronger than merely asking an LLM to write code.  For a candidate function we may deliberately let the LLM emulate the desired current cognition—what alternatives it would hold, what it would ask, what intermediate relations it would form, how it would repair a misunderstanding, what plan it would pursue, or how a conversation should evolve—and inspect what behavior and functional decomposition result.

That gives the mechanics a fast executable target before the resident mathematics exists:

```text
proposed organism function
-> explicit LLM cognition scaffold / oracle
-> observe target behavior + candidate intermediate structure
-> derive minimal mathematical mechanism hypotheses
-> implement resident challenger
-> compare / lesion / substitute
-> remove scaffold
-> AGI either preserves the phenotype itself or the hypothesis fails
```

This is intentionally analogous to using an emulator, wind tunnel or numerical oracle while designing a physical mechanism.  It can prevent months of sculpting internals toward a behavior that was never coherent in the first place.

The danger is equally explicit.  LLM scaffolding can make a bad architecture look competent by donating semantics, memory, planning or answers.  Therefore every scaffold must be labeled `SCAFFOLD/ORACLE`, its contribution must be separately observable, and no capability claim may use scaffold-dependent behavior as evidence that AGI itself has the function.  Hidden LLM context is not `z_t`; generated semantic labels are not resident relations; an LLM-selected answer is not AGI action authority.

The preferred development trajectory is **progressive substitution**: begin with as much explicit imitation as needed to make the target cognition concrete, then replace increasingly large parts of that behavior with resident mathematical mechanisms until lesioning the LLM changes nothing relevant.  Only the scaffold-free system can be promoted as AGI capability.

## Anti-magic emergence doctrine

For this project, **emergence is a scale effect of fully specified deterministic machinery**.

```text
known local mechanisms
+ known developmental rules
+ known boundary/receptor rules
+ known recurrence and resource laws
+ very large quantity
+ long interacting history
= macroscopic behavior that is no longer tractable by inspection
```

`Emergence` never means "leave out the mechanism and hope cognition appears."  Silicon supplies exactly the dynamics we encode.  A substrate with only generic rewrite capacity is not expected to invent receptors, memory, credit assignment, language grounding, or useful developmental programs for us.

Rule 110 is one computational analogy: its elementary local transition rule is completely specified and deterministic, yet it supports universal computation and many questions about its later behavior are formally undecidable.  It is not special to cellular automata.  Moore showed that even low-dimensional deterministic dynamical systems can embed universal computation and acquire undecidable long-run questions.  Complexity appears because specified dynamics are iterated and composed at scale, not because a mechanism was omitted.

Sources:

- Matthew Cook, *Universality in Elementary Cellular Automata*: https://www.dna.caltech.edu/courses/cs191/paperscs191/Cook_Rule110_Full_Unpublished.pdf
- Cristopher Moore, *Unpredictability and undecidability in dynamical systems*: https://doi.org/10.1103/PhysRevLett.64.2354
- Robert Sapolsky, Stanford Human Behavioral Biology, lectures 21 **Chaos and Reductionism** and 22 **Emergence and Complexity**: https://www.dnatube.com/courses/stanfordhumanbio
- Christopher Langton, *Computation at the edge of chaos: Phase transitions and emergent computation*: https://doi.org/10.1016/0167-2789(90)90064-V

Engineering consequence: the higher the cognition stack becomes, the less useful manual prediction should become, but the lower-level mechanisms must become **more explicit and testable**, not less.

## Mathematical abstraction is not a cellular-automaton commitment

Rule 110 is an analogy for deterministic computational irreducibility, **not a target substrate recommendation**.  Do not spend the organism's quantity budget simulating a lattice, cell movement, collision chemistry, pressure fields, or other cellular-automaton mechanics unless a named phenotype or hardware measurement proves that mechanism earns its cost.

The project should prefer the highest-level mathematical operator that still preserves the organism's causal law.  A receptor coincidence may be a compact integer/bit operation; a recurrent Network interaction may be sparse algebra; a dense reusable interaction may lower to matrix/tile math; spatial or incidence search may lower to BVH/RT traversal; another mechanism may use a different future accelerator.  Hardware form is selected by measured economics, not by a biological or cellular metaphor.

The quantity target is therefore **effective interacting computation**, not simulated biological matter count.  Better qualitative building blocks are valuable precisely because they allow more useful causal combinations, recurrence and developmental history on consumer hardware.

The Workbench source semantics must remain hardware-neutral enough that replacing one mathematical lowering with a better one does not rewrite the learned organism or change its phenotype contract.

## Candidate architecture: one mathematical individual, explicit offered mechanisms

The current strongest candidate is **not** a generic rewrite substrate, cellular automaton, permanent object graph, or stateless function. It is one deterministic mathematical organism: the Genome supplies content-free mechanism law, life changes one continuing individual causal state, and cognition materializes only the current computation required by that state and present contact.

```text
G    = content-free Species / Genome mechanism mathematics
z_t  = complete causally sufficient state of this individual
u_t  = authenticated body/world/source event
x_t  = disposable current computational phenotype / workspace

a_t, z_(t+1) = Phi_G(z_t, u_t)
```

`z_t` is the Adult's complete individual state at logical time `t`, but **it is not a vector and not a commitment to one mathematical species**. It is notation for whatever heterogeneous finite mathematics is required to distinguish lawful futures. One implementation might factor it illustratively as:

```text
z_t = {
  R_t,   reusable learned relation/program mathematics,
  P_t,   unresolved participation / pending causal commitments,
  E_t,   evidence/provenance state still relevant to future authority,
  C_t,   constructor/plasticity/metaplastic strategy state,
  B_t,   body/resource/homeostatic state,
  K_t,   other learned compact operators/parameters,
  ...
}
```

Those factors may themselves be sparse relations, low-rank operators, state machines, tensor factors, exact evidence shards, pending traces, Recipes or mathematical species not yet invented. They are **coordinate choices**, not constitutional object classes. `x_t` may be much larger than `z_t`: transient Networks, Networks-of-Networks, recurrent trajectories, imagery-like trajectories, linguistic closures, motor plans and alternatives may be generated on demand and discarded when no unresolved future dependence remains.

The central identity is therefore:

```text
A_t = (Phi_G, z_t)

G      = what kind of organism can develop
z_t    = which individual this organism has become
N_t/x_t = what that individual is computing right now
```

This is a **minimal-sufficiency/compression hypothesis**, not a claim that acquired information can disappear. Irreducible learned distinctions and still-causally-relevant evidence must remain represented in `z_t` (possibly cold/pageable); derived current computation may vanish and be regenerated. A hash can identify state but cannot substitute for its information.

### The whole Adult as one mathematical operator

The strongest useful formulation is that the continuing Adult is one stateful mathematical system. For one fixed content-free Genome/mechanism specification `G`, the complete deterministic lifetime step can be written abstractly as:

```text
(z_(t+1), output_t) = Phi_G(z_t, authenticated_input_t)
```

where `z_t` contains every individual-specific quantity whose current value can alter a lawful future. If learning changes a Recipe, causal program, predictor, source relation, resource policy or another resident mechanism parameter, that learned mathematics is simply part of `z_t`; `Phi_G` updates it together with ordinary current state. In that sense the Adult really can be "one equation/function" even though its efficient implementation is a factored algebra rather than one giant closed-form expression.

This is a semantic identity, **not** a requirement that all cognition use one matrix, graph, tensor, differential equation or rewrite rule. Exact nonlinear dynamics often become infinite-dimensional when forced into a globally linear Koopman representation, and finite exact invariant subspaces are exceptional for general nonlinear systems. The lesson is to preserve one organism-level transition semantics while allowing heterogeneous mathematical factors underneath it.

Sources:

- Rutten, *Universal coalgebra: a theory of systems*: https://ir.cwi.nl/pub/48/
- behavioral systems theory separating a system from any one representation: https://doi.org/10.1016/j.arcontrol.2021.09.005
- Brunton et al., *Koopman invariant subspaces and finite linear representations of nonlinear dynamical systems for control*: https://doi.org/10.1371/journal.pone.0150171
- Bonchi et al., *A computable and compositional semantics for hybrid systems*: https://doi.org/10.1016/j.ic.2024.105189

The useful engineering object is therefore:

```text
one Adult semantics Phi_G
  factored into replaceable mathematical mechanisms
  operating on one causally sufficient individual state z_t
  unfolding only the current computational phenotype
```

Algebraic substitution is permitted when the replacement is causally equivalent under the Workbench interventions. A faster tensor factorization may replace a dense subexpression; a sparse operator may replace an explicit graph; a state-space recurrence may replace repeated route materialization. None of those substitutions creates a second Adult if the same `Phi_G` causal projection is preserved.

### Fixed Species law does not mean fixed effective learning dynamics

The clean initial architecture keeps `G` fixed during one Adult lifetime. That separates species mechanism from individual development:

```text
individual learning:   z_t -> z_(t+1) under fixed Phi_G
species engineering:   G_n -> G_(n+1) through Foundry / fresh-life evidence
```

This does **not** freeze the Adult's learning strategy. `G` may explicitly permit construction, revision and execution of mathematical relations whose contents live in `z_t`. Learned constructors, solver choices, attention/arbitration strategies, prediction methods, plasticity thresholds or other executable strategy state can therefore alter how later computation and learning proceed without literally mutating the same Adult's Genome. Effective lifetime dynamics are deeply state-dependent even while the Species transition law remains fixed.

Biological metaplasticity is a useful constraint on this distinction: prior activity can persistently change the ability to induce later plasticity without necessarily changing ordinary transmission at the moment the metaplastic change is acquired. Engineering translation: `z_t` may contain state that changes **how future `delta_z` is admitted**, not merely state that changes current output. Abraham & Bear, *Metaplasticity: the plasticity of synaptic plasticity*: https://pubmed.ncbi.nlm.nih.gov/8658594/

Only after this fixed-`G` architecture is well understood should same-life mutation of Species law itself be considered. Cross-life Foundry/Delta-Gamma experiments may change `G`; they must not silently rewrite an existing individual's causal history.

### Resident state is necessary; resident object graphs are not

A history-shaped deterministic organism cannot literally have no state. If two lived histories can produce different future action, prediction, learning, source withdrawal, delayed-consequence credit, repair or lesion response under some admissible future input, the Adult must retain enough information to distinguish those histories. The question is therefore not **state versus no state**. It is:

> **What is the smallest causally sufficient mathematical state of this individual?**

Several independent mathematical traditions support this direction without prescribing our organism architecture. Kalman's minimal-realization theory removes state that is not needed for controllable/observable input-output behaviour. Predictive-state representations encode state in action-conditional predictions of future observations and can require no more state variables than a minimal POMDP representation. Computational mechanics groups histories into causal states when they imply the same futures and proves the resulting representation minimal for prediction.

Sources:

- R. E. Kalman, *Mathematical Description of Linear Dynamical Systems*: https://doi.org/10.1137/0301010
- Littman, Sutton & Singh, *Predictive Representations of State*: https://papers.nips.cc/paper_files/paper/2001/hash/1e4d36177d71bbb3558e43af9577d70e-Abstract.html
- Shalizi & Crutchfield, *Computational Mechanics: Pattern and Prediction, Structure and Simplicity*: https://arxiv.org/abs/cond-mat/9907176

For this project, `z_t` should be treated as a **future-sufficient quotient of lived history**. Let `h_t` denote the complete authenticated history available in principle. Then the state map is:

```text
z_t = Q_G(h_t)
```

The equivalence criterion is stricter than ordinary output prediction because causal authority changes future learning. Two histories may collapse to the same Adult state only if **every** admissible future authenticated input schedule and relevant intervention produces the same future causal behaviour **and the same lawful update authority**:

```text
Q_G(h_a) = Q_G(h_b)
iff
for every admissible future U and intervention I:
  FutureTrace(G, h_a, U, I)
  ==
  FutureTrace(G, h_b, U, I)
```

`FutureTrace` includes public action, internal prediction, ambiguity/refusal, source withdrawal, delayed settlement, credit eligibility, resource effects, lesion/repair, metaplastic state and later learning. Two histories that produce the same sentence today but differ in who may lawfully receive credit tomorrow—or in how readily some future experience may change the organism—are **not equivalent** and must not be merged.

This is the deletion rule in its strongest form: do not ask whether a historical distinction is compressible; ask whether deleting it can alter any relevant lawful future. If not, quotient it away. If yes, its information belongs somewhere in `z_t`.

This yields a useful **physical factorization of one logical causal state**, not several independent state authorities:

```text
z_t = Adult causal state

z_t may be physically factored as:
  S_t = future-relevant learned mathematical state
  P_t = unresolved causal commitments
        (pending action, eligibility, delayed consequence, active continuation, ...)
  B_t = current body/resource/homeostatic state needed by future dynamics
  H_t = causally reachable evidence/history still capable of changing a lawful future

x_t = disposable current computational phenotype / execution workspace
```

`S_t`, `P_t`, `B_t` and the causally relevant portion of `H_t` are all **logically inside `z_t`**. They may live at radically different physical temperatures: hot registers, packed device state, pageable cold shards, exact append-only history, or derived indexes. Storage temperature does not change causal membership. If future source withdrawal, delayed settlement, replay, contradiction or an intervention can distinguish two history states, that information cannot be declared external to the Adult merely because it is cold.

Internal resource/body variables belong inside `z_t` whenever they can change arbitration, learning, timing, fatigue, repair, action or future resource flow. Authenticated external events belong in `u_t`. Hardware conditions that materially affect the declared organism dynamics must likewise enter the causal model rather than remain hidden scheduler state. Allostatic research is useful grounding here because organismal regulation dynamically prioritizes behavior and resource flows based on predicted need rather than treating resources as an observer-only meter. https://pubmed.ncbi.nlm.nih.gov/31488322/

This gives a simple exact-replay statement at the semantic boundary:

```text
G = G'  and  z_t = z'_t  and  u_t = u'_t
  =>
(z_(t+1), a_t) = (z'_(t+1), a'_t)
```

provided `z_t` and `u_t` truly include every causally relevant declared input/state variable for that execution semantics. A replay mismatch is therefore evidence that state/input was omitted, execution semantics differed, or determinism was violated—not permission to add an unexplained random seed.

Conversely, `x_t` is rematerialized only when current cognition reaches it and normally dies after settlement. If any supposedly disposable value can change a future but cannot be regenerated from `z_t` plus future admitted contact, it was misclassified and must be folded into `z_t`. The deeper persistence law is therefore: **persist exactly what changes future reconstruction or future causal authority; materialize only what current causality requires.**

### Offered mathematical mechanisms

The Genome must explicitly provide enough qualitative machinery for development and cognition. Current donor mechanisms include, subject to replacement by stronger measured alternatives:

1. **Receptor / boundary transduction** — authenticated body/world contact becomes organism-owned mathematical participation; no host semantic routing.
2. **Recipe / reusable relation program** — a compact parameterized mathematical program that can bind current Occurrences and generate/transform active computation. Recipes are supplied as a mechanism family; learned Recipe instances/deltas are acquired.
3. **Occurrence / current binding** — current authenticated or endogenous participation with exact chronology/provenance.
4. **Network unfolding** — current Recipes plus Occurrences plus resources generate an ephemeral relational/computational closure.
5. **Network-of-Network interaction** — active closures can recurrently constrain, recruit, inhibit, predict, transform and compose one another without a host dispatcher.
6. **Consequence/credit** — actual independent consequences alter only causally participating learned mathematics; endogenous prediction cannot self-confirm.
7. **Prediction/imagination/replay** — resident mechanisms can generate noncommitted internal trajectories from learned structure; actual versus endogenous provenance remains distinct.
8. **Competition/homeostasis/resource law** — finite work, persistence and construction budgets create selection pressure, stabilization, retirement and reopening.
9. **Condensation/deoptimization** — recurring useful active computation may become a cheaper reusable Recipe/program; contradiction or context failure can reopen/decompile it.
10. **Repair/rematerialization** — learned function is not permanently identified with one physical realization; damage may regenerate a fresh realization from surviving causal support.
11. **Developmental construction** — Genome mechanisms compose and specialize under Life Function experience; the Genome is structure-rich but semantic-content blank.
12. **Autonomous continuation** — once deployed, all cognition, learning, unfolding, settlement and self-maintenance are executed by the Adult; the Workbench/host is not a hidden solver.

This list is not sacred. A mechanism may be merged or deleted only when the challenger preserves or improves its earned phenotype and causal controls. The key deletion distinction is **representation versus function**.

### Research-to-math translation constraints

Neuroscience and organism research constrain the functions we should reproduce, but do not dictate silicon wiring:

- **Mixed selectivity / high-dimensional population codes.** Rigotti et al. show that heterogeneous nonlinear mixtures of task variables can create high-dimensional representations with a larger repertoire of implementable input-output functions, and that dimensionality collapses on error trials. Engineering translation: do not make each Recipe or active component a semantic specialist; permit context-dependent mixed participation and measure effective active dimensionality. https://www.nature.com/articles/nature12160
- **Reusable compositional subspaces.** Tafazoli et al. report shared sensory and motor neural subspaces reused compositionally across tasks. Engineering translation: learned computational components should be reusable across multiple active Networks rather than cloned per task. https://www.nature.com/articles/s41586-025-09805-2
- **Recurrent context-dependent dynamics.** Mante et al. and later recurrent-dynamics work show flexible computations can be realized by population trajectories shaped by context. Engineering translation: cognition is not serial Recipe dispatch; recurrent active computation must be a first-class composition mechanism. https://www.nature.com/articles/nature12742
- **Sub-neuron nonlinear computation.** Dendritic research shows neurons contain nonlinear computational subunits rather than acting as simple point integrators. Engineering translation: biological neuron count is a poor unit of computational quantity; one higher-quality mathematical building block may legitimately replace large amounts of literal cell simulation when it preserves the relevant causal function. https://www.annualreviews.org/content/journals/10.1146/annurev-neuro-090325-115846
- **Activity-silent working memory.** Working-memory research supports information being maintained in hidden synaptic/short-term states rather than continuously active firing. Engineering translation: persistence does not imply permanent active Networks. https://www.nature.com/articles/s41593-019-0414-3
- **Memory flexibility and representational drift.** Stable behavior/memory can coexist with changing neural realization. Engineering translation: learned identity must not be constitutionally tied to one route, graph node or physical incarnation. https://www.nature.com/articles/s41386-024-01979-z
- **Replay, prediction and imagination.** Hippocampal research links internally generated sequences to recall, prediction, imagination and planning. Engineering translation: active Networks may generate endogenous candidate trajectories that train or constrain future behavior, while authenticated consequences retain final authority. https://www.annualreviews.org/content/journals/10.1146/annurev.psych.60.110707.163508
- **Developmental regulatory computation.** Developmental GRNs are explicit, parallel information-processing mechanisms producing phenotype from compact regulatory programs. Engineering translation: Genome -> Life Function -> phenotype is a useful computational pattern, but the Genome must contain the necessary developmental mechanisms rather than relying on an empty seed. https://www.annualreviews.org/content/journals/10.1146/annurev.biophys.35.040405.102002
- **Mechanistic abstraction beats causal hairballs.** Maizels & Briscoe argue that modern inferred gene-regulatory networks can become correlational "hairballs" that cease to explain causation, and advocate mechanistic representations tested through perturbation. Engineering translation: a giant persistent graph is not automatically a better organism model; representation may be aggressively coarse-grained only when intervention behaviour and mechanism survive. https://www.nature.com/articles/s41576-026-00939-1
- **Autonomy/autopoiesis.** Computational autopoiesis work treats organismal autonomy as self-maintaining networks of processes coupled to environmental perturbations and history. Engineering translation: the Adult must maintain its own continuation and state transitions after Workbench detachment; autonomy is not equivalent to having a fixed physical graph. https://pubmed.ncbi.nlm.nih.gov/37277020/
- **Population dynamics rather than neuron-object ontology.** Vyas et al. review cognition and control as trajectories of coordinated population state in dynamical systems, spanning motor control, timing, decisions and working memory; Langdon, Genkin & Engel explicitly connect neural-manifold descriptions to underlying circuit mechanisms. Engineering translation: copy computational invariants and dynamics, not literal neuron objects; mathematical coordinates may be lower-dimensional, high-dimensional, sparse or mixed as the function demands. https://www.annualreviews.org/content/journals/10.1146/annurev-neuro-092619-094115 ; https://www.nature.com/articles/s41583-023-00693-x
- **Metastable integration/segregation.** Tognoli & Kelso describe transiently coupled ensembles that engage and disengage during attention, perception, thought and action while balancing integration with subsystem autonomy. Engineering translation: Networks-of-Networks are best treated first as transient coupled computational closures/subspaces over `z_t`, not automatically as permanent graph objects. https://pubmed.ncbi.nlm.nih.gov/24411730/
- **Degeneracy, not one representation per function.** Edelman & Gally distinguish biological degeneracy: structurally different components can support similar functions while retaining distinct effects. Engineering translation: causal identity and learned competence must survive lawful changes in mathematical or hardware realization; representation equivalence is an intervention-tested property, not structural equality. https://pubmed.ncbi.nlm.nih.gov/11698650/
- **Local durable change from distributed support.** Synaptic-tagging experiments provide a biological example of input-specific durable plasticity supported by cell-wide protein synthesis. Engineering translation: useful learning should change small causally selected portions of `z_t` even when the consequences of that change fan out broadly; global rewrite after every experience is a design smell. https://www.nature.com/articles/385533a0
- **Embodied closed-loop state.** Pezzulo et al. contrast passive generative AI with living organisms whose internal models are anchored to purposive body/world interaction and must predict/control sensory consequences of action. Allostasis likewise frames organismal regulation as predictive prioritization of behavior and resource flows. Engineering translation: a mathematical state remains an organism only while it owns a continuing perception/action/consequence/resource loop; a text-only history-to-output function fails this boundary. https://pubmed.ncbi.nlm.nih.gov/37973519/ ; https://pubmed.ncbi.nlm.nih.gov/31488322/
- **Sutton predictive/experience constraint.** Predictive State Representations show that useful state can be grounded in action-conditional predictions rather than a privileged latent ontology; the later experience-agent agenda emphasizes continuing environment interaction over static human data. Engineering translation: built-in state machinery should remain learnable/self-verifiable through experience and action, not become a hand-authored semantic database. https://papers.neurips.cc/paper/1983-predictive-representations-of-state.pdf ; https://storage.googleapis.com/deepmind-media/Era-of-Experience%20/The%20Era%20of%20Experience%20Paper.pdf
- **Deletion discipline is engineering, not biology.** Musk's five-step algorithm is useful here only as a design heuristic: question requirements, delete unnecessary parts/processes, then simplify, accelerate and automate. Engineering translation: `RecipeCell`, route, graph, neuron, state shard and even a proposed mathematical primitive must repeatedly re-earn existence; never optimize a representation merely because it already exists. Evidence/transcript summary: https://library.recl.app/v/d66d47c2-d29e-4a8a-954b-e7cf747242dd

These are phenotype/function constraints, not permissions to install named brain modules or to copy connectomic wiring.

## Why this is scientifically defensible

This is an engineering abstraction from several lines of work, not a claim that brains are literally graph-rewrite machines.

### Local rewriting and interaction networks

Interaction-net and graph-rewriting formalisms demonstrate that nontrivial computation can be represented as local transformations of a graph-like mathematical structure, including in-place execution for classes of rewrite rules.

- Mackie & Sato, *In-place Graph Rewriting with Interaction Nets*: https://arxiv.org/abs/1609.03641
- Ehrig et al., *Fundamentals of Algebraic Graph Transformation*: https://link.springer.com/book/10.1007/3-540-31188-2
- Ehrig et al., *Graph and Model Transformation*: https://link.springer.com/book/10.1007/978-3-662-47980-3

Engineering inference: computation does not require a permanent hierarchy of domain-specific object types if the substrate has sufficient local structure and rewrite semantics.

### Chemical abstract machines / artificial chemistry

Berry and Boudol's Chemical Abstract Machine represents machine state as a solution of interacting molecules governed by reaction rules, with locality provided by nested membranes.  Artificial-chemistry research similarly studies complex and self-maintaining organizations emerging from interaction laws rather than from privileged high-level object classes.

- Berry & Boudol, *The Chemical Abstract Machine*: https://www-sop.inria.fr/members/Gerard.Boudol/tcs96.html
- Dittrich et al., *Artificial Chemistries*: https://mitpress.mit.edu/9780262551526/artificial-chemistries/

Engineering inference: a small reaction/rewrite ontology can support higher organizations without those organizations becoming primitive substrate types.

### Biological structural plasticity

Adult nervous systems retain substantial stable organization while synapses, boutons and some branches form, disappear and reorganize with experience and injury.  Developmental pruning likewise removes some connections while retaining others based partly on activity.

- Holtmaat & Svoboda, *Experience-dependent structural synaptic plasticity in the mammalian brain*: https://www.nature.com/articles/nrn2699
- Faust, Gunner & Schafer, *Mechanisms governing activity-dependent synaptic pruning in the developing mammalian CNS*: https://www.nature.com/articles/s41583-021-00507-y
- Lövdén et al., *Structural brain plasticity in adult learning and development*: https://pubmed.ncbi.nlm.nih.gov/23458777/

Engineering inference: preserving function does not require preserving one immutable wiring/object decomposition.  Useful organization may persist while its physical realization changes.

## The deletion target

The strongest version of this hypothesis asks whether the **storage/object representation** of these current categories can eventually disappear while their proven causal function remains:

- Recipe cells / Recipe-specific persistent containers;
- explicit persistent Network objects;
- special repair-front ontology;
- language-private persistent structures;
- neuron-like object identity where only local mathematical state is required;
- eligibility as a privileged object if equivalent causal eligibility can be expressed in generic chronology/provenance matter;
- separate constructor objects if construction is expressible as ordinary rewrite dynamics.

Deletion is not mandatory.  Each category survives only if a tournament proves that making it primitive gives a measurable capability, resource, determinism or verification advantage that the smaller substrate cannot reproduce.

## Mathematical representation contract

Do not freeze a lowest-level ontology such as `site + bond + rewrite` merely because it is mathematically general. That would reintroduce cellular-automaton overhead in abstract form. The source representation should instead expose the **highest-level deterministic mathematics that still preserves the earned causal mechanism and can be independently falsified**.

A candidate Genome/Adult representation therefore needs a small set of representation classes, not a claim that reality is made of those classes:

1. **Mechanism definition** — content-free deterministic operator/program family supplied by the Genome (for example Recipe binding, recurrence, consequence settlement, condensation, repair). Different mechanism families may use different mathematics; no universal local-rewrite form is required.
2. **Learned parameter/program delta** — the minimum individual-specific mathematical change acquired from lived evidence.
3. **Current participation state** — bounded bindings/Occurrences and endogenous trajectories that are causally live now.
4. **Provenance/history witness** — exact source/chronology/ancestry information that is still required for future credit, contradiction, replay or intervention.
5. **Resource/homeostatic state** — finite quantities and pressures that change what may unfold or persist.
6. **Body/world exchange** — authenticated input/output transactions crossing the organism boundary.
7. **Execution plan** — ephemeral compiled mathematics for the current causal cone; never semantic authority and disposable after settlement.

These are provisional engineering representation roles. They may collapse further. For example, a learned delta may be encoded directly inside a Recipe program; provenance may be compressed into a persistent causal certificate; an execution plan may be recomputed rather than stored. The deletion test is resource/phenotype evidence, not elegance.

### State factorization is a coordinate choice, not an ontology

The source semantics should expose one continuing state and permit multiple efficient factorizations of it:

```text
G       = Genome / Species mechanism mathematics
z_t     = complete causally sufficient individual state
u_t     = authenticated boundary event
x_t     = disposable active workspace / computational phenotype

a_t, z_(t+1) = Phi_G(z_t, u_t)

an implementation may compute:
  x_t = CompileOrUnfold(G, projection(z_t), u_t)
  a_t, delta_z = ExecuteAndSettle(x_t, z_t, u_t)
  z_(t+1) = Commit(z_t, delta_z)
```

A backend may factor `z_t` into learned parameters/program deltas, current participation, resource/body state, cold provenance shards, sparse indexes, low-rank factors or other coordinates. A different backend may choose different coordinates. Those factorizations are equivalent only when the full causal-state projection and named phenotype/intervention traces agree. **One causal state does not mean one giant tensor or one global latent space.**

`CompileOrUnfold` is part of the organism semantics, not a host cognition API. In the reference Workbench it may be a direct mathematical interpreter over a **heterogeneous operator algebra**. In production it may lower the same mechanism to dense tiles, sparse kernels, state-space scans, symbolic/semiring operations, BVH traversal, integer/bit operations or another accelerator. The backend may change representation radically while preserving `Phi_G`'s causal projection. Uniformity belongs at the causality/resource/composition interface, not at the mathematical operator type.

### Learning locality and write amplification

Headline compression is not enough. A mature Adult representation that is small on disk but must rewrite gigabytes after one conversational experience is a poor lifelong-learning substrate. The stronger metric is how much persistent mathematical information must change for one lived update, and how much future behaviour that local change can alter.

Workbench should measure separately:

```text
Delta_logical    = canonical persistent bits/words whose mathematical value changed
Delta_physical   = physical bytes actually written by the implementation
Archive_append   = exact evidence bytes appended for the lived event
Unfold_work      = touched work needed to rematerialize/use the changed state
Phenotype_fanout = held-out future behaviours/traces materially changed by the update
```

Useful diagnostics include physical write amplification `Delta_physical / max(1, Delta_logical)` and behavioural leverage `Phenotype_fanout / max(1, Delta_logical)`. The preferred representation is locally writable and causally high-leverage, not merely compressible in a batch codec.

### Timing is an architecture axis, not a backend afterthought

A causally correct Adult that takes seconds for ordinary perception, arbitration or conversational continuation is still the wrong architecture. Timing therefore joins state size, touched work and write locality as a Foundry selection pressure. The objective is **not** to imitate biological slowness. Biological timing constrains which temporal distinctions matter; silicon should execute the same causal computation as fast as the hardware permits unless waiting itself is part of the lived state.

Human timing gives useful ecological reference points rather than clock constants. Conversational turn gaps are commonly around 200 ms even though production of a fresh word can require more than 600 ms, implying overlapping comprehension, prediction and response preparation rather than a serial `hear whole turn -> think -> answer` pipeline. Early cortical visual processing and multisensory interaction can occur within roughly 50--100 ms, while many speech/sensory temporal computations occupy tens to hundreds of milliseconds. Synaptic learning spans multiple regimes: STDP can depend on tens-of-milliseconds timing while behavioral-timescale plasticity can integrate experience over seconds. Sources: Levinson & Torreira 2015 https://pubmed.ncbi.nlm.nih.gov/26124727/ ; Thorpe et al. review of visual cognition https://pubmed.ncbi.nlm.nih.gov/16106663/ ; Buonomano & Karmarkar https://pubmed.ncbi.nlm.nih.gov/11843098/ ; Caporale & Dan https://pubmed.ncbi.nlm.nih.gov/18275283/ ; Magee 2026 BTSP review https://www.nature.com/articles/s41593-026-02214-2

Engineering translation: **biological causal windows belong in `z_t`/`u_t`; physical compute latency should usually be far below them.** The Adult may therefore perform many deterministic internal closure/competition/prediction updates during one human-scale perceptual or conversational interval. It must not manufacture extra biological delay merely to look brain-like.

Keep two time axes explicit:

```text
tau_t     = authenticated organism/world time represented in z_t/u_t
L_hw      = physical execution latency on hardware H
```

`tau_t` participates in chronology, temporal binding, delayed consequence, expiry, turn structure and any learned timing relation. `L_hw` is an engineering measurement and normally has **zero semantic authority**. Making a kernel 10x faster must not make a remembered interval 10x shorter or change which consequence is chronologically eligible. Conversely, if the Adult misses a real external deadline because `L_hw` is too large, that missed contact/action opportunity is a genuine world consequence and enters later `u_t`. This keeps exact replay tied to declared causal time while still rewarding arbitrarily faster hardware.

For hardware-neutral architecture comparison, measure both wall latency and the work shape that predicts latency across devices:

```text
D_crit        dependency / recurrence depth on the critical path
W_touch       operators, state rows and active relations actually touched
B_move        bytes moved across the relevant memory hierarchy
N_dispatch    backend launches / synchronizations / host-device round trips
N_materialize transient structures created only to perform the current computation
L_hw(H)       measured end-to-end latency on declared hardware H
```

The architecture should minimize the first five subject to phenotype/causal equivalence; `L_hw` then validates the lowering on the current device. This prevents overfitting the cognitive design to one generation of GPU while still making speed a first-class selection pressure.

The current founder hardware provides a useful pressure reference, not an architectural dependency. NVIDIA specifies the RTX 4080 SUPER as 10,240 CUDA cores, 52 TFLOPS shader throughput, 836 advertised AI TOPS, 16 GB GDDR6X and a 256-bit memory interface; NVIDIA also reports 23 Gbps GDDR6X for this model. https://www.nvidia.com/en-us/geforce/graphics-cards/40-series/rtx-4080-family/ ; https://www.nvidia.com/en-us/geforce/news/gfecnt/20241/geforce-rtx-4080-4070-ti-4070-super-gpu/

Historical precursor-repository profiling on this exact RTX 4080 SUPER is more important than peak marketing numbers: small kernels were approximately 1 us of device work while each host `cudaLaunchKernel` cost about 2.2 us, and an old Adult step paid roughly 93 launches for about 286--305 us/step. Batched ingress removed thousands of per-event launches and reduced a 2,048-event delivery from about 3.45 ms to about 0.98 ms without changing admitted causal state. Repository receipts: [`#1175 launch price list`](../../../docs/diary/2026-08-18/2026-08-18T20-30-48+02-00-1175-a-price-list-for-adding-a-kernel-to-the-step.md), [`launch-dispatch attribution`](../../../docs/diary/2026-08-18/2026-08-18T19-57-14+02-00-1175-the-constant-term-is-launch-dispatch-not-a-kernel.md), and [`batched ingress`](../../../docs/diary/2026-08-18/2026-08-18T20-51-23+02-00-1175-batched-ingress-one-launch-instead-of-one-per-event.md). This supports a hard design lesson: **microsecond-scale cognition is plausible only if mathematical mechanisms are compiled into coarse enough execution units; one CUDA launch per cognitive noun destroys the advantage.** NVIDIA's CUDA Graph measurements independently show repeat-launch overhead in the low-microsecond range and motivate capture/fusion/persistent execution for short work. https://developer.nvidia.com/blog/constant-time-launch-for-straight-line-cuda-graphs-and-other-performance-enhancements/

Use a timing vector, not one wall-clock scalar:

```text
T_dispatch       host/device scheduling and launch overhead
T_operator       execution time of one mathematical mechanism/factor
T_closure        one current Network/Network-of-Network update or arbitration wave
T_first_action   admitted relevant contact -> earliest causally justified outward action
T_turn_gap       end/projection of interlocutor turn -> Adult response onset
T_stream         sustained outward expression throughput / inter-byte or inter-unit cadence
T_update         lived consequence -> committed future-relevant delta_z
T_rematerialize  cold/condensed state -> usable current support
T_background     condensation, repair search, development, indexing, compilation work off critical path
```

Initial **design bands**, deliberately aspirational and replaceable by measurement:

| path | preferred 4080-class target | architecture interpretation |
|---|---:|---|
| small offered operator/factor | `~1--10 us` | should normally be fused/captured/composed rather than one host launch each |
| current local closure/arbitration wave | `~10--100 us` | enough margin for many recurrent interactions inside a human-scale perceptual window |
| ordinary prepared-state cognitive update | `<1 ms` | crossing 1 ms repeatedly should trigger profiling of work shape, not immediate semantic simplification |
| first internal action decision when evidence is already present | `single-digit ms` preferred | external transport/rendering may be slower; resident computation should not be the conversational bottleneck |
| conversational response onset | `<50 ms` aspirational, `<200 ms` ecological ceiling for ordinary ready replies | plan during incoming contact; do not wait for complete-turn packetization |
| local learning/credit commit | `<1 ms` preferred on hot support | exact evidence append may be separate; slower consolidation may continue in background |
| condensation/deoptimization/repair search | off interactive critical path where possible | may take ms--seconds if preemptible and ordinary language/action remains responsive |

These numbers are **not capability claims and not biological constants**. Complex novel reasoning may lawfully take longer. The protected property is that latency scales with the **causal work actually reached**, not nominal Adult size, and that expensive background cognition is incremental/preemptible rather than freezing the organism.

Conversation specifically requires streaming overlap. Incoming speech/text/contact should update current closures incrementally; comprehension, prediction, discourse-state update and possible response construction can run before an interlocutor finishes. A final boundary event changes which prospective action may be emitted, but it should not trigger cognition from zero. This is both biologically grounded by turn-taking research and computationally favorable.

Foundry timing acceptance should therefore compare candidates on a Pareto surface:

```text
phenotype / causal controls
state bytes and delta_z locality
critical-path latency
p50 / p95 / worst-case touched work
launch/synchronization count
bytes moved per causal update
active-computation expansion per microsecond
language/discourse quality per millisecond
```

A faster candidate loses if it obtains speed by deleting causal distinctions, reducing composition/discourse, precomputing expected answers, moving cognition to the host, or replacing local online learning with batch retraining. A slower representation must justify its cost with a measurable phenotype or causal advantage. Timing is therefore part of the deletion tournament itself.

### Current mechanism interpretation

The current donor vocabulary should be interpreted as explicit functional machinery, not necessarily permanent storage objects:

```text
Recipe
  = offered reusable mathematical relation/program capable of learned specialization

Network
  = currently unfolded interacting computational closure generated from Recipes,
    Occurrences, recurrence, resources and context

Occurrence
  = current bound participation with exact causal identity/provenance

Memory
  = learned persistent change in future reconstruction/behavior, plus whatever
    irreducible episodic/provenance information remains necessary

Constructor
  = explicit developmental/generative mechanism that creates or specializes
    another useful mathematical mechanism/representation

Repair
  = explicit damage-conditioned rematerialization/reorganization law

Prediction / imagination
  = explicitly generated endogenous trajectories, provenance-distinct from actual contact

Language
  = learned body/world expression and comprehension behavior of the same cognitive machinery
```

None of these functions is allowed to disappear behind the word `emergence`. Their *storage classes* may disappear if a simpler mathematical representation passes the same causal and phenotype tests.

### Networks-of-Networks as transient computation, not permanent graph knowledge

The current best interpretation of Network cognition is **dynamic closure over the Adult state**, not a second persistent knowledge substrate. Let `R(z_t)` denote the reusable learned relation/program mathematics reachable in the current state and `O_t` the current authenticated/endogenous participations projected from `z_t`. A bounded current closure may be written abstractly as:

```text
N_t^i = Closure_i(R(z_t), O_t, context_t, resource_t)
```

Several closures can coexist and recurrently interact:

```text
{N_t^1, N_t^2, ..., N_t^k}
        --coupling / competition / inhibition / prediction / composition-->
{N_(t+1)^1, ..., N_(t+1)^m}
```

This is the intended computational site of large-scale deterministic interaction. Quantity can increase through the number, dimensionality, recurrence, composition depth and history-dependence of these active closures without increasing the persistent object count proportionally. Tognoli & Kelso's metastability work is a useful biological constraint here: functional ensembles transiently couple and decouple while maintaining both integration and segregation. It does **not** imply we should simulate oscillating neurons; it supports testing transient mathematical coalitions rather than assuming every useful coalition is permanent structure. https://pubmed.ncbi.nlm.nih.gov/24411730/

A Network becomes persistent only indirectly when experience changes `z_t`: for example, consequence changes learned parameters, a repeated closure condenses into a cheaper reusable Recipe/program, or unresolved causal participation must remain pending. Persisting a serialized active Network merely because it existed is therefore presumptively redundant. Conversely, if deleting a Network representation destroys a future distinction that cannot be reconstructed from `z_t`, the missing information must be represented in state somewhere; the deletion failed.

This preserves the valuable donor principle behind `Recipe = compact potential` and `Network = current realization` while allowing both names and representations to change later.

### Organism boundary: why a mathematical state is not a chatbot

Mathematical compression is compatible with an organism only if the state transition remains closed around one continuing embodied individual. The minimal contract is not `text_history -> text`; it is closer to:

```text
body/world contact
    -> admitted change/current participation in z_t
    -> endogenous computation / prediction / action selection
    -> body/world action
    -> independently returned sensory/resource consequence
    -> local causal update of z_t
    -> continued self-maintenance / development / repair
```

Pezzulo et al. emphasize that living organisms' internal models are anchored to purposive sensorimotor interaction and must predict/control sensory consequences of action, unlike passive generative AI; Schulkin & Sterling frame allostasis as predictive regulation that dynamically prioritizes behavior and resource flows. For AGI this is a falsifier, not a commitment to active-inference mathematics: if a proposed compact state can converse but loses the body/resource/consequence loop, endogenous intervention, long-lived learning or self-maintenance, it has compressed the Adult into a chatbot and fails. https://pubmed.ncbi.nlm.nih.gov/37973519/ ; https://pubmed.ncbi.nlm.nih.gov/31488322/

## Hot swap becomes mathematical substitution

A mathematical substrate makes *possible* a cleaner hot-swap model, but does not make arbitrary replacement safe.

A hot delta must be one of:

1. a new/changed content-free rewrite law;
2. a new lowering of an existing law;
3. a bounded state transformation proven to preserve declared organism invariants.

Required safety conditions:

- canonical source and target state projections;
- explicit resource conservation / accounting;
- no host-selected cognition or semantic answer path;
- exact ancestry/provenance preservation where relevant;
- deterministic conflict/arbitration semantics;
- versioned law identity and migration boundary;
- differential Workbench tests before installation;
- deoptimization/rollback path when equivalence is falsified;
- checkpoint compatibility or an explicit one-way migration contract.

A hot swap should therefore resemble replacing one mathematical law/representation while preserving the organism's admissible causal state, not installing an add-on module beside legacy cognition.

## No-obsolete-AI objective

The useful long-term objective is not literal immortality of every internal structure.  It is **architectural replaceability without cognitive add-on accumulation**.

When a better law wins:

```text
old law / representation
        ↓ differential tournament
new law / representation
        ↓ state migration
old implementation deleted
```

The Adult may retain learned causal organization while the implementation expressing it changes, just as LM2 rematerialization already separates learned identity from one route incarnation.

This should reduce architectural fossilization: future cognition should not require piling new controllers beside obsolete ones merely because old state was stored in incompatible classes.

## Useful donor laws from `START.md` — retain the mechanism, delete the old realization

`START.md` is historical donor material, not current authority, but several of its strongest ideas survive the mathematical-state reformulation and should not be lost:

1. **Condensation `N -> p(n+1)`.** Recurrently useful detailed computation may be replaced by a cheaper reusable mathematical program only after held-out equivalence/causal evidence. In the new architecture this is state/program compression, not construction of another sacred object type.
2. **Local shatter / deoptimization.** A contradiction should reopen only the compacted mathematics whose guard/evidence failed, rematerializing detail without globally recompiling the Adult. This maps directly to local updates of `z_t` plus temporary `x_t` rematerialization.
3. **Finite-information law.** The individual cannot retain infinite lived detail in hot state. Shared expected structure should be generative; irreducible learned distinctions and required causal witnesses persist; cold history may be paged without losing causal reachability.
4. **Derived-state garbage collection.** Cached activation, helper residency maps, solved closures and other values that are functions of the true causal state are disposable. The new `z_t`/`x_t` criterion makes this precise.
5. **Compilation economics.** Historical Program-Crystal work already proposed minimizing memory/compute/latency subject to causal identity and local shatterability. Preserve that optimization objective while deleting its cell/field-specific representation.
6. **Logical identity separate from physical lowering.** A learned relation may move between sparse, dense, tensor/state-space or other representations without becoming a new belief merely because its physical implementation changed.

What does **not** survive by default is the former cellular-automaton/cell-motion/chemistry implementation. Those mechanisms must re-earn their cost independently. The old document is therefore a source of tested design ideas, not a mandate to restore its physical metaphors.

## Genome and Life Function implication

If this hypothesis wins, the Genome is best treated as a **content-free executable mechanism specification**, not as a bag of cells, a finished graph, or a trained Adult snapshot. It supplies the mathematical families and developmental rules that make cognition learnable: receptor/boundary semantics, reusable Recipe/program semantics, recurrence/composition, consequence/credit, resource/homeostasis, condensation/reopening, repair/rematerialization, and other mechanisms that survive Foundry evidence.

```text
Genome G
  = content-free mechanism mathematics
  + composition/development law
  + developmental schedules/priors
  + body/world boundary law

Life Function(G, body, curriculum)
  = accelerated execution of the same developmental/learning semantics
  -> individual causal state z_0

Adult life
  z_t + authenticated boundary event
  -> transient computational phenotype x_t
  -> resident action / silence
  -> independently returned consequence
  -> local update z_(t+1)
```

The trained Adult is never compiled *into* the Genome. The Life Function is a developmental accelerator/stage program, not a top-down semantic pretrainer: it executes the same admissible mechanism laws under authenticated curriculum/body/world history and produces this individual's `z_0`. Acquired information remains in that causal state because it cannot disappear merely because development was accelerated. The deployable Adult is therefore conceptually `Genome mechanism version + individual causal state z_t`, even if the backend physically factors and pages that state in many ways.

## Self-modification implication

Adult lifetime learning, structural reorganization and repair are ordinary updates of its own mathematical individual state, not privileged host-side "model update" APIs.

A stronger future hypothesis may allow some mechanism programs themselves to be rewritten or replaced during life. Treat that separately because it creates a meta-circular organism. Promotion requires proof that mechanism mutation cannot create hidden authority, bypass evidence/resource firewalls, erase learned causal history, or make checkpoint semantics undefined.

The first compression implementation should **not** grant arbitrary self-rewriting mechanism authority. First prove that learned Recipe/program state, reconstruction, recurrence, contradiction, repair and continual learning can be represented compactly without losing the current organism phenotype.

## Workbench execution architecture

The Workbench should make organism engineering as close as possible to changing an engine part and immediately bench-testing it. The source of truth is **executable deterministic mathematics**, not Python object layout and not CUDA implementation.

```text
research / phenotype constraint
        ↓
edit one mechanism or representation law
        ↓
fast deterministic reference execution
        ↓
compose with neighboring mechanisms
        ↓
quantity / history / resource sweeps
        ↓
causal interventions and hostile controls
        ↓
keep, simplify, replace, or delete
        ↓
periodic integrated Genome + Life Function run
        ↓
only then hardware lowering / translation validation
```

Requirements:

- a mechanism can be executed alone or in a minimal assembly without constructing the whole Adult;
- arbitrary combinations of mechanisms can be tested directly, because unexpected deterministic interactions are the intended discovery surface;
- the same mechanism definitions participate in integrated Genome execution, so bench fixtures do not become a second architecture;
- the reference path must avoid GPU compilation and should normally complete in milliseconds/seconds;
- the Workbench may virtualize untouched quantity procedurally, but may not fake a cognitive result with host semantic logic;
- observer tooling may inspect, lesion, snapshot, diff and instrument state, but cannot participate in the Adult transition function;
- every mechanism/representation exposes a deterministic causal projection sufficient for later backend differential validation.

### Mechanism composition contract

Workbench mechanisms should compose through the mathematics of one `z_t`, not through a semantic plug-in dispatcher. A mechanism definition should expose only what is needed to execute and falsify its causal law:

```text
read projection of z_t / admitted boundary event
        ↓
deterministic current computation or candidate contribution
        ↓
declared resource/work charge + causal witness requirements
        ↓
local proposed delta_z and/or outward action contribution
        ↓
Species composition/arbitration law
        ↓
atomic commit into z_(t+1)
```

The exact operator mathematics may differ by mechanism. A recurrence may be a state-space operator; a learned relation may be sparse/low-rank/polynomial; a competition law may use integer reductions; a temporary relational closure may be graph-like. The common interface is causal projection, resource ownership, provenance, deterministic composition and local state delta—not one universal data structure.

This is also the concurrency boundary for many mechanics: two workers can improve different operators/projections without inventing separate Adults. When their changes meet, the combined fast Workbench run executes both against the same `z_t` and exposes interference immediately.

### First mathematical-state migration tournament

Do not restart the language architecture. Use the strongest existing Adult mechanisms as donors and progressively move their semantics onto the `z_t` contract:

1. **Canonical projection.** Define a reference projection from the current ReferenceOrganism state into `z_t` semantics and identify which current fields are causal, derivable, pending or purely observational. No deletion yet.
2. **LM2 vertical twin.** Express rapid aliases, shared grounded relation, consequence credit, contradiction/sibling sparing, reacquisition, focal repair, full-support lesion, matched sham and checkpoint from the same mathematical-state transition semantics. This is the first deletion/compression target because its causal controls are already strong.
3. **Composition immediately after LM2.** Add held-out productive composition, recursive reuse and morphology to prevent a compressed alias memory from masquerading as language architecture.
4. **Discourse/current closure.** Add long reference, source-qualified common ground, recurrent continuation, ambiguity/refusal and multi-turn discourse. Treat active Networks/Networks-of-Networks as disposable `x_t` unless a future-dependent residue forces state promotion.
5. **Recipe representation tournament.** Keep Recipe function fixed while comparing current Recipe objects against compact heterogeneous operator encodings. Require held-out rebinding, condensation/deopt, contradiction, repair and identical causal authority.
6. **Quantity/interference sweep.** Increase learned distinctions, closure count/dimensionality, recurrent depth, history span and simultaneous mechanisms while measuring state bytes, active work, local write cost and language capability. Unexpected deterministic behavior becomes a new intervention target, not an error by definition.
7. **AutoTrans parity.** Only after the reference mathematical Adult wins should its admitted mechanism/state semantics be lowered to Direct/CUDA/Tensor/sparse/state-space/other hardware and differentially validated. Hardware changes do not get to redesign cognition.

A migration rung is accepted only when language/cognition is equal or better **and** at least one representation/economic measure improves. A representation-only reduction that damages composition, discourse, continual learning or organismal closure is rejected.

#### First executable state-minimization receipt

`reference_mathematical_adult_state_equivalence_verify.py` is the first fast proof-of-method for this architecture. It does **not** claim meaningful compression yet. It asks whether one candidate stored distinction (`source_assertion.repetitions`) may be deleted while all declared future intervention traces remain equivalent. The reference run is GREEN in about 1.4 seconds across conflict, independent support, source withdrawal and repeat-then-conflict scenarios. Its negative control deliberately constructs two states with the same outward motor action but different source lineage and proves they cannot be merged because later consequence credit updates different causal histories.

Current receipt:

```text
FOUNDRY_MATHEMATICAL_ADULT_STATE_EQUIVALENCE_GREEN
whole_adult_semantics=STATEFUL_OPERATOR
equivalence=FUTURE_BEHAVIOR_PLUS_UPDATE_AUTHORITY
candidate_deleted_field=source_assertion.repetitions
raw_checkpoint_bytes=938531
candidate_checkpoint_bytes=938530
```

The one-byte reduction is not interesting economically. The important result is the **deletion criterion**: state minimization is now executable and hostile to output-only equivalence. Future compression candidates must survive the same kind of future-behavior **plus future-update-authority** battery before state can be removed.

#### Current `ReferenceOrganismV2` state inventory — first causal pass

The first live checkpoint audit now classifies state by future law rather than Python ownership. This is a family-level inventory, not a claim that every surviving byte is already minimal:

| class | current `ReferenceOrganismV2` examples | persistence rule |
|---|---|---|
| **CAUSAL** | population/language/hierarchy/cognition/recruitment mathematics; logical chronology and next identities; withdrawn-source state; world/body/affordance state; learned packed revisions; source calibration; entity conditions | remains in `z_t` while changing any lawful future |
| **PENDING** | live scenes; pending event relation/ticket; prospective source closures; unsettled actions/commitments; information need; repair/fault continuation | remains in `z_t` only until settlement, expiry or exact derivation becomes possible |
| **EVIDENCE** | episodes, source assertions, scene/shared-episode relations, source-qualified entity support, delayed-consequence witnesses | may be cold/pageable but remains logically in `z_t` while future authority can reach it |
| **DERIVED** | `_episode_by_id`, incidence maps, active-feature cache, pending heap, source/selection indexes, `last_shared_*` lookup maps | rebuild from causal/evidence state; never checkpoint authority |
| **OBSERVATIONAL** | `last_retrieval` and `last_*_touches` counters | may exist live for assays, but never changes replay identity or checkpoint state |

The audit is intentionally conservative. `exploration_trials`, pending communication/discourse state, action/motor records and exact evidence remain persistent until an intervention battery proves a smaller causal projection. A plausible deletion is not enough.

#### First live observational-state deletion

`reference_mathematical_adult_observational_state_deletion_verify.py` turns that inventory into an actual `ReferenceOrganismV2` representation change. `last_retrieval` remains available immediately after retrieval for observer assays, but it is no longer serialized and legacy checkpoints containing it restore with an empty diagnostic. Repository-wide inbound-use inspection found no later organism transition that reads the previous diagnostic value.

Current reference receipt:

```text
FOUNDRY_MATHEMATICAL_ADULT_OBSERVATIONAL_STATE_DELETION_GREEN
state_role=OBSERVATIONAL
deleted_persistent_field=last_retrieval
legacy_checkpoint_bytes=938534
compact_checkpoint_bytes=938463
bytes_saved=71
future_scenarios=4
negative_control=source_assertions deletion rejected in 3 scenarios
added_transition_touches=0
added_materializations=0
focused_elapsed_ms≈2221
```

The 71-byte saving is again economically small; the architectural gain is that an actual `ReferenceOrganismV2` checkpoint now excludes a proven non-causal distinction. The focused battery preserves public behaviour, causal traces and future update authority across conflict, independent support, source withdrawal and repeat-then-conflict. The language-mastery factory includes this verifier as a standing fast gate, so future changes cannot silently promote this observer diagnostic back into `z_t`.

#### Conditional factorization inside pending action state

The next pass moved from purely observational state into **PENDING** state. An unsettled language action must remain in `z_t` until consequence/repair settles, but not every coordinate inside that action must be duplicated. In the ordinary path, `planned_payload == payload`; the plan is therefore exactly reconstructable from the actual emitted bytes. In a fault path, the values diverge and both become causally necessary because repair and selection credit distinguish intended from actual output.

`reference_mathematical_adult_pending_action_factorization_verify.py` makes that conditional derivation executable. `ReferenceOrganismV2.checkpoint()` now omits `actions[].planned_payload` only when it equals `actions[].payload`; restore already reconstructs the omitted plan from `payload`. Faulted checkpoints retain the distinct plan exactly.

Current reference receipt:

```text
FOUNDRY_MATHEMATICAL_ADULT_PENDING_ACTION_FACTORIZATION_GREEN
state_role=PENDING_WITH_CONDITIONAL_DERIVATION
factored_field=actions[].planned_payload
derivation_guard=planned_payload == payload
legacy_checkpoint_bytes=946650
compact_checkpoint_bytes=946497
bytes_saved_for_one_pending_action=153
fault_negative_control=payload != planned_payload retains both
fault_replay=endogenous repair exact
added_transition_touches=0
added_runtime_materializations=0
focused_elapsed_ms≈601
```

This is the stronger deletion pattern the migration needs: **do not delete a pending causal object merely because most of it is derivable; factor the object so only irreducible distinctions persist.** The ordinary action loses duplicate bytes, while the fault/repair intervention proves the conditional boundary and prevents an over-aggressive compression from erasing future repair authority. This verifier is also part of the language-mastery factory gate.

#### Evidence retention without duplicate retrieval coordinates

The third live V2 deletion applies the same law to **EVIDENCE** rather than observer or pending state. `EpisodeV2` must remain because its lived context/atoms/source/tick can change future episodic completion and ambiguity. But the persisted `EpisodeV2.signature` is an exact deterministic function of `(context, atoms)` plus the fixed `PopulationBankV1` topology; it is a retrieval coordinate over evidence, not an independent learned distinction.

`reference_mathematical_adult_episode_signature_deletion_verify.py` removes `episodes[].signature` from checkpoints and reconstructs it during restore. Legacy signatures are ignored, including a deliberately forged signature, so a redundant cached coordinate cannot become hidden checkpoint authority. The episode evidence itself remains and is the negative control: deleting the target episode changes the later partial-cue future.

Current reference receipt:

```text
FOUNDRY_MATHEMATICAL_ADULT_EPISODE_SIGNATURE_DELETION_GREEN
state_role=EVIDENCE_WITH_DERIVED_RETRIEVAL_COORDINATE
deleted_persistent_field=episodes[].signature
episode_count=18
derived_signature_ints_removed=336
legacy_checkpoint_bytes=938196
compact_checkpoint_bytes=936070
bytes_saved=2126
bytes_saved_per_episode≈118.1
unique_retrieval + settlement identical
ambiguity/silence identical
forged_legacy_signature_has_no_authority=true
episode_evidence_negative_control=rejected
interactive_transition_delta=0
restore_signature_rebuilds=18
focused_elapsed_ms≈1275
```

This is the first deletion in the sequence with meaningful repeated-state leverage: the saving scales with lived episode count while ordinary cognition touches exactly the same retrieval candidates as before. The cost moves to deterministic checkpoint rematerialization, where each episode signature is rebuilt once from retained evidence. The result strengthens the physical/logical split: evidence remains in `z_t`; rebuildable incidence coordinates do not.

#### Canonical Species substrate plus sparse population deltas

The next migration step attacks the largest checkpoint owner rather than another small field. `PopulationBankV1` previously serialized every nominal site and edge coordinate even though most values were exact birth defaults. This confused **Species/birth realization** with the individual's `z_t`. The replacement keeps the same live population mathematics but checkpoints only non-default site/edge state plus explicit topology/territory deviations.

Research and donor constraints are recorded in `docs/research/foundry_population_delta_checkpoint_grounding_2026-08-31.md`. Activity-silent memory research supports separating future-relevant stored state from currently active realization; incremental-checkpoint literature supplies the engineering principle that sparse deltas are economical when the changed state is small; and the existing PyCUDA procedural population already checkpoints touched site/edge state rather than dumping nominal capacity. The host reference adds a stricter lesion rule: canonical topology may be regenerated from Species law, but an individual topology/territory lesion is a causal deviation and must persist as an override.

Current reference receipt:

```text
FOUNDRY_MATHEMATICAL_ADULT_POPULATION_DELTA_CHECKPOINT_GREEN
representation=CANONICAL_SPECIES_SUBSTRATE_PLUS_SPARSE_INDIVIDUAL_DELTAS
legacy_population_checkpoint_bytes=922694
compact_population_checkpoint_bytes=5314
bytes_saved=917380
compression_ratio≈0.00576
site_rows=96
edge_rows=192
canonical_topology_overrides=0
canonical_territory_overrides=0
lesioned_topology_overrides=1
lesioned_territory_overrides=1
legacy_restore + canonicalization=true
future_learning_exact=true
live_eligibility + learned_edge_weight=true
checkpoint_generation≈0.3 ms at 32K and 131K nominal sites in the trained V2 fixture
```

The important result is not JSON compression. The causal factorization is now explicit: deterministic Species/birth substrate is reconstructable; learned, pending and damaged deviations remain individual state. Derived touched-site/edge indexes make checkpoint work proportional to lived/touched population state rather than nominal population size, while sparse lesion override maps prevent reconstruction from silently healing the individual. The first scanning prototype took roughly 55 ms at 32K and 220 ms at 131K; tracking the derived sparse frontier reduced the same checkpoint generation to roughly 0.3 ms without changing the causal state projection.

#### Procedural Species topology in the live population

Once checkpoint persistence no longer required canonical topology arrays, their runtime materialization had to re-earn its cost. It did not. Research grounding and the predeclared falsifier are in `docs/research/foundry_population_procedural_topology_grounding_2026-08-31.md`: implicit graph representations are a valid representation class when adjacency is a function of compact codes, but this is used narrowly because the current Species topology is already an exact deterministic function of `(PopulationSpecV1, site, lane)`. Individual lesions remain explicit sparse overrides in `z_t`.

The live `PopulationBankV1` now computes canonical edge targets and territory values on demand and materializes only `_edge_target_overrides` / `_territory_overrides`. Learned support, timing and edge weights remain ordinary mutable population state in this slice; the next subsection separately factors pending eligibility without changing its causal window.

```text
FOUNDRY_MATHEMATICAL_ADULT_POPULATION_PROCEDURAL_TOPOLOGY_GREEN
representation=IMPLICIT_SPECIES_TOPOLOGY_PLUS_SPARSE_LESION_OVERRIDES
131072-site numeric allocation: 20.0 -> 11.0 bytes/site
131072-site init: ~203 ms donor -> ~0.63 ms challenger
524288-site init: ~886 ms donor -> ~2.46 ms challenger
signature latency: ~7.9 us donor -> ~13.1 us challenger
signature target: <50 us
canonical signature exact=true
topology lesion replay=true
territory lesion replay=true
population delta checkpoint remains GREEN
ReferenceOrganismV2 whole-subject verifier≈77.5 ms in the measured run
language-mastery factory with neighboring mechanisms≈1.09 s
```

This is a deliberate memory/compute exchange: one local signature pays several deterministic integer mixes instead of reading a permanently materialized topology array. The local mechanism remains in the tens-of-microseconds lane, while nominal-population construction and memory fall sharply. More importantly, the representation now matches the causal ownership law: Species structure is executable mathematics; only the individual's learned or damaged deviation is stored as individual state.

#### Pending eligibility as sparse absolute expiry

Eligibility remains causal before expiry: deleting or shortening it changes which future independent consequence may revise the Adult. The removable duplication is the **dense countdown realization**, not the credit window itself. The host reference advances ordinary population `tick` on contact/activation but ages eligibility only on `decay()`, so the sufficient coordinate is an absolute expiry in a separate decay epoch `d`:

```text
e(d) = max(0, tau - d)
```

`PopulationBankV1` now keeps sparse site/edge expiry maps plus expiry buckets and exposes the old fixed-domain `eligibility[i]` / `edge_eligibility[e]` interface as a derived view. `decay()` touches only rows whose expiry equals the new decay epoch. Legacy countdown checkpoints migrate into the sparse representation; repeated activation moves the expiry forward; final-live and first-expired settlement boundaries are unchanged.

```text
FOUNDRY_MATHEMATICAL_ADULT_POPULATION_SPARSE_EXPIRY_GREEN
representation=SPARSE_ABSOLUTE_ELIGIBILITY_EXPIRY
524288 nominal sites; active pending rows=24 sites + 48 edges
numeric resident arrays=8.0 bytes/site
pre-expiry decay touches=0
expiry-boundary decay touches=72
524288-site one-step decay≈0.003 ms in measured run
final live tick credits exactly
first expired tick credits zero
reactivation refresh exact
checkpoint + legacy replay exact
anti-stranding metaplasticity GREEN
ReferenceOrganismV2 GREEN≈26.8 ms in measured run
```

The deletion also produces an honest outward resource-bounded language gain rather than a cosmetic surface change. The removed donor loop visited every site and edge each dialogue tick: at 65,536 sites / fanout 2 that is exactly 196,608 decay visits per turn. Under one fixed decay-work envelope equal to 32 donor turns, the donor law can reach 32 coherent turns while the sparse-expiry Adult reaches all 64 tested turns with the same learned discourse surfaces:

```text
FOUNDRY_MATHEMATICAL_ADULT_POPULATION_EXPIRY_DIALOGUE_BUDGET_GREEN
donor_turns_under_budget=32
challenger_turns_under_budget=64
visible_turn_gain=32
donor_decay_touches_for_64=12582912
challenger_decay_touches_for_64=3012
first="the careful engineer tests the sensor."
last=" also, the quiet engineer tests the valve."
```

This is a PENDING-state factorization: `z_t` still contains the future credit authority, but only as the minimal expiry distinctions that can change a later update. Dormant nominal population capacity no longer receives one countdown byte or one decrement visit merely because it could have become eligible.

Current Workbench falsifiers now strengthen the economic/language side of the same architecture:

```text
FOUNDRY_MATHEMATICAL_ADULT_LEARNING_LOCALITY_GREEN
  one lived lexical confirmation
  -> changed_active_lexeme_rows=1
  -> archive_events_appended=1
  -> active_lexical_json_growth_bytes=5
  -> heldout_language_fanout=4

FOUNDRY_MATHEMATICAL_ADULT_OPERATOR_FACTORIZATION_GREEN
  integrated depth-4 branch: 16 held-out clauses / 639 public bytes
  scale branch: 64 held-out clauses / depth 6 / 2495 public bytes
  current decision width=1
  persisted composite closures=0
  causal-program tree is sole recursive public-surface owner
  source-withdrawal + focal-lesion + incremental-reafference controls preserved

FOUNDRY_MATHEMATICAL_ADULT_CAUSAL_PROGRAM_FACTOR_OWNER_GREEN
  generic causal-program bank owns opaque program->factor relation
  Adult has no second surface shadow namespace
  factor checkpoint replays exact surface
  focal factor lesion blocks transduction while program/credit survive
```

These are reference-only receipts, not Direct capability promotion and not a new sacred causal-program ontology. Together they show the intended direction more concretely than compression percentage alone: a small local `delta_z` can change several held-out language futures, while one already-earned recursive factorization can serve execution and public rematerialization without a duplicate persistent closure tree. Any later representation may replace causal programs too if it preserves or improves these phenotypes, controls, locality and economics. The normal factory should keep these receipts fast enough to remain architecture-iteration gates; if a state-minimization battery becomes slow because of nuisance scale, split the scale claim from the causal equivalence battery rather than accepting a slow inner loop.

### Shared-main mechanic model

The intended development process is concurrent architecture sculpting by many frontier-model mechanics in the one canonical shared `main` checkout. Auxiliary worktrees/clones may assist with read-only inspection or compute, but frontier changes are applied, committed and pushed from shared main. This favors small mechanism definitions, stable executable contracts and local replacement over giant monolithic files or generated compatibility layers.

Concurrent edits and races are expected. Re-read the live source before editing/finalizing, preserve superior concurrent improvements, rerun affected fast assays, and converge on one shared mechanism. Do not create parallel architecture authorities or private development lines merely to avoid collisions.

Capability promotion, constitutional graph state and final hardware evidence retain their proof boundaries. The shared Workbench is the hypothesis/source-semantics construction surface; Direct remains the production qualification backend.

## Workbench tournament

Do not rewrite Direct first.  The first tournament is **not** current cognition versus an intentionally mechanism-poor generic rewrite substrate.  That would confound representation compression with deletion of already-earned cognitive machinery.

Start from the current successful ReferenceOrganism / Recipe / Network / incidence semantics as the donor and replace one representation boundary at a time while preserving its explicit causal function.  Examples include compact mathematical Recipe encoding, lazy/procedural Network realization, compressed learned deltas, or alternative recurrence/interaction algebra.  A deeper mechanism replacement is allowed only after the challenger explicitly reproduces the donor mechanism's phenotype and causal controls.

Run identical curricula and interventions for:

- LM2 rapid aliases + contradiction/reacquisition;
- productive held-out composition;
- recursive reuse;
- long reference under distractors;
- world-language reactivation;
- source withdrawal/reacquisition;
- focal lesion, coalition lesion and matched sham;
- continual interference / anti-stranding;
- checkpoint replay;
- resource exhaustion;
- large procedural population scaling.

Measure:

- phenotype success, with language composition/discourse as a mandatory ratchet rather than an optional downstream check;
- authored law count;
- primitive persistent state types;
- canonical logical state size and physical state bytes;
- `Delta_logical`, physical write amplification and exact archive append per lived update;
- active/persistent expansion ratio: how much current `x_t` computation can be generated from the smaller `z_t` description;
- causally effective interactions and active dimensionality per touched byte/work unit;
- touched work and rematerialization cost;
- stranded persistent matter;
- revision locality and interference with unrelated competence;
- migration/hot-swap cost;
- deterministic replay complexity;
- consumer-hardware latency, bandwidth and energy proxies once the mathematical candidate reaches lowering.

The mathematical substrate wins only if it is equal or better behaviorally while deleting meaningful ontology/legacy cost.

## Failure criteria

Reject or weaken the hypothesis if the generic substrate requires so much hidden metadata or rewrite dispatch that it simply recreates Recipes/Networks under worse names; if language/cognition becomes harder to learn or revise; if hot swaps require global semantic migrations; if deterministic conflict resolution becomes a centralized solver; or if scalable hardware lowering is substantially worse than specialized structures without compensating organism benefit.

## Immediate design consequence

Do **not** add new persistent resident object classes while this tournament is open unless a fast falsifier demonstrates they are necessary.

For new language/cognition work, first ask:

> Can this capability be expressed as a local change in the Adult's existing mathematical state/operator factorization while preserving one organism-level transition semantics?

Do not force it into local rewrite dynamics merely for uniformity. Introduce a new named primitive or mathematical family only when the existing heterogeneous operator algebra cannot reproduce the required phenotype/causal controls efficiently enough.
