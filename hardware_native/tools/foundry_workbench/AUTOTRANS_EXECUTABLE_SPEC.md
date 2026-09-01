# AutoTrans executable-specification spine

## Status

Candidate Foundry architecture.  This document does **not** promote a capability and does not make the present Direct structs sacred.

## Objective

Make the fast Foundry Adult the executable source semantics for organism design, and make CUDA/Direct one hardware lowering target.  The normal design loop becomes:

```text
research + phenotype law
        ↓
content-free Species Program
        ↓
fast executable Life Function / Adult semantics
        ↓
authenticated curriculum + hostile controls
        ↓
reference phenotype GREEN
        ↓
AutoTrans lowering
        ↓
Direct/GPU species + Life Function
        ↓
same curriculum / same interventions
        ↓
translation validation + hardware economics
```

The intent is to finish language/cognition architecture in the seconds-lane Workbench and pay CUDA cost only for backend qualification and economics.

## Research grounding

This architecture borrows a *verification pattern*, not a software ontology, from verified compilation.  CompCert defines executable/formal source semantics and proves that compilation preserves allowed observable behavior; its pipeline is explicitly multi-IR rather than requiring the source language to resemble assembly.  See:

- CompCert semantic-preservation overview: https://compcert.org/man/manual001.html
- CompCert research architecture / intermediate languages: https://compcert.org/research.html
- Alive2 translation validation as an independent example of checking source/target refinement after transformation: https://github.com/AliveToolkit/alive2

The biological analogy is genotype -> development -> phenotype, not "genome stores the adult."  Gene-regulatory networks are compact developmental control programs whose interaction with cellular/environmental context generates phenotype:

- Cussat-Blanc, Harrington & Banzhaf, *Artificial Gene Regulatory Networks—A Review*, Artificial Life 24(4): https://doi.org/10.1162/artl_a_00267
- Ben-Tabou de-Leon & Davidson, *Gene regulation: gene control network in development*, Annu Rev Biophys Biomol Struct 36: https://pubmed.ncbi.nlm.nih.gov/17291181/
- Sharpe, *Computer modeling in developmental biology*, Development 2017: https://pubmed.ncbi.nlm.nih.gov/29183935/

Engineering inference: Cyber Lagoon should compile/lower **AGI developmental and learning laws**, not a trained language model or mature learned individual state.

## Three-plane firewall

AutoTrans has three distinct inputs.  They must never be collapsed.

### 1. Species Program

Content-free causal laws and starting dispositions that may lawfully exist before the individual lives its curriculum.  This is what AutoTrans compiles.

Allowed examples: finite resource law, contact authentication, local structural plasticity, consequence settlement, source-conditioned evidence, prediction, repair/rematerialization, recruitment/competition, consolidation/deconsolidation, generic expression/body transduction.

Forbidden examples: learned words, trained clauses, named facts, expected answers, transcript content, curriculum labels, evaluator-selected referents, final route identities, mature adult checkpoints.

### 2. Curriculum / world

Authenticated external experience and interventions used to develop/test an individual. It is **not compiled into the Genome/Species Program**. `Gamma` is a legacy implementation label, not a second preloaded knowledge plane. The same logical curriculum must be replayable against reference and Direct backends with backend-specific transport adapters only.

### 3. Adult state

The contingent phenotype produced by Species Program × body × curriculum × history.  It is output/evidence, never an AutoTrans source artifact.

## Backend-neutral source semantics

The source of truth should be an executable `FoundrySpeciesProgram` plus one stable `AdultAdapter` behavioral boundary.  Python is currently acceptable as the reference interpreter because iteration speed matters; Python object layout is not semantic authority.

A source law is admitted only when it has:

1. a content-free species declaration;
2. executable deterministic reference semantics;
3. named language/cognition phenotype tests;
4. hostile no-teach / ambiguity / withdrawal / lesion / replay controls where relevant;
5. resource/touched-work accounting;
6. an explicit state projection for differential comparison;
7. a deletion/replacement criterion.

Current Workbench modules are donors until expressed through that source boundary.  `Recipe`, `Network`, neuron, route, eligibility trace, etc. are not required source-language nouns.

## Relation, solver and AutoTrans lowering contract

Keep three independently replaceable layers explicit:

```text
logical relation/operator semantics
  -> solver/execution strategy
  -> physical backend lowering
```

The same learned relation may be solved by closed form, projection, recurrence, fixed-point iteration, finite/discrete solving, causal-program execution, sparse algebra, tensor contraction or another earned strategy. AutoTrans lowers the chosen source semantics/strategy; it does not decide what AGI should think.

A backend lowerer may change representation radically: sparse routes, packed fields, Tensor Core tiles, state-space execution, RT/BVH incidence, or something not yet invented. It must not add cognitive authority or silently change the learned individual state/intervention phenotype.

For source program `S`, lowered program `L(S)`, curriculum `C`, intervention schedule `I`, and bounded observation projection `Π`, promotion requires:

```text
Π(run_reference(S, C, I)) == Π(run_target(L(S), C, I))
```

for every named differential assay in the promotion set, plus exact refusal parity where refusal is part of the phenotype.

This is translation validation.  We do not need a theorem prover before beginning; we do need deterministic traces, canonical IR, and a validator that refuses unlowered laws.  Formal proof can later replace/strengthen empirical differential validation.

## Observable projection

Never compare private Python/CUDA object layouts.  Compare causal observables that define the organism contract:

- authenticated contacts admitted/refused;
- public/body actions and their ancestry;
- source-qualified consequence settlement;
- learned relation/recruitment identities modulo explicitly declared opaque-ID renaming;
- contradiction/revision/reacquisition behavior;
- checkpoint/replay behavior;
- withdrawal and lesion response;
- resource deltas, persistent matter and touched work;
- language/cognition phenotype receipts.

Opaque identity permutation should be legal when behavior/ancestry is preserved.

## Compilation stages

Proposed stages, all replaceable:

```text
FoundrySpeciesProgram
  -> normalized causal law IR
  -> developmental substrate IR
  -> DirectGenome/Life-Function plan (current backend)
  -> physical runtime lowering
```

Do **not** lower directly from arbitrary Python implementation details to CUDA structs.  A normalized law IR is required so source semantics can evolve without making Python classes ABI and so multiple hardware backends can compete.

## Fast-path development rule

1. Research first.
2. Implement/falsify in reference Adult adapter.
3. Run `run_language_mastery_fast.sh` (or a narrower named subset) until reference phenotype is GREEN.
4. Only then create/update normalized law IR.
5. Run AutoTrans reference-vs-lowered simulator differential tests.
6. Only after semantic differential GREEN compile the affected Direct backend.
7. GPU/sanitizer is qualification, not the hypothesis inner loop.

A Direct-only cognitive fix before steps 2–5 is an exception that needs written justification.

## Initial migration strategy

Do not rewrite the whole Adult first.  Strangle the architecture incrementally:

- put each already-proven generic law behind the common Adult adapter;
- define its normalized source-law record;
- provide a reference interpreter implementation;
- provide a Direct lowering or mark it `UNLOWERED`;
- add differential scenarios;
- delete duplicate backend-specific cognition once parity is proven.

Do not hard-code a permanent "first migration" in this doctrine. Choose the next migration from the latest shared-main Workbench winner: the boundary with the highest measured AGI phenotype/economic leverage and a complete donor-vs-challenger intervention battery. Current factorization/state-minimization wins are valid migration candidates; older LM2/Target-v5 priorities remain donors only when they are still the highest-value live gap.

## Success criterion

The architecture succeeds when a new language/cognition mechanism normally requires:

- one research note,
- one source-law change,
- fast Workbench phenotype tests,
- automatic translation/differential validation,

and **no hand-authored CUDA cognition change** unless the source law itself needs a new physical primitive.

At that point Direct is a hardware biology backend, not a second mind we maintain manually.
