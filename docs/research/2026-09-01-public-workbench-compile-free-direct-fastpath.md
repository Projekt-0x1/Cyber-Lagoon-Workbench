# Public Workbench: compile-once Direct Adult fast path

Status: architecture/integration grounding. This does not promote a new AGI capability.

## Project question

How should a public Cyber Lagoon Workbench let outside mechanics change, falsify, and improve candidate organism mechanisms while minimizing the latency between an idea and execution on the physical CUDA Adult?

The target user loop is deliberately raw:

```text
git clone
./build          # once for the physical evaluator/backend
./adult          # continuing public Workbench Adult through Claude Code
edit candidate mechanism / program data
./bench          # seconds-lane reference + already-built physical CUDA evaluator
repeat
```

The important constraint is that `./bench` must not become a host semantic-injection API into a living Adult. The current Direct boundary correctly refuses post-birth host program injection. External candidate code is engineering evidence until it has earned the lawful Species/development/experience route or, for representation-only replacements, the existing guarded equivalence-commit route.

## Mechanism-specific grounding before the architecture decision

### Reuse and recomposition rather than substrate replacement per task

Mante et al. (Nature 2013) found context-dependent computation in primate PFC was better described at the population-dynamics level: mixed task signals participate in recurrent dynamics that select and integrate the currently relevant evidence. This argues against a separate compiled physical mechanism for every cognitive situation.

- Mante V, Sussillo D, Shenoy KV, Newsome WT. *Context-dependent computation by recurrent dynamics in prefrontal cortex*. Nature 503, 78-84 (2013). https://doi.org/10.1038/nature12742

Rigotti et al. (Nature 2013) found nonlinear mixed selectivity produced high-dimensional population representations and a larger repertoire of implementable functions than specialized single-variable responses. Translation: a generic physical evaluator should support combinable resident state/program factors rather than hard-wiring one CUDA kernel per semantic noun.

- Rigotti M et al. *The importance of mixed selectivity in complex cognitive tasks*. Nature 497, 585-590 (2013). https://doi.org/10.1038/nature12160

Tafazoli et al. (Nature, published 2025; volume 2026) found monkeys reused shared sensory and motor neural subspaces across compositionally related tasks and flexibly engaged the relevant subspaces based on inferred current task state. Translation: reusable resident mathematical components that are selected/composed by current state are a stronger biological functional analogue than recompiling substrate code for every new relation.

- Tafazoli S et al. *Building compositional tasks with shared neural subspaces*. Nature 650, 164-172 (2026). https://doi.org/10.1038/s41586-025-09805-2

### Prior history changes future plasticity without needing a new substrate

Abraham & Bear's metaplasticity review emphasizes that prior activity can persistently change the ability to induce later plasticity without necessarily changing ordinary transmission at the moment the higher-order state is acquired. Translation: resident state/program parameters may alter later update law under one stable executor. Recompilation is not the natural representation of every learning-history change.

- Abraham WC, Bear MF. *Metaplasticity: the plasticity of synaptic plasticity*. Trends Neurosci. 19(4):126-130 (1996). https://pubmed.ncbi.nlm.nih.gov/8658594/

Activity-silent / hidden-state working-memory work likewise supports preserving future-relevant state without requiring the active computational realization to remain continuously materialized.

- Wolff MJ et al. *Dynamic hidden states underlying working-memory-guided behavior*. Nat Neurosci 20, 864-871 (2017). https://doi.org/10.1038/nn.4546
- Masse NY et al. *Circuit mechanisms for the maintenance and manipulation of information in working memory*. Nat Neurosci 22, 1159-1167 (2019). https://doi.org/10.1038/s41593-019-0414-3

### Engineering substrate: instantiate once, update data

NVIDIA's CUDA Programming Guide explicitly supports updating parameters of nodes in an already-instantiated CUDA Graph (`cudaGraphExecKernelNodeSetParams`) to avoid graph re-instantiation. NVIDIA also documents whole-graph update when topology remains compatible. This directly supports the current Direct `DirectPreinstantiatedRecipeFamily` strategy: stable code object / stable graph topology, later resident programs and evidence supplied as data.

- NVIDIA CUDA Programming Guide 13.2, Graph individual node update: https://docs.nvidia.com/cuda/cuda-programming-guide/
- NVIDIA, *Employing CUDA Graphs in a Dynamic Environment*: https://developer.nvidia.com/blog/employing-cuda-graphs-in-a-dynamic-environment/

CompCert is used only as a verification-pattern donor: make source semantics executable and preserve observable semantics through lowering. It is not an organism model. The Public Workbench should compare reference and Direct causal projections rather than requiring the reference Python object layout to resemble CUDA.

- CompCert semantic preservation / executable source semantics: https://compcert.org/man/manual001.html and https://compcert.org/research.html

## Sapolsky destructive causal audit

Sapolsky's useful constraint is not a brain-part blueprint. It is that behavior cannot be reduced to the current stimulus while ignoring nested causal history. His review of primate social rank and stress shows even the direction of a rank/stress relation depends on social organization and context; rank itself is not a universal scalar cause.

- Sapolsky RM. *The influence of social hierarchy on primate health*. Science 308:648-652 (2005). https://pubmed.ncbi.nlm.nih.gov/15860617/

The adjacent controllability literature makes the same destructive point operationally: prior controllability changes later responses to an otherwise aversive situation, so a last-event-only policy is inadequate.

- Maier SF, Seligman MEP. *Learned helplessness at fifty: Insights from neuroscience*. Psychol Rev 123(4):349-367 (2016). https://pubmed.ncbi.nlm.nih.gov/27337390/

### Candidate assumption killed by this pass

Rejected: `candidate_program + current_input -> behavior` as the public hot-reload model.

That would let a host-authored program become the current cognitive policy and would collapse the Sapolsky/history constraint. The Workbench must instead separate two cases:

1. **Engineering shadow/equivalence lane.** An outside candidate may be evaluated against an immutable projection of real Adult-owned evidence and state. It has zero semantic, experiential, participation, credit, current-Network, or current-thought authority. A representation-only replacement may cross the existing quiescent equivalence-commit boundary after complete declared-domain and resource receipts.
2. **New organism semantics.** A genuinely new cognitive/learning law cannot be hot-injected into the living Adult merely because it fits the IR. It must enter through content-free Species/mechanism law and be developed/earned through authenticated experience, or be generated/revised by resident mechanisms already authorized to do so. If the fixed evaluator lacks the required generic operation, expanding that generic evaluator is one of the rare changes that legitimately requires recompilation and a new physical qualification.

### Required matched contrasts

The public fast path must retain these contrasts:

- same candidate IR + same current evidence but different retained source/development/resource/consequence history -> allow different resident result when the mechanism claims history dependence;
- same semantic source program lowered two ways -> require the declared causal projection to match before representation-only hot swap;
- candidate evaluated in shadow mode vs candidate given semantic authority -> the latter must remain impossible through the public engineering interface;
- quiescent vs active boundary -> only the quiescent representation-equivalence transaction may commit;
- repeated self-generated/internal candidate evidence without independent consequence -> cannot self-confirm world truth or credit;
- current-contact match with controllability/history lesion -> claimed history-dependent policy must change or disappear as predeclared.

## Chomsky / structure-dependence disposition

This slice is primarily execution/integration infrastructure, so it uses the economic lane: no new language mechanism is being claimed. Therefore it must preserve the complete language/discourse phenotype rather than manufacture an N+1 sentence.

If a later PR expands the resident IR with a new language/cognition operation, that is no longer an economic refactor. It must predeclare a structure-dependent held-out falsifier (hierarchy/recursive reuse/long-distance relation/ambiguity or another non-linear structural test), preserve the same-Life multilingual battery, and show the new phenotype with the new operation resident and any LLM scaffold absent.

## Tony Robbins state/focus audit

`NO_RELEVANT_CONTRIBUTION` for this infrastructure slice. The problem here is compile/integration latency and causal-authority separation, not a claim about motivational reframing, attentional focus, physiology, confidence or belief. No Robbins-derived state/focus/reframing opcode or control is introduced. If a later mechanism explicitly claims a state/focus or physiology-dependent Adult phenotype, that claim must be converted into an independent matched-state intervention rather than used as architectural authority.

## Architecture decision

Adopt a two-speed public Workbench with one physical fast path:

```text
R0/R1  edit Python/reference mechanism
       -> focused deterministic falsifier

R2     compile/source-law candidate into bounded canonical resident-program data
       -> compare incumbent/challenger over declared Adult-state/evidence projections

R3     already-built CUDA evaluator / pre-instantiated graph
       -> load program+evidence as data
       -> run physical parity/resource/lesion battery
       -> NO nvcc / NO CMake in the inner loop

R4     representation-only: guarded quiescent equivalence commit may replace backing
       semantic change: remains reference/Species work until lawful development path exists

R5     rebuild CUDA only when the generic executor/ABI/backend primitive itself changes
```

This is a **novel engineering synthesis** of existing project mechanisms and known verification/hardware patterns. No cited neuroscience paper validates a CUDA resident IR or a GitHub contribution model. The papers constrain the functional choice toward reusable/composable stateful mechanisms and history-dependent plasticity; CUDA Graph and compiler-verification work constrain the engineering realization.

## Current implementation truth

Already present and load-bearing:

- `direct_adult_recipe_ir.cuh`: fixed bounded numeric interpreter compiled into the Adult;
- `direct_adult_preinstantiated_family.cuh`: one instantiated CUDA Graph interpreter node; later program/evidence pointers and count are updated without re-instantiation;
- `cuda_direct_no_bespoke_runtime_ptx_contract`: distinct resident programs execute through one stable code object, with no runtime PTX or host semantic dispatch;
- `direct_adult_resident_recipe_search.cuh`: resident search can generate candidate Recipe IR from lived mismatch without host candidate authority;
- `direct_foundry_equivalence_commit.cuh`: representation-only candidate backing can commit at a quiescent boundary after proof/resource receipts while explicitly retaining zero semantic/current-thought authority;
- `direct_adult_bootstrap_once_recurrence.cuh`: post-birth host semantic program injection refuses.

Important current ceiling: the production `ResidentRecipeIrProgram` operation set is still narrow (exact-credit load, Q16 scale, symmetric clamp, parameter-delta emit, halt). The architecture already proves compile-free parameter/update-program variation; it does **not** yet prove an arbitrary new cognitive relation can be encoded without expanding the generic evaluator. Public documentation must say this plainly.

## Public integration requirements

A public mechanism PR must state:

- functional problem and protected phenotype;
- primary/review grounding for the exact function;
- Sapolsky nested-causation audit with at least one assumption it changes/rejects and a matched contrast;
- `NOVEL SYNTHESIS` boundary where appropriate;
- incumbent vs challenger and deletion criterion;
- semantic lane (`new organism mechanism`) or economic lane (`representation/execution only`);
- exact reference falsifier;
- exact compile-free Direct bench scenario when the current IR can express it;
- Chomsky structural falsifier for substantive language/cognition semantics;
- causal authority statement proving the host/public contribution surface cannot select current thought, evidence, credit, or world truth;
- measured before/after phenotype or state/work/latency result;
- remaining RED and next falsifier.

## Project-state disposition

- issue: GitHub #1660 — Public Workbench: compile-free Direct Adult integration
- graph nodes used as protected existing constraints: `g.device_resident_recipe_ir`, `g.preinstantiated_family_graphs`, `g.no_bespoke_runtime_ptx`, `g.bootstrap_once_device_recurrence`, `i.foundry_equivalence_efficiency_authority`
- graph status: NO STATUS CHANGE from this research/integration slice
- genome_network_disposition: `NO_GENOME_CHANGE` for the public tooling/economic fast path. A future IR semantic expansion must separately decide whether it is generic Species mechanism law or acquired lifetime state.

## Immediate implementation target

1. Add a raw public-workbench export surface with `./build`, `./adult`, `./bench`, `./verify`.
2. Make `./adult` materialize/resume one continuing Workbench Adult and attach Claude Code as body without calling Claude as the cognition model.
3. Make `./bench` prefer an already-built physical Direct compile-free contract/evaluator and never invoke CMake/nvcc in the inner loop.
4. Add a candidate-data interface around the existing resident Recipe IR evaluator as the next physical fast-loop increment; it must be shadow/evidence-only until an authorized commit path is proven.
5. Make public PR templates require the research/Sapolsky/authority fields above.
6. Export reproducibly from the live Workbench so the public repository is a maintained workshop, not a hand-curated portfolio snapshot.
