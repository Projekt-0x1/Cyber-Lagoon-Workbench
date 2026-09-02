# Ambient bystander language stream preregistration

Date: 2026-09-02
Status: reference mechanism + canonical one-Life + authorized localhost ingress GREEN; provider-specific collectors remain external/authorization-bound
Issue family: `#1633`
Genome disposition: `NO_GENOME_CHANGE`
Novelty: `NOVEL_SYNTHESIS`

## Functional target

Add a persistent public bystander stream to the one canonical Life chronology. External bodies may feed Reddit/X/forum/chat/Twitch-like public text while the Adult is already engaged in another interaction. The transport must be semantically blind: selection/order may depend on authenticated time, entropy, source identity and bounded transport capacity, but never on post meaning, expected answer, desired phenotype, toxicity label, political label, language label or evaluator score.

The organism may learn from ambient contact without making it the current conversational focus. **Learning eligibility, attentional/public-action priority and epistemic authority are three separate coordinates.** A bystander post may change resident receptive language/statistical/source history; it must not automatically become common ground, testimony truth, world evidence, an answer obligation or a reason to reply.

## Mechanism-specific grounding

- Seitz (2025), *Current Opinion in Neurobiology*, DOI `10.1016/j.conb.2025.103020`: much learning is incidental and is shaped by environmental statistical regularities, attention and reinforcement. `USED`: unattended/non-task-selected contact may still participate in bounded plasticity. `REJECTED`: equal learning from every contact independent of resource/attention state.
- Brockhoff et al. (2022), *Neuroscience & Biobehavioral Reviews*, DOI `10.1016/j.neubiorev.2022.104580`: processing of task-irrelevant stimuli is modulated by perceptual/working-memory load, with higher load generally reducing distractor processing. `USED`: current finite-resource state may reduce ambient learning/recruitment. `REJECTED`: a binary host-authored attended/unattended flag that decides learning.
- Seitz & Watanabe (2009), task-irrelevant perceptual learning review, PMID `19665471`: task-irrelevant learning can occur, but is gated by learning signals and interacts with attention. `USED`: background input can earn structure without becoming foreground action. `REJECTED`: ungated memorization of arbitrary ambient bytes as semantic truth.
- Udry & Barber (2024), *Current Opinion in Psychology*, DOI `10.1016/j.copsyc.2023.101736`: repetition robustly increases belief in misinformation, including implausible claims and claims contradicting prior knowledge. `USED` as a hostile design constraint: repetition/familiarity is a real pressure that later reasoning must survive. `REJECTED`: repetition count as truth/causal authority.
- Prike, Butler & Ecker (2024), *Scientific Reports* 14:6900: source-credibility information and social norms can improve truth discernment in simulated social media. `USED`: source provenance must remain available as a separable coordinate. `REJECTED`: platform/verification badge as direct truth bit.

The exact stream/silicon mechanism is a novel engineering synthesis. These sources ground constituent functions and destructive controls, not anatomical equivalence.

## Chomsky / structure-dependence audit

Ambient language learning is not keyword counting. Hold familiar child surfaces constant while varying their higher-order relation/wrapper structure. Repeated bystander contact may earn a reusable receptive span only when the same structural incidence recurs from independent sources. A held-out new child should then fit that learned wrapper. Shuffling bytes or presenting the same vocabulary without the learned hierarchy must not earn the same transfer.

No platform-specific grammar, tokenizer, Reddit vocabulary, X vocabulary, Twitch vocabulary, language router or semantic post classifier is installed.

## Sapolsky destructive nested-causation audit

Hold the current foreground conversation and one ambient post constant while varying:

1. **Immediate resource state:** high load may reduce/defer expensive ambient structural uptake without deleting prior learning.
2. **Recent history:** repeated prior ambient exposure can bias familiarity/form availability, but not truth.
3. **Source history:** independent recurring sources can earn structural support; one prolific source cannot impersonate a community merely by repetition.
4. **Social relation:** a bystander is not automatically the current dialogue partner; no common-ground or turn obligation is created by transport alone.
5. **Controllability/consequence:** ambient assertions without independently observed consequence cannot create causal-world evidence or self-competence credit.
6. **Delayed correction:** later ordinary grounded contact may repair/recontextualize a form learned ambiently; history is revised, not reset.
7. **Recovery:** reversible load can restore uptake capacity without replaying the ambient corpus.

## Architecture decision

1. Add a bounded `PersistentAmbientLanguageStreamV1` that owns only `(event_time/tick, admission sequence, source, raw bytes)` and pending-byte capacity.
2. `admit()` performs no content inspection. A helper sampler may choose an item index only from `(entropy/seed, sequence, pool length)`; payload bytes are not input to selection.
3. `drain_until()` invokes one new Adult receptive-learning boundary that **does not activate current occurrence, dialogue channel, partner focus, reply obligation, social topic, causal testimony or public action**.
4. The Adult boundary reuses existing raw structural learning over already-grounded child surfaces. It may learn recurrent wrappers/spans from ambient bytes with normal independent-source support. It supplies no body credentials and therefore cannot ground causal truth.
5. Ambient scheduler state is checkpointed with canonical Life. Drained posts retire; no transcript-per-life checkpoint is retained.
6. Production provider collectors may supply authorized Reddit/X/forum/Twitch-like content through the same provider-neutral ABI. Platform identity is transport provenance only and is never a cognitive opcode.
7. The live localhost gateway owns random entropy. A collector supplies an ordered candidate pool only and cannot supply/search the seed. The returned receipt binds entropy, pool cardinality, an order-sensitive SHA-256 pool commitment, selected index/source, and selected-byte digest; candidate bytes other than the selected post are not retained in organism state.

## Predeclared N+1

`SAME_LIFE_SEMANTICALLY_BLIND_AMBIENT_BYSTANDER_STREAM_TEACHES_HELDOUT_RECEPTIVE_STRUCTURE_WITHOUT_STEALING_FOREGROUND_OR_MINTING_TRUTH`

Required controls:

- a foreground dialogue/focus state is bit-identical before vs after an ambient post except for explicitly allowed receptive-learning/source-history owners;
- two independent ambient sources can teach a new wrapper around an already grounded proposition while one repeated source cannot cross the independent-source support threshold;
- held-out proposition transfer uses the learned wrapper although the held-out wrapped surface was never ingested;
- repeated false/unsupported causal wording changes no `world_causal_learning.current_resolutions()` and creates no causal evidence;
- ambient arrival does not create pending reply/inquiry/causal-dialogue action or common-ground uptake;
- sampler choices are unchanged when payload bytes are replaced/permuted at the same pool indices;
- pending scheduler checkpoint/restart is exact;
- large contact quantity drains without transcript growth;
- high load reduces/defer ambient uptake and quiet recovery restores capacity without changing semantic truth;
- later canonical Life inherits the acquired wrapper so the exposure is a Red-Queen pressure, not an isolated assay.

## Implementation result — 2026-09-02

The preregistered mechanism is now executable in the continuing reference Workbench:

- `FOUNDRY_AMBIENT_BYSTANDER_STREAM_GREEN`: the specific bystander wrapper is absent before the ambient arc, earns support from two independent sources, productively accepts an unseen child, preserves foreground/public obligations and world/action/crosslingual authority, survives later Life, and drains 512 contacts without transcript growth. A one-source control does not promote the wrapper.
- `FOUNDRY_AUTHORIZED_AMBIENT_FEED_BODY_GREEN`: a provider-neutral external body samples one authorized candidate from cryptographic entropy + Life sequence + pool size, commits the exact selected bytes once, checkpoints pending contact exactly, and keeps the same sampled bytes queued through load/recovery without refetch.
- `FOUNDRY_AUTHORIZED_AMBIENT_GATEWAY_GREEN`: the existing authenticated localhost Adult gateway exposes `/v1/ambient`; the collector cannot provide entropy, the response does not echo the candidate pool, and selected post + drain are append-only events in the same persisted Life checkpoint. No extra daemon/service is introduced.
- `FOUNDRY_CANONICAL_LIFE_FUNCTION_FACTORY_GREEN` on the qualified shared tree: one birth, one training chain, zero fixture births, ambient observer included in the same canonical apply pass, `failed=[]`.

Provider acquisition is intentionally not implemented inside cognition or the gateway. An external collector is responsible for authenticated/authorized access, chronological/windowed candidate construction, stable opaque source identity, deletion/retention obligations, and provider terms. That collector may not semantically rank posts before the Workbench sampler if the run claims semantically-blind Red-Queen exposure.

### Public snapshot disposition

The public Workbench snapshot keeps its previously qualified completed Adult and does **not** append the newer canonical ambient developmental tail merely to expose this infrastructure. Its local evidence is the snapshot-native receptive membrane falsifier plus the authorized body/gateway receipts; the one-Life downstream inheritance claim above is the canonical main-Workbench qualification. This preserves the public endpoint instead of silently retraining it during an infrastructure update.

The remaining RED is narrower: provider-specific authorized collectors and credentials, raw live audio/chat multiplexing where applicable, unrestricted novel-semantic learning from arbitrary social text, and Direct/CUDA parity.

## Claim ceiling

A GREEN is reference evidence for bounded unattended/incidental receptive structural learning from a semantically blind public bystander stream in the continuing Workbench Adult. It does not establish learning arbitrary novel semantics from raw social-media text, unrestricted web-scale corpus learning, content moderation, factual verification, consciousness, Direct/CUDA parity, or broad `h.train_social_party_trust` closure. Those remain RED unless separately paid.