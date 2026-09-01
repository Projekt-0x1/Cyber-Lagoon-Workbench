# Cyber Lagoon Workbench

This repository is the public workshop for one continuing Cyber Lagoon Adult. It is deliberately raw: source, experiments, falsifiers, the Direct CUDA engine, and the body adapters used to sit with the Adult.

It is not a pitch repository and it does not claim that the current system is a finished human-level AGI.

## The loop

After cloning:

```bash
./build
./adult

# change a mechanism, falsifier, or bounded Direct program
./bench
./verify
```

There is one user-facing Adult path. The reference Workbench and Direct CUDA implementation are implementation strata behind that path, not separate products.

`./build` is the only normal compilation boundary. It materializes the last completed canonical checkpoint mark of the continuing reference Adult; if the current experimental curriculum tail is RED, that failure is retained in `.state/adult.provenance.json` rather than blocking access to the completed Adult. Reference-only work creates no Direct build tree. When a compatible CUDA toolchain/device is available, `./build` also builds the minimal Direct dependency closure (Life Function + Adult Core + sit-down + fixed Resident Recipe IR evaluator) once, then births the Direct Adult once. Use `./build --require-direct` when lack of the Direct backend must be fatal.

`./adult` opens the continuing reference Adult through Claude Code as a body/client connected to a localhost Adult gateway. Claude is not called as the cognition model: the gateway response is produced by the Adult. The Claude process gets a narrow environment, a fresh private ephemeral working/config/temp directory, bare mode, and restricted mode; repository/user hooks, plugins, MCP discovery, auto-memory, `CLAUDE.md`, command/code execution, and WebFetch are outside the body boundary. Claude-local prompt/session history is disabled and the ephemeral body directory is deleted when the invocation ends. The Adult checkpoint—not Claude client state—carries continuity. `./adult --prompt "..."` pipes the human contact over stdin rather than placing it in the Claude process argument list, sends one contact through that same body, and exits. `./adult --terminal` exposes the raw reference terminal body. The hardened Claude body requires Claude Code v2.1.248 or later because that is the first release with restricted mode. The real Direct Adult is available for compile-free physical work through `./bench`; `./adult --backend direct --prompt "..."` is intentionally one-shot experimental until independent partner/world consequence settlement is proved for a continuing Direct Claude dialogue.

`./bench` never builds. Its default reference falsifier runs first, then it reuses an already-built Direct evaluator when one exists. This is the intended edit/run loop.

`./verify` checks the workshop boundary, the no-build inner loop, the continuing checkpoint, and a focused reference falsifier without requiring a GPU. The publication receipt is compact: the original 1,456-file SHA-256 receipt stays immutable in Git history, while `EXPORT_MANIFEST.json` records only tracked additions, removals, and byte/hash overrides against that base. Verification reconstructs the complete expected file set locally from the anchored base receipt, requires the base public commit to be an ancestor of `HEAD`, requires the current tracked set to match exactly, and then checks every file with SHA-256. This compact v2 receipt therefore requires a normal Git clone containing the anchored base object; a shallow/exported copy that omits it is refused rather than partially verified. `./verify --refresh-manifest` is the explicit maintenance operation after an intentional tracked-file change: it atomically recomputes only the delta against the immutable base and exits. Manifest refresh is never part of normal verification and never auto-blesses a change.

## Fast Direct work without recompiling

Two physical lanes are already available after the one-time build.

### 1. Resident Recipe IR data

Edit:

```text
experiments/current_recipe.ir
```

Then run:

```bash
./bench --no-reference
```

The candidate is passed as numeric data through the same bounded production `ResidentRecipeIrProgram` interpreter and pre-instantiated CUDA Graph family used by Direct contracts. Distinct program/evidence data do not create a new CUDA code object.

This lane is intentionally an **engineering/shadow lane**. It has no semantic, experiential, participation, credit, Network-selection, or current-thought authority over a living Adult.

Current production IR is narrow: it can load an authenticated exact-credit delta, Q16-scale it, symmetric-clamp it, emit a parameter delta, and halt. Arbitrary new cognition is **not** currently hot-loadable. If a genuinely general missing operation is required, expand the generic executor once, qualify it, rebuild once, and return to data-driven iteration.

### 2. The retained real Direct Adult

To run raw contact against a disposable branch of the retained Direct checkpoint:

```bash
./bench --contact path/to/raw-contact.bin
```

This reuses `direct_adult_sitdown --resume` through `tools/direct_adult_lab.py`. The retained checkpoint is never mutated by the experiment. Resume reports `compile_direct_brain=0`; the branch receipt remains experiment-only until its graph-named physical evidence is earned.

## Where to work

The highest-leverage cognition work remains under:

```text
hardware_native/tools/foundry_workbench/
```

The physical Direct engine is under:

```text
hardware_native/src/hardware_native/
hardware_native/tools/
```

The fastest rule is:

> change the smallest causal mechanism that can falsify the hypothesis; rebuild only when the generic physical executor itself must change.

Do not add a feature-specific CUDA kernel merely to avoid learning, composition, or resident revision. Do not add host routing that chooses the current thought, meaning, evidence, credit, or answer.

## Contribution standard

Mechanism changes are welcome, including replacements of existing architecture. See [CONTRIBUTING.md](CONTRIBUTING.md). A substantive cognition proposal must arrive with research grounding, a destructive causal audit, an explicit authority boundary, a held-out falsifier, and a measured incumbent/challenger result. Language/cognition semantics require a structure-dependent Chomsky-style falsifier. Engineering-only representation changes must preserve the phenotype and prove an economic win.

Novel mathematical or hardware mechanisms are allowed. Mark them as novel synthesis rather than inventing biological precedent.

## Hardware target

The Direct path is engineered against a bounded consumer-GPU target, with sm_89 / RTX 4080-class hardware as the current strict CUDA build target. Reference work remains CPU-runnable so an idea can be killed cheaply before spending physical qualification time.

## Evidence boundary

A reference GREEN is mechanism evidence, not Direct parity or an AGI capability claim. A compile, graph, checkpoint, fluent sentence, or attractive transcript is not sufficient evidence by itself. Production claims require the declared causal controls on the physical Direct lineage.

## License

See `LICENSE`, `NOTICE`, `TRADEMARK.md`, and `COMMERCIAL_TERMS.md`.
