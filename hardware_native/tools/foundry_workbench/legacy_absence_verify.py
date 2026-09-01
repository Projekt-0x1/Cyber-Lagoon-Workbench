#!/usr/bin/env python3
from __future__ import annotations
import ast
import hashlib
import re
from pathlib import Path

ROOT = Path(__file__).parent
REMOVED = frozenset("""
audit audit_workbench authoring_planes authoring_planes_verify bitbus_observer
bitbus_observer_verify bootstrap_brain build_language_direct_relseq_cuda_contract
cognitive_architecture_v1 cognitive_architecture_v2 cognitive_architecture_v3
cognitive_architecture_verify condensation_verify continuity_architecture_v1
continuity_architecture_v2 continuity_architecture_verify control_architecture_v1
control_architecture_verify control_constructor_architecture_v1
control_constructor_architecture_verify dependency_graph_verify design_catalog_verify
design_registry foundry full_loop hive_beedance hive_beedance_verify
language_agreement_nonlocal language_ambiguity language_architecture_v1
language_architecture_v2 language_architecture_v2_verify language_architecture_v3
language_architecture_v3_verify language_architecture_v4 language_architecture_v4_verify
language_architecture_verify language_common_ground
language_crossmodal_prose_snapshot_v1 language_developmental_snapshot
language_developmental_snapshot_v2 language_developmental_snapshot_v6
language_direct_relseq_parity language_distributed_lexicon_snapshot_v1
language_foundry_verify language_hierarchical_snapshot_v1
language_learned_morphology_snapshot_v1 language_learned_reference_snapshot_v1
language_learner_quantity_audit language_learning_architecture_v1
language_learning_architecture_v2 language_learning_architecture_v3
language_learning_architecture_v4 language_learning_architecture_v5
language_learning_architecture_v6 language_learning_architecture_v7
language_learning_architecture_v8 language_learning_architecture_v9
language_learning_architecture_v10 language_learning_architecture_v11
language_learning_architecture_v12 language_lexical_competition
language_message_style language_multicontext_prose_snapshot_v1
language_paragraph_autonomy language_population_quantity_verify
language_productive_morphology_snapshot_v1 language_quantity_scaling_v1
language_recursive_prose_snapshot_v1 language_rhetorical_prose language_rung1
language_source_epistemics learning_dynamics_verify lived_organism_simulator
organism_hot_sync_verify organism_runtime planning_architecture_v1
planning_architecture_v2 planning_architecture_verify pool_io pool_io_smoke
quantity_scaling_audit reference_vm reference_organism_v1 reference_organism_verify reference_partner_history_verify
reference_lexical_competition_verify reference_construction_competition_verify
reference_source_epistemics_verify reference_surface_segmentation_verify reference_integrated_organism_verify
starting_cognitive_organism_v1
starting_cognitive_organism_verify starting_cognitive_vehicle_v1
starting_cognitive_vehicle_v2 starting_cognitive_vehicle_v2_verify
starting_cognitive_vehicle_verify strict_authority strict_authority_verify
strict_runtime vehicle_hot_sync_verify vehicle_runtime
""".split())


def imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text())
    out: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            out.add(node.module.split(".")[0])
    return out


def main() -> None:
    present = sorted(name for name in REMOVED if (ROOT / f"{name}.py").exists())
    if present:
        raise SystemExit("legacy:present:" + ",".join(present))
    stranded = {
        path.name: sorted(imports(path) & REMOVED)
        for path in ROOT.glob("*.py")
        if path.stem not in REMOVED and imports(path) & REMOVED
    }
    if stranded:
        raise SystemExit("legacy:stranded_import:" + repr(stranded))
    source = "\n".join(path.read_text() for path in ROOT.glob("*.py"))
    forbidden = {
        "host_goal": r"def\s+enqueue_goal\s*\(",
        "runtime_applicability": r"\.applicable\s*\(",
        "caller_pass": r"(?:facts\s*\[\s*['\"]pass|facts\.get\(\s*['\"]pass)",
        "runtime_transformer": r"subprocess\.run\s*\(\s*command",
    }
    hits = [name for name, pattern in forbidden.items() if re.search(pattern, source)]
    if hits:
        raise SystemExit("legacy:authority_primitive:" + ",".join(hits))
    root = hashlib.sha256("\n".join(sorted(REMOVED)).encode()).hexdigest()
    print(
        "FOUNDRY_LEGACY_ABSENCE_GREEN "
        f"removed_python_modules={len(REMOVED)} removed_set_sha256={root} "
        "stranded_imports=0 host_goal=0 runtime_applicability=0 caller_pass=0"
    )


if __name__ == "__main__":
    main()
