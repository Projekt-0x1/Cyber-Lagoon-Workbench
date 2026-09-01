#!/usr/bin/env python3
"""RED/contrast: earned program span pieces must have one lifetime owner.

Language evidence already retains exact historical span mathematics plus source provenance.
An earned causal-program factor names that witness. The Adult must not copy the same
piece tuple into a second template table merely to survive withdrawal or later ambiguity.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from reference_hierarchical_composition_v1 import HierarchicalRefuse
from reference_mathematical_adult_operator_factorization_verify import build, JOIN


def main():
    started=time.perf_counter()
    adult,leaves,_deep,top,_programs,_roots=build(True)
    pid=top.identity
    expected=adult.public_surface(pid)
    factor=int(adult.programs.factor(pid))
    historical=adult.language.historical_span_pieces(factor)
    historical_rows=[key for key in adult.language._span_sources
                     if adult.language.span_factor_identity(key[0],key[1],key[2])==factor]

    # Competing current evidence must affect new induction without rewriting the
    # exact historical factor already earned by the existing causal program.
    adult.observe_join(JOIN,leaves[0],leaves[1],6101,separator=b' / ')
    adult.observe_join(JOIN,leaves[1],leaves[0],6102,separator=b' / ')
    current_ambiguous=adult.language.span_template(JOIN,2) is None
    try:
        adult.compose(JOIN,top.identity,leaves[0]);new_structure_refused=False
    except HierarchicalRefuse:
        new_structure_refused=True

    adult.language.withdraw_source(5002)
    after_withdrawal=adult.language.historical_span_pieces(factor)
    without_copy=adult.public_surface(pid)

    checkpoint=adult.program_surface_checkpoint()
    adult._surface_leaf_surfaces={};adult._surface_leaf_families={};adult._surface_leaf_family_index={};adult.programs.factors={}
    adult.restore_program_surface_checkpoint(checkpoint)
    replay=adult.public_surface(pid)

    checks={
        'historical_span_identity_is_unique':historical is not None and len(historical_rows)==1 and adult.language.span_factor_identity(JOIN,2,historical)==factor,
        'competing_evidence_ambiguates_new_structure':current_ambiguous and new_structure_refused,
        'historical_witness_survives_source_withdrawal':after_withdrawal==historical,
        'adult_template_copy_is_deleted':not hasattr(adult,'_surface_templates'),
        'earned_program_survives_without_template_copy':without_copy==expected,
        'surface_checkpoint_has_no_template_payload':'templates' not in checkpoint,
        'checkpoint_replays_through_language_witness':replay==expected,
        'bounded_fast_path':time.perf_counter()-started<1.0,
    }
    failed=[key for key,value in checks.items() if not value]
    result={
        'schema':'cyber-lagoon.reference-mathematical-adult-span-template-owner.v1',
        'pass':not failed,
        'reference_only':True,
        'state_boundary':'LANGUAGE_EVIDENCE_OWNS_HISTORICAL_SPAN_PIECES_CAUSAL_PROGRAM_OWNS_FACTOR',
        'factor':factor,
        'recursive_bytes':len(expected),
        'checkpoint_bytes':len(json.dumps(checkpoint,sort_keys=True,separators=(',',':')).encode()),
        'checks':checks,
        'elapsed_ms':round((time.perf_counter()-started)*1000.0,3),
    }
    print('FOUNDRY_MATHEMATICAL_ADULT_SPAN_TEMPLATE_OWNER_'+('GREEN' if not failed else 'RED'))
    print(json.dumps(result,indent=2,sort_keys=True))
    if failed:raise SystemExit(1)


if __name__=='__main__':main()
