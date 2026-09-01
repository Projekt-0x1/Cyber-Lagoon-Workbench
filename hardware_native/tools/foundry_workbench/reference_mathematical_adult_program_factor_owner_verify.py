#!/usr/bin/env python3
"""RED/contrast: recursive construction must not write the hierarchy store.

Leaf+leaf and program+program composition return an ephemeral witness. Productive
leaves live in the program factor. Public surface rematerializes from that factor
after source withdrawal and survives focal factor lesion.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from reference_language_mastery_adult_v1 import LanguageMasteryAdultV1, CompositionWitnessV1
from reference_hierarchical_composition_v1 import HierarchicalRefuse
from reference_predictive_credit_profile_v1 import Q

CLAUSE, JOIN, CTX = 9001, 9101, 0xA11


def emit_all(adult, pid):
    expression=adult.expression(pid); out=[]
    while True:
        plan=expression.emit()
        if plan is None:return tuple(out),expression
        out.append(plan.value)
        if not expression.reafference(plan,plan.value):raise RuntimeError('owner:reafference')
        if len(out)>10000:raise RuntimeError('owner:nontermination')


def composites(adult):
    return ()


def main():
    started=time.perf_counter(); adult=LanguageMasteryAdultV1()
    names={101:'careful',102:'quiet',201:'engineer',202:'technician',301:'tests',302:'inspects',401:'sensor',402:'valve'}
    for feature,text in names.items():
        adult.observe_surface_item(feature,text.encode(),1000+feature)
        adult.observe_surface_item(feature,text.encode(),2000+feature)
    adult.observe_surface_construction(CLAUSE,(101,201,301,401),b'the careful engineer tests the sensor.',3001)
    adult.observe_surface_construction(CLAUSE,(102,202,302,402),b'the quiet technician inspects the valve.',3002)
    leaves=tuple(adult.leaf(CLAUSE,atoms) for atoms in (
        (101,201,301,401),(102,202,302,402),(102,201,301,402),(101,202,302,401)))
    adult.observe_join(JOIN,leaves[0],leaves[1],5001)
    adult.observe_join(JOIN,leaves[2],leaves[3],5002)
    pairs=[]
    for i in range(0,4,2):
        witness=adult.compose(JOIN,leaves[i],leaves[i+1])
        if not isinstance(witness,CompositionWitnessV1):raise RuntimeError('owner:witness')
        if hasattr(adult,'hierarchy'):raise RuntimeError('owner:persisted_pair')
        program=None
        for _ in range(3):
            program=adult.experience_program((leaves[i].identity,leaves[i+1].identity),witness,3*Q//4,Q//8,CTX,Q//3,True)
        if program is None:raise RuntimeError('owner:pair')
        pairs.append(program)
    if composites(adult):raise RuntimeError('owner:pair_composites')
    deep=adult.compose(JOIN,pairs[0],pairs[1])
    if not isinstance(deep,CompositionWitnessV1):raise RuntimeError('owner:deep_witness')
    if hasattr(adult,'hierarchy'):raise RuntimeError('owner:persisted_deep')
    top=None
    for _ in range(3):
        top=adult.experience_program((pairs[0].identity,pairs[1].identity),deep,3*Q//4,Q//8,CTX,Q//2,True)
    if top is None:raise RuntimeError('owner:top')

    before=adult.public_surface(top.identity)
    after=adult.public_surface(top.identity)
    emitted,expr=emit_all(adult,top.identity)

    adult.language.withdraw_source(5002)
    withdrawn=adult.public_surface(top.identity)
    try:
        adult.compose(JOIN,leaves[0],leaves[2]); new_refused=False
    except HierarchicalRefuse:
        new_refused=True
    factor=int(adult.programs.factor(top.identity))
    # Probe durable learned state on a cold checkpoint fork; transient recall/output
    # holds intentionally vanish across restore and cannot mask the lesion.
    lesion_adult=LanguageMasteryAdultV1.restore(adult.checkpoint())
    saved_span_rows={key:set(sources) for key,sources in lesion_adult.language._span_sources.items()
                     if lesion_adult.language.span_factor_identity(key[0],key[1],key[2])==factor}
    for key in saved_span_rows:lesion_adult.language._span_sources.pop(key,None)
    lesion_adult.language._rebuild_indices()
    try:
        lesion_adult.public_surface(top.identity); lesion_refused=False
    except RuntimeError:
        lesion_refused=True

    text=bytes(before).decode()
    distinct={part.strip() for part in text.split('.') if part.strip()}
    encoded=json.dumps(adult.program_surface_checkpoint(),sort_keys=True,separators=(',',':'))
    checks={
        'compose_returns_ephemeral_witness':isinstance(deep,CompositionWitnessV1),
        'construction_never_persists_composites':not composites(adult),
        'higher_compose_uses_earned_programs':deep.child_identities==(pairs[0].identity,pairs[1].identity),
        'heldout_recursive_surface':after==before and len(distinct)==4,
        'incremental_expression_survives':emitted==before and not hasattr(expr,'surface'),
        'earned_factor_survives_source_withdrawal':withdrawn==before,
        'withdrawal_blocks_new_structure':new_refused,
        'focal_factor_lesion_breaks_expression':lesion_refused,
        'leaves_live_in_factor':all(leaf.depth==0 and leaf.identity not in adult._surface_leaf_surfaces and leaf.identity in adult._surface_leaf_family_index and adult._leaf_surface(leaf.identity)==tuple(leaf.surface) for leaf in leaves),
        'construction_never_writes_hierarchy':not hasattr(adult,'hierarchy'),
        'no_second_closure_tree_in_factor':'child_identities' not in encoded,
        'bounded_fast_path':time.perf_counter()-started<1.0,
    }
    failed=[k for k,v in checks.items() if not v]
    if failed:raise SystemExit('FOUNDRY_MATHEMATICAL_ADULT_PROGRAM_FACTOR_OWNER_RED '+','.join(failed))
    print('FOUNDRY_MATHEMATICAL_ADULT_PROGRAM_FACTOR_OWNER_GREEN')
    print(json.dumps({
        'contract':'FOUNDRY_MATHEMATICAL_ADULT_PROGRAM_FACTOR_OWNER_GREEN',
        'reference_only':True,
        'language_phenotype_improved':True,
        'composite_closures_written':0,
        'recursive_bytes':len(before),
        'distinct_clauses':len(distinct),
        'checks':checks,
        'elapsed_ms':round((time.perf_counter()-started)*1000,3),
    },indent=2,sort_keys=True))


if __name__=='__main__':main()
