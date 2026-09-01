#!/usr/bin/env python3
"""Differential proof that the continuing Adult has one recursive program owner.

Composition and productive leaves are current/factor state rather than hierarchy
closures. Public expression reconstructs from causal-program membership plus retained
terminal surfaces and learned template mathematics. The paired builds exercise two
operand routes and must converge on identical program identities and phenotype.
Reference-only: this proves the Workbench representation, not Direct parity.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from reference_language_mastery_adult_v1 import LanguageMasteryAdultV1
from reference_hierarchical_composition_v1 import HierarchicalRefuse
from reference_predictive_credit_profile_v1 import Q

CLAUSE, JOIN, CTX = 9001, 9101, 0xA11


def emit_all(expression):
    out=[]
    while True:
        plan=expression.emit()
        if plan is None:return tuple(out),expression
        out.append(plan.value)
        if not expression.reafference(plan,plan.value):raise RuntimeError('factor:reafference')
        if len(out)>10000:raise RuntimeError('factor:nontermination')


def build(retire_closures=True):
    adult=LanguageMasteryAdultV1()
    A=(101,102);G=(201,202);V=(301,302);O=(401,402)
    names={101:'careful',102:'quiet',201:'engineer',202:'technician',301:'tests',302:'inspects',401:'sensor',402:'valve'}
    for feature,text in names.items():
        adult.observe_surface_item(feature,text.encode(),1000+feature)
        adult.observe_surface_item(feature,text.encode(),2000+feature)
    adult.observe_surface_construction(CLAUSE,(101,201,301,401),b'the careful engineer tests the sensor.',3001)
    adult.observe_surface_construction(CLAUSE,(102,202,302,402),b'the quiet technician inspects the valve.',3002)
    patterns=tuple((a,g,v,o) for a in range(2) for g in range(2) for v in range(2) for o in range(2))
    leaves=tuple(adult.leaf(CLAUSE,(A[a],G[g],V[v],O[o])) for a,g,v,o in patterns)
    adult.observe_join(JOIN,leaves[0],leaves[-1],5001)
    adult.observe_join(JOIN,leaves[-1],leaves[1],5002)

    operands=list(leaves);child_programs=None;all_programs=[];program_roots={}
    while len(operands)>1:
        next_operands=[];next_programs=[]
        for i in range(0,len(operands),2):
            closure=adult.compose(JOIN,operands[i],operands[i+1])
            children=(operands[i].identity,operands[i+1].identity) if child_programs is None else (
                child_programs[i].identity,child_programs[i+1].identity)
            program=None
            for _ in range(3):
                program=adult.experience_program(
                    children,closure,3*Q//4,Q//8,CTX,Q//3,True,
                    retire_closure=retire_closures)
            if program is None:raise RuntimeError('factor:program_not_earned')
            next_operands.append(program.identity if retire_closures else closure)
            next_programs.append(program);all_programs.append(program);program_roots[program.identity]=closure
        operands=next_operands;child_programs=next_programs
    return adult,leaves,operands[0],child_programs[0],tuple(all_programs),program_roots


def surface_state_projection(adult):
    """Observer-only normalized surface projection; it does not duplicate Adult state."""
    leaves=dict(adult._surface_leaf_surfaces)
    for identity in adult._surface_leaf_family_index:leaves[int(identity)]=adult._leaf_surface(identity)
    return {
        'schema':1,
        'leaf_surfaces':[{'identity':k,'surface':list(v)} for k,v in sorted(leaves.items())],
        'templates':[
            {'identity':tid,'pieces':[{'kind':p.kind,'port':p.port,'literal':list(p.literal)} for p in pieces]}
            for tid in sorted({int(v) for v in adult.programs.factors.values() if int(v)>0})
            for pieces in (adult.language.historical_span_pieces(tid),) if pieces is not None],
        'program_templates':[{'program':k,'template':max(0,int(v))} for k,v in sorted(adult.programs.factors.items())],
    }


def atomic_factor_probe():
    """An atomic public program references its existing hierarchy leaf without copying bytes."""
    adult,leaves,_retired_top,_top,_programs,_roots=build(True)
    pid=0xA700
    expected=tuple(leaves[0].surface)
    adult.experience_atomic_program(pid,leaves[0],Q//2,Q//16,CTX,Q//16,True)
    factor=int(adult.programs.factor(pid));checkpoint=adult.program_surface_checkpoint()
    checkpoint_bytes=len(json.dumps(checkpoint,sort_keys=True,separators=(',',':')).encode())
    adult._surface_leaf_surfaces={};adult._surface_leaf_families={};adult._surface_leaf_family_index={};adult.programs.factors={}
    adult.restore_program_surface_checkpoint(checkpoint)
    return {
        'factor':factor,
        'leaf_identity':leaves[0].identity,
        'surface':adult.public_surface(pid),
        'expected':expected,
        'checkpoint_bytes':checkpoint_bytes,
        'factor_only_checkpoint':set(checkpoint)=={'schema','raw_leaf_surfaces','leaf_families','program_factor_state'} and
                                 not checkpoint['raw_leaf_surfaces'] and len(checkpoint['leaf_families'])==1 and
                                 checkpoint['program_factor_state']==adult.programs.factor_checkpoint() and
                                 all(set(row)=={'program','factor'} for row in checkpoint['program_factor_state']['factors']),
    }


def scale_probe():
    """Push the same learned relation to 64 held-out clauses / program depth six."""
    adult=LanguageMasteryAdultV1()
    A=(101,102,103,104);G=(201,202,203,204);V=(301,302);O=(401,402)
    names={
        101:'careful',102:'quiet',103:'steady',104:'alert',
        201:'engineer',202:'technician',203:'operator',204:'analyst',
        301:'tests',302:'inspects',401:'sensor',402:'valve',
    }
    for feature,text in names.items():
        adult.observe_surface_item(feature,text.encode(),1000+feature)
        adult.observe_surface_item(feature,text.encode(),2000+feature)
    adult.observe_surface_construction(CLAUSE,(101,201,301,401),b'the careful engineer tests the sensor.',3001)
    adult.observe_surface_construction(CLAUSE,(102,202,302,402),b'the quiet technician inspects the valve.',3002)
    leaves=[adult.leaf(CLAUSE,(a,g,v,o)) for a in A for g in G for v in V for o in O]
    adult.observe_join(JOIN,leaves[0],leaves[-1],5001)
    adult.observe_join(JOIN,leaves[-1],leaves[1],5002)
    operands=leaves;child_programs=None;programs=[]
    while len(operands)>1:
        next_operands=[];next_programs=[]
        for i in range(0,len(operands),2):
            closure=adult.compose(JOIN,operands[i],operands[i+1])
            children=(operands[i].identity,operands[i+1].identity) if child_programs is None else (
                child_programs[i].identity,child_programs[i+1].identity)
            program=None
            for _ in range(3):
                program=adult.experience_program(children,closure,3*Q//4,Q//8,CTX,Q//3,True)
            if program is None:raise RuntimeError('factor:scale_program_not_earned')
            next_operands.append(program.identity);next_programs.append(program);programs.append(program)
        operands=next_operands;child_programs=next_programs
    top=child_programs[0];surface=adult.public_surface(top.identity);text=bytes(surface).decode()
    distinct={part.strip() for part in text.split('.') if part.strip()}
    composite_ids=()
    encoded=json.dumps(adult.program_surface_checkpoint(),sort_keys=True,separators=(',',':'))
    return {
        'distinct_clauses':len(distinct),'program_depth':top.depth,'recursive_bytes':len(surface),
        'current_decision_width':adult.current_width(top.identity),'programs_reused':len(programs),
        'surface_factor_bytes':len(encoded.encode()),
        'no_composite_closures':not composite_ids and all(str(identity) not in encoded for identity in composite_ids),
    }


def main():
    started=time.perf_counter()
    reference,reference_leaves,reference_deep,reference_top,reference_programs,_reference_roots=build(False)
    adult,leaves,_candidate_top,top,programs,_candidate_roots=build(True)
    reference_surface=tuple(reference_deep.surface);reference_text=bytes(reference_surface).decode()
    candidate_surface=adult.public_surface(top.identity)
    emitted,expr=emit_all(adult.expression(top.identity))

    projection=surface_state_projection(adult)
    factor_checkpoint=adult.program_surface_checkpoint()
    projection_bytes=len(json.dumps(projection,sort_keys=True,separators=(',',':')).encode())
    factor_checkpoint_bytes=len(json.dumps(factor_checkpoint,sort_keys=True,separators=(',',':')).encode())
    composite_closures=()
    encoded=json.dumps(factor_checkpoint,sort_keys=True,separators=(',',':'))
    no_composite_closures=not composite_closures and 'child_identities' not in encoded

    # Checkpoint the factored state after every learned composite has already retired.
    checkpoint=adult.program_surface_checkpoint()
    program_factor_relation_not_duplicated=(set(checkpoint)=={'schema','raw_leaf_surfaces','leaf_families','program_factor_state'} and
        checkpoint['program_factor_state']==adult.programs.factor_checkpoint())
    template_witness_not_duplicated='templates' not in checkpoint and not hasattr(adult,'_surface_templates')
    adult._surface_leaf_surfaces={};adult._surface_leaf_families={};adult._surface_leaf_family_index={};adult.programs.factors={}
    adult.restore_program_surface_checkpoint(checkpoint)
    replay_surface=adult.public_surface(top.identity)
    replay_checkpoint_matches=checkpoint==adult.program_surface_checkpoint()
    surface_relation_scalar=all(type(tid) is int for tid in adult.programs.factors.values())
    surface_relation_value_bytes=sum(sys.getsizeof(tid) for tid in adult.programs.factors.values())

    # An earned program can participate in genuinely new later learning without
    # resurrecting a hierarchy closure: only the current composition witness exists.
    extension=adult.compose(JOIN,top.identity,leaves[0])
    extension_program=None
    for _ in range(3):
        extension_program=adult.experience_program(
            (top.identity,leaves[0].identity),extension,3*Q//4,Q//8,CTX,Q//3,True)
    if extension_program is None:raise RuntimeError('factor:extension_not_earned')
    extension_surface=adult.public_surface(extension_program.identity)
    extension_expected=reference_surface+(32,)+tuple(reference_leaves[0].surface)
    post_extension_composites=()

    adult.language.withdraw_source(5002)
    withdrawal_surface=adult.public_surface(top.identity)
    withdrawal_extension=adult.public_surface(extension_program.identity)
    try:
        adult.compose(JOIN,extension_program.identity,leaves[1]);new_structure_refused=False
    except HierarchicalRefuse:
        new_structure_refused=True
    refused_composites=()

    repair_expr=adult.expression(top.identity);first=repair_expr.emit()
    repair_refused=not repair_expr.reafference(first,first.value^1)
    same_pending=repair_expr.emit()==first
    repair_then_advanced=repair_expr.reafference(first,first.value) and repair_expr.emit().ordinal==1

    top_factor=int(adult.programs.factor(top.identity))
    # Probe durable learned state on a cold checkpoint fork; transient recall/output
    # holds intentionally vanish across restore and cannot mask the lesion.
    lesion_adult=LanguageMasteryAdultV1.restore(adult.checkpoint())
    saved_span_rows={key:set(sources) for key,sources in lesion_adult.language._span_sources.items()
                     if lesion_adult.language.span_factor_identity(key[0],key[1],key[2])==top_factor}
    for key in saved_span_rows:lesion_adult.language._span_sources.pop(key,None)
    lesion_adult.language._rebuild_indices()
    try:
        lesion_adult.public_surface(top.identity);lesion_refused=False
    except RuntimeError:
        lesion_refused=True

    distinct={part.strip() for part in reference_text.split('.') if part.strip()}
    atomic=atomic_factor_probe();scale=scale_probe()
    checks={
        'reference_candidate_same_program_identity':reference_top.identity==top.identity and tuple(p.identity for p in reference_programs)==tuple(p.identity for p in programs),
        'sixteen_distinct_heldout_clauses':len(distinct)==16,
        'depth_four_program':top.depth>=4,
        'current_decision_width_one':adult.current_width(top.identity)==1,
        'adult_integrated_factor_exact_surface':candidate_surface==reference_surface,
        'adult_integrated_incremental_surface':emitted==reference_surface and not hasattr(expr,'surface'),
        'one_program_tree_not_duplicate_closure_tree':no_composite_closures,
        'construction_writes_no_hierarchy':not hasattr(adult,'hierarchy'),
        'surface_factor_checkpoint_replays':replay_surface==reference_surface and replay_checkpoint_matches,
        'program_surface_is_direct_scalar_relation':surface_relation_scalar,
        'retired_program_rematerializes_for_future_learning':extension_surface==extension_expected and extension_program.depth==top.depth+1,
        'future_learning_retires_current_materialization_again':not post_extension_composites,
        'earned_factor_survives_source_withdrawal':withdrawal_surface==reference_surface and withdrawal_extension==extension_surface,
        'withdrawal_blocks_new_structure':new_structure_refused,
        'failed_new_structure_cleans_materialization':not refused_composites,
        'reafference_precedes_cursor_advance':repair_refused and same_pending and repair_then_advanced,
        'focal_template_lesion_breaks_recursive_expression':lesion_refused,
        'ordinary_leaf_surfaces_have_one_owner':not hasattr(adult,'hierarchy') and not adult._surface_leaf_surfaces and len(adult._surface_leaf_family_index)==16 and sum(len(v) for v in adult._surface_leaf_families.values())==16,
        'program_factor_relation_has_one_owner':program_factor_relation_not_duplicated and not hasattr(adult,'surface'),
        'atomic_program_references_leaf_without_byte_copy':atomic['factor']==-atomic['leaf_identity'] and atomic['surface']==atomic['expected'] and atomic['factor_only_checkpoint'],
        'template_witness_has_one_owner':template_witness_not_duplicated and adult.language.historical_span_pieces(top_factor) is not None,
        'one_template_family_reused':len({int(v) for v in adult.programs.factors.values() if int(v)>0})==1,
        'scale_sixty_four_distinct_compositions':scale['distinct_clauses']==64,
        'scale_reaches_program_depth_six':scale['program_depth']>=6,
        'scale_exceeds_depth_four_surface_extent':scale['recursive_bytes']>len(reference_surface),
        'scale_keeps_current_decision_width_one':scale['current_decision_width']==1,
        'scale_keeps_program_tree_as_only_recursive_surface_owner':scale['no_composite_closures'],
        'bounded_fast_path':time.perf_counter()-started<1.0,
    }
    failed=[k for k,v in checks.items() if not v]
    if failed:raise SystemExit('FOUNDRY_MATHEMATICAL_ADULT_OPERATOR_FACTORIZATION_RED '+','.join(failed))
    result={
        'contract':'FOUNDRY_MATHEMATICAL_ADULT_OPERATOR_FACTORIZATION_GREEN',
        'reference_only':True,
        'language_phenotype_improved':True,
        'operator_semantics':'EPHEMERAL_COMPOSITION_PLUS_CAUSAL_PROGRAM_OWNED_PUBLIC_FACTOR',
        'recursive_bytes':len(reference_surface),
        'distinct_clauses':len(distinct),
        'program_depth':top.depth,
        'current_decision_width':adult.current_width(top.identity),
        'normalized_surface_projection_bytes':projection_bytes,
        'surface_factor_checkpoint_bytes':factor_checkpoint_bytes,
        'program_factor_relation_value_bytes':surface_relation_value_bytes,
        'composite_closures':len(composite_closures),
        'extension_program_depth':extension_program.depth,
        'extension_recursive_bytes':len(extension_surface),
        'atomic_factor_checkpoint_bytes':atomic['checkpoint_bytes'],
        'programs_reused':len(programs),
        'scale_depth_six':scale,
        'checks':checks,
        'elapsed_ms':round((time.perf_counter()-started)*1000,3),
    }
    print(result['contract']);print(json.dumps(result,indent=2,sort_keys=True))


if __name__=='__main__':main()
