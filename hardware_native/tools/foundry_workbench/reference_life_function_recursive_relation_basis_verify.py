#!/usr/bin/env python3
"""Canonical-life N+1: silent exact closures become reusable relation matter and speech."""
from __future__ import annotations
import copy,json,time
from reference_life_function_curriculum_v1 import *
from reference_recursive_relation_basis_v1 import RecursiveRelationBasisV1


def endpoint(basis):
    rows=[basis.relations[r] for r in basis.active]
    lefts={r.left_space for r in rows};rights={r.right_space for r in rows}
    starts=lefts-rights;ends=rights-lefts
    return (next(iter(starts)),next(iter(ends))) if len(starts)==len(ends)==1 else (0,0)


def main():
    t=time.perf_counter();checks={};program=canonical_species_program_v2();curriculum=canonical_life_function_curriculum_v2()
    cold=ReferenceLifeFunctionRuntimeV2(program).run(curriculum.prefix_at_mark('relation_primitives'))
    start,goal=endpoint(cold.adult.relation_basis);raw=cold.adult.relation_basis.resolve(start,goal);cold_cp=copy.deepcopy(cold.checkpoint())
    hot=ReferenceLifeFunctionRuntimeV2.restore(program,copy.deepcopy(cold_cp));hot.run(curriculum.prefix_at_mark('self_derived_basis'))
    best=hot.adult.relation_basis.best_derived();fast=hot.adult.relation_basis.resolve(start,goal);spaces=hot.adult.relation_basis.expand_spaces(best)
    checks['canonical_species_has_no_constructor_depth_law']=(
        'generic_constructor_metaplasticity' not in {law.law for law in program.laws}
        and getattr(hot.adult.language_adult,'_constructor_metaplasticity',None) is None)
    checks['same_life_silent_recurrence_builds_recursive_relation_basis']=(
        raw is not None and fast is not None and len(spaces)==9
        and hot.adult.relation_basis.generation(best)>=3
        and len(hot.adult.relation_basis.relations)>len(cold.adult.relation_basis.relations))
    checks['self_derived_basis_reduces_later_exact_work']=(
        raw.boundary_q16==fast.boundary_q16 and raw.relation_touches==8 and fast.relation_touches==1)

    expression=hot.adult.relation_basis_expression();pre=copy.deepcopy(hot.adult.checkpoint());first=expression.emit()
    wrong=expression.reafference(first,int(first[0])^1)
    checks['wrong_public_reafference_cannot_create_self_derived_action']=(
        not wrong and expression.receipt is None and not hot.adult.pending_relation_actions
        and hot.adult.checkpoint()==pre)
    assert expression.reafference(first,first[0])
    out=[first[0]]
    while expression.receipt is None:
        step=expression.emit();out.append(step[0]);assert expression.reafference(step,step[0])
    spoken=bytes(out);receipt=expression.receipt
    contacts=tuple(bytes(e.payload) for e in curriculum.events[:curriculum.mark_cursor('self_derived_public')]
                   if e.lane in {'surface','discourse_surface','utterance','authenticated_utterance'}
                   and e.payload and all(isinstance(x,int) and 0<=x<=255 for x in e.payload))
    checks['exact_completion_creates_ancestry_bearing_public_action']=(
        receipt is not None and receipt.relation_identity==best
        and receipt.identity in hot.adult.pending_relation_actions
        and hot.adult.relation_public_count==1)
    checks['visible_n_plus_one_is_nine_sentence_never_taught_composition']=(
        spoken.count(b'.')==9 and len(spoken)>250 and spoken not in contacts
        and all(bytes(hot.adult.language_adult._leaf_surface(x)) in spoken for x in spaces))

    top_cut=ReferenceLifeFunctionRuntimeV2.restore(program,copy.deepcopy(hot.checkpoint()));top_cut.adult.relation_basis.remove_derived(best)
    slower=top_cut.adult.relation_basis.resolve(start,goal)
    checks['focal_highest_abstraction_lesion_restores_lower_basis']=(
        slower is not None and slower.boundary_q16==fast.boundary_q16 and 1<slower.relation_touches<raw.relation_touches)
    premise_cut=ReferenceLifeFunctionRuntimeV2.restore(program,copy.deepcopy(hot.checkpoint()))
    primitive=next(r for r in premise_cut.adult.relation_basis.relations.values() if r.direct_evidence and start in (r.left_space,r.right_space))
    premise_cut.adult.relation_basis.withdraw_evidence(primitive.direct_evidence[0])
    checks['primitive_source_withdrawal_shatters_dependent_self_derived_claim']=(
        premise_cut.adult.relation_basis.resolve(start,goal) is None
        and premise_cut.adult.project_relation_basis(best) is None)

    replay=ReferenceLifeFunctionRuntimeV2.restore(program,copy.deepcopy(hot.checkpoint()));rspoken,_=replay.adult.externalize_relation_basis()
    blob=json.dumps(hot.checkpoint(),sort_keys=True).lower()
    checks['checkpoint_replays_self_derived_public_structure_without_generated_paragraph']=(
        rspoken==spoken and spoken.decode() not in blob and 'best_paths' not in blob and 'active' not in hot.adult.relation_basis.checkpoint())
    checks['no_new_language_evidence_between_primitive_and_self_derived_marks']=(
        cold.adult.language_adult.language.checkpoint()==hot.adult.language_adult.language.checkpoint())
    checks['bounded_fast_path']=time.perf_counter()-t<0.5
    failed=[k for k,v in checks.items() if not v]
    result={'schema':'cyber-lagoon.life-function-recursive-relation-basis.v1','contract':'FOUNDRY_LIFE_FUNCTION_RECURSIVE_RELATION_BASIS_'+('GREEN' if not failed else 'RED'),'pass':not failed,'reference_only':True,'language_phenotype_improved':checks['visible_n_plus_one_is_nine_sentence_never_taught_composition'],'visible_language_gain':spoken.decode(errors='replace'),'species_root':program.root(),'curriculum_root':curriculum.root(),'primitive_touches':raw.relation_touches if raw else -1,'retained_touches':fast.relation_touches if fast else -1,'generation':hot.adult.relation_basis.generation(best),'self_derived_relation_count':len(hot.adult.relation_basis.relations)-len(cold.adult.relation_basis.relations),'visible':spoken.decode(errors='replace'),'checks':checks,'failed':failed,'remaining_red':['ENDOGENOUS_AUTHORITY_CALIBRATION','WORLD_TAIL_PARITY','DIRECT_RECURSIVE_RELATION_BASIS_PARITY'],'elapsed_ms':round((time.perf_counter()-t)*1000,3)}
    print(result['contract']);print(json.dumps(result,indent=2,sort_keys=True));return 0 if not failed else 1
if __name__=='__main__':raise SystemExit(main())
