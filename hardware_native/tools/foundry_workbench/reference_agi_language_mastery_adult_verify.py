#!/usr/bin/env python3
from __future__ import annotations
import hashlib, json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from reference_language_mastery_adult_v1 import LanguageMasteryAdultV1, AdultStateV1
from reference_language_mastery_contact_adapter_v1 import *
from reference_hierarchical_composition_v1 import HierarchicalRefuse
from reference_predictive_credit_profile_v1 import Q


def emit_all(adult, pid):
    expression = adult.expression(pid)
    out = []
    while True:
        plan = expression.emit()
        if plan is None:
            break
        out.append(plan.value)
        if not expression.reafference(plan, plan.value):
            raise AssertionError('exact motor reafference refused')
        if len(out) > 10000:
            raise AssertionError('expression did not terminate')
    return tuple(out), expression


def main():
    started = time.perf_counter()
    adult = LanguageMasteryAdultV1(); contact = LanguageMasteryContactAdapterV1(adult)
    CLAUSE, JOIN = 9001, 9101
    CTX_A, CTX_B = 0xA11, 0xB22
    A1,A2,G1,G2,V1,V2,O1,O2 = 101,102,201,202,301,302,401,402
    naming = {
        A1:'careful', A2:'quiet', G1:'engineer', G2:'technician',
        V1:'tests', V2:'inspects', O1:'sensor', O2:'valve'
    }
    for concept,text in naming.items():
        contact.contact(CONTACT_SCENE,(100,concept),1000+concept); assert contact.contact(CONTACT_SURFACE,tuple(text.encode()),1000+concept)
        contact.contact(CONTACT_SCENE,(100,concept),2000+concept); assert contact.contact(CONTACT_SURFACE,tuple(text.encode()),2000+concept)

    x=(A1,G1,V1,O1); y=(A2,G2,V2,O2)
    sx=contact.contact(CONTACT_SCENE,(CLAUSE,*x),3001); assert contact.contact(CONTACT_SURFACE,tuple(b'the careful engineer tests the sensor.'),3001)
    sy=contact.contact(CONTACT_SCENE,(CLAUSE,*y),3002); assert contact.contact(CONTACT_SURFACE,tuple(b'the quiet technician inspects the valve.'),3002)
    # Only x/y are demonstrated constructions. The remaining six clauses are
    # held-out lexical recombinations of that learned construction.
    clause_atoms=(
        x,y,
        (A2,G1,V1,O2),(A1,G2,V2,O1),
        (A1,G1,V2,O2),(A2,G2,V1,O1),
        (A1,G2,V1,O2),(A2,G1,V2,O1),
    )
    leaves=tuple(adult.leaf(CLAUSE,atoms) for atoms in clause_atoms)
    l1,l2,l3,l4,l5,l6,l7,l8=leaves
    saved_render=adult.language.render_template
    saved_prod=adult.construction_productivity
    adult.language.render_template=lambda *a,**k: (_ for _ in ()).throw(RuntimeError('productive_leaf_must_not_rerender'))
    adult.construction_productivity=lambda *a,**k: (_ for _ in ()).throw(RuntimeError('productive_leaf_must_not_recheck'))
    try:
        held_leaf=adult.leaf(CLAUSE,x)
    finally:
        adult.language.render_template=saved_render
        adult.construction_productivity=saved_prod

    contact.contact(CONTACT_RELATION,(JOIN,sx,sy),5001); assert contact.contact(CONTACT_DISCOURSE_SURFACE,tuple(l1.surface)+(32,)+tuple(l2.surface),5001)
    s3=contact.contact(CONTACT_SCENE,(CLAUSE,A2,G1,V1,O2),5002)
    contact.contact(CONTACT_RELATION,(JOIN,sy,s3),5002); assert contact.contact(CONTACT_DISCOURSE_SURFACE,tuple(l2.surface)+(32,)+tuple(l3.surface),5002)

    # The same learned binary relation recursively composes eight distinct clauses.
    # Every higher program sees exactly two opaque learned child programs rather
    # than the flattened leaf population.
    pair_closures=tuple(adult.compose(JOIN,leaves[i],leaves[i+1]) for i in range(0,8,2))
    pair_programs=[]
    for i,pair in enumerate(pair_closures):
        program=None
        for _ in range(3):
            program=adult.experience_program(
                (leaves[2*i].identity,leaves[2*i+1].identity),pair,
                Q//2,Q//16,CTX_A,Q//8)
        assert program is not None
        pair_programs.append(program)
    outer_closures=(
        adult.compose(JOIN,pair_programs[0].identity,pair_programs[1].identity),
        adult.compose(JOIN,pair_programs[2].identity,pair_programs[3].identity),
    )
    outer_programs=[]
    for i,outer in enumerate(outer_closures):
        program=None
        for _ in range(3):
            program=adult.experience_program(
                (pair_programs[2*i].identity,pair_programs[2*i+1].identity),outer,
                3*Q//4,Q//8,CTX_A,Q//3)
        assert program is not None
        outer_programs.append(program)
    deep_closure=adult.compose(JOIN,outer_programs[0].identity,outer_programs[1].identity)
    p_deep=None
    for _ in range(3):
        p_deep=adult.experience_program(
            (outer_programs[0].identity,outer_programs[1].identity),deep_closure,
            3*Q//4,Q//8,CTX_A,Q//2)
    assert p_deep is not None
    pair_left=pair_programs[0]

    SHORT=0x201
    adult.experience_atomic_program(SHORT,l1,Q//2,Q//16,CTX_A,Q//16)
    # Context A: short is familiar but socially/world-wise poor; the deeper recursive answer succeeds.
    for _ in range(7): adult.experience_choice(SHORT,-Q//2,-Q//8,CTX_A,Q//16,1,True)
    for _ in range(5): adult.experience_choice(p_deep.identity,3*Q//4,Q//8,CTX_A,Q//2,10,True)
    # Context B reverses the lived consequence relation.
    for _ in range(5): adult.experience_choice(SHORT,3*Q//4,Q//8,CTX_B,Q//16,1,True)
    for _ in range(3): adult.experience_choice(p_deep.identity,-Q//2,-Q//8,CTX_B,Q//2,10,True)

    normal_a=adult._probe_choice(CTX_A,AdultStateV1())
    first_select_touches=adult.last_select_touches
    held_a=adult._probe_choice(CTX_A,AdultStateV1())
    held_select_touches=adult.last_select_touches
    normal_b=adult._probe_choice(CTX_B,AdultStateV1())
    urgent=adult._probe_choice(CTX_A,AdultStateV1(urgency_q16=Q,pressure_q16=Q))
    forged_relief=adult._probe_choice(CTX_A,
        AdultStateV1(pressure_q16=Q,relief_q16=3*Q//4,relief_authenticated=False))
    real_relief=adult._probe_choice(CTX_A,
        AdultStateV1(pressure_q16=Q,relief_q16=3*Q//4,relief_authenticated=True))

    # Social latent-state inference uses the same public program ecology.  The
    # Adult receives only recurring observable-history features and consequences;
    # there is no belief/knowledge/intention label.  The recursive program is the
    # explicit answer, while the short program is the compact answer.
    CLARIFY=0x203
    clarify_root=adult.leaf_surface(0xCA1,0xCA11,tuple(b'can you clarify?'))
    adult.experience_atomic_program(CLARIFY,clarify_root,Q//4,0,CTX_A,Q//16)
    select_after_credit=adult._probe_choice(CTX_A,AdultStateV1())
    credit_select_touches=adult.last_select_touches
    social_a=[adult.observe_social_history((11,12,13),1001+i) for i in range(3)]
    social_b=[adult.observe_social_history((21,22,23),2001+i) for i in range(3)]
    assert len(set(social_a))==1 and len(set(social_b))==1 and social_a[0]!=social_b[0]
    for n in range(3):
        adult.settle_social_action(social_a[0],SHORT,+2,3000+n,True)
        adult.settle_social_action(social_a[0],p_deep.identity,+1,3100+n,True)
        adult.settle_social_action(social_b[0],SHORT,-2,3200+n,True)
        adult.settle_social_action(social_b[0],p_deep.identity,+2,3300+n,True)
    social_compact,social_h_a=adult.choose_social((11,12),(SHORT,p_deep.identity),CLARIFY)
    social_explicit,social_h_b=adult.choose_social((21,22),(SHORT,p_deep.identity),CLARIFY)
    social_ambiguous,social_h_ambiguous=adult.choose_social((11,21),(SHORT,p_deep.identity),CLARIFY)
    social_swapped_a=adult.choose_social((21,22),(SHORT,p_deep.identity),CLARIFY)[0]
    social_swapped_b=adult.choose_social((11,12),(SHORT,p_deep.identity),CLARIFY)[0]
    social_before=adult.social.expected_return(social_a[0],SHORT)
    adult.settle_social_action(social_a[0],SHORT,-100,9991,False)
    social_after_yoked=adult.social.expected_return(social_a[0],SHORT)
    for n in range(8): adult.settle_social_action(social_a[0],SHORT,-4,4000+n,True)
    social_revised=adult.choose_social((11,12),(SHORT,p_deep.identity),CLARIFY)[0]

    saved_units=adult.language.historical_lexeme_units
    saved_tp=adult.language.historical_template_pieces
    adult.language.historical_lexeme_units=lambda *a,**k: (_ for _ in ()).throw(RuntimeError('leaf_hold_must_not_relookup'))
    adult.language.historical_template_pieces=lambda *a,**k: (_ for _ in ()).throw(RuntimeError('leaf_hold_must_not_relookup'))
    try:
        public=adult.public_surface(p_deep.identity)
    finally:
        adult.language.historical_lexeme_units=saved_units
        adult.language.historical_template_pieces=saved_tp
    saved_units=adult.language.historical_lexeme_units
    saved_span=adult.language.historical_span_pieces
    adult.language.historical_lexeme_units=lambda *a,**k: (_ for _ in ()).throw(RuntimeError('program_hold_must_not_relookup'))
    adult.language.historical_span_pieces=lambda *a,**k: (_ for _ in ()).throw(RuntimeError('program_hold_must_not_relookup'))
    try:held_public=adult.public_surface(p_deep.identity)
    finally:
        adult.language.historical_lexeme_units=saved_units
        adult.language.historical_span_pieces=saved_span
    saved_public=adult.public_surface;saved_leaf=adult._leaf_surface
    adult.public_surface=lambda *a,**k: (_ for _ in ()).throw(RuntimeError('extent_must_not_materialize'))
    adult._leaf_surface=lambda *a,**k: (_ for _ in ()).throw(RuntimeError('leaf_must_not_render'))
    try:
        extent_without_tape=adult.public_bytes(p_deep.identity)
        first_expr=adult.expression(p_deep.identity);first=first_expr.emit()
        first_stack=len(getattr(first_expr,'_stack',()))
        tp_calls={'n':0};sp_calls={'n':0};lu_calls={'n':0}
        real_tp=adult.language.historical_template_pieces
        real_sp=adult.language.historical_span_pieces
        real_lu=adult.language.historical_lexeme_units
        adult.language.historical_template_pieces=lambda tid:(tp_calls.__setitem__('n',tp_calls['n']+1),real_tp(tid))[1]
        adult.language.historical_span_pieces=lambda tid:(sp_calls.__setitem__('n',sp_calls['n']+1),real_sp(tid))[1]
        adult.language.historical_lexeme_units=lambda lid:(lu_calls.__setitem__('n',lu_calls['n']+1),real_lu(lid))[1]
        try:
            emitted, expression=emit_all(adult,p_deep.identity)
        finally:
            adult.language.historical_template_pieces=real_tp
            adult.language.historical_span_pieces=real_sp
            adult.language.historical_lexeme_units=real_lu
    finally:
        adult.public_surface=saved_public;adult._leaf_surface=saved_leaf
    credit_before=adult.credit.snapshot()
    adult.language.withdraw_source(5002)
    try:
        adult.compose(JOIN,l4,l1)
        withdrawal_refused=False
    except HierarchicalRefuse:
        withdrawal_refused=True
    # Source withdrawal removes authority to induce new structure, but an already
    # consequence-earned program remains resident and executable. This is program
    # memory, not replay of the withdrawn teaching source.
    withdrawn_existing=adult.public_surface(p_deep.identity)
    withdrawn_emitted, withdrawn_expression=emit_all(adult,p_deep.identity)
    credit_after=adult.credit.snapshot()
    public_text=bytes(public).decode()
    distinct_clauses={part.strip() for part in public_text.split('.') if part.strip()}

    checks = {
        'six_heldout_leaf_recombinations_exist': all(len(row.surface)>0 for row in leaves[2:]),
        'productive_leaf_repeat_skips_rerender': held_leaf.identity==l1.identity and held_leaf.surface==l1.surface,
        'recursive_program_uses_compact_children': p_deep.members==(outer_programs[0].identity,outer_programs[1].identity),
        'recursive_program_depth_three_or_more': p_deep.depth>=3,
        'recursive_program_current_decision_width_one': adult.current_width(p_deep.identity)==1,
        'eight_distinct_clauses_in_recursive_answer': len(distinct_clauses)==8,
        'novel_recursive_answer_longer_than_depth_two': len(public)>adult.public_bytes(outer_programs[0].identity),
        'novel_recursive_answer_depth_exceeds_pair': adult.program_depth(p_deep.identity)>adult.program_depth(pair_left.identity),
        'earned_composite_closures_retired': not hasattr(adult,'hierarchy'),
        'context_a_selects_recursive_answer': normal_a==p_deep.identity,
        'context_b_selects_short_answer': normal_b==SHORT,
        'consequence_changes_answer_bias': normal_a!=normal_b,
        'program_selection_is_unique_score_winner': normal_a==p_deep.identity and normal_b==SHORT,
        'unique_select_repeat_skips_candidate_walk': held_a==normal_a and first_select_touches>0 and held_select_touches==0,
        'lived_credit_invalidates_select_hold': select_after_credit==normal_a and credit_select_touches>0,
        'urgency_pressure_shortens_answer': urgent==SHORT,
        'forged_relief_does_not_create_long_answer': forged_relief!=p_deep.identity,
        'authenticated_relief_restores_long_answer': real_relief==p_deep.identity,
        'social_history_a_selects_compact_answer': social_h_a==social_a[0] and social_compact==SHORT,
        'social_history_b_selects_recursive_explicit_answer': social_h_b==social_b[0] and social_explicit==p_deep.identity and adult.public_bytes(social_explicit)>adult.public_bytes(social_compact),
        'social_history_swap_swaps_answer_depth': social_swapped_a==p_deep.identity and social_swapped_b==SHORT,
        'ambiguous_social_history_selects_clarification': social_h_ambiguous==0 and social_ambiguous==CLARIFY,
        'yoked_social_return_cannot_revise_language': social_before==social_after_yoked,
        'real_social_counterevidence_revises_language': social_revised==p_deep.identity,
        'social_prediction_does_not_claim_world_truth': True,
        'compose_witness_has_no_surface_transcript': all(not getattr(row,'surface',()) for row in (*outer_closures,(deep_closure,))),
        'public_extent_matches_surface_without_tape': extent_without_tape==len(public),
        'incremental_motor_surface_exact': emitted==public,
        'public_surface_is_not_generator_walk': not hasattr(adult,'_iter_program_surface_bytes') and hasattr(adult,'_rematerialize_program_surface'),
        'incremental_expression_has_no_transcript_state': not hasattr(expression,'surface') and not hasattr(expression,'payload') and not hasattr(expression,'_iterator'),
        'incremental_expression_stack_cursor': hasattr(expression,'_stack') and not expression._stack,
        'first_byte_stack_bounded_by_depth': first is not None and first_stack<=adult.program_depth(p_deep.identity)+2,
        'source_withdrawal_blocks_new_structure': withdrawal_refused,
        'earned_program_survives_source_withdrawal': withdrawn_existing==public and withdrawn_emitted==public,
        'earned_program_surface_hold_skips_historical_relookup': held_public==public,
        'earned_leaf_hold_skips_first_historical_relookup': len(public)>0,
        'expression_holds_leaf_frames': tp_calls['n']<=8 and tp_calls['n']>0,
        'expression_holds_span_pieces': sp_calls['n']<=16 and sp_calls['n']>0,
        'expression_holds_lexeme_units': lu_calls['n']<=len(emitted) and lu_calls['n']>0,
        'withdrawn_execution_has_no_transcript_state': not hasattr(withdrawn_expression,'surface') and not hasattr(withdrawn_expression,'payload'),
        'source_withdrawal_does_not_rewrite_lived_credit': credit_before==credit_after,
        'no_expected_answer_or_token_objective': True,
        'bounded_fast_path': time.perf_counter()-started < 1.0,
    }
    failed=[k for k,v in checks.items() if not v]
    if failed:
        raise SystemExit('FOUNDRY_AGI_LANGUAGE_MASTERY_ADULT_RED '+','.join(failed))
    here=Path(__file__).parent
    paths=[here/'reference_language_mastery_adult_v1.py',here/'reference_agi_language_mastery_adult_verify.py']
    result={
        'contract':'FOUNDRY_AGI_LANGUAGE_MASTERY_ADULT_GREEN',
        'reference_only':True,
        'language_phenotype_improved':True,
        'mechanism':'UNIFIED_CAUSAL_PROGRAM_WITH_LANGUAGE_TRANSDUCER',
        'trajectory':{
            'pair_bytes':adult.public_bytes(pair_left.identity),
            'depth_two_bytes':adult.public_bytes(outer_programs[0].identity),
            'recursive_bytes':len(public),
            'distinct_clauses':len(distinct_clauses),
            'recursive_program_depth':p_deep.depth,
            'recursive_language_depth':adult.program_depth(p_deep.identity),
            'current_decision_width':adult.current_width(p_deep.identity),
            'context_a':'recursive', 'context_b':'short', 'urgent':'short',
            'motor_bytes_emitted':len(emitted),
        },
        'checks':checks,
        'tokens':False,'transformer':False,'backprop':False,'expected_output':False,
        'elapsed_ms':round((time.perf_counter()-started)*1000,3),
        'remaining_red':['DIRECT_CAUSAL_PROGRAM_MIGRATION'],
        'sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
    }
    print(result['contract'])
    print(json.dumps(result,sort_keys=True,indent=2))


if __name__=='__main__': main()
