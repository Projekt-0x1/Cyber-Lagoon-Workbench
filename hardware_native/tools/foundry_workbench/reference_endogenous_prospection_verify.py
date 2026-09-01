#!/usr/bin/env python3
"""Falsifier for receipt-backed endogenous prospective Recipe condensation.

The continuing organism may compose one-shot lived transitions into a shadow
Recipe and unfold it as an ephemeral Network.  That shadow may nominate a real
intervention, but rehearsal alone cannot mint evidence or causal credit.
"""
from __future__ import annotations

import copy
import json
import time

from reference_organism_v2 import *
from reference_cognition_v1 import (
    MAX_PLAN_DEPTH,PROSPECTIVE_TTL_TICKS,EXPERT_MIN_COMPLETIONS,
    EXPERT_PROBATION_PASSES,
)
from reference_hierarchical_composition_v1 import TransientSequencePlanV1
from reference_incremental_expression_v1 import (
    IncrementalTransientExpressionV1,IncrementalTransientSequenceExpressionV1,
)
from reference_population_v1 import PopulationSpecV1

START,MID,GOAL,ALT_MID,DEEP_MID,WRONG=101,102,103,104,105,199
YOKED_START,YOKED_MID,YOKED_GOAL=201,202,203
FIRST,SECOND,DISTRACTOR,THIRD,FOURTH,FIFTH,SIXTH=7101,7102,7103,7104,7105,7106,7107
EDGE_SOURCE1,EDGE_SOURCE2,TEST_SOURCE=9001,9002,9003
EXPRESSION_CONTEXT,EXPRESSION_RELATION=0xE710,0xE711
PARTNER_A,PARTNER_B=0xE801,0xE802
BODY_MARKER=0xE901
BODY_A,BODY_B=(0xEA11,),(0xEA12,)
BODY_A_SOURCE,BODY_B_SOURCE=0xEA21,0xEA22
EXPERT_ROUTE_SOURCE,EXPERT_ALT_SOURCE,EXPERT_BODY_SOURCE=0xFB01,0xFB02,0xFB03


def stage(o,state,target,actions,source):
    o.contact(CONTACT_WORLD_STATE,(state,),source,True,True)
    o.contact(CONTACT_BODY_TARGET,(target,),5001,True,True)
    o.contact(CONTACT_AFFORDANCES,tuple(actions),5002,True,True)


def settle(o,action,source,next_state,effect=1,independent=True):
    return o.contact(CONTACT_MOTOR_CONSEQUENCE,
                     (action.ticket,int(effect),1,int(next_state)),
                     source,True,independent)


def live_edge(o,state,target,action,source,effect=1):
    stage(o,state,target,(action,),source)
    issued=o.tick()
    if not isinstance(issued,MotorActionV2) or issued.action_id!=action:
        raise AssertionError('resident action')
    settle(o,issued,source,target,effect)


def one_shot_organism(spec):
    o=ReferenceOrganismV2(spec)
    live_edge(o,START,MID,FIRST,EDGE_SOURCE1)
    live_edge(o,MID,GOAL,SECOND,EDGE_SOURCE2)
    return o


def reversed_one_shot_organism(spec):
    o=ReferenceOrganismV2(spec)
    live_edge(o,MID,GOAL,SECOND,EDGE_SOURCE2)
    live_edge(o,START,MID,FIRST,EDGE_SOURCE1)
    return o


def _partner(o,partner,source):
    o.contact(CONTACT_PARTNER_CONTEXT,(1,7,int(partner)),int(source),True,True)


def _surface(o,text,source):
    o.contact(CONTACT_SURFACE,tuple(text.encode()),int(source),True,True)


def teach_prospective_expression(o):
    """Author communication topology; learn every surface through ordinary contact."""
    ecologies=(
        (PARTNER_A,{
            FIRST:'inspect',MID:'middle',SECOND:'verify',GOAL:'goal',
            THIRD:'probe',ALT_MID:'alternate',FOURTH:'confirm',
            FIFTH:'consider',DEEP_MID:'deeper',SIXTH:'finish'},' then '),
        (PARTNER_B,{
            FIRST:'pruefen',MID:'mitte',SECOND:'testen',GOAL:'ziel',
            THIRD:'sondieren',ALT_MID:'alternative',FOURTH:'bestaetigen',
            FIFTH:'erwaegen',DEEP_MID:'tiefer',SIXTH:'abschliessen'},' danach '),
    )
    clauses=((FIRST,MID),(SECOND,GOAL),(THIRD,ALT_MID),(FOURTH,GOAL),
             (FIFTH,DEEP_MID),(SIXTH,GOAL))
    binary_routes=(((FIRST,MID),(SECOND,GOAL)),((THIRD,ALT_MID),(FOURTH,GOAL)))
    for ecology_index,(partner,words,joiner) in enumerate(ecologies):
        _partner(o,partner,partner)
        sources=(partner,partner+0x100+ecology_index)
        for source in sources:
            for atom,text in words.items():
                o.contact(CONTACT_SCENE,(7,0,1,atom),source,True,True)
                _surface(o,text,source)
            for action_id,state_id in clauses:
                o.contact(CONTACT_SCENE,(7,EXPRESSION_CONTEXT,2,action_id,state_id),source,True,True)
                _surface(o,words[action_id]+' '+words[state_id],source)
        # Mature clauses first; binary span contact then teaches a recurrent join law.
        for source in sources:
            for left_clause,right_clause in binary_routes:
                left=o.contact(CONTACT_SCENE,(7,EXPRESSION_CONTEXT,2,*left_clause),source,True,True)
                right=o.contact(CONTACT_SCENE,(7,EXPRESSION_CONTEXT,2,*right_clause),source,True,True)
                o.contact(CONTACT_SCENE_LINK,(left,right,EXPRESSION_RELATION),source,True,True)
                o.contact(CONTACT_DISCOURSE_SURFACE,tuple((
                    words[left_clause[0]]+' '+words[left_clause[1]]+joiner+
                    words[right_clause[0]]+' '+words[right_clause[1]]).encode()),
                    source,True,True)
    return ecologies


def emit_transient_expression(o,expression_plan):
    trajectory=(IncrementalTransientSequenceExpressionV1(o.language,expression_plan,leaves=o.utterances)
                if isinstance(expression_plan,TransientSequencePlanV1)
                else IncrementalTransientExpressionV1(o.language,expression_plan,leaves=o.utterances));out=[]
    while True:
        byte_plan=trajectory.emit()
        if byte_plan is None:break
        out.append(byte_plan.value)
        if not trajectory.reafference(byte_plan,byte_plan.value):
            raise AssertionError('prospective expression reafference')
    return bytes(out),trajectory


def learn_body_marker(o,action_id,entity,body_values,body_source,effect,scene_source):
    o.contact(CONTACT_BODY_STATE,tuple(body_values),int(body_source),True,True)
    _partner(o,PARTNER_A,scene_source+1)
    o.contact(CONTACT_SCENE,(7,EXPRESSION_CONTEXT,2,int(action_id),int(entity)),scene_source,True,True)
    action=o.tick()
    if not isinstance(action,ActionV2):raise AssertionError('somatic marker action')
    learned=o.contact(CONTACT_CONSEQUENCE,(action.ticket,int(effect)),PARTNER_A,True,True)
    if learned.get('somatic_updates',0)<=0:raise AssertionError('somatic marker update')
    return action


def main():
    started=time.perf_counter();checks={};spec=PopulationSpecV1(32768,2,4,42,8)
    untrained=ReferenceOrganismV2(spec)
    stage(untrained,START,GOAL,(FIRST,DISTRACTOR),TEST_SOURCE);untrained.tick()
    checks['no_training_cannot_generate_prospective_recipe']=(
        not untrained.cognition._prospective_recipes
        and untrained.last_prospective_recipe==0)

    o=one_shot_organism(spec)
    teach_prospective_expression(o)
    o.contact(CONTACT_BODY_STATE,(BODY_MARKER,),0xE902,True,True)
    checks['one_shot_contacts_are_not_authoritative_edges']=(
        len(o.cognition._evidence)==2 and not o.cognition.edges())
    checks['no_lived_end_to_end_episode']=(
        not any(key[0]==(START,) and key[2]==(GOAL,) for key in o.cognition._evidence))

    stage(o,START,GOAL,(FIRST,DISTRACTOR),TEST_SOURCE)
    checks['ordinary_counterfactual_prefers_untried_distractor']=(
        o._exploration_candidate()==DISTRACTOR)
    evidence_before=copy.deepcopy(o.cognition._evidence)
    credit_before=o.population.credit_events;revision_before=o.population.revision_events
    retained_before=len(o.population.occurrences)
    action=o.tick();recipe=next(iter(o.cognition._prospective_recipes.values()))
    retained_ids={row.identity for row in o.population.occurrences}
    checks['resident_recurrence_composes_novel_two_edge_recipe']=(
        recipe.actions==(FIRST,SECOND) and recipe.states==((START,),(MID,),(GOAL,))
        and recipe.sources==(EDGE_SOURCE1,EDGE_SOURCE2)
        and recipe.shadow_credit==1 and recipe.shadow_counter==0)
    checks['shadow_recipe_changes_real_intervention_nomination']=(
        isinstance(action,MotorActionV2) and action.action_id==FIRST
        and o.last_prospective_recipe==recipe.identity)
    checks['prospective_network_is_ephemeral']=(
        len(o.last_prospective_occurrences)==6
        and all(identity not in retained_ids for identity in o.last_prospective_occurrences)
        and len(o.population.occurrences)==retained_before+1)
    checks['shadow_work_mints_no_evidence_or_actual_credit']=(
        o.cognition._evidence==evidence_before
        and o.population.credit_events==credit_before
        and o.population.revision_events==revision_before)
    checks['shadow_credit_is_separate_and_provenance_bearing']=(
        recipe.shadow_credit==1 and recipe.sources==(EDGE_SOURCE1,EDGE_SOURCE2)
        and not o.cognition.edges())

    _partner(o,PARTNER_A,0xEA01)
    expression_a=o.current_prospective_expression_plan(EXPRESSION_CONTEXT,EXPRESSION_RELATION)
    opportunity_a,expression_plan_a=(expression_a if expression_a is not None else (None,None))
    emitted_a,trajectory_a=(emit_transient_expression(o,expression_plan_a)
                            if expression_plan_a is not None else (b'',None))
    _partner(o,PARTNER_B,0xEA02)
    expression_b=o.current_prospective_expression_plan(EXPRESSION_CONTEXT,EXPRESSION_RELATION)
    opportunity_b,expression_plan_b=(expression_b if expression_b is not None else (None,None))
    emitted_b,trajectory_b=(emit_transient_expression(o,expression_plan_b)
                            if expression_plan_b is not None else (b'',None))
    checks['prospective_recipe_supplies_language_content_without_world_laundering']=(
        opportunity_a is not None and opportunity_b is not None
        and opportunity_a.recipe_identity==opportunity_b.recipe_identity==recipe.identity
        and opportunity_a.actions==opportunity_b.actions==recipe.actions==(FIRST,SECOND)
        and opportunity_a.start==(START,) and opportunity_a.goal==(GOAL,)
        and opportunity_a.world_occurrence==opportunity_b.world_occurrence==o.world_state_occurrence
        and opportunity_a.body_occurrence==opportunity_b.body_occurrence==o.body_state_occurrence
        and opportunity_a.identity!=opportunity_b.identity
    )
    checks['same_prospective_content_recruits_partner_conditioned_surface_trajectories']=(
        emitted_a==b'inspect middle then verify goal'
        and emitted_b==b'pruefen mitte danach testen ziel'
        and emitted_a!=emitted_b
        and expression_plan_a.identity!=expression_plan_b.identity
        and not hasattr(o,'hierarchy')
        and trajectory_a.complete and trajectory_b.complete
    )
    cognition_state=json.dumps(o.cognition.checkpoint(),sort_keys=True,separators=(',',':')).lower()
    checks['prospective_checkpoint_contains_no_expression_surface']=(
        'inspect' not in cognition_state and 'verify' not in cognition_state
        and 'pruefen' not in cognition_state and 'testen' not in cognition_state
        and 'danach' not in cognition_state
    )

    def competing_frontier(preferred_effect):
        x=ReferenceOrganismV2(spec)
        live_edge(x,START,MID,FIRST,0xF101,preferred_effect)
        live_edge(x,MID,GOAL,SECOND,0xF102,preferred_effect)
        live_edge(x,START,ALT_MID,THIRD,0xF103,1)
        live_edge(x,ALT_MID,GOAL,FOURTH,0xF104,1)
        teach_prospective_expression(x)
        x.contact(CONTACT_BODY_STATE,(BODY_MARKER,),0xF105,True,True)
        stage(x,START,GOAL,(FIRST,THIRD),0xF106)
        _partner(x,PARTNER_A,0xF107)
        return x

    tied=competing_frontier(1)
    tied_frontier=tied.current_prospective_expression_frontier(
        EXPRESSION_CONTEXT,EXPRESSION_RELATION)
    checks['equal_consequence_keeps_multiple_prospective_expression_closures_alive']=(
        len(tied_frontier)==2
        and {tuple(row[0].actions) for row in tied_frontier}
            == {(FIRST,SECOND),(THIRD,FOURTH)}
        and len({row[0].score for row in tied_frontier})==1
        and tied.current_prospective_expression_plan(
            EXPRESSION_CONTEXT,EXPRESSION_RELATION) is None
        and not tied.cognition.edges()
    )

    biased=competing_frontier(2)
    biased_frontier=biased.current_prospective_expression_frontier(
        EXPRESSION_CONTEXT,EXPRESSION_RELATION)
    selected=biased.current_prospective_expression_plan(
        EXPRESSION_CONTEXT,EXPRESSION_RELATION)
    selected_opportunity,selected_expression=(selected if selected is not None else (None,None))
    selected_bytes,_selected_trajectory=(emit_transient_expression(biased,selected_expression)
        if selected_expression is not None else (b'',None))
    _partner(biased,PARTNER_B,0xF108)
    selected_b=biased.current_prospective_expression_plan(
        EXPRESSION_CONTEXT,EXPRESSION_RELATION)
    selected_opportunity_b,selected_expression_b=(selected_b if selected_b is not None else (None,None))
    selected_bytes_b,_selected_trajectory_b=(emit_transient_expression(biased,selected_expression_b)
        if selected_expression_b is not None else (b'',None))
    checks['independent_consequence_bias_selects_one_prospective_expression_closure']=(
        len(biased_frontier)==2 and biased_frontier[0][0].score>biased_frontier[1][0].score
        and selected_opportunity is not None
        and selected_opportunity.actions==(FIRST,SECOND)
        and selected_bytes==b'inspect middle then verify goal'
        and selected_opportunity_b is not None
        and selected_opportunity_b.actions==selected_opportunity.actions
        and selected_bytes_b==b'pruefen mitte danach testen ziel'
        and selected_bytes_b!=selected_bytes
        and not biased.cognition.edges()
    )

    deep=ReferenceOrganismV2(spec)
    live_edge(deep,START,MID,FIRST,0xF201,1)
    live_edge(deep,MID,GOAL,SECOND,0xF202,1)
    live_edge(deep,MID,DEEP_MID,FIFTH,0xF203,1)
    live_edge(deep,DEEP_MID,GOAL,SIXTH,0xF204,1)
    teach_prospective_expression(deep)
    deep.contact(CONTACT_BODY_STATE,(BODY_MARKER,),0xF205,True,True)
    stage(deep,START,GOAL,(FIRST,),0xF206);_partner(deep,PARTNER_A,0xF207)
    deep_frontier=deep.current_prospective_expression_frontier(
        EXPRESSION_CONTEXT,EXPRESSION_RELATION)
    deep_row=next((row for row in deep_frontier if len(row[0].actions)==3),None)
    deep_bytes,deep_trajectory=(emit_transient_expression(deep,deep_row[2])
        if deep_row is not None else (b'',None))
    checks['multi_depth_frontier_reuses_shared_prefix_and_binary_span_recipe']=(
        {tuple(row[0].actions) for row in deep_frontier}
            == {(FIRST,SECOND),(FIRST,FIFTH,SIXTH)}
        and deep_row is not None and isinstance(deep_row[2],TransientSequencePlanV1)
        and deep_bytes==b'inspect middle then consider deeper then finish goal'
        and deep_trajectory.complete
        and not hasattr(deep,'hierarchy')
        and deep.current_prospective_expression_plan(
            EXPRESSION_CONTEXT,EXPRESSION_RELATION)[0].actions==(FIRST,SECOND)
    )

    somatic=ReferenceOrganismV2(spec)
    live_edge(somatic,START,MID,FIRST,0xF301,1)
    live_edge(somatic,MID,GOAL,SECOND,0xF302,1)
    live_edge(somatic,START,ALT_MID,THIRD,0xF303,1)
    live_edge(somatic,ALT_MID,GOAL,FOURTH,0xF304,1)
    teach_prospective_expression(somatic)
    learn_body_marker(somatic,FIRST,MID,BODY_A,BODY_A_SOURCE,1,0xF310)
    learn_body_marker(somatic,THIRD,ALT_MID,BODY_B,BODY_B_SOURCE,1,0xF320)
    somatic_checkpoint=copy.deepcopy(somatic.checkpoint())
    def select_under_body(body_values,body_source,source):
        y=ReferenceOrganismV2.restore(copy.deepcopy(somatic_checkpoint))
        y.contact(CONTACT_BODY_STATE,tuple(body_values),int(body_source),True,True)
        stage(y,START,GOAL,(FIRST,THIRD),source);_partner(y,PARTNER_A,source+1)
        frontier=y.cognition.prospective_frontier((START,),(GOAL,),depth_slack=2)
        biases={tuple(plan.actions):y._prospective_somatic_bias(plan) for plan in frontier}
        selected=y.current_prospective_expression_plan(EXPRESSION_CONTEXT,EXPRESSION_RELATION)
        opportunity,expression=(selected if selected is not None else (None,None))
        raw,_trajectory=(emit_transient_expression(y,expression)
                         if expression is not None else (b'',None))
        return opportunity,raw,biases
    body_a_op,body_a_bytes,body_a_bias=select_under_body(BODY_A,BODY_A_SOURCE,0xF330)
    body_b_op,body_b_bytes,body_b_bias=select_under_body(BODY_B,BODY_B_SOURCE,0xF340)
    checks['learned_current_body_state_reorders_same_prospective_population']=(
        body_a_op is not None and body_b_op is not None
        and body_a_op.actions==(FIRST,SECOND)
        and body_b_op.actions==(THIRD,FOURTH)
        and body_a_bytes==b'inspect middle then verify goal'
        and body_b_bytes==b'probe alternate then confirm goal'
        and body_a_bias[(FIRST,SECOND)]>body_a_bias[(THIRD,FOURTH)]
        and body_b_bias[(THIRD,FOURTH)]>body_b_bias[(FIRST,SECOND)]
        and not somatic.cognition.edges()
    )
    checks['same_scene_body_history_selects_distinct_incremental_motors']=(
        checks['learned_current_body_state_reorders_same_prospective_population']
        and body_a_bytes!=body_b_bytes and not hasattr(somatic,'hierarchy')
    )
    checks['language_phenotype_improved']=checks['same_scene_body_history_selects_distinct_incremental_motors']
    checks['visible_discussion_improvement']=checks['language_phenotype_improved']

    def quantity_frontier(order):
        q=ReferenceOrganismV2(PopulationSpecV1(65536,2,4,42,8))
        for index in order:
            middle=0x11000+index;first_action=0x12000+index*2;second_action=first_action+1
            live_edge(q,START,middle,first_action,0x13000+index*2,1)
            live_edge(q,middle,GOAL,second_action,0x13001+index*2,1)
        rows=q.cognition.prospective_frontier((START,),(GOAL,),max_candidates=8,depth_slack=2)
        return q,tuple((row.recipe_identity,row.actions,row.states,row.score) for row in rows)
    quantity_a,quantity_rows_a=quantity_frontier(range(12))
    quantity_b,quantity_rows_b=quantity_frontier(reversed(range(12)))
    checks['prospective_population_is_bounded_and_contact_order_deterministic']=(
        len(quantity_rows_a)==len(quantity_rows_b)==8
        and quantity_rows_a==quantity_rows_b
        and not quantity_a.cognition.edges() and not quantity_b.cognition.edges()
        and len(quantity_a.cognition._evidence)==len(quantity_b.cognition._evidence)==24
    )

    expert_o=ReferenceOrganismV2(spec)
    live_edge(expert_o,START,MID,FIRST,EXPERT_ROUTE_SOURCE,2)
    live_edge(expert_o,MID,GOAL,SECOND,EXPERT_ROUTE_SOURCE,2)
    live_edge(expert_o,START,ALT_MID,THIRD,EXPERT_ALT_SOURCE,1)
    live_edge(expert_o,ALT_MID,GOAL,FOURTH,EXPERT_ALT_SOURCE,1)
    teach_prospective_expression(expert_o)
    learn_body_marker(expert_o,FIRST,MID,BODY_A,EXPERT_BODY_SOURCE,1,0xFC10)
    learn_body_marker(expert_o,THIRD,ALT_MID,BODY_B,BODY_B_SOURCE,1,0xFC20)
    def complete_expert_route():
        stage(expert_o,START,GOAL,(FIRST,SECOND,THIRD,FOURTH),EXPERT_ROUTE_SOURCE)
        expert_o.contact(CONTACT_BODY_STATE,BODY_A,EXPERT_BODY_SOURCE,True,True)
        _partner(expert_o,PARTNER_A,0xFC30)
        first=expert_o.tick()
        if not isinstance(first,MotorActionV2) or first.action_id!=FIRST or not first.prospective_snapshot:
            raise AssertionError('expert first prospective action')
        context=int(first.prospective_context_signature)
        settle(expert_o,first,EXPERT_ROUTE_SOURCE,MID,2,True)
        second=expert_o.tick()
        if (not isinstance(second,MotorActionV2) or second.action_id!=SECOND
                or second.prospective_context_signature!=context):
            raise AssertionError('expert continuation prospective action')
        learned=settle(expert_o,second,EXPERT_ROUTE_SOURCE,GOAL,2,True)
        return context,learned
    nomination_receipts=[];expert_context=0
    for _ in range(EXPERT_MIN_COMPLETIONS):
        expert_context,receipt=complete_expert_route();nomination_receipts.append(receipt)
    expert_key=((START,),(GOAL,),expert_context)
    expert_recipe=expert_o.cognition._prospective_experts.get(expert_key)
    checks['whole_route_independent_recurrence_nominates_dormant_expert']=(
        expert_recipe is not None and expert_recipe.active==0
        and expert_recipe.actions==(FIRST,SECOND)
        and expert_recipe.states==((START,),(MID,),(GOAL,))
        and expert_recipe.completion_count==EXPERT_MIN_COMPLETIONS
        and expert_recipe.completion_sources==(EXPERT_ROUTE_SOURCE,)
        and expert_recipe.frontier_revision==expert_o.cognition._frontier_revision
        and nomination_receipts[-1].get('prospective_expert_nomination')==expert_recipe.identity
    )
    stage(expert_o,START,GOAL,(FIRST,SECOND,THIRD,FOURTH),EXPERT_ROUTE_SOURCE)
    expert_o.contact(CONTACT_BODY_STATE,BODY_A,EXPERT_BODY_SOURCE,True,True);_partner(expert_o,PARTNER_A,0xFC40)
    probation_touches=[]
    for _ in range(EXPERT_PROBATION_PASSES):
        trial=expert_o.current_prospective_expression_plan(EXPRESSION_CONTEXT,EXPRESSION_RELATION)
        if trial is None:raise AssertionError('expert probation expression')
        probation_touches.append(expert_o.cognition.last_plan_touches)
    expert_recipe=expert_o.cognition._prospective_experts[expert_key]
    pre_fast_frontier=expert_o.cognition.prospective_frontier((START,),(GOAL,),depth_slack=2)
    lower_witness_ids=tuple(row.recipe_identity for row in pre_fast_frontier)
    fast=expert_o.current_prospective_expression_plan(EXPRESSION_CONTEXT,EXPRESSION_RELATION)
    fast_opportunity,fast_expression=(fast if fast is not None else (None,None))
    fast_bytes,fast_trajectory=(emit_transient_expression(expert_o,fast_expression)
        if fast_expression is not None else (b'',None))
    fast_touches=expert_o.cognition.last_plan_touches
    checks['shadow_equivalence_activates_cheaper_n_plus_one_expert']=(
        expert_recipe.active==1 and expert_recipe.probation_passes==EXPERT_PROBATION_PASSES
        and fast_opportunity is not None and fast_opportunity.recipe_identity==expert_recipe.identity
        and fast_opportunity.actions==(FIRST,SECOND)
        and fast_bytes==b'inspect middle then verify goal' and fast_trajectory.complete
        and fast_touches<min(probation_touches)
        and tuple(expert_recipe.witness_recipes)==lower_witness_ids
        and {tuple(row.actions) for row in pre_fast_frontier}=={(FIRST,SECOND),(THIRD,FOURTH)}
    )
    expert_checkpoint=copy.deepcopy(expert_o.checkpoint())
    expert_restored=ReferenceOrganismV2.restore(copy.deepcopy(expert_checkpoint))
    restored_recipe=expert_restored.cognition._prospective_experts.get(expert_key)
    checks['checkpoint_preserves_authenticated_expert_and_lower_witness']=(
        restored_recipe is not None and restored_recipe.active==1
        and restored_recipe.identity==expert_recipe.identity
        and expert_restored.cognition.checkpoint()==expert_o.cognition.checkpoint()
    )
    corrupt_expert=copy.deepcopy(expert_checkpoint)
    corrupt_expert['cognition']['prospective_experts'][0]['identity']+=1
    try:
        ReferenceOrganismV2.restore(corrupt_expert);expert_corruption_refused=False
    except ValueError:
        expert_corruption_refused=True
    checks['expert_checkpoint_corruption_refused']=expert_corruption_refused
    body_remat=ReferenceOrganismV2.restore(copy.deepcopy(expert_checkpoint))
    body_remat.contact(CONTACT_BODY_STATE,BODY_B,BODY_B_SOURCE,True,True)
    body_before_uses=body_remat.cognition._prospective_experts[expert_key].uses
    body_selected=body_remat.current_prospective_expression_plan(EXPRESSION_CONTEXT,EXPRESSION_RELATION)
    body_opportunity,body_expression=(body_selected if body_selected is not None else (None,None))
    body_bytes,_body_trajectory=(emit_transient_expression(body_remat,body_expression)
        if body_expression is not None else (b'',None))
    checks['body_domain_change_rematerializes_without_erasing_expert']=(
        body_opportunity is not None and body_opportunity.actions==(THIRD,FOURTH)
        and body_bytes==b'probe alternate then confirm goal'
        and body_remat.cognition._prospective_experts[expert_key].active==1
        and body_remat.cognition._prospective_experts[expert_key].uses==body_before_uses
        and body_remat.cognition.last_plan_touches>fast_touches
    )
    source_cut=ReferenceOrganismV2.restore(copy.deepcopy(expert_checkpoint))
    cut_recipe=source_cut.cognition._prospective_experts[expert_key];before_deopt=cut_recipe.deoptimizations
    source_cut.contact(CONTACT_WITHDRAW_SOURCE,(EXPERT_ROUTE_SOURCE,),0xFC50,True,True)
    # Withdrawal invalidates both learned route support and the current world contact.
    # Re-contact the same current state from a neutral source so lower surviving
    # cognition can rematerialize without reviving the withdrawn route evidence.
    stage(source_cut,START,GOAL,(FIRST,SECOND,THIRD,FOURTH),0xFC51)
    source_cut.contact(CONTACT_BODY_STATE,BODY_A,EXPERT_BODY_SOURCE,True,True);_partner(source_cut,PARTNER_A,0xFC52)
    source_selected=source_cut.current_prospective_expression_plan(EXPRESSION_CONTEXT,EXPRESSION_RELATION)
    source_opportunity,source_expression=(source_selected if source_selected is not None else (None,None))
    source_bytes,_source_trajectory=(emit_transient_expression(source_cut,source_expression)
        if source_expression is not None else (b'',None))
    checks['owning_source_withdrawal_deoptimizes_and_rematerializes_lower_frontier']=(
        cut_recipe.active==0 and cut_recipe.deoptimizations==before_deopt+1
        and source_opportunity is not None and source_opportunity.actions==(THIRD,FOURTH)
        and source_bytes==b'probe alternate then confirm goal'
        and source_cut.cognition.last_plan_touches>fast_touches
    )
    checks['expert_optimization_mints_no_world_authority']=(
        not hasattr(expert_recipe,'effect') and not hasattr(expert_recipe,'reward')
        and expert_recipe.completion_count==EXPERT_MIN_COMPLETIONS
    )

    shadow_credit=recipe.shadow_credit;evidence_count=len(o.cognition._evidence)
    rehearsal=ReferenceOrganismV2.restore(copy.deepcopy(o.checkpoint()))
    rehearsal_credit=rehearsal.population.credit_events;rehearsal.tick()
    rehearsed_recipe=next(iter(rehearsal.cognition._prospective_recipes.values()))
    checks['repeated_endogenous_use_cannot_self_confirm_shadow']=(
        rehearsed_recipe.rehearsals==recipe.rehearsals
        and rehearsed_recipe.shadow_credit==shadow_credit
        and len(rehearsal.cognition._evidence)==evidence_count
        and rehearsal.population.credit_events==rehearsal_credit)

    try:
        o.cognition._condense_prospective((START,),(GOAL,),object())
        direct_host_authoring_refused=False
    except ValueError:
        direct_host_authoring_refused=True
    checks['direct_host_candidate_authoring_refused']=direct_host_authoring_refused

    permuted=reversed_one_shot_organism(spec)
    stage(permuted,START,GOAL,(FIRST,DISTRACTOR),TEST_SOURCE);permuted.tick()
    permuted_recipe=next(iter(permuted.cognition._prospective_recipes.values()))
    checks['contact_order_permutation_preserves_recipe']=(
        permuted_recipe.identity==recipe.identity
        and permuted_recipe.actions==recipe.actions
        and len(permuted.cognition._evidence)==len(evidence_before))

    other_context=ReferenceOrganismV2(spec)
    live_edge(other_context,YOKED_START,YOKED_MID,FIRST,EDGE_SOURCE1)
    live_edge(other_context,YOKED_MID,YOKED_GOAL,SECOND,EDGE_SOURCE2)
    stage(other_context,START,GOAL,(FIRST,DISTRACTOR),TEST_SOURCE);other_context.tick()
    checks['yoked_other_context_does_not_supply_route']=(
        not other_context.cognition._prospective_recipes
        and other_context.last_prospective_recipe==0
        and len(other_context.cognition._evidence)==len(evidence_before))

    # Work-only load control: remote decoy transitions are observer-injected and
    # never used as behavioral or causal evidence for the path. This bounds wall
    # time; it does not claim a local index or constant global-scan cost.
    loaded=one_shot_organism(spec);remote_decoy_edges=4096
    for index in range(remote_decoy_edges):
        state=(10000+index,);action_id=20000+index;next_state=(30000+index,)
        loaded.cognition.observe(state,action_id,next_state,1,40000+2*index,True)
        loaded.cognition.observe(state,action_id,next_state,1,40001+2*index,True)
    stage(loaded,START,GOAL,(FIRST,DISTRACTOR),TEST_SOURCE)
    loaded_started=time.perf_counter();loaded_action=loaded.tick()
    loaded_elapsed_ms=(time.perf_counter()-loaded_started)*1000
    loaded_recipe=next(iter(loaded.cognition._prospective_recipes.values()))
    checks['remote_evidence_preserves_result_with_bounded_wall_time']=(
        loaded_action.action_id==FIRST and loaded_recipe.identity==recipe.identity
        and loaded.last_prospective_touches==2 and loaded_elapsed_ms<250)

    checkpoint=copy.deepcopy(o.checkpoint());restored=ReferenceOrganismV2.restore(copy.deepcopy(checkpoint))
    restored_plan=restored.cognition.plan((START,),(GOAL,))
    checks['checkpoint_preserves_compact_recipe_not_network']=(
        restored_plan.recipe_identity==recipe.identity
        and restored_plan.actions==(FIRST,SECOND)
        and not hasattr(restored,'prospective_network'))
    corrupt=copy.deepcopy(checkpoint)
    corrupt['cognition']['prospective_recipes'][0]['identity']+=1
    try:
        ReferenceOrganismV2.restore(corrupt);corrupt_refused=False
    except ValueError:
        corrupt_refused=True
    checks['prospective_checkpoint_corruption_refused']=corrupt_refused
    corrupt_expiry=copy.deepcopy(checkpoint)
    corrupt_expiry['cognition']['prospective_recipes'][0]['expires_tick']+=1
    try:
        ReferenceOrganismV2.restore(corrupt_expiry);expiry_corrupt_refused=False
    except ValueError:
        expiry_corrupt_refused=True
    checks['prospective_checkpoint_expiry_corruption_refused']=expiry_corrupt_refused

    reinforced=ReferenceOrganismV2.restore(copy.deepcopy(checkpoint))
    pending=next(action for action in reinforced.motor_actions if not action.settled)
    settle(reinforced,pending,TEST_SOURCE,MID,1,True)
    reinforced_checkpoint=copy.deepcopy(reinforced.checkpoint())
    reinforced_restored=ReferenceOrganismV2.restore(reinforced_checkpoint)
    checks['new_actual_constituent_support_invalidates_stale_shadow']=(
        not reinforced.cognition._prospective_recipes
        and not reinforced_restored.cognition._prospective_recipes
        and TEST_SOURCE in reinforced.cognition._evidence[((START,),FIRST,(MID,),1)])
    stage(reinforced,START,GOAL,(FIRST,DISTRACTOR),9902);reinforced.tick()
    rebuilt=next(iter(reinforced.cognition._prospective_recipes.values()))
    checks['revised_shadow_requires_fresh_endogenous_recondensation']=(
        rebuilt.identity!=recipe.identity and TEST_SOURCE in rebuilt.sources
        and rebuilt.shadow_credit==1 and len(reinforced.cognition.edges())==1
        and reinforced.cognition.plan((START,),(GOAL,)).recipe_identity==rebuilt.identity)

    withdrawn=ReferenceOrganismV2.restore(copy.deepcopy(checkpoint))
    withdrawn.contact(CONTACT_WITHDRAW_SOURCE,(EDGE_SOURCE1,),9901,True,True)
    checks['source_withdrawal_defeats_shadow_recipe']=(
        withdrawn.cognition.plan((START,),(GOAL,)).status==0)

    expiry=ReferenceOrganismV2.restore(copy.deepcopy(checkpoint))
    expiry_evidence=copy.deepcopy(expiry.cognition._evidence)
    expiry_credit=expiry.population.credit_events
    checks['prospective_valid_through_declared_deadline']=(
        expiry.cognition.plan((START,),(GOAL,),current_tick=recipe.expires_tick).recipe_identity==recipe.identity)
    checks['prospective_expires_without_minting_counterevidence']=(
        expiry.cognition.plan((START,),(GOAL,),current_tick=recipe.expires_tick+1).status==0
        and expiry.cognition._evidence==expiry_evidence
        and expiry.population.credit_events==expiry_credit)

    lesion=ReferenceOrganismV2.restore(copy.deepcopy(checkpoint));lesion_evidence=copy.deepcopy(lesion.cognition._evidence)
    lesion.cognition._prospective_recipes.clear()
    checks['focal_shadow_lesion_preserves_lived_fragments']=(
        lesion.cognition._evidence==lesion_evidence
        and lesion.cognition.plan((START,),(GOAL,)).status==0)

    two_recipe=one_shot_organism(spec)
    live_edge(two_recipe,YOKED_START,YOKED_MID,FIRST,9011)
    live_edge(two_recipe,YOKED_MID,YOKED_GOAL,SECOND,9012)
    stage(two_recipe,YOKED_START,YOKED_GOAL,(FIRST,),9013)
    remote_action=two_recipe.tick();settle(two_recipe,remote_action,9013,YOKED_MID,1,False)
    stage(two_recipe,START,GOAL,(FIRST,DISTRACTOR),9014);two_recipe.tick()
    two_checkpoint=copy.deepcopy(two_recipe.checkpoint())
    target_key=((START,),(GOAL,));remote_key=((YOKED_START,),(YOKED_GOAL,))
    sham=ReferenceOrganismV2.restore(copy.deepcopy(two_checkpoint));sham.cognition._prospective_recipes.pop(remote_key)
    focal=ReferenceOrganismV2.restore(copy.deepcopy(two_checkpoint));focal.cognition._prospective_recipes.pop(target_key)
    checks['equal_work_remote_recipe_lesion_preserves_target_nomination']=(
        len(sham.cognition._prospective_recipes)==len(focal.cognition._prospective_recipes)==1
        and sham.cognition.plan((START,),(GOAL,)).recipe_identity!=0
        and focal.cognition.plan((START,),(GOAL,)).status==0)

    learned=settle(o,action,TEST_SOURCE,WRONG,-1,True)
    checks['host_staged_contradiction_counters_shadow']=(
        learned.get('credit',0)>0 and recipe.shadow_counter==1
        and recipe.counter_sources==(TEST_SOURCE,))
    stage(o,START,GOAL,(FIRST,DISTRACTOR),9004);after=o.tick()
    checks['host_staged_return_overrides_shadow_route']=(
        isinstance(after,MotorActionV2) and after.action_id==DISTRACTOR
        and o.last_prospective_recipe==0)

    yoked=one_shot_organism(spec);yoked_credit=yoked.population.credit_events
    yoked_evidence=copy.deepcopy(yoked.cognition._evidence)
    for index,state in enumerate((START,MID,GOAL)):
        yoked.population.activate((PROSPECTIVE_STATE_TAG,index,state),retain=False)
    checks['host_scripted_activation_is_not_endogenous_recipe']=(
        not yoked.cognition._prospective_recipes
        and yoked.cognition._evidence==yoked_evidence
        and yoked.population.credit_events==yoked_credit)

    authoritative=ReferenceOrganismV2(spec)
    for source in (9101,9102):live_edge(authoritative,START,MID,FIRST,source)
    for source in (9201,9202):live_edge(authoritative,MID,GOAL,SECOND,source)
    authoritative.cognition._prospective_recipes.clear()
    checks['repeated_actual_contact_earns_authoritative_plan']=(
        len(authoritative.cognition.edges())==2
        and authoritative.cognition.plan((START,),(GOAL,)).actions==(FIRST,SECOND))

    checks['no_named_dream_or_language_module']=(
        not any(hasattr(o,name) for name in ('think','imagine','dream','sleep','answer','prompt')))
    checks['bounded_sparse_awake_work']=(
        o.last_prospective_touches<=2
        and len(o.last_prospective_occurrences)<=2*MAX_PLAN_DEPTH+2)

    state=json.dumps(checkpoint,sort_keys=True,separators=(',',':'))
    contract_ok=all(checks.values())
    result={
        'schema':'agi.reference-endogenous-prospection.v1',
        'contract_status':'CONTRACT' if contract_ok else 'RED','checks':checks,
        'shadow_recipe':{'identity':recipe.identity,'actions':list(recipe.actions),
                         'sources':list(recipe.sources),'reconstruction_support':recipe.shadow_credit,
                         'world_credit':0,
                         'counter':recipe.shadow_counter,'rehearsals':recipe.rehearsals},
        'prospective_expression':{
            'partner_a':emitted_a.decode(), 'partner_b':emitted_b.decode(),
            'same_recipe_identity':opportunity_a.recipe_identity if opportunity_a else 0,
            'opportunity_a':opportunity_a.identity if opportunity_a else 0,
            'opportunity_b':opportunity_b.identity if opportunity_b else 0,
            'plan_a':expression_plan_a.identity if expression_plan_a else 0,
            'plan_b':expression_plan_b.identity if expression_plan_b else 0,
            'surface_bytes_in_prospective_checkpoint':False,
            'equal_frontier':[{'actions':list(row[0].actions),'score':row[0].score}
                              for row in tied_frontier],
            'biased_frontier':[{'actions':list(row[0].actions),'score':row[0].score}
                               for row in biased_frontier],
            'biased_partner_a':selected_bytes.decode(),
            'biased_partner_b':selected_bytes_b.decode(),
            'multi_depth_frontier':[list(row[0].actions) for row in deep_frontier],
            'recurrent_three_step_surface':deep_bytes.decode(),
            'body_a_selected':[] if body_a_op is None else list(body_a_op.actions),
            'body_b_selected':[] if body_b_op is None else list(body_b_op.actions),
            'body_a_bias':{str(k):v for k,v in body_a_bias.items()},
            'body_b_bias':{str(k):v for k,v in body_b_bias.items()},
            'expert_condensation':{
                'identity':expert_recipe.identity,'actions':list(expert_recipe.actions),
                'completion_count':expert_recipe.completion_count,
                'completion_sources':list(expert_recipe.completion_sources),
                'probation_passes':expert_recipe.probation_passes,'active':expert_recipe.active,
                'uses':expert_recipe.uses,'deoptimizations':expert_recipe.deoptimizations,
                'frontier_witness_count':len(expert_recipe.witness_recipes),
                'probation_touches':probation_touches,'fast_touches':fast_touches,
                'fast_surface':fast_bytes.decode(),'source_withdraw_surface':source_bytes.decode(),
            },
        },
        'quantity':{'resident_sites':o.population.spec.site_count,
                    'materialized_sites':o.population.materialized_site_count(),
                    'retained_occurrences':len(o.population.occurrences),
                    'ephemeral_network_occurrences':6,
                    'prospective_touches':2,
                    'prospective_population_contacted':12,
                    'prospective_population_live_bound':len(quantity_rows_a),
                    'prospective_ttl_ticks':PROSPECTIVE_TTL_TICKS,
                    'remote_decoy_edges':remote_decoy_edges,
                    'loaded_plan_ms':round(loaded_elapsed_ms,3),
                    'checkpoint_bytes':len(state)},
        'schedule':'awake_interleaved','runtime_llm':False,'adult_attached':False,
        'reference_only':True,'production_ir':'ResidentRecipeIrProgram.vcurrent',
        'translation_status':'UNDEFINED',
        'candidate_authority':'CAPABILITY_GUARDED_REFERENCE/NOT_SECURITY_BOUNDARY',
        'checkpoint_authentication':'STRUCTURAL_RECOMPUTE_ONLY/AUTHENTICATION_RED',
        'consequence_ingress':'HOST_STAGED_REFERENCE/PHYSICAL_AUTH_RED',
        'actual_credit_shared_with_shadow':False,'graph_flip':False,
        'physical_direct_parity':'NOT_RUN/RED',
        'contract_scope':'TICK_TRIGGERED_TWO_FRAGMENT_SHADOW_PATH_COMPOSITION',
        'claim':'DETERMINISTIC_REFERENCE_TICK_TRIGGERED_TWO_FRAGMENT_SHADOW_PATH_COMPOSITION',
        'remaining_red':['PHYSICAL_CONSEQUENCE_AUTHENTICATION','CHECKPOINT_AUTHENTICATION',
                         'HOST_STAGED_WORLD_TARGET_AFFORDANCE_INGRESS',
                         'AUTONOMOUS_GOAL_FORMATION_NOT_RUN',
                         'SCHEDULE_COMPETITION',
                         'PRODUCTION_TRANSLATION'],
        'elapsed_ms':round((time.perf_counter()-started)*1000,3),
    }
    print('FOUNDRY_ENDOGENOUS_PROSPECTION '+('GREEN' if contract_ok else 'RED')+
          ' reference_only=1 adult=0 graph_flip=0')
    print(json.dumps(result,indent=2,sort_keys=True))
    raise SystemExit(0 if contract_ok else 1)


if __name__=='__main__':main()
