#!/usr/bin/env python3
"""Resident cognition scheduling extracted from the oversized organism shell."""
from __future__ import annotations

from reference_cognition_v1 import PlanV1

def _resident_retrieval_context(o):
    """Bounded current participation cue; no transcript or surface bytes."""
    if not o.partner_present or int(o.partner_source)<=0:return ()
    partner=int(o.partner_source)
    closure=int(o.last_shared_closure_by_partner.get(partner,0))
    context=(partner,closure) if closure>0 else (partner,)
    # A just-heard surface participates only through residently reconstructed,
    # learned structure. Older assertions do not become a hidden text buffer.
    structures={tuple(int(atom) for atom in row.binding_atoms)
                for row in o.source_assertions
              if row.active and row.source not in o.withdrawn_sources
              and int(row.language_binding)>0
              and row.binding_atoms
              and int(row.lexical_tick)==int(o.tick_count)-1}
    if len(structures)>1:return ()
    return context+(next(iter(structures)) if structures else ())


def cognitive_tick(o,resident_authority,prospective_state_tag,prospective_edge_tag,
                   prospective_network_tag):
    # Actual contradictory returns create only a resident reconciliation priority in
    # TransitionEcologyV1.  Let that uncertainty buy bounded private work on the
    # ordinary cognitive clock instead of requiring a host/test to call a special
    # reconsideration API.  Repeated/multiple discrepancies increase effort, while
    # _reconcile_local remains prospective-only: it cannot add evidence or credit.
    priorities=tuple(max(0,int(value)) for value in o.cognition._reconcile_priority.values())
    if priorities:
        budget=min(4,max(1,sum(priorities)))
        o._reconcile_low_pressure(budget)
    if o.world_state is None or not o.body_target:return None
    if o.cognition.satisfies(o.world_state,o.body_target):o.information_need=();return None
    retrieval_context=_resident_retrieval_context(o)
    recalled=o.cognition.reactivate_intention(
        o.world_state,o.body_target,resident_authority,o.tick_count,retrieval_context)
    recalled_snapshot=();recalled_context_signature=0
    if recalled.status:
        plan=recalled
        # Cue overlap may retrieve a route for consideration, but only exact
        # learned state can authorize its motor transition. Learned speech can
        # expose the still-hypothetical remainder under a richer cue.
        if recalled.status==1 and not recalled.cue_exact:
            expression=o._emit_resident_prospective_expression(recalled)
            if expression is not None:return expression
            plan=PlanV1(0,(),(tuple(o.world_state),),0,0,())
        elif recalled.status==1:
            intention=o.cognition._prospective_intentions.get(recalled.intention_identity)
            if intention is None:raise ValueError('organism:prospective_intention_missing')
            recalled_snapshot=o._prospective_snapshot_from_intention(intention)
            recalled_context_signature=int(intention.context_signature)
    else:
        plan=o.cognition.plan(o.world_state,o.body_target,current_tick=o.tick_count)
    if plan.status==0:
        prospective=o.cognition._condense_prospective(
            o.world_state,o.body_target,resident_authority,current_tick=o.tick_count)
        if prospective.recipe_identity:plan=prospective
        elif o.partner_present:
            selected=o._resident_selected_prospective_plan()
            if (selected is not None and selected.actions
                    and int(selected.actions[0]) not in o.affordances):
                prospective_expression=o._emit_resident_prospective_expression(selected)
                if prospective_expression is not None:return prospective_expression
    o.last_prospective_recipe=0;o.last_prospective_touches=0;o.last_prospective_occurrences=()
    prospective_context_signature=(
        (recalled_context_signature or o._prospective_expert_context_signature())
        if plan.status==1 and plan.recipe_identity else 0)
    if plan.status==1 and plan.recipe_identity:
        if prospective_context_signature<=0:raise ValueError('organism:prospective_context')
        transient=[]
        for index,state in enumerate(plan.states):
            transient.append(o.population.activate((prospective_state_tag,index,*state),retain=False).identity)
            if index<len(plan.actions):
                transient.append(o.population.activate(
                    (prospective_edge_tag,index,plan.actions[index],*state,*plan.states[index+1]),
                    retain=False).identity)
        transient.append(o.population.activate(
            (prospective_network_tag,plan.recipe_identity,len(plan.actions),plan.shadow_credit,
             prospective_context_signature),retain=False).identity)
        o.last_prospective_recipe=int(plan.recipe_identity)
        o.last_prospective_touches=int(o.cognition.last_plan_touches)
        o.last_prospective_occurrences=tuple(transient)
    if plan.status==1 and plan.actions:
        action=plan.actions[0]
        if not o.affordances or action in o.affordances:
            if plan.recipe_identity and not plan.intention_identity:
                o.cognition._retain_prospective_intention(
                    plan,resident_authority,o.tick_count,retrieval_context,
                    prospective_context_signature)
            recipe_identity=(plan.recipe_identity if recalled_snapshot or
                any(row.identity==plan.recipe_identity
                    for row in o.cognition._prospective_recipes.values()) else 0)
            return o._issue_motor(action,prospective_recipe=recipe_identity,
                prospective_snapshot=recalled_snapshot,
                prospective_context_signature=prospective_context_signature)
        prospective_expression=o._emit_resident_prospective_expression(plan)
        if prospective_expression is not None:return prospective_expression
    source_action,assertions,source_occurrences,source_alternatives=o._source_nomination(True)
    if source_action is not None:
        o.information_need=();o.information_need_asked=False
        baseline=o._exploration_candidate();return o._issue_motor(
            source_action,assertions,o._source_context_signature(),source_occurrences,baseline,
            o._source_world_occurrences(assertions),o._source_somatic_occurrences(assertions))
    if plan.status==2:
        need=(2,*plan.alternative_actions)
        if o.information_need!=need:
            o.information_need=need;o.information_need_asked=False
        if not any(int(action) in o.affordances for action in plan.alternative_actions):
            selected=o._resident_selected_prospective_plan()
            if selected is not None:
                prospective_expression=o._emit_resident_prospective_expression(selected)
                if prospective_expression is not None:return prospective_expression
        if not o.information_need_asked:
            inquiry=o._emit_information_request(plan.alternative_actions)
            if inquiry is not None:return inquiry
        return None
    if len(source_alternatives)>1:
        need=(3,*source_alternatives)
        if o.information_need!=need:
            o.information_need=need;o.information_need_asked=False
        if not o.information_need_asked:
            inquiry=o._emit_information_request(source_alternatives)
            if inquiry is not None:return inquiry
        return None
    baseline=o._exploration_candidate()
    if not baseline:
        if o.affordances:o.information_need=(4,*sorted(o.affordances))
        return None
    return o._issue_motor(baseline)
