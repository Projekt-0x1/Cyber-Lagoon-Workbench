#!/usr/bin/env python3
"""N+1: self-initiated world topic shifts and later returns without transcript memory."""
from __future__ import annotations

import copy,json,time
from reference_language_mastery_contact_adapter_v1 import LanguageMasteryContactAdapterV1
from reference_lived_world_conversation_terminal_v1 import quiet
from reference_lived_world_followup_bridge_v1 import LivedWorldFollowupBridgeV1
from reference_lived_world_followup_terminal_v1 import respond_followup
from reference_organism_v2 import ReferenceOrganismV2
from reference_predictive_credit_profile_v1 import Q
from reference_self_initiated_world_followup_verify import QCTX,QMARK,prepared
from reference_world_derived_proposition_frontier_verify import WORLD_A,WORLD_B,SOURCE_A,SOURCE_B,world

ENGINEER=201;TECHNICIAN=202;OPERATOR=203;ANALYST=204;QMARK2=0xD703
visible_language_gain='SELF_INITIATED_TOPIC_SHIFTS_BETWEEN_LIVED_WORLDS_AND_LATER_RETURNS_WITH_FOCUSED_HUMAN_FOLLOWUP'
language_phenotype_improved=True
future_update_authority_preserved=True


def broaden_query_productivity(adult):
    contact=LanguageMasteryContactAdapterV1(adult)
    from reference_language_mastery_contact_adapter_v1 import CONTACT_SCENE,CONTACT_SURFACE
    for k in range(2):
        contact.contact(CONTACT_SCENE,(0xD700,QMARK2),7000+k)
        contact.contact(CONTACT_SURFACE,b'regarding',7100+k)
    for marker,prefix,base in ((QMARK,b'about ',8000),(QMARK2,b'regarding ',9000)):
        for atom,word in ((401,b'sensor?'),(402,b'valve?')):
            for k in range(2):
                contact.contact(CONTACT_SCENE,(QCTX,marker,atom),base+atom*10+k)
                contact.contact(CONTACT_SURFACE,prefix+word,base+atom*10+100+k)


def train_query_context(adult,organism,state,source,atom):
    adult._clear_current_occurrence();world(organism,state,source);spoken,last=quiet(adult,organism,0)
    if not spoken:raise RuntimeError('topic-return:development_first_turn')
    context,frontier=LivedWorldFollowupBridgeV1.activate_frontier(adult,organism,QCTX,(QMARK,atom),last)
    if context<=0 or not frontier:raise RuntimeError('topic-return:development_frontier')
    for leaf in frontier:
        for _ in range(2):
            adult.experience_discourse_candidate(leaf.identity,Q,context=context)
            adult.experience_discourse_background(leaf.identity,False)
    for left,right in zip(frontier,frontier[1:]):
        for _ in range(2):adult.experience_discourse_transition(left.identity,right.identity,context)
    return tuple(x.identity for x in frontier)


def ask(adult,organism,last,text):
    return respond_followup(adult,organism,LanguageMasteryContactAdapterV1(adult),text,last)


def main():
    started=time.perf_counter();checks={};adult,organism=prepared();broaden_query_productivity(adult)
    # Counterbalanced query-marker/entity development makes held-out combinations productive.
    for text,atom in ((b'about engineer?',ENGINEER),(b'about analyst?',ANALYST)):
        rows=adult.language.invert_surface(tuple(text))
        checks['productive_query_'+str(atom)]=(adult.construction_productivity(QCTX,(QMARK,atom))>=2 and len(rows)==1 and tuple(rows[0].atoms)==(QMARK,atom))
    train_query_context(adult,organism,WORLD_A,SOURCE_A,ENGINEER)
    train_query_context(adult,organism,WORLD_B,SOURCE_B,ANALYST)
    adult._clear_current_occurrence();world(organism,WORLD_A,SOURCE_A);last=0

    a1,last=quiet(adult,organism,last);a_engineer=ask(adult,organism,last,b'about engineer?')
    checks['world_a_self_initiates_and_human_can_focus_a_only_entity']=(
        len(a1)>300 and 0<len(a_engineer)<len(a1) and b'engineer' in a_engineer
        and b'technician' not in a_engineer and b'analyst' not in a_engineer and b'operator' not in a_engineer)

    world(organism,WORLD_B,SOURCE_B);b1,last=quiet(adult,organism,last)
    leaked_a=ask(adult,organism,last,b'about engineer?')
    b_analyst=ask(adult,organism,last,b'about analyst?')
    checks['topic_shift_changes_self_initiated_turn']=(len(b1)>300 and b1!=a1)
    checks['old_a_only_entity_cannot_leak_into_current_b_topic']=(leaked_a==b'')
    checks['current_b_only_entity_supports_focused_followup']=(
        0<len(b_analyst)<len(b1) and b'analyst' in b_analyst
        and b'engineer' not in b_analyst and b'technician' not in b_analyst and b'operator' not in b_analyst)

    # Interrupt at a natural boundary. No active topic/query is checkpoint authority.
    adult_cp=copy.deepcopy(adult.checkpoint());org_cp=copy.deepcopy(organism.checkpoint());last_cp=last
    checkpoint_text=json.dumps({'adult':adult_cp,'organism':org_cp,'last':last_cp},sort_keys=True)
    restored=type(adult).restore(adult_cp);restored_org=ReferenceOrganismV2.restore(org_cp)
    world(restored_org,WORLD_A,SOURCE_A);a2,last2=quiet(restored,restored_org,last_cp)
    returned_engineer=ask(restored,restored_org,last2,b'about engineer?')
    wrong_b=ask(restored,restored_org,last2,b'about analyst?')
    checks['genuine_world_a_return_reinstates_original_self_initiated_topic']=(a2==a1 and last2!=last_cp)
    checks['returned_a_topic_reinstates_focused_human_followup']=(returned_engineer==a_engineer)
    checks['b_only_entity_does_not_leak_after_return_to_a']=(wrong_b==b'')
    checks['checkpoint_has_no_transcript_active_topic_or_query_frontier']=(
        all(token not in checkpoint_text for token in ('transcript','conversation_buffer','context_window','active_topic','topic_frontier','query_frontier')))

    # Same physical world occurrence cannot create a topic-return event by cadence alone.
    same,last_same=quiet(restored,restored_org,last2)
    checks['unchanged_returned_world_does_not_repeat_by_idle_cadence']=(same==b'' and last_same==last2)
    checks['bounded_fast_path']=time.perf_counter()-started<0.2

    failed=[k for k,v in checks.items() if not v]
    result={'schema':'cyber-lagoon.self-initiated-topic-shift-return.v1','pass':not failed,'reference_only':True,'language_phenotype_improved':language_phenotype_improved,'future_update_authority_preserved':future_update_authority_preserved,'visible_language_gain':visible_language_gain,'bytes':{'a_spontaneous':len(a1),'a_engineer':len(a_engineer),'b_spontaneous':len(b1),'b_analyst':len(b_analyst),'returned_a':len(a2)},'checks':checks,'failed':failed,'remaining_red':['OPEN_QUESTION_FOLLOWUP_BINDING','ELLIPSIS_BEYOND_ENTITY_QUERY','OPEN_ENDED_CONVERSATIONAL_GENERATION','NOVEL_TOPIC_LEARNING_WITHOUT_PRETRAINED_RELEVANCE'],'elapsed_ms':round((time.perf_counter()-started)*1000,3)}
    print('FOUNDRY_SELF_INITIATED_TOPIC_SHIFT_RETURN_'+('GREEN' if not failed else 'RED'));print('visible_language_gain='+visible_language_gain);print(json.dumps(result,indent=2,sort_keys=True));return 0 if not failed else 1

if __name__=='__main__':raise SystemExit(main())
