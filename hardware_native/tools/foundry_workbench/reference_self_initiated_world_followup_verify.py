#!/usr/bin/env python3
"""N+1: human learned entity follow-up continues the AGI's own spontaneous world turn."""
from __future__ import annotations

import copy
import inspect
import json
import select
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from reference_global_discourse_relevance_verify import fresh
from reference_language_mastery_contact_adapter_v1 import CONTACT_SCENE,CONTACT_SURFACE,LanguageMasteryContactAdapterV1
from reference_lived_world_conversation_runtime_verify import train_world_discourse
from reference_lived_world_conversation_terminal_v1 import quiet,save_session
from reference_lived_world_followup_bridge_v1 import LivedWorldFollowupBridgeV1
from reference_lived_world_followup_terminal_v1 import respond_followup
from reference_organism_v2 import ReferenceOrganismV2
from reference_population_v1 import PopulationSpecV1
from reference_predictive_credit_profile_v1 import Q
from reference_world_derived_proposition_frontier_verify import SOURCE_A,SOURCE_B,WORLD_A,WORLD_B,activate,world

QCTX=0xD701;QMARK=0xD702;SENSOR=401;VALVE=402
visible_language_gain='HUMAN_ENTITY_FOLLOWUP_CONTINUES_AGI_SELF_INITIATED_LIVED_WORLD_TURN_WITHOUT_TRANSCRIPT'
language_phenotype_improved=True
future_update_authority_preserved=True


def train_queries(adult):
    contact=LanguageMasteryContactAdapterV1(adult)
    for k in range(2):
        contact.contact(CONTACT_SCENE,(0xD700,QMARK),900+k)
        contact.contact(CONTACT_SURFACE,b'about',1000+k)
    for atom,word,base in ((SENSOR,b'about sensor?',2000),(VALVE,b'about valve?',3000)):
        for k in range(2):
            contact.contact(CONTACT_SCENE,(QCTX,QMARK,atom),base+k)
            contact.contact(CONTACT_SURFACE,word,base+100+k)


def train_followup_discourse(adult,organism,state,source):
    adult._clear_current_occurrence();world(organism,state,source)
    spoken,last=quiet(adult,organism,0)
    if not spoken:raise RuntimeError('followup:development_first_turn')
    for atom in (SENSOR,VALVE):
        context,frontier=LivedWorldFollowupBridgeV1.activate_frontier(
            adult,organism,QCTX,(QMARK,atom),last)
        if context<=0 or not frontier:raise RuntimeError('followup:development_frontier')
        for leaf in frontier:
            for _ in range(2):
                adult.experience_discourse_candidate(leaf.identity,Q,context=context)
                adult.experience_discourse_background(leaf.identity,False)
        for left,right in zip(frontier,frontier[1:]):
            for _ in range(2):
                adult.experience_discourse_transition(
                    left.identity,right.identity,context)


def prepared():
    adult,_host,_factors,_ca,_cb=fresh();organism=ReferenceOrganismV2(PopulationSpecV1(32768,2,4,42,8))
    for state,source in ((WORLD_A,SOURCE_A),(WORLD_B,SOURCE_B)):
        world(organism,state,source);context,frontier=activate(adult,organism)
        train_world_discourse(adult,frontier,context)
    train_queries(adult)
    train_followup_discourse(adult,organism,WORLD_A,SOURCE_A)
    train_followup_discourse(adult,organism,WORLD_B,SOURCE_B)
    adult._clear_current_occurrence();world(organism,WORLD_A,SOURCE_A)
    return adult,organism


def process_dialogue(terminal,checkpoint,query,idle_ms=1):
    p=subprocess.Popen((sys.executable,str(terminal),'--resume',str(checkpoint),'--idle-ms',str(idle_ms)),stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    first=second=b''
    try:
        ready=select.select((p.stdout,),(),(),1.0)[0]
        if ready:first=p.stdout.readline().rstrip(b'\r\n')
        p.stdin.write(query+b'\n');p.stdin.flush()
        ready=select.select((p.stdout,),(),(),1.0)[0]
        if ready:second=p.stdout.readline().rstrip(b'\r\n')
        p.stdin.close();p.wait(timeout=2)
    finally:
        if p.poll() is None:p.terminate();p.wait(timeout=2)
    return first,second,p.returncode


def query_only(terminal,checkpoint,query):
    p=subprocess.Popen((sys.executable,str(terminal),'--resume',str(checkpoint),'--idle-ms','1000'),stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    try:
        p.stdin.write(query+b'\n');p.stdin.flush()
        ready=select.select((p.stdout,),(),(),0.25)[0]
        out=p.stdout.readline().rstrip(b'\r\n') if ready else b''
        p.stdin.close();p.wait(timeout=2)
    finally:
        if p.poll() is None:p.terminate();p.wait(timeout=2)
    return out,p.returncode


def main():
    started=time.perf_counter();checks={};adult,organism=prepared()
    base_adult=copy.deepcopy(adult.checkpoint());base_org=copy.deepcopy(organism.checkpoint())

    # Resident expected behavior, independently of subprocess transport.
    expected_adult=type(adult).restore(copy.deepcopy(base_adult));expected_org=ReferenceOrganismV2.restore(copy.deepcopy(base_org))
    spontaneous,last=quiet(expected_adult,expected_org,0)
    contact=LanguageMasteryContactAdapterV1(expected_adult)
    sensor=respond_followup(expected_adult,expected_org,contact,b'about sensor?',last)
    checks['resident_self_initiated_turn_then_sensor_followup_is_focused']=(len(spontaneous)>300 and 120<len(sensor)<len(spontaneous))
    checks['focused_followup_mentions_only_queried_world_object']=(b'sensor' in sensor and b'valve' not in sensor)

    valve_adult=type(adult).restore(copy.deepcopy(base_adult));valve_org=ReferenceOrganismV2.restore(copy.deepcopy(base_org));sp2,last2=quiet(valve_adult,valve_org,0)
    valve=respond_followup(valve_adult,valve_org,LanguageMasteryContactAdapterV1(valve_adult),b'about valve?',last2)
    checks['same_self_initiated_turn_supports_complementary_valve_followup']=(sp2==spontaneous and 120<len(valve)<len(spontaneous) and b'valve' in valve and b'sensor' not in valve and valve!=sensor)

    terminal=Path(__file__).with_name('reference_lived_world_followup_terminal_v1.py')
    with tempfile.TemporaryDirectory() as directory:
        d=Path(directory);session=d/'session.json';save_session(session,type(adult).restore(copy.deepcopy(base_adult)),ReferenceOrganismV2.restore(copy.deepcopy(base_org)),0)
        first,second,rc=process_dialogue(terminal,session,b'about sensor?')
        checks['physical_multiturn_session_self_initiates_then_answers_human_followup']=(rc==0 and first==spontaneous and second==sensor)
        saved=json.loads(session.read_text())
        checks['checkpoint_contains_no_transcript_or_topic_sidecar']=all(k not in json.dumps(saved,sort_keys=True) for k in ('transcript','conversation_buffer','context_window','topic_id','query_frontier'))

        # Exact same learned state/query before self-initiation has no episode license.
        before=d/'before.json';save_session(before,type(adult).restore(copy.deepcopy(base_adult)),ReferenceOrganismV2.restore(copy.deepcopy(base_org)),0)
        early,early_rc=query_only(terminal,before,b'about sensor?')
        checks['same_query_before_self_initiated_turn_stays_silent']=(early_rc==0 and early==b'')

        # Natural turn boundary: self-initiate in-process, checkpoint, restart, then query.
        mid_adult=type(adult).restore(copy.deepcopy(base_adult));mid_org=ReferenceOrganismV2.restore(copy.deepcopy(base_org));mid_spoken,mid_last=quiet(mid_adult,mid_org,0)
        mid=d/'mid.json';save_session(mid,mid_adult,mid_org,mid_last)
        resumed,resumed_rc=query_only(terminal,mid,b'about sensor?')
        checks['restart_after_self_initiated_turn_preserves_followup_without_replaying_turn']=(mid_spoken==spontaneous and resumed_rc==0 and resumed==sensor)

    bridge_source=inspect.getsource(LivedWorldFollowupBridgeV1)
    terminal_source=terminal.read_text()
    checks['bridge_uses_learned_concept_identity_not_surface_string_matching']=(
        'historical_lexeme_binding' in bridge_source and 'query_atoms' in bridge_source
        and 'bytes(' not in bridge_source and '.decode(' not in bridge_source and 'sensor' not in bridge_source.lower() and 'valve' not in bridge_source.lower())
    checks['followup_requires_same_spoken_world_occurrence']=(
        'occurrence!=int(spoken_world_occurrence)' in bridge_source.replace(' ',''))
    checks['terminal_has_no_host_topic_expected_answer_or_transcript_authority']=(
        all(token not in terminal_source.lower() for token in ('topic_id','expected_answer','transcript','conversation_buffer')))
    checks['bounded_fast_path']=time.perf_counter()-started<2.0

    failed=[k for k,v in checks.items() if not v]
    result={'schema':'cyber-lagoon.self-initiated-world-followup.v1','pass':not failed,'reference_only':True,'language_phenotype_improved':language_phenotype_improved,'future_update_authority_preserved':future_update_authority_preserved,'visible_language_gain':visible_language_gain,'bytes':{'spontaneous':len(spontaneous),'sensor_followup':len(sensor),'valve_followup':len(valve)},'checks':checks,'failed':failed,'remaining_red':['OPEN_QUESTION_FOLLOWUP_BINDING','ELLIPSIS_BEYOND_ENTITY_QUERY','TOPIC_SHIFT_AND_RETURN_ACROSS_SELF_INITIATED_TURNS','OPEN_ENDED_CONVERSATIONAL_GENERATION'],'elapsed_ms':round((time.perf_counter()-started)*1000,3)}
    print('FOUNDRY_SELF_INITIATED_WORLD_FOLLOWUP_'+('GREEN' if not failed else 'RED'));print('visible_language_gain='+visible_language_gain);print(json.dumps(result,indent=2,sort_keys=True));return 0 if not failed else 1


if __name__=='__main__':raise SystemExit(main())
