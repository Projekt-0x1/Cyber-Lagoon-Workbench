#!/usr/bin/env python3
"""Visible N+1: continuing Workbench Adult originates discourse from lived world without user prompt."""
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
from reference_language_mastery_adult_v1 import LanguageMasteryAdultV1
from reference_organism_v2 import ReferenceOrganismV2,CONTACT_WORLD_STATE
from reference_population_v1 import PopulationSpecV1
from reference_predictive_credit_profile_v1 import Q
from reference_world_derived_proposition_frontier_verify import (
    SOURCE_A,SOURCE_B,SOURCE_C,WORLD_A,WORLD_B,WORLD_C,
    activate,train_current_world,world,
)

visible_language_gain=(
    'LIVED_WORLD_CHANGE_ORIGINATES_LONG_FORM_TERMINAL_DISCOURSE_'
    'WITHOUT_USER_PROMPT_OR_HOST_FRONTIER'
)


def write_json(path,payload):
    path.write_text(json.dumps(payload,separators=(',',':'),sort_keys=True))


def run_once(terminal,checkpoint,expect_output,probe_seconds=0.15):
    p=subprocess.Popen(
        (sys.executable,str(terminal),'--resume',str(checkpoint),'--idle-ms','1'),
        stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    first=b'';repeat=b''
    try:
        ready=select.select((p.stdout,),(),(),0.8)[0]
        if ready:first=p.stdout.readline()
        if expect_output:
            again=select.select((p.stdout,),(),(),probe_seconds)[0]
            if again:repeat=p.stdout.readline()
        else:
            # Keep the process alive through several physical no-contact opportunities.
            time.sleep(probe_seconds)
            again=select.select((p.stdout,),(),(),0)[0]
            if again:repeat=p.stdout.readline()
        p.stdin.close();p.wait(timeout=2)
    finally:
        if p.poll() is None:
            p.terminate();p.wait(timeout=2)
    return first.rstrip(b'\r\n'),repeat.rstrip(b'\r\n'),p.returncode


def session_payload(adult,organism,last=0):
    return {
        'adult':copy.deepcopy(adult.checkpoint()),
        'organism':copy.deepcopy(organism.checkpoint()),
        'last_spoken_world_occurrence':int(last),
    }


def train_world_discourse(adult,frontier,context):
    learned=train_current_world(adult,frontier)
    for left,right in zip(frontier,frontier[1:]):
        for _ in range(2):
            adult.experience_discourse_transition(
                int(left.identity),int(right.identity),int(context))
    return learned


def main():
    started=time.perf_counter();checks={}
    adult,host_frontier,_factors,_ca,_cb=fresh()
    organism=ReferenceOrganismV2(PopulationSpecV1(32768,2,4,42,8))

    # Development uses the already-paid world-derived proposition frontier.
    world(organism,WORLD_A,SOURCE_A);ctx_a,frontier_a=activate(adult,organism)
    expected_a=train_world_discourse(adult,frontier_a,ctx_a)
    world(organism,WORLD_B,SOURCE_B);ctx_b,frontier_b=activate(adult,organism)
    expected_b=train_world_discourse(adult,frontier_b,ctx_b)
    adult._clear_current_occurrence()

    here=Path(__file__).resolve().parent
    legacy=here/'reference_language_mastery_terminal_v1.py'
    terminal=here/'reference_lived_world_conversation_terminal_v1.py'

    with tempfile.TemporaryDirectory() as directory:
        d=Path(directory)
        # Baseline terminal has the same learned Adult but no organism/world bridge.
        adult_only=d/'adult-only.json';write_json(adult_only,adult.checkpoint())
        legacy_p=subprocess.Popen(
            (sys.executable,str(legacy),'--resume',str(adult_only),'--idle-ms','1'),
            stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        legacy_ready=select.select((legacy_p.stdout,),(),(),0.15)[0]
        legacy_output=legacy_p.stdout.readline() if legacy_ready else b''
        legacy_p.stdin.close();legacy_p.wait(timeout=2)
        checks['baseline_same_learned_adult_stays_silent_without_lived_world_body']=(
            legacy_output==b'' and legacy_p.returncode==0)

        # Re-enter A and let physical idle time supply only an opportunity. No stdin.
        world(organism,WORLD_A,SOURCE_A);occ_a=int(organism.world_state_occurrence)
        session=d/'world-session.json';write_json(session,session_payload(adult,organism,0))
        a,repeat_a,rc_a=run_once(terminal,session,True)
        saved_a=json.loads(session.read_text())
        checks['world_a_originates_long_form_without_user_prompt']=(
            rc_a==0 and len(a)>300 and saved_a['last_spoken_world_occurrence']==occ_a)
        checks['same_world_occurrence_does_not_repeat_on_continuing_quiet']=(repeat_a==b'')

        # Restart from the exact combined checkpoint: same occurrence remains silent.
        same,repeat_same,rc_same=run_once(terminal,session,False)
        checks['restart_does_not_repeat_already_spoken_world_occurrence']=(
            rc_same==0 and same==b'' and repeat_same==b'')

        # A new current world occurrence is body state, not a host topic/frontier.
        saved=json.loads(session.read_text())
        restored_org=ReferenceOrganismV2.restore(saved['organism'])
        restored_org.contact(CONTACT_WORLD_STATE,WORLD_B,SOURCE_B,True,True)
        occ_b=int(restored_org.world_state_occurrence)
        saved['organism']=restored_org.checkpoint();write_json(session,saved)
        b,repeat_b,rc_b=run_once(terminal,session,True)
        saved_b=json.loads(session.read_text())
        checks['world_b_new_occurrence_originates_different_long_form']=(
            rc_b==0 and len(b)>300 and b!=a and
            saved_b['last_spoken_world_occurrence']==occ_b and occ_b!=occ_a)
        checks['world_b_also_emits_only_once']=(repeat_b==b'')

        # A broader novel world has mechanically available propositions but no
        # relevance history; novelty alone must not force the Adult to talk.
        saved=json.loads(session.read_text())
        restored_org=ReferenceOrganismV2.restore(saved['organism'])
        restored_org.contact(CONTACT_WORLD_STATE,WORLD_C,SOURCE_C,True,True)
        saved['organism']=restored_org.checkpoint();write_json(session,saved)
        c,repeat_c,rc_c=run_once(terminal,session,False)
        checks['novel_world_without_learned_relevance_remains_silent']=(
            rc_c==0 and c==b'' and repeat_c==b'')

    source=terminal.read_text()
    checks['runtime_derives_frontier_inside_body_not_from_stdin_or_host_topic']=(
        'WorldDiscourseSituationBridgeV1.activate_frontier(adult,organism)' in source
        and 'adult.organize_relevant_frontier(frontier)' in source
        and 'topic' not in source.lower()
        and 'expected' not in source.lower())
    checks['ordinary_text_contact_reuses_existing_contact_adapter']=(
        'CONTACT_UTTERANCE' in source and 'LanguageMasteryContactAdapterV1' in source
        and 'adult.choose_public_plan()' in source)
    checks['development_frontiers_are_current_world_derived_not_host_frontier']=(
        len(frontier_a)==len(frontier_b)==8
        and tuple(x.identity for x in frontier_a)==expected_a
        and tuple(x.identity for x in frontier_b)==expected_b
        and set(expected_a).isdisjoint(set(expected_b)))
    checks['bounded_process_runtime_assay']=time.perf_counter()-started<4.0

    failed=[k for k,v in checks.items() if not v]
    result={
        'schema':'cyber-lagoon.lived-world-conversation-runtime.v1',
        'pass':not failed,
        'reference_only':True,
        'language_phenotype_improved':True,
        'visible_language_gain':visible_language_gain,
        'world_contexts':{'a':ctx_a,'b':ctx_b},
        'output_bytes':{'baseline':len(legacy_output),'a':len(a),'b':len(b),'c':len(c)},
        'checks':checks,'failed':failed,
        'remaining_red':[
            'OPEN_WORLD_ENTITY_DISCOVERY_FROM_RAW_SENSORY_STREAM',
            'AUTONOMOUS_NOVEL_TOPIC_LEARNING_WITHOUT_PRETRAINED_RELEVANCE',
            'CONVERSATIONAL_FOLLOWUP_OVER_SELF_INITIATED_DISCOURSE',
        ],
        'elapsed_ms':round((time.perf_counter()-started)*1000,3),
    }
    print('FOUNDRY_LIVED_WORLD_CONVERSATION_RUNTIME_'+('GREEN' if not failed else 'RED'))
    print('visible_language_gain='+visible_language_gain)
    print(json.dumps(result,indent=2,sort_keys=True))
    return 0 if not failed else 1


if __name__=='__main__':raise SystemExit(main())
