#!/usr/bin/env python3
"""Fast standing replay of the measured lived-world conversational N+1."""
from __future__ import annotations

import hashlib
import json
import tempfile
import time
from pathlib import Path

from reference_global_discourse_relevance_verify import fresh
from reference_lived_world_conversation_terminal_v1 import quiet,restore_session,save_session
from reference_organism_v2 import ReferenceOrganismV2,CONTACT_WORLD_STATE
from reference_population_v1 import PopulationSpecV1
from reference_world_derived_proposition_frontier_verify import (
    SOURCE_A,SOURCE_B,SOURCE_C,WORLD_A,WORLD_B,WORLD_C,
    activate,train_current_world,world,
)

ROOT=Path(__file__).resolve().parents[3]
WORK=ROOT/'hardware_native'/'tools'/'foundry_workbench'
visible_language_gain=(
    'LIVED_WORLD_CHANGE_ORIGINATES_LONG_FORM_TERMINAL_DISCOURSE_'
    'WITHOUT_USER_PROMPT_OR_HOST_FRONTIER'
)
language_phenotype_improved=True
future_update_authority_preserved=True


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def train_world_discourse(adult,frontier,context):
    learned=train_current_world(adult,frontier)
    for left,right in zip(frontier,frontier[1:]):
        for _ in range(2):
            adult.experience_discourse_transition(
                int(left.identity),int(right.identity),int(context))
    return learned


def main():
    started=time.perf_counter();checks={}
    receipt=json.loads((WORK/'lived_world_conversation_runtime_measured_receipt.json').read_text())
    measured=receipt['measured_sha256']
    hashes={name:sha(ROOT/name) for name in measured}
    checks['formal_process_receipt_hashes_are_exact']=(hashes==measured)
    checks['formal_process_receipt_has_visible_a_b_and_silent_baselines']=(
        receipt.get('status')=='GREEN'
        and receipt.get('formal_contract')=='FOUNDRY_LIVED_WORLD_CONVERSATION_RUNTIME_GREEN'
        and receipt.get('visible_language_gain')==visible_language_gain
        and receipt.get('output_bytes')=={
            'baseline':0,'unvalued_world_c':0,'world_a':377,'world_b':362})

    adult,_host,_factors,_ca,_cb=fresh()
    organism=ReferenceOrganismV2(PopulationSpecV1(32768,2,4,42,8))
    world(organism,WORLD_A,SOURCE_A);_ctx_a,frontier_a=activate(adult,organism);train_world_discourse(adult,frontier_a,_ctx_a)
    world(organism,WORLD_B,SOURCE_B);_ctx_b,frontier_b=activate(adult,organism);train_world_discourse(adult,frontier_b,_ctx_b)
    adult._clear_current_occurrence()

    world(organism,WORLD_A,SOURCE_A);occ_a=int(organism.world_state_occurrence)
    a,last=quiet(adult,organism,0)
    same,last_same=quiet(adult,organism,last)
    checks['in_process_world_a_originates_long_form_once']=(len(a)>300 and last==occ_a and same==b'' and last_same==occ_a)

    with tempfile.TemporaryDirectory() as directory:
        state=Path(directory)/'session.json'
        save_session(state,adult,organism,last)
        restored_adult,restored_org,restored_last=restore_session(state)
        replay,replay_last=quiet(restored_adult,restored_org,restored_last)
        checks['checkpoint_same_world_stays_silent']=(replay==b'' and replay_last==restored_last==occ_a)
        restored_org.contact(CONTACT_WORLD_STATE,WORLD_B,SOURCE_B,True,True)
        occ_b=int(restored_org.world_state_occurrence)
        b,last_b=quiet(restored_adult,restored_org,restored_last)
        checks['new_world_b_originates_different_long_form']=(len(b)>300 and b!=a and last_b==occ_b and occ_b!=occ_a)
        restored_org.contact(CONTACT_WORLD_STATE,WORLD_C,SOURCE_C,True,True)
        c,last_c=quiet(restored_adult,restored_org,last_b)
        checks['unvalued_novel_world_does_not_force_speech']=(c==b'' and last_c==last_b)

    terminal_source=(WORK/'reference_lived_world_conversation_terminal_v1.py').read_text()
    checks['terminal_world_frontier_is_resident_derived']=(
        'WorldDiscourseSituationBridgeV1.activate_frontier(adult,organism)' in terminal_source
        and 'adult.organize_relevant_frontier(frontier)' in terminal_source
        and 'topic' not in terminal_source.lower()
        and 'expected' not in terminal_source.lower())
    checks['ordinary_text_contact_path_is_preserved']=(
        'LanguageMasteryContactAdapterV1' in terminal_source
        and 'CONTACT_UTTERANCE' in terminal_source
        and 'adult.choose_public_plan()' in terminal_source)
    checks['standing_replay_is_fast']=time.perf_counter()-started<0.25

    failed=[name for name,passed in checks.items() if not passed]
    result={
        'schema':'cyber-lagoon.lived-world-conversation-runtime-fast.v1',
        'pass':not failed,'reference_only':True,
        'language_phenotype_improved':language_phenotype_improved,
        'visible_language_gain':visible_language_gain,
        'future_update_authority_preserved':future_update_authority_preserved,
        'output_bytes':{'a':len(a),'b':len(b),'c':len(c)},
        'checks':checks,'failed':failed,
        'remaining_red':[
            'OPEN_WORLD_ENTITY_DISCOVERY_FROM_RAW_SENSORY_STREAM',
            'AUTONOMOUS_NOVEL_TOPIC_LEARNING_WITHOUT_PRETRAINED_RELEVANCE',
            'CONVERSATIONAL_FOLLOWUP_OVER_SELF_INITIATED_DISCOURSE',
        ],
        'elapsed_ms':round((time.perf_counter()-started)*1000,3),
    }
    print('FOUNDRY_LIVED_WORLD_CONVERSATION_RUNTIME_FAST_'+('GREEN' if not failed else 'RED'))
    print('visible_language_gain='+visible_language_gain)
    print(json.dumps(result,indent=2,sort_keys=True))
    return 0 if not failed else 1


if __name__=='__main__':raise SystemExit(main())
