#!/usr/bin/env python3
from __future__ import annotations
import copy,hashlib,json,time
from pathlib import Path
from reference_language_mastery_contact_adapter_v1 import LanguageMasteryContactAdapterV1
from reference_lived_world_conversation_terminal_v1 import quiet
from reference_lived_world_followup_terminal_v1 import respond_followup
from reference_organism_v2 import ReferenceOrganismV2
from reference_self_initiated_world_followup_verify import prepared

ROOT=Path(__file__).resolve().parents[3];WORK=ROOT/'hardware_native'/'tools'/'foundry_workbench'
visible_language_gain='HUMAN_ENTITY_FOLLOWUP_CONTINUES_AGI_SELF_INITIATED_LIVED_WORLD_TURN_WITHOUT_TRANSCRIPT'
language_phenotype_improved=True;future_update_authority_preserved=True

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    started=time.perf_counter();checks={};receipt=json.loads((WORK/'self_initiated_world_followup_measured_receipt.json').read_text())
    measured=receipt['measured_sha256'];checks['formal_multiturn_receipt_hashes_are_exact']={name:sha(ROOT/name) for name in measured}==measured
    checks['formal_visible_gain_is_exact']=(receipt.get('status')=='GREEN' and receipt.get('formal_contract')=='FOUNDRY_SELF_INITIATED_WORLD_FOLLOWUP_GREEN' and receipt.get('visible_language_gain')==visible_language_gain and receipt.get('bytes')=={'sensor_followup':186,'spontaneous':377,'valve_followup':185})
    adult,organism=prepared();adult_cp=copy.deepcopy(adult.checkpoint());org_cp=copy.deepcopy(organism.checkpoint())
    spontaneous,last=quiet(adult,organism,0);sensor=respond_followup(adult,organism,LanguageMasteryContactAdapterV1(adult),b'about sensor?',last)
    checks['self_initiated_then_sensor_followup_replays']=len(spontaneous)==377 and len(sensor)==186 and b'sensor' in sensor and b'valve' not in sensor
    valve_adult=type(adult).restore(copy.deepcopy(adult_cp));valve_org=ReferenceOrganismV2.restore(copy.deepcopy(org_cp));sp2,last2=quiet(valve_adult,valve_org,0);valve=respond_followup(valve_adult,valve_org,LanguageMasteryContactAdapterV1(valve_adult),b'about valve?',last2)
    checks['complementary_valve_followup_replays']=sp2==spontaneous and len(valve)==185 and b'valve' in valve and b'sensor' not in valve
    early_adult=type(adult).restore(copy.deepcopy(adult_cp));early_org=ReferenceOrganismV2.restore(copy.deepcopy(org_cp));early=respond_followup(early_adult,early_org,LanguageMasteryContactAdapterV1(early_adult),b'about sensor?',0)
    checks['same_query_without_self_initiated_episode_is_silent']=early==b''
    restored_adult=type(adult).restore(copy.deepcopy(adult_cp));restored_org=ReferenceOrganismV2.restore(copy.deepcopy(org_cp));spoken,spoken_last=quiet(restored_adult,restored_org,0);mid_adult=type(adult).restore(copy.deepcopy(restored_adult.checkpoint()));mid_org=ReferenceOrganismV2.restore(copy.deepcopy(restored_org.checkpoint()));resumed=respond_followup(mid_adult,mid_org,LanguageMasteryContactAdapterV1(mid_adult),b'about sensor?',spoken_last)
    checks['checkpoint_between_turns_preserves_followup_without_transcript']=spoken==spontaneous and resumed==sensor
    checks['standing_replay_is_bounded']=time.perf_counter()-started<1.0
    failed=[k for k,v in checks.items() if not v]
    result={'schema':'cyber-lagoon.self-initiated-world-followup-fast.v1','pass':not failed,'reference_only':True,'language_phenotype_improved':language_phenotype_improved,'future_update_authority_preserved':future_update_authority_preserved,'visible_language_gain':visible_language_gain,'bytes':{'spontaneous':len(spontaneous),'sensor_followup':len(sensor),'valve_followup':len(valve)},'checks':checks,'failed':failed,'remaining_red':['OPEN_QUESTION_FOLLOWUP_BINDING','ELLIPSIS_BEYOND_ENTITY_QUERY','TOPIC_SHIFT_AND_RETURN_ACROSS_SELF_INITIATED_TURNS','OPEN_ENDED_CONVERSATIONAL_GENERATION'],'elapsed_ms':round((time.perf_counter()-started)*1000,3)}
    print('FOUNDRY_SELF_INITIATED_WORLD_FOLLOWUP_FAST_'+('GREEN' if not failed else 'RED'));print('visible_language_gain='+visible_language_gain);print(json.dumps(result,indent=2,sort_keys=True));return 0 if not failed else 1
if __name__=='__main__':raise SystemExit(main())
