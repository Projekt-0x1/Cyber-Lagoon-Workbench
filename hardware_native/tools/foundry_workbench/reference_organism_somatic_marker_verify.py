#!/usr/bin/env python3
from __future__ import annotations
import copy,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
from reference_organism_v2 import *
from reference_population_v1 import PopulationSpecV1

CTX=8801;P=9901;BODY_SOURCE=9902
ALICE,BOB,SEE,DOG,TEST,VALVE,INSPECT,SENSOR=201,202,301,401,302,402,303,403
ALICE2,REMOTE=501,601
FA=(11,12,13,14);FA2=(11,12,13,99);FR=(71,72,73,74)

def u(s):return tuple(s.encode())
def partner(o,p=P):o.contact(CONTACT_PARTNER_CONTEXT,(1,7,p),70000+p,True,True)
def body(o,src=BODY_SOURCE):return o.contact(CONTACT_BODY_STATE,(5011,5012,5013),src,True,True)
def feat(o,e,f,src):o.contact(CONTACT_ENTITY_FEATURES,(e,len(f),*f),src,True,True)
def name(o,e,text,src):
    o.contact(CONTACT_SCENE,(7,0,1,e),src,True,True);o.contact(CONTACT_SURFACE,u(text),src+1000,True,True)
def clause(o,atoms,text,src):
    o.contact(CONTACT_SCENE,(7,CTX,len(atoms),*atoms),src,True,True);o.contact(CONTACT_SURFACE,u(text),src+2000,True,True)

def train(o):
    for e,f,src in ((ALICE,FA,8001),(BOB,(21,22,23,24),8002),(ALICE2,FA2,8003),(REMOTE,FR,8004),
                    (SEE,(31,32),8011),(DOG,(41,42),8012),(TEST,(33,34),8013),(VALVE,(43,44),8014),(INSPECT,(35,36),8015),(SENSOR,(45,46),8016)):
        feat(o,e,f,src)
    for e,s in ((ALICE,'alice'),(BOB,'bob'),(REMOTE,'zoe'),(SEE,'sees'),(DOG,'dog'),(TEST,'tests'),(VALVE,'valve'),(INSPECT,'inspects'),(SENSOR,'sensor')):
        name(o,e,s,10000+e);name(o,e,s,11000+e)
    clause(o,(ALICE,SEE,DOG),'alice sees the dog.',30001);clause(o,(ALICE,SEE,DOG),'alice sees the dog.',30002)
    clause(o,(ALICE,INSPECT,SENSOR),'alice inspects the sensor.',30003);clause(o,(ALICE,INSPECT,SENSOR),'alice inspects the sensor.',30004)
    clause(o,(BOB,TEST,VALVE),'bob tests the valve.',30005);clause(o,(BOB,TEST,VALVE),'bob tests the valve.',30006)
    clause(o,(REMOTE,INSPECT,SENSOR),'zoe inspects the sensor.',30007);clause(o,(REMOTE,INSPECT,SENSOR),'zoe inspects the sensor.',30008)
    assert o.language.template(CTX,3) is not None

def speak(o,atoms,src):
    partner(o);o.contact(CONTACT_SCENE,(7,CTX,len(atoms),*atoms),src,True,True);return o.tick()

def main():
    t=time.perf_counter();checks={};spec=PopulationSpecV1(65536,2,4,42,8)
    o=ReferenceOrganismV2(spec);train(o)
    body_occurrence=body(o)
    first=speak(o,(ALICE,SEE,DOG),41001)
    checks['body_state_is_actual_action_participant']=(isinstance(first,ActionV2) and first.payload==u('alice sees the dog.') and first.body_occurrence==body_occurrence and first.body_occurrence in first.contributors and first.body_signature!=0 and first.body_source==BODY_SOURCE)
    before_lang=len(o.selection_configuration_revisions)
    o.contact(CONTACT_CONSEQUENCE,(first.ticket,-1),P,True,True)
    checks['language_credit_stays_configuration_local']=len(o.selection_configuration_revisions)==before_lang+1
    checks['somatic_marker_is_sibling_store']=o._somatic_revisions.row_count==3 and ALICE in o._somatic_marked_entities()
    cp=copy.deepcopy(o.checkpoint())
    checks['checkpoint_has_no_media_blob']=all(k not in cp for k in ('image','audio','png','wav','pixels','samples')) and isinstance(cp.get('somatic_revisions_packed'),str)
    other_o=ReferenceOrganismV2.restore(copy.deepcopy(cp));other=speak(other_o,(ALICE,INSPECT,SENSOR),42001)
    checks['negative_marker_transfers_across_constructions']=isinstance(other,ActionV2) and other.payload==u('alice inspects the sensor.') and bool(other.somatic_occurrences) and all(oid in other.contributors for oid in other.somatic_occurrences)
    bob_o=ReferenceOrganismV2.restore(copy.deepcopy(cp));bob=speak(bob_o,(BOB,TEST,VALVE),43001)
    checks['unmarked_referent_still_speaks']=isinstance(bob,ActionV2) and bob.payload==u('bob tests the valve.') and not bob_o._somatic_state_occurrences(CTX,(BOB,TEST,VALVE))
    remat_o=ReferenceOrganismV2.restore(copy.deepcopy(cp));remat=speak(remat_o,(ALICE2,INSPECT,SENSOR),44001)
    checks['overlapping_identity_recruits_same_marker']=isinstance(remat,ActionV2) and bool(remat.somatic_occurrences) and ALICE in remat_o._overlapping_entities(ALICE2) and all(oid in remat.contributors for oid in remat.somatic_occurrences)
    remote_o=ReferenceOrganismV2.restore(copy.deepcopy(cp));remote=speak(remote_o,(REMOTE,INSPECT,SENSOR),45001)
    checks['unrelated_features_do_not_inherit_marker']=isinstance(remote,ActionV2) and remote.payload==u('zoe inspects the sensor.') and not remote_o._somatic_state_occurrences(CTX,(REMOTE,INSPECT,SENSOR))
    no_body=ReferenceOrganismV2(spec);train(no_body);na=speak(no_body,(ALICE,SEE,DOG),45501)
    no_body.contact(CONTACT_CONSEQUENCE,(na.ticket,-1),P,True,True)
    checks['consequence_without_body_occurrence_cannot_mint_marker']=no_body._somatic_revisions.row_count==0
    late=ReferenceOrganismV2(spec);train(late);la=speak(late,(ALICE,SEE,DOG),45502);body(late)
    late.contact(CONTACT_CONSEQUENCE,(la.ticket,-1),P,True,True)
    checks['body_contact_after_action_cannot_retroactively_participate']=late._somatic_revisions.row_count==0
    yoked=ReferenceOrganismV2(spec);train(yoked);body(yoked);ya=speak(yoked,(ALICE,SEE,DOG),46001)
    yoked.contact(CONTACT_CONSEQUENCE,(ya.ticket,-1),P,True,False)
    checks['yoked_return_cannot_write_somatic_marker']=yoked._somatic_revisions.row_count==0 and isinstance(speak(yoked,(ALICE,INSPECT,SENSOR),46002),ActionV2)
    r=ReferenceOrganismV2.restore(copy.deepcopy(cp));ra=speak(r,(ALICE,INSPECT,SENSOR),47001)
    checks['checkpoint_keeps_somatic_recruitment']=isinstance(ra,ActionV2) and bool(ra.somatic_occurrences) and all(oid in ra.contributors for oid in ra.somatic_occurrences)
    cut=ReferenceOrganismV2.restore(copy.deepcopy(cp));cut.contact(CONTACT_WITHDRAW_SOURCE,(P,),88001,True,True);ca=speak(cut,(ALICE,INSPECT,SENSOR),48001)
    checks['source_withdrawal_removes_somatic_recruitment']=isinstance(ca,ActionV2) and not cut._somatic_state_occurrences(CTX,(ALICE,INSPECT,SENSOR))
    body_cut=ReferenceOrganismV2.restore(copy.deepcopy(cp));body_cut.contact(CONTACT_WITHDRAW_SOURCE,(BODY_SOURCE,),88002,True,True);bca=speak(body_cut,(ALICE,INSPECT,SENSOR),48002)
    checks['body_source_withdrawal_removes_somatic_recruitment']=isinstance(bca,ActionV2) and not body_cut._somatic_state_occurrences(CTX,(ALICE,INSPECT,SENSOR))
    checks['no_stored_picture_or_audio']=not hasattr(o,'image') and not hasattr(o,'audio') and not hasattr(o,'imagine') and not hasattr(o,'valence')
    checks['no_emotion_opcode']=not hasattr(o,'feel') and not hasattr(o,'somatic_marker') and PREF_SOMA not in (PREF_TEMPLATE,PREF_LEXEME,PREF_FORM)
    result={'schema':'agi.reference-organism-body-history-marker.v2','pass':all(checks.values()),'checks':checks,'runtime_llm':False,'graph_flip':False,'papers':['Garfinkel et al. 2013 Psychophysiology PMID 23521494','Bechara et al. 1997 Science PMID 9036851','Maia and McClelland 2004 PNAS critique'],'claim':'ACTUAL_BODY_OCCURRENCE_CONDITIONS_SOURCE_SPECIFIC_LANGUAGE_BIAS_WITHOUT_MEDIA_OR_VALENCE_SCALAR_REFERENCE_ONLY','checkpoint_bytes':len(json.dumps(cp,sort_keys=True,separators=(',',':')).encode()),'somatic_recipe_bytes':len(bytes.fromhex(cp['somatic_revisions_packed'])),'somatic_rows':o._somatic_revisions.row_count,'body_occurrence_sites':len(next(row for row in o.population.occurrences if row.identity==body_occurrence).sites),'resident_sites':o.population.spec.site_count,'last_somatic_touches':other_o.last_somatic_touches,'elapsed_ms':round((time.perf_counter()-t)*1000,3)}
    print('FOUNDRY_REFERENCE_ORGANISM_SOMATIC_MARKER '+('GREEN' if result['pass'] else 'RED')+' body_occurrence=1 transfer=1 overlap=1 media=0 language_outer=1')
    print(json.dumps(result,indent=2,sort_keys=True));raise SystemExit(0 if result['pass'] else 1)
if __name__=='__main__':main()
