#!/usr/bin/env python3
from __future__ import annotations
import copy,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
from reference_organism_v2 import *
from reference_population_v1 import PopulationSpecV1

CTX=8801;P=9901;P2=9902;SEE=77001;EN,EN2,DE,DE2=1101,1102,2101,2102
ALICE,BOB,INSPECT,SENSOR,TEST,VALVE=201,202,303,403,302,402
FA=(11,12,13,14);FS=(45,46,47,48)

def u(s):return tuple(s.encode())
def partner(o,p=P):o.contact(CONTACT_PARTNER_CONTEXT,(1,7,p),70000+p,True,True)
def feat(o,e,f,src):o.contact(CONTACT_ENTITY_FEATURES,(e,len(f),*f),src,True,True)
def name(o,e,text,src):
    o.contact(CONTACT_SCENE,(7,0,1,e),src,True,True);o.contact(CONTACT_SURFACE,u(text),src,True,True)
def clause(o,atoms,text,src):
    o.contact(CONTACT_SCENE,(7,CTX,len(atoms),*atoms),src,True,True);o.contact(CONTACT_SURFACE,u(text),src,True,True)
def see(o,entity,src,independent=True):
    return o.contact(CONTACT_WORLD_STATE,(entity,),src,True,independent)
def speak(o,atoms,src,p=P):
    partner(o,p);o.contact(CONTACT_SCENE,(7,CTX,len(atoms),*atoms),src,True,True);return o.tick()

def train(o):
    feat(o,ALICE,FA,8001);feat(o,BOB,(21,22,23,24),8002)
    feat(o,INSPECT,(35,36),8015);feat(o,SENSOR,FS,8016);feat(o,TEST,(33,34),8013);feat(o,VALVE,(43,44),8014)
    for e,s in ((ALICE,'alice'),(BOB,'bob'),(INSPECT,'inspects'),(SENSOR,'sensor'),(TEST,'tests'),(VALVE,'valve')):
        name(o,e,s,10000+e);name(o,e,s,11000+e)
    clause(o,(ALICE,INSPECT,SENSOR),'alice inspects the sensor.',30003);clause(o,(ALICE,INSPECT,SENSOR),'alice inspects the sensor.',30004)
    clause(o,(BOB,TEST,SENSOR),'bob tests the sensor.',30007);clause(o,(BOB,TEST,SENSOR),'bob tests the sensor.',30008)
    clause(o,(BOB,TEST,VALVE),'bob tests the valve.',30005);clause(o,(BOB,TEST,VALVE),'bob tests the valve.',30006)
    assert o.language.template(CTX,3) is not None

def establish(o,p,atoms,src):
    a=speak(o,atoms,src,p)
    if not isinstance(a,ActionV2):raise AssertionError(('establish',a))
    o.contact(CONTACT_CONSEQUENCE,(a.ticket,1),p,True,True);return a

def teach_it(o,p,feature,src):
    partner(o,p)
    for i in range(2):
        o.contact(CONTACT_SCENE,(7,0,1,feature),src+i,True,True);o.contact(CONTACT_SURFACE,u('it'),src+100+i,True,True)
    assert o.language.form(feature,(COND_REINSTATED,),require_conditioned=True)==u('it')

def teach_compact(o,p,atoms,text,src):
    partner(o,p)
    o.contact(CONTACT_SCENE,(7,CTX,len(atoms),*atoms),src,True,True)
    o.contact(CONTACT_SURFACE,u(text),src,True,True)

def main():
    t=time.perf_counter();checks={};spec=PopulationSpecV1(65536,2,4,42,8)
    o=ReferenceOrganismV2(spec);train(o)
    establish(o,P,(BOB,TEST,SENSOR),41001)
    teach_it(o,P,SENSOR,41101)
    teach_compact(o,P,(ALICE,INSPECT,SENSOR),'alice inspects it.',43101)
    teach_compact(o,P,(ALICE,INSPECT,SENSOR),'alice inspects it.',43102)
    checks['talking_is_not_seeing']=o._world_revisions.row_count==0 and ALICE not in o._world_marked_entities() and SENSOR not in o._world_marked_entities()
    see(o,SENSOR,SEE,True)
    checks['seeing_writes_world_not_language']=SENSOR in o._world_marked_entities() and ALICE not in o._world_marked_entities()
    cp=copy.deepcopy(o.checkpoint())
    c_o=ReferenceOrganismV2.restore(copy.deepcopy(cp));c=speak(c_o,(ALICE,INSPECT,SENSOR),44001,P);cw=c_o._world_state_occurrences(CTX,(ALICE,INSPECT,SENSOR))
    checks['compact_surface_keeps_seen_world']=isinstance(c,ActionV2) and c.payload==u('alice inspects it.') and {row[0] for row in cw}=={SENSOR} and all(row[3] in c.contributors for row in cw)
    fresh=ReferenceOrganismV2.restore(copy.deepcopy(cp));f=speak(fresh,(ALICE,INSPECT,SENSOR),44002,P2);fw=fresh._world_state_occurrences(CTX,(ALICE,INSPECT,SENSOR))
    checks['fresh_partner_explicit_still_recruits_world']=isinstance(f,ActionV2) and f.payload==u('alice inspects the sensor.') and {row[0] for row in fw}=={SENSOR} and all(row[3] in f.contributors for row in fw)
    bob_o=ReferenceOrganismV2.restore(copy.deepcopy(cp));bob=speak(bob_o,(BOB,TEST,VALVE),45001,P)
    checks['unseen_referent_speaks_without_world']=isinstance(bob,ActionV2) and bob.payload==u('bob tests the valve.') and not bob_o._world_state_occurrences(CTX,(BOB,TEST,VALVE))
    yoked=ReferenceOrganismV2(spec);train(yoked);see(yoked,SENSOR,SEE,False)
    checks['yoked_see_cannot_write_world']=yoked._world_revisions.row_count==0
    cut=ReferenceOrganismV2.restore(copy.deepcopy(cp));cut.contact(CONTACT_WITHDRAW_SOURCE,(SEE,),88002,True,True);ca=speak(cut,(ALICE,INSPECT,SENSOR),46001,P)
    checks['world_withdrawal_keeps_compact_drops_world']=isinstance(ca,ActionV2) and ca.payload==u('alice inspects it.') and not cut._world_state_occurrences(CTX,(ALICE,INSPECT,SENSOR))
    ALICE2=501;a2=ReferenceOrganismV2.restore(copy.deepcopy(cp));feat(a2,ALICE2,(11,12,13,99),8003)
    name(a2,ALICE2,'alice',12001);name(a2,ALICE2,'alice',12002)
    first=speak(a2,(ALICE,INSPECT,SENSOR),53001,P)
    if isinstance(first,ActionV2):a2.contact(CONTACT_CONSEQUENCE,(first.ticket,-1),P,True,True)
    leak=speak(a2,(ALICE2,INSPECT,SENSOR),53002,P)
    checks['punished_compact_silences_overlapping_named_subject']=isinstance(first,ActionV2) and first.payload==u('alice inspects it.') and (leak is None or not leak.form_slots)
    both=ReferenceOrganismV2(spec)
    feat(both,ALICE,FA,8001);feat(both,BOB,(21,22,23,24),8002)
    feat(both,INSPECT,(35,36),8015);feat(both,SENSOR,FS,8016);feat(both,TEST,(33,34),8013);feat(both,VALVE,(43,44),8014)
    for src,alice,inspect,sensor,bob,test,valve in (
        (EN,'alice','inspects','sensor','bob','tests','valve'),
        (EN2,'alice','inspects','sensor','bob','tests','valve'),
        (DE,'Alice','prueft','Sensor','Bob','testet','Ventil'),
        (DE2,'Alice','prueft','Sensor','Bob','testet','Ventil')):
        name(both,ALICE,alice,src);name(both,INSPECT,inspect,src);name(both,SENSOR,sensor,src)
        name(both,BOB,bob,src);name(both,TEST,test,src);name(both,VALVE,valve,src)
    clause(both,(ALICE,INSPECT,SENSOR),'alice inspects sensor.',EN);clause(both,(ALICE,INSPECT,SENSOR),'alice inspects sensor.',EN2)
    clause(both,(ALICE,INSPECT,SENSOR),'Alice prueft Sensor.',DE);clause(both,(ALICE,INSPECT,SENSOR),'Alice prueft Sensor.',DE2)
    clause(both,(BOB,TEST,SENSOR),'bob tests sensor.',EN);clause(both,(BOB,TEST,SENSOR),'bob tests sensor.',EN2)
    clause(both,(BOB,TEST,SENSOR),'Bob testet Sensor.',DE);clause(both,(BOB,TEST,SENSOR),'Bob testet Sensor.',DE2)
    establish(both,EN,(BOB,TEST,SENSOR),41021);establish(both,DE,(BOB,TEST,SENSOR),41031)
    raw=copy.deepcopy(both.checkpoint())
    partner(both,EN)
    for src in (EN,EN2):
        both.contact(CONTACT_SCENE,(7,0,1,SENSOR),src,True,True);both.contact(CONTACT_SURFACE,u('it'),src,True,True)
    partner(both,DE)
    for src in (DE,DE2):
        both.contact(CONTACT_SCENE,(7,0,1,SENSOR),src,True,True);both.contact(CONTACT_SURFACE,u('es'),src,True,True)
    teach_compact(both,EN,(ALICE,INSPECT,SENSOR),'alice inspects it.',EN)
    teach_compact(both,EN,(ALICE,INSPECT,SENSOR),'alice inspects it.',EN2)
    teach_compact(both,DE,(ALICE,INSPECT,SENSOR),'Alice prueft es.',DE)
    teach_compact(both,DE,(ALICE,INSPECT,SENSOR),'Alice prueft es.',DE2)
    see(both,SENSOR,SEE,True);buried=copy.deepcopy(both.checkpoint())
    en_o=ReferenceOrganismV2.restore(copy.deepcopy(buried));en=speak(en_o,(ALICE,INSPECT,SENSOR),44011,EN);ew=en_o._world_state_occurrences(CTX,(ALICE,INSPECT,SENSOR))
    de_o=ReferenceOrganismV2.restore(copy.deepcopy(buried));de=speak(de_o,(ALICE,INSPECT,SENSOR),44012,DE);dw=de_o._world_state_occurrences(CTX,(ALICE,INSPECT,SENSOR))
    checks['english_partner_keeps_compact_it']=isinstance(en,ActionV2) and en.payload==u('alice inspects it.') and bool(en.form_slots)
    checks['german_partner_keeps_compact_es']=isinstance(de,ActionV2) and de.payload==u('Alice prueft es.') and bool(de.form_slots)
    checks['both_compact_surfaces_recruit_same_seen_world']=isinstance(en,ActionV2) and isinstance(de,ActionV2) and {row[0] for row in ew}=={row[0] for row in dw}=={SENSOR} and all(row[3] in en.contributors for row in ew) and all(row[3] in de.contributors for row in dw)
    NEW=404
    en_n=ReferenceOrganismV2.restore(copy.deepcopy(buried));feat(en_n,NEW,(45,46,47,99),8020);en_ov=speak(en_n,(ALICE,INSPECT,NEW),44021,EN)
    de_n=ReferenceOrganismV2.restore(copy.deepcopy(buried));feat(de_n,NEW,(45,46,47,99),8020);de_ov=speak(de_n,(ALICE,INSPECT,NEW),44022,DE)
    checks['english_partner_compacts_overlapping_identity']=isinstance(en_ov,ActionV2) and en_ov.payload==u('alice inspects it.') and bool(en_ov.form_slots)
    checks['german_partner_compacts_overlapping_identity']=isinstance(de_ov,ActionV2) and de_ov.payload==u('Alice prueft es.') and bool(de_ov.form_slots)
    def punished_explicit(p,text,src):
        x=ReferenceOrganismV2.restore(copy.deepcopy(raw));see(x,SENSOR,SEE,True)
        first=speak(x,(ALICE,INSPECT,SENSOR),src,p)
        if not isinstance(first,ActionV2) or first.payload!=u(text):return first,None
        x.contact(CONTACT_CONSEQUENCE,(first.ticket,-1),p,True,True)
        feat(x,NEW,(45,46,47,99),src+2);return first,speak(x,(ALICE,INSPECT,NEW),src+3,p)
    en_p,en_leak=punished_explicit(EN,'alice inspects sensor.',45011);de_p,de_leak=punished_explicit(DE,'Alice prueft Sensor.',45012)
    checks['english_punished_name_silences_overlapping']=isinstance(en_p,ActionV2) and en_leak is None
    checks['german_punished_name_silences_overlapping']=isinstance(de_p,ActionV2) and de_leak is None
    def punished_own_form(p,form,text,srcs,src):
        x=ReferenceOrganismV2.restore(copy.deepcopy(buried));first=speak(x,(ALICE,INSPECT,SENSOR),src,p)
        if not isinstance(first,ActionV2) or first.payload!=u(text):return first,None
        x.contact(CONTACT_CONSEQUENCE,(first.ticket,-1),p,True,True)
        feat(x,NEW,(45,46,47,99),src+2);partner(x,p)
        for s in srcs:
            x.contact(CONTACT_SCENE,(7,0,1,NEW),s,True,True);x.contact(CONTACT_SURFACE,u(form),s,True,True)
        return first,speak(x,(ALICE,INSPECT,NEW),src+3,p)
    en_f,en_fleak=punished_own_form(EN,'it','alice inspects it.',(EN,EN2),47011)
    de_f,de_fleak=punished_own_form(DE,'es','Alice prueft es.',(DE,DE2),47012)
    checks['english_new_form_reacquires_after_exact_negative']=isinstance(en_f,ActionV2) and isinstance(en_fleak,ActionV2) and en_fleak.payload==u('alice inspects it.') and bool(en_fleak.form_slots)
    checks['german_new_form_reacquires_after_exact_negative']=isinstance(de_f,ActionV2) and isinstance(de_fleak,ActionV2) and de_fleak.payload==u('Alice prueft es.') and bool(de_fleak.form_slots)
    checks['no_pronoun_or_imagine_opcode']=not hasattr(o,'pronoun') and not hasattr(o,'imagine') and not hasattr(o,'translate')
    checks['no_stored_media']=all(k not in cp for k in ('image','audio','png','wav'))
    result={'schema':'0x1.reference-organism-world-compact.v1','pass':all(checks.values()),'checks':checks,'runtime_llm':False,'graph_flip':False,'compact':bytes(c.payload).decode() if isinstance(c,ActionV2) else '','claim':'COMPACT_SURFACE_KEEPS_SEEN_WORLD_TALKING_IS_NOT_SEEING_REFERENCE_ONLY','elapsed_ms':round((time.perf_counter()-t)*1000,3)}
    print('FOUNDRY_REFERENCE_ORGANISM_WORLD_COMPACT '+('GREEN' if result['pass'] else 'RED')+' talking_not_seeing=1 compact_keeps_world=1 pronoun=0 media=0')
    print(json.dumps(result,indent=2,sort_keys=True));raise SystemExit(0 if result['pass'] else 1)
if __name__=='__main__':main()
