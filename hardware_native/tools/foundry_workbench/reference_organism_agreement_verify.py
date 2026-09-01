#!/usr/bin/env python3
from __future__ import annotations
import copy,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
from reference_organism_v2 import *
from reference_population_v1 import PopulationSpecV1

CLAUSE=8801;SING,PLUR=7001,7002;P=9901
CAT,SEE,DOG=101,102,103
HCAT,HSEE,HDOG=201,202,203
MCAT,MSEE,MDOG=301,302,303

def u(s):return tuple(s.encode())
def partner(o,p=P):o.contact(CONTACT_PARTNER_CONTEXT,(1,7,p),70000+p,True,True)
def bind(o,entity,ident,cond,src):
    payload=(entity,len(ident),*ident)+(cond,) if cond else (entity,len(ident),*ident)
    o.contact(CONTACT_ENTITY_FEATURES,payload,src,True,True)
def name(o,entity,text,src):
    o.contact(CONTACT_SCENE,(7,0,1,entity),src,True,True);o.contact(CONTACT_SURFACE,u(text),src+1000,True,True)
def form(o,entity,cond,text,src):
    bind(o,entity,(entity,),cond,src);name(o,entity,text,src)
def clause(o,atoms,conds,text,src):
    for atom,cond in zip(atoms,conds):bind(o,atom,(atom,),cond,src+atom)
    o.contact(CONTACT_SCENE,(7,CLAUSE,len(atoms),*atoms),src,True,True);o.contact(CONTACT_SURFACE,u(text),src+2000,True,True)

def train(o):
    for e,s in ((CAT,'cat'),(SEE,'see'),(DOG,'dog'),(HCAT,'cat'),(HSEE,'see'),(HDOG,'dog'),(MCAT,'cat'),(MSEE,'see'),(MDOG,'dog')):
        name(o,e,s,10000+e);name(o,e,s,11000+e)
    o.contact(CONTACT_SCENE,(7,CLAUSE,3,CAT,SEE,DOG),29001,True,True);o.contact(CONTACT_SURFACE,u('the cat see the dog.'),31001,True,True)
    o.contact(CONTACT_SCENE,(7,CLAUSE,3,CAT,SEE,DOG),29002,True,True);o.contact(CONTACT_SURFACE,u('the cat see the dog.'),31002,True,True)
    for src,e,c,text in ((21001,CAT,SING,'cat'),(21002,CAT,SING,'cat'),(21003,CAT,PLUR,'cats'),(21004,CAT,PLUR,'cats'),
                         (22001,SEE,SING,'sees'),(22002,SEE,SING,'sees'),(22003,SEE,PLUR,'see'),(22004,SEE,PLUR,'see'),
                         (23001,DOG,SING,'dog'),(23002,DOG,SING,'dog'),(23003,DOG,PLUR,'dogs'),(23004,DOG,PLUR,'dogs'),
                         (24001,HCAT,SING,'cat'),(24002,HCAT,SING,'cat'),(24003,HCAT,PLUR,'cats'),(24004,HCAT,PLUR,'cats'),
                         (25001,HSEE,SING,'sees'),(25002,HSEE,SING,'sees'),(25003,HSEE,PLUR,'see'),(25004,HSEE,PLUR,'see'),
                         (26001,HDOG,SING,'dog'),(26002,HDOG,SING,'dog'),(26003,HDOG,PLUR,'dogs'),(26004,HDOG,PLUR,'dogs')):
        form(o,e,c,text,src)
    clause(o,(CAT,SEE,DOG),(PLUR,PLUR,SING),'the cats see the dog.',30001)
    clause(o,(CAT,SEE,DOG),(SING,SING,PLUR),'the cat sees the dogs.',30002)
    assert o.language.template(CLAUSE,3) is not None

def main():
    t=time.perf_counter();checks={};o=ReferenceOrganismV2(PopulationSpecV1(65536,2,4,42,8));train(o)
    checks['compatibility_pattern_learned']=o.language.compatible(CLAUSE,(PLUR,PLUR,SING)) and o.language.compatible(CLAUSE,(SING,SING,PLUR)) and not o.language.compatible(CLAUSE,(PLUR,SING,SING))
    partner(o)
    for e,c in ((HCAT,PLUR),(HSEE,PLUR),(HDOG,SING)):bind(o,e,(e,),c,40000+e)
    o.contact(CONTACT_SCENE,(7,CLAUSE,3,HCAT,HSEE,HDOG),41001,True,True);agree=o.tick()
    checks['heldout_agreeing_ports_emit']=isinstance(agree,ActionV2) and agree.payload==u('the cats see the dog.')
    checks['agreement_forms_join_selection_network']=agree is not None and any(k==PREF_FORM for k,_,_,_ in agree.selection_occurrences)
    if agree:o.contact(CONTACT_CONSEQUENCE,(agree.ticket,1),P,True,True)
    cp=copy.deepcopy(o.checkpoint())
    for e,c in ((MCAT,PLUR),(MSEE,SING),(MDOG,SING)):bind(o,e,(e,),c,42000+e)
    o.contact(CONTACT_SCENE,(7,CLAUSE,3,MCAT,MSEE,MDOG),43001,True,True);mismatch=o.tick()
    checks['heldout_mismatch_stays_silent']=mismatch is None
    checks['unconditioned_lexeme_does_not_bypass_agreement']=o.language.lexeme(MCAT)==u('cat')
    r=ReferenceOrganismV2.restore(copy.deepcopy(cp));partner(r)
    for e,c in ((HCAT,SING),(HSEE,SING),(HDOG,PLUR)):bind(r,e,(e,),c,44000+e)
    r.contact(CONTACT_SCENE,(7,CLAUSE,3,HCAT,HSEE,HDOG),45001,True,True);rb=r.tick()
    checks['checkpoint_other_agreeing_pattern']=isinstance(rb,ActionV2) and rb.payload==u('the cat sees the dogs.')
    cut=ReferenceOrganismV2.restore(copy.deepcopy(cp));cut.contact(CONTACT_WITHDRAW_SOURCE,(32002,),88001,True,True)
    partner(cut)
    for e,c in ((HCAT,SING),(HSEE,SING),(HDOG,PLUR)):bind(cut,e,(e,),c,46000+e)
    cut.contact(CONTACT_SCENE,(7,CLAUSE,3,HCAT,HSEE,HDOG),47001,True,True)
    checks['compat_source_withdrawal_silences']=cut.tick() is None
    checks['no_number_opcode']=not hasattr(o,'plural') and not hasattr(o,'agree') and not hasattr(o,'number')
    EN,EN2,DE,DE2,NEUTRAL=1101,1102,2101,2102,9999
    def partner_form(o,entity,cond,text,src):
        bind(o,entity,(entity,),cond,src)
        o.contact(CONTACT_SCENE,(7,0,1,entity),src,True,True)
        o.contact(CONTACT_SURFACE,u(text),src,True,True)
    both=ReferenceOrganismV2(PopulationSpecV1(65536,2,4,42,8))
    for src,cat,see,dog in ((EN,'cat','see','dog'),(EN2,'cat','see','dog'),(DE,'Katze','sehen','Hund'),(DE2,'Katze','sehen','Hund')):
        for e,text in ((CAT,cat),(SEE,see),(DOG,dog),(HCAT,cat),(HSEE,see),(HDOG,dog),(MCAT,cat),(MSEE,see),(MDOG,dog)):
            name(both,e,text,src)
    for src,text in ((EN,'cat see dog.'),(EN2,'cat see dog.'),(DE,'Katze sehen Hund.'),(DE2,'Katze sehen Hund.')):
        both.contact(CONTACT_SCENE,(7,CLAUSE,3,CAT,SEE,DOG),src,True,True);both.contact(CONTACT_SURFACE,u(text),src,True,True)
    for src,rows in (
        (EN,((CAT,SING,'cat'),(CAT,PLUR,'cats'),(SEE,SING,'sees'),(SEE,PLUR,'see'),(DOG,SING,'dog'),(DOG,PLUR,'dogs'),
             (HCAT,SING,'cat'),(HCAT,PLUR,'cats'),(HSEE,SING,'sees'),(HSEE,PLUR,'see'),(HDOG,SING,'dog'),(HDOG,PLUR,'dogs'))),
        (EN2,((CAT,SING,'cat'),(CAT,PLUR,'cats'),(SEE,SING,'sees'),(SEE,PLUR,'see'),(DOG,SING,'dog'),(DOG,PLUR,'dogs'),
              (HCAT,SING,'cat'),(HCAT,PLUR,'cats'),(HSEE,SING,'sees'),(HSEE,PLUR,'see'),(HDOG,SING,'dog'),(HDOG,PLUR,'dogs'))),
        (DE,((CAT,SING,'Katze'),(CAT,PLUR,'Katzen'),(SEE,SING,'sieht'),(SEE,PLUR,'sehen'),(DOG,SING,'Hund'),(DOG,PLUR,'Hunde'),
             (HCAT,SING,'Katze'),(HCAT,PLUR,'Katzen'),(HSEE,SING,'sieht'),(HSEE,PLUR,'sehen'),(HDOG,SING,'Hund'),(HDOG,PLUR,'Hunde'))),
        (DE2,((CAT,SING,'Katze'),(CAT,PLUR,'Katzen'),(SEE,SING,'sieht'),(SEE,PLUR,'sehen'),(DOG,SING,'Hund'),(DOG,PLUR,'Hunde'),
              (HCAT,SING,'Katze'),(HCAT,PLUR,'Katzen'),(HSEE,SING,'sieht'),(HSEE,PLUR,'sehen'),(HDOG,SING,'Hund'),(HDOG,PLUR,'Hunde')))):
        for e,c,text in rows:partner_form(both,e,c,text,src)
    for src,atoms,conds,text in (
        (EN,(CAT,SEE,DOG),(PLUR,PLUR,SING),'cats see dog.'),(EN2,(CAT,SEE,DOG),(PLUR,PLUR,SING),'cats see dog.'),
        (DE,(CAT,SEE,DOG),(PLUR,PLUR,SING),'Katzen sehen Hund.'),(DE2,(CAT,SEE,DOG),(PLUR,PLUR,SING),'Katzen sehen Hund.'),
        (EN,(CAT,SEE,DOG),(SING,SING,PLUR),'cat sees dogs.'),(EN2,(CAT,SEE,DOG),(SING,SING,PLUR),'cat sees dogs.'),
        (DE,(CAT,SEE,DOG),(SING,SING,PLUR),'Katze sieht Hunde.'),(DE2,(CAT,SEE,DOG),(SING,SING,PLUR),'Katze sieht Hunde.')):
        clause(both,atoms,conds,text,src)
    buried=copy.deepcopy(both.checkpoint())
    def heldout(p,conds,src):
        x=ReferenceOrganismV2.restore(copy.deepcopy(buried));partner(x,p)
        for e,c in zip((HCAT,HSEE,HDOG),conds):bind(x,e,(e,),c,src+e)
        x.contact(CONTACT_SCENE,(7,CLAUSE,3,HCAT,HSEE,HDOG),src,True,True);return x.tick()
    en=heldout(EN,(PLUR,PLUR,SING),41011);de=heldout(DE,(PLUR,PLUR,SING),41012)
    checks['english_partner_keeps_agreeing_forms']=isinstance(en,ActionV2) and en.payload==u('cats see dog.') and any(k==PREF_FORM for k,_,_,_ in en.selection_occurrences)
    checks['german_partner_keeps_agreeing_forms']=isinstance(de,ActionV2) and de.payload==u('Katzen sehen Hund.') and any(k==PREF_FORM for k,_,_,_ in de.selection_occurrences)
    checks['neutral_partner_does_not_bypass_with_stem']=heldout(NEUTRAL,(PLUR,PLUR,SING),41013) is None
    checks['bilingual_mismatch_stays_silent']=heldout(EN,(PLUR,SING,SING),41014) is None
    FD,NEW=(45,46,47,48),404
    def teach_compact_agree(p,compact,clause_text,src,dog_features=(DOG,)):
        x=ReferenceOrganismV2.restore(copy.deepcopy(buried));partner(x,p)
        for e,c in ((CAT,PLUR),(SEE,PLUR)):bind(x,e,(e,),c,src+e)
        bind(x,DOG,dog_features,SING,src+DOG)
        x.contact(CONTACT_SCENE,(7,CLAUSE,3,CAT,SEE,DOG),src,True,True);prior=x.tick()
        if not isinstance(prior,ActionV2):return None
        x.contact(CONTACT_CONSEQUENCE,(prior.ticket,1),p,True,True)
        pair=(EN,EN2) if p==EN else (DE,DE2)
        for repetition in range(3):
            for src_p in pair:
                contact_source=src_p+repetition*100000
                x.contact(CONTACT_SCENE,(7,0,1,DOG),contact_source,True,True);x.contact(CONTACT_SURFACE,u(compact),contact_source,True,True)
        bind(x,DOG,dog_features,SING,src+15)
        for repetition in range(3):
            for src_p in pair:
                contact_source=src_p+repetition*100000
                partner(x,p);x.contact(CONTACT_SCENE,(7,CLAUSE,3,CAT,SEE,DOG),contact_source,True,True);x.contact(CONTACT_SURFACE,u(clause_text),contact_source,True,True)
        x.contact(CONTACT_WORLD_STATE,(DOG,),77001,True,True);return x
    def emit_agree(x,p,atoms,src):
        if x is None:return None,()
        partner(x,p);x.contact(CONTACT_SCENE,(7,CLAUSE,3,*atoms),src,True,True);a=x.tick()
        return a,x._world_state_occurrences(CLAUSE,atoms)
    en_c,ew=emit_agree(teach_compact_agree(EN,'it','cats see it.',51000),EN,(CAT,SEE,DOG),51030)
    de_c,dw=emit_agree(teach_compact_agree(DE,'es','Katzen sehen es.',52000),DE,(CAT,SEE,DOG),52030)
    checks['english_reinstated_object_stays_compact']=isinstance(en_c,ActionV2) and en_c.payload==u('cats see it.')
    checks['german_reinstated_object_stays_compact']=isinstance(de_c,ActionV2) and de_c.payload==u('Katzen sehen es.')
    checks['compact_agreeing_surfaces_recruit_same_seen_world']=isinstance(en_c,ActionV2) and isinstance(de_c,ActionV2) and {row[0] for row in ew}=={row[0] for row in dw}=={DOG} and all(row[3] in en_c.contributors for row in ew) and all(row[3] in de_c.contributors for row in dw)
    def overlap_emit(p,compact,clause_text,src):
        x=teach_compact_agree(p,compact,clause_text,src,FD)
        if x is None:return None,()
        x.contact(CONTACT_ENTITY_FEATURES,(NEW,4,45,46,47,99),src+2,True,True)
        partner(x,p)
        for e,c in ((CAT,PLUR),(SEE,PLUR)):bind(x,e,(e,),c,src+50+e)
        return emit_agree(x,p,(CAT,SEE,NEW),src+30)
    en_n,enw=overlap_emit(EN,'it','cats see it.',53000);de_n,dnw=overlap_emit(DE,'es','Katzen sehen es.',54000)
    checks['english_overlapping_identity_keeps_compact']=isinstance(en_n,ActionV2) and en_n.payload==u('cats see it.')
    checks['german_overlapping_identity_keeps_compact']=isinstance(de_n,ActionV2) and de_n.payload==u('Katzen sehen es.')
    checks['overlapping_compact_recruits_donor_world']=isinstance(en_n,ActionV2) and isinstance(de_n,ActionV2) and {row[0] for row in enw}=={row[0] for row in dnw}=={DOG} and all(row[3] in en_n.contributors for row in enw) and all(row[3] in de_n.contributors for row in dnw)
    def punished_overlap(p,compact,clause_text,src):
        x=teach_compact_agree(p,compact,clause_text,src,FD)
        first,_=emit_agree(x,p,(CAT,SEE,DOG),src+30)
        if not isinstance(first,ActionV2):return first,None
        x.contact(CONTACT_CONSEQUENCE,(first.ticket,-1),p,True,True)
        x.contact(CONTACT_ENTITY_FEATURES,(NEW,4,45,46,47,99),src+2,True,True)
        partner(x,p)
        for e,c in ((CAT,PLUR),(SEE,PLUR)):bind(x,e,(e,),c,src+50+e)
        return first,emit_agree(x,p,(CAT,SEE,NEW),src+32)[0]
    en_p,en_leak=punished_overlap(EN,'it','cats see it.',55000);de_p,de_leak=punished_overlap(DE,'es','Katzen sehen es.',56000)
    checks['english_punished_map_silences_overlapping']=isinstance(en_p,ActionV2) and en_p.payload==u('cats see it.') and en_leak is None
    checks['german_punished_map_silences_overlapping']=isinstance(de_p,ActionV2) and de_p.payload==u('Katzen sehen es.') and de_leak is None
    result={'schema':'0x1.reference-organism-agreement.v1','pass':all(checks.values()),'checks':checks,'agreeing':bytes(agree.payload).decode() if agree else '','runtime_llm':False,'graph_flip':False,'claim':'PORT_COMPATIBILITY_AGREEMENT_ON_ORGANISM_SELECTION_NETWORK_REFERENCE_ONLY','elapsed_ms':round((time.perf_counter()-t)*1000,3)}
    print('FOUNDRY_REFERENCE_ORGANISM_AGREEMENT '+('GREEN' if result['pass'] else 'RED')+' heldout=1 mismatch_silent=1 form_network=1 category_opcodes=0')
    print(json.dumps(result,indent=2,sort_keys=True));raise SystemExit(0 if result['pass'] else 1)
if __name__=='__main__':main()
