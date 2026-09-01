#!/usr/bin/env python3
from __future__ import annotations
import copy,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
from reference_organism_v2 import *
from reference_population_v1 import PopulationSpecV1

NAME=100;CTX=9001;PARTNER=9701
M={101:'careful',102:'quiet',201:'engineer',202:'technician',301:'tests',302:'inspects',401:'sensor',402:'valve'}
def u(s):return tuple(s.encode())
def scene(o,atoms,source,context=CTX):return o.contact(CONTACT_SCENE,(7,context,len(atoms),*atoms),source,True,True)
def surface(o,text,source):return o.contact(CONTACT_SURFACE,u(text),source,True,True)
def partner(o,p=PARTNER):return o.contact(CONTACT_PARTNER_CONTEXT,(1,7,p),70000+p,True,True)
def settle(o,a,effect,independent=True,p=PARTNER):return o.contact(CONTACT_CONSEQUENCE,(a.ticket,effect),p,True,independent)
def config(a):return tuple(row[:3] for row in a.selection_occurrences)
def config_value(o,ctx,a):return o._selection_configuration_evidence(ctx,config(a))[0]

def train_base(o):
    for f,text in M.items():
        for k in range(2):scene(o,(f,),10000+f*10+k,NAME);surface(o,text,20000+f*10+k)
    rows=((101,201,301,401),(102,202,302,402))
    texts=('the careful engineer tests the sensor.','the quiet technician inspects the valve.')
    for k,(atoms,text) in enumerate(zip(rows,texts)):
        scene(o,atoms,30001+k);surface(o,text,31001+k)
    assert o.language.template(CTX,4) is not None

def add_device_advantage(o):
    for k in range(3):scene(o,(401,),40001+k,NAME);surface(o,'device',41001+k)
    assert o.language.lexeme(401)==u('device')

def add_object_first_advantage(o):
    rows=((101,201,301,401),(102,202,302,402),(101,202,302,401))
    texts=('the device, the careful engineer tests.','the valve, the quiet technician inspects.','the device, the careful technician inspects.')
    for k,(atoms,text) in enumerate(zip(rows,texts)):
        scene(o,atoms,42001+k);surface(o,text,43001+k)
    assert len(o.language.template_candidates(CTX,4))==2

def staged():
    o=ReferenceOrganismV2(PopulationSpecV1(32768,2,4,42,8));train_base(o);add_device_advantage(o);add_object_first_advantage(o)
    prior=(101,201,301,401);current=(102,202,301,401)
    partner(o);sid1=scene(o,prior,50001);a1=o.tick();assert a1 is not None;settle(o,a1,1)
    sid2=scene(o,current,50002);ctx=o._selection_preference_context(o.current_scene);assert ctx!=0
    return o,ctx,sid1,sid2,current

def equalize_developmental_support(o):
    # Plain developmental evidence is learned outside an active partner/common-ground
    # episode; otherwise the same contact correctly becomes conditioned evidence.
    o.contact(CONTACT_PARTNER_CONTEXT,(0,0,0),50999,True,True)
    scene(o,(401,),51001,NAME);surface(o,'sensor',52001)
    scene(o,(101,202,302,402),51002);surface(o,'the careful technician inspects the valve.',52002)
    partner(o)

def ids(o):
    lex={bytes(units).decode():o.language.lexeme_identity(401,units) for _support,units,_sources in o.language.lexeme_candidates(401)}
    templates={}
    for row in o.language.template_candidates(CTX,4):
        # identify by first literal scaffold, enough for this assay only
        rendered=o.language.render_template(row,(u('careful'),u('engineer'),u('tests'),u('device')))
        templates[bytes(rendered).decode()]=int(row.identity[:15],16)
    return lex,templates

def main():
    t=time.perf_counter();checks={};o,ctx,_sid1,_sid2,current=staged();lex,templates=ids(o)
    initial=o.tick();assert initial is not None
    checks['developmental_support_selects_initial_candidates']=bytes(initial.payload).decode()=='the device, the quiet technician tests.' and lex['device'] in initial.lexical_identities
    checks['actual_selection_occurrences_participate']=len(initial.selection_occurrences)==5 and all(row[3] in initial.contributors for row in initial.selection_occurrences)
    positive=settle(o,initial,1,True)
    checks['independent_consequence_credits_selection_network']=positive.get('selection_credit',0)>0 and positive.get('selection_revisions',0)>0 and positive.get('selection_network_updates',0)==1
    chosen_template=initial.template_identity;chosen_lexeme=lex['device']
    checks['exact_network_preference_separate_from_developmental_support']=config_value(o,ctx,initial)==1

    equalize_developmental_support(o)
    lex_rows=o.language.lexeme_candidates(401);template_rows=o.language.template_candidates(CTX,4)
    checks['developmental_evidence_equalized']=sorted(x[0] for x in lex_rows)==[3,3] and sorted(x.support for x in template_rows)==[3,3]
    base_t,talts=o._select_explicit_template(CTX,4,0);base_l,_,lalts=o._select_explicit_lexeme(401,0)
    checks['equal_support_refuses_without_lived_context']=base_t is None and talts==2 and base_l is None and lalts==2
    selected,selected_template,selected_lexemes,_=o._realize_explicit_selected(CTX,current,ctx)
    checks['lived_network_breaks_joint_tie']=selected is not None and selected_template is not None and int(selected_template.identity[:15],16)==chosen_template and chosen_lexeme in selected_lexemes

    # New actual use of the credited alternatives, followed by negative consequence, reopens both ties.
    third_event=current;sid3=scene(o,third_event,53001);third=o.tick();checks['credited_selection_network_drives_later_actual_expression']=third is not None and third.template_identity==chosen_template and chosen_lexeme in third.lexical_identities and third.selection_context==ctx
    negative=settle(o,third,-1,True);re_surface,_,_,_=o._realize_explicit_selected(CTX,current,ctx)
    checks['independent_negative_reopens_joint_tie']=negative.get('selection_network_updates',0)==1 and config_value(o,ctx,third)==0 and re_surface is None

    cp=o.checkpoint();r=ReferenceOrganismV2.restore(copy.deepcopy(cp));checks['checkpoint_preserves_competition_state']=r.digest()==o.digest() and config_value(r,ctx,third)==0

    # Same initial path with a yoked/non-independent positive return cannot mint context preference.
    y,yctx,*_=staged();ya=y.tick();yr=settle(y,ya,1,False);checks['nonindependent_return_no_network_preference']=yr.get('selection_network_updates',0)==0 and config_value(y,yctx,ya)==0

    # Preference rows are source-qualified and source withdrawal makes them inert.
    z,zctx,*_=staged();za=z.tick();settle(z,za,1,True);checks['source_qualified_network_preference_exists']=config_value(z,zctx,za)==1;z.contact(CONTACT_WITHDRAW_SOURCE,(PARTNER,),99999,True,True);checks['source_withdrawal_removes_network_preference_authority']=config_value(z,zctx,za)==0

    # Competing current occurrences are not a queue.  Hold learned language
    # competence fixed, vary only lived consequence, then reverse insertion.
    A=(101,201,301,401);B=(102,202,302,402)
    c=ReferenceOrganismV2(PopulationSpecV1(32768,2,4,42,8));train_base(c);partner(c)
    scene(c,A,80001);ca=c.tick();assert ca is not None;settle(c,ca,-1)
    scene(c,B,80002);cb=c.tick();assert cb is not None;settle(c,cb,1)
    learned=copy.deepcopy(c.checkpoint());left=ReferenceOrganismV2.restore(learned);right=ReferenceOrganismV2.restore(learned)
    scene(left,A,81001);left_b=scene(left,B,81002);la=left.tick()
    right_b=scene(right,B,82001);scene(right,A,82002);ra=right.tick()
    checks['consequence_supported_scene_wins_both_insertion_orders']=(
        isinstance(la,ActionV2) and isinstance(ra,ActionV2)
        and la.scene_identity==left_b and ra.scene_identity==right_b
        and la.payload==ra.payload==u('the quiet technician inspects the valve.'))

    equal=ReferenceOrganismV2(PopulationSpecV1(32768,2,4,42,8));train_base(equal);partner(equal)
    equal_cp=copy.deepcopy(equal.checkpoint());ab=ReferenceOrganismV2.restore(equal_cp);ba=ReferenceOrganismV2.restore(equal_cp)
    scene(ab,A,83001);scene(ab,B,83002);scene(ba,B,84001);scene(ba,A,84002)
    checks['equal_unearned_current_scenes_remain_unresolved']=ab.tick() is None and ba.tick() is None
    checks['visible_discussion_improvement']=(
        checks['consequence_supported_scene_wins_both_insertion_orders']
        and checks['equal_unearned_current_scenes_remain_unresolved'])
    checks['no_prompt_answer_think_speak_api']=all(not hasattr(o,n) for n in ('prompt','answer','think','speak','enqueue_goal'))
    result={'schema':'0x1.reference-organism-competition.v3','pass':all(checks.values()),'checks':checks,'context':ctx,'selection_occurrences':len(initial.selection_occurrences),'configuration_revision_rows':len(o.selection_configuration_revisions),'lexical_alternatives':len(lex_rows),'construction_alternatives':len(template_rows),'claim':'CAUSAL_CONTEXT_SELECTION_NETWORK_COMPETITION_REFERENCE_NOT_DIRECT_PARITY_OR_HUMAN_LANGUAGE','elapsed_ms':round((time.perf_counter()-t)*1000,3)}
    print('FOUNDRY_REFERENCE_ORGANISM_COMPETITION '+('GREEN' if result['pass'] else 'RED')+' lexical=2 constructions=2 selection_occurrences=1 causal_preference=1 prompt=0 llm=0')
    print(json.dumps(result,indent=2,sort_keys=True));raise SystemExit(0 if result['pass'] else 1)
if __name__=='__main__':main()
