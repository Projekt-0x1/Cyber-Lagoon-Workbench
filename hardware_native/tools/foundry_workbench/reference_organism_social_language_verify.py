#!/usr/bin/env python3
from __future__ import annotations
import copy,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
from reference_organism_v2 import *
from reference_population_v1 import PopulationSpecV1

NAME=100;CTX=9001
M={101:'careful',102:'quiet',201:'engineer',202:'technician',301:'tests',302:'inspects',401:'sensor',402:'valve'}

def u(s):return tuple(s.encode())
def scene(o,context,atoms,source,channel=7):return o.contact(CONTACT_SCENE,(channel,context,len(atoms),*atoms),source,True,True)
def surface(o,text,source):return o.contact(CONTACT_SURFACE,u(text),source,True,True)
def partner(o,p,channel=7):return o.contact(CONTACT_PARTNER_CONTEXT,(1,channel,p),60000+p,True,True)
def settle(o,a,p,effect=1,independent=True):return o.contact(CONTACT_CONSEQUENCE,(a.ticket,effect),p,True,independent)

def train_base(o):
    for f,text in M.items():
        scene(o,NAME,(f,),10000+f);surface(o,text,20000+f)
        scene(o,NAME,(f,),11000+f);surface(o,text,21000+f)
    a=(101,201,301,401);b=(102,202,302,402)
    scene(o,CTX,a,30001);surface(o,'the careful engineer tests the sensor.',31001)
    scene(o,CTX,b,30002);surface(o,'the quiet technician inspects the valve.',31002)
    assert o.language.template(CTX,4) is not None
    return a,b

def establish(o,p,atoms,source):
    partner(o,p);scene(o,CTX,atoms,source);a=o.tick();assert isinstance(a,ActionV2);settle(o,a,p,1,True);return a

def teach_conditioned_form(o,p,feature,start_source):
    partner(o,p)
    for i in range(2):
        scene(o,NAME,(feature,),start_source+i);surface(o,'it',start_source+100+i)
    assert o.language.form(feature,(COND_REINSTATED,),require_conditioned=True)==u('it')

def main():
    t=time.perf_counter();checks={};spec=PopulationSpecV1(65536,fanout=2,sites_per_feature=4,eligibility_horizon=8)
    o=ReferenceOrganismV2(spec);sensor_prior,valve_prior=train_base(o)
    P_SENSOR,P_VALVE=9001,9002
    establish(o,P_SENSOR,sensor_prior,40001);teach_conditioned_form(o,P_SENSOR,401,41001)
    establish(o,P_VALVE,valve_prior,40002);teach_conditioned_form(o,P_VALVE,402,42001)

    # Two different referents, same shared-position geometry -> one reusable compact context.
    sensor_current=(102,202,302,401);valve_current=(101,201,301,402)
    partner(o,P_SENSOR);scene(o,CTX,sensor_current,43001);ctx1,cond1=o._surface_context(o.current_scene);surface(o,'the quiet technician inspects it.',43101)
    partner(o,P_VALVE);scene(o,CTX,valve_current,43002);ctx2,cond2=o._surface_context(o.current_scene);surface(o,'the careful engineer tests it.',43102)
    checks['same_reinstatement_geometry_same_context']=ctx1==ctx2 and ctx1!=CTX and cond1[-1]==cond2[-1]==(COND_REINSTATED,)
    checks['compact_template_learned_from_two_referents']=o.language.template(ctx1,4) is not None

    # Novel lexical combination after a shared sensor episode compresses only the carried referent.
    P3=9010;p3_prior=(101,202,301,401);establish(o,P3,p3_prior,44001);history_at_cp=dict(o.last_shared_episode_by_partner);cp=o.checkpoint()
    novel=(102,201,302,401);scene(o,CTX,novel,44002);compact=o.tick()
    checks['heldout_compact_reference']=isinstance(compact,ActionV2) and compact.payload==u('the quiet engineer inspects it.') and compact.source==P3
    settle(o,compact,P3,1,True)

    # A new partner cannot borrow P3's common ground.
    P4=9011;partner(o,P4);scene(o,CTX,novel,44003);explicit=o.tick()
    checks['fresh_partner_explicit_reference']=isinstance(explicit,ActionV2) and explicit.payload==u('the quiet engineer inspects the sensor.')
    settle(o,explicit,P4,1,True)

    # Restore at the point after P3's prior shared episode: same partner must compact identically.
    r=ReferenceOrganismV2.restore(copy.deepcopy(cp));partner(r,P3);scene(r,CTX,novel,44002);rr=r.tick()
    checks['checkpoint_preserves_partner_common_ground']=isinstance(rr,ActionV2) and rr.payload==u('the quiet engineer inspects it.') and r.last_shared_episode_by_partner==history_at_cp

    # One of two form sources removed -> conditioned route loses support and whole expression falls back explicitly.
    cut=ReferenceOrganismV2.restore(copy.deepcopy(cp));cut.language.withdraw_source(41102);partner(cut,P3);scene(cut,CTX,novel,45001);fallback=cut.tick()
    checks['conditioned_form_loss_falls_back_whole_construction']=isinstance(fallback,ActionV2) and fallback.payload==u('the quiet engineer inspects the sensor.') and b'it' not in bytes(fallback.payload)

    # Negative and endogenous/yoked returns do not mint shared partner history.
    neg=ReferenceOrganismV2(spec);train_base(neg);partner(neg,9020);scene(neg,CTX,sensor_prior,46001);na=neg.tick();settle(neg,na,9020,-1,True)
    checks['negative_return_no_common_ground']=9020 not in neg.last_shared_episode_by_partner
    yoked=ReferenceOrganismV2(spec);train_base(yoked);partner(yoked,9021);scene(yoked,CTX,sensor_prior,46002);ya=yoked.tick();settle(yoked,ya,9021,1,False)
    checks['nonindependent_return_no_common_ground']=9021 not in yoked.last_shared_episode_by_partner

    checks['no_semantic_pronoun_opcode']=not hasattr(o,'pronoun') and COND_REINSTATED not in M
    checks['runtime_llm_absent']=not hasattr(o,'prompt') and not hasattr(o,'complete')
    result={'schema':'0x1.reference-organism-social-language.v1','pass':all(checks.values()),'checks':checks,
      'compact_context':ctx1,'partner_histories':dict(sorted(o.last_shared_episode_by_partner.items())),
      'conditioned_form_evidence':2,'compact_surface':bytes(compact.payload).decode() if compact else '',
      'explicit_surface':bytes(explicit.payload).decode() if explicit else '',
      'claim':'PARTNER_CONDITIONED_LEARNED_REFERENCE_NOT_THEORY_OF_MIND_OR_HUMAN_LANGUAGE','elapsed_ms':round((time.perf_counter()-t)*1000,3)}
    print('FOUNDRY_REFERENCE_ORGANISM_SOCIAL_LANGUAGE '+('GREEN' if result['pass'] else 'RED')+' partner_history=1 conditioned_form=1 compact_reference=1 prompt=0 llm=0')
    print(json.dumps(result,indent=2,sort_keys=True));raise SystemExit(0 if result['pass'] else 1)
if __name__=='__main__':main()
