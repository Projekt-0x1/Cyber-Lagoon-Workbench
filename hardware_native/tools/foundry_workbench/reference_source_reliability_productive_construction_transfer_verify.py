#!/usr/bin/env python3
"""Destructive transfer of learned source reliability across a new productive form."""
from __future__ import annotations
import copy,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
from reference_language_mastery_adult_v1 import AdultStateV1,LanguageMasteryAdultV1
from reference_language_mastery_contact_adapter_v1 import CONTACT_UTTERANCE,LanguageMasteryContactAdapterV1
from reference_predictive_credit_profile_v1 import Q

language_phenotype_improved=True
future_update_authority_preserved=True
CLAUSE=0x6A01; A1,A2,G1,G2,V1,V2,O1,O2=101,102,201,202,301,302,401,402
P_ACCEPT,P_VERIFY=0x6B01,0x6B02; REL,UNREL,NOVEL=0x6C01,0x6C02,0x6C03
X=(A1,G1,V1,O1);Y=(A2,G2,V2,O2)
BASE_X=b'the careful engineer tests the sensor.';BASE_Y=b'the quiet technician inspects the valve.'
BASE_AMB=b'the careful engineer tests the probe.';BASE_ANS=BASE_X
ALT_X=b'after review, the sensor is what the careful engineer tests.'
ALT_Y=b'after review, the valve is what the quiet technician inspects.'
ALT_AMB=b'after review, the probe is what the careful engineer tests.'
ALT_ANS=b'after review, the dax is what the careful engineer tests.'
OTHER_AMB=b'after review, the sensor is what the careful marker tests.'

def emit_all(a,pid):
    e=a.expression(pid);out=bytearray()
    while (step:=e.emit()) is not None:
        out.append(step.value)
        if not e.reafference(step,step.value):raise RuntimeError('source_transfer:reafference')
    return bytes(out)

def settle(a):
    for _ in range(64):
        if not a.internal_work_pending():return 0
        p=a.internal_tick()
        if p:return int(p)
    raise RuntimeError('source_transfer:settle')

def bindings(a,raw):
    return tuple(sorted((int(r.context),tuple(map(int,r.atoms)),str(r.template_identity)) for r in a.language.invert_surface(tuple(raw))))

def relations(a,raw):return tuple(sorted((c,x) for c,x,_ in bindings(a,raw)))

def repair(a,channel,amb,ans,counter):
    counter[0]+=4;base=counter[0];a._clear_current_occurrence();m=LanguageMasteryContactAdapterV1(a)
    gap=m.contact(CONTACT_UTTERANCE,tuple(amb),base,channel);q=settle(a)
    if q!=P_VERIFY:raise RuntimeError('source_transfer:question')
    question=emit_all(a,q);scene=m.contact(CONTACT_UTTERANCE,tuple(ans),base+1,channel)
    return int(a.choose(AdultStateV1())),int(scene),int(a._current_partner_context),question,int(gap)

def teach_lexemes(a):
    for concept,surface in ((A1,b'careful'),(A2,b'quiet'),(G1,b'engineer'),(G2,b'technician'),(V1,b'tests'),(V2,b'inspects'),(O1,b'sensor'),(O2,b'valve')):
        for n in range(3):a.observe_surface_item(concept,tuple(surface),1000+concept*10+n)
    for concept,surface,off in ((O1,b'probe',0),(O2,b'probe',20),(O1,b'dax',40),(G1,b'marker',60),(G2,b'marker',80)):
        for n in range(2):a.observe_surface_item(concept,tuple(surface),4000+concept*10+off+n)

def teach_base(a):
    for s in (5001,5002,5003):assert a.observe_surface_construction(CLAUSE,X,tuple(BASE_X),s)
    for s in (5011,5012,5013):assert a.observe_surface_construction(CLAUSE,Y,tuple(BASE_Y),s)

def teach_alt(a):
    for s in (6001,6002):assert a.observe_surface_construction(CLAUSE,X,tuple(ALT_X),s)
    for s in (6011,6012):assert a.observe_surface_construction(CLAUSE,Y,tuple(ALT_Y),s)

def response_ecology(a,amb,matched):
    alts=relations(a,amb)
    if len(alts)!=2 or matched not in alts:raise RuntimeError('source_transfer:ambiguity')
    comp=a._language_competition_context(alts);rep=a._language_repair_context(alts,matched)
    accept=a.leaf_surface(0x6D01,1,tuple(b'okay.'));verify=a.leaf_surface(0x6D02,2,tuple(b'can you verify?'))
    for _ in range(3):a.experience_atomic_program(P_VERIFY,verify,3*Q//4,0,comp,Q//8,True)
    for _ in range(3):
        a.experience_atomic_program(P_ACCEPT,accept,3*Q//4,0,rep,Q//16,True)
        a.experience_atomic_program(P_VERIFY,verify,Q//4,0,rep,Q//8,True)
    return int(comp),int(rep)

def main():
    t=time.perf_counter();a=LanguageMasteryAdultV1();teach_lexemes(a);teach_base(a)
    base_ids={tid for c,x,tid in bindings(a,BASE_ANS) if c==CLAUSE and x==X}
    obj_comp,obj_rep=response_ecology(a,BASE_AMB,(CLAUSE,X));events=[0x7000]
    rel_train=[];unrel_train=[]
    for _ in range(2):
        ch,*_=repair(a,REL,BASE_AMB,BASE_ANS,events);rel_train.append(ch);a.experience_partner_choice(ch,Q);a.experience_partner_background(ch,False)
    for _ in range(2):
        ch,*_=repair(a,UNREL,BASE_AMB,BASE_ANS,events);unrel_train.append(ch);a.experience_partner_choice(ch,-Q);a.experience_partner_background(ch,False)
    partner_before=json.dumps(a.partner_credit.checkpoint(),sort_keys=True,separators=(',',':'));reliability_cp=copy.deepcopy(a.checkpoint())
    teach_alt(a);partner_after=json.dumps(a.partner_credit.checkpoint(),sort_keys=True,separators=(',',':'))
    templates=a.language.template_candidates(CLAUSE,4);alt_rows=bindings(a,ALT_ANS);alt_ids={tid for c,x,tid in alt_rows if c==CLAUSE and x==X};transfer_cp=copy.deepcopy(a.checkpoint())
    tested=LanguageMasteryAdultV1.restore(copy.deepcopy(transfer_cp))
    rc,_,rctx,rq,_=repair(tested,REL,ALT_AMB,ALT_ANS,events);uc,_,uctx,uq,_=repair(tested,UNREL,ALT_AMB,ALT_ANS,events);nc,_,nctx,nq,_=repair(tested,NOVEL,ALT_AMB,ALT_ANS,events)
    different=LanguageMasteryAdultV1.restore(copy.deepcopy(transfer_cp));other_comp,other_rep=response_ecology(different,OTHER_AMB,(CLAUSE,X));oc,_,octx,_,_=repair(different,UNREL,OTHER_AMB,ALT_ANS,events)
    lesion_cp=copy.deepcopy(transfer_cp);unrel_obj_ctx=a._repair_partner_context(UNREL,obj_rep)
    for row in lesion_cp['partner_credit']['rows']:row['contexts']=[c for c in row['contexts'] if int(c['identity'])!=unrel_obj_ctx]
    lesioned=LanguageMasteryAdultV1.restore(lesion_cp);lc,_,lctx,_,_=repair(lesioned,UNREL,ALT_AMB,ALT_ANS,events)
    canonical=max(templates,key=lambda x:x.support);alt=next(x for x in templates if x.identity in alt_ids)
    cports=tuple(p.port for p in canonical.pieces if p.kind==2);aports=tuple(p.port for p in alt.pieces if p.kind==2)
    checks={
      'reliability_calibrated_only_on_canonical_form':rel_train==[P_ACCEPT,P_ACCEPT] and unrel_train==[P_ACCEPT,P_ACCEPT] and ALT_X.decode() not in json.dumps(reliability_cp) and ALT_ANS.decode() not in json.dumps(reliability_cp),
      'new_form_learning_does_not_rewrite_source_credit':partner_before==partner_after,
      'new_productive_form_is_active_but_not_canonical':len(templates)==2 and sorted(x.support for x in templates)==[4,6] and bytes(a.language.realize(CLAUSE,X) or ())==BASE_X,
      'template_identity_and_port_order_are_distinct':len(base_ids)==len(alt_ids)==1 and base_ids.isdisjoint(alt_ids) and cports==(0,1,2,3) and aports==(3,0,1,2),
      'heldout_new_form_reconstructs_same_repair_binding':alt_rows==((CLAUSE,X,next(iter(alt_ids))),) and relations(a,ALT_AMB)==relations(a,BASE_AMB),
      'same_new_form_testimony_uses_prior_source_accuracy':rc==P_ACCEPT and uc==P_VERIFY and nc==P_ACCEPT and rq==uq==nq==b'can you verify?',
      'policy_coordinate_is_repair_relation_not_surface_template':rctx==a._repair_partner_context(REL,obj_rep) and uctx==a._repair_partner_context(UNREL,obj_rep) and nctx==a._repair_partner_context(NOVEL,obj_rep),
      'adverse_source_history_does_not_globalize_to_other_relation':other_rep!=obj_rep and other_comp!=obj_comp and octx==different._repair_partner_context(UNREL,other_rep) and oc==P_ACCEPT,
      'focal_source_relation_lesion_abolishes_transfer':lctx==lesioned._repair_partner_context(UNREL,obj_rep) and lc==P_ACCEPT,
      'checkpoint_has_no_current_source_or_complete_test_surface':'current_partner_context' not in transfer_cp and ALT_ANS.decode() not in json.dumps(reliability_cp) and LanguageMasteryAdultV1.restore(copy.deepcopy(transfer_cp)).choose()==0,
      'both_fast_unions_run_this_transfer_audit':(
          'source-reliability-transfer:reference_source_reliability_productive_construction_transfer_verify.py' in (Path(__file__).parent/'run_language_mastery_fast.sh').read_text()
          and 'reference_source_reliability_productive_construction_transfer_verify.py' in (Path(__file__).parent/'run_language_mastery_factory_fast.sh').read_text()),
      'bounded_fast_path':time.perf_counter()-t<1.0,
    }
    failed=[k for k,v in checks.items() if not v]
    result={'schema':'cyber-lagoon.reference-source-reliability-productive-construction-transfer.v1','pass':not failed,'reference_only':True,'language_phenotype_improved':language_phenotype_improved,'visible_language_gain':'PRIOR_SOURCE_ACCURACY_TRANSFERS_ACROSS_HELDOUT_REORDERED_PRODUCTIVE_CONSTRUCTION','future_update_authority_preserved':future_update_authority_preserved,'canonical_ports':list(cports),'transfer_ports':list(aports),'choices':{'reliable_new_form':rc,'unreliable_new_form':uc,'novel_new_form':nc,'unreliable_other_relation':oc,'unreliable_after_lesion':lc},'checks':checks,'failed':failed,'remaining_red':['RECURSIVE_LONG_DISTANCE_EMBEDDED_TESTIMONY_TRANSFER','MULTISOURCE_CONSENSUS_AND_DECEPTION','DIRECT_TESTIMONY_PARITY'],'elapsed_ms':round((time.perf_counter()-t)*1000,3)}
    print('FOUNDRY_SOURCE_RELIABILITY_PRODUCTIVE_CONSTRUCTION_TRANSFER_'+('GREEN' if not failed else 'RED'));print(json.dumps(result,indent=2,sort_keys=True));return 0 if not failed else 1
if __name__=='__main__':raise SystemExit(main())
