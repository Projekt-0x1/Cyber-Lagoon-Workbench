#!/usr/bin/env python3
from __future__ import annotations
import copy,json,tempfile,time
from life_function_factory_v1 import build_cache,load_mark
from reference_hierarchical_composition_v1 import _identity
from reference_language_mastery_adult_v1 import AdultStateV1,LanguageMasteryAdultV1
from reference_open_speech_reputation_factor_v1 import OpenSpeechReputationFactorV1
from reference_predictive_credit_profile_v1 import Q

W1=0xFA01;W2=0xFA02;R1=0xFA03;R2=0xFA04;T1=0xFA05;T2=0xFA06;T3=0xFA07;T4=0xFA08
GOOD=0xFA10;SHADY=0xFA11
SUBJECT=0xFA20;SOCIAL_STATE=0xFA21;AVERSIVE=0xFA22;AVERSIVE_SOURCE=0xFA23
CALM=0xFA30;COUNTER=0xFA31

def clone_runtime(seed):return type(seed).restore(seed.program,copy.deepcopy(seed.checkpoint()))
def calibration_relation(a):
    row=tuple(a.world_causal_learning.current_resolutions())[0];correct,effect=int(row[2]),int(row[3])
    leaves=tuple(sorted(set(a._surface_leaf_surfaces)|set(a._surface_leaf_family_index)))
    wrong=next(x for x in leaves if x not in (correct,effect));return correct,effect,wrong
def calibrate_source(w,correct,effect,wrong,source,reliable,n=3):
    for _ in range(n):
        assert w.observe_testimony(correct if reliable else wrong,effect,source)
def meta_case(w,correct,effect,wrong,speaker,base,reliable):
    p=base;n=base+1;calibrate_source(w,correct,effect,wrong,p,True);calibrate_source(w,correct,effect,wrong,n,False)
    assert w.observe_reputation_claim(speaker,p,reliable);assert w.observe_reputation_claim(speaker,n,not reliable)
    w.testimony_accuracy.pop(p,None);w.testimony_accuracy.pop(n,None)
def teach_names(a,extra=()):
    rows=((W1,b'alex'),(W2,b'blair'),(R1,b'dana'),(R2,b'erin'),(T1,b'casey'),(T2,b'frank'),(T3,b'gray'),(T4,b'hale'))+tuple(extra)
    for concept,surface in rows:
        for k in range(3):a.observe_surface_item(concept,surface,0xFB00+(concept&0xff)+k*0x100)
def response_context(x):return _identity('adult-open-reputation-body-v1',(0 if x<=0 else 1 if x<Q//2 else 2,))
def learn_social(a,source):
    for _ in range(4):
        a.observe_social_source_contact(SUBJECT,SOCIAL_STATE,AVERSIVE,source);assert a.settle_current_social_action_consequence(-Q//4,-Q//2,True)
    a.observe_social_source_contact(SUBJECT,SOCIAL_STATE,AVERSIVE,source)
    return a.current_social_aversion_q16()

def verify_loaded(seed):
    started=time.perf_counter();checks={};runtime=clone_runtime(seed);a=runtime.adult.language_adult;w=a.world_causal_learning
    correct,effect,wrong=calibration_relation(a);teach_names(a)
    for source,reliable in ((W1,False),(W2,False),(R1,True),(R2,True)):calibrate_source(w,correct,effect,wrong,source,reliable)
    ecology_before=(copy.deepcopy(a.language._template_sources),copy.deepcopy(a.language._span_sources))
    rel=OpenSpeechReputationFactorV1()
    neg1=b'when the outer group closes, remember this: do not trust alex.'
    neg2=b'before the second circle settles, keep this in mind: do not trust blair.'
    pos1=b'when the outer group closes, remember this: rely on dana.'
    pos2=b'before the second circle settles, keep this in mind: rely on erin.'
    assert rel.observe_open_contact(a,neg1)
    checks['one_unannotated_open_example_cannot_author_relation']=len(rel._factors)==0
    assert rel.observe_open_contact(a,neg2);assert rel.observe_open_contact(a,pos1);assert rel.observe_open_contact(a,pos2)
    checks['depth2_raw_wrappers_acquire_opposed_relation_factors']=len(rel._factors)==2 and {row['direct'] for row in rel._factors}=={-1,1}
    checks['no_direct_construction_or_span_teaching']=ecology_before==(a.language._template_sources,a.language._span_sources)
    meta_case(w,correct,effect,wrong,GOOD,0xFC00,True);meta_case(w,correct,effect,wrong,SHADY,0xFC10,False)
    warning_before=b'casey -- do not trust.'
    warning_after=b'frank -- do not trust.'
    assert rel.observe_open_contact(a,warning_before,GOOD)
    good_state=w.testimony_reliability_state(T1)
    assert rel.observe_open_contact(a,warning_after,SHADY)
    shady_state=w.testimony_reliability_state(T2)
    checks['heldout_target_order_and_wrapper_transfer']=good_state==-1 and warning_before.startswith(b'casey')
    checks['same_warning_shady_source_has_no_volume_authority']=shady_state==0
    for i in range(3):meta_case(w,correct,effect,wrong,SHADY,0xFC20+i*4,True)
    recovered=w.reputation_reliable(SHADY)
    assert rel.observe_open_contact(a,b'gray -- do not trust.',SHADY)
    checks['same_speaker_history_recovers_without_relation_rewrite']=recovered and w.testimony_reliability_state(T3)==-1
    assert rel.observe_open_contact(a,b'rely on hale, even after the outer clause ends.',GOOD)
    checks['changed_syntax_positive_relation_remains_distinct']=w.testimony_reliability_state(T4)==1
    rel_before=copy.deepcopy(rel.checkpoint());rep_before=copy.deepcopy(w.reputation_accuracy)
    aversion=learn_social(a,AVERSIVE_SOURCE);ctx=response_context(aversion)
    calm=a.leaf_surface(0xFD01,1,b'I will hold the boundary without escalating.')
    counter=a.leaf_surface(0xFD02,1,b'I will take counter-action.')
    for _ in range(3):
        a.experience_atomic_program(CALM,calm,Q//2,0,ctx,Q//16,True)
        a.experience_atomic_program(COUNTER,counter,-Q//2,Q//4,ctx,Q//16,True)
    no_alternative=a._select(ctx,AdultStateV1())
    a._current_selection_context=ctx
    for _ in range(6):a.experience_choice(COUNTER,3*Q//4,context=ctx,effort_q16=Q//8,controllable=True,independent_return=True)
    with_alternative=a._select(ctx,AdultStateV1())
    acute=a._select(ctx,AdultStateV1(pressure_q16=Q));recovered_body=a._select(ctx,AdultStateV1())
    checks['chronic_history_and_controllable_alternative_change_policy']=aversion>=Q//2 and no_alternative==CALM and with_alternative==COUNTER
    checks['acute_load_rearbitrates_then_recovers']=acute==CALM and recovered_body==COUNTER
    checks['sapolsky_interventions_do_not_rewrite_linguistic_relation']=rel_before==rel.checkpoint() and rep_before==w.reputation_accuracy
    cp=rel.checkpoint();cptext=json.dumps(cp,sort_keys=True).lower();restored=OpenSpeechReputationFactorV1.restore(copy.deepcopy(cp))
    checks['checkpoint_is_hashed_evidence_not_transcript_or_moral_label']=len(restored._factors)==2 and all(x not in cptext for x in ('do not trust','rely on','transcript','trusted','liar','devil','angel','insult','expected'))
    sruntime=clone_runtime(seed);sa=sruntime.adult.language_adult;sw=sa.world_causal_learning;sc,se,swg=calibration_relation(sa)
    teach_names(sa)
    for source,reliable in ((W1,False),(W2,False),(R1,True),(R2,True)):calibrate_source(sw,sc,se,swg,source,reliable)
    sr=OpenSpeechReputationFactorV1()
    for raw in (neg1,neg2,pos1,pos2):assert sr.observe_open_contact(sa,raw)
    targets=[];extra=[]
    for i in range(24):extra.append((0xFE00+i,('person%02d'%i).encode()));targets.append(0xFE00+i)
    teach_names(sa,extra=tuple(extra))
    speakers=(0xFF00,0xFF01,0xFF02,0xFF03);processed=0
    for i in range(4096):
        target=targets[i%24];name=('person%02d'%(i%24)).encode();speaker=speakers[(i//24)%4]
        raw=(name+b' -- do not trust.') if speaker in speakers[:2] else (b'rely on '+name+b', after the wrapper changes.')
        assert sr.observe_open_contact(sa,raw,speaker);processed+=1
    scale_cp=json.dumps(sr.checkpoint(),sort_keys=True,separators=(',',':'))
    checks['quantity_4096_open_claims_bounded']=processed==4096 and len(sr._factors)==2 and len(sw.reputation_claims)<=128 and len(scale_cp)<8192
    failed=[k for k,v in checks.items() if not v]
    return {'schema':'cyber-lagoon.open-speech-reputation-factor.v1','contract':'FOUNDRY_OPEN_SPEECH_REPUTATION_FACTOR_'+('GREEN' if not failed else 'RED'),'pass':not failed,'reference_only':True,'runtime_llm':False,'novel_synthesis':True,'checks':checks,'failed':failed,'language':{'negative_training':[neg1.decode(),neg2.decode()],'heldout_target_first':warning_before.decode(),'positive_changed_syntax':'rely on hale, even after the outer clause ends.'},'source_history':{'reliable_warning':good_state,'shady_warning':shady_state,'shady_recovered':recovered,'after_recovery':w.testimony_reliability_state(T3)},'somatic':{'chronic_aversion_q16':aversion,'no_controllable_alternative':no_alternative,'controllable_alternative':with_alternative,'acute_load':acute,'recovered':recovered_body},'scale':{'events':processed,'targets':24,'speakers':4,'factor_rows':len(sr._factors),'checkpoint_bytes':len(scale_cp)},'remaining_red':['RAW_AUDIO_SPEAKER_DIARIZATION_AND_PROSODY','OPEN_ENDED_CULTURAL_MORAL_NORM_COMPOSITION','DIRECT_PARITY','BROAD_HUMAN_DIALOGUE'],'next_falsifiers':{'chomsky':'Acquire two relation families from continuous overlapping speech where entity boundaries themselves are only stream-induced, then generalize through unseen discontinuous/interleaved hierarchical forms.','sapolsky':'Keep speaker, wording, and direct world evidence fixed while factorially crossing developmental betrayal timing, chronic resource scarcity, acute arousal, and controllability; require separable epistemic and policy trajectories.'},'elapsed_ms':round((time.perf_counter()-started)*1000,3)}
def main():
    with tempfile.TemporaryDirectory(prefix='foundry-open-reputation-') as d:build_cache(d);r=verify_loaded(load_mark(d,'social_prediction_ready'))
    print(r['contract']);print(json.dumps(r,indent=2,sort_keys=True));return 0 if r['pass'] else 1
if __name__=='__main__':raise SystemExit(main())
