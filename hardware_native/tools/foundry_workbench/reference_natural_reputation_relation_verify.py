#!/usr/bin/env python3
from __future__ import annotations
import copy,json,tempfile,time
from life_function_factory_v1 import build_cache,load_mark
from reference_hierarchical_composition_v1 import _identity
from reference_language_mastery_adult_v1 import AdultStateV1,LanguageMasteryAdultV1
from reference_natural_reputation_relation_v1 import NaturalReputationRelationV1
from reference_predictive_credit_profile_v1 import Q
WARNING_CTX=0xE900;ENDORSE_CTX=0xE901;OUTER_CTX=0xE902
W1=0xE910;W2=0xE911;R1=0xE912;R2=0xE913;TARGET=0xE914;NESTED_TARGET=0xE915
WARNER=0xE920;RIVAL=0xE921;ALLY=0xE922
SUBJECT=0xE930;SOCIAL_STATE=0xE931;AVERSIVE=0xE932;NEUTRAL=0xE933;URGE=0xE934
AVERSIVE_SOURCE=0xE940;OTHER_SOURCE=0xE941;URGE_SOURCE=0xE942;CALM=0xE950;COUNTER=0xE951

def clone_runtime(seed):return type(seed).restore(seed.program,copy.deepcopy(seed.checkpoint()))
def calibration_relation(a):
    row=tuple(a.world_causal_learning.current_resolutions())[0];correct,effect=int(row[2]),int(row[3])
    leaves=tuple(sorted(set(a._surface_leaf_surfaces)|set(a._surface_leaf_family_index)))
    wrong=next(x for x in leaves if x not in (correct,effect));return correct,effect,wrong
def calibrate_source(w,correct,effect,wrong,source,reliable):
    for _ in range(3):
        if not w.observe_testimony(correct if reliable else wrong,effect,source):raise RuntimeError('source-calibration')
def calibrate_meta(w,correct,effect,wrong,speaker,base):
    p=base;n=base+1;calibrate_source(w,correct,effect,wrong,p,True);calibrate_source(w,correct,effect,wrong,n,False)
    assert w.observe_reputation_claim(speaker,p,True);assert w.observe_reputation_claim(speaker,n,False)
    w.testimony_accuracy.pop(p,None);w.testimony_accuracy.pop(n,None)
def teach_names(a):
    for concept,surface in ((W1,b'alex'),(W2,b'blair'),(R1,b'dana'),(R2,b'erin'),(TARGET,b'casey'),(NESTED_TARGET,b'frank')):
        for k in range(3):a.observe_surface_item(concept,surface,0xEA00+concept%256+k*0x100)
def teach_relations(a):
    assert a.observe_surface_construction(WARNING_CTX,(W1,),b'do not trust alex.',0xEB01)
    assert a.observe_surface_construction(WARNING_CTX,(W2,),b'do not trust blair.',0xEB02)
    assert a.observe_surface_construction(ENDORSE_CTX,(R1,),b'rely on dana.',0xEB11)
    assert a.observe_surface_construction(ENDORSE_CTX,(R2,),b'rely on erin.',0xEB12)
def surface(a,ctx,target):return bytes(a.leaf(ctx,(target,)).surface)
def learn_social(a,source,action,outcome,somatic):
    for _ in range(3):
        a.observe_social_source_contact(SUBJECT,SOCIAL_STATE,action,source);assert a.settle_current_social_action_consequence(outcome,somatic,True)
    a.observe_social_source_contact(SUBJECT,SOCIAL_STATE,action,source);return a.current_social_action_consequence(),a.current_social_aversion_q16()
def response_context(x):return _identity('adult-natural-reputation-boundary-v1',(0 if x<=0 else 1 if x<Q//2 else 2,))
def epistemic_context(x):return _identity('adult-natural-reputation-public-v1',(int(x),))
def public_response(a,state,surfaces):
    pid=a._select(epistemic_context(state),AdultStateV1());return surfaces.get(pid,b'')

def verify_loaded(seed):
    t=time.perf_counter();checks={};runtime=clone_runtime(seed);a=runtime.adult.language_adult;w=a.world_causal_learning
    correct,effect,wrong=calibration_relation(a);teach_names(a);teach_relations(a)
    for source,reliable in ((W1,False),(W2,False),(R1,True),(R2,True)):calibrate_source(w,correct,effect,wrong,source,reliable)
    rel=NaturalReputationRelationV1();one=NaturalReputationRelationV1()
    w1=surface(a,WARNING_CTX,W1);w2=surface(a,WARNING_CTX,W2);r1=surface(a,ENDORSE_CTX,R1);r2=surface(a,ENDORSE_CTX,R2)
    checks['one_example_cannot_author_relation_semantics']=one.calibrate_from_direct_life(a,w1) and one.relation_polarity(WARNING_CTX) is None
    for raw in (w1,w2,r1,r2):assert rel.calibrate_from_direct_life(a,raw)
    checks['lived_targets_calibrate_opposed_relations']=rel.relation_polarity(WARNING_CTX) is False and rel.relation_polarity(ENDORSE_CTX) is True
    held_warning=surface(a,WARNING_CTX,TARGET);held_endorse=surface(a,ENDORSE_CTX,TARGET)
    checks['heldout_target_productive']=held_warning==b'do not trust casey.' and held_endorse==b'rely on casey.' and a.construction_productivity(WARNING_CTX,(TARGET,))>=2
    wa=surface(a,WARNING_CTX,W1);wb=surface(a,WARNING_CTX,W2)
    assert a.language.observe_span(OUTER_CTX,(b'for this decision',wa),b'for this decision: '+wa,0xEC01)
    assert a.language.observe_span(OUTER_CTX,(b'for later',wb),b'for later: '+wb,0xEC02)
    nested=bytes(a.language.realize_span(OUTER_CTX,(b'for this case',surface(a,WARNING_CTX,NESTED_TARGET))))
    checks['recursive_merge_exposes_embedded_reputation_binding']=any(c==WARNING_CTX and target==NESTED_TARGET and d>0 for c,target,d in rel._all_bindings(a,nested))
    calibrate_meta(w,correct,effect,wrong,WARNER,0xED10);calibrate_meta(w,correct,effect,wrong,ALLY,0xED20)
    p=0xED30;n=0xED31;calibrate_source(w,correct,effect,wrong,p,True);calibrate_source(w,correct,effect,wrong,n,False)
    assert w.observe_reputation_claim(RIVAL,p,False);assert w.observe_reputation_claim(RIVAL,n,True);w.testimony_accuracy.pop(p,None);w.testimony_accuracy.pop(n,None)
    checks['speaker_history_separate_from_relation_semantics']=w.reputation_reliable(WARNER) and w.reputation_reliable(ALLY) and not w.reputation_reliable(RIVAL)
    assert rel.observe_natural_claim(a,held_warning,WARNER);warned=w.testimony_reliability_state(TARGET)
    for _ in range(32):assert rel.observe_natural_claim(a,held_endorse,RIVAL)
    checks['warning_beats_bad_advice_volume']=warned==-1 and w.testimony_reliability_state(TARGET)==-1
    assert rel.observe_natural_claim(a,held_endorse,ALLY);checks['credible_conflict_is_unresolved']=w.testimony_reliability_state(TARGET)==0
    assert rel.observe_natural_claim(a,nested,WARNER);checks['nested_warning_preserves_source_target']=w.testimony_reliability_state(NESTED_TARGET)==-1
    warning_leaf=a.leaf_surface(0xEE01,1,b'I will verify that source first.');conflict_leaf=a.leaf_surface(0xEE02,1,b'I need more independent source evidence.');direct_leaf=a.leaf_surface(0xEE03,1,b'My direct evidence now supports that source.')
    for _ in range(3):
        a.experience_atomic_program(0xEE11,warning_leaf,Q//2,0,epistemic_context(-1),Q//32,True)
        a.experience_atomic_program(0xEE12,conflict_leaf,Q//2,0,epistemic_context(0),Q//32,True)
        a.experience_atomic_program(0xEE13,direct_leaf,Q//2,0,epistemic_context(1),Q//32,True)
    responses={0xEE11:bytes(warning_leaf.surface),0xEE12:bytes(conflict_leaf.surface),0xEE13:bytes(direct_leaf.surface)}
    conflict_public=public_response(a,w.testimony_reliability_state(TARGET),responses);checks['conflict_changes_visible_discourse']=conflict_public==bytes(conflict_leaf.surface)
    calibrate_source(w,correct,effect,wrong,TARGET,True);direct_state=w.testimony_reliability_state(TARGET);direct_public=public_response(a,direct_state,responses)
    checks['direct_life_overrides_reputation_and_revises_language']=direct_state==1 and direct_public==bytes(direct_leaf.surface)
    negative,aversion=learn_social(a,AVERSIVE_SOURCE,AVERSIVE,-Q//4,-Q//2);neutral,other=learn_social(a,OTHER_SOURCE,AVERSIVE,0,0);_,same_other=learn_social(a,AVERSIVE_SOURCE,NEUTRAL,0,0)
    checks['visceral_aversion_is_action_source_history_local']=negative[2]>=2 and aversion>=Q//2 and neutral[2]>=2 and other==0 and same_other==0
    ctx=response_context(aversion);calm=a.leaf_surface(0xEF01,1,b'I will hold the boundary without escalating.');counter=a.leaf_surface(0xEF02,1,b'I will take counter-action.')
    for _ in range(3):a.experience_atomic_program(CALM,calm,Q//2,0,ctx,Q//16,True);a.experience_atomic_program(COUNTER,counter,-Q//2,Q//4,ctx,Q//16,True)
    baseline=a._select(ctx,AdultStateV1());a.observe_social_source_contact(SUBJECT,SOCIAL_STATE,URGE,URGE_SOURCE);urged=a._select(ctx,AdultStateV1());checks['urge_has_no_policy_authority']=baseline==CALM and urged==CALM
    a._current_selection_context=ctx
    for _ in range(6):a.experience_choice(COUNTER,3*Q//4,context=ctx,effort_q16=Q//8,controllable=True,independent_return=True)
    evidence_before=(copy.deepcopy(rel.checkpoint()),copy.deepcopy(w.reputation_accuracy),copy.deepcopy(w.reputation_claims),copy.deepcopy(a.credit.checkpoint()))
    revised=a._select(ctx,AdultStateV1());loaded=a._select(ctx,AdultStateV1(pressure_q16=Q));recovered=a._select(ctx,AdultStateV1())
    evidence_after=(copy.deepcopy(rel.checkpoint()),copy.deepcopy(w.reputation_accuracy),copy.deepcopy(w.reputation_claims),copy.deepcopy(a.credit.checkpoint()))
    checks['body_load_strictly_rearbitrates_then_recovers']=revised==COUNTER and loaded==CALM and recovered==COUNTER and evidence_before==evidence_after
    relcp=copy.deepcopy(rel.checkpoint());cptext=json.dumps(relcp,sort_keys=True).lower();restored_rel=NaturalReputationRelationV1.restore(copy.deepcopy(relcp));restored_a=LanguageMasteryAdultV1.restore(copy.deepcopy(a.checkpoint()))
    checks['checkpoint_has_evidence_not_transcript_or_moral_labels']=restored_rel.relation_polarity(WARNING_CTX) is False and all(x not in cptext for x in ('do not trust','rely on','transcript','trusted','liar','devil','angel','insult','expected'))
    checks['checkpoint_preserves_future_update_authority']=restored_rel.observe_natural_claim(restored_a,held_warning,WARNER) and restored_a.world_causal_learning.reputation_claims.get((WARNER,TARGET)) is False
    sruntime=clone_runtime(seed);sa=sruntime.adult.language_adult;sw=sa.world_causal_learning;sc,se,swg=calibration_relation(sa);teach_names(sa);teach_relations(sa)
    for source,reliable in ((W1,False),(W2,False),(R1,True),(R2,True)):calibrate_source(sw,sc,se,swg,source,reliable)
    sr=NaturalReputationRelationV1()
    for raw in (surface(sa,WARNING_CTX,W1),surface(sa,WARNING_CTX,W2),surface(sa,ENDORSE_CTX,R1),surface(sa,ENDORSE_CTX,R2)):assert sr.calibrate_from_direct_life(sa,raw)
    targets=[]
    for i in range(24):
        c=0xF100+i;targets.append(c);name=('person%02d'%i).encode()
        for k in range(3):sa.observe_surface_item(c,name,0xF400+i+k*0x100)
    speakers=(0xF200,0xF201,0xF202,0xF203);processed=0
    for i in range(2048):
        target=targets[i%24];speaker=speakers[(i//24)%4];rctx=WARNING_CTX if speaker in speakers[:2] else ENDORSE_CTX
        assert sr.observe_natural_claim(sa,surface(sa,rctx,target),speaker);processed+=1
    scalecp=json.dumps(sr.checkpoint(),sort_keys=True,separators=(',',':'))
    checks['quantity_2048_claims_bounded']=processed==2048 and sum(len(x['examples']) for x in sr.checkpoint()['relations'])==4 and len(sw.reputation_claims)<=128 and len(scalecp)<512
    failed=[k for k,v in checks.items() if not v]
    gain='HELDOUT_PRODUCTIVE_AND_RECURSIVELY_EMBEDDED_REPUTATION_LANGUAGE_NOW_CHANGES_SOURCE_CALIBRATION_AND_PUBLIC_EVIDENCE_SEEKING_WITH_DIRECT_LIFE_REVISION'
    return {'schema':'cyber-lagoon.natural-reputation-relation.v1','contract':'FOUNDRY_NATURAL_REPUTATION_RELATION_'+('GREEN' if not failed else 'RED'),'pass':not failed,'reference_only':True,'runtime_llm':False,'novel_synthesis':True,'language_phenotype_improved':not failed,'future_update_authority_preserved':not failed,'visible_language_gain':gain,'checks':checks,'failed':failed,'natural_language':{'warning':held_warning.decode(),'endorsement':held_endorse.decode(),'nested_warning':nested.decode(),'conflict_public':conflict_public.decode(),'after_direct_life':direct_public.decode()},'epistemics':{'warned':warned,'after_credible_conflict':0 if checks['credible_conflict_is_unresolved'] else None,'after_direct_life':direct_state,'nested_target':w.testimony_reliability_state(NESTED_TARGET)},'somatic':{'aversion_q16':aversion,'same_action_other_source_q16':other,'normal':bytes(counter.surface).decode() if revised==COUNTER else '','under_load':bytes(calm.surface).decode() if loaded==CALM else '','recovered':bytes(counter.surface).decode() if recovered==COUNTER else ''},'scale':{'events':processed,'targets':24,'speakers':4,'reputation_claim_rows':len(sw.reputation_claims),'relation_checkpoint_bytes':len(scalecp)},'remaining_red':['RAW_AUDIO_SPEAKER_DIARIZATION_AND_PROSODY','UNSUPERVISED_REPUTATION_RELATION_ACQUISITION_FROM_UNANNOTATED_OPEN_SPEECH','OPEN_ENDED_CULTURAL_MORAL_NORM_COMPOSITION','DIRECT_PARITY','BROAD_HUMAN_DIALOGUE'],'next_falsifiers':{'chomsky':'Acquire the warning relation itself inside a novel depth-2 raw wrapper without direct construction/span teaching, then transfer across held-out syntax and target order.','sapolsky':'Hold identical nested warning wording fixed while varying betrayal/recovery history, chronic-vs-acute load, and independently controllable alternatives.'},'elapsed_ms':round((time.perf_counter()-t)*1000,3)}
def main():
    with tempfile.TemporaryDirectory(prefix='foundry-natural-reputation-') as d:build_cache(d);r=verify_loaded(load_mark(d,'social_prediction_ready'))
    print(r['contract']);print('visible_language_gain='+r['visible_language_gain']);print(json.dumps(r,indent=2,sort_keys=True));return 0 if r['pass'] else 1
if __name__=='__main__':raise SystemExit(main())
