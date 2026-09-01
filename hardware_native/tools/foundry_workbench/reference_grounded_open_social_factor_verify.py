#!/usr/bin/env python3
from __future__ import annotations
import copy,json,tempfile,time
from life_function_factory_v1 import build_cache,load_mark
from reference_grounded_open_social_factor_v1 import GroundedOpenSocialFactorV1

N1=0xF901;N2=0xF902;P1=0xF903;P2=0xF904;GOOD=0xF910;BAD=0xF911

def clone(seed):return type(seed).restore(seed.program,copy.deepcopy(seed.checkpoint()))
def relation(a):
    row=tuple(a.world_causal_learning.current_resolutions())[0]
    c,e=int(row[2]),int(row[3]);leaves=tuple(sorted(set(a._surface_leaf_surfaces)|set(a._surface_leaf_family_index)))
    return c,e,next(x for x in leaves if x not in (c,e))
def calibrate(w,c,e,x,source,reliable,n=3):
    for _ in range(n):assert w.observe_testimony(c if reliable else x,e,source)
def meta(w,c,e,x,speaker,base,reliable):
    calibrate(w,c,e,x,base,True);calibrate(w,c,e,x,base+1,False)
    assert w.observe_reputation_claim(speaker,base,reliable);assert w.observe_reputation_claim(speaker,base+1,not reliable)
    w.testimony_accuracy.pop(base,None);w.testimony_accuracy.pop(base+1,None)
def pair(r,a,left,right,lt,rt,source):
    for i,(lf,rf) in enumerate(((left[:8],right[:8]),(left[8:21],right[8:21]),(left[21:],right[21:]))):
        assert r.admit_fragment(a,source,lf,grounded_target=lt,boundary=i==2)
        assert r.admit_fragment(a,source+1,rf,grounded_target=rt,boundary=i==2)

def verify_loaded(seed):
    t0=time.perf_counter();checks={};a=clone(seed).adult.language_adult;w=a.world_causal_learning;c,e,x=relation(a)
    for source,reliable in ((N1,False),(N2,False),(P1,True),(P2,True)):calibrate(w,c,e,x,source,reliable)
    lex=copy.deepcopy(a.language._lexeme_sources);r=GroundedOpenSocialFactorV1()
    pair(r,a,b'outer alex marker: avoid contact now.',b'outer blair marker: avoid contact now.',N1,N2,0xFA00)
    pair(r,a,b'alex after clause: avoid contact now.',b'blair after clause: avoid contact now.',N1,N2,0xFA10)
    pair(r,a,b'outer dana marker: prefer contact now.',b'outer erin marker: prefer contact now.',P1,P2,0xFA20)
    pair(r,a,b'dana after clause: prefer contact now.',b'erin after clause: prefer contact now.',P1,P2,0xFA30)
    checks['joint_entity_and_relation_induction']=set(r._relations)=={-1,1} and all(r._targets[k].get('entity') for k in (N1,N2,P1,P2))
    checks['no_prelearned_entity_lexeme_needed']=lex==a.language._lexeme_sources
    checks['interleaved_sources_never_cross_concatenate']=r.live_source_count==0
    for k in (N1,N2,P1,P2):w.testimony_accuracy.pop(k,None)
    meta(w,c,e,x,GOOD,0xFB00,True);meta(w,c,e,x,BAD,0xFB10,False)
    assert r.observe_open_utterance(a,b'alex -- wrapper -- avoid contact now.',GOOD)
    assert r.observe_open_utterance(a,b'dana -- wrapper -- prefer contact now.',GOOD)
    checks['heldout_wrapper_transfers_both_polarities']=w.testimony_reliability_state(N1)==-1 and w.testimony_reliability_state(P1)==1
    for _ in range(8):assert r.observe_open_utterance(a,b'blair -- avoid contact now.',BAD)
    checks['meta_unreliable_repetition_has_no_authority']=w.testimony_reliability_state(N2)==0
    cp0=copy.deepcopy(r.checkpoint())
    for i in range(3):meta(w,c,e,x,BAD,0xFC00+i*4,True)
    assert r.observe_open_utterance(a,b'blair -- avoid contact now.',BAD)
    checks['source_recovery_changes_authority_not_language']=w.testimony_reliability_state(N2)==-1 and cp0==r.checkpoint()
    calibrate(w,c,e,x,N2,True,n=6)
    checks['later_direct_life_overrides_social_factor']=w.testimony_reliability_state(N2)==1
    cp=r.checkpoint();text=json.dumps(cp,sort_keys=True).lower();restored=GroundedOpenSocialFactorV1.restore(copy.deepcopy(cp))
    checks['checkpoint_is_hashed_not_transcript']=restored.live_source_count==0 and all(s not in text for s in ('alex','blair','dana','erin','avoid contact','prefer contact','transcript','live'))
    done=0
    for i in range(4096):
        assert r.observe_open_utterance(a,b'alex -- avoid contact now.' if i&1 else b'dana -- prefer contact now.',GOOD);done+=1
    scale=json.dumps(r.checkpoint(),sort_keys=True,separators=(',',':'))
    checks['quantity_4096_is_bounded']=done==4096 and len(r._targets)==4 and len(r._relations)==2 and len(scale)<32768
    failed=[k for k,v in checks.items() if not v]
    return {'schema':'cyber-lagoon.grounded-open-social-factor.v1','contract':'FOUNDRY_GROUNDED_OPEN_SOCIAL_FACTOR_'+('GREEN' if not failed else 'RED'),'pass':not failed,'reference_only':True,'checks':checks,'failed':failed,'scale':{'events':done,'checkpoint_bytes':len(scale)},'remaining_red':['RAW_AUDIO_DIARIZATION_PROSODY','UNSEEN_IDENTITY_WITHOUT_GROUNDING','CULTURAL_MORAL_LONG_HISTORY','DIRECT_PARITY','BROAD_HUMAN_DIALOGUE'],'elapsed_ms':round((time.perf_counter()-t0)*1000,3)}
def main():
    with tempfile.TemporaryDirectory(prefix='foundry-grounded-open-') as d:build_cache(d);r=verify_loaded(load_mark(d,'social_prediction_ready'))
    print(r['contract']);print(json.dumps(r,indent=2,sort_keys=True));return 0 if r['pass'] else 1
if __name__=='__main__':raise SystemExit(main())
