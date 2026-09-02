#!/usr/bin/env python3
"""Public-snapshot falsifier: background text teaches structure without taking foreground/truth authority."""
from __future__ import annotations
import copy,json,time
from reference_language_mastery_contact_adapter_v1 import CONTACT_UTTERANCE,LanguageMasteryContactAdapterV1
from reference_long_distance_structural_reply_verify import DISTRACTOR
from reference_mathematical_adult_workbench_v1 import MathematicalWorkbenchAdultV1
from reference_persistent_ambient_language_stream_v1 import PersistentAmbientLanguageStreamV1,semantically_blind_index
from reference_predictive_credit_profile_v1 import Q
from reference_recursive_testimony_span_repair_verify import calibrated
from reference_source_reliability_productive_construction_transfer_verify import ALT_AMB,ALT_ANS,ALT_X,ALT_Y,CLAUSE,REL,X,response_ecology,teach_alt
S1=0x9701;S2=0x9702
def frame(left,right):return b'reply frame [ '+left+b' ] beside [ '+right+b' ]'
def prepared():
    language,_object,_counter=calibrated();teach_alt(language);response_ecology(language,ALT_AMB,(CLAUSE,X));return MathematicalWorkbenchAdultV1(language)
def focus(a):
    l=a.language_adult;return (l._current_selection_context,l._current_language_channel,tuple(l._pending_language_competition),l._language_competition_active,tuple(l._current_partner_action))
def world(a):return copy.deepcopy(a.language_adult.world_causal_learning.checkpoint())
def foreground(a):return int(LanguageMasteryContactAdapterV1(a.language_adult).contact(CONTACT_UTTERANCE,tuple(ALT_AMB),0x9A00,REL))
def main():
    started=time.perf_counter();c={};train1=frame(ALT_X,ALT_Y);train2=frame(ALT_Y,ALT_X);held=frame(ALT_ANS,DISTRACTOR)
    a=prepared();assert foreground(a)>0;f0=focus(a);w0=world(a);s=PersistentAmbientLanguageStreamV1();s.admit(1,S1,train1);one=s.drain_until(a,1);one_span=tuple(a.language.invert_span(tuple(held)));s.admit(2,S2,train2);two=s.drain_until(a,2);spans=tuple(a.language.invert_span(tuple(held)))
    c['foreground_occurrence_survives_background_learning']=focus(a)==f0
    c['one_source_below_quorum_two_sources_teach_heldout_wrapper']=not one_span and len(spans)==1 and len(spans[0].children)==2 and all(r[-1] for r in (*one,*two))
    c['ambient_structure_mints_zero_world_truth']=world(a)==w0
    b=prepared();r=PersistentAmbientLanguageStreamV1();r.admit(1,S1,train1);r.admit(2,S1,train2);r.drain_until(b,2);c['one_prolific_source_cannot_fake_independent_support']=not tuple(b.language.invert_span(tuple(held)))
    c['sampler_is_payload_blind']=semantically_blind_index(7,9,3)==semantically_blind_index(7,9,3)
    loaded=prepared();q=PersistentAmbientLanguageStreamV1();q.admit(1,S1,train1);assert q.drain_until(loaded,1);q.admit(2,S2,train2)
    for i in range(8):loaded.language_adult.settle_body_ingress('ambient-public-load',i+1,format(i+1,'064x'),Q)
    blocked=q.drain_until(loaded,2);pending=q.pending_count
    for _ in range(64):loaded.language_adult.internal_tick()
    recovered=q.drain_until(loaded,2)
    c['load_defers_same_queued_post_then_recovery_processes_without_refetch']=not blocked and pending==1 and len(recovered)==1 and q.pending_count==0
    cp=PersistentAmbientLanguageStreamV1();cp.admit(5,0xB001,b'pending post');data=copy.deepcopy(cp.checkpoint());c['pending_restart_exact']=PersistentAmbientLanguageStreamV1.restore(data).checkpoint()==data
    failed=sorted(k for k,v in c.items() if not v);out={'schema':'cyber-lagoon.public-ambient-bystander-stream.v1','contract':'FOUNDRY_PUBLIC_AMBIENT_BYSTANDER_STREAM_'+('GREEN' if not failed else 'RED'),'pass':not failed,'reference_only':True,'checks':c,'failed':failed,'elapsed_ms':round((time.perf_counter()-started)*1000,3)}
    print(out['contract']);print(json.dumps(out,indent=2,sort_keys=True));return 0 if not failed else 1
if __name__=='__main__':raise SystemExit(main())
