#!/usr/bin/env python3
"""Hostile receipt for persistent overlapping multi-party social contact."""
from __future__ import annotations
import copy,json,time
from reference_hierarchical_composition_v1 import _identity
from reference_language_mastery_adult_v1 import LanguageMasteryAdultV1,AdultStateV1
from reference_persistent_multi_agent_social_stream_v1 import PersistentMultiAgentSocialStreamV1
from reference_predictive_credit_profile_v1 import Q

SUB=0xD101;STATE=0xD102;INSULTISH=0xD103;NEUTRAL=0xD104;URGE=0xD105
CALM=0xD201;RETALIATE=0xD202
SRC_TARGET=0xDA01;SRC_OTHER=0xDA02;SRC_URGE=0xDB01


def response_context(aversion_q16):
    bucket=0 if int(aversion_q16)<=0 else (1 if int(aversion_q16)<Q//2 else 2)
    return _identity('adult-social-response-state-v1',(bucket,))


def social_projection(adult,sources):
    rows=[]
    for source,action in sources:
        adult.observe_social_source_contact(SUB,STATE,action,source)
        rows.append((source,action,adult.current_social_action_consequence(),adult.current_social_aversion_q16()))
    return tuple(rows)


def train_response(adult,aversion):
    ctx=response_context(aversion)
    calm=adult.leaf_surface(0xD301,1,b'I will not escalate this.')
    retaliate=adult.leaf_surface(0xD302,1,b'I will strike back.')
    for _ in range(3):
        adult.experience_atomic_program(CALM,calm,Q//2,0,ctx,Q//16,True)
        adult.experience_atomic_program(RETALIATE,retaliate,-Q//2,Q//4,ctx,Q//16,True)
    baseline=adult._select(ctx,AdultStateV1())
    adult._current_selection_context=ctx
    for _ in range(6):
        adult.experience_choice(RETALIATE,3*Q//4,context=ctx,effort_q16=Q//8,
                                controllable=True,independent_return=True)
    return ctx,calm,retaliate,baseline


def main():
    started=time.perf_counter();checks={};adult=LanguageMasteryAdultV1();stream=PersistentMultiAgentSocialStreamV1()

    # Three actors overlap at the same and adjacent logical times.  Arrival order is
    # chronology/sequence only: same opaque act is aversive for one source and neutral
    # for another, while an urging third party has no direct policy authority.
    for repeat in range(3):
        tick=10+repeat
        stream.admit(tick,SRC_TARGET,SUB,STATE,INSULTISH,-Q//4,-Q//2,True)
        stream.admit(tick,SRC_OTHER,SUB,STATE,INSULTISH,0,0,True)
        stream.admit(tick,SRC_URGE,SUB,STATE,URGE,None,0,False)
    due=stream.drain_until(adult,12)
    checks['overlapping_three_actor_contacts_are_all_admitted_once']=(
        len(due)==9 and len({sequence for _,sequence,_,_ in due})==9 and stream.pending_count==0)
    projection=social_projection(adult,((SRC_TARGET,INSULTISH),(SRC_OTHER,INSULTISH),(SRC_URGE,URGE)))
    focal_aversion=projection[0][3];other_aversion=projection[1][3]
    checks['same_act_keeps_source_history_dissociation']=(focal_aversion>=Q//2 and other_aversion==0)

    ctx,calm,retaliate,baseline=train_response(adult,focal_aversion)
    urged=adult._select(ctx,AdultStateV1())
    checks['third_party_urge_has_no_policy_authority']=(baseline==CALM and urged==RETALIATE)

    # Strict Sapolsky matched-history intervention: pressure changes the whole-action
    # winner; removing pressure restores the learned preference without rewriting it.
    response_state_before=copy.deepcopy(adult.credit.checkpoint())
    normal=adult._select(ctx,AdultStateV1())
    loaded=adult._select(ctx,AdultStateV1(pressure_q16=Q))
    recovered=adult._select(ctx,AdultStateV1())
    response_state_after=copy.deepcopy(adult.credit.checkpoint())
    checks['body_load_causally_reorders_and_recovery_restores']=(
        normal==RETALIATE and loaded==CALM and recovered==RETALIATE and response_state_before==response_state_after)

    # Checkpoint with several future contacts still pending, then replay the exact same
    # pending chronology in a clean restored Adult.  Current occurrences are deliberately
    # not checkpointed; pending boundary events rematerialize them.
    for tick,source,action,outcome,somatic in (
        (30,SRC_TARGET,NEUTRAL,0,0),(30,SRC_OTHER,INSULTISH,0,0),
        (30,SRC_TARGET,INSULTISH,-Q//4,-Q//2),(31,SRC_URGE,URGE,0,0)):
        stream.admit(tick,source,SUB,STATE,action,outcome,somatic,True)
    adult_cp=copy.deepcopy(adult.checkpoint());stream_cp=copy.deepcopy(stream.checkpoint())
    restored_adult=LanguageMasteryAdultV1.restore(copy.deepcopy(adult_cp))
    restored_stream=PersistentMultiAgentSocialStreamV1.restore(copy.deepcopy(stream_cp))
    due_a=stream.drain_until(adult,31);due_b=restored_stream.drain_until(restored_adult,31)
    final_a=social_projection(adult,((SRC_TARGET,INSULTISH),(SRC_OTHER,INSULTISH),(SRC_TARGET,NEUTRAL)))
    final_b=social_projection(restored_adult,((SRC_TARGET,INSULTISH),(SRC_OTHER,INSULTISH),(SRC_TARGET,NEUTRAL)))
    checks['midstream_checkpoint_replay_is_exact']=(due_a==due_b and final_a==final_b and stream.pending_count==restored_stream.pending_count==0)

    # Quantity: 1,024 contacts arrive across repeated overlapping triplets.  Drained
    # events retire from scheduler state; no transcript/event-per-life checkpoint grows.
    scale=PersistentMultiAgentSocialStreamV1();scale_adult=LanguageMasteryAdultV1()
    processed=0
    for base in range(0,1024,128):
        count=min(128,1024-base)
        for i in range(count):
            source=(SRC_TARGET,SRC_OTHER,SRC_URGE)[(base+i)%3]
            action=(INSULTISH,INSULTISH,URGE)[(base+i)%3]
            outcome=(-Q//8 if source==SRC_TARGET else 0)
            somatic=(-Q//8 if source==SRC_TARGET else 0)
            scale.admit(base//128,source,SUB,STATE,action,outcome,somatic,True)
        processed+=len(scale.drain_until(scale_adult,base//128))
    empty_checkpoint_bytes=len(json.dumps(scale.checkpoint(),sort_keys=True,separators=(',',':')))
    checks['quantity_1024_contacts_retire_without_scheduler_history_growth']=(
        processed==1024 and scale.pending_count==0 and empty_checkpoint_bytes<160)

    failed=[name for name,value in checks.items() if not value]
    surfaces={CALM:bytes(calm.surface).decode(),RETALIATE:bytes(retaliate.surface).decode()}
    result={
        'schema':'cyber-lagoon.persistent-multi-agent-social-stream.v1',
        'contract':'FOUNDRY_PERSISTENT_MULTI_AGENT_SOCIAL_STREAM_'+('GREEN' if not failed else 'RED'),
        'pass':not failed,'reference_only':True,'runtime_llm':False,
        'language_phenotype_improved':not failed,
        'visible_language_gain':'OVERLAPPING_MULTI_PARTY_HISTORY_PLUS_BODY_LOAD_CAUSES_RETALIATE_TO_CALM_TO_RETALIATE_PUBLIC_TRAJECTORY_WITH_EXACT_CHECKPOINT_REPLAY',
        'checks':checks,'failed':failed,
        'overlap':{'processed':len(due),'sources':3,'same_tick_contacts':3},
        'source_local_aversion_q16':{'focal':focal_aversion,'same_action_other':other_aversion},
        'sapolsky_public_trajectory':{
            'normal':surfaces.get(normal,''),'under_load':surfaces.get(loaded,''),'recovered':surfaces.get(recovered,'')},
        'checkpoint':{'pending_at_checkpoint':len(stream_cp['pending']),'replay_exact':checks['midstream_checkpoint_replay_is_exact']},
        'scale':{'contacts':processed,'pending_after_drain':scale.pending_count,'empty_scheduler_checkpoint_bytes':empty_checkpoint_bytes},
        'remaining_red':['RAW_AUDIO_SPEAKER_DIARIZATION','NATURAL_LANGUAGE_REPUTATION_CLAIM_BINDING',
                         'DIRECT_SOCIAL_REPUTATION_PARITY','CULTURAL_NORM_ACQUISITION_FROM_LARGE_CORPUS'],
        'elapsed_ms':round((time.perf_counter()-started)*1000,3)}
    print(result['contract']);print(json.dumps(result,indent=2,sort_keys=True));return 0 if not failed else 1

if __name__=='__main__':raise SystemExit(main())
