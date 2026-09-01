#!/usr/bin/env python3
"""N+1: partner returns are source/order/integrity authenticated before exact-ticket social consequence settlement."""
from __future__ import annotations
import copy,json,time
from reference_authenticated_body_protocol_verify import VSRC,DSRC,TSRC,WSRC,visual,joint,motion_to
from reference_body_event_ingress_v1 import BodyEventIngressV1
from reference_consequence_qualified_joint_attention_memory_v1 import ConsequenceQualifiedJointAttentionMemoryV1
from reference_consequence_qualified_joint_attention_memory_verify import prepare
from reference_continuous_visual_sensor_ownership_verify import A1,A2,train_level1
from reference_embodied_multimodal_process_v1 import EmbodiedMultimodalRuntimeV1
from reference_gaze_sensor_ingress_v1 import GazeSensorIngressV1
from reference_gesture_sensor_ingress_v1 import GestureSensorIngressV1
from reference_multiple_visual_object_files_verify import train_temporal
from reference_predictive_credit_profile_v1 import Q
from reference_visual_sensor_ingress_v1 import VisualSensorIngressV1

def return_event(source,ticket,plan,sequence,digest=None):
    payload={'ticket':int(ticket),'plan':int(plan),'outcome':Q,'independent':True}
    return {'kind':'return','source':int(source),'sequence':int(sequence),'ticket':int(ticket),'plan':int(plan),'outcome':Q,'independent':True,
            'digest':digest or BodyEventIngressV1.payload_digest('return',payload)}
def snap(rt):return copy.deepcopy(rt.checkpoint())

def main():
    started=time.perf_counter();checks={};adult,o,g,_tracker,_left,_left_pos,_event=prepare();level1=train_level1();temporal=train_temporal(level1)
    rt=EmbodiedMultimodalRuntimeV1(adult,o,ConsequenceQualifiedJointAttentionMemoryV1(),g,level1,temporal,VisualSensorIngressV1(),GestureSensorIngressV1(),GazeSensorIngressV1(),int(o.world_state_occurrence))
    rt.event(visual(1,A1));rt.event(visual(2,A2));target=sorted((e,row[0],row[1]) for e,row in rt.tracker.active.items() if row and int(row[-1])==0)[0];motion=motion_to(target[1],target[2]);action=rt.event(joint(motion))
    ticket,plan=int(action['ticket']),int(action['plan']);checks['joint_action_opens_one_pending_exact_ticket_return']=(ticket>0 and ticket in rt.memory.pending and rt.memory.pending[ticket][1]==TSRC)

    before=snap(rt);wrong_source=False
    try:rt.event(return_event(DSRC,ticket,plan,1))
    except ValueError:wrong_source=True
    checks['wrong_authenticated_source_cannot_claim_pending_partner_return']=(wrong_source and snap(rt)==before)

    forged=False
    try:rt.event(return_event(TSRC,ticket,plan,1,'0'*64))
    except ValueError:forged=True
    checks['forged_return_digest_refuses_before_ticket_or_memory_mutation']=(forged and snap(rt)==before)

    wrong_plan=False
    try:rt.event(return_event(TSRC,ticket,plan+1,1))
    except ValueError:wrong_plan=True
    checks['wrong_plan_cannot_claim_exact_ticket_even_with_right_source']=(wrong_plan and snap(rt)==before)

    settled=rt.event(return_event(TSRC,ticket,plan,1));after=snap(rt)
    checks['valid_authenticated_return_settles_exact_ticket_and_commits_return_cursor']=(settled=={'settled':True} and ticket not in rt.memory.pending and rt.body.last_sequence.get(('return',TSRC))==1)

    duplicate=False
    try:rt.event(return_event(TSRC,ticket,plan,1))
    except ValueError:duplicate=True
    checks['duplicate_return_cannot_double_settle_or_advance_cursor']=(duplicate and snap(rt)==after)

    cp=copy.deepcopy(rt.checkpoint());restored=EmbodiedMultimodalRuntimeV1.restore(cp)
    checks['return_provenance_cursor_survives_restart_without_outcome_payload']=(restored.body.last_sequence.get(('return',TSRC))==1)
    checks['body_checkpoint_contains_only_lane_source_sequence_provenance']=(
        all(set(row)=={'lane','source','sequence'} for row in cp['body'].get('last_sequence',()))
        and set(cp['body'])=={'schema','last_sequence','withdrawn_sources'})
    checks['bounded_fast_path']=time.perf_counter()-started<1.0
    failed=[k for k,v in checks.items() if not v]
    result={'schema':'cyber-lagoon.reference-authenticated-partner-return-protocol.v1','contract':'FOUNDRY_AUTHENTICATED_PARTNER_RETURN_PROTOCOL_'+('GREEN' if not failed else 'RED'),'pass':not failed,'reference_only':True,'language_phenotype_improved':not failed,'future_update_authority_preserved':True,
            'visible_language_gain':'PARTNER_RETURNS_NOW_REQUIRE_AUTHENTICATED_SOURCE_FRESH_SEQUENCE_INTEGRITY_AND_EXACT_ADULT_ACTION_TICKET_BEFORE_SOCIAL_MEMORY_CONSEQUENCE_SETTLES',
            'checks':checks,'failed':failed,'remaining_red':['CONTINUOUS_STREAM_GESTURE_SEGMENTATION','DIRECT_AUTHENTICATED_BODY_PROTOCOL_PARITY'],'elapsed_ms':round((time.perf_counter()-started)*1000,3)}
    print(result['contract']);print(json.dumps(result,indent=2,sort_keys=True));return 0 if not failed else 1
if __name__=='__main__':raise SystemExit(main())
