#!/usr/bin/env python3
"""N+1: continuous authenticated hand samples segment into pointing strokes from movement/hold dynamics only."""
from __future__ import annotations
import copy,json,time
from reference_continuous_gesture_segmenter_v1 import ContinuousGestureSegmenterV1
from reference_consequence_qualified_joint_attention_memory_verify import prepare
from reference_raw_pointing_motion_v1 import RawPointingMotionV1

SOURCE=0xFC11

def feed(seg,source,seq,point):
    y,x=map(int,point)
    return seg.ingest(source,seq,y,x,ContinuousGestureSegmenterV1.sample_digest(y,x))

def line(a,b,steps=3):
    ay,ax=a;by,bx=b
    return tuple((round(ay+(by-ay)*i/steps),round(ax+(bx-ax)*i/steps)) for i in range(steps+1))

def main():
    started=time.perf_counter();checks={};adult,o,g,tracker,left,left_pos,event=prepare();visible=sorted((int(e),int(row[0]),int(row[1])) for e,row in tracker.active.items() if row and int(row[-1])==0);right=next(row for row in visible if row[0]!=left);right_pos=(right[1],right[2])
    origin=(5,0);seg=ContinuousGestureSegmenterV1();seq=0;completed=[]
    stream=[origin,origin,*line(origin,left_pos)[1:],left_pos,left_pos,*line(left_pos,right_pos)[1:],right_pos,right_pos]
    for point in stream:
        seq+=1;row=feed(seg,SOURCE,seq,point)
        if row:completed.append(row)
    checks['one_uninterrupted_stream_emits_two_strokes_without_boundary_labels']=(len(completed)==2)
    checks['first_segment_resolves_left_visible_object']=(len(completed)>=1 and RawPointingMotionV1.target(tracker,completed[0])==left)
    checks['second_segment_resolves_right_visible_object']=(len(completed)>=2 and RawPointingMotionV1.target(tracker,completed[1])==right[0])
    checks['idle_and_holds_do_not_emit_extra_gestures']=(len(completed)==2)

    # Replay and forged samples fail before provenance or partial-segment mutation.
    before=copy.deepcopy(seg.checkpoint());active_before=copy.deepcopy(seg.active);replay=False
    try:feed(seg,SOURCE,seq,stream[-1])
    except ValueError:replay=True
    checks['duplicate_sample_sequence_refuses_atomically']=(replay and seg.checkpoint()==before and seg.active==active_before)
    forged=False
    try:seg.ingest(SOURCE,seq+1,9,9,'0'*64)
    except ValueError:forged=True
    checks['forged_sample_digest_refuses_before_partial_stroke_mutation']=(forged and seg.checkpoint()==before and seg.active==active_before)

    # Partial stroke is transient: restart preserves sequence ownership but not old motion.
    partial=ContinuousGestureSegmenterV1();s=0
    for point in (origin,(5,3),(6,6)):
        s+=1;feed(partial,SOURCE,s,point)
    cp=copy.deepcopy(partial.checkpoint());restored=ContinuousGestureSegmenterV1.restore(cp)
    r1=feed(restored,SOURCE,s+1,(7,9));r2=feed(restored,SOURCE,s+2,(7,9));r3=feed(restored,SOURCE,s+3,(7,9))
    checks['checkpoint_preserves_sample_cursor_but_breaks_partial_gesture_continuity']=(not r1 and not r2 and not r3 and restored.ingress.last_sequence.get(('gesture-sample',SOURCE))==s+3)
    blob=json.dumps(cp,sort_keys=True)
    checks['checkpoint_contains_no_partial_points_or_completed_trajectory']=(all(token not in blob for token in ('active','points','trajectory',str(origin))))

    # Withdrawal removes source authority and transient partial state.
    seg.withdraw_source(SOURCE);withdrawn=False
    try:feed(seg,SOURCE,seq+1,(10,10))
    except ValueError:withdrawn=True
    checks['source_withdrawal_blocks_future_samples_and_clears_partial_state']=(withdrawn and SOURCE not in seg.active)
    checks['bounded_fast_path']=time.perf_counter()-started<1.0
    failed=[k for k,v in checks.items() if not v]
    result={'schema':'cyber-lagoon.reference-continuous-gesture-segmentation.v1','contract':'FOUNDRY_CONTINUOUS_GESTURE_SEGMENTATION_'+('GREEN' if not failed else 'RED'),'pass':not failed,'reference_only':True,'language_phenotype_improved':not failed,'future_update_authority_preserved':True,
            'visible_language_gain':'CONTINUOUS_AUTHENTICATED_HAND_SAMPLES_NOW_SELF_SEGMENT_INTO_MULTIPLE_POINTING_STROKES_FROM_MOVEMENT_AND_HOLD_DYNAMICS_WITHOUT_HOST_START_END_LABELS',
            'checks':checks,'failed':failed,'remaining_red':['SAME_PROCESS_GESTURE_SAMPLE_STREAM_TO_SHARED_ATTENTION','LEARNED_GESTURE_STYLE_VARIATION','DIRECT_CONTINUOUS_GESTURE_SEGMENTATION_PARITY'],'elapsed_ms':round((time.perf_counter()-started)*1000,3)}
    print(result['contract']);print(json.dumps(result,indent=2,sort_keys=True));return 0 if not failed else 1
if __name__=='__main__':raise SystemExit(main())
