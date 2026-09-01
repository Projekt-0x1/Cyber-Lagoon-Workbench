#!/usr/bin/env python3
"""N+1: raw embodied conversation ingress arbitrates remembered unnamed-event recall with ordinary text affordances."""
from __future__ import annotations
import copy,json,tempfile,time
from pathlib import Path
from reference_consequence_qualified_joint_attention_memory_v1 import ConsequenceQualifiedJointAttentionMemoryV1
from reference_consequence_qualified_joint_attention_memory_verify import prepare
from reference_embodied_conversation_terminal_v1 import EmbodiedConversationAffordanceV1,checkpoint,respond,restore,save
from reference_language_mastery_adult_v1 import AdultStateV1
from reference_language_mastery_terminal_v1 import emit_choice
from reference_nonvisible_unnamed_deictic_event_verify import THAT,AGAIN,RECALL_CTX,CHANNEL,event_surface
from reference_predictive_credit_profile_v1 import Q

S1=0xEA11;S2=0xEA12;COMPETING=0xEA21

def install_memory(adult,o,g,tracker,left_pos,memory):
    for source in (S1,S2):
        staged=memory.stage(adult,o,tracker,g,b'that',THAT,*left_pos,CHANNEL,source)
        if staged is None:raise RuntimeError('embodied_memory:stage')
        ticket,response,_episode=staged
        spoken=emit_choice(adult,response)
        if not spoken or not memory.settle_partner_return(adult,ticket,response,Q,True):raise RuntimeError('embodied_memory:settle')
    learned=memory.resolve(adult,o,b'that again',CHANNEL)
    if learned is None:raise RuntimeError('embodied_memory:not_learned')
    return learned

def main():
    started=time.perf_counter();checks={};adult,o,g,tracker,left,left_pos,event=prepare();memory=ConsequenceQualifiedJointAttentionMemoryV1();learned=install_memory(adult,o,g,tracker,left_pos,memory)
    # The visual file itself is no longer needed: raw text must resolve through durable
    # joint-attention memory inside the embodied proposal tournament.
    tracker.active_entity=0 if hasattr(tracker,'active_entity') else 0
    raw=b'that again';last=int(getattr(o,'world_state_occurrence',0))
    proposals=EmbodiedConversationAffordanceV1.proposals(adult,o,memory,raw,CHANNEL,last)
    out=respond(adult,o,memory,raw,CHANNEL,last)
    checks['raw_ingress_has_one_remembered_event_proposal_without_test_side_resolver']=(len(proposals)==1 and int(proposals[0].context)==int(learned.event))
    checks['raw_embodied_respond_externalizes_remembered_event_through_normal_program_surface']=(out==event_surface(adult,learned.event)==b'the careful engineer tests the sensor.')

    # Session checkpoint/restore must preserve Adult + organism + memory, with no transcript.
    payload=checkpoint(adult,o,memory,last);blob=json.dumps(payload,sort_keys=True)
    ra,ro,rm,rlast=restore(copy.deepcopy(payload));replay=respond(ra,ro,rm,raw,CHANNEL,rlast)
    checks['embodied_session_checkpoint_restart_repeats_memory_grounded_conversation']=(replay==out and rlast==last)
    checks['session_checkpoint_contains_no_query_or_answer_transcript']=(all(token not in blob for token in ('that again','tests the sensor','conversation_buffer','transcript','context_window')))

    # Empty memory must not manufacture this answer. The same Adult/world pair alone is insufficient.
    empty=ConsequenceQualifiedJointAttentionMemoryV1();empty_adult=type(adult).restore(copy.deepcopy(adult.checkpoint()));empty_org=type(o).restore(copy.deepcopy(o.checkpoint()))
    empty_rows=EmbodiedConversationAffordanceV1.proposals(empty_adult,empty_org,empty,raw,CHANNEL,last)
    empty_out=respond(empty_adult,empty_org,empty,raw,CHANNEL,last)
    checks['same_adult_and_world_without_joint_memory_do_not_recall_absent_event']=(not empty_rows and empty_out==b'')

    # Genuine tournament tie: give the exact raw `that again` an ordinary action plan
    # different from the remembered event response. Memory must not get priority.
    tied=type(adult).restore(copy.deepcopy(adult.checkpoint()));tied_org=type(o).restore(copy.deepcopy(o.checkpoint()));tied_memory=ConsequenceQualifiedJointAttentionMemoryV1.restore(copy.deepcopy(memory.checkpoint()))
    ordinary_leaf=tied.leaf(RECALL_CTX,(THAT,AGAIN))
    # A second Program identity over the same resident event leaf creates a true
    # action-level competitor without authoring any new response surface.
    alt_root=type(ordinary_leaf)(int(learned.event),int(getattr(ordinary_leaf,'context',RECALL_CTX)),tied._leaf_surface(int(learned.event)))
    for _ in range(3):tied.experience_atomic_program(COMPETING,alt_root,Q,context=ordinary_leaf.identity,effort_q16=Q//32,controllable=True)
    tied.experience_program_background(COMPETING,False,context=ordinary_leaf.identity)
    tied_rows=EmbodiedConversationAffordanceV1.proposals(tied,tied_org,tied_memory,raw,CHANNEL,last)
    tied_out=respond(tied,tied_org,tied_memory,raw,CHANNEL,last)
    checks['ordinary_text_and_memory_proposals_compete_without_memory_priority']=(len(tied_rows)>=2 and len({(r.context,r.plan,r.kind) for r in tied_rows})>=2 and tied_out==b'')

    # Atomic file save path exercises the same checkpoint shape used by a persistent body process.
    with tempfile.TemporaryDirectory() as td:
        path=Path(td)/'session.json';save(path,adult,o,memory,last);disk=json.loads(path.read_text());da,do,dm,dl=restore(disk);disk_out=respond(da,do,dm,raw,CHANNEL,dl)
    checks['atomic_session_file_roundtrip_preserves_direct_memory_conversation']=(disk_out==out)
    checks['bounded_fast_path']=time.perf_counter()-started<1.0
    failed=[k for k,v in checks.items() if not v]
    result={'schema':'cyber-lagoon.reference-embodied-memory-conversation.v1','contract':'FOUNDRY_EMBODIED_MEMORY_CONVERSATION_'+('GREEN' if not failed else 'RED'),'pass':not failed,'reference_only':True,'language_phenotype_improved':not failed,'future_update_authority_preserved':True,'visible_language_gain':'RAW_EMBODIED_CONVERSATION_INGRESS_NOW_ARBITRATES_UNNAMED_ABSENT_EVENT_MEMORY_WITH_OTHER_RESIDENT_TEXT_AFFORDANCES_WITHOUT_A_HOST_MEMORY_DISPATCHER','conversation':['that again',out.decode() if out else ''],'checks':checks,'failed':failed,'remaining_red':['EMBODIED_SESSION_OWNS_JOINT_ATTENTION_ACQUISITION_CONTACT','PHYSICAL_GESTURE_AND_TEXT_MULTIMODAL_BODY_INGRESS','DIRECT_EMBODIED_MEMORY_CONVERSATION_PARITY'],'elapsed_ms':round((time.perf_counter()-started)*1000,3)}
    print(result['contract']);print(json.dumps(result,indent=2,sort_keys=True));return 0 if not failed else 1
if __name__=='__main__':raise SystemExit(main())
