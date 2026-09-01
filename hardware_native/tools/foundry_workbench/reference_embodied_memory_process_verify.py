#!/usr/bin/env python3
"""N+1: physical persistent embodied process answers raw stdin from durable unnamed-event memory."""
from __future__ import annotations
import copy,json,subprocess,sys,tempfile,time
from pathlib import Path
from reference_consequence_qualified_joint_attention_memory_v1 import ConsequenceQualifiedJointAttentionMemoryV1
from reference_embodied_memory_conversation_verify import install_memory
from reference_embodied_conversation_terminal_v1 import checkpoint,restore
from reference_consequence_qualified_joint_attention_memory_verify import prepare
from reference_nonvisible_unnamed_deictic_event_verify import CHANNEL

TERMINAL=Path(__file__).with_name('reference_embodied_conversation_terminal_v1.py')

def run(path,raw):
    return subprocess.run([sys.executable,str(TERMINAL),'--resume',str(path),'--channel',str(CHANNEL),'--idle-ms','1000'],input=raw+b'\n',stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=5,check=True)

def main():
    started=time.perf_counter();checks={};adult,o,g,tracker,left,left_pos,event=prepare();memory=ConsequenceQualifiedJointAttentionMemoryV1();learned=install_memory(adult,o,g,tracker,left_pos,memory);tracker.active={};last=int(o.world_state_occurrence)
    with tempfile.TemporaryDirectory() as td:
        path=Path(td)/'embodied.json';path.write_text(json.dumps(checkpoint(adult,o,memory,last),sort_keys=True))
        result=run(path,b'that again');saved=json.loads(path.read_text());ra,ro,rm,rlast=restore(saved)
        checks['persistent_process_raw_stdin_produces_memory_grounded_stdout']=(result.stdout==b'the careful engineer tests the sensor.\n' and result.stderr==b'')
        checks['process_exit_atomically_saves_adult_organism_memory_session']=(rm.resolve(ra,ro,b'that again',CHANNEL) is not None and rlast==last)
        blob=path.read_text();checks['saved_process_session_contains_no_stdin_or_stdout_transcript']=(all(token not in blob for token in ('that again','tests the sensor','stdin','stdout','transcript')))

        # Same durable Adult/world with no joint memory has no remembered proposal and no output.
        empty_path=Path(td)/'empty.json';empty=ConsequenceQualifiedJointAttentionMemoryV1();empty_path.write_text(json.dumps(checkpoint(type(adult).restore(copy.deepcopy(adult.checkpoint())),type(o).restore(copy.deepcopy(o.checkpoint())),empty,last),sort_keys=True))
        empty_result=run(empty_path,b'that again')
        checks['physical_process_with_empty_memory_stays_silent_for_same_raw_input']=(empty_result.stdout==b'' and empty_result.stderr==b'')

        # Two raw turns in one process use the same continuing Adult instance; no test-side resolver runs between them.
        two_path=Path(td)/'two.json';two_path.write_text(json.dumps(checkpoint(type(adult).restore(copy.deepcopy(adult.checkpoint())),type(o).restore(copy.deepcopy(o.checkpoint())),ConsequenceQualifiedJointAttentionMemoryV1.restore(copy.deepcopy(memory.checkpoint())),last),sort_keys=True))
        two=subprocess.run([sys.executable,str(TERMINAL),'--resume',str(two_path),'--channel',str(CHANNEL),'--idle-ms','1000'],input=b'that again\nthat again\n',stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=5,check=True)
        checks['one_persistent_process_handles_multiple_raw_memory_turns_without_host_dispatch']=(two.stdout==b'the careful engineer tests the sensor.\nthe careful engineer tests the sensor.\n' and two.stderr==b'')
    checks['bounded_fast_path']=time.perf_counter()-started<2.0
    failed=[k for k,v in checks.items() if not v]
    result={'schema':'cyber-lagoon.reference-embodied-memory-process.v1','contract':'FOUNDRY_EMBODIED_MEMORY_PROCESS_'+('GREEN' if not failed else 'RED'),'pass':not failed,'reference_only':True,'language_phenotype_improved':not failed,'future_update_authority_preserved':True,'visible_language_gain':'A_SINGLE_PERSISTENT_STDIN_STDOUT_BODY_PROCESS_NOW_ANSWERS_RAW_TEXT_FROM_DURABLE_UNNAMED_SHARED_ATTENTION_MEMORY_WITHOUT_TEST_SIDE_RESOLUTION','checks':checks,'failed':failed,'remaining_red':['SAME_PROCESS_VISUAL_GESTURE_GAZE_ACQUISITION_BEFORE_TEXT_RECALL','CONTINUOUS_MULTIMODAL_EVENT_PROTOCOL','DIRECT_EMBODIED_PROCESS_PARITY'],'elapsed_ms':round((time.perf_counter()-started)*1000,3)}
    print(result['contract']);print(json.dumps(result,indent=2,sort_keys=True));return 0 if not failed else 1
if __name__=='__main__':raise SystemExit(main())
