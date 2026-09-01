#!/usr/bin/env python3
from __future__ import annotations
import copy,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
from reference_life_function_curriculum_v1 import (
    LifeCurriculumEventV2,LifeFunctionCurriculumV2,ReferenceLifeFunctionRuntimeV2,
    canonical_life_function_curriculum_v2,canonical_species_program_v2,
)
from reference_life_extension_history_matrix_v1 import B


def at_mark(curriculum,species,mark):
    cursor=curriculum.mark_cursor(mark)
    return ReferenceLifeFunctionRuntimeV2(species).run(LifeFunctionCurriculumV2(curriculum.events[:cursor]))


def main():
    started=time.perf_counter();checks={}
    curriculum=canonical_life_function_curriculum_v2();species=canonical_species_program_v2()
    runtime=at_mark(curriculum,species,'endogenous_state_inquiry_loaded');adult=runtime.adult
    expected=adult.endogenous_state_inquiry(B);before=copy.deepcopy(adult.checkpoint());base=runtime.cursor
    first=runtime.apply(LifeCurriculumEventV2(base+1,'endogenous_inquiry_opportunity',0xE510,(B,)))
    second=runtime.apply(LifeCurriculumEventV2(base+2,'endogenous_inquiry_opportunity',0xE511,(B,)))
    action_id=int(first[1]) if first else 0
    checks['natural_runtime_opportunity_originates_learned_inquiry']=(bool(expected) and first[0]==expected and action_id>0)
    checks['repeated_idle_opportunity_is_inhibited_while_need_remains']=(second==(b'',0) and len(adult.pending_endogenous_inquiry_actions)==1)
    pending=copy.deepcopy(adult.checkpoint());restarted=type(adult).restore(copy.deepcopy(pending))
    checks['pending_turn_inhibition_survives_checkpoint_exactly']=(restarted.checkpoint()==pending and restarted.externalize_endogenous_inquiry(0xE512,B)==(b'',None))
    before_motor=adult.settle_endogenous_inquiry_resolution(next(iter(adult.pending_endogenous_inquiry_actions.values())),0xE513)
    motor=runtime.apply(LifeCurriculumEventV2(base+3,'endogenous_inquiry_motor_return',B,(base+1,)))
    unrelated=adult.settle_endogenous_inquiry_resolution(next(iter(adult.pending_endogenous_inquiry_actions.values())),0xE513)
    runtime.apply(LifeCurriculumEventV2(base+4,'quiet',0,(64,)))
    closed=runtime.apply(LifeCurriculumEventV2(base+5,'endogenous_inquiry_resolution',0xE513,(base+1,)))
    third=runtime.apply(LifeCurriculumEventV2(base+6,'endogenous_inquiry_opportunity',0xE514,(B,)))
    checks['only_motor_realized_evidence_obsolete_inquiry_deactivates']=(not before_motor and motor is True and not unrelated and closed is True and third==(b'',0) and adult.endogenous_state_inquiry(B) is None)
    checks['public_inquiry_does_not_mutate_language_or_world_truth']=before['language_adult']['language']==adult.checkpoint()['language_adult']['language'] and before['language_adult']['world_causal_learning']==adult.checkpoint()['language_adult']['world_causal_learning']
    failed=[k for k,v in checks.items() if not v]
    result={'schema':'cyber-lagoon.endogenous-inquiry-transport.v2','pass':not failed,'checks':checks,'failed':failed,'question':(expected or b'').decode(errors='replace'),'public_count':adult.endogenous_inquiry_public_count,'elapsed_ms':round((time.perf_counter()-started)*1000,3),'claim':'MOTOR_REALIZED_SELF_STATE_INQUIRY_PERSISTS_THROUGH_UNRELATED_CONTACT_AND_DEACTIVATES_ONLY_AFTER_BODY_RECOVERY','runtime_llm':False,'reference_only':True}
    print('FOUNDRY_ENDOGENOUS_INQUIRY_TRANSPORT '+('GREEN' if not failed else 'RED'))
    print(json.dumps(result,indent=2,sort_keys=True));return 0 if not failed else 1

if __name__=='__main__':raise SystemExit(main())
