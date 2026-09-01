#!/usr/bin/env python3
import json,time
from reference_authenticated_omission_window_v1 import *

def main():
    t=time.perf_counter(); e=ExpectedOutcomeV1(0xA1,77,20,'the valve should open.','the valve did not open.')
    pre=TickOnlyOmissionPredecessor(e); cur=AuthenticatedOmissionWindowV1(e)
    pre.advance_internal(21); cur.advance_internal(21)
    before=pre.discuss(); idle=cur.discuss()
    wrong_source=cur.contact(ExternalProgressV1(0xA2,77,22,True))
    endogenous=cur.contact(ExternalProgressV1(0xA1,77,22,False))
    wrong_context=cur.contact(ExternalProgressV1(0xA1,78,22,True))
    closed=cur.contact(ExternalProgressV1(0xA1,77,22,True)); duplicate=cur.contact(ExternalProgressV1(0xA1,77,23,True))
    after=cur.discuss()
    checks={
      'predecessor_falsely_retracts_on_idle_processor_time':before=='the valve did not open.',
      'corrected_adult_preserves_expectation_without_world_evidence':idle=='the valve should open.',
      'wrong_source_does_not_close':not wrong_source,
      'endogenous_activity_does_not_close':not endogenous,
      'wrong_context_does_not_close':not wrong_context,
      'authenticated_later_same_source_context_closes':closed and after=='the valve did not open.',
      'closure_is_exactly_once':not duplicate and cur.closure_tick==22,
      'visible_discussion_improvement':before!=idle and idle=='the valve should open.' and after=='the valve did not open.',
      'bounded':time.perf_counter()-t<0.1,
    }
    failed=[k for k,v in checks.items() if not v]
    print('FOUNDRY_AUTHENTICATED_OMISSION_LANGUAGE_'+('GREEN' if not failed else 'RED'))
    print("predecessor_after_idle=['the valve did not open.']")
    print("corrected_after_idle=['the valve should open.']")
    print("corrected_after_authenticated_closure=['the valve did not open.']")
    print(json.dumps({'checks':checks,'failed':failed,'language_phenotype_improved':not failed,'visible_language_gain':'IDLE_TIME_NO_LONGER_FABRICATES_OMISSION_DISCOURSE','reference_only':True},indent=2,sort_keys=True))
    return 0 if not failed else 1
if __name__=='__main__': raise SystemExit(main())
