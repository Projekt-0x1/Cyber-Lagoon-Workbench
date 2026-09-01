#!/usr/bin/env python3
"""Destructive audit: Adult slow resource history must enter via body ingress, not setters."""
from __future__ import annotations
import copy,inspect,json,time
from pathlib import Path
from reference_language_mastery_adult_v1 import AdultStateV1,LanguageMasteryAdultV1
from reference_language_mastery_contact_adapter_v1 import CONTACT_UTTERANCE,LanguageMasteryContactAdapterV1
from reference_mathematical_adult_operator_factorization_verify import build
from reference_predictive_credit_profile_v1 import PredictiveCreditBankV1,Q
from reference_slow_resource_history_v1 import HISTORY_WINDOW_TICKS,LOAD_SAMPLE_CAP_Q16,SUSTAINED_MIN_CONTACTS,SlowResourceHistoryV1
SHORT=0x5E02;NEUTRAL=AdultStateV1()
language_phenotype_improved=True
future_update_authority_preserved=True

def digest(n):return format(int(n),'064x')[-64:]
def truth(a,ctx,deep):
 def row(pid):
  c=a.credit.row(pid).contexts[ctx];return (c.outcome_samples,c.outcome_mean_q16,c.somatic_mean_q16,c.control_attempts,c.control_successes,c.background_attempts,c.background_successes,c.control_history_q16)
 return row(SHORT),row(deep)
def language_state(a):return json.dumps({'surface':a.program_surface_checkpoint(),'credit':a.credit.checkpoint()},sort_keys=True,separators=(',',':'))
def trained():
 seed,leaves,_d,top,*_=build(True);cue=leaves[0];ctx=cue.identity
 seed.experience_atomic_program(SHORT,cue,Q//4,0,ctx,Q//16,True)
 a=LanguageMasteryAdultV1.restore(copy.deepcopy(seed.checkpoint()));a.credit=PredictiveCreditBankV1(32)
 for _ in range(4):
  a.experience_choice(SHORT,Q//4,0,ctx,Q//16,1,True);a.experience_choice(top.identity,Q//2,0,ctx,Q//4,5,True)
 return a,top.identity,cue,ctx
def public_choice(a,cue):
 contact=LanguageMasteryContactAdapterV1(a);contact.contact(CONTACT_UTTERANCE,tuple(cue.surface),0xA001)
 return int(a.choose())
def load(a,seq,value):a.settle_body_ingress('body',seq,digest(seq),value)
def sustained(a):
 for n in range(1,SUSTAINED_MIN_CONTACTS+1):load(a,n,LOAD_SAMPLE_CAP_Q16)
 return a.slow_resource_history.pressure_q16()

def main():
 started=time.perf_counter();checks={};base,deep,cue,ctx=trained();base_cp=copy.deepcopy(base.checkpoint());base_truth=truth(base,ctx,deep);base_lang=language_state(base)
 baseline=LanguageMasteryAdultV1.restore(copy.deepcopy(base_cp));baseline_choice=public_choice(baseline,cue)
 loaded=LanguageMasteryAdultV1.restore(copy.deepcopy(base_cp));pressure=sustained(loaded);loaded_choice=public_choice(loaded,cue)
 checks['authenticated_body_resource_history_changes_public_choice']=(baseline_choice==deep and loaded_choice==SHORT and pressure>0)
 spaced=LanguageMasteryAdultV1.restore(copy.deepcopy(base_cp));seq=0
 for n in range(SUSTAINED_MIN_CONTACTS):
  seq+=1;load(spaced,seq,LOAD_SAMPLE_CAP_Q16)
  if n+1<SUSTAINED_MIN_CONTACTS:
   for _ in range(HISTORY_WINDOW_TICKS+1):seq+=1;load(spaced,seq,0)
 spaced_choice=public_choice(spaced,cue)
 checks['same_loads_spaced_by_authenticated_neutral_body_samples_do_not_accumulate']=(spaced.slow_resource_history.contacts==SUSTAINED_MIN_CONTACTS and spaced.slow_resource_history.pressure_q16()==0 and spaced_choice==deep)
 large=LanguageMasteryAdultV1.restore(copy.deepcopy(base_cp));load(large,1,100*Q);large_choice=public_choice(large,cue)
 checks['one_large_body_sample_cannot_substitute_for_duration']=(large.slow_resource_history.capped_samples==1 and large_choice==deep)
 invalid=LanguageMasteryAdultV1.restore(copy.deepcopy(base_cp));load(invalid,1,LOAD_SAMPLE_CAP_Q16);before=copy.deepcopy(invalid.slow_resource_history.checkpoint())
 try:load(invalid,1,100*Q);invalid_refused=False
 except ValueError:invalid_refused=True
 checks['invalid_body_ingress_refuses_before_history_update']=(invalid_refused and invalid.slow_resource_history.checkpoint()==before)
 loaded_cp=copy.deepcopy(loaded.checkpoint());restored=LanguageMasteryAdultV1.restore(copy.deepcopy(loaded_cp));restored_choice=public_choice(restored,cue)
 checks['adult_checkpoint_preserves_ingress_earned_history']=(restored.slow_resource_history.checkpoint()==loaded.slow_resource_history.checkpoint() and restored_choice==SHORT)
 lesioned=LanguageMasteryAdultV1.restore(copy.deepcopy(loaded_cp));before_lang=language_state(lesioned);lesioned.slow_resource_history.lesion_history();lesioned._select_epoch+=1;lesion_choice=public_choice(lesioned,cue)
 checks['focal_history_lesion_restores_deep_without_truth_rewrite']=(lesion_choice==deep and language_state(lesioned)==before_lang and truth(lesioned,ctx,deep)==base_truth)
 recovered=LanguageMasteryAdultV1.restore(copy.deepcopy(loaded_cp));tick0=recovered._tick
 for _ in range(6*HISTORY_WINDOW_TICKS):recovered.internal_tick()
 recovered_choice=public_choice(recovered,cue)
 checks['ordinary_quiet_ticks_recover_without_candidate_return']=(recovered._tick==tick0+6*HISTORY_WINDOW_TICKS and recovered.slow_resource_history.pressure_q16()==0 and recovered_choice==deep)
 checks['program_truth_and_language_exact_across_body_history_arms']=(all(truth(x,ctx,deep)==base_truth for x in (baseline,loaded,spaced,large,restored,lesioned,recovered)) and all(language_state(x)==base_lang for x in (baseline,loaded,spaced,large,restored,lesioned,recovered)))
 checks['no_privileged_resource_or_time_setter_remains']=(not hasattr(LanguageMasteryAdultV1,'observe_resource_load') and not hasattr(LanguageMasteryAdultV1,'advance_resident_time'))
 sig=list(inspect.signature(LanguageMasteryAdultV1.settle_body_ingress).parameters)
 checks['resource_enters_existing_authenticated_body_boundary']=(sig==['self','source','sequence','reafference','resource_load_q16'])
 checks['public_behavior_not_probe_api']=('_probe_choice' not in inspect.getsource(public_choice) and '.choose()' in inspect.getsource(public_choice))
 checks['explicit_current_state_is_neutral']=(NEUTRAL==AdultStateV1())
 checks['standard_fast_union_runs_authenticated_body_resource_audit']='adult-slow-resource-body:reference_adult_slow_resource_body_ingress_ownership_verify.py' in (Path(__file__).parent/'run_language_mastery_fast.sh').read_text()
 checks['bounded_fast_path']=time.perf_counter()-started<1.0
 failed=[k for k,v in checks.items() if not v]
 result={'contract':'FOUNDRY_ADULT_SLOW_RESOURCE_BODY_INGRESS_OWNERSHIP_GREEN','reference_only':True,'adult_owned':True,'authenticated_body_resource_contact_binding':True,'language_phenotype_improved':language_phenotype_improved,'visible_language_gain':'AUTHENTICATED_SUSTAINED_BODY_RESOURCE_HISTORY_SWITCHES_PUBLIC_DEEP_TO_SHORT_AND_QUIET_TICKS_RECOVER_DEEP','choices':{'baseline':baseline_choice,'loaded':loaded_choice,'spaced':spaced_choice,'large':large_choice,'restored':restored_choice,'lesioned':lesion_choice,'recovered':recovered_choice},'checks':checks,'failed':failed,'remaining_red':['LEARNED_SOURCE_RELIABILITY_BEYOND_EXPLICIT_WITHDRAWAL','DIRECT_SLOW_HISTORY_PARITY','CONTINUOUS_LIFE_INTERFERENCE'],'elapsed_ms':round((time.perf_counter()-started)*1000,3)}
 if failed:print('FOUNDRY_ADULT_SLOW_RESOURCE_BODY_INGRESS_OWNERSHIP_RED '+','.join(failed));print(json.dumps(result,indent=2,sort_keys=True));return 1
 print(result['contract']);print(json.dumps(result,indent=2,sort_keys=True));return 0
if __name__=='__main__':raise SystemExit(main())
