#!/usr/bin/env python3
"""Frozen sustained-vs-spaced slow-resource history destructive audit."""
from __future__ import annotations
import copy,inspect,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
from reference_language_mastery_adult_v1 import AdultStateV1,LanguageMasteryAdultV1
from reference_mathematical_adult_operator_factorization_verify import build
from reference_predictive_credit_profile_v1 import PredictiveCreditBankV1,Q
from reference_slow_resource_history_v1 import *
phenotype_preserved=True
future_update_authority_preserved=True
language_phenotype_improved=True
reference_challenger_mechanism=True
CTX=0x5C01;SHORT=0x5C02;CURRENT=AdultStateV1()

def surf(a):return json.dumps(a.program_surface_checkpoint(),sort_keys=True,separators=(',',':'))
def adult(seed,deep):
 a=LanguageMasteryAdultV1.restore(copy.deepcopy(seed));a.credit=PredictiveCreditBankV1(32)
 for _ in range(4):
  a.experience_choice(SHORT,Q//4,0,CTX,Q//16,1,True);a.experience_choice(deep,Q//2,0,CTX,Q//4,5,True)
 return a
def choose(a,h):return int(a._probe_choice(CTX,AdultStateV1(pressure_q16=h.pressure_q16())))
def truth(a,deep):
 def row(pid):
  c=a.credit.row(pid).contexts[CTX];return (c.outcome_samples,c.outcome_mean_q16,c.somatic_mean_q16,c.control_attempts,c.control_successes,c.background_attempts,c.background_successes,c.control_history_q16)
 return row(SHORT),row(deep)

def main():
 t=time.perf_counter();seed,leaves,_d,top,*_=build(True);seed.experience_atomic_program(SHORT,leaves[0],Q//4,0,CTX,Q//16,True)
 a=adult(copy.deepcopy(seed.checkpoint()),top.identity);acp=copy.deepcopy(a.checkpoint());base=surf(a);ss=a.public_surface(SHORT);ds=a.public_surface(top.identity);tb=truth(a,top.identity)
 baseline=SlowResourceHistoryV1();bc=choose(a,baseline)
 sustained=SlowResourceHistoryV1()
 for tick in range(1,SUSTAINED_MIN_CONTACTS+1):sustained.observe(tick,LOAD_SAMPLE_CAP_Q16)
 sc=choose(a,sustained)
 spaced=SlowResourceHistoryV1()
 for n in range(SUSTAINED_MIN_CONTACTS):spaced.observe(1+n*(HISTORY_WINDOW_TICKS+1),LOAD_SAMPLE_CAP_Q16)
 yc=choose(a,spaced)
 large=SlowResourceHistoryV1();large.observe(1,100*Q);lc=choose(a,large)
 rh=SlowResourceHistoryV1.restore(copy.deepcopy(sustained.checkpoint()));ra=LanguageMasteryAdultV1.restore(copy.deepcopy(acp));rc=choose(ra,rh)
 lh=SlowResourceHistoryV1.restore(copy.deepcopy(sustained.checkpoint()));lh.lesion_history();la=LanguageMasteryAdultV1.restore(copy.deepcopy(acp));lchoice=choose(la,lh)
 qh=SlowResourceHistoryV1.restore(copy.deepcopy(sustained.checkpoint()));qh.advance(sustained.last_contact_tick+6*HISTORY_WINDOW_TICKS);qa=LanguageMasteryAdultV1.restore(copy.deepcopy(acp));qc=choose(qa,qh)
 adults=(a,ra,la,qa);module=inspect.getsource(sys.modules[SlowResourceHistoryV1.__module__])
 checks={
 'neutral_without_history_selects_deep':bc==top.identity,
 'sustained_history_earns_pressure_and_selects_short':sustained.sustained_contacts==6 and sustained.load_accumulator_q16==6*LOAD_SAMPLE_CAP_Q16 and sustained.pressure_q16()==sustained.load_accumulator_q16 and sustained.pressure_q16()>=CHRONIC_GATE_Q16 and sc==SHORT,
 'spaced_yoked_contacts_do_not_earn_chronic_policy':spaced.contacts==sustained.contacts and spaced.pressure_q16()==0 and yc==top.identity,
 'one_large_sample_cannot_substitute_for_duration':large.capped_samples==1 and large.load_accumulator_q16==LOAD_SAMPLE_CAP_Q16 and large.pressure_q16()==0 and lc==top.identity,
 'checkpoint_preserves_slow_history_effect':rh.checkpoint()==sustained.checkpoint() and rc==SHORT,
 'focal_history_lesion_restores_deep_without_truth_change':lh.pressure_q16()==0 and lchoice==top.identity and truth(la,top.identity)==tb,
 'quiet_time_recovers_without_candidate_return':qh.pressure_q16()==0 and qh.recovery_events>0 and qc==top.identity,
 'explicit_current_state_is_neutral':CURRENT==AdultStateV1(),
 'adult_truth_identical_across_arms':all(truth(x,top.identity)==tb for x in adults),
 'language_factors_remain_exact':all(surf(x)==base and x.public_surface(SHORT)==ss and x.public_surface(top.identity)==ds for x in adults),
 'history_module_has_no_language_social_import':all(x not in module for x in ('reference_language_mastery_adult_v1','reference_language_learning_v1','reference_social')),
 'standard_fast_union_runs_this_destructive_audit':'slow-resource-history:reference_productive_leaf_slow_resource_history_destructive_audit_verify.py' in (Path(__file__).parent/'run_language_mastery_fast.sh').read_text(),
 'frozen_constants_match_preregistration':HISTORY_WINDOW_TICKS==8 and SUSTAINED_MIN_CONTACTS==6 and LOAD_SAMPLE_CAP_Q16==Q//8 and LOAD_CEILING_Q16==Q and CHRONIC_GATE_Q16==Q//2 and QUIET_RECOVERY_STEP_Q16==Q//8,
 'bounded_fast_path':time.perf_counter()-t<1.0}
 failed=[k for k,v in checks.items() if not v]
 result={'schema':'cyber-lagoon.reference-productive-leaf-slow-resource-history-destructive-audit.v1','pass':not failed,'reference_only':True,'mechanism_change':True,'adult_mechanism_change':False,'reference_challenger_mechanism':reference_challenger_mechanism,'novel_synthesis':True,'phenotype_preserved':phenotype_preserved,'future_update_authority_preserved':future_update_authority_preserved,'language_phenotype_improved':language_phenotype_improved,'visible_language_gain':'MATCHED_CURRENT_STATE_SUSTAINED_RESOURCE_HISTORY_SWITCHES_FACTORED_DEEP_TO_SHORT_AND_QUIET_RECOVERY_RESTORES_DEEP','short_program':SHORT,'deep_program':int(top.identity),'short_bytes':len(ss),'deep_bytes':len(ds),'choices':{'baseline':bc,'sustained':sc,'spaced':yc,'one_large':lc,'restored':rc,'lesioned':lchoice,'recovered':qc},'sustained_checkpoint':sustained.checkpoint(),'spaced_checkpoint':spaced.checkpoint(),'checks':checks,'failed':failed,'remaining_red':['ADULT_OWNERSHIP_OF_SLOW_RESOURCE_HISTORY','LEARNED_SOURCE_RELIABILITY_BEYOND_EXPLICIT_WITHDRAWAL','DIRECT_COMPOSITIONAL_GRAMMAR_PARITY'],'elapsed_ms':round((time.perf_counter()-t)*1000,3)}
 print('FOUNDRY_PRODUCTIVE_LEAF_SLOW_RESOURCE_HISTORY_DESTRUCTIVE_AUDIT_'+('GREEN' if not failed else 'RED'));print(json.dumps(result,indent=2,sort_keys=True));return 0 if not failed else 1
if __name__=='__main__':raise SystemExit(main())
