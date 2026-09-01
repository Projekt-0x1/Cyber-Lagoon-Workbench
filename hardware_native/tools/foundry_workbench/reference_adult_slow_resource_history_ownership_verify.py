#!/usr/bin/env python3
"""Adult-owned slow resource history across program and long-form discourse competition."""
from __future__ import annotations

import copy
import inspect
import json
import time

from reference_language_mastery_adult_v1 import AdultStateV1,LanguageMasteryAdultV1
from reference_mathematical_adult_operator_factorization_verify import build
from reference_partner_specific_discourse_selection_verify import prepare,establish_partner,select
from reference_partner_specific_pragmatic_language_verify import PARTNER_A
from reference_predictive_credit_profile_v1 import PredictiveCreditBankV1,Q
from reference_slow_resource_history_v1 import (
    CHRONIC_GATE_Q16,HISTORY_WINDOW_TICKS,LOAD_SAMPLE_CAP_Q16,
    SUSTAINED_MIN_CONTACTS,SlowResourceHistoryV1,
)

CTX=0x5D01;SHORT=0x5D02;NEUTRAL=AdultStateV1()


def truth(a,deep):
    def row(pid):
        c=a.credit.row(pid).contexts[CTX]
        return (c.outcome_samples,c.outcome_mean_q16,c.somatic_mean_q16,
                c.control_attempts,c.control_successes,c.background_attempts,
                c.background_successes,c.control_history_q16)
    return row(SHORT),row(deep)


def language_state(a):
    return json.dumps({
        'surface':a.program_surface_checkpoint(),
        'credit':a.credit.checkpoint(),
        'organization':a.organization_credit.checkpoint(),
        'discourse':a.discourse_credit.checkpoint(),
        'partner':a.partner_credit.checkpoint(),
    },sort_keys=True,separators=(',',':'))


def trained_adult():
    seed,leaves,_d,top,*_=build(True)
    seed.experience_atomic_program(SHORT,leaves[0],Q//4,0,CTX,Q//16,True)
    a=LanguageMasteryAdultV1.restore(copy.deepcopy(seed.checkpoint()))
    a.credit=PredictiveCreditBankV1(32)
    for _ in range(4):
        a.experience_choice(SHORT,Q//4,0,CTX,Q//16,1,True)
        a.experience_choice(top.identity,Q//2,0,CTX,Q//4,5,True)
    return a,top.identity


def choose(a):return int(a._probe_choice(CTX,NEUTRAL))


def digest(n):return format(int(n),'064x')[-64:]


def body_load(a,seq,value):
    return a.settle_body_ingress('resource-body',int(seq),digest(seq),int(value))


def sustained(a):
    for seq in range(1,SUSTAINED_MIN_CONTACTS+1):body_load(a,seq,LOAD_SAMPLE_CAP_Q16)
    return a.slow_resource_history.pressure_q16()


def main():
    started=time.perf_counter();checks={}
    base,deep=trained_adult();base_cp=copy.deepcopy(base.checkpoint());base_truth=truth(base,deep);base_lang=language_state(base)
    baseline_choice=choose(base);baseline_cached=choose(base)

    loaded=LanguageMasteryAdultV1.restore(copy.deepcopy(base_cp));loaded_before=choose(loaded);loaded_pressure=sustained(loaded);loaded_choice=choose(loaded)
    checks['adult_owns_sustained_history_and_invalidates_cached_choice']=(
        loaded_before==deep and loaded_choice==SHORT and loaded_pressure>=CHRONIC_GATE_Q16)

    spaced=LanguageMasteryAdultV1.restore(copy.deepcopy(base_cp));seq=0
    for n in range(SUSTAINED_MIN_CONTACTS):
        seq+=1;body_load(spaced,seq,LOAD_SAMPLE_CAP_Q16)
        if n+1<SUSTAINED_MIN_CONTACTS:
            for _ in range(HISTORY_WINDOW_TICKS+1):seq+=1;body_load(spaced,seq,0)
    spaced_choice=choose(spaced)
    checks['adult_spaced_yoked_contacts_do_not_earn_chronic_pressure']=(
        spaced.slow_resource_history.contacts==SUSTAINED_MIN_CONTACTS
        and spaced.slow_resource_history.pressure_q16()==0 and spaced_choice==deep)

    large=LanguageMasteryAdultV1.restore(copy.deepcopy(base_cp));body_load(large,1,100*Q);large_choice=choose(large)
    checks['adult_one_large_sample_cannot_substitute_for_duration']=(
        large.slow_resource_history.capped_samples==1
        and large.slow_resource_history.load_accumulator_q16==LOAD_SAMPLE_CAP_Q16
        and large.slow_resource_history.pressure_q16()==0 and large_choice==deep)

    loaded_cp=copy.deepcopy(loaded.checkpoint());restored=LanguageMasteryAdultV1.restore(copy.deepcopy(loaded_cp));restored_choice=choose(restored)
    checks['adult_checkpoint_owns_slow_resource_history']=(
        restored.slow_resource_history.checkpoint()==loaded.slow_resource_history.checkpoint()
        and restored_choice==SHORT)

    lesioned=LanguageMasteryAdultV1.restore(copy.deepcopy(loaded_cp));before_lesion_lang=language_state(lesioned);lesioned.slow_resource_history.lesion_history();lesioned._select_epoch+=1;lesion_choice=choose(lesioned)
    checks['focal_adult_history_lesion_restores_deep_without_truth_rewrite']=(
        lesion_choice==deep and language_state(lesioned)==before_lesion_lang and truth(lesioned,deep)==base_truth)

    recovered=LanguageMasteryAdultV1.restore(copy.deepcopy(loaded_cp))
    for _ in range(6*HISTORY_WINDOW_TICKS):recovered.internal_tick()
    recovered_choice=choose(recovered)
    checks['adult_quiet_time_recovers_without_candidate_return']=(
        recovered.slow_resource_history.pressure_q16()==0
        and recovered.slow_resource_history.recovery_events>0 and recovered_choice==deep)

    checks['program_specific_truth_and_language_exact_across_history_arms']=(
        all(truth(x,deep)==base_truth for x in (base,loaded,spaced,large,restored,lesioned,recovered))
        and all(language_state(x)==base_lang for x in (base,loaded,spaced,large,restored,lesioned,recovered)))
    checks['explicit_current_state_remains_neutral']=(NEUTRAL==AdultStateV1())

    # Long-form discourse: the Adult itself owns the prior load.  Use the current
    # public partner-discourse fixture rather than importing its private constants.
    # Partner A's even proposition set is trained with higher duration so sustained
    # history can suppress some resident matter under the same neutral explicit state.
    discourse,frontier,_generic=prepare()
    # Establish A's ordinary even-proposition precedent, but keep exactly one
    # long-duration detail marginal enough for the history pressure to suppress.
    from reference_partner_specific_pragmatic_language_verify import social_contact
    social_contact(discourse,PARTNER_A,0x5D20)
    for i,leaf in enumerate(frontier):
        if i%2:
            outcome,effort,duration=-Q,Q//16,1
        elif i==0:
            outcome,effort,duration=Q//4,Q//16,4
        else:
            outcome,effort,duration=Q,Q//16,1
        for _ in range(2):
            discourse.experience_partner_choice(
                leaf.identity,outcome,effort_q16=effort,duration=duration)
            discourse.experience_partner_background(leaf.identity,False)
    _pre_root,pre_ids,_pre_surface,_pre_generic,_pre_partner=select(
        discourse,frontier,PARTNER_A,0x5D21,NEUTRAL)
    sustained(discourse)
    _loaded_root,loaded_ids,_loaded_surface,_loaded_generic,_loaded_partner=select(
        discourse,frontier,PARTNER_A,0x5D22,NEUTRAL)
    for _ in range(6*HISTORY_WINDOW_TICKS):discourse.internal_tick()
    _rec_root,recovered_ids,_rec_surface,_rec_generic,_rec_partner=select(
        discourse,frontier,PARTNER_A,0x5D23,NEUTRAL)
    checks['adult_slow_history_modulates_long_form_discourse_under_matched_current_state']=(
        len(pre_ids)>0 and len(loaded_ids)<len(pre_ids) and recovered_ids==pre_ids)
    checks['slow_history_does_not_author_new_discourse_content']=(set(loaded_ids).issubset(set(pre_ids)))

    legacy=copy.deepcopy(base_cp);legacy.pop('slow_resource_history',None);legacy_adult=LanguageMasteryAdultV1.restore(legacy)
    checks['legacy_checkpoint_without_slow_history_restores_empty_history']=(
        legacy_adult.slow_resource_history.checkpoint()==SlowResourceHistoryV1().checkpoint()
        and choose(legacy_adult)==deep)

    history_source=inspect.getsource(SlowResourceHistoryV1)
    checks['adult_resource_ingress_is_content_free']=(
        list(inspect.signature(LanguageMasteryAdultV1.settle_body_ingress).parameters)
        ==['self','source','sequence','reafference','resource_load_q16']
        and not hasattr(LanguageMasteryAdultV1,'observe_resource_load')
        and not hasattr(LanguageMasteryAdultV1,'advance_resident_time')
        and all(token not in history_source for token in ('program','language','partner','grammar','surface')))
    checks['bounded_fast_path']=time.perf_counter()-started<1.0

    failed=[k for k,v in checks.items() if not v]
    result={
        'contract':'FOUNDRY_ADULT_SLOW_RESOURCE_HISTORY_OWNERSHIP_GREEN',
        'reference_only':True,'adult_owned':True,
        'choices':{'baseline':baseline_choice,'loaded':loaded_choice,'spaced':spaced_choice,
                   'large':large_choice,'restored':restored_choice,'lesioned':lesion_choice,'recovered':recovered_choice},
        'pressure':{'loaded_q16':loaded_pressure,'recovered_q16':recovered.slow_resource_history.pressure_q16()},
        'discourse_counts':{'baseline':len(pre_ids),'loaded':len(loaded_ids),'recovered':len(recovered_ids)},
        'checks':checks,'failed':failed,
        'remaining_red':['DIRECT_SLOW_HISTORY_PARITY','CONTINUOUS_LIFE_INTERFERENCE'],
        'elapsed_ms':round((time.perf_counter()-started)*1000,3),
    }
    if failed:
        print('FOUNDRY_ADULT_SLOW_RESOURCE_HISTORY_OWNERSHIP_RED '+','.join(failed));print(json.dumps(result,indent=2,sort_keys=True));return 1
    print(result['contract']);print(json.dumps(result,indent=2,sort_keys=True));return 0

if __name__=='__main__':raise SystemExit(main())
