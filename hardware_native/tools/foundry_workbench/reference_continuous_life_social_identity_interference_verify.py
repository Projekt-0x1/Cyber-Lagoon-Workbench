#!/usr/bin/env python3
"""Continuous-life-style interference audit for source-derived social identity."""
from __future__ import annotations
import copy,json,time
from reference_multimodal_social_identity_convergence_verify import NEW_SOURCE,contact,pair
from reference_partner_specific_pragmatic_language_verify import (
    ACTION,OBJECT,PARTNER_A,P_COMPACT,P_EXPLICIT,SOURCE_A,STATE,establish,prepare,
)
from reference_partner_specific_discourse_selection_verify import (
    establish_partner,prepare as prepare_discourse,
)
from reference_predictive_credit_profile_v1 import Q
from reference_slow_resource_history_v1 import LOAD_SAMPLE_CAP_Q16,SUSTAINED_MIN_CONTACTS,HISTORY_WINDOW_TICKS

N_INTERFERENCE_BRIDGES=32
BODY='identity-life-body'


def digest(n):return format(int(n),'064x')[-64:]


def body_load(a,seq,value):
    return a.settle_body_ingress(BODY,int(seq),digest(seq),int(value))


def seed_anchor(a,agent,source):
    for _ in range(2):
        a.observe_social_contact(agent,OBJECT,STATE,ACTION,source)
        if not a.observe_social_behavior(agent,OBJECT,ACTION,True):
            raise RuntimeError('identity_interference:anchor_behavior')
    if a._resolved_social_source(source)!=source:
        raise RuntimeError('identity_interference:anchor_unresolved')


def learn_bridge(a,agent,anchor,alternate):
    seed_anchor(a,agent,anchor)
    for _ in range(2):
        a.observe_social_contact(agent,OBJECT,STATE,ACTION,anchor)
        a.observe_social_contact(agent,OBJECT,STATE,ACTION,alternate)
        a.experience_recent_social_source_continuity()
        a.experience_recent_social_source_background(False)
    if a._resolved_social_source(alternate)!=anchor:
        raise RuntimeError('identity_interference:bridge_unresolved')


def train_a_short():
    a,compact,explicit,generic=prepare();establish(a)
    for _ in range(2):pair(a,SOURCE_A,NEW_SOURCE)
    contact(a,NEW_SOURCE);choice=a.choose()
    if choice!=P_COMPACT:raise RuntimeError('identity_interference:a_setup')
    return a,compact,explicit,generic


def main():
    started=time.perf_counter();checks={}
    adult,compact,explicit,generic=train_a_short()
    a_partner_context=int(adult._current_partner_context)
    initial_a_candidates=tuple(adult.social_identity_credit.candidates(NEW_SOURCE))
    initial_credit=copy.deepcopy(adult.social_identity_credit.checkpoint())
    initial_partner=copy.deepcopy(adult.partner_credit.checkpoint())
    initial_language=json.dumps(adult.program_surface_checkpoint(),sort_keys=True,separators=(',',':'))

    # Later life: learn many source-local identities/bridges unrelated to A.
    unrelated=[]
    for i in range(N_INTERFERENCE_BRIDGES):
        anchor=0xE100+i*2;alternate=anchor+1;agent=0xF100+i
        learn_bridge(adult,agent,anchor,alternate);unrelated.append((anchor,alternate,agent))
        # Additional unpaired contact adds traffic without plasticity toward A.
        adult.observe_social_contact(agent,OBJECT,STATE,ACTION,0xF800+i)

    rows_after_interference=len(adult.social_identity_credit.rows)
    contexts_after_interference=len(adult.social_identity_credit.context_members)
    a_candidates_after=tuple(adult.social_identity_credit.candidates(NEW_SOURCE))
    contact(adult,NEW_SOURCE);a_after_choice=adult.choose();a_after_context=int(adult._current_partner_context)
    checks['early_a_bridge_survives_32_later_unrelated_bridges']=(
        a_after_choice==P_COMPACT and a_after_context==a_partner_context
        and a_candidates_after==initial_a_candidates==(SOURCE_A,))
    checks['lifetime_identity_bank_grows_while_a_retrieval_stays_source_local']=(
        rows_after_interference>=N_INTERFERENCE_BRIDGES+2
        and contexts_after_interference>=N_INTERFERENCE_BRIDGES+2
        and len(a_candidates_after)==1)

    # Sapolsky resource-history arm: body load occurs during unrelated life but must
    # not rewrite identity truth. Reentry is tested after ordinary quiet recovery.
    seq=0
    for _ in range(SUSTAINED_MIN_CONTACTS):seq+=1;body_load(adult,seq,LOAD_SAMPLE_CAP_Q16)
    loaded_pressure=adult.slow_resource_history.pressure_q16()
    for _ in range(6*HISTORY_WINDOW_TICKS):adult.internal_tick()
    recovered_pressure=adult.slow_resource_history.pressure_q16()
    contact(adult,NEW_SOURCE);after_resource_choice=adult.choose()
    checks['resource_history_modulates_economics_without_erasing_a_identity']=(
        loaded_pressure>0 and recovered_pressure==0 and after_resource_choice==P_COMPACT
        and tuple(adult.social_identity_credit.candidates(NEW_SOURCE))==(SOURCE_A,))

    checkpoint=copy.deepcopy(adult.checkpoint());restored=type(adult).restore(checkpoint)
    checks['checkpoint_drops_active_identity_occurrence_after_long_interference']=(
        restored._current_social_source==0 and restored._recent_social_source==0
        and restored._current_partner_context==0 and restored.choose()==0)
    contact(restored,NEW_SOURCE);restored_a=restored.choose()
    checks['checkpoint_reentry_recovers_early_a_after_interference']=(restored_a==P_COMPACT)

    # Sample unrelated bridges after checkpoint; all must still resolve exactly.
    sampled=unrelated[::7]
    unrelated_ok=True
    for anchor,alternate,agent in sampled:
        restored.observe_social_contact(agent,OBJECT,STATE,ACTION,alternate)
        unrelated_ok=unrelated_ok and restored._resolved_social_source(alternate)==anchor
    checks['unrelated_identity_bridges_survive_checkpoint_too']=unrelated_ok

    # Focal recent adverse history revises A only. Unrelated bridges must remain exact.
    contact(restored,NEW_SOURCE);before_revision=restored.choose()
    for _ in range(5):
        restored.observe_social_contact(PARTNER_A,OBJECT,STATE,ACTION,SOURCE_A)
        restored.observe_social_contact(PARTNER_A,OBJECT,STATE,ACTION,NEW_SOURCE)
        restored.experience_recent_social_source_background(True)
    after_revision=restored.choose()
    unrelated_after_revision=True
    for anchor,alternate,agent in sampled:
        restored.observe_social_contact(agent,OBJECT,STATE,ACTION,alternate)
        unrelated_after_revision=unrelated_after_revision and restored._resolved_social_source(alternate)==anchor
    checks['focal_a_adverse_history_changes_a_not_unrelated_identities']=(
        before_revision==P_COMPACT and after_revision==P_EXPLICIT and unrelated_after_revision)

    # Reacquire A through the no-anchor recent-source path and prove partner/language
    # truth was not globally rewritten by interference or revision.
    for _ in range(5):pair(restored,SOURCE_A,NEW_SOURCE)
    reacquired=restored.choose()
    checks['a_reacquires_after_interference_without_global_relearning']=(reacquired==P_COMPACT)
    checks['partner_and_language_truth_survive_identity_interference']=(
        restored.partner_credit.checkpoint()==initial_partner
        and json.dumps(restored.program_surface_checkpoint(),sort_keys=True,separators=(',',':'))==initial_language)

    # Plasticity remains open after the long epoch: learn a genuinely new bridge now.
    post_anchor=0xEE10;post_alt=0xEE11;post_agent=0xFE10
    learn_bridge(restored,post_agent,post_anchor,post_alt)
    restored.observe_social_contact(post_agent,OBJECT,STATE,ACTION,post_alt)
    checks['new_identity_bridge_can_be_learned_after_long_interference']=(
        restored._resolved_social_source(post_alt)==post_anchor)

    # Separate long-form phenotype must survive the same style of unrelated identity
    # development and return A-specific matter rather than generic 16-proposition output.
    discourse,frontier,_=prepare_discourse()
    establish_partner(discourse,frontier,PARTNER_A,0,0xEF20)
    for _ in range(2):pair(discourse,SOURCE_A,NEW_SOURCE)
    for i in range(N_INTERFERENCE_BRIDGES):
        learn_bridge(discourse,0xFA00+i,0xEA00+i*2,0xEA01+i*2)
    contact(discourse,NEW_SOURCE);root=discourse.organize_relevant_frontier(frontier)
    rows=tuple(discourse.last_discourse_selected)
    even=tuple(leaf.identity for i,leaf in enumerate(frontier) if i%2==0)
    checks['a_long_form_discourse_survives_unrelated_identity_interference']=(
        root is not None and rows==even and len(rows)==8)

    checks['no_person_table_or_global_identity_router']=(
        not hasattr(adult,'person_table') and not hasattr(adult,'identity_router')
        and not hasattr(adult,'modality_identity_map'))
    checks['bounded_reference_work']=time.perf_counter()-started<1.0

    failed=[k for k,v in checks.items() if not v]
    result={
        'contract':'FOUNDRY_CONTINUOUS_LIFE_SOCIAL_IDENTITY_INTERFERENCE_GREEN',
        'reference_only':True,'graph_flip':False,'mechanism_change':False,
        'interference_bridges':N_INTERFERENCE_BRIDGES,
        'identity_rows':rows_after_interference,'identity_contexts':contexts_after_interference,
        'a_candidates_after_interference':list(a_candidates_after),
        'resource_pressure':{'loaded':loaded_pressure,'recovered':recovered_pressure},
        'choices':{'a_after_interference':a_after_choice,'checkpoint_reentry':restored_a,
                   'before_revision':before_revision,'after_revision':after_revision,'reacquired':reacquired},
        'long_form_count':len(rows),'checks':checks,'failed':failed,
        'remaining_red':['DIRECT_RAW_SENSORY_SOCIAL_IDENTITY_PARITY','PHYSICAL_MODALITY_FEATURE_EXTRACTION','UNBOUNDED_LIFETIME_IDENTITY_CAPACITY'],
        'elapsed_ms':round((time.perf_counter()-started)*1000,3),
    }
    if failed:
        print('FOUNDRY_CONTINUOUS_LIFE_SOCIAL_IDENTITY_INTERFERENCE_RED '+','.join(failed));print(json.dumps(result,indent=2,sort_keys=True));return 1
    print(result['contract']);print(json.dumps(result,indent=2,sort_keys=True));return 0

if __name__=='__main__':raise SystemExit(main())
