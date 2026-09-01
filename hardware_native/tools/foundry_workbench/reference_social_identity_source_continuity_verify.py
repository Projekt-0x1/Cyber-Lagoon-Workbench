#!/usr/bin/env python3
"""Source-earned social identity blocks copied partner-history authority."""
from __future__ import annotations

import copy
import json
import time

from reference_partner_specific_pragmatic_language_verify import (
    ACTION, OBJECT, PARTNER_A, PARTNER_B, P_COMPACT, P_EXPLICIT, SOURCE_A, SOURCE_B,
    STATE, establish, prepare,
)
from reference_partner_specific_discourse_selection_verify import (
    establish_partner, prepare as prepare_discourse,
)

UNBOUND_COPY_SOURCE=0xDCBAD
EARNED_NEW_SOURCE=0xDCA2
AMBIGUOUS_SOURCE=0xDCA3


def main():
    started=time.perf_counter();checks={}

    short,compact,explicit,generic=prepare();establish(short)
    # This source never participated in A's development. The caller merely asserts
    # A's integer; a source-earned identity path must not grant A-local credit.
    short.observe_social_contact(PARTNER_A,OBJECT,STATE,ACTION,UNBOUND_COPY_SOURCE)
    copied_generic=int(short._current_selection_context);copied_local=int(short._current_partner_context)
    copied_choice=short.choose()
    checks['unbound_source_claiming_a_falls_back_to_generic_short_language']=(
        copied_generic==generic and copied_choice==P_EXPLICIT
        and short.public_surface(copied_choice)==explicit.surface)

    discourse,frontier,dgeneric=prepare_discourse()
    establish_partner(discourse,frontier,PARTNER_A,0,0xDC10)
    establish_partner(discourse,frontier,PARTNER_B,1,0xDC20)
    discourse.observe_social_contact(PARTNER_A,OBJECT,STATE,ACTION,UNBOUND_COPY_SOURCE)
    copied_dgeneric=int(discourse._current_selection_context)
    copied_dlocal=int(discourse._current_partner_context)
    copied_root=discourse.organize_relevant_frontier(frontier)
    copied_rows=tuple(discourse.last_discourse_selected)
    copied_surface=() if copied_root is None else tuple(copied_root.surface)
    checks['unbound_source_claiming_a_falls_back_to_generic_discourse']=(
        copied_dgeneric==dgeneric and len(copied_rows)==len(frontier)==16
        and len(copied_surface)>700)

    # Two independently returned continuity events let a new opaque source
    # reactivate A without changing the asserted agent or generic relation.
    short.observe_social_contact(PARTNER_A,OBJECT,STATE,ACTION,EARNED_NEW_SOURCE)
    for _ in range(2):
        short.experience_social_source_continuity(SOURCE_A)
        short.experience_social_source_background(SOURCE_A,False)
    earned_choice=short.choose();earned_context=int(short._current_partner_context)
    checks['independently_earned_cross_source_continuity_retrieves_a']=(
        earned_context!=0 and earned_choice==P_COMPACT)
    checkpoint=copy.deepcopy(short.checkpoint())
    restored=type(short).restore(checkpoint)
    checks['checkpoint_does_not_restore_active_social_identity']=(
        restored._current_partner_context==0 and restored.choose()==0)
    restored.observe_social_contact(PARTNER_A,OBJECT,STATE,ACTION,EARNED_NEW_SOURCE)
    checks['checkpoint_fresh_source_reactivates_earned_identity']=(
        restored.choose()==P_COMPACT and restored._current_partner_context==earned_context)

    # Equal source evidence for A and B has no unique identity winner.
    ambiguous,_,amb_explicit,_=prepare();establish(ambiguous)
    ambiguous.observe_social_contact(PARTNER_A,OBJECT,STATE,ACTION,AMBIGUOUS_SOURCE)
    for anchor in (SOURCE_A,SOURCE_B):
        for _ in range(2):
            ambiguous.experience_social_source_continuity(anchor)
            ambiguous.experience_social_source_background(anchor,False)
    checks['equal_a_b_source_evidence_refuses_identity_specific_bias']=(
        ambiguous._current_partner_context==0 and ambiguous.choose()==P_EXPLICIT
        and ambiguous.public_surface(P_EXPLICIT)==amb_explicit.surface)

    forged,_,_,_=prepare();establish(forged)
    forged.observe_social_contact(PARTNER_A,OBJECT,STATE,ACTION,UNBOUND_COPY_SOURCE)
    for _ in range(2):
        forged.experience_social_source_continuity(SOURCE_A,independent_return=False)
    checks['non_independent_source_continuity_cannot_retrieve_a']=(
        forged._current_partner_context==0 and forged.choose()==P_EXPLICIT)

    yoked,_,_,_=prepare();establish(yoked)
    yoked.observe_social_contact(PARTNER_A,OBJECT,STATE,ACTION,UNBOUND_COPY_SOURCE)
    for _ in range(2):
        yoked.experience_social_source_continuity(SOURCE_A)
        yoked.experience_social_source_background(SOURCE_A,True)
    checks['yoked_source_continuity_cannot_retrieve_a']=(
        yoked._current_partner_context==0 and yoked.choose()==P_EXPLICIT)

    # Recent mismatch suspends the source anchor; repeated ordinary contingent
    # continuity reacquires it without changing language or partner-credit state.
    revised=type(short).restore(checkpoint)
    revised.observe_social_contact(PARTNER_A,OBJECT,STATE,ACTION,EARNED_NEW_SOURCE)
    before_revision=revised.choose()
    for _ in range(5):revised.experience_social_source_background(SOURCE_A,True)
    after_revision=revised.choose()
    for _ in range(5):
        revised.experience_social_source_continuity(SOURCE_A)
        revised.experience_social_source_background(SOURCE_A,False)
    after_reacquisition=revised.choose()
    checks['recent_source_mismatch_extinguishes_then_reacquires_identity']=(
        before_revision==P_COMPACT and after_revision==P_EXPLICIT
        and after_reacquisition==P_COMPACT)
    checks['claimed_agent_and_unbound_source_are_not_same_causal_coordinate']=(
        copied_local==0 and copied_dlocal==0)
    checks['no_host_agent_identity_authority']=not getattr(short,'trust_agent_identity',False)
    checks['bounded_fast_path']=time.perf_counter()-started<1.0

    failed=[name for name,value in checks.items() if not value]
    result={
        'contract':'FOUNDRY_SOCIAL_IDENTITY_SOURCE_CONTINUITY_GREEN',
        'reference_only':True,'graph_flip':False,
        'language_phenotype_improved':True,
        'visible_language_gain':'SOURCE_EARNED_SOCIAL_IDENTITY_BLOCKS_COPIED_PARTNER_HISTORY',
        'generic_context':generic,
        'claimed_a_contexts':{'short':copied_local,'discourse':copied_dlocal},
        'copied_choice':copied_choice,'earned_choice':earned_choice,
        'copied_discourse_propositions':len(copied_rows),
        'checks':checks,
        'remaining_red':['DIRECT_RAW_SENSORY_SOCIAL_IDENTITY_PARITY',
                         'MULTIMODAL_IDENTITY_CONVERGENCE',
                         'CONTINUOUS_LIFE_IDENTITY_INTERFERENCE'],
        'elapsed_ms':round((time.perf_counter()-started)*1000,3),
    }
    if failed:
        print('FOUNDRY_SOCIAL_IDENTITY_SOURCE_CONTINUITY_RED '+','.join(failed))
        print(json.dumps(result,sort_keys=True,indent=2));return 1
    print('FOUNDRY_SOCIAL_IDENTITY_SOURCE_CONTINUITY_GREEN')
    print(json.dumps(result,sort_keys=True,indent=2));return 0


if __name__=='__main__':raise SystemExit(main())
