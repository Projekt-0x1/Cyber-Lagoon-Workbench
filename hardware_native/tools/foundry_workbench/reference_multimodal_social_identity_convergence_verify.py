#!/usr/bin/env python3
"""Resident recent-source competition for multimodal-like social identity convergence."""
from __future__ import annotations
import copy,inspect,json,time
from reference_partner_specific_pragmatic_language_verify import (
    ACTION,OBJECT,PARTNER_A,PARTNER_B,P_COMPACT,P_EXPLICIT,SOURCE_A,SOURCE_B,
    STATE,establish,prepare,
)
from reference_partner_specific_discourse_selection_verify import (
    establish_partner,prepare as prepare_discourse,
)
from reference_language_mastery_adult_v1 import AdultStateV1
from reference_predictive_credit_profile_v1 import Q

NEW_SOURCE=0xDD01
AMBIG_SOURCE=0xDD02
SPACED_SOURCE=0xDD03


def contact(a,source,claimed=PARTNER_A):
    a.observe_social_contact(claimed,OBJECT,STATE,ACTION,source)
    return int(a._current_partner_context)


def pair(a,anchor_source,new_source,claimed=PARTNER_A,outcome_q16=Q,background=False):
    contact(a,anchor_source,claimed)
    contact(a,new_source,claimed)
    a.experience_recent_social_source_continuity(outcome_q16=outcome_q16)
    a.experience_recent_social_source_background(bool(background))
def main():
    started=time.perf_counter();checks={}
    short,compact,explicit,generic=prepare();establish(short)

    # Unfamiliar current modality/source has no partner-local authority.
    contact(short,NEW_SOURCE)
    unbound_choice=short.choose()
    checks['unearned_new_modality_falls_back_to_generic_language']=(
        short._current_partner_context==0 and unbound_choice==P_EXPLICIT)

    # One cross-source pairing is developmentally insufficient.
    one=type(short).restore(copy.deepcopy(short.checkpoint()))
    pair(one,SOURCE_A,NEW_SOURCE)
    one_choice=one.choose()
    checks['one_crossmodal_pair_is_insufficient']=(
        one._current_partner_context==0 and one_choice==P_EXPLICIT)

    # Repeated matched pairing earns A without any anchor argument at plastic return.
    matched=type(short).restore(copy.deepcopy(short.checkpoint()))
    for _ in range(2):pair(matched,SOURCE_A,NEW_SOURCE)
    matched_choice=matched.choose();matched_ctx=int(matched._current_partner_context)
    checks['matched_recent_source_history_converges_on_a']=(
        matched_ctx!=0 and matched_choice==P_COMPACT
        and matched.public_surface(matched_choice)==compact.surface)

    # Same events, but temporal separation beyond resident coincidence window, refuse.
    spaced=type(short).restore(copy.deepcopy(short.checkpoint()))
    refused=0
    for _ in range(2):
        contact(spaced,SOURCE_A)
        for _ in range(9):spaced.internal_tick()
        contact(spaced,SPACED_SOURCE)
        try:spaced.experience_recent_social_source_continuity()
        except RuntimeError:refused+=1
    checks['spaced_yoked_source_events_do_not_converge']=(
        refused==2 and spaced._current_partner_context==0 and spaced.choose()==P_EXPLICIT)

    yoked=type(short).restore(copy.deepcopy(short.checkpoint()))
    for _ in range(2):pair(yoked,SOURCE_A,NEW_SOURCE,background=True)
    checks['yoked_crossmodal_outcome_cannot_converge']=(
        yoked._current_partner_context==0 and yoked.choose()==P_EXPLICIT)

    # Identity is learned from contingency, not reward sign. Aversive but
    # action-contingent cross-source returns still identify the same partner.
    aversive=type(short).restore(copy.deepcopy(short.checkpoint()))
    for _ in range(2):pair(aversive,SOURCE_A,NEW_SOURCE,outcome_q16=-Q)
    checks['consequence_valence_does_not_author_social_identity']=(
        aversive._current_partner_context!=0 and aversive.choose()==P_COMPACT)

    # Matched support for two lived anchors is unresolved rather than arbitrary identity.
    ambiguous=type(short).restore(copy.deepcopy(short.checkpoint()))
    for anchor in (SOURCE_A,SOURCE_B):
        for _ in range(2):pair(ambiguous,anchor,AMBIG_SOURCE)
    checks['equal_a_b_crossmodal_support_refuses_identity_specific_bias']=(
        ambiguous._current_partner_context==0 and ambiguous.choose()==P_EXPLICIT)

    # Checkpoint keeps learned bridge, not active/current/recent source occurrence.
    checkpoint=copy.deepcopy(matched.checkpoint());restored=type(matched).restore(checkpoint)
    checks['checkpoint_drops_active_multimodal_occurrences']=(
        restored._current_social_source==0 and restored._recent_social_source==0
        and restored._current_partner_context==0 and restored.choose()==0)
    contact(restored,NEW_SOURCE)
    checks['fresh_new_modality_retrieves_checkpointed_convergence']=(
        restored.choose()==P_COMPACT and restored._current_partner_context==matched_ctx)

    # Focal bridge lesion preserves source-A competence and generic language.
    lesioned=type(matched).restore(checkpoint);row=lesioned.social_identity_credit.row(SOURCE_A)
    row.contexts.pop(NEW_SOURCE,None);lesioned.social_identity_credit.context_members.get(NEW_SOURCE,set()).discard(SOURCE_A)
    contact(lesioned,NEW_SOURCE)
    checks['focal_crossmodal_bridge_lesion_restores_generic_not_silence']=(
        lesioned._current_partner_context==0 and lesioned.choose()==P_EXPLICIT)
    contact(lesioned,SOURCE_A)
    checks['crossmodal_bridge_lesion_preserves_anchor_modality_identity']=(
        lesioned._current_partner_context!=0 and lesioned.choose()==P_COMPACT)

    # Recent mismatches can suspend and matched history reacquire the bridge.
    revised=type(matched).restore(checkpoint);contact(revised,NEW_SOURCE)
    before=revised.choose()
    for _ in range(5):
        contact(revised,SOURCE_A);contact(revised,NEW_SOURCE)
        revised.experience_recent_social_source_background(True)
    after=revised.choose()
    for _ in range(5):pair(revised,SOURCE_A,NEW_SOURCE)
    reacquired=revised.choose()
    checks['recent_mismatch_extinguishes_then_matched_history_reacquires']=(
        before==P_COMPACT and after==P_EXPLICIT and reacquired==P_COMPACT)

    # The same learned convergence must retrieve already-earned long-form partner matter.
    discourse,frontier,dgeneric=prepare_discourse()
    establish_partner(discourse,frontier,PARTNER_A,0,0xDD20)
    establish_partner(discourse,frontier,PARTNER_B,1,0xDD30)
    contact(discourse,NEW_SOURCE);generic_root=discourse.organize_relevant_frontier(frontier)
    generic_rows=tuple(discourse.last_discourse_selected)
    for _ in range(2):pair(discourse,SOURCE_A,NEW_SOURCE)
    converged_root=discourse.organize_relevant_frontier(frontier)
    converged_rows=tuple(discourse.last_discourse_selected)
    even=tuple(leaf.identity for i,leaf in enumerate(frontier) if i%2==0)
    checks['converged_modality_retrieves_a_long_form_matter']=(
        generic_root is not None and converged_root is not None
        and len(generic_rows)==16 and converged_rows==even
        and tuple(generic_root.surface)!=tuple(converged_root.surface))

    # Present allostatic state changes whether partner-specific history wins,
    # without rewriting the learned identity; neutral recovery retrieves it.
    stateful=type(matched).restore(checkpoint);contact(stateful,NEW_SOURCE)
    stateful_context=int(stateful._current_partner_context)
    neutral_choice=stateful.choose()
    overloaded_choice=stateful.choose(AdultStateV1(pressure_q16=Q))
    recovered_choice=stateful.choose()
    checks['allostatic_state_suppresses_weak_partner_use_then_recovers']=(
        stateful_context!=0 and stateful._current_partner_context==stateful_context
        and neutral_choice==P_COMPACT and overloaded_choice==P_EXPLICIT
        and recovered_choice==P_COMPACT)

    # Winning plastic API has no anchor/person argument.
    sig=list(inspect.signature(type(short).experience_recent_social_source_continuity).parameters)
    checks['winning_plastic_boundary_has_no_anchor_nomination']=(
        'anchor_source' not in sig and 'independent_return' not in sig
        and sig==['self','outcome_q16','effort_q16','duration'])
    checks['recent_source_occurrence_not_checkpointed']=(
        'recent_social_source' not in json.dumps(checkpoint,sort_keys=True))
    checks['bounded_fast_path']=time.perf_counter()-started<1.0

    failed=[k for k,v in checks.items() if not v]
    result={
        'contract':'FOUNDRY_MULTIMODAL_SOCIAL_IDENTITY_CONVERGENCE_GREEN',
        'reference_only':True,'graph_flip':False,
        'language_phenotype_improved':True,
        'visible_language_gain':'ALLOSTATIC_STATE_AND_CONTINGENCY_CHANGE_SOURCE_QUALIFIED_LANGUAGE_WITH_RECOVERY',
        'choices':{'unbound':unbound_choice,'one':one_choice,'matched':matched_choice,
                   'before_mismatch':before,'after_mismatch':after,'reacquired':reacquired},
        'discourse_counts':{'generic':len(generic_rows),'converged':len(converged_rows)},
        'checks':checks,'failed':failed,
        'remaining_red':['DIRECT_RAW_SENSORY_SOCIAL_IDENTITY_PARITY','PHYSICAL_FACE_VOICE_FEATURE_EXTRACTION','CONTINUOUS_LIFE_IDENTITY_INTERFERENCE'],
        'elapsed_ms':round((time.perf_counter()-started)*1000,3),
    }
    if failed:
        print('FOUNDRY_MULTIMODAL_SOCIAL_IDENTITY_CONVERGENCE_RED '+','.join(failed));print(json.dumps(result,indent=2,sort_keys=True));return 1
    print(result['contract']);print(json.dumps(result,indent=2,sort_keys=True));return 0

if __name__=='__main__':raise SystemExit(main())
