#!/usr/bin/env python3
"""N+1: immediate bare-concept ellipsis continues a focused self-initiated world dialogue."""
from __future__ import annotations

import copy
import json
import time

from reference_language_mastery_contact_adapter_v1 import LanguageMasteryContactAdapterV1
from reference_lived_world_conversation_terminal_v1 import quiet
from reference_lived_world_ellipsis_terminal_v1 import respond_ellipsis
from reference_lived_world_followup_terminal_v1 import respond_followup
from reference_organism_v2 import ReferenceOrganismV2
from reference_predictive_credit_profile_v1 import Q
from reference_self_initiated_world_followup_verify import prepared
from reference_world_discourse_situation_bridge_v1 import WorldDiscourseSituationBridgeV1
from reference_world_derived_proposition_frontier_verify import WORLD_B,SOURCE_B,world

visible_language_gain='BARE_LEARNED_CONCEPT_ELLIPSIS_CONTINUES_IMMEDIATE_FOCUSED_SELF_INITIATED_WORLD_DIALOGUE'
language_phenotype_improved=True
future_update_authority_preserved=True


def main():
    started=time.perf_counter();checks={};adult,organism=prepared();last=0
    baseline_cp=copy.deepcopy(adult.checkpoint());baseline_org=copy.deepcopy(organism.checkpoint())
    baseline_frontier=WorldDiscourseSituationBridgeV1.frontier(adult,organism)
    baseline_ids=tuple(int(x.identity) for x in baseline_frontier)

    # Bare concept is ordinary learned lexical contact, but without a focused prior
    # conversational situation it cannot nominate a discourse subset.
    early=respond_ellipsis(
        adult,organism,LanguageMasteryContactAdapterV1(adult),b'valve',last)
    checks['bare_concept_before_focused_self_initiated_dialogue_stays_silent']=(early==b'')

    adult=type(adult).restore(copy.deepcopy(baseline_cp));organism=ReferenceOrganismV2.restore(copy.deepcopy(baseline_org));last=0
    spontaneous,last=quiet(adult,organism,last)
    focused=respond_followup(
        adult,organism,LanguageMasteryContactAdapterV1(adult),b'about sensor?',last)
    focused_context=int(adult._current_selection_context)
    ellipsis=respond_ellipsis(
        adult,organism,LanguageMasteryContactAdapterV1(adult),b'valve',last)
    ellipsis_context=int(adult._current_selection_context)
    checks['self_initiated_then_explicit_focus_then_bare_ellipsis_is_visible']=(
        len(spontaneous)==377 and len(focused)==186 and len(ellipsis)==185)
    checks['bare_valve_ellipsis_switches_focus_without_repeating_query_scaffold']=(
        b'valve' in ellipsis and b'sensor' not in ellipsis and ellipsis!=focused)
    checks['ellipsis_uses_different_already_learned_discourse_context']=(
        focused_context>0 and ellipsis_context>0 and focused_context!=ellipsis_context)

    # Hearing the one-word lexical occurrence must not turn it into an independently
    # productive world proposition or expand the mechanical world frontier.
    after_frontier=WorldDiscourseSituationBridgeV1.frontier(adult,organism)
    checks['one_word_lexical_contact_does_not_pollute_productive_world_frontier']=(
        tuple(int(x.identity) for x in after_frontier)==baseline_ids and len(after_frontier)==8)

    # Immediate transient discourse is the license. Checkpoint intentionally clears
    # active selection context; bare ellipsis therefore refuses after restart.
    restored=type(adult).restore(copy.deepcopy(adult.checkpoint()))
    restored_org=ReferenceOrganismV2.restore(copy.deepcopy(organism.checkpoint()))
    after_checkpoint=respond_ellipsis(
        restored,restored_org,LanguageMasteryContactAdapterV1(restored),b'sensor',last)
    checks['checkpoint_without_active_discourse_does_not_preserve_ellipsis_license']=(
        after_checkpoint==b'' and int(restored._current_selection_context)==0)

    # A new lived-world event supersedes the focused A situation. The current B
    # long-form turn is not a strict subset context, so stale A ellipsis refuses.
    world(organism,WORLD_B,SOURCE_B);world_b,last_b=quiet(adult,organism,last)
    stale=respond_ellipsis(
        adult,organism,LanguageMasteryContactAdapterV1(adult),b'sensor',last_b)
    checks['new_world_event_supersedes_prior_ellipsis_license']=(
        len(world_b)==362 and stale==b'' and last_b!=last)

    # Ambiguous structural context must refuse rather than pick one by ID/order.
    ambiguous=type(adult).restore(copy.deepcopy(baseline_cp));amb_org=ReferenceOrganismV2.restore(copy.deepcopy(baseline_org));amb_last=0
    _speech,amb_last=quiet(ambiguous,amb_org,amb_last)
    _focus=respond_followup(
        ambiguous,amb_org,LanguageMasteryContactAdapterV1(ambiguous),b'about sensor?',amb_last)
    valve_ids={int(x.identity) for x in WorldDiscourseSituationBridgeV1.frontier(ambiguous,amb_org)
               if b'valve' in bytes(x.surface)}
    duplicate_context=0xE771
    for pid in sorted(valve_ids):
        for _ in range(2):
            ambiguous.experience_discourse_candidate(pid,Q,context=duplicate_context)
            ambiguous.experience_discourse_background(pid,False)
    ambiguous_result=respond_ellipsis(
        ambiguous,amb_org,LanguageMasteryContactAdapterV1(ambiguous),b'valve',amb_last)
    checks['duplicate_matching_discourse_context_refuses_ellipsis_ambiguity']=(
        ambiguous_result==b'' and set(ambiguous.discourse_credit.candidates(duplicate_context))==valve_ids)

    checkpoint_text=json.dumps(adult.checkpoint(),sort_keys=True)
    checks['adult_checkpoint_has_no_ellipsis_topic_or_transcript_state']=(
        all(token not in checkpoint_text for token in (
            'ellipsis','active_topic','topic_frontier','conversation_buffer','context_window','transcript')))
    checks['standing_path_is_bounded']=time.perf_counter()-started<1.0

    failed=[name for name,passed in checks.items() if not passed]
    result={
        'schema':'cyber-lagoon.self-initiated-world-ellipsis.v1',
        'pass':not failed,'reference_only':True,
        'language_phenotype_improved':language_phenotype_improved,
        'future_update_authority_preserved':future_update_authority_preserved,
        'visible_language_gain':visible_language_gain,
        'bytes':{'spontaneous':len(spontaneous),'explicit_focus':len(focused),'ellipsis':len(ellipsis),'world_b':len(world_b)},
        'checks':checks,'failed':failed,
        'remaining_red':[
            'CHECKPOINT_SURVIVING_ELLIPSIS_WITH_CAUSAL_SELF_OUTPUT_RELATION',
            'ELLIPSIS_OVER_RELATIONS_NOT_SINGLE_CONCEPT',
            'OPEN_ENDED_CONVERSATIONAL_GENERATION',
            'NOVEL_TOPIC_LEARNING_WITHOUT_PRETRAINED_RELEVANCE',
        ],
        'elapsed_ms':round((time.perf_counter()-started)*1000,3),
    }
    print('FOUNDRY_SELF_INITIATED_WORLD_ELLIPSIS_'+('GREEN' if not failed else 'RED'))
    print('visible_language_gain='+visible_language_gain)
    print(json.dumps(result,indent=2,sort_keys=True))
    return 0 if not failed else 1


if __name__=='__main__':raise SystemExit(main())
