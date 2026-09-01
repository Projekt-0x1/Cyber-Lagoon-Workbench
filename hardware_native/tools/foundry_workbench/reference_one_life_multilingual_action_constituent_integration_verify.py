#!/usr/bin/env python3
"""N+1 receipt: one Life learns action constituents across seven surface ecologies.

The donor reproduces the prior prefix-dominant transfer law. The challenger is the
current OpenLanguageActionAffordanceV1: it may retain bounded competing structural
hypotheses and uses raw constituent/flank evidence without language IDs, translation
or an imperative ontology. The same learner receives every register in one interleaved
chronology; held-out third variants are never demonstrated.
"""
from __future__ import annotations
import copy,json,time

from reference_life_extension_recursive_self_culture_control_v1 import CLAUSES,B1,B2
from reference_open_language_action_affordance_v1 import (
    MAX_FACTOR_MEMBERS,MIN_TRANSFER_BYTES,OpenLanguageActionAffordanceV1,
)
from reference_mathematical_adult_workbench_v1 import MathematicalWorkbenchAdultV1

VISIBLE_GAIN='ONE_CONTINUING_LIFE_MOVES_FROM_PREFIX_ONLY_MULTILINGUAL_ACTION_FAILURE_TO_14_OF_14_HELDOUT_STRUCTURAL_ACTION_REALIZATIONS_WITH_QUOTED_WRAPPERS_STILL_REFUSED'
language_phenotype_improved=True
future_update_authority_preserved=True


class PrefixDonor(OpenLanguageActionAffordanceV1):
    """Exact old architectural limitation: only a sufficiently long common prefix transfers."""
    @staticmethod
    def _common_family(left,right):
        common=set(row for row in left if len(row)==3)&set(row for row in right if len(row)==3)
        prefixes=sorted((row for row in common if row[0]=='P'),key=lambda row:(-row[1],row[2]))
        if not prefixes or int(prefixes[0][1])<MIN_TRANSFER_BYTES:return ()
        suffixes=sorted((row for row in common if row[0]=='S'),key=lambda row:(-row[1],row[2]))
        return tuple((prefixes+suffixes)[:MAX_FACTOR_MEMBERS])


def train_one_life(learner):
    tick=1;timeline=[];registers=tuple(CLAUSES)
    for witness in range(2):
        order=registers if witness==0 else tuple(reversed(registers))
        for name in order:
            for action,forms in ((B1,CLAUSES[name][0]),(B2,CLAUSES[name][1])):
                source=0xD000+tick
                raw=forms[witness].encode('utf-8')
                if not learner.observe_language(raw,source,tick):raise RuntimeError('one-life-action:language')
                timeline.append((tick,name,'language',source,forms[witness]));tick+=1
                if not learner.observe_action(action,source,tick):raise RuntimeError('one-life-action:action')
                timeline.append((tick,name,'action',source,action));tick+=1
    return tick,tuple(timeline)


def heldout(learner):
    rows={};wrapped={}
    for name,pairs in CLAUSES.items():
        for action,forms in ((B1,pairs[0]),(B2,pairs[1])):
            key=f'{name}:{action}'
            raw=forms[2].encode('utf-8')
            candidates=learner.candidates(raw)
            rows[key]=bool(candidates and int(candidates[0][0])==int(action))
            wrapped[key]=not learner.candidates(('someone said: '+forms[2]).encode('utf-8'))
    return rows,wrapped


def extend_unseen_eighth(learner,tick):
    """Sparse no-space ecology acquired after all seven resident surface ecologies."""
    rows=(
      ('⟪ra⟫moruNEXAA',B1),('⟪ra⟫taviNEXAA',B1),
      ('⟪ra⟫moruFIXER',B2),('⟪ra⟫taviFIXER',B2),
      ('zuKEMAQmiretalonaki',0xE812),('niKEMAQmiretalonaki',0xE812),
      ('zuVOKERmiretalonaki',0xE811),('niVOKERmiretalonaki',0xE811),
    )
    timeline=[]
    for index,(text,action) in enumerate(rows):
        source=0xF700+index;raw=text.encode('utf-8')
        if not learner.observe_language(raw,source,tick):raise RuntimeError('one-life-action:eighth-language')
        timeline.append((tick,'eighth','language',source,text));tick+=1
        if not learner.observe_action(action,source,tick):raise RuntimeError('one-life-action:eighth-action')
        timeline.append((tick,'eighth','action',source,action));tick+=1
    return tick,tuple(timeline)


def eighth_probe(learner):
    intended='⟪ra⟫KEMAQmiretalonaki'.encode('utf-8');rival='⟪ra⟫VOKERmiretalonaki'.encode('utf-8')
    unknown='⟪ra⟫ZULIPmiretalonaki'.encode('utf-8');wrapped='report⟪ra⟫KEMAQmiretalonaki'.encode('utf-8')
    return {
      'intended':learner.candidates(intended),'rival':learner.candidates(rival),
      'unknown':learner.candidates(unknown),'wrapped':learner.candidates(wrapped),
      'intended_raw':intended,
    }


def main():
    started=time.perf_counter();donor=PrefixDonor();challenger=OpenLanguageActionAffordanceV1()
    donor_tick,donor_timeline=train_one_life(donor);tick,timeline=train_one_life(challenger);seven_tick=tick
    donor_rows,donor_wrapped=heldout(donor);rows,wrapped=heldout(challenger)
    tick,eighth_timeline=extend_unseen_eighth(challenger,tick);timeline=tuple((*timeline,*eighth_timeline));eighth=eighth_probe(challenger)
    restored=OpenLanguageActionAffordanceV1.restore(copy.deepcopy(challenger.checkpoint()))
    replay_rows,replay_wrapped=heldout(restored);replay_eighth=eighth_probe(restored)
    donor_count=sum(donor_rows.values());challenger_count=sum(rows.values())
    seven_language_sequence=tuple(row[1] for row in timeline[:56] if row[2]=='language')
    language_sequence=tuple(row[1] for row in timeline if row[2]=='language')

    # Recursive hierarchy is a stricter scope owner once it exists. Transplant the exact
    # same learned action ecology into the Mathematical Adult, then teach only a higher
    # one-child wrapper relation. The flat structural candidate exists inside the parent,
    # but hierarchy-aware interpretation must refuse parent force until the parent itself
    # receives an independent action consequence.
    hierarchy=MathematicalWorkbenchAdultV1();hierarchy.language_action_affordances=OpenLanguageActionAffordanceV1.restore(copy.deepcopy(challenger.checkpoint()))
    hctx=0xF811
    child_a=b'zuKEMAQmiretalonaki';child_b=b'niKEMAQmiretalonaki'
    hierarchy.language.observe_span(hctx,(child_a,),b'report'+child_a,0xF812)
    hierarchy.language.observe_span(hctx,(child_b,),b'report'+child_b,0xF813)
    wrapped_eighth=b'report'+eighth['intended_raw'];hierarchy_spans=tuple(hierarchy.language.invert_span(wrapped_eighth))
    flat_inside_parent=hierarchy.language_action_affordances.structural_candidates(wrapped_eighth)
    hierarchy_before=hierarchy.hierarchy_aware_action_candidates(wrapped_eighth)
    hierarchy.language_action_affordances.observe_language(wrapped_eighth,0xF814,tick);hierarchy.language_action_affordances.observe_action(0xE812,0xF814,tick+1)
    hierarchy_after=hierarchy.hierarchy_aware_action_candidates(wrapped_eighth)
    checks={
      'one_continuing_interleaved_life_contains_all_seven_registers':(
          seven_tick==donor_tick and len(timeline)>=56 and set(seven_language_sequence)==set(CLAUSES)
          and len(seven_language_sequence)==28
          and set(seven_language_sequence[:14])==set(CLAUSES)
          and set(seven_language_sequence[14:])==set(CLAUSES)
          and seven_language_sequence[:14]!=seven_language_sequence[14:]),
      'same_life_then_acquires_unseen_eighth_without_language_branch':(
          len(timeline)==72 and language_sequence[-8:]==('eighth',)*8
          and not hasattr(challenger,'language_id')),
      'eighth_no_space_heldout_recombines_construction_and_action_constituent':(
          len(eighth['intended'])==1 and int(eighth['intended'][0][0])==0xE812
          and len(eighth['rival'])==1 and int(eighth['rival'][0][0])==0xE811),
      'eighth_unknown_predicate_and_outer_embedding_fail_closed':(
          not eighth['unknown'] and not eighth['wrapped']),
      'eighth_capability_survives_checkpoint':replay_eighth==eighth,
      'learned_hierarchy_blocks_child_action_force_at_parent_scope':(
          len(hierarchy_spans)==1 and bool(flat_inside_parent) and not hierarchy_before),
      'parent_can_gain_its_own_action_force_only_after_lived_parent_consequence':(
          bool(hierarchy_after) and int(hierarchy_after[0][0])==0xE812
          and hierarchy.language_action_affordances.witnessed_action_surface(wrapped_eighth,0xE812)),
      'prefix_donor_is_predeclared_red_on_typological_displacement':donor_count<14,
      'challenger_is_14_of_14_on_same_heldout_third_variants':challenger_count==14,
      'visible_n_plus_one_over_immediate_prefix_donor':challenger_count>donor_count,
      'unseen_outer_reported_speech_still_refuses_action_force':all(wrapped.values()) and all(donor_wrapped.values()),
      'checkpoint_replays_same_structural_action_capability':replay_rows==rows and replay_wrapped==wrapped,
      'mechanism_has_no_language_translation_or_imperative_classifier':all(not hasattr(challenger,name) for name in ('language_id','translation','imperative')),
      'future_update_authority_is_still_lived_action_chronology':all(len(f.sources)>=2 for f in challenger._factors),
    }
    failed=[name for name,passed in checks.items() if not passed]
    result={
      'schema':'cyber-lagoon.one-life-multilingual-action-constituent-integration.v1',
      'contract':'FOUNDRY_ONE_LIFE_MULTILINGUAL_ACTION_CONSTITUENT_INTEGRATION_'+('GREEN' if not failed else 'RED'),
      'pass':not failed,'reference_only':True,'runtime_llm':False,'novel_synthesis':True,
      'language_phenotype_improved':not failed and language_phenotype_improved,
      'future_update_authority_preserved':not failed and future_update_authority_preserved,
      'visible_language_gain':VISIBLE_GAIN,
      'donor_heldout_green':donor_count,'challenger_heldout_green':challenger_count,
      'heldout':rows,'donor_heldout':donor_rows,'wrapper_refusal':wrapped,
      'factor_rows':challenger.factor_count,'checks':checks,'failed':failed,
      'next_falsifiers':{
        'chomsky':'Acquire unseen recursive/embedded action constructions where the invariant constituent changes hierarchical position, while matched linear-overlap foils refuse.',
        'sapolsky':'Hold the same multilingual action surface fixed while varying source betrayal/recovery, current affordances, prior motor competence and body-resource history; current availability may gate execution but must not erase learned competence.'},
      'elapsed_ms':round((time.perf_counter()-started)*1000,3),
    }
    print(result['contract']);print('visible_language_gain='+VISIBLE_GAIN)
    print(json.dumps(result,ensure_ascii=False,indent=2,sort_keys=True));return 0 if not failed else 1


if __name__=='__main__':raise SystemExit(main())
