#!/usr/bin/env python3
"""N+1: completed self-output licenses bare ellipsis across checkpoint without transcript state."""
from __future__ import annotations

import copy
import inspect
import json
import time

from reference_language_mastery_contact_adapter_v1 import CONTACT_UTTERANCE,LanguageMasteryContactAdapterV1
from reference_lived_world_conversation_terminal_v1 import quiet
from reference_lived_world_followup_bridge_v1 import LivedWorldFollowupBridgeV1
from reference_lived_world_followup_terminal_v1 import respond_followup
from reference_lived_world_ellipsis_terminal_v1 import respond_ellipsis
from reference_organism_v2 import ReferenceOrganismV2
from reference_self_initiated_world_followup_verify import prepared
from reference_world_derived_proposition_frontier_verify import WORLD_B,SOURCE_B,world

visible_language_gain='COMPLETED_SELF_OUTPUT_LICENSES_BARE_ELLIPSIS_ACROSS_CHECKPOINT_WITHOUT_TRANSCRIPT'
language_phenotype_improved=True
future_update_authority_preserved=True


def partial_focus(adult,organism,last,raw=b'about sensor?',nbytes=8,wrong=False):
    contact=LanguageMasteryContactAdapterV1(adult)
    identity=contact.contact(CONTACT_UTTERANCE,tuple(raw),adult._advance(),0)
    scene=contact.scenes.get(int(identity)) if int(identity)>0 else None
    if scene is None:return b'',False
    _context,frontier=LivedWorldFollowupBridgeV1.activate_frontier(
        adult,organism,scene.context,scene.atoms,last)
    root=adult.organize_relevant_frontier(frontier) if frontier else None
    if root is None:return b'',False
    expression=adult.expression(root);out=bytearray();bad_refused=False
    for index in range(int(nbytes)):
        step=expression.emit()
        if step is None:break
        observed=(step.value^1) if wrong and index==0 else step.value
        ok=expression.reafference(step,observed)
        if not ok:
            bad_refused=True;break
        out.append(step.value)
    return bytes(out),bad_refused


def main():
    started=time.perf_counter();checks={};adult,organism=prepared();last=0

    # Complete world speech then a complete focused answer. Only final motor
    # reafference may settle the opaque future-relevant context.
    spontaneous,last=quiet(adult,organism,last)
    world_context=adult.last_completed_public_context()
    focused=respond_followup(
        adult,organism,LanguageMasteryContactAdapterV1(adult),b'about sensor?',last)
    focused_context=adult.last_completed_public_context()
    checks['complete_public_actions_settle_distinct_opaque_contexts']=(
        len(spontaneous)==377 and len(focused)==186 and world_context>0
        and focused_context>0 and focused_context!=world_context)

    cp=copy.deepcopy(adult.checkpoint());org_cp=copy.deepcopy(organism.checkpoint())
    restored=type(adult).restore(copy.deepcopy(cp));restored_org=ReferenceOrganismV2.restore(copy.deepcopy(org_cp))
    checks['checkpoint_preserves_completed_context_but_not_active_selection']=(
        restored.last_completed_public_context()==focused_context
        and int(restored._current_selection_context)==0)
    ellipsis=respond_ellipsis(
        restored,restored_org,LanguageMasteryContactAdapterV1(restored),b'valve',last)
    valve_context=restored.last_completed_public_context()
    checks['checkpoint_surviving_bare_ellipsis_is_visible']=(
        len(ellipsis)==185 and b'valve' in ellipsis and b'sensor' not in ellipsis
        and valve_context>0 and valve_context!=focused_context)

    # Incomplete public expression carries a pending prefix but cannot become the
    # completed self-output relation. Restart must therefore refuse bare ellipsis.
    partial_adult,partial_org=prepared();partial_last=0
    _speech,partial_last=quiet(partial_adult,partial_org,partial_last)
    prior_completed=partial_adult.last_completed_public_context()
    prefix,_=partial_focus(partial_adult,partial_org,partial_last,nbytes=8)
    partial_cp=copy.deepcopy(partial_adult.checkpoint())
    partial_restored=type(partial_adult).restore(copy.deepcopy(partial_cp))
    partial_org_restored=ReferenceOrganismV2.restore(copy.deepcopy(partial_org.checkpoint()))
    partial_ellipsis=respond_ellipsis(
        partial_restored,partial_org_restored,LanguageMasteryContactAdapterV1(partial_restored),b'valve',partial_last)
    checks['incomplete_public_expression_does_not_license_checkpoint_ellipsis']=(
        len(prefix)==8 and partial_adult.last_completed_public_context()==prior_completed
        and bool(partial_cp.get('pending_public_expression')) and partial_ellipsis==b'')

    # Wrong reafference cannot settle the relation either.
    bad_adult,bad_org=prepared();bad_last=0
    _bad_speech,bad_last=quiet(bad_adult,bad_org,bad_last)
    bad_prior=bad_adult.last_completed_public_context()
    bad_prefix,bad_refused=partial_focus(bad_adult,bad_org,bad_last,nbytes=1,wrong=True)
    checks['wrong_motor_reafference_cannot_settle_completed_context']=(
        bad_prefix==b'' and bad_refused and bad_adult.last_completed_public_context()==bad_prior)

    # Later completed public action supersedes rather than accumulating dialogue state.
    world(restored_org,WORLD_B,SOURCE_B);world_b,last_b=quiet(restored,restored_org,last)
    b_context=restored.last_completed_public_context()
    restarted_b=type(restored).restore(copy.deepcopy(restored.checkpoint()))
    restarted_b_org=ReferenceOrganismV2.restore(copy.deepcopy(restored_org.checkpoint()))
    stale=respond_ellipsis(
        restarted_b,restarted_b_org,LanguageMasteryContactAdapterV1(restarted_b),b'sensor',last_b)
    checks['later_completed_world_turn_overwrites_prior_ellipsis_relation']=(
        len(world_b)==362 and b_context>0 and b_context!=valve_context and stale==b'')

    # Persistent matter is exactly one scalar context plus the pre-existing pending
    # prefix owner. There is no public setter or persisted surface/constituent list.
    checkpoint=json.loads(json.dumps(restored.checkpoint()))
    source=inspect.getsource(type(restored))
    checks['completed_relation_is_one_scalar_not_transcript_or_plan']=(
        isinstance(checkpoint.get('last_completed_public_context'),int)
        and checkpoint['last_completed_public_context']==b_context
        and all(key not in checkpoint for key in (
            'last_completed_public_surface','last_completed_public_plan',
            'last_completed_public_constituents','conversation_buffer','transcript','context_window')))
    checks['completed_context_has_read_only_api_no_host_setter']=(
        'def last_completed_public_context(self):' in source
        and 'def set_last_completed_public_context' not in source
        and 'def settle_last_completed_public_context' not in source)
    checks['bounded_fast_path']=time.perf_counter()-started<1.0

    failed=[name for name,passed in checks.items() if not passed]
    result={
        'schema':'cyber-lagoon.checkpoint-surviving-world-ellipsis.v1',
        'pass':not failed,'reference_only':True,
        'language_phenotype_improved':language_phenotype_improved,
        'future_update_authority_preserved':future_update_authority_preserved,
        'visible_language_gain':visible_language_gain,
        'bytes':{'spontaneous':len(spontaneous),'focused':len(focused),'ellipsis':len(ellipsis),'world_b':len(world_b)},
        'contexts':{'world_a':world_context,'focused':focused_context,'ellipsis':valve_context,'world_b':b_context},
        'checks':checks,'failed':failed,
        'remaining_red':[
            'ELLIPSIS_OVER_RELATIONS_NOT_SINGLE_CONCEPT',
            'MULTI_PARTNER_SELF_OUTPUT_RELATION_FACTORIZATION',
            'OPEN_ENDED_CONVERSATIONAL_GENERATION',
            'NOVEL_TOPIC_LEARNING_WITHOUT_PRETRAINED_RELEVANCE',
        ],
        'elapsed_ms':round((time.perf_counter()-started)*1000,3),
    }
    print('FOUNDRY_CHECKPOINT_SURVIVING_WORLD_ELLIPSIS_'+('GREEN' if not failed else 'RED'))
    print('visible_language_gain='+visible_language_gain)
    print(json.dumps(result,indent=2,sort_keys=True))
    return 0 if not failed else 1


if __name__=='__main__':raise SystemExit(main())
