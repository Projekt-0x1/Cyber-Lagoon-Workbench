#!/usr/bin/env python3
"""Whole-subject checks for the strict post-legacy population organism."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))

from reference_organism_v2 import *
from reference_population_v1 import PopulationSpecV1


def u(text):return tuple(text.encode('utf-8'))

def scene(o,context,atoms,source,channel=7):return o.contact(CONTACT_SCENE,(channel,context,len(atoms),*atoms),source,True,True)
def surface(o,text,source):return o.contact(CONTACT_SURFACE,u(text),source,True,True)
def partner(o,source,channel=7):return o.contact(CONTACT_PARTNER_CONTEXT,(1,channel,source),77000+int(source),True,True)

def train(o):
    NAME=100;CTX=9001
    mapping={101:'careful',102:'quiet',201:'engineer',202:'technician',301:'tests',302:'inspects',401:'sensor',402:'valve'}
    for feature,text in mapping.items():
        scene(o,NAME,(feature,),50000+feature);surface(o,text,1000+feature)
        scene(o,NAME,(feature,),60000+feature);surface(o,text,2000+feature)
    # Repeated raw sentence contacts; no template/order enters through the API.
    scene(o,CTX,(101,201,301,401),8101);surface(o,'the careful engineer tests the sensor.',3001)
    scene(o,CTX,(102,202,302,402),8102);surface(o,'the quiet technician inspects the valve.',3002)
    return NAME,CTX,mapping


def main():
    started=time.perf_counter();checks={};spec=PopulationSpecV1(131072,fanout=2,sites_per_feature=4,eligibility_horizon=8)
    blank=ReferenceOrganismV2(spec);checks['blank_birth_silent']=blank.tick() is None and not blank.actions
    checks['no_host_goal_api']=not hasattr(blank,'enqueue_goal') and not hasattr(blank,'prompt') and not hasattr(blank,'complete')

    o=ReferenceOrganismV2(spec);NAME,CTX,mapping=train(o);checks['learned_not_born']=o.language.template(CTX,4) is not None and blank.language.template(CTX,4) is None
    held=(102,201,301,402);scene(o,CTX,held,9001);checks['no_partner_scene_silence']=o.tick() is None;partner(o,9001);action=o.tick()
    checks['endogenous_heldout_expression']=action is not None and action.payload==u('the quiet engineer tests the valve.')
    checks['action_causal_trace']=action is not None and action.scene_identity>0 and action.population_occurrence>0 and action.template_identity>0 and all(action.contributors)
    held_training={u('the careful engineer tests the sensor.'),u('the quiet technician inspects the valve.')};checks['not_transcript_replay']=action.payload not in held_training

    # A later lived episode may produce its own exact public commitment while
    # the first awaits independently returned consequence.
    partner(o,9002);scene(o,CTX,(101,202,302,401),9002);next_action=o.tick()
    checks['independent_later_scene_does_not_wait_for_prior_public_consequence']=(
        next_action is not None and next_action.ticket!=action.ticket
        and next_action.payload==u('the careful technician inspects the sensor.'))
    before_credit=o.population.credit_events;before_revision=o.population.revision_events
    learned=o.contact(CONTACT_CONSEQUENCE,(action.ticket,1),9001,True,True)
    checks['causal_consequence_revises_population']=learned['credit']>0 and learned['revisions']>0 and o.population.credit_events>before_credit and o.population.revision_events>before_revision

    checks['settling_first_preserves_second_exact_commitment']=(
        next_action is not None and not next_action.settled
        and {a.ticket for a in o.actions if not a.settled}=={next_action.ticket})
    checks['continuing_subject_resumes']=next_action is not None
    o.contact(CONTACT_CONSEQUENCE,(next_action.ticket,0),9002,True,True)

    # Episodic reinstatement: cue with one unknown slot retrieves a unique lived episode.
    partner(o,9100);scene(o,CTX,(101,201,301,0),9100);memory_action=o.tick();checks['episodic_completion']=memory_action is not None and memory_action.payload==u('the careful engineer tests the sensor.') and o.current_scene.completed_from_episode>0
    checks['retrieval_was_unique']=o.last_retrieval['status']==1 and o.last_retrieval['winner']==o.current_scene.completed_from_episode
    o.contact(CONTACT_CONSEQUENCE,(memory_action.ticket,1),9100,True,True)

    # Create a second episode with same known cue and another object, then partial cue must remain unresolved/silent.
    scene(o,CTX,(101,201,301,402),9200);surface(o,'the careful engineer tests the valve.',3003)
    scene(o,CTX,(101,201,301,0),9300);ambiguous=o.tick();checks['episodic_ambiguity_silent']=ambiguous is None and o.last_retrieval['status']==2 and o.last_retrieval['alternatives']>=2

    # Source-qualified construction support: removing one of the two original template witnesses disables that exact learned construction family.
    w=ReferenceOrganismV2(spec);train(w);w.contact(CONTACT_WITHDRAW_SOURCE,(3002,),9999,True,True);partner(w,9400);scene(w,CTX,held,9400);checks['construction_source_withdrawal_silences']=w.tick() is None

    # Checkpoint/replay before a novel scene gives identical action/state.
    base=ReferenceOrganismV2(spec);train(base);cp=base.checkpoint();left=ReferenceOrganismV2.restore(cp);right=ReferenceOrganismV2.restore(cp);partner(left,9500);partner(right,9500);scene(left,CTX,held,9500);scene(right,CTX,held,9500);la=left.tick();ra=right.tick();checks['checkpoint_replay']=la==ra and left.digest()==right.digest()

    # Opaque ID permutation cannot alter the learned surface phenotype.
    perm=ReferenceOrganismV2(spec);rename={k:k+70000 for k in mapping}
    for feature,text in mapping.items():
        scene(perm,NAME,(rename[feature],),50000+feature);surface(perm,text,1000+feature);scene(perm,NAME,(rename[feature],),60000+feature);surface(perm,text,2000+feature)
    scene(perm,CTX,tuple(rename[x] for x in (101,201,301,401)),8101);surface(perm,'the careful engineer tests the sensor.',3001)
    scene(perm,CTX,tuple(rename[x] for x in (102,202,302,402)),8102);surface(perm,'the quiet technician inspects the valve.',3002)
    partner(perm,9600);scene(perm,CTX,tuple(rename[x] for x in held),9600);pa=perm.tick();checks['opaque_id_permutation']=pa is not None and pa.payload==action.payload

    q=o.population.quantity_vector(None,alternatives=o.last_retrieval.get('alternatives',0),horizon=1,trajectory=max((len(a.payload) for a in o.actions),default=0))
    checks['quantity_is_real_state']=q['R']==spec.site_count and q['I']==spec.site_count*spec.fanout and q['F']>0 and q['O']>0
    result={'schema':'0x1.reference-organism-v2.verify','pass':all(checks.values()),'checks':checks,'quantity':q,'actions':len(o.actions),'episodes':len(o.episodes),'language_template_count':len(o.language._template_sources),'population_digest':o.population.digest(),'organism_digest':o.digest(),'claim':'POST_LEGACY_STRICT_CONTINUING_REFERENCE_ORGANISM_NOT_HUMAN_LEVEL_OR_PHYSICAL_ADULT','elapsed_ms':round((time.perf_counter()-started)*1000,3)}
    print('FOUNDRY_REFERENCE_ORGANISM_V2 '+('GREEN' if result['pass'] else 'RED')+' continuing=1 prompt=0 llm=0 population=1 learned_language=1 episodic=1 ambiguity=1 consequence=1')
    print(json.dumps(result,indent=2,sort_keys=True));raise SystemExit(0 if result['pass'] else 1)

if __name__=='__main__':main()
