#!/usr/bin/env python3
from __future__ import annotations

import copy,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))

from reference_organism_v2 import *
from reference_population_v1 import PopulationSpecV1
from reference_raw_surface_recipe_critic_v1 import RawSurfaceRecipeCriticV1
import reference_ephemeral_language_recipe_verify as alice_recipe
import reference_organism_world_language_verify as world

ALICE_PATH=Path(__file__).resolve().parents[2]/'data'/'alice.txt'
P=world.P
NOVEL=(world.BOB,world.INSPECT,world.SENSOR)
FORMAL=b'bob inspects the sensor.'
ALTERNATE=b'sensor, bob inspects.'


def build_raw():
    raw=ALICE_PATH.read_bytes()[:131072]
    surface,ecology,_state,_surface_ms,_recipe_ms=alice_recipe._alice_build(raw,0)
    ecology.compact_training_buffer()
    return surface,ecology


def prepare():
    o=ReferenceOrganismV2(PopulationSpecV1(65536,2,4,42,8));world.train(o)
    # First earn the nonlinguistic role/construction history while the original
    # surface is still unique. No held-out NOVEL event is stored as a whole.
    world.qualify_expression(o,(world.ALICE,world.INSPECT,world.SENSOR),(81001,81002))
    world.qualify_expression(o,(world.ALICE,world.TEST,world.VALVE),(81003,81004))
    world.qualify_expression(o,(world.BOB,world.TEST,world.VALVE),(81005,81006))
    assert not any(ep.atoms==NOVEL for ep in o.episodes)

    # Ground the common-ground port forms first through ordinary singleton contact.
    # Conditioned construction induction requires its live port surfaces to exist.
    world.partner(o,P)
    forms=(
        (world.ALICE,'alice'),(world.BOB,'bob'),
        (world.INSPECT,'inspects'),(world.TEST,'tests'),
        (world.SENSOR,'sensor'),(world.VALVE,'valve'),
    )
    for index,(atom,text) in enumerate(forms):
        for witness in range(2):
            source=84000+index*10+witness
            o.contact(CONTACT_SCENE,(7,0,1,atom),source,True,True)
            active,used,_binding,_relations=o._surface_view(o.current_scene)
            _ctx,conditions=o._surface_context(o.current_scene,used,active)
            assert conditions and conditions[0]
            o.contact(CONTACT_SURFACE,tuple(text.encode()),source+2000,True,True)

    # Learn two equal constructions inside the actual reinstated/common-ground
    # context. The alternate is object-fronted, so it cannot be absorbed as a
    # one-port form mutation. Only already-lived role examples are demonstrated.
    demos=(
        ((world.ALICE,world.INSPECT,world.SENSOR),'alice inspects the sensor.','sensor, alice inspects.'),
        ((world.ALICE,world.TEST,world.VALVE),'alice tests the valve.','valve, alice tests.'),
        ((world.BOB,world.TEST,world.VALVE),'bob tests the valve.','valve, bob tests.'),
    )
    conditioned_context=0
    for style_index in (1,0):  # alternate first, then standard; both receive 8 sources
        for witness in range(8):
            atoms,formal,alternate=demos[witness%len(demos)]
            source=85000+style_index*1000+witness
            o.contact(CONTACT_SCENE,(7,world.CTX,3,*atoms),source,True,True)
            active,used,_binding,_relations=o._surface_view(o.current_scene)
            ctx,conditions=o._surface_context(o.current_scene,used,active)
            conditioned_context=conditioned_context or int(ctx)
            assert int(ctx)==conditioned_context and all(conditions)
            o.contact(CONTACT_SURFACE,tuple((alternate if style_index else formal).encode()),source+2000,True,True)
    rows=o.language.template_candidates(conditioned_context,3)
    assert len(rows)==2 and len({r.support for r in rows})==1 and rows[0].support==8
    assert not any(ep.atoms==NOVEL for ep in o.episodes)
    return o


def stage_world(o,source):
    o.contact(CONTACT_WORLD_STATE,NOVEL,source,True,True)
    world.partner(o,P)
    before=len(o.episodes)
    scene=o._resident_world_scene()
    return scene,before


def main():
    started=time.perf_counter();checks={}
    base=prepare();checkpoint=copy.deepcopy(base.checkpoint())
    _surface,raw=build_raw();critic=RawSurfaceRecipeCriticV1(raw)

    baseline=ReferenceOrganismV2.restore(copy.deepcopy(checkpoint))
    bscene,before=stage_world(baseline,92001)
    checks['actual_world_constructs_heldout_content_frontier']=(
        bscene is not None and bscene.atoms==baseline.world_state==NOVEL
        and len(baseline.episodes)==before
        and baseline.last_world_scene_candidate_touches==1
        and not any(ep.atoms==NOVEL for ep in base.episodes)
    )
    bcands=critic.current_candidates(baseline,bscene)
    checks['world_content_has_two_equal_resident_surfaces']=(
        len(bcands)==2 and set(row.surface for row in bcands)=={FORMAL,ALTERNATE}
        and len({row.developmental_support for row in bcands})==1
    )
    # With no raw structural proposal the content is known but expression remains tied.
    checks['content_known_surface_tie_refuses_without_raw_pressure']=baseline.tick() is None

    expressed=ReferenceOrganismV2.restore(copy.deepcopy(checkpoint))
    escene,episodes_before=stage_world(expressed,92002)
    proposal=critic.propose(expressed,escene)
    action=expressed.tick(proposal)
    checks['raw_structure_realizes_world_originated_content']=(
        proposal is not None and isinstance(action,ActionV2)
        and action.payload==tuple(FORMAL)
        and action.scene_identity==escene.identity
        and escene.atoms==NOVEL
        and expressed.world_state_occurrence in action.contributors
        and len(expressed.episodes)==episodes_before
    )
    checks['surface_proposal_carries_no_world_or_text_payload']=(
        proposal is not None and not hasattr(proposal,'surface') and not hasattr(proposal,'atoms')
        and tuple(proposal.lexical_identities)==tuple(action.lexical_identities)
    )
    learned=expressed.contact(CONTACT_CONSEQUENCE,(action.ticket,1),P,True,True)
    checks['world_expression_earns_only_after_independent_return']=(
        learned.get('selection_credit',0)>0 and learned.get('selection_network_updates',0)>=1
    )

    # Same world event with raw higher Recipes removed remains semantically available
    # but cannot emit because surface selection is unresolved.
    cut_raw=copy.deepcopy(raw);cut_raw.recipes.clear();cut_raw._rebuild_index()
    cut=ReferenceOrganismV2.restore(copy.deepcopy(checkpoint));cscene,_=stage_world(cut,92003)
    cut_critic=RawSurfaceRecipeCriticV1(cut_raw);cut_proposal=cut_critic.propose(cut,cscene)
    checks['raw_recipe_lesion_preserves_content_drops_expression']=(
        cscene is not None and cscene.atoms==NOVEL and cut_proposal is None
        and cut.tick() is None and not cut.actions
    )

    # If the actual world changes after proposal formation, exact Scene validation blocks it.
    stale=ReferenceOrganismV2.restore(copy.deepcopy(checkpoint));sscene,_=stage_world(stale,92004)
    stale_proposal=critic.propose(stale,sscene)
    stale.contact(CONTACT_WORLD_STATE,(world.ALICE,world.TEST,world.SENSOR),92005,True,True)
    checks['changed_world_invalidates_surface_proposal']=stale.tick(stale_proposal) is None and not stale.actions

    formal_occ=len(raw.unfold_all(FORMAL));alternate_occ=len(raw.unfold_all(ALTERNATE))
    checks['alice_structure_independently_prefers_world_surface']=formal_occ>alternate_occ and formal_occ>=1 and alternate_occ==0
    checks['bounded_raw_and_world_candidate_work']=(
        critic.last_surface_candidates==2 and critic.last_raw_touches<64
        and expressed.last_world_scene_candidate_touches==1
    )
    checks['no_prompt_semantic_planner_or_runtime_llm']=all(
        not hasattr(critic,n) for n in ('prompt','semantic_plan','answer','llm','tokenizer')
    ) and not hasattr(expressed,'prompt')

    report={
        'schema':'0x1.reference-world-raw-corpus-expression.v1','pass':all(checks.values()),
        'checks':checks,'world_atoms':list(NOVEL),'candidate_surfaces':[row.surface.decode() for row in bcands],
        'selected':'' if action is None else bytes(action.payload).decode(),
        'formal_higher_closures':formal_occ,'alternate_higher_closures':alternate_occ,
        'raw_recipe_bytes':len(raw.packed_state()),'raw_candidate_touches':critic.last_raw_touches,
        'world_candidate_touches':expressed.last_world_scene_candidate_touches,
        'runtime_llm':False,'graph_flip':False,
        'claim':'ACTUAL_WORLD_EVENT_COMPOSES_HELDOUT_CONTENT_AND_REAL_RAW_CORPUS_RECIPES_SELECT_ONLY_ITS_SURFACE_REFERENCE_ONLY_NOT_LLM_PARITY',
        'elapsed_ms':round((time.perf_counter()-started)*1000,3),
    }
    print('FOUNDRY_WORLD_RAW_CORPUS_EXPRESSION '+('GREEN' if report['pass'] else 'RED')+
          f" world_candidates={report['world_candidate_touches']} raw_touches={report['raw_candidate_touches']} selected={report['selected']!r}")
    print(json.dumps(report,indent=2,sort_keys=True));raise SystemExit(0 if report['pass'] else 1)

if __name__=='__main__':main()
