#!/usr/bin/env python3
from __future__ import annotations
import copy,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
from reference_organism_v2 import *
from reference_population_v1 import PopulationSpecV1

CTX=8801;P=9901;SEE_V=77001;SEE_A=77002
ALICE,BOB,INSPECT,SENSOR,TEST,VALVE=201,202,303,403,302,402
ALICE2,REMOTE,VOICE,CARA,DANA,EVE=501,601,701,203,204,205
FA=(11,12,13,14);FA2=(11,12,13,99);FR=(71,72,73,74);FV=(11,12,13,88)
FC=(21,22,23,25);FD=(81,82,83,84);FE=(21,22,91,92)

def u(s):return tuple(s.encode())
def partner(o,p=P):o.contact(CONTACT_PARTNER_CONTEXT,(1,7,p),70000+p,True,True)
def feat(o,e,f,src):o.contact(CONTACT_ENTITY_FEATURES,(e,len(f),*f),src,True,True)
def name(o,e,text,src):
    o.contact(CONTACT_SCENE,(7,0,1,e),src,True,True);o.contact(CONTACT_SURFACE,u(text),src+1000,True,True)
def clause(o,atoms,text,src):
    o.contact(CONTACT_SCENE,(7,CTX,len(atoms),*atoms),src,True,True);o.contact(CONTACT_SURFACE,u(text),src+2000,True,True)
def see(o,entity,src,independent=True):
    return o.contact(CONTACT_WORLD_STATE,(entity,),src,True,independent)

def train(o):
    for e,f,src in ((ALICE,FA,8001),(BOB,(21,22,23,24),8002),(ALICE2,FA2,8003),(REMOTE,FR,8004),
                    (VOICE,FV,8005),(INSPECT,(35,36),8015),(SENSOR,(45,46),8016),(TEST,(33,34),8013),(VALVE,(43,44),8014)):
        feat(o,e,f,src)
    for e,s in ((ALICE,'alice'),(ALICE2,'alice'),(BOB,'bob'),(REMOTE,'zoe'),(INSPECT,'inspects'),(SENSOR,'sensor'),(TEST,'tests'),(VALVE,'valve')):
        name(o,e,s,10000+e);name(o,e,s,11000+e)
    clause(o,(ALICE,INSPECT,SENSOR),'alice inspects the sensor.',30003);clause(o,(ALICE,INSPECT,SENSOR),'alice inspects the sensor.',30004)
    clause(o,(ALICE,TEST,VALVE),'alice tests the valve.',30007);clause(o,(ALICE,TEST,VALVE),'alice tests the valve.',30008)
    clause(o,(BOB,TEST,VALVE),'bob tests the valve.',30005);clause(o,(BOB,TEST,VALVE),'bob tests the valve.',30006)
    clause(o,(REMOTE,INSPECT,SENSOR),'zoe inspects the sensor.',30009);clause(o,(REMOTE,INSPECT,SENSOR),'zoe inspects the sensor.',30010)
    assert o.language.template(CTX,3) is not None

def speak(o,atoms,src,p=P):
    partner(o,p);o.contact(CONTACT_SCENE,(7,CTX,len(atoms),*atoms),src,True,True);return o.tick()

def qualify_expression(o,atoms,sources,p=P,independent=True):
    actions=[]
    for source in sources:
        action=speak(o,atoms,source,p)
        assert isinstance(action,ActionV2)
        o.contact(CONTACT_CONSEQUENCE,(action.ticket,1),p,True,independent)
        actions.append(action)
    return tuple(actions)

def world_expression(o,atoms,source,p=P):
    o.contact(CONTACT_WORLD_STATE,atoms,source,True,True);world_occurrence=o.world_state_occurrence
    partner(o,p);before_episodes=len(o.episodes);action=o.tick()
    return action,world_occurrence,before_episodes

def main():
    t=time.perf_counter();checks={};spec=PopulationSpecV1(65536,2,4,42,8)
    o=ReferenceOrganismV2(spec);train(o)
    checks['no_language_action_before_world']=not o.actions
    before_lang=len(o.selection_configuration_revisions)
    see(o,ALICE,SEE_V,True);see(o,VOICE,SEE_A,True)
    checks['world_see_writes_shared_world_network']=o._world_revisions.row_count==2 and ALICE in o._world_marked_entities() and VOICE in o._world_marked_entities()
    checks['world_see_does_not_credit_language_or_soma']=len(o.selection_configuration_revisions)==before_lang and o._somatic_revisions.row_count==0 and not o.actions
    cp=o.checkpoint()
    checks['checkpoint_has_no_media_blob']=all(k not in cp for k in ('image','audio','png','wav','pixels','samples')) and isinstance(cp.get('world_revisions_packed'),str)
    en_o=ReferenceOrganismV2.restore(cp);en=speak(en_o,(ALICE,INSPECT,SENSOR),42001);ew=en_o._world_state_occurrences(CTX,(ALICE,INSPECT,SENSOR))
    checks['later_language_recruits_world_and_voice']=isinstance(en,ActionV2) and en.payload==u('alice inspects the sensor.') and len(ew)>=2 and all(row[3] in en.contributors for row in ew) and not any(k==PREF_VIEW for k,_,_,_ in en.selection_occurrences)
    other_o=ReferenceOrganismV2.restore(cp);other=speak(other_o,(ALICE,TEST,VALVE),43001);ow=other_o._world_state_occurrences(CTX,(ALICE,TEST,VALVE))
    checks['other_construction_recruits_same_world']=isinstance(other,ActionV2) and bool(ow) and all(row[3] in other.contributors for row in ow)
    remat_o=ReferenceOrganismV2.restore(cp);remat=speak(remat_o,(ALICE2,INSPECT,SENSOR),44001);rw=remat_o._world_state_occurrences(CTX,(ALICE2,INSPECT,SENSOR))
    checks['overlapping_identity_recruits_world']=isinstance(remat,ActionV2) and bool(rw) and ALICE in remat_o._overlapping_entities(ALICE2) and all(row[3] in remat.contributors for row in rw)
    pun=ReferenceOrganismV2.restore(cp);first=speak(pun,(ALICE,INSPECT,SENSOR),53001)
    if isinstance(first,ActionV2):pun.contact(CONTACT_CONSEQUENCE,(first.ticket,-1),P,True,True)
    leak=speak(pun,(ALICE2,INSPECT,SENSOR),53002)
    checks['punished_name_silences_overlapping_named_identity']=isinstance(first,ActionV2) and leak is None
    bob_o=ReferenceOrganismV2.restore(cp);bob=speak(bob_o,(BOB,TEST,VALVE),45001)
    checks['unmarked_referent_still_speaks']=isinstance(bob,ActionV2) and bob.payload==u('bob tests the valve.') and not bob_o._world_state_occurrences(CTX,(BOB,TEST,VALVE))
    remote_o=ReferenceOrganismV2.restore(cp);remote=speak(remote_o,(REMOTE,INSPECT,SENSOR),46001)
    checks['unrelated_features_do_not_inherit_world']=isinstance(remote,ActionV2) and remote.payload==u('zoe inspects the sensor.') and not remote_o._world_state_occurrences(CTX,(REMOTE,INSPECT,SENSOR))
    yoked=ReferenceOrganismV2(spec);train(yoked);see(yoked,ALICE,SEE_V,False)
    checks['yoked_see_cannot_write_world']=yoked._world_revisions.row_count==0 and isinstance(speak(yoked,(ALICE,INSPECT,SENSOR),47001),ActionV2)
    cut=ReferenceOrganismV2.restore(cp);cut.contact(CONTACT_WITHDRAW_SOURCE,(SEE_V,),88002,True,True);ca=speak(cut,(ALICE,INSPECT,SENSOR),48001);cw=cut._world_state_occurrences(CTX,(ALICE,INSPECT,SENSOR))
    checks['visual_source_withdrawal_keeps_voice_only']=isinstance(ca,ActionV2) and bool(cw) and all(row[0]==VOICE for row in cw)
    before=tuple(cut._world_revisions.iter_revisions());see(cut,ALICE,SEE_V,True)
    checks['withdrawn_see_cannot_rewrite_world']=tuple(cut._world_revisions.iter_revisions())==before
    poison=ReferenceOrganismV2(spec);train(poison);see(poison,ALICE,SEE_A,True)
    poison.contact(CONTACT_WITHDRAW_SOURCE,(SEE_V,),88010,True,True)
    poison.contact(CONTACT_WORLD_STATE,(BOB,),SEE_V,True,True)
    checks['withdrawn_see_cannot_drive_planning']=poison.world_state==(ALICE,) and poison.world_source==SEE_A and BOB not in poison._world_marked_entities()
    gone=ReferenceOrganismV2(spec);train(gone);see(gone,ALICE,SEE_A,True)
    gone.contact(CONTACT_WITHDRAW_SOURCE,(SEE_A,),88012,True,True)
    checks['withdrawn_see_clears_planning_world']=gone.world_state is None and gone.world_source==0
    mute_p=ReferenceOrganismV2(spec);train(mute_p);partner(mute_p,P)
    mute_p.contact(CONTACT_SCENE,(7,CTX,3,ALICE,INSPECT,SENSOR),51001,True,True)
    mute_p.contact(CONTACT_WITHDRAW_SOURCE,(P,),88020,True,True)
    checks['withdrawn_partner_does_not_author_language']=mute_p.tick() is None and not mute_p.partner_present
    hold=ReferenceOrganismV2(spec);train(hold);partner(hold,P)
    live=hold.contact(CONTACT_SCENE,(7,CTX,3,ALICE,INSPECT,SENSOR),52001,True,True);DEAD=43001
    hold.contact(CONTACT_WITHDRAW_SOURCE,(DEAD,),88030,True,True)
    try:
        hold.contact(CONTACT_SCENE,(7,CTX,3,BOB,TEST,VALVE),DEAD,True,True);planted=True
    except ValueError:
        planted=False
    checks['withdrawn_source_cannot_plant_scene']=not planted and hold.current_scene is not None and hold.current_scene.identity==live
    mute=ReferenceOrganismV2.restore(cp);mute.contact(CONTACT_WITHDRAW_SOURCE,(SEE_A,),88003,True,True);ma=speak(mute,(ALICE,INSPECT,SENSOR),49001);mw=mute._world_state_occurrences(CTX,(ALICE,INSPECT,SENSOR))
    checks['voice_source_withdrawal_keeps_visual_only']=isinstance(ma,ActionV2) and bool(mw) and all(row[0]==ALICE for row in mw)
    both=ReferenceOrganismV2.restore(cp);both.contact(CONTACT_WITHDRAW_SOURCE,(SEE_V,),88004,True,True);both.contact(CONTACT_WITHDRAW_SOURCE,(SEE_A,),88005,True,True)
    checks['withdrawing_both_drops_world_recruitment']=isinstance(speak(both,(ALICE,INSPECT,SENSOR),50001),ActionV2) and not both._world_state_occurrences(CTX,(ALICE,INSPECT,SENSOR))

    # Actual nonlinguistic event contact must be able to unfold a previously
    # consequence-qualified shared construction without a fresh host scene.
    resident=ReferenceOrganismV2(spec);train(resident)
    qualify_expression(resident,(ALICE,INSPECT,SENSOR),(61001,61002))
    resident_qualified_checkpoint=resident.checkpoint()
    resident.contact(CONTACT_WORLD_STATE,(ALICE,INSPECT,SENSOR),71001,True,True)
    resident_world_occurrence=resident.world_state_occurrence;partner(resident,P)
    resident_checkpoint=resident.checkpoint();episodes_before=len(resident.episodes)
    resident_action=resident.tick();resident_touches=resident.last_world_scene_touches
    replay=ReferenceOrganismV2.restore(resident_checkpoint);replay_action=replay.tick()
    checks['actual_world_occurrence_reinstates_outward_construction']=(
        isinstance(resident_action,ActionV2)
        and resident_action.payload==u('alice inspects the sensor.')
        and resident_world_occurrence in resident_action.contributors
        and all(row[3] in resident_action.contributors
                for row in resident_action.selection_occurrences)
        and len(resident.episodes)==episodes_before
    )
    checks['world_reinstatement_checkpoint_replays']=(
        isinstance(replay_action,ActionV2)
        and replay_action.payload==resident_action.payload
        and replay_action.planned_payload==resident_action.planned_payload
        and replay.world_state_occurrence in replay_action.contributors
    )
    checks['world_reinstatement_is_sparse_over_shared_relations']=(
        resident_touches==2 and len(resident.shared_episode_relations)==2
    )
    source_variant=ReferenceOrganismV2.restore(resident_qualified_checkpoint)
    variant_action,variant_occurrence,_=world_expression(
        source_variant,(ALICE,INSPECT,SENSOR),71999,
    )
    checks['new_world_source_preserves_learned_expression_envelope']=(
        isinstance(variant_action,ActionV2)
        and variant_action.payload==resident_action.payload
        and variant_occurrence in variant_action.contributors
        and source_variant.world_source==71999!=resident.world_source
    )

    silent=ReferenceOrganismV2(spec);train(silent)
    silent.contact(CONTACT_WORLD_STATE,(ALICE,INSPECT,SENSOR),71002,True,True)
    partner(silent,P);checks['language_demonstration_without_consequence_stays_silent']=silent.tick() is None and not silent.shared_episode_relations
    yoked=ReferenceOrganismV2(spec);train(yoked)
    qualify_expression(yoked,(ALICE,INSPECT,SENSOR),(62001,62002),independent=False)
    yoked.contact(CONTACT_WORLD_STATE,(ALICE,INSPECT,SENSOR),71003,True,True);partner(yoked,P)
    checks['yoked_expression_return_cannot_author_reinstatement']=yoked.tick() is None and not yoked.shared_episode_relations

    no_partner=ReferenceOrganismV2.restore(resident_checkpoint)
    no_partner.contact(CONTACT_PARTNER_CONTEXT,(0,0,0),71004,True,True)
    checks['actual_world_without_partner_stays_silent']=no_partner.tick() is None
    superseded=ReferenceOrganismV2.restore(resident_checkpoint)
    superseded.contact(CONTACT_SCENE,(7,CTX,1,99991),71009,True,True)
    checks['later_actual_scene_retires_stale_world_expression']=(
        superseded.world_state==(ALICE,INSPECT,SENSOR)
        and superseded.world_state_occurrence==0 and superseded.tick() is None
    )
    world_cut=ReferenceOrganismV2.restore(resident_checkpoint)
    world_cut.contact(CONTACT_WITHDRAW_SOURCE,(71001,),71005,True,True)
    checks['world_source_withdrawal_blocks_reinstatement']=world_cut.tick() is None and world_cut.world_state is None
    partner_cut=ReferenceOrganismV2.restore(resident_checkpoint)
    partner_cut.contact(CONTACT_WITHDRAW_SOURCE,(P,),71006,True,True)
    checks['partner_withdrawal_blocks_reinstatement']=partner_cut.tick() is None and not partner_cut.partner_present
    focal=ReferenceOrganismV2.restore(resident_checkpoint);focal.shared_episode_relations.clear();focal._rebuild_runtime_indices()
    checks['focal_shared_relation_lesion_blocks_reinstatement']=focal.tick() is None
    expression_cut=ReferenceOrganismV2.restore(resident_qualified_checkpoint)
    expression_cut.contact(CONTACT_WITHDRAW_SOURCE,(61001,),72001,True,True)
    expression_cut.contact(CONTACT_WITHDRAW_SOURCE,(61002,),72002,True,True)
    cut_action,_,_=world_expression(expression_cut,(ALICE,INSPECT,SENSOR),72003)
    checks['expression_episode_source_withdrawal_blocks_reinstatement']=cut_action is None

    ambiguous=ReferenceOrganismV2(spec);train(ambiguous)
    clause(ambiguous,(SENSOR,INSPECT,ALICE),'the sensor inspects alice.',63001)
    clause(ambiguous,(SENSOR,INSPECT,ALICE),'the sensor inspects alice.',63002)
    qualify_expression(ambiguous,(ALICE,INSPECT,SENSOR),(63011,63012))
    qualify_expression(ambiguous,(SENSOR,INSPECT,ALICE),(63021,63022))
    ambiguous.contact(CONTACT_WORLD_STATE,(ALICE,INSPECT,SENSOR),71007,True,True);partner(ambiguous,P)
    ambiguity_action=ambiguous.tick()
    checks['role_shuffled_equal_support_remains_ambiguous']=(
        ambiguity_action is None and ambiguous.information_need[:1]==(7,)
        and len(ambiguous.information_need)==3
    )
    stronger=ReferenceOrganismV2(spec);train(stronger)
    clause(stronger,(SENSOR,INSPECT,ALICE),'the sensor inspects alice.',64001)
    clause(stronger,(SENSOR,INSPECT,ALICE),'the sensor inspects alice.',64002)
    qualify_expression(stronger,(ALICE,INSPECT,SENSOR),(64011,64012))
    qualify_expression(stronger,(SENSOR,INSPECT,ALICE),(64021,))
    stronger_action,_,_=world_expression(stronger,(ALICE,INSPECT,SENSOR),71008)
    checks['independent_expression_support_resolves_role_competition']=(
        isinstance(stronger_action,ActionV2)
        and stronger_action.payload==u('alice inspects the sensor.')
    )

    # Productive composition: every current filler has consequence-qualified
    # positional history, but neither held-out event was ever stored as a whole.
    productive=ReferenceOrganismV2(spec);train(productive)
    for entity,features,label,source in (
        (CARA,FC,'cara',8050),(DANA,FD,'dana',8060),(EVE,FE,'eve',8070),
    ):
        feat(productive,entity,features,source)
        name(productive,entity,label,12000+entity);name(productive,entity,label,13000+entity)
    qualify_expression(productive,(ALICE,INSPECT,SENSOR),(81001,81002))
    qualify_expression(productive,(ALICE,TEST,VALVE),(81003,81004))
    qualify_expression(productive,(BOB,TEST,VALVE),(81005,81006))
    productive_checkpoint=productive.checkpoint()
    novel=(BOB,INSPECT,SENSOR)
    novel_o=ReferenceOrganismV2.restore(productive_checkpoint)
    novel_action,novel_occurrence,novel_episode_count=world_expression(
        novel_o,novel,82001,
    )
    novel_scene=(novel_o._scene_by_id.get(novel_action.scene_identity)
                 if isinstance(novel_action,ActionV2) else None)
    checks['heldout_event_composes_only_current_world_fillers']=(
        isinstance(novel_action,ActionV2)
        and novel_action.payload==u('bob inspects the sensor.')
        and novel_scene is not None and novel_scene.atoms==novel_o.world_state==novel
        and novel_occurrence in novel_action.contributors
        and len(novel_o.episodes)==novel_episode_count
        and not any(ep.atoms==novel for ep in productive.episodes)
    )
    checks['novel_composition_is_sparse_and_unique']=(
        novel_o.last_world_scene_candidate_touches==1
        and novel_o.last_world_scene_touches==6
    )
    reordered=ReferenceOrganismV2.restore(productive_checkpoint)
    reordered_action,_,_=world_expression(reordered,(SENSOR,BOB,INSPECT),82002)
    checks['raw_world_payload_order_does_not_select_roles']=(
        isinstance(reordered_action,ActionV2)
        and reordered_action.payload==novel_action.payload
    )
    second_novel=ReferenceOrganismV2.restore(productive_checkpoint)
    second_action,_,_=world_expression(second_novel,(ALICE,TEST,SENSOR),82003)
    checks['second_heldout_combination_uses_same_role_law']=(
        isinstance(second_action,ActionV2)
        and second_action.payload==u('alice tests the sensor.')
        and not any(ep.atoms==(ALICE,TEST,SENSOR) for ep in productive.episodes)
    )

    # A newly grounded and named entity has no stored clause or role episode.
    # Distributed overlap with Bob may nominate Bob's lived first-position
    # support, but the actual world Occurrence supplies the transient filler.
    cara=ReferenceOrganismV2.restore(productive_checkpoint)
    cara_action,cara_occurrence,cara_episode_count=world_expression(
        cara,(CARA,INSPECT,SENSOR),82010,
    )
    cara_scene=(cara._scene_by_id.get(cara_action.scene_identity)
                if isinstance(cara_action,ActionV2) else None)
    checks['new_grounded_filler_enters_learned_role_without_stored_clause']=(
        isinstance(cara_action,ActionV2)
        and cara_action.payload==u('cara inspects the sensor.')
        and cara_scene is not None and cara_scene.atoms==(CARA,INSPECT,SENSOR)
        and cara_occurrence in cara_action.contributors
        and len(cara.episodes)==cara_episode_count
        and not any(CARA in ep.atoms and ep.context==CTX for ep in productive.episodes)
    )
    checks['distributed_role_transfer_is_sparse_and_transient']=(
        cara.last_world_scene_touches==6
        and cara.last_world_scene_feature_touches==20
        and cara.last_world_scene_candidate_touches==1
    )
    cara_second=ReferenceOrganismV2.restore(productive_checkpoint)
    cara_second_action,_,_=world_expression(cara_second,(CARA,TEST,VALVE),82011)
    checks['new_filler_reuses_second_heldout_event_construction']=(
        isinstance(cara_second_action,ActionV2)
        and cara_second_action.payload==u('cara tests the valve.')
        and not any(CARA in ep.atoms and ep.context==CTX for ep in productive.episodes)
    )
    cara_reordered=ReferenceOrganismV2.restore(productive_checkpoint)
    cara_reordered_action,_,_=world_expression(cara_reordered,(SENSOR,CARA,INSPECT),82012)
    checks['new_filler_world_order_does_not_select_roles']=(
        isinstance(cara_reordered_action,ActionV2)
        and cara_reordered_action.payload==cara_action.payload
    )
    cara_pre=ReferenceOrganismV2.restore(productive_checkpoint)
    cara_pre.contact(CONTACT_WORLD_STATE,(CARA,INSPECT,SENSOR),82013,True,True);partner(cara_pre,P)
    cara_checkpoint=cara_pre.checkpoint();cara_direct=cara_pre.tick()
    cara_replay=ReferenceOrganismV2.restore(cara_checkpoint);cara_replayed=cara_replay.tick()
    checks['distributed_role_transfer_checkpoint_replays']=(
        isinstance(cara_direct,ActionV2) and isinstance(cara_replayed,ActionV2)
        and cara_direct.payload==cara_replayed.payload
        and cara_direct.planned_payload==cara_replayed.planned_payload
    )
    unrelated=ReferenceOrganismV2.restore(productive_checkpoint)
    unrelated_action,_,_=world_expression(unrelated,(DANA,INSPECT,SENSOR),82014)
    checks['unrelated_grounded_features_do_not_inherit_role']=unrelated_action is None
    threshold=ReferenceOrganismV2.restore(productive_checkpoint)
    threshold_action,_,_=world_expression(threshold,(EVE,INSPECT,SENSOR),82015)
    checks['subthreshold_feature_overlap_does_not_inherit_role']=threshold_action is None
    cara_cut=ReferenceOrganismV2.restore(productive_checkpoint)
    cara_cut.contact(CONTACT_WITHDRAW_SOURCE,(81005,),82016,True,True)
    cara_cut.contact(CONTACT_WITHDRAW_SOURCE,(81006,),82017,True,True)
    cara_cut_action,_,_=world_expression(cara_cut,(CARA,INSPECT,SENSOR),82018)
    checks['qualifying_role_source_withdrawal_blocks_feature_transfer']=cara_cut_action is None
    cara_lesion=ReferenceOrganismV2.restore(productive_checkpoint)
    cara_lesion.shared_episode_relations=[row for row in cara_lesion.shared_episode_relations
        if cara_lesion._episode_by_id[row.episode_identity].atoms[0]!=BOB]
    cara_lesion._rebuild_runtime_indices()
    cara_lesion_action,_,_=world_expression(cara_lesion,(CARA,INSPECT,SENSOR),82019)
    checks['focal_role_history_lesion_blocks_feature_transfer']=cara_lesion_action is None

    feature_cut=ReferenceOrganismV2.restore(productive_checkpoint)
    feature_cut.contact(CONTACT_WITHDRAW_SOURCE,(8050,),82200,True,True)
    feature_cut_action,_,_=world_expression(feature_cut,(CARA,INSPECT,SENSOR),82201)
    checks['sole_feature_source_withdrawal_blocks_role_transfer']=(
        feature_cut_action is None
        and feature_cut.entity_features[CARA]==FC
        and not feature_cut._active_entity_features(CARA)
    )
    dual_feature=ReferenceOrganismV2.restore(productive_checkpoint)
    feat(dual_feature,CARA,FC,8051);dual_feature_checkpoint=dual_feature.checkpoint()
    dual_feature.contact(CONTACT_WITHDRAW_SOURCE,(8050,),82202,True,True)
    dual_feature_action,_,_=world_expression(dual_feature,(CARA,INSPECT,SENSOR),82203)
    checks['redundant_feature_source_survives_one_withdrawal']=(
        isinstance(dual_feature_action,ActionV2)
        and dual_feature_action.payload==u('cara inspects the sensor.')
        and dual_feature._active_entity_features(CARA)==FC
    )
    both_feature_cut=ReferenceOrganismV2.restore(dual_feature_checkpoint)
    checks['feature_source_sets_checkpoint_exactly']=(
        both_feature_cut.entity_feature_sources==dual_feature.entity_feature_sources
        and both_feature_cut.checkpoint()==dual_feature_checkpoint
    )
    both_feature_cut.contact(CONTACT_WITHDRAW_SOURCE,(8050,),82204,True,True)
    both_feature_cut.contact(CONTACT_WITHDRAW_SOURCE,(8051,),82205,True,True)
    both_feature_action,_,_=world_expression(both_feature_cut,(CARA,INSPECT,SENSOR),82206)
    checks['all_feature_sources_withdrawn_blocks_role_transfer']=(
        both_feature_action is None and not both_feature_cut._active_entity_features(CARA)
    )
    malformed_organism=ReferenceOrganismV2(spec);feat(malformed_organism,CARA,FC,8050)
    malformed=malformed_organism.checkpoint();malformed['entity_feature_sources'].pop()
    try:ReferenceOrganismV2.restore(malformed)
    except ValueError:malformed_feature_sources_refused=True
    else:malformed_feature_sources_refused=False
    checks['malformed_feature_source_checkpoint_refused']=malformed_feature_sources_refused
    withdrawn_ingress=ReferenceOrganismV2(spec)
    withdrawn_ingress.contact(CONTACT_WITHDRAW_SOURCE,(82900,),82207,True,True)
    before_features=dict(withdrawn_ingress.entity_features)
    try:feat(withdrawn_ingress,CARA,FC,82900)
    except ValueError:withdrawn_feature_contact_refused=True
    else:withdrawn_feature_contact_refused=False
    checks['withdrawn_source_cannot_reassert_entity_features']=(
        withdrawn_feature_contact_refused
        and withdrawn_ingress.entity_features==before_features
    )
    replaced_feature=feature_cut
    feat(replaced_feature,CARA,FD,8051)
    replaced_feature_action,_,_=world_expression(replaced_feature,(CARA,INSPECT,SENSOR),82208)
    checks['changed_feature_contact_replaces_prior_vector_and_provenance']=(
        replaced_feature_action is None
        and replaced_feature._active_entity_features(CARA)==FD
        and replaced_feature.entity_feature_sources[CARA]=={8051}
    )

    unqualified=ReferenceOrganismV2(spec);train(unqualified)
    feat(unqualified,CARA,FC,82100);name(unqualified,CARA,'cara',82101);name(unqualified,CARA,'cara',82102)
    unqualified_action,_,_=world_expression(unqualified,(CARA,INSPECT,SENSOR),82103)
    checks['feature_overlap_without_positive_role_consequence_stays_silent']=unqualified_action is None

    # Two new fillers that overlap both lived role populations equally leave
    # the transient bindings unresolved rather than letting feature order win.
    collision=ReferenceOrganismV2(spec)
    old_a,old_b,new_a,new_b,obj=9301,9302,9303,9304,9305
    for entity,features,label,source in (
        (old_a,(1,2,3,4),'olda',92001),(old_b,(1,2,3,5),'oldb',92002),
        (new_a,(1,2,3,6),'newa',92003),(new_b,(1,2,3,7),'newb',92004),
        (obj,(8,9),'object',92005),
    ):
        feat(collision,entity,features,source)
        name(collision,entity,label,source+100);name(collision,entity,label,source+200)
    clause(collision,(old_a,old_b,obj),'olda oldb the object.',92100)
    clause(collision,(old_a,old_b,obj),'olda oldb the object.',92101)
    qualify_expression(collision,(old_a,old_b,obj),(92110,92111))
    collision_action,_,_=world_expression(collision,(new_a,new_b,obj),92120)
    checks['equal_cross_role_feature_support_remains_ambiguous']=(
        collision_action is None and collision.information_need[:1]==(7,)
        and len(collision.information_need)==3
        and collision.last_world_scene_candidate_touches==2
    )
    novel_pre=ReferenceOrganismV2.restore(productive_checkpoint)
    novel_pre.contact(CONTACT_WORLD_STATE,novel,82004,True,True);partner(novel_pre,P)
    novel_pre_checkpoint=novel_pre.checkpoint();novel_direct=novel_pre.tick()
    novel_replay=ReferenceOrganismV2.restore(novel_pre_checkpoint);novel_replayed=novel_replay.tick()
    checks['novel_role_composition_checkpoint_replays']=(
        isinstance(novel_direct,ActionV2) and isinstance(novel_replayed,ActionV2)
        and novel_direct.payload==novel_replayed.payload
        and novel_direct.planned_payload==novel_replayed.planned_payload
    )
    role_cut=ReferenceOrganismV2.restore(productive_checkpoint)
    role_cut.contact(CONTACT_WITHDRAW_SOURCE,(81005,),83001,True,True)
    role_cut.contact(CONTACT_WITHDRAW_SOURCE,(81006,),83002,True,True)
    role_cut_action,_,_=world_expression(role_cut,novel,83003)
    checks['role_source_withdrawal_blocks_novel_composition']=role_cut_action is None
    role_lesion=ReferenceOrganismV2.restore(productive_checkpoint)
    role_lesion.shared_episode_relations=[row for row in role_lesion.shared_episode_relations
        if role_lesion._episode_by_id[row.episode_identity].atoms[0]!=BOB]
    role_lesion._rebuild_runtime_indices();role_lesion_action,_,_=world_expression(role_lesion,novel,83004)
    checks['focal_position_history_lesion_blocks_novel_composition']=role_lesion_action is None

    # Two equally supported assignments over the same current atoms remain
    # unresolved; one additional actual consequence resolves the competition.
    role_amb=ReferenceOrganismV2(spec);train(role_amb)
    clause(role_amb,(INSPECT,ALICE,VALVE),'inspects alice the valve.',84001)
    clause(role_amb,(INSPECT,ALICE,VALVE),'inspects alice the valve.',84002)
    clause(role_amb,(ALICE,BOB,SENSOR),'alice bob the sensor.',84003)
    clause(role_amb,(ALICE,BOB,SENSOR),'alice bob the sensor.',84004)
    qualify_expression(role_amb,(BOB,TEST,VALVE),(84011,))
    qualify_expression(role_amb,(ALICE,INSPECT,SENSOR),(84012,))
    qualify_expression(role_amb,(INSPECT,ALICE,VALVE),(84013,))
    qualify_expression(role_amb,(ALICE,BOB,SENSOR),(84014,))
    role_amb_checkpoint=role_amb.checkpoint()
    role_amb_action,_,_=world_expression(role_amb,novel,84020)
    checks['equal_composed_role_assignments_remain_ambiguous']=(
        role_amb_action is None and role_amb.information_need[:1]==(7,)
        and len(role_amb.information_need)==3
        and role_amb.last_world_scene_candidate_touches==2
    )
    role_strong=ReferenceOrganismV2.restore(role_amb_checkpoint)
    qualify_expression(role_strong,(BOB,TEST,VALVE),(84015,))
    role_strong_action,_,_=world_expression(role_strong,novel,84021)
    checks['one_more_lived_role_consequence_resolves_composition']=(
        isinstance(role_strong_action,ActionV2)
        and role_strong_action.payload==u('bob inspects the sensor.')
    )

    # A causally earned high-fanout position ecology must refuse before factorial
    # enumeration.  Seven fillers each have lived support in all seven positions.
    bounded=ReferenceOrganismV2(spec);current_atoms=tuple(range(9001,9008));decoys=tuple(range(9101,9108))
    for atom in (*current_atoms,*decoys):
        name(bounded,atom,f'w{atom}',85000+atom);name(bounded,atom,f'w{atom}',86000+atom)
    decoy_surface=' '.join(f'w{atom}' for atom in decoys)+'.'
    clause(bounded,decoys,decoy_surface,87001);clause(bounded,decoys,decoy_surface,87002)
    for position in range(7):
        for offset,atom in enumerate(current_atoms):
            configuration=list(decoys);configuration[position]=atom
            qualify_expression(bounded,tuple(configuration),(88000+position*7+offset,))
    bounded_action,_,_=world_expression(bounded,current_atoms,89001)
    checks['causally_earned_role_frontier_refuses_before_explosion']=(
        bounded_action is None and bounded.information_need==(7,)
        and bounded.last_world_scene_candidate_touches==0
        and bounded.last_world_scene_touches==49
    )

    corrupt=copy.deepcopy(resident_checkpoint);corrupt['world_state_occurrence']=(1<<62)
    try:ReferenceOrganismV2.restore(corrupt)
    except ValueError:world_occurrence_tamper_refused=True
    else:world_occurrence_tamper_refused=False
    checks['world_occurrence_checkpoint_tamper_refused']=world_occurrence_tamper_refused
    checks['language_is_outer_not_the_model']=not hasattr(o,'translate') and not hasattr(o,'imagine') and not hasattr(o,'picture') and not hasattr(o,'hear')
    checks['no_stored_media']=not hasattr(o,'image') and not hasattr(o,'audio') and PREF_VIEW not in (PREF_TEMPLATE,PREF_LEXEME,PREF_FORM)
    result={'schema':'agi.reference-organism-world-language.v5','pass':all(checks.values()),'checks':checks,'check_count':len(checks),'runtime_llm':False,'graph_flip':False,'organism_checkpoint_schema':resident_checkpoint['schema'],'resident_world_scene_touches':resident_touches,'resident_world_checkpoint_bytes':len(json.dumps(resident_checkpoint,sort_keys=True,separators=(',',':'))),'novel_role_relation_touches':novel_o.last_world_scene_touches,'novel_role_candidate_touches':novel_o.last_world_scene_candidate_touches,'distributed_role_relation_touches':cara.last_world_scene_touches,'distributed_role_feature_touches':cara.last_world_scene_feature_touches,'distributed_role_candidate_touches':cara.last_world_scene_candidate_touches,'feature_source_rows':len(dual_feature.entity_feature_sources),'bounded_role_relations':bounded.last_world_scene_touches,'papers':['Griffin Bock 2000 event apprehension precedes formulation','Bock 1986 syntactic persistence','Akhtar Tomasello 1997 developmental productivity','Frankland Greene 2015 transient semantic role values','Boyd Goldberg 2012 graded construction generalization','Johnson Hashtroudi Lindsay 1993 source monitoring','Newcombe et al 2011 developmental source binding'],'claim':'ACTUAL_WORLD_OCCURRENCE_COMPOSES_SOURCE_RETAINED_CONSEQUENCE_QUALIFIED_DISTRIBUTED_ROLE_FILLER_CONSTRUCTION_REFERENCE_ONLY_NOT_DIRECT_PARITY_OR_HUMAN_LANGUAGE','elapsed_ms':round((time.perf_counter()-t)*1000,3)}
    print('FOUNDRY_REFERENCE_ORGANISM_WORLD_LANGUAGE '+('GREEN' if result['pass'] else 'RED')+' world_to_language=source_retained_role_composition voice_overlap=1 media=0 language_outer=1')
    print(json.dumps(result,indent=2,sort_keys=True));raise SystemExit(0 if result['pass'] else 1)
if __name__=='__main__':main()
