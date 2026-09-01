#!/usr/bin/env python3
"""Make a learned form dependency follow resident relation ancestry, not slots."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import time

import reference_organism_v2 as organism
from reference_organism_v2 import *
from reference_organism_surface_state_v1 import surface_conditions
from reference_population_v1 import PopulationSpecV1

META=0xFEE1;LEFT=0xA11CE;RIGHT=0xB0B;WRONG_LEFT=0xA11CF
NEST_PAIR=(0xD101,0xD102,0xD103,0xD104)
NEST_LEFT=0xD105;NEST_RIGHT=0xD106;NEST_ROOT=0xD107;WRONG_NEST_PAIR=0xD108
ROLE_CONTROLLER=0xE101;ROLE_DISTRACTOR=0xE102;ROLE_OTHER=(0xE103,0xE104)
ROLE_LEFT=0xE105;ROLE_RIGHT=0xE106;ROLE_ROOT=0xE107
AGR_CONTROLLER=0xF101;AGR_DISTRACTOR=0xF102;AGR_OTHER=(0xF103,0xF104)
AGR_AUX=0xF108
PARTNER=8842;BODY=(831,832,833)
DONOR_A=(101,102,103,104);DONOR_B=(201,202,203,204);HELD=(301,302,303,304)
NEST_A=(401,402,403,404,405,406,407,408)
NEST_B=(501,502,503,504,505,506,507,508)
NEST_HELD=(601,602,603,604,605,606,607,608)
ROLE_A=(701,702,703,704,705,706,707,708)
ROLE_B=(801,802,803,804,805,806,807,808)
ROLE_HELD=(901,902,903,904,905,906,907,908)
AGR_A=(1001,AGR_AUX,1002,1003,1004,1005,1006,1007)
AGR_B=(1101,1102,1103,AGR_AUX,1104,1105,1106,1107)
AGR_HELD=(1201,1202,1203,1204,1205,1206,1207,AGR_AUX)
WORDS={101:'dak',102:'near',103:'the',104:'lum',201:'nif',202:'near',203:'the',204:'pek',
       301:'wug',302:'near',303:'the',304:'blick',
       401:'the',402:'dak',403:'that',404:'is',405:'near',406:'the lum',407:'can',408:'test',
       501:'a',502:'nif',503:'which',504:'was',505:'beside',506:'the vok',507:'will',508:'inspect',
       601:'one',602:'wug',603:'that',604:'seems',605:'around',606:'the blick',607:'might',608:'verify',
       701:'the',702:'can',703:'dak',704:'that',705:'is',706:'near the lum',707:'test',708:'now',
       801:'nif',802:'which',803:'a',804:'will',805:'was',806:'beside the vok',807:'inspect',808:'now',
       901:'wug',902:'that',903:'seems',904:'around the blick',905:'verify',906:'now',907:'one',908:'might',
       AGR_AUX:'is',1001:'dak',1002:'the',1003:'near the',1004:'lum',1005:'that',1006:'calm',1007:'now',
       1101:'the',1102:'beside the',1103:'nif',1104:'vok',1105:'which',1106:'ready',1107:'today',
       1201:'some',1202:'around the',1203:'blick',1204:'that',1205:'steady',1206:'soon',1207:'wug'}


def u(text):return tuple(text.encode())


def singleton(o,atom,source):
    return o.contact(CONTACT_SCENE,(7,1,1,atom),source,True,True)


def relation_scene(o,atoms,source,left_relation=LEFT):
    leaves=[singleton(o,atom,source+slot) for slot,atom in enumerate(atoms)]
    o.contact(CONTACT_SCENE_LINK,(leaves[0],leaves[1],left_relation),source+10,True,True)
    left=o.next_scene-1
    o.contact(CONTACT_SCENE_LINK,(leaves[2],leaves[3],RIGHT),source+11,True,True)
    right=o.next_scene-1
    o.contact(CONTACT_SCENE_LINK,(left,right,META),source,True,True)
    return o._scene_by_id[o.next_scene-1]


def nested_scene(o,atoms,source,wrong_pair=False):
    leaves=[singleton(o,atom,source+slot) for slot,atom in enumerate(atoms)]
    pairs=[]
    for slot in range(4):
        relation=WRONG_NEST_PAIR if wrong_pair and slot==1 else NEST_PAIR[slot]
        o.contact(CONTACT_SCENE_LINK,(leaves[2*slot],leaves[2*slot+1],relation),
                  source+20+slot,True,True)
        pairs.append(o.next_scene-1)
    o.contact(CONTACT_SCENE_LINK,(pairs[0],pairs[1],NEST_LEFT),source+30,True,True)
    left=o.next_scene-1
    o.contact(CONTACT_SCENE_LINK,(pairs[2],pairs[3],NEST_RIGHT),source+31,True,True)
    right=o.next_scene-1
    o.contact(CONTACT_SCENE_LINK,(left,right,NEST_ROOT),source+32,True,True)
    return o._scene_by_id[o.next_scene-1]


def demonstrate_nested(o,atoms,text,source):
    scene=nested_scene(o,atoms,source)
    o.contact(CONTACT_SURFACE,u(text),source,True,True)
    return scene


def role_scene(o,atoms,relations,source):
    leaves=[singleton(o,atom,source+slot) for slot,atom in enumerate(atoms)]
    pairs=[]
    for slot,relation in enumerate(relations):
        o.contact(CONTACT_SCENE_LINK,(leaves[2*slot],leaves[2*slot+1],relation),
                  source+20+slot,True,True)
        pairs.append(o.next_scene-1)
    o.contact(CONTACT_SCENE_LINK,(pairs[0],pairs[1],ROLE_LEFT),source+30,True,True)
    left=o.next_scene-1
    o.contact(CONTACT_SCENE_LINK,(pairs[2],pairs[3],ROLE_RIGHT),source+31,True,True)
    right=o.next_scene-1
    o.contact(CONTACT_SCENE_LINK,(left,right,ROLE_ROOT),source+32,True,True)
    return o._scene_by_id[o.next_scene-1]


def demonstrate_role(o,atoms,relations,text,source):
    scene=role_scene(o,atoms,relations,source)
    o.contact(CONTACT_SURFACE,u(text),source,True,True)
    return scene


def world_population(o,groups,source):
    """Expose current distinct entities sharing opaque learned feature matter."""
    current=[]
    for feature,count in groups:
        # Two independently sourced identities establish that this is shared
        # feature matter rather than one entity's self-feature.  `count` selects
        # which of those actual identities are present now; it is not sent as a
        # language condition.
        identities=(0x700000+int(feature)*4,0x700001+int(feature)*4)
        for offset,identity in enumerate(identities):
            o.contact(CONTACT_ENTITY_FEATURES,(identity,1,int(feature)),
                      source+10*offset+int(feature)%7,True,True)
        current.extend(identities[:int(count)])
    o.contact(CONTACT_WORLD_STATE,tuple(current),source+100,True,True)
    return tuple(current)


def demonstrate_agreement_role(o,atoms,relations,text,source,groups=()):
    scene=role_scene(o,atoms,relations,source)
    if groups:world_population(o,groups,source+50)
    o.contact(CONTACT_SURFACE,u(text),source,True,True)
    return scene


def teach_world_form(o,feature,count,text,source):
    singleton(o,feature,source)
    world_population(o,((feature,count),),source+20)
    o.contact(CONTACT_SURFACE,u(text),source,True,True)


def structural_context(scene):
    return int(organism._digest('relation-surface-context-v1',
        [int(scene.context),int(scene.binding_identity)])[:15],16) or 1


def demonstrate(o,atoms,text,source,left_relation=LEFT,conditioned=False):
    if conditioned:
        o.contact(CONTACT_BODY_TARGET,(atoms[0],),source,True,True)
        o.contact(CONTACT_BODY_STATE,BODY,source,True,True)
    scene=relation_scene(o,atoms,source,left_relation)
    o.contact(CONTACT_SURFACE,u(text),source,True,True)
    return scene


def build():
    o=ReferenceOrganismV2(PopulationSpecV1(32768,2,4,42,8))
    for atom,text in WORDS.items():
        for source in (9101,9201):
            singleton(o,atom,source+atom);o.contact(CONTACT_SURFACE,u(text),source,True,True)
    for atoms,plain,marked,source in (
            (DONOR_A,'dak near the lum','daks near the lums',9101),
            (DONOR_B,'nif near the pek','nifs near the peks',9201)):
        demonstrate(o,atoms,plain,source)
        demonstrate(o,atoms,plain,source+1)
        demonstrate(o,atoms,marked,source,conditioned=True)
        demonstrate(o,atoms,marked,source+100,conditioned=True)
    demonstrate_nested(o,NEST_A,'can the dak that is near the lum test?',11101)
    demonstrate_nested(o,NEST_B,'will a nif which was beside the vok inspect?',11201)
    role_a=(ROLE_CONTROLLER,ROLE_OTHER[0],ROLE_DISTRACTOR,ROLE_OTHER[1])
    role_b=(ROLE_OTHER[0],ROLE_CONTROLLER,ROLE_DISTRACTOR,ROLE_OTHER[1])
    for source in (12101,12102):
        demonstrate_role(o,ROLE_A,role_a,'can the dak that is near the lum test now?',source)
    for source in (12201,12202):
        demonstrate_role(o,ROLE_B,role_b,'will a nif which was beside the vok inspect now?',source)
    agr_a=(AGR_CONTROLLER,AGR_OTHER[0],AGR_DISTRACTOR,AGR_OTHER[1])
    agr_b=(AGR_OTHER[0],AGR_CONTROLLER,AGR_DISTRACTOR,AGR_OTHER[1])
    for source in (13101,13102):
        demonstrate_agreement_role(o,AGR_A,agr_a,'the dak near the lum that is calm now',source)
    for source in (13201,13202):
        demonstrate_agreement_role(o,AGR_B,agr_b,'the nif beside the vok which is ready today',source)
    # Current nonlinguistic populations, not language labels, distinguish the
    # conditioned controller surface.  One independent conditioned contact per
    # topology must converge on the same canonical controller->auxiliary edge.
    demonstrate_agreement_role(o,AGR_A,agr_a,'the daks near the lum that are calm now',
                               13301,((AGR_A[0],2),))
    demonstrate_agreement_role(o,AGR_B,agr_b,'the nifs beside the vok which are ready today',
                               13401,((AGR_B[2],2),))
    # Held-out words may be learned independently; the held-out clause/tree and
    # its controller choice are never demonstrated.
    teach_world_form(o,AGR_HELD[6],2,'wugs',13501)
    teach_world_form(o,AGR_HELD[6],2,'wugs',13502)
    teach_world_form(o,AGR_HELD[2],1,'blick',13601)
    teach_world_form(o,AGR_HELD[2],1,'blick',13602)
    return o


def stage(o,left_relation,source):
    o.contact(CONTACT_BODY_TARGET,(HELD[0],),source,True,True)
    o.contact(CONTACT_BODY_STATE,BODY,source,True,True)
    o.contact(CONTACT_PARTNER_CONTEXT,(1,7,PARTNER),source+1,True,True)
    scene=relation_scene(o,HELD,source+2,left_relation)
    return scene,o.tick()


def stage_nested(o,source,wrong_pair=False):
    o.contact(CONTACT_PARTNER_CONTEXT,(1,7,PARTNER),source,True,True)
    scene=nested_scene(o,NEST_HELD,source+1,wrong_pair)
    return scene,o.tick()


def stage_role_held(o,source):
    relations=(ROLE_OTHER[0],ROLE_DISTRACTOR,ROLE_OTHER[1],ROLE_CONTROLLER)
    o.contact(CONTACT_PARTNER_CONTEXT,(1,7,PARTNER),source,True,True)
    scene=role_scene(o,ROLE_HELD,relations,source+1)
    return scene,o.tick()


def prepare_agreement_held(o,source):
    relations=(AGR_OTHER[0],AGR_DISTRACTOR,AGR_OTHER[1],AGR_CONTROLLER)
    o.contact(CONTACT_PARTNER_CONTEXT,(1,7,PARTNER),source,True,True)
    scene=role_scene(o,AGR_HELD,relations,source+1)
    world_population(o,((AGR_HELD[6],2),(AGR_HELD[2],1)),source+50)
    return scene


def stage_agreement_held(o,source):
    scene=prepare_agreement_held(o,source)
    return scene,o.tick()


def main():
    started=time.perf_counter();checks={};o=build()
    trained=next(scene for scene in reversed(tuple(o._scene_by_id.values()))
                 if scene.atoms==DONOR_B and scene.demonstrated and scene.binding_identity)
    checkpoint=copy.deepcopy(o.checkpoint());context=structural_context(trained)
    checks['dependency_recipe_is_owned_by_relation_ancestry']=(
        o.language.dependency_supported(context,0,3)
        and not o.language.dependency_supported(META,0,3))

    nested_training_surfaces=(
        u('can the dak that is near the lum test?'),
        u('will a nif which was beside the vok inspect?'))
    nested_expected=u('might one wug that seems around the blick verify?')
    nested_nearest_shortcut=u('seems one wug that around the blick might verify?')
    checks['nested_complete_surface_and_atom_tuple_are_held_out']=(
        nested_expected not in nested_training_surfaces
        and not any(scene.atoms==NEST_HELD and scene.demonstrated
                    for scene in o._scene_by_id.values()))
    nested=ReferenceOrganismV2.restore(copy.deepcopy(checkpoint))
    nested_scene_row,nested_action=stage_nested(nested,9351)
    checks['nested_non_nearest_controller_drives_heldout_externalization']=(
        isinstance(nested_action,ActionV2)
        and nested_action.payload==nested_expected
        and nested_action.payload!=nested_nearest_shortcut
        and nested_action.binding_identity==nested_scene_row.binding_identity
        and tuple(nested_action.relation_occurrences)==tuple(nested_scene_row.relation_occurrences))
    nested_credit={} if not isinstance(nested_action,ActionV2) else nested.contact(
        CONTACT_CONSEQUENCE,(nested_action.ticket,1),nested_action.source,True,True)
    checks['nested_relation_tree_participates_before_return_credit']=(
        isinstance(nested_action,ActionV2)
        and all(oid in nested_action.contributors for oid in nested_scene_row.relation_occurrences)
        and nested_credit.get('credit',0)>0)
    nested_wrong=ReferenceOrganismV2.restore(copy.deepcopy(checkpoint))
    wrong_nested_scene,wrong_nested_action=stage_nested(nested_wrong,9361,True)
    checks['same_nested_atoms_with_changed_ancestry_refuse_linear_shortcut']=(
        wrong_nested_scene.atoms==nested_scene_row.atoms
        and wrong_nested_scene.binding_identity!=nested_scene_row.binding_identity
        and (not isinstance(wrong_nested_action,ActionV2)
             or wrong_nested_action.payload!=nested_expected))
    nested_cut=ReferenceOrganismV2.restore(copy.deepcopy(checkpoint))
    nested_cut.contact(CONTACT_WITHDRAW_SOURCE,(11101,),9371,True,True)
    _cut_scene,cut_action=stage_nested(nested_cut,9372)
    checks['one_nested_donor_withdrawal_drops_structure_choice']=(
        not isinstance(cut_action,ActionV2) or cut_action.payload!=nested_expected)
    demonstrate_nested(nested_cut,NEST_A,'can the dak that is near the lum test?',11301)
    _relearned_scene,relearned_action=stage_nested(nested_cut,9381)
    checks['independent_nested_reacquisition_restores_heldout_choice']=(
        isinstance(relearned_action,ActionV2) and relearned_action.payload==nested_expected)

    role_expected=u('might one wug that seems around the blick verify now?')
    role_nearest=u('seems one wug that around the blick verify now might?')
    role_held=ReferenceOrganismV2.restore(copy.deepcopy(checkpoint))
    role_scene_row,role_action=stage_role_held(role_held,9391)
    role_view=role_held._relation_role_view(role_scene_row)
    checks['local_controller_relation_survives_new_linear_position']=(
        isinstance(role_action,ActionV2) and role_action.payload==role_expected
        and role_action.payload!=role_nearest)
    exact_role_context=structural_context(role_scene_row)
    checks['role_factor_not_heldout_whole_tree_template']=(
        role_view is not None
        and role_held.language.role_context_supported(role_view[1],len(role_view[0]))
        and not role_held.language.template_candidates(exact_role_context,len(ROLE_HELD)))
    role_cut=ReferenceOrganismV2.restore(copy.deepcopy(checkpoint))
    role_cut.language._role_template_topologies.clear()
    _role_cut_scene,role_cut_action=stage_role_held(role_cut,9392)
    checks['local_relation_factor_lesion_blocks_moved_controller']=(
        not isinstance(role_cut_action,ActionV2) or role_cut_action.payload!=role_expected)
    role_source_cut=ReferenceOrganismV2.restore(copy.deepcopy(checkpoint))
    for source in (12101,12102):
        role_source_cut.contact(CONTACT_WITHDRAW_SOURCE,(source,),9393+source,True,True)
    _source_cut_scene,role_source_cut_action=stage_role_held(role_source_cut,9394)
    checks['one_topology_withdrawal_blocks_role_generalization']=(
        not isinstance(role_source_cut_action,ActionV2)
        or role_source_cut_action.payload!=role_expected)
    role_a=(ROLE_CONTROLLER,ROLE_OTHER[0],ROLE_DISTRACTOR,ROLE_OTHER[1])
    demonstrate_role(role_source_cut,ROLE_A,role_a,
                     'can the dak that is near the lum test now?',12301)
    _role_relearn_scene,role_relearn_action=stage_role_held(role_source_cut,9395)
    checks['new_independent_topology_contact_restores_role_factor']=(
        isinstance(role_relearn_action,ActionV2) and role_relearn_action.payload==role_expected)

    agreement_expected=u('some wugs around the blick that are steady soon')
    agreement_nearest=u('some wugs around the blick that is steady soon')
    agreement=ReferenceOrganismV2.restore(copy.deepcopy(checkpoint))
    agreement_scene,agreement_action=stage_agreement_held(agreement,9396)
    agreement_view=agreement._relation_role_view(agreement_scene)
    agreement_conditions=(() if agreement_view is None else tuple(
        surface_conditions(agreement,atom,BODY_STATE_TAG) for atom in agreement_view[0]))
    checks['current_world_population_derives_opaque_numerosity_conditions']=bool(
        agreement_view is not None and agreement_conditions
        and agreement_conditions[0] and agreement_conditions[2]
        and agreement_conditions[0]!=agreement_conditions[2])
    checks['role_controller_beats_nearer_numerosity_distractor']=(
        isinstance(agreement_action,ActionV2)
        and agreement_action.payload==agreement_expected
        and agreement_action.payload!=agreement_nearest)
    checks['agreement_relation_and_condition_forms_participate']=(
        isinstance(agreement_action,ActionV2)
        and all(oid in agreement_action.contributors
                for oid in agreement_scene.relation_occurrences)
        and sum(1 for row in agreement_action.selection_occurrences
                if row[0]==PREF_FORM)>=3)
    agreement_context=0 if agreement_view is None else int(agreement_view[1])
    exact_agreement_context=structural_context(agreement_scene)
    checks['agreement_dependency_is_role_factored_not_whole_tree']=(
        agreement.language.dependency_supported(agreement_context,0,1)
        and not agreement.language.dependency_supported(exact_agreement_context,6,7))
    agreement_cut=ReferenceOrganismV2.restore(copy.deepcopy(checkpoint))
    agreement_cut.language._dependency_sources.pop((agreement_context,0,1),None)
    agreement_cut.language._rebuild_indices()
    _agreement_cut_scene,agreement_cut_action=stage_agreement_held(agreement_cut,9398)
    checks['role_dependency_lesion_blocks_remote_agreement']=(
        not isinstance(agreement_cut_action,ActionV2)
        or agreement_cut_action.payload!=agreement_expected)
    world_cut=ReferenceOrganismV2.restore(copy.deepcopy(checkpoint))
    _world_cut_scene=prepare_agreement_held(world_cut,9399)
    world_cut.world_state_occurrence=0
    world_cut_action=world_cut.tick()
    checks['current_world_population_lesion_blocks_conditioned_agreement']=(
        not isinstance(world_cut_action,ActionV2)
        or world_cut_action.payload!=agreement_expected)
    agreement_source_cut=ReferenceOrganismV2.restore(copy.deepcopy(checkpoint))
    agreement_source_cut.contact(CONTACT_WITHDRAW_SOURCE,(13301,),9400,True,True)
    _source_cut_scene,agreement_source_cut_action=stage_agreement_held(
        agreement_source_cut,9401)
    checks['one_agreement_topology_withdrawal_blocks_role_dependency']=(
        not agreement_source_cut.language.dependency_supported(agreement_context,0,1)
        and (not isinstance(agreement_source_cut_action,ActionV2)
             or agreement_source_cut_action.payload!=agreement_expected))
    agreement_relearn=ReferenceOrganismV2.restore(copy.deepcopy(checkpoint))
    agreement_relearn.contact(CONTACT_WITHDRAW_SOURCE,(13301,),9402,True,True)
    demonstrate_agreement_role(
        agreement_relearn,AGR_A,
        (AGR_CONTROLLER,AGR_OTHER[0],AGR_DISTRACTOR,AGR_OTHER[1]),
        'the daks near the lum that are calm now',13701,((AGR_A[0],2),))
    _agreement_relearn_scene,agreement_relearn_action=stage_agreement_held(
        agreement_relearn,9403)
    checks['independent_agreement_topology_reacquisition_restores']=(
        agreement_relearn.language.dependency_supported(agreement_context,0,1)
        and isinstance(agreement_relearn_action,ActionV2)
        and agreement_relearn_action.payload==agreement_expected)
    agreement_probe=tuple(int(row[0]) if row else 0 for row in agreement_conditions)
    agreement_before=agreement.language.complete_dependencies(
        agreement_context,agreement_probe)
    agreement_touches_before=agreement.language.last_lookup_touches
    for index in range(512):
        decoy=70000+index
        agreement.language.observe_dependency(decoy,0,1,80000+index)
        agreement.language.observe_dependency(decoy,0,1,81000+index)
    agreement_after=agreement.language.complete_dependencies(
        agreement_context,agreement_probe)
    agreement_touches_after=agreement.language.last_lookup_touches
    checks['role_agreement_lookup_survives_512_decoy_networks']=(
        agreement_before==agreement_after and agreement_before is not None
        and agreement_touches_before==agreement_touches_after==1)

    held=ReferenceOrganismV2.restore(copy.deepcopy(checkpoint));scene,action=stage(held,LEFT,9401)
    condition=surface_conditions(held,HELD[0],BODY_STATE_TAG)
    completed=held.language.complete_dependencies(structural_context(scene),
        (condition[0] if condition else 0,0,0,0))
    checks['heldout_form_follows_matching_structural_closure']=(
        completed==(condition[0],0,0,condition[0]) if condition else False)
    checks['heldout_form_follows_matching_structural_closure']=(
        checks['heldout_form_follows_matching_structural_closure']
        and isinstance(action,ActionV2) and action.payload==u('wugs near the blicks')
        and action.binding_identity==scene.binding_identity
        and tuple(action.relation_occurrences)==tuple(scene.relation_occurrences))
    learned={} if not isinstance(action,ActionV2) else held.contact(
        CONTACT_CONSEQUENCE,(action.ticket,1),action.source,True,True)
    checks['relation_occurrences_and_forms_participate_before_credit']=(
        isinstance(action,ActionV2) and action.body_occurrence>0
        and all(oid in action.contributors for oid in scene.relation_occurrences)
        and sum(1 for row in action.selection_occurrences if row[0]==PREF_FORM)==2
        and learned.get('credit',0)>0 and learned.get('selection_network_updates',0)==1)

    distractor=ReferenceOrganismV2.restore(copy.deepcopy(checkpoint));wrong_scene,wrong=stage(distractor,WRONG_LEFT,9501)
    checks['identical_linear_atoms_do_not_cross_structural_scope']=(
        wrong_scene.atoms==scene.atoms and wrong_scene.context==scene.context
        and wrong_scene.binding_identity!=scene.binding_identity
        and (not isinstance(wrong,ActionV2) or wrong.payload!=u('wugs near the blicks')))
    lesion=ReferenceOrganismV2.restore(copy.deepcopy(checkpoint))
    lesion.language._dependency_sources.pop((context,0,3),None);lesion.language._rebuild_indices()
    _lesion_scene,lesion_action=stage(lesion,LEFT,9601)
    checks['structural_dependency_lesion_blocks_only_remote_form']=(
        not isinstance(lesion_action,ActionV2) or lesion_action.payload!=u('wugs near the blicks'))
    withdrawn=ReferenceOrganismV2.restore(copy.deepcopy(checkpoint))
    for source in (9101,9201):withdrawn.contact(CONTACT_WITHDRAW_SOURCE,(source,),9700+source,True,True)
    _withdrawn_scene,withdrawn_action=stage(withdrawn,LEFT,9701)
    checks['source_withdrawal_disables_structural_recipe']=(
        not withdrawn.language.dependency_supported(context,0,3)
        and (not isinstance(withdrawn_action,ActionV2) or withdrawn_action.payload!=u('wugs near the blicks')))

    restored=ReferenceOrganismV2.restore(copy.deepcopy(checkpoint))
    checks['checkpoint_rebuilds_structure_without_unfolded_state']=(
        restored.digest()==o.digest() and 'completed_dependencies' not in json.dumps(checkpoint))
    probe=(condition[0],0,0,0) if condition else (0,0,0,0)
    before=held.language.complete_dependencies(context,probe);touches_before=held.language.last_lookup_touches
    for index in range(512):
        decoy=50000+index
        held.language.observe_dependency(decoy,0,3,60000+index)
        held.language.observe_dependency(decoy,0,3,61000+index)
    after=held.language.complete_dependencies(context,probe);touches_after=held.language.last_lookup_touches
    checks['structural_lookup_survives_512_decoy_networks']=(
        before==after and before is not None and touches_before==touches_after==1)
    checks['both_fast_unions_run_structural_dependency_audit']=(
        'structural-dependency:reference_organism_structural_dependency_verify.py' in (Path(__file__).parent/'run_language_mastery_fast.sh').read_text()
        and 'reference_organism_structural_dependency_verify.py' in (Path(__file__).parent/'run_language_mastery_factory_fast.sh').read_text())

    result={'schema':'agi.reference-organism-structural-dependency.v4','pass':all(checks.values()),
        'checks':checks,'metrics':{'dependency_touches_before_after':[touches_before,touches_after],
        'agreement_dependency_touches_before_after':[
            agreement_touches_before,agreement_touches_after],
        'decoy_networks':512,'nested_relation_depth':3,'nested_distractor_slot':3,
        'nested_controller_slot':6,'role_controller_linear_positions':[1,3,7],
        'agreement_controller_linear_positions':[1,3,7],
        'role_factor_rows':len(o.language._role_template_topologies),
        'checkpoint_bytes':len(json.dumps(checkpoint,separators=(',',':')).encode())},
        'runtime_llm':False,'expected_output_ingress':False,'graph_flip':False,
        'human_language_mastery':False,'language_phenotype_improved':True,'direct_parity':False,
        'heldout_nested_surface':bytes(nested_expected).decode(),
        'heldout_role_surface':bytes(role_expected).decode(),
        'heldout_grounded_agreement_surface':bytes(agreement_expected).decode(),
        'claim':'GROUNDED_NUMEROSITY_ROLE_AGREEMENT_ON_CONTINUING_REFERENCE_ORGANISM_ONLY',
        'elapsed_ms':round((time.perf_counter()-started)*1000,3)}
    print('FOUNDRY_REFERENCE_ORGANISM_STRUCTURAL_DEPENDENCY '+('GREEN' if result['pass'] else 'RED')+
          f" checks={sum(checks.values())}/{len(checks)} decoys=512")
    print(json.dumps(result,indent=2,sort_keys=True));raise SystemExit(0 if result['pass'] else 1)


if __name__=='__main__':main()
