#!/usr/bin/env python3
"""R1: one sensory-minted individual coordinates reference and certified cause."""
from __future__ import annotations
import copy,inspect,json,time
from reference_crossmodal_concept_grounding_v1 import CrossmodalConceptGroundingV1
from reference_language_mastery_claude_gateway_v1 import body_source_identity
from reference_language_learning_v1 import PIECE_LITERAL,PIECE_PORT
from reference_life_function_curriculum_v1 import (
    C as CLAUSE,ReferenceLifeFunctionRuntimeV2,canonical_life_function_curriculum_v2,
    canonical_species_program_v2,
)
from reference_organism_v2 import ReferenceOrganismV2,CONTACT_ENTITY_FEATURES,CONTACT_WORLD_STATE
from reference_population_v1 import PopulationSpecV1
from reference_resident_relation_certificate_v1 import ResidentRelationCertificateV1
from reference_temporal_visual_continuity_v1 import TemporalVisualContinuityV1
from reference_unsupervised_perceptual_features_v1 import FEATURE_QUORUM
from reference_visual_joint_attention_naming_v1 import VisualJointAttentionNamingV1
from reference_visual_object_file_tracker_v1 import VisualObjectFileTrackerV1
from reference_visual_sensor_ingress_v1 import VisualSensorIngressV1
from reference_world_discourse_situation_bridge_v1 import WorldDiscourseSituationBridgeV1

TRACK_SOURCE=0xD001;WORLD_SOURCE=0xD002;KNOWN=0xD003
NORA=b'nora';LIAM=b'liam';MIRA=b'mira'
VIEW_A=((0,0,0),(0,255,0),(0,0,0));VIEW_B=((0,0,0),(0,192,64),(0,0,0))
FEATURE_A=0xD101;FEATURE_B=0xD102
RAW_ADJ=0xD201;RAW_TEST=0xD202;RAW_INSPECT=0xD203;RAW_SENSOR=0xD204;RAW_VALVE=0xD205

def _developed_adult():
    runtime=ReferenceLifeFunctionRuntimeV2(canonical_species_program_v2())
    runtime.run(canonical_life_function_curriculum_v2());adult=runtime.adult.language_adult
    factors=[]
    for separator in (b' ',b' because ',b' since '):
        rows=[]
        for (context,arity,pieces),sources in adult.language._span_sources.items():
            if int(arity)!=2 or len([s for s in sources if s not in adult.language._withdrawn])<2:continue
            ports=[i for i,piece in enumerate(pieces) if int(piece.kind)==PIECE_PORT]
            if len(ports)!=2:continue
            found=b''.join(bytes(piece.literal) for piece in pieces[ports[0]+1:ports[1]]
                           if int(piece.kind)==PIECE_LITERAL and piece.literal)
            if found==separator:rows.append(adult.language.span_factor_identity(context,arity,pieces))
        if rows:factors.append(min(map(int,rows)))
    if len(factors)<3:raise RuntimeError('participant-discourse:relation-factors')
    return runtime,adult,tuple(factors[:3])

def _train_temporal():
    temporal=TemporalVisualContinuityV1()
    for _ in range(FEATURE_QUORUM):
        temporal.observe_features((FEATURE_A,));temporal.observe_features((FEATURE_B,));temporal.gap()
    return temporal

def _discover(tracker,organism,sensor,temporal,start):
    first=tracker.observe(organism,sensor,temporal,TRACK_SOURCE,start,VIEW_A,FEATURE_A,
                          VisualSensorIngressV1.frame_digest(VIEW_A))[0]
    second,relation,same,_=tracker.observe(
        organism,sensor,temporal,TRACK_SOURCE,start+1,VIEW_B,FEATURE_B,
        VisualSensorIngressV1.frame_digest(VIEW_B))
    return first if first==second and same and relation else 0

def _name(adult,tracker,surface,left_source,right_source):
    VisualJointAttentionNamingV1.observe(adult,tracker,surface,left_source)
    VisualJointAttentionNamingV1.observe(adult,tracker,surface,right_source)

def _teach_participant(adult,tracker,organism,sensor,temporal,start,name,sources,atoms,surface):
    entity=_discover(tracker,organism,sensor,temporal,start);_name(adult,tracker,name,*sources)
    for source in (sources[0]+0x10,sources[1]+0x10):
        adult.observe_surface_construction(CLAUSE,atoms(entity),surface(name),source)
    return entity

def _pair(grounding,adult,organism,raw,concept,source):
    organism.contact(CONTACT_WORLD_STATE,(int(raw),),source,True,True)
    grounding.observe_world(organism)
    if grounding.observe_language_surface(adult,adult.language.lexeme(int(concept)))!=int(concept):
        raise RuntimeError('participant-discourse:lexical-pair')
    grounding.settle_current_pair(source,1,True)

def _ground(grounding,adult,organism,raw,concept,source):
    _pair(grounding,adult,organism,raw,concept,source)
    _pair(grounding,adult,organism,raw,concept,source+1)

def _intervene(adult,receipt,true_cause):
    binding=adult.world_causal_learning.bindings[int(receipt)]
    true_slot=binding.slots[binding.causes.index(int(true_cause))]
    for source in (0xD401,0xD402,0xD403):
        for _ in range(4):
            nomination,occurrence=adult.world_causal_learning.nominate_intervention(receipt)
            effect=1 if true_slot in set(map(int,nomination.coalition)) else 0
            adult.world_causal_learning.settle_intervention(nomination,occurrence,source,effect,True)

def main():
    started=time.perf_counter();checks={};temporal=_train_temporal()
    runtime,adult,factors=_developed_adult();organism=ReferenceOrganismV2(PopulationSpecV1(32768,2,4,42,8))
    mechanism_started=time.perf_counter()
    sensor=VisualSensorIngressV1();tracker=VisualObjectFileTrackerV1();grounding=CrossmodalConceptGroundingV1()

    nora=_teach_participant(adult,tracker,organism,sensor,temporal,1,NORA,(0xC201,0xC202),
                            lambda e:(e,201,301,401),lambda n:n+b' warms the greenhouse steadily.')
    liam=_teach_participant(adult,tracker,organism,sensor,temporal,4,LIAM,(0xC221,0xC222),
                            lambda e:(e,202,302,402),lambda n:n+b' dries the soil quickly.')
    mira=_discover(tracker,organism,sensor,temporal,7)
    before_name=adult.language.lexeme(mira);_name(adult,tracker,MIRA,0xC241,0xC242)
    named=adult.leaf(CLAUSE,(mira,201,301,401))
    checks['participant_is_sensory_minted_before_name_and_not_training_individual']=(
        mira>0 and mira not in (nora,liam) and before_name is None)
    checks['heldout_name_productively_fills_learned_argument_slot']=(
        bytes(named.surface)==b'mira warms the greenhouse steadily.')

    invariant=organism._active_entity_features(mira)[0]
    organism.contact(CONTACT_ENTITY_FEATURES,(KNOWN,1,invariant),0xD100,True,True)
    _ground(grounding,adult,organism,KNOWN,101,0xD110)
    for offset,(raw,concept) in enumerate(((RAW_ADJ,201),(RAW_TEST,301),(RAW_INSPECT,302),
                                           (RAW_SENSOR,401),(RAW_VALVE,402))):
        _ground(grounding,adult,organism,raw,concept,0xD120+offset*4)
    organism.contact(CONTACT_WORLD_STATE,
                     (RAW_ADJ,mira,RAW_TEST,RAW_INSPECT,RAW_SENSOR,RAW_VALVE),
                     WORLD_SOURCE,True,True)
    closure=grounding.resolve_current_world(adult,organism)
    _context,frontier=WorldDiscourseSituationBridgeV1.activate_frontier(adult,organism,grounding)
    named_rows=tuple(row for row in frontier if row.identity==named.identity)
    checks['one_sparse_world_closure_keeps_individual_and_grounded_category']=(mira in closure and 101 in closure)
    checks['current_world_recruits_named_participant_without_mira_sentence_contact']=(
        len(named_rows)==1 and named_rows[0].identity==named.identity)

    others=tuple(row for row in frontier if row.identity!=named.identity)
    if len(others)<2:raise RuntimeError('participant-discourse:frontier')
    rival,effect=others[0],others[-1];operator=int(factors[1])
    receipt=adult.world_causal_learning.participate((named.identity,rival.identity),effect.identity,64)
    _intervene(adult,receipt,named.identity)
    ecology=adult.world_causal_learning.ecology
    stale_after_participate=tuple(
        key for key in runtime.adult._settled_causal_action_lineage
        if int(key[1]) in adult.world_causal_learning.bindings
        and int(key[1]) not in ecology.pending)
    causal_grounding=adult.world_causal_learning.grounding
    causal_grounding.observe(operator,effect.identity,named.identity,
                              adult.world_causal_learning,receipt,0xD410,adult)
    causal_grounding.observe(operator,effect.identity,named.identity,
                              adult.world_causal_learning,receipt,0xD411,adult)
    program=causal_grounding.materialize(adult,adult.world_causal_learning,receipt,operator)
    if program is None:raise RuntimeError('participant-discourse:causal-program')
    certificate=ResidentRelationCertificateV1.causal(
        adult,receipt,program.identity,operator,causal_grounding)
    explanation=bytes(adult.public_surface(program.identity))
    checks['same_participant_enters_consequence_qualified_causal_relation']=(
        certificate.left==named.identity and certificate.right==effect.identity
        and certificate.source_blocks==3 and bytes(named.surface) in explanation)
    checks['public_connective_has_resident_relation_certificate']=(
        b' because ' in explanation and certificate.program==program.identity
        and b'. because ' not in explanation and b'..' not in explanation)
    checks['embedding_changes_only_transient_boundary_not_resident_child']=(
        bytes(named.surface).endswith(b'.') and explanation.endswith(b'.')
        and bytes(named.surface) in explanation)
    question=(b'why is it the case that '+bytes(effect.surface).rstrip(b'.?').lower()+b'?')
    runtime_checkpoint=copy.deepcopy(runtime.checkpoint())
    stored_bindings={int(row['receipt']) for row in
                     runtime_checkpoint['adult']['language_adult']['world_causal_learning']['bindings']}
    stored_lineages=tuple(runtime_checkpoint['adult']['settled_causal_action_lineage'])
    checks['expired_unresolved_lineage_is_reconciled_without_losing_resolved_relation']=(
        bool(stale_after_participate) and receipt in stored_bindings
        and all(int(row['causal_receipt']) in stored_bindings for row in stored_lineages))
    direct=ReferenceLifeFunctionRuntimeV2.restore(
        canonical_species_program_v2(),runtime_checkpoint)
    channel=body_source_identity('participant-direct-control')
    direct_surface,direct_action=direct.contact_utterance(question,channel,channel)
    direct_receipt=direct.adult.pending_causal_dialogue_actions.get(direct_action)
    direct_coordinates=(() if direct_receipt is None else
                        direct.adult._causal_action_coordinates(direct_receipt))
    checks['ordinary_body_contact_selects_participant_explanation']=(
        direct_action>0 and direct_coordinates
        and any(adult.leaf_equivalent(cause,named.identity)
                and adult.leaf_equivalent(row_effect,effect.identity)
                for _receipt,cause,row_effect in direct_coordinates)
        and bytes(named.surface).rstrip(b'.?!') in direct_surface
        and bytes(effect.surface).rstrip(b'.?!') in direct_surface
        and b'..' not in direct_surface and b'.,' not in direct_surface)

    checkpoint=copy.deepcopy(adult.checkpoint());restarted=type(adult).restore(checkpoint)
    category_only=adult.leaf(CLAUSE,(101,201,301,401))
    checks['category_substitution_cannot_impersonate_discovered_individual']=(
        category_only.identity!=named.identity and bytes(category_only.surface)!=bytes(named.surface))
    checks['restart_rematerializes_reference_and_explanation_without_stored_sentences']=(
        bytes(restarted.leaf(CLAUSE,(mira,201,301,401)).surface)==bytes(named.surface)
        and bytes(restarted.public_surface(program.identity))==explanation
        and bytes(named.surface).decode() not in json.dumps(checkpoint,sort_keys=True))
    source=(inspect.getsource(CrossmodalConceptGroundingV1.resolve_world_atoms)
            +inspect.getsource(ReferenceOrganismV2.mint_visual_entity)).lower()
    checks['identity_path_has_no_expected_answer_or_named_identity_authority']=(
        all(token not in source for token in ('expected','answer_key','engineer','mira')))
    checks['bounded_sparse_path']=(grounding.last_touches<16 and time.perf_counter()-mechanism_started<1.0)
    failed=[key for key,value in checks.items() if not value]
    result={
        'schema':'cyber-lagoon.reference-resident-visual-participant-discourse.v1',
        'contract':'FOUNDRY_RESIDENT_VISUAL_PARTICIPANT_DISCOURSE_'+('GREEN' if not failed else 'RED'),
        'pass':not failed,'reference_only':True,'runtime_llm':False,
        'language_phenotype_improved':not failed,'future_update_authority_preserved':not failed,
        'conversation':[bytes(named.surface).decode(),question.decode(),bytes(direct_surface).decode(),
                        bytes(restarted.leaf(CLAUSE,(mira,201,301,401)).surface).decode()],
        'resident_relation_surface':explanation.decode(),
        'natural_question':question.decode(),'direct_body_surface':bytes(direct_surface).decode(),
        'direct_relation_coordinates':[list(map(int,row)) for row in direct_coordinates],
        'stale_lineage_rows_reconciled':len(stale_after_participate),
        'participant':mira,'world_closure':closure,'relation_certificate':certificate.__dict__,
        'checks':checks,'failed':failed,
        'remaining_red':['CANONICAL_LIFE_CURRICULUM_VISUAL_CONTACT','MULTITURN_CLAUDE_PARTICIPANT_DIALOGUE',
                         'TEMPORAL_AND_CONTRAST_CERTIFICATES','DIRECT_PARITY'],
        'elapsed_ms':round((time.perf_counter()-started)*1000,3),
    }
    print(result['contract']);print(json.dumps(result,indent=2,sort_keys=True));return 0 if not failed else 1

if __name__=='__main__':raise SystemExit(main())
