#!/usr/bin/env python3
"""Novel entities inherit grounded concepts only through resident feature-family resemblance."""
from __future__ import annotations
import copy,inspect,json,time
from reference_crossmodal_concept_grounding_v1 import CrossmodalConceptGroundingV1
from reference_global_discourse_relevance_verify import fresh
from reference_organism_v2 import ReferenceOrganismV2,CONTACT_ENTITY_FEATURES,CONTACT_WORLD_STATE,CONTACT_WITHDRAW_SOURCE
from reference_population_v1 import PopulationSpecV1
from reference_predictive_credit_profile_v1 import Q
from reference_raw_sensory_shared_concept_grounding_verify import SURFACE,pair
from reference_world_discourse_situation_bridge_v1 import WorldDiscourseSituationBridgeV1

A=0xE101;B=0xE102;NOVEL_A=0xE201;WEAK=0xE202;AMBIG=0xE203
FA=(11,12,13,14);FB=(11,12,21,22);FNA=(11,12,13,15);FWEAK=(11,12,98,99);FAMB=(11,12,13,21)
FEATURE_SOURCE=0xEA01;WORLD_SOURCE=0xEA02
RAW_ADJ=0xEB01;RAW_TEST=0xEB02;RAW_INSPECT=0xEB03;RAW_SENSOR=0xEB04;RAW_VALVE=0xEB05
DIRECT=((RAW_ADJ,101),(RAW_TEST,301),(RAW_INSPECT,302),(RAW_SENSOR,401),(RAW_VALVE,402))


def features(o,entity,row,source=FEATURE_SOURCE):
    o.contact(CONTACT_ENTITY_FEATURES,(int(entity),len(row),*tuple(row)),int(source),True,True)


def world(o,atoms,source=WORLD_SOURCE):
    o.contact(CONTACT_WORLD_STATE,tuple(map(int,atoms)),int(source),True,True)


def ground_entity(g,adult,o,entity,concept,source):
    for n in range(2):pair(g,adult,o,entity,SURFACE[concept],source+n)


def train_direct(g,adult,o):
    for i,(raw,concept) in enumerate(DIRECT):
        for n in range(2):pair(g,adult,o,raw,SURFACE[concept],0xEC00+i*4+n)


def main():
    started=time.perf_counter();checks={}
    adult,host_frontier,_f,_ca,_cb=fresh();o=ReferenceOrganismV2(PopulationSpecV1(32768,2,4,42,8));g=CrossmodalConceptGroundingV1()

    # Two experienced exemplars have independently learned feature geometry and language grounding.
    features(o,A,FA);features(o,B,FB)
    ground_entity(g,adult,o,A,201,0xED10);ground_entity(g,adult,o,B,203,0xED20)
    checks['direct_exemplars_keep_distinct_grounded_concepts']=(g.resolve_raw_feature(adult,A)==201 and g.resolve_raw_feature(adult,B)==203)

    # Novel entities receive only perceptual features; they are never paired with language.
    features(o,NOVEL_A,FNA);features(o,WEAK,FWEAK);features(o,AMBIG,FAMB)
    checks['novel_family_resemblance_entity_inherits_a_without_language_pair']=(
        g.resolve_raw_feature(adult,NOVEL_A)==0 and g.resolve_world_atom(adult,o,NOVEL_A)==201)
    checks['below_overlap_threshold_novel_entity_stays_unresolved']=(g.resolve_world_atom(adult,o,WEAK)==0)
    checks['cross_category_family_resemblance_ambiguity_refuses']=(g.resolve_world_atom(adult,o,AMBIG)==0)
    overlaps=o._overlapping_entities(NOVEL_A);geometry_touches=o.last_entity_candidate_touches
    checks['category_transfer_uses_multi_feature_geometry']=(
        A in overlaps and B not in overlaps
        and geometry_touches<=len(FNA)*len(o.entity_features))

    # Feature-source withdrawal abolishes generalization, not direct exemplar grounding.
    withdrawn_o=ReferenceOrganismV2.restore(copy.deepcopy(o.checkpoint()));withdrawn_g=CrossmodalConceptGroundingV1.restore(copy.deepcopy(g.checkpoint()))
    withdrawn_o.contact(CONTACT_WITHDRAW_SOURCE,(FEATURE_SOURCE,),0xEE01,True,True)
    checks['feature_source_withdrawal_removes_generalization_not_direct_grounding']=(
        withdrawn_g.resolve_world_atom(adult,withdrawn_o,NOVEL_A)==0
        and withdrawn_g.resolve_raw_feature(adult,A)==201)

    # Focal cross-modal lesion of exemplar A removes transfer while preserving geometry and word.
    lesioned=CrossmodalConceptGroundingV1.restore(copy.deepcopy(g.checkpoint()));lesioned.lesion_raw_feature(A)
    checks['focal_exemplar_bridge_lesion_removes_novel_transfer_preserves_features_and_word']=(
        lesioned.resolve_world_atom(adult,o,NOVEL_A)==0
        and o._active_entity_features(NOVEL_A)==FNA
        and adult.language.lexeme(201)==tuple(SURFACE[201]))

    # Three-owner checkpoint retains geometry + grounding, no active category occurrence.
    cp_o=copy.deepcopy(o.checkpoint());cp_g=copy.deepcopy(g.checkpoint());ro=ReferenceOrganismV2.restore(cp_o);rg=CrossmodalConceptGroundingV1.restore(cp_g)
    checks['checkpoint_preserves_category_transfer_without_active_bridge_occurrence']=(
        rg.resolve_world_atom(adult,ro,NOVEL_A)==201 and not rg._current_world and rg._current_language is None)

    # Later unrelated exemplar learning does not alter the prior novel category judgment.
    interference=CrossmodalConceptGroundingV1.restore(cp_g);io=ReferenceOrganismV2.restore(cp_o)
    for i in range(12):
        entity=0xF000+i;concept=(202 if i%2==0 else 204);row=(100+i*4,101+i*4,102+i*4,103+i*4)
        features(io,entity,row,FEATURE_SOURCE+100+i);ground_entity(interference,adult,io,entity,concept,0xF500+i*4)
    checks['later_unrelated_category_learning_does_not_overwrite_novel_a']=(
        interference.resolve_world_atom(adult,io,NOVEL_A)==201)

    # Strong phenotype: a never-language-paired novel agent enters a world-derived discourse frontier.
    train_direct(g,adult,o)
    world(o,(RAW_ADJ,A,RAW_TEST,RAW_INSPECT,RAW_SENSOR,RAW_VALVE));ctx_a,frontier_a=WorldDiscourseSituationBridgeV1.activate_frontier(adult,o,g)
    for leaf in frontier_a:
        for _ in range(2):
            adult.experience_discourse_candidate(leaf.identity,Q,context=None,effort_q16=Q//16,duration=1)
            adult.experience_discourse_background(leaf.identity,False)
    world(o,(RAW_ADJ,NOVEL_A,RAW_TEST,RAW_INSPECT,RAW_SENSOR,RAW_VALVE));ctx_n,frontier_n=WorldDiscourseSituationBridgeV1.activate_frontier(adult,o,g)
    # Novel world has a different physical situation context, so give that situation ordinary relevance consequence using its mechanically recruited matter.
    for leaf in frontier_n:
        for _ in range(2):
            adult.experience_discourse_candidate(leaf.identity,Q,context=None,effort_q16=Q//16,duration=1)
            adult.experience_discourse_background(leaf.identity,False)
    world(o,(RAW_ADJ,NOVEL_A,RAW_TEST,RAW_INSPECT,RAW_SENSOR,RAW_VALVE));re_ctx,re_frontier=WorldDiscourseSituationBridgeV1.activate_frontier(adult,o,g)
    root=adult.organize_relevant_frontier(re_frontier);selected=tuple(adult.last_discourse_selected)
    expected_agent=tuple(x.identity for x in host_frontier if b'engineer' in bytes(x.surface) and b'careful' in bytes(x.surface))
    checks['novel_perceptual_entity_can_drive_long_form_grounded_discourse']=(
        ctx_n==re_ctx and len(re_frontier)==4 and root is not None and selected==tuple(x.identity for x in re_frontier)
        and set(selected)==set(expected_agent))
    checks['novel_entity_id_was_never_directly_grounded']=(g.resolve_raw_feature(adult,NOVEL_A)==0)

    source=inspect.getsource(CrossmodalConceptGroundingV1.resolve_world_atom)
    checks['no_persistent_category_table_or_category_label']=(
        all(token not in source for token in ('category_table','category_id','prototype_store','expected_category')))
    checks['bounded_reference_work']=time.perf_counter()-started<1.0

    failed=[k for k,v in checks.items() if not v]
    result={'contract':'FOUNDRY_OPEN_WORLD_PERCEPTUAL_CATEGORY_GROUNDING_GREEN','reference_only':True,'graph_flip':False,
            'novel_resolution':g.resolve_world_atom(adult,o,NOVEL_A),'frontier_count':len(re_frontier),'checks':checks,'failed':failed,
            'remaining_red':['RAW_PIXEL_AUDIO_FEATURE_EXTRACTION','UNSUPERVISED_FEATURE_DIMENSION_DISCOVERY','DIRECT_PERCEPTUAL_CATEGORY_PARITY'],
            'elapsed_ms':round((time.perf_counter()-started)*1000,3)}
    if failed:
        print('FOUNDRY_OPEN_WORLD_PERCEPTUAL_CATEGORY_GROUNDING_RED '+','.join(failed));print(json.dumps(result,indent=2,sort_keys=True));return 1
    print(result['contract']);print(json.dumps(result,indent=2,sort_keys=True));return 0

if __name__=='__main__':raise SystemExit(main())
