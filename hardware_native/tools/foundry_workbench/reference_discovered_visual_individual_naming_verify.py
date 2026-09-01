#!/usr/bin/env python3
"""N+1: partner joint attention gives organism-discovered individuals productive learned names."""
from __future__ import annotations
import copy,inspect,json,time
from reference_continuous_visual_sensor_ownership_verify import A1,A2,train_level1,vf
from reference_global_discourse_relevance_verify import fresh
from reference_multi_visual_object_file_tracker_v1 import MultiVisualObjectFileTrackerV1
from reference_multiple_visual_object_files_verify import canvas as multi_canvas,train_temporal
from reference_organism_v2 import ReferenceOrganismV2
from reference_population_v1 import PopulationSpecV1
from reference_temporal_visual_continuity_v1 import TemporalVisualContinuityV1
from reference_visual_joint_attention_naming_v1 import VisualJointAttentionNamingV1
from reference_visual_object_file_tracker_v1 import VisualObjectFileTrackerV1
from reference_visual_sensor_ingress_v1 import VisualSensorIngressV1

SOURCE=0xC101;CLAUSE=103
NORA=b'nora';LIAM=b'liam';MIRA=b'mira'

def obs(t,o,s,temp,level1,seq,image):
    return t.observe(o,s,temp,SOURCE,seq,image,vf(level1,image),VisualSensorIngressV1.frame_digest(image))

def discover(t,o,s,temp,level1,start):
    e1,_r1,_same1,_=obs(t,o,s,temp,level1,start,A1)
    e2,r2,same2,_=obs(t,o,s,temp,level1,start+1,A2)
    if e1!=e2 or not same2:return 0
    return e1

def teach_name(adult,tracker,name,s1,s2):
    VisualJointAttentionNamingV1.observe(adult,tracker,name,s1)
    one=adult.language.lexeme(VisualJointAttentionNamingV1.visible_entities(tracker)[0])
    VisualJointAttentionNamingV1.observe(adult,tracker,name,s2)
    return one,adult.language.lexeme(VisualJointAttentionNamingV1.visible_entities(tracker)[0])

def main():
    started=time.perf_counter();checks={};level1=train_level1();temporal=train_temporal(level1);ta=temporal.relation(vf(level1,A1),vf(level1,A2))
    adult,host,*_=fresh();o=ReferenceOrganismV2(PopulationSpecV1(32768,2,4,42,8));sensor=VisualSensorIngressV1();tracker=VisualObjectFileTrackerV1()

    # Two independently discovered/nameable individuals provide developmental
    # subject-slot diversity. Their full clauses vary every other slot as well.
    nora=discover(tracker,o,sensor,temporal,level1,1);_one,nora_name=teach_name(adult,tracker,NORA,0xC201,0xC202)
    for src in (0xC211,0xC212):
        adult.observe_surface_construction(CLAUSE,(101,nora,301,401),b'the careful nora tests the sensor.',src)
    liam=discover(tracker,o,sensor,temporal,level1,4);_one2,liam_name=teach_name(adult,tracker,LIAM,0xC221,0xC222)
    for src in (0xC231,0xC232):
        adult.observe_surface_construction(CLAUSE,(102,liam,302,402),b'the quiet liam inspects the valve.',src)
    checks['two_training_individuals_are_organism_discovered_not_fixture_ids']=(nora>0 and liam>0 and nora!=liam and nora_name==tuple(NORA) and liam_name==tuple(LIAM))

    # Held-out third individual: no sentence containing MIRA is ever demonstrated.
    mira=discover(tracker,o,sensor,temporal,level1,7)
    checks['heldout_individual_exists_before_any_name_or_sentence_contact']=(mira>0 and mira not in (nora,liam) and adult.language.lexeme(mira) is None)
    VisualJointAttentionNamingV1.observe(adult,tracker,MIRA,0xC241)
    one=adult.language.lexeme(mira)
    VisualJointAttentionNamingV1.observe(adult,tracker,MIRA,0xC241)
    repeated=adult.language.lexeme(mira)
    VisualJointAttentionNamingV1.observe(adult,tracker,MIRA,0xC242)
    learned=adult.language.lexeme(mira)
    checks['one_source_or_repetition_cannot_install_heldout_individual_name']=(one is None and repeated is None)
    checks['second_independent_joint_attention_source_installs_heldout_name']=(learned==tuple(MIRA))

    named_leaf=adult.leaf(CLAUSE,(101,mira,301,401));named_surface=bytes(named_leaf.surface)
    checks['heldout_named_individual_productively_enters_new_clause_without_sentence_demo']=(
        named_surface==b'the careful mira tests the sensor.')

    # Same visual individual changes view and retains lexical identity.
    mira2,_r,same,_=obs(tracker,o,sensor,temporal,level1,9,A1)
    checks['individual_name_survives_view_change_on_same_visual_identity']=(mira2==mira and same and adult.language.lexeme(mira)==tuple(MIRA))

    # Gap starts another same-category individual; arbitrary name must not transfer.
    fresh_entity,_rf,_sf,_=obs(tracker,o,sensor,temporal,level1,11,A1)
    checks['new_same_category_individual_does_not_inherit_mira_name']=(fresh_entity!=mira and adult.language.lexeme(fresh_entity) is None)

    # Ambiguous joint attention refuses naming when two files are visible.
    mo=ReferenceOrganismV2(PopulationSpecV1(32768,2,4,42,8));ms=VisualSensorIngressV1();mt=MultiVisualObjectFileTrackerV1();mtemp=TemporalVisualContinuityV1.restore(copy.deepcopy(temporal.checkpoint()))
    frame=multi_canvas(((A1,1,1),(A1,1,14)));mt.observe(mo,ms,mtemp,level1,SOURCE,1,frame,VisualSensorIngressV1.frame_digest(frame))
    before=copy.deepcopy(adult.language.checkpoint());ambiguous=VisualJointAttentionNamingV1.observe(adult,mt,b'nova',0xC250)
    checks['multiple_visible_individuals_refuse_unresolved_joint_attention_name']=(ambiguous==0 and adult.language.checkpoint()==before)

    # Withdrawal affects lexical authority only; visual identity persists and a new
    # independent naming source can restore the name.
    adult.language.withdraw_source(0xC242);withdrawn=adult.language.lexeme(mira)
    # Return attention to MIRA requires a fresh contiguous MIRA occurrence; a new gap
    # would mint a different entity, so restore the visual tracker occurrence only for
    # this language-source lesion control.
    tracker.active_entity=mira
    VisualJointAttentionNamingV1.observe(adult,tracker,MIRA,0xC243);reacquired=adult.language.lexeme(mira)
    checks['name_source_withdrawal_and_reacquisition_do_not_erase_visual_individual']=(withdrawn is None and reacquired==tuple(MIRA) and o._active_entity_features(mira)==(ta,))

    checkpoint=adult.checkpoint();blob=json.dumps(checkpoint,sort_keys=True);source=inspect.getsource(VisualJointAttentionNamingV1).lower();params=inspect.signature(VisualJointAttentionNamingV1.observe).parameters
    checks['naming_bridge_receives_no_host_entity_argument_or_semantic_role']=(
        'entity' not in params and all(token not in source for token in ('category','concept','expected','sentence','engineer')))
    checks['checkpoint_has_name_relations_but_not_heldout_mira_sentence']=(named_surface.decode() not in blob)
    checks['bounded_fast_path']=time.perf_counter()-started<1.0
    failed=[k for k,v in checks.items() if not v]
    result={'schema':'cyber-lagoon.reference-discovered-visual-individual-naming.v2','contract':'FOUNDRY_DISCOVERED_VISUAL_INDIVIDUAL_NAMING_'+('GREEN' if not failed else 'RED'),'pass':not failed,'reference_only':True,'language_phenotype_improved':not failed,'future_update_authority_preserved':True,'visible_language_gain':'JOINT_ATTENTION_TEACHES_AN_ARBITRARY_NAME_TO_A_HELDOUT_ORGANISM_DISCOVERED_INDIVIDUAL_AND_PRODUCTIVE_GRAMMAR_USES_IT_IN_A_NEW_SENTENCE','entities':{'nora':nora,'liam':liam,'mira':mira,'fresh_same_category':fresh_entity},'named_surface':named_surface.decode(),'checks':checks,'failed':failed,'remaining_red':['NAME_USE_IN_PARTNER_QUERY_TO_SELECT_VISUAL_INDIVIDUAL','DEICTIC_THIS_THAT_GROUNDING','NAME_RETENTION_THROUGH_LONG_OCCLUSION_AND_REIDENTIFICATION','DIRECT_DISCOVERED_INDIVIDUAL_NAMING_PARITY'],'elapsed_ms':round((time.perf_counter()-started)*1000,3)}
    print(result['contract']);print(json.dumps(result,indent=2,sort_keys=True));return 0 if not failed else 1
if __name__=='__main__':raise SystemExit(main())
