#!/usr/bin/env python3
"""Transfer raw-pair grouping through lived distributed entity features."""
from __future__ import annotations

import copy
import json
import time

from reference_organism_v2 import *
from reference_population_v1 import PopulationSpecV1

META=7811;LEFT=7812;RIGHT=7813;PARTNER=7814
D1=(101,102,103,104);D2=(201,202,203,204);HELD=(301,302,303,304);NOVEL=(401,402,403,404)
G1=(501,502,503,504);G2=(601,602,603,604);GHELD=(701,702,703,704)
ALL=(*D1,*D2,*HELD,*NOVEL,*G1,*G2,*GHELD);WORDS={atom:f'g{atom}' for atom in ALL}
ROLES=(9101,9102,9103,9104);HUB=9199
EN=(51001,51002);DE=(52001,52002)
EN_WORDS=('red','fox','greets','bird');DE_WORDS=('roter','Fuchs','gruesst','Vogel')
DE_ORDER=(0,1,3,2)


def u(text):return tuple(text.encode())
def surface(atoms):return ' '.join(WORDS[atom] for atom in atoms)


def set_features(o,atoms,base,roles=ROLES):
    sources=[]
    for index,(atom,role) in enumerate(zip(atoms,roles)):
        source=base+index;o.contact(CONTACT_ENTITY_FEATURES,(atom,3,role,100000+atom,HUB),source,True,True);sources.append(source)
    return tuple(sources)


def feature_rows(o,atoms,rows,base):
    for index,(atom,features) in enumerate(zip(atoms,rows)):
        o.contact(CONTACT_ENTITY_FEATURES,(atom,len(features),*features),base+index,True,True)


def tie_features(o,base):
    rows=((ROLES[0],100000+HELD[0],HUB),(ROLES[1],ROLES[2],100000+HELD[1],HUB),
          (ROLES[2],ROLES[1],100000+HELD[2],HUB),(ROLES[3],100000+HELD[3],HUB))
    feature_rows(o,HELD,rows,base)


def name(o,atom):
    for rank in range(2):
        source=8000+atom*4+rank;o.contact(CONTACT_SCENE,(7,1,1,atom),source,True,True)
        o.contact(CONTACT_SURFACE,u(WORDS[atom]),source,True,True)


def pair_history(o,atoms,base):
    for rank,(pair,context) in enumerate(((atoms[:2],LEFT),(atoms[2:],RIGHT))):
        for witness in range(2):
            source=base+rank*10+witness;o.contact(CONTACT_SCENE,(7,context,2,*pair),source,True,True)
            o.contact(CONTACT_SURFACE,u(surface(pair)),source,True,True)


def demonstrate(o,atoms,source):
    o.contact(CONTACT_SCENE,(7,META,4,*atoms),source,True,True)
    o.contact(CONTACT_SURFACE,u(surface(atoms)),source,True,True)


def multilingual_exemplar(o,atoms,words,sources,order=(0,1,2,3)):
    for atom,word in zip(atoms,words):
        for source in sources:
            o.contact(CONTACT_SCENE,(7,1,1,atom),source,True,True)
            o.contact(CONTACT_SURFACE,u(word),source,True,True)
    for pair,context in ((atoms[:2],LEFT),(atoms[2:],RIGHT)):
        phrase=' '.join(words[atoms.index(atom)] for atom in pair)
        for source in sources:
            o.contact(CONTACT_SCENE,(7,context,2,*pair),source,True,True)
            o.contact(CONTACT_SURFACE,u(phrase),source,True,True)
    phrase=' '.join(words[index] for index in order)
    for source in sources:
        o.contact(CONTACT_SCENE,(7,META,4,*atoms),source,True,True)
        o.contact(CONTACT_SURFACE,u(phrase),source,True,True)


def multilingual_relation_build(english=True,german=True):
    o=ReferenceOrganismV2(PopulationSpecV1(32768,2,4,47,8))
    for rank,atoms in enumerate((D1,D2,HELD)):
        rows=tuple((ROLES[slot],HUB,100000+atom) for slot,atom in enumerate(atoms))
        feature_rows(o,atoms,rows,93001+rank*100)
    if english:multilingual_exemplar(o,D1,EN_WORDS,EN)
    if german:multilingual_exemplar(o,D2,DE_WORDS,DE,DE_ORDER)
    return o


def build():
    o=ReferenceOrganismV2(PopulationSpecV1(32768,2,4,43,8))
    for rank,atoms in enumerate((D1,D2,HELD)):set_features(o,atoms,90001+rank*100)
    set_features(o,NOVEL,90301,(9201,9202,9203,9204))
    for atom in ALL:name(o,atom)
    for rank,atoms in enumerate((D1,D2)):
        pair_history(o,atoms,10001+rank*100);demonstrate(o,atoms,12001+rank)
    return o


def late_feature_build():
    o=ReferenceOrganismV2(PopulationSpecV1(32768,2,4,46,8))
    for atom in (*D1,*D2,*HELD):name(o,atom)
    for rank,atoms in enumerate((D1,D2)):
        pair_history(o,atoms,14001+rank*100);demonstrate(o,atoms,14201+rank)
    before=(len(o.episodes),len(o.actions),o._selection_revisions.row_count)
    for rank,atoms in enumerate((D1,D2,HELD)):set_features(o,atoms,92001+rank*100)
    return o,before


def graded_build():
    o=ReferenceOrganismV2(PopulationSpecV1(32768,2,4,45,8))
    rows=[]
    for exemplar,offset in ((G1,0),(G2,2)):
        features=tuple((9301+slot*10+offset,9302+slot*10+offset,HUB) for slot in range(4))
        feature_rows(o,exemplar,features,91001+offset*10);rows.append(features)
    held=tuple((rows[0][slot][0],rows[1][slot][0],HUB) for slot in range(4))
    feature_rows(o,GHELD,held,91101)
    for atom in (*G1,*G2,*GHELD):name(o,atom)
    for rank,atoms in enumerate((G1,G2)):
        pair_history(o,atoms,13001+rank*100);demonstrate(o,atoms,13201+rank)
    return o


def stage(o,atoms,base,partner=PARTNER):
    o.contact(CONTACT_PARTNER_CONTEXT,(1,7,partner),base,True,True)
    raw=o.contact(CONTACT_SCENE,(7,META,4,*atoms),base+1,True,True)
    return raw,o.tick()


def qualify_geometry(o,independent=True):
    actions=[]
    for rank,atoms in enumerate((D1,D2)):
        _raw,action=stage(o,atoms,18001+rank*100)
        assert isinstance(action,ActionV2)
        o.contact(CONTACT_CONSEQUENCE,(action.ticket,1),action.source,True,independent);actions.append(action)
    return tuple(actions)


def main():
    started=time.perf_counter();checks={};o=build();checkpoint=copy.deepcopy(o.checkpoint());checkpoint_text=json.dumps(checkpoint)
    checks['training_has_no_authored_scene_links']=not o.scene_links
    held=ReferenceOrganismV2.restore(copy.deepcopy(checkpoint));raw,action=stage(held,HELD,20001)
    checks['distributed_pair_geometry_transfers_to_novel_members']=(
        isinstance(action,ActionV2) and action.binding_identity>0 and raw!=action.scene_identity
        and action.payload==u(surface(HELD)) and len(action.relation_occurrences)>=6)
    learned={} if not isinstance(action,ActionV2) else held.contact(CONTACT_CONSEQUENCE,(action.ticket,1),action.source,True,True)
    checks['aligned_history_participates_before_independent_credit']=(
        isinstance(action,ActionV2) and all(x in action.contributors for x in action.relation_occurrences)
        and learned.get('credit',0)>0 and learned.get('selection_network_updates',0)==1)

    late,late_before=late_feature_build();late_checkpoint=copy.deepcopy(late.checkpoint())
    checks['late_feature_contact_does_not_create_episode_action_or_credit']=(
        late_before==(len(late.episodes),len(late.actions),late._selection_revisions.row_count))
    _raw,late_online_action=stage(late,HELD,20301)
    late_restored=ReferenceOrganismV2.restore(copy.deepcopy(late_checkpoint))
    _raw,late_restored_action=stage(late_restored,HELD,20401)
    checks['late_features_integrate_lived_episodes_online_like_restore']=(
        isinstance(late_online_action,ActionV2) and isinstance(late_restored_action,ActionV2)
        and late_online_action.binding_identity==late_restored_action.binding_identity
        and late_online_action.payload==late_restored_action.payload==u(surface(HELD)))
    replaced=ReferenceOrganismV2.restore(copy.deepcopy(late_checkpoint))
    incident=set(replaced._pair_episode_by_atom[D1[0]])
    old_keys={key for identity in incident for key in replaced._pair_feature_keys_by_episode.get(identity,())}
    replaced.contact(CONTACT_ENTITY_FEATURES,(D1[0],3,990001,990002,990003),92501,True,True)
    replacement_touches=replaced.last_relation_feature_reindex_touches
    stale=any(identity in replaced._pair_feature_episode_index.get(key,()) for key in old_keys for identity in incident)
    replacement_checkpoint=copy.deepcopy(replaced.checkpoint())
    _raw,replaced_action=stage(replaced,HELD,20501)
    replacement_restored=ReferenceOrganismV2.restore(copy.deepcopy(replacement_checkpoint))
    _raw,replacement_replay=stage(replacement_restored,HELD,20601)
    checks['feature_replacement_removes_stale_episode_access']=(
        replacement_touches==len(incident)==2 and not stale
        and replaced_action is None and replacement_replay is None)
    late_withdrawn=ReferenceOrganismV2.restore(copy.deepcopy(late_checkpoint))
    late_withdrawn.contact(CONTACT_WITHDRAW_SOURCE,(92001,),20701,True,True)
    _raw,late_withdrawn_action=stage(late_withdrawn,HELD,20801)
    checks['late_feature_source_withdrawal_prevents_transfer']=late_withdrawn_action is None

    multilingual=multilingual_relation_build();multilingual_checkpoint=copy.deepcopy(multilingual.checkpoint())
    _raw,english_relation=stage(multilingual,HELD,20851,EN[0])
    german_relation_o=ReferenceOrganismV2.restore(copy.deepcopy(multilingual_checkpoint))
    _raw,german_relation=stage(german_relation_o,HELD,20861,DE[0])
    german_surface=' '.join(DE_WORDS[index] for index in DE_ORDER)
    checks['cross_language_exemplars_share_relation_despite_divergent_surface_order']=(
        isinstance(english_relation,ActionV2) and isinstance(german_relation,ActionV2)
        and english_relation.binding_identity==german_relation.binding_identity
        and english_relation.template_identity!=german_relation.template_identity
        and english_relation.payload==u(' '.join(EN_WORDS))
        and german_relation.payload==u(german_surface))
    multilingual_credit={} if not isinstance(english_relation,ActionV2) else multilingual.contact(
        CONTACT_CONSEQUENCE,(english_relation.ticket,1),english_relation.source,True,True)
    checks['cross_language_relation_occurrences_participate_before_credit']=(
        isinstance(english_relation,ActionV2)
        and all(identity in english_relation.contributors for identity in english_relation.relation_occurrences)
        and multilingual_credit.get('credit',0)>0 and multilingual_credit.get('selection_network_updates',0)==1)
    multilingual_yoked=ReferenceOrganismV2.restore(copy.deepcopy(multilingual_checkpoint))
    _raw,multilingual_yoked_action=stage(multilingual_yoked,HELD,20866,EN[0])
    multilingual_yoked_credit={} if not isinstance(multilingual_yoked_action,ActionV2) else multilingual_yoked.contact(
        CONTACT_CONSEQUENCE,(multilingual_yoked_action.ticket,1),multilingual_yoked_action.source,True,False)
    checks['yoked_cross_language_return_cannot_write_credit']=(
        isinstance(multilingual_yoked_action,ActionV2) and multilingual_yoked_credit.get('credit',0)==0
        and multilingual_yoked_credit.get('selection_network_updates',0)==0)
    english_only=multilingual_relation_build(german=False)
    _raw,english_only_action=stage(english_only,HELD,20871,EN[0])
    german_only=multilingual_relation_build(english=False)
    _raw,german_only_action=stage(german_only,HELD,20881,DE[0])
    checks['one_language_one_exemplar_cannot_manufacture_transfer']=(
        english_only_action is None and german_only_action is None)
    neutral_relation_o=ReferenceOrganismV2.restore(copy.deepcopy(multilingual_checkpoint))
    _raw,neutral_relation=stage(neutral_relation_o,HELD,20891,99999)
    checks['unfamiliar_partner_preserves_multilingual_surface_ambiguity']=neutral_relation is None
    withdrawn_language=ReferenceOrganismV2.restore(copy.deepcopy(multilingual_checkpoint))
    for source in EN:withdrawn_language.contact(CONTACT_WITHDRAW_SOURCE,(source,),20901+source,True,True)
    _raw,withdrawn_language_action=stage(withdrawn_language,HELD,20911,DE[0])
    checks['withdrawing_one_language_exemplar_removes_shared_geometry']=withdrawn_language_action is None
    checks['multilingual_relation_has_no_router_or_translation_path']=all(not hasattr(multilingual,name) for name in
        ('language_id','language_router','english','german','translate'))
    multilingual_quantity=ReferenceOrganismV2.restore(copy.deepcopy(multilingual_checkpoint))
    for index in range(512):
        atom=700000+index;feature_source=710000+index*3
        multilingual_quantity.contact(CONTACT_ENTITY_FEATURES,(atom,2,HUB,720000+index),feature_source,True,True)
        for witness in range(2):
            source=feature_source+1+witness
            multilingual_quantity.contact(CONTACT_SCENE,(7,1,1,atom),source,True,True)
            multilingual_quantity.contact(CONTACT_SURFACE,u(f'noise-{index}'),source,True,True)
    _raw,quantity_multilingual_action=stage(multilingual_quantity,HELD,740001,EN[0])
    lexeme_feature_touches=multilingual_quantity.last_entity_candidate_touches
    checks['diagnostic_lexeme_geometry_ignores_512_common_hub_competitors']=(
        isinstance(quantity_multilingual_action,ActionV2)
        and quantity_multilingual_action.payload==u(' '.join(EN_WORDS))
        and 0<lexeme_feature_touches<=16 and lexeme_feature_touches*32<512)

    graded=graded_build();graded_checkpoint=copy.deepcopy(graded.checkpoint())
    graded_raw,graded_action=stage(graded,GHELD,20501)
    checks['graded_reliability_beats_weak_hub_alternatives']=(
        isinstance(graded_action,ActionV2) and graded_action.binding_identity>0
        and graded_raw!=graded_action.scene_identity and graded_action.payload==u(surface(GHELD)))
    graded_learned={} if not isinstance(graded_action,ActionV2) else graded.contact(
        CONTACT_CONSEQUENCE,(graded_action.ticket,1),graded_action.source,True,True)
    checks['graded_relation_occurrences_participate_before_credit']=(
        isinstance(graded_action,ActionV2)
        and all(x in graded_action.contributors for x in graded_action.relation_occurrences)
        and graded_learned.get('credit',0)>0 and graded_learned.get('selection_network_updates',0)==1)
    graded_restored=ReferenceOrganismV2.restore(copy.deepcopy(graded_checkpoint))
    _raw,graded_replay=stage(graded_restored,GHELD,20601)
    checks['graded_reliability_rebuilds_from_checkpoint']=(
        isinstance(graded_replay,ActionV2) and graded_replay.binding_identity==graded_action.binding_identity
        and 'pair_feature_episode_index' not in json.dumps(graded_checkpoint))

    causal=build();qualified=qualify_geometry(causal);causal_checkpoint=copy.deepcopy(causal.checkpoint())
    tie_features(causal,20701)
    _raw,causal_tie_action=stage(causal,HELD,20801)
    checks['independent_causal_history_resolves_equal_geometry']=(
        isinstance(causal_tie_action,ActionV2)
        and causal_tie_action.binding_identity==qualified[0].binding_identity)
    yoked_history=build();qualify_geometry(yoked_history,False)
    tie_features(yoked_history,20901)
    _raw,yoked_tie_action=stage(yoked_history,HELD,21001)
    checks['yoked_history_cannot_resolve_equal_geometry']=yoked_tie_action is None
    restored_history=ReferenceOrganismV2.restore(copy.deepcopy(causal_checkpoint))
    tie_features(restored_history,21101)
    _raw,restored_tie_action=stage(restored_history,HELD,21201)
    checks['causal_tie_preference_survives_checkpoint']=(
        isinstance(restored_tie_action,ActionV2)
        and restored_tie_action.binding_identity==qualified[0].binding_identity)
    countered_history=ReferenceOrganismV2.restore(copy.deepcopy(causal_checkpoint))
    _raw,countered_action=stage(countered_history,D1,21251)
    assert isinstance(countered_action,ActionV2)
    countered_history.contact(CONTACT_CONSEQUENCE,(countered_action.ticket,-1),countered_action.source,True,True)
    tie_features(countered_history,21261);_raw,countered_tie_action=stage(countered_history,HELD,21271)
    checks['independent_counterevidence_reopens_geometry_tie']=countered_tie_action is None
    withdrawn_history=ReferenceOrganismV2.restore(copy.deepcopy(causal_checkpoint))
    withdrawn_history.contact(CONTACT_WITHDRAW_SOURCE,(PARTNER,),21301,True,True)
    tie_features(withdrawn_history,21401)
    _raw,withdrawn_tie_action=stage(withdrawn_history,HELD,21501,PARTNER+1)
    checks['credit_source_withdrawal_reopens_geometry_tie']=withdrawn_tie_action is None
    lesioned_history=ReferenceOrganismV2.restore(copy.deepcopy(causal_checkpoint))
    lesioned_history._selection_construction_index.clear()
    tie_features(lesioned_history,21601)
    _raw,lesioned_tie_action=stage(lesioned_history,HELD,21701)
    checks['causal_selection_index_lesion_reopens_geometry_tie']=lesioned_tie_action is None

    missing=ReferenceOrganismV2.restore(copy.deepcopy(checkpoint));_raw,missing_action=stage(missing,NOVEL,21001)
    checks['unmatched_features_cannot_manufacture_grouping']=missing_action is None
    same_source=ReferenceOrganismV2(PopulationSpecV1(8192,2,4,44,8))
    for rank,atoms in enumerate((D1,D2,HELD)):set_features(same_source,atoms,21101+rank*10)
    for atoms in (D1,D2):
        for pair,context in ((atoms[:2],LEFT),(atoms[2:],RIGHT)):
            same_source.contact(CONTACT_SCENE,(7,context,2,*pair),21999,True,True)
    same_source.contact(CONTACT_SCENE,(7,META,4,*HELD),21999,True,True)
    checks['one_source_across_exemplars_cannot_supply_diversity']=(
        same_source._resident_relation_hypothesis(same_source.current_scene) is None)
    ambiguous=ReferenceOrganismV2.restore(copy.deepcopy(checkpoint))
    tie_features(ambiguous,22001)
    _raw,ambiguous_action=stage(ambiguous,HELD,22101)
    checks['two_feature_aligned_covers_preserve_ambiguity']=ambiguous_action is None
    withdrawn=ReferenceOrganismV2.restore(copy.deepcopy(checkpoint))
    withdrawn.contact(CONTACT_WITHDRAW_SOURCE,(90201,),23001,True,True)
    _raw,withdrawn_action=stage(withdrawn,HELD,23101)
    checks['feature_source_withdrawal_prevents_transfer']=withdrawn_action is None
    lesion=ReferenceOrganismV2.restore(copy.deepcopy(checkpoint))
    if hasattr(lesion,'_pair_feature_episode_index'):lesion._pair_feature_episode_index.clear()
    _raw,lesion_action=stage(lesion,HELD,24001)
    checks['feature_pair_index_lesion_prevents_transfer']=lesion_action is None

    restored=ReferenceOrganismV2.restore(copy.deepcopy(checkpoint));before=restored.digest();_raw,restored_action=stage(restored,HELD,25001)
    checks['checkpoint_rebuilds_geometry_without_persisted_index']=(
        before==ReferenceOrganismV2.restore(copy.deepcopy(checkpoint)).digest()
        and isinstance(restored_action,ActionV2) and all(name not in checkpoint_text for name in
            ('pair_episode_by_atom','pair_feature_episode_index','pair_feature_keys_by_episode')))
    baseline=ReferenceOrganismV2.restore(copy.deepcopy(checkpoint));_raw,baseline_action=stage(baseline,HELD,26001)
    baseline_touches=getattr(baseline,'last_relation_hypothesis_touches',0)
    quantity=ReferenceOrganismV2.restore(copy.deepcopy(checkpoint))
    for index in range(512):
        atoms=(10000+index*2,10001+index*2);source=30000+index*3
        set_features(quantity,atoms,source,(50000+index*2,50001+index*2))
        quantity.contact(CONTACT_SCENE,(7,60000+index,2,*atoms),source+2,True,True)
        quantity.contact(CONTACT_SURFACE,u('x'),source+2,True,True)
    focal=880001;partners=(880011,880012,880013,880014)
    for index,partner in enumerate(partners):
        source=890001+index*3
        quantity.contact(CONTACT_SCENE,(7,890101+index,2,focal,partner),source,True,True)
        quantity.contact(CONTACT_SURFACE,u('x'),source,True,True)
        quantity.contact(CONTACT_ENTITY_FEATURES,(partner,1,891001+index),source+1,True,True)
    quantity.contact(CONTACT_ENTITY_FEATURES,(focal,1,892001),892101,True,True)
    feature_reindex_touches=quantity.last_relation_feature_reindex_touches
    _raw,quantity_action=stage(quantity,HELD,70001);touches=getattr(quantity,'last_relation_hypothesis_touches',0)
    checks['geometry_lookup_stays_bounded_under_512_common_hub_pairs']=(
        isinstance(baseline_action,ActionV2) and isinstance(quantity_action,ActionV2)
        and 0<baseline_touches<=touches<=128 and touches*4<512)
    checks['late_feature_reindex_touches_only_incident_pairs']=(
        feature_reindex_touches==len(partners) and len(quantity.episodes)>512
        and feature_reindex_touches*100<len(quantity.episodes))

    result={'schema':'agi.reference-organism-relation-geometry.v1','pass':all(checks.values()),'checks':checks,
        'metrics':{'decoy_feature_pairs':512,'touches_before_after':[baseline_touches,touches],
            'late_feature_reindex_touches':feature_reindex_touches,'replacement_touches':replacement_touches,
            'lexeme_feature_touches':lexeme_feature_touches,
            'surface_port_orders':[[0,1,2,3],list(DE_ORDER)]},
        'runtime_llm':False,'authored_scene_links':False,'graph_flip':False,'human_language_mastery':False,
        'direct_parity':False,'claim':'DISTRIBUTED_RELATION_GEOMETRY_REFERENCE_ONLY',
        'elapsed_ms':round((time.perf_counter()-started)*1000,3)}
    print('FOUNDRY_REFERENCE_ORGANISM_RELATION_GEOMETRY '+('GREEN' if result['pass'] else 'RED')+
          f" checks={sum(checks.values())}/{len(checks)} decoys=512 touches={touches}")
    print(json.dumps(result,indent=2,sort_keys=True));raise SystemExit(0 if result['pass'] else 1)


if __name__=='__main__':main()
