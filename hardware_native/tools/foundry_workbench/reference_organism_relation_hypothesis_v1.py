#!/usr/bin/env python3
"""Transient structural nomination from independently witnessed raw pair episodes."""
from __future__ import annotations

RELATION_HYPOTHESIS_TAG=0x1E1A710A
PAIR_SOURCE_SUPPORT=2
PAIR_POSTING_LIMIT=64
PAIR_SCORE_ONE=1<<16
PAIR_CONTEXT_FLOOR=PAIR_SCORE_ONE
ENTITY_GEOMETRY_FLOOR=PAIR_SCORE_ONE//3
PAIR_COVERS=(((0,1),(2,3)),((0,2),(1,3)),((0,3),(1,2)))


def _overlap_score(left,right):
    left=set(left);right=set(right)
    return len(left&right)*PAIR_SCORE_ONE//max(1,len(left),len(right))


def recruit_entity_geometry(owner,entity):
    scores={};touches=0;features=owner._active_entity_features(int(entity));feature_set=set(features)
    for feature in features:
        bucket=owner._entity_feature_index.get(int(feature),())
        if len(bucket)>PAIR_POSTING_LIMIT:continue
        touches+=len(bucket);weight=PAIR_SCORE_ONE//max(1,len(bucket))
        for candidate in bucket:
            candidate=int(candidate);scores[candidate]=scores.get(candidate,0)+weight
    owner.last_entity_candidate_touches=touches
    return tuple(sorted(candidate for candidate,score in scores.items()
        if score>=ENTITY_GEOMETRY_FLOOR and sum(feature in feature_set for feature in owner._active_entity_features(candidate))*2>len(features)))


def index_pair_episode_features(owner,episode):
    identity=int(episode.identity)
    for key in owner._pair_feature_keys_by_episode.pop(identity,()):
        bucket=owner._pair_feature_episode_index.get(key)
        if bucket:
            bucket.discard(identity)
            if not bucket:owner._pair_feature_episode_index.pop(key,None)
    if len(episode.atoms)!=2 or episode.source in owner.withdrawn_sources:return
    left=owner._active_entity_features(int(episode.atoms[0]))
    right=owner._active_entity_features(int(episode.atoms[1]))
    keys=[]
    for left_feature in left:
        for right_feature in right:
            key=(int(left_feature),int(right_feature))
            bucket=owner._pair_feature_episode_index.setdefault(key,set())
            if identity in bucket or len(bucket)<PAIR_POSTING_LIMIT:
                bucket.add(identity);keys.append(key)
    if keys:owner._pair_feature_keys_by_episode[identity]=tuple(keys)


def reindex_entity_pair_features(owner,entity):
    identities=owner._pair_episode_by_atom.get(int(entity),())
    owner.last_relation_feature_reindex_touches=len(identities)
    for identity in tuple(identities):
        episode=owner._episode_by_id.get(int(identity))
        if episode is not None:index_pair_episode_features(owner,episode)


def _supported_pairs(owner,pair,posting_cache):
    exact=owner._pair_episode_index.get(tuple(pair),());owner.last_relation_hypothesis_touches+=len(exact)
    if exact:ids=exact
    else:
        ids=set();left=owner._active_entity_features(int(pair[0])) if int(pair[0]) in owner.entity_features else ()
        right=owner._active_entity_features(int(pair[1])) if int(pair[1]) in owner.entity_features else ()
        for left_feature in left:
            for right_feature in right:
                key=(int(left_feature),int(right_feature))
                if key not in posting_cache:
                    posting_cache[key]=owner._pair_feature_episode_index.get(key,())
                    owner.last_relation_hypothesis_touches+=len(posting_cache[key])
                ids.update(posting_cache[key])
    contexts={}
    for identity in ids:
        episode=owner._episode_by_id.get(int(identity))
        if episode is None or episode.source in owner.withdrawn_sources:continue
        if exact:
            if tuple(episode.atoms)!=tuple(pair):continue
            score=PAIR_SCORE_ONE
        else:
            left_score=_overlap_score(left,owner._active_entity_features(int(episode.atoms[0])))
            right_score=_overlap_score(right,owner._active_entity_features(int(episode.atoms[1])))
            score=min(left_score,right_score)
            if not score:continue
        scene=owner._scene_by_id.get(int(episode.scene_identity))
        if scene is None or not scene.population_occurrence:continue
        row=contexts.setdefault(int(episode.context),{}).setdefault(tuple(episode.atoms),[score,{}])
        row[0]=max(row[0],score);row[1][int(episode.source)]=int(scene.population_occurrence)
    supported=[]
    for context,exemplars in sorted(contexts.items()):
        if exact:
            sources=next(iter(exemplars.values()))[1]
            witnesses=tuple(value for _source,value in sorted(sources.items())[:PAIR_SOURCE_SUPPORT])
            support_score=len(witnesses)*PAIR_SCORE_ONE
        else:
            witnesses=[];used_sources=set();support_score=0
            ranked=sorted(exemplars.items(),key=lambda row:(-row[1][0],row[0]))
            for _atoms,(score,sources) in ranked:
                witness=next(((source,value) for source,value in sorted(sources.items()) if source not in used_sources),None)
                if witness is not None:used_sources.add(witness[0]);witnesses.append(witness[1]);support_score+=score
                if len(witnesses)>=PAIR_SOURCE_SUPPORT:break
            witnesses=tuple(witnesses)
        if len(witnesses)>=PAIR_SOURCE_SUPPORT and support_score>=PAIR_CONTEXT_FLOOR:
            supported.append((context,witnesses,support_score))
    return tuple(supported)


def recruit_relation_hypothesis(owner,scene,digest):
    owner.last_relation_hypothesis_touches=0
    if scene is None or len(scene.atoms)!=4 or scene.binding_identity or scene.relation_occurrences:return None
    atoms=tuple(int(x) for x in scene.atoms);candidates=[];posting_cache={}
    for left_slots,right_slots in PAIR_COVERS:
        left=tuple(atoms[index] for index in left_slots);right=tuple(atoms[index] for index in right_slots)
        for left_context,left_occurrences,left_score in _supported_pairs(owner,left,posting_cache):
            for right_context,right_occurrences,right_score in _supported_pairs(owner,right,posting_cache):
                candidates.append((min(left_score,right_score),left_slots,right_slots,left_context,right_context,left_occurrences,right_occurrences))
    if not candidates:return None
    best=max(row[0] for row in candidates);winners=[row for row in candidates if row[0]==best]
    if len(winners)!=1:
        preference=owner._selection_preference_context(scene);causal=[]
        for row in winners:
            _score,left_slots,right_slots,left_context,right_context,*_rest=row
            binding=int(digest('relation-binding-v3',[int(scene.context),*left_slots,left_context,*right_slots,right_context])[:15],16) or 1
            surface_context=int(digest('relation-surface-context-v1',[int(scene.context),binding])[:15],16) or 1
            template=owner.language.template(surface_context,len(atoms))
            evidence=0 if template is None else owner._selection_construction_evidence(
                preference,int(template.identity[:15],16),binding)
            causal.append((evidence,row))
        peak=max(score for score,_row in causal);winners=[row for score,row in causal if score==peak]
        if peak<=0 or len(winners)!=1:return None
    _score,left_slots,right_slots,left_context,right_context,left_occurrences,right_occurrences=winners[0]
    binding=int(digest('relation-binding-v3',[int(scene.context),*left_slots,left_context,*right_slots,right_context])[:15],16) or 1
    supports=tuple(dict.fromkeys((int(scene.population_occurrence),*left_occurrences,*right_occurrences)))
    occurrence=owner.population.recruit((RELATION_HYPOTHESIS_TAG,int(scene.population_occurrence),binding,left_context,right_context,*supports))
    relations=(*supports,int(occurrence.identity));scene.demonstrated=True
    hypothesis=type(scene)(owner.next_scene,int(scene.channel),int(scene.context),atoms,int(scene.source),
        int(occurrence.identity),False,False,0,binding,tuple(relations))
    owner.next_scene+=1;owner.current_scene=hypothesis;owner.pending_scenes.append(hypothesis);owner._index_scene(hypothesis)
    return hypothesis
