#!/usr/bin/env python3
"""Bounded partner-specific memory of source-qualified joint-attention entity/event episodes."""
from __future__ import annotations
from dataclasses import dataclass
from reference_visual_deictic_reference_v1 import VisualDeicticReferenceV1
from reference_world_discourse_situation_bridge_v1 import WorldDiscourseSituationBridgeV1

MAX_JOINT_EPISODES=32
MIN_JOINT_SOURCES=2
MAX_EPISODE_SOURCES=8

@dataclass(frozen=True)
class JointAttentionEpisodeV1:
    channel:int
    marker:int
    entity:int
    event:int

class JointAttentionEpisodeMemoryV1:
    def __init__(self):self.support={}

    @staticmethod
    def candidate(adult,organism,tracker,grounding,raw_marker,marker_feature,
                  point_y2,point_x2,channel):
        channel=int(channel);marker_feature=int(marker_feature)
        if channel<=0:return None
        entity=VisualDeicticReferenceV1.resolve(
            adult,tracker,raw_marker,marker_feature,point_y2,point_x2)
        if not entity or entity not in tuple(getattr(organism,'world_state',()) or ()):return None
        frontier=WorldDiscourseSituationBridgeV1.frontier(adult,organism,grounding)
        if len(frontier)!=1:return None
        return JointAttentionEpisodeV1(channel,marker_feature,int(entity),int(frontier[0].identity))

    def observe(self,adult,organism,tracker,grounding,raw_marker,marker_feature,
                point_y2,point_x2,channel,source):
        source=int(source);episode=self.candidate(
            adult,organism,tracker,grounding,raw_marker,marker_feature,
            point_y2,point_x2,channel)
        if episode is None or source<=0:return None
        key=(episode.channel,episode.marker,episode.entity,episode.event)
        if key not in self.support and len(self.support)>=MAX_JOINT_EPISODES:return None
        rows=self.support.setdefault(key,set())
        if source not in rows and len(rows)>=MAX_EPISODE_SOURCES:return None
        rows.add(source);return episode

    @staticmethod
    def _contents(adult,raw):
        bindings=tuple(adult.language.invert_surface(tuple(raw)))
        contents={tuple(map(int,row.atoms)) for row in bindings if row.atoms}
        return next(iter(contents)) if len(contents)==1 else ()

    @staticmethod
    def _event_atoms(adult,event):
        event=int(event);tid=adult._surface_leaf_family_index.get(event)
        if tid is None:return ()
        lexemes=adult._surface_leaf_families.get(int(tid),{}).get(event)
        if lexemes is None:return ()
        atoms=[]
        for lexeme in lexemes:
            row=adult.language.historical_lexeme_binding(int(lexeme))
            if row is None:return ()
            atoms.append(int(row[0]))
        return tuple(atoms)

    def resolve(self,adult,organism,raw,channel):
        channel=int(channel);atoms=self._contents(adult,raw)
        if channel<=0 or not atoms:return None
        rows=[]
        for (row_channel,marker,entity,event),sources in self.support.items():
            if row_channel!=channel or marker not in atoms:continue
            active=sum(1 for source in sources if int(source) not in adult.language._withdrawn)
            if active<MIN_JOINT_SOURCES:continue
            if not organism._active_entity_features(int(entity)) or not adult._has_leaf(int(event)):continue
            rows.append(JointAttentionEpisodeV1(row_channel,marker,entity,event))
        if len(rows)==1:return rows[0]
        if len(rows)<2:return None
        scores=[]
        for row in rows:
            event_atoms=set(self._event_atoms(adult,row.event));cue=set(atoms)-{int(row.marker)}
            scores.append((len(event_atoms.intersection(cue)),row))
        best=max(score for score,_row in scores)
        winners=[row for score,row in scores if score==best]
        return winners[0] if best>0 and len(winners)==1 else None

    def checkpoint(self):
        return {'schema':1,'episodes':[{'channel':c,'marker':m,'entity':e,'event':v,'sources':sorted(srcs)}
            for (c,m,e,v),srcs in sorted(self.support.items())]}

    @classmethod
    def restore(cls,data):
        if int(data.get('schema',0))!=1:raise ValueError('joint_episode:checkpoint')
        out=cls()
        for row in data.get('episodes',()):
            key=(int(row.get('channel',0)),int(row.get('marker',0)),int(row.get('entity',0)),int(row.get('event',0)))
            sources=set(map(int,row.get('sources',())))
            if min(key)<=0 or key in out.support or not sources or min(sources)<=0 or len(sources)>MAX_EPISODE_SOURCES:raise ValueError('joint_episode:checkpoint')
            out.support[key]=sources
        if len(out.support)>MAX_JOINT_EPISODES:raise ValueError('joint_episode:capacity')
        return out
