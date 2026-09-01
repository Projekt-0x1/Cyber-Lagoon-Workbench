#!/usr/bin/env python3
"""Reference-only contextual invariant consolidation mechanism.

This is a novel silicon synthesis grounded in functional constraints from episodic
specificity, statistical structure learning, contextual variability, consolidation,
and Sapolsky-style nested causation. It is not a model of named brain regions.

Exact lived episode/context evidence remains separable from the slower invariant
candidate. Repetition can strengthen evidence; only genuinely distinct contexts
increase diversity. Consolidation can strengthen/reorganize already-lived evidence
but cannot mint a new context/source or external authority.
"""
from __future__ import annotations
from dataclasses import dataclass,field
import hashlib,json


def _fold(tag:str,values)->int:
    body=json.dumps([tag,*map(int,values)],separators=(',',':')).encode()
    return int(hashlib.sha256(body).hexdigest()[:16],16)


@dataclass(frozen=True)
class ContextualEpisodeV1:
    identity:int
    relation_identity:int
    context_identity:int
    source_identity:int
    witness_identity:int
    consequence_q16:int=0
    controllable:int=0


@dataclass
class ContextualInvariantV1:
    relation_identity:int
    total_evidence:int=0
    consolidated_strength:int=0
    consequence_sum_q16:int=0
    controllable_uses:int=0
    context_counts:dict[int,int]=field(default_factory=dict)
    source_counts:dict[int,int]=field(default_factory=dict)
    episode_ids:list[int]=field(default_factory=list)
    lesioned:int=0

    @property
    def context_diversity(self)->int:return len(self.context_counts)
    @property
    def source_diversity(self)->int:return len(self.source_counts)
    @property
    def peak_context_repetition(self)->int:return max(self.context_counts.values(),default=0)
    @property
    def invariant_strength(self)->int:
        # Novel Workbench hypothesis: integrated evidence is amplified by diversity
        # and bounded consolidation, while a dominant single context remains a
        # strong episodic competitor. This equation itself is NOT biology authority.
        return self.context_diversity*(self.total_evidence+self.consolidated_strength)
    @property
    def episodic_competitor(self)->int:
        return self.peak_context_repetition*self.peak_context_repetition
    def reusable(self)->bool:
        return (not self.lesioned and self.context_diversity>=2 and
                self.invariant_strength>self.episodic_competitor)


class ContextualInvariantBankV1:
    def __init__(self):
        self.episodes:dict[int,ContextualEpisodeV1]={}
        self.invariants:dict[int,ContextualInvariantV1]={}
        self.consolidation_passes=0
        self.last_touched_relations=0

    def observe(self,relation_identity:int,context_identity:int,source_identity:int,
                witness_identity:int,consequence_q16:int=0,controllable:bool=False)->int:
        relation_identity=int(relation_identity);context_identity=int(context_identity)
        source_identity=int(source_identity);witness_identity=int(witness_identity)
        if not relation_identity or not context_identity or not source_identity or not witness_identity:
            raise ValueError('contextual_invariant:episode_identity')
        eid=_fold('contextual-episode-v1',(relation_identity,context_identity,source_identity,witness_identity,len(self.episodes)+1))
        if eid in self.episodes:raise ValueError('contextual_invariant:episode_collision')
        ep=ContextualEpisodeV1(eid,relation_identity,context_identity,source_identity,witness_identity,int(consequence_q16),int(bool(controllable)))
        self.episodes[eid]=ep
        row=self.invariants.setdefault(relation_identity,ContextualInvariantV1(relation_identity))
        row.total_evidence+=1;row.context_counts[context_identity]=row.context_counts.get(context_identity,0)+1
        row.source_counts[source_identity]=row.source_counts.get(source_identity,0)+1
        row.episode_ids.append(eid);row.consequence_sum_q16+=int(consequence_q16);row.controllable_uses+=int(bool(controllable));row.lesioned=0
        return eid

    def consolidate(self,plasticity_q16:int=65536,resource_q16:int=65536)->int:
        plasticity_q16=max(0,min(65536,int(plasticity_q16)));resource_q16=max(0,min(65536,int(resource_q16)))
        gain_scale=(plasticity_q16*resource_q16)//65536
        touched=0
        for row in self.invariants.values():
            if not row.episode_ids or row.lesioned:continue
            # Consolidation strengthens already-lived integration only. Diversity is
            # deliberately untouched, so replay cannot forge external corroboration.
            base=max(1,row.context_diversity)
            gain=(base*gain_scale)//65536
            if gain:
                row.consolidated_strength=min(1_000_000,row.consolidated_strength+gain);touched+=1
        self.consolidation_passes+=1;self.last_touched_relations=touched
        return touched

    def reusable(self,relation_identity:int)->bool:
        row=self.invariants.get(int(relation_identity));return bool(row and row.reusable())

    def lesion_episode(self,episode_identity:int)->bool:
        ep=self.episodes.pop(int(episode_identity),None)
        if ep is None:return False
        row=self.invariants.get(ep.relation_identity)
        if row is None:return False
        row.episode_ids=[x for x in row.episode_ids if x!=ep.identity]
        row.total_evidence=max(0,row.total_evidence-1)
        row.consequence_sum_q16-=ep.consequence_q16;row.controllable_uses=max(0,row.controllable_uses-ep.controllable)
        for table,key in ((row.context_counts,ep.context_identity),(row.source_counts,ep.source_identity)):
            n=table.get(key,0)-1
            if n>0:table[key]=n
            else:table.pop(key,None)
        return True

    def lesion_invariant(self,relation_identity:int)->bool:
        row=self.invariants.get(int(relation_identity))
        if row is None:return False
        row.lesioned=1;return True

    def repair_invariant(self,relation_identity:int)->bool:
        row=self.invariants.get(int(relation_identity))
        if row is None or not row.episode_ids:return False
        row.lesioned=0;row.consolidated_strength=0
        return True

    def checkpoint(self)->dict:
        return {'schema':1,'consolidation_passes':self.consolidation_passes,
          'episodes':[[e.identity,e.relation_identity,e.context_identity,e.source_identity,e.witness_identity,e.consequence_q16,e.controllable] for e in self.episodes.values()],
          'invariants':[[r.relation_identity,r.total_evidence,r.consolidated_strength,r.consequence_sum_q16,r.controllable_uses,sorted(r.context_counts.items()),sorted(r.source_counts.items()),list(r.episode_ids),r.lesioned] for r in self.invariants.values()]}

    @classmethod
    def restore(cls,d):
        if d.get('schema')!=1:raise ValueError('contextual_invariant:checkpoint')
        x=cls();x.consolidation_passes=int(d.get('consolidation_passes',0))
        for v in d.get('episodes',()):
            e=ContextualEpisodeV1(*map(int,v));x.episodes[e.identity]=e
        for v in d.get('invariants',()):
            rid,total,consolidated,consequence,controllable,contexts,sources,episodes,lesioned=v
            r=ContextualInvariantV1(int(rid),int(total),int(consolidated),int(consequence),int(controllable),dict((int(k),int(n)) for k,n in contexts),dict((int(k),int(n)) for k,n in sources),list(map(int,episodes)),int(lesioned));x.invariants[r.relation_identity]=r
        return x
