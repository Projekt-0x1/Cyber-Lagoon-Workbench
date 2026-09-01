#!/usr/bin/env python3
"""Selective consolidation and source withdrawal over the developmental amplifier.

Consolidation is evidence compression, not truth creation. A transition may be compacted
only after source-diverse lived support remains unopposed and at least one organism-owned
experiment has been resolved by later authenticated contact. The compact row preserves the
exact contributing source set so later withdrawal or contradiction can reopen uncertainty.
"""
from __future__ import annotations
from dataclasses import dataclass

from reference_developmental_predictive_amplifier_v2 import (
    DevelopmentalPredictiveAmplifierV2,
    DevelopmentalExperimentV1,
)
from reference_developmental_predictive_amplifier_v1 import (
    MultimodalConceptV1,
    SurfaceFamilyV1,
    StructuredEpisodeV1,
    ReplayHypothesisV1,
    _id,
    Q,
    MAX_EPISODES,
)

MIN_CONSOLIDATION_SOURCES=3
MIN_INTERVENTION_CONFIRMATIONS=1


@dataclass(frozen=True)
class ConsolidatedTransitionV1:
    relation:int
    left:int
    right:int
    sources:tuple[int,...]
    first_tick:int
    last_tick:int


class DevelopmentalPredictiveAmplifierV3(DevelopmentalPredictiveAmplifierV2):
    """Future-sufficient invariant state plus still-revisable lived evidence."""

    def __init__(self):
        super().__init__()
        self._consolidated=[]
        self._intervention_confirmations={}
        self._withdrawn_sources=set()

    def _active_sources(self,sources):
        return tuple(sorted(int(s) for s in sources if int(s) not in self._withdrawn_sources))

    def withdraw_source(self,source):
        source=int(source)
        if source<=0:raise ValueError('amplifier-v3:withdraw-source')
        self._withdrawn_sources.add(source)
        self._examples=[x for x in self._examples if int(x['source'])!=source]
        self._surface_examples=[x for x in self._surface_examples if int(x['source'])!=source]
        self._episodes=[x for x in self._episodes if int(x.source)!=source]
        live_concepts=[];dead=set()
        for concept in self._concepts:
            sources=self._active_sources(concept.sources)
            if len(sources)<2:dead.add(int(concept.identity));continue
            live_concepts.append(MultimodalConceptV1(int(concept.identity),tuple(concept.feature_members),sources))
        self._concepts=live_concepts
        families=[]
        for family in self._surface_families:
            if int(family.concept_identity) in dead:continue
            sources=self._active_sources(family.sources)
            if len(sources)<2:continue
            families.append(SurfaceFamilyV1(int(family.concept_identity),tuple(family.members),sources))
        self._surface_families=families
        compact=[]
        for row in self._consolidated:
            sources=self._active_sources(row.sources)
            if not sources:continue
            compact.append(ConsolidatedTransitionV1(row.relation,row.left,row.right,sources,row.first_tick,row.last_tick))
        self._consolidated=compact
        for key,sources in tuple(self._intervention_confirmations.items()):
            remain={int(s) for s in sources if int(s)!=source}
            if remain:self._intervention_confirmations[key]=remain
            else:self._intervention_confirmations.pop(key,None)
        return source

    def observe_structured_episode(self,relations,anchors,source,tick,independent=True):
        self._advance(tick);relations=tuple(map(int,relations));anchors=tuple(map(int,anchors));source=int(source)
        if (not independent or source<=0 or source in self._withdrawn_sources or len(anchors)<2 or len(relations)!=len(anchors)-1):return False
        added=False
        for relation,left,right in zip(relations,anchors,anchors[1:]):
            if min(relation,left,right)<=0:continue
            existing=next((x for x in self._consolidated if x.relation==relation and x.left==left and x.right==right),None)
            if existing is not None:
                if source not in existing.sources:
                    idx=self._consolidated.index(existing)
                    self._consolidated[idx]=ConsolidatedTransitionV1(relation,left,right,tuple(sorted((*existing.sources,source))),existing.first_tick,int(tick));added=True
                continue
            row=StructuredEpisodeV1(relation,left,right,source,int(tick))
            if row not in self._episodes:
                if len(self._episodes)>=MAX_EPISODES:self._episodes.pop(0)
                self._episodes.append(row);added=True
        return added

    def confirm_intervention(self,relation,left,right,source,tick,independent=True):
        self._advance(tick);relation=int(relation);left=int(left);right=int(right);source=int(source)
        if (not independent or min(relation,left,right,source)<=0 or source in self._withdrawn_sources):return False
        exists=any(x.relation==relation and x.left==left and x.right==right and x.source==source for x in self._episodes)
        if not exists:exists=any(x.relation==relation and x.left==left and x.right==right and source in x.sources for x in self._consolidated)
        if not exists:return False
        self._intervention_confirmations.setdefault((relation,left,right),set()).add(source)
        return True

    def _edge_sources(self):
        grouped={}
        for row in self._consolidated:
            grouped.setdefault((int(row.relation),int(row.left),int(row.right)),set()).update(self._active_sources(row.sources))
        for row in self._episodes:
            if int(row.source) in self._withdrawn_sources:continue
            grouped.setdefault((int(row.relation),int(row.left),int(row.right)),set()).add(int(row.source))
        return grouped

    def consolidate(self,tick,min_sources=MIN_CONSOLIDATION_SOURCES,min_confirmations=MIN_INTERVENTION_CONFIRMATIONS):
        """Compact only intervention-surviving, source-diverse, unopposed lived edges."""
        self._advance(tick);min_sources=max(2,int(min_sources));min_confirmations=max(1,int(min_confirmations))
        grouped=self._edge_sources();by_context={}
        for (relation,left,right),sources in grouped.items():by_context.setdefault((relation,left),{})[right]=set(sources)
        promoted=0
        for (relation,left),rights in by_context.items():
            if len(rights)!=1:continue
            right,sources=next(iter(rights.items()))
            if len(sources)<min_sources:continue
            confirms=self._intervention_confirmations.get((relation,left,right),set())
            if len(self._active_sources(confirms))<min_confirmations:continue
            evidence=[x for x in self._episodes if x.relation==relation and x.left==left and x.right==right and x.source not in self._withdrawn_sources]
            prior=next((x for x in self._consolidated if x.relation==relation and x.left==left and x.right==right),None)
            ticks=[int(x.tick) for x in evidence]
            if prior is not None:ticks.extend((int(prior.first_tick),int(prior.last_tick)))
            first_tick=min(ticks) if ticks else int(tick);last_tick=max(ticks) if ticks else int(tick)
            compact=ConsolidatedTransitionV1(relation,left,right,tuple(sorted(sources)),first_tick,last_tick)
            if prior is None:self._consolidated.append(compact);promoted+=1
            else:self._consolidated[self._consolidated.index(prior)]=compact
            self._episodes=[x for x in self._episodes if not (x.relation==relation and x.left==left and x.right==right)]
        return promoted

    def predict_completion(self,relation,left):
        relation=int(relation);left=int(left);by_right={}
        for (rel,lft,right),sources in self._edge_sources().items():
            if rel==relation and lft==left and sources:by_right.setdefault(right,set()).update(sources)
        if not by_right:return {'status':0,'winner':0,'alternatives':(),'uncertainty':0}
        ranked=sorted(((len(srcs),right) for right,srcs in by_right.items()),reverse=True);top=ranked[0][0];winners=sorted(right for support,right in ranked if support==top);alternatives=tuple(sorted(by_right))
        if len(winners)!=1:return {'status':0,'winner':0,'alternatives':tuple(winners),'uncertainty':len(winners)}
        return {'status':1,'winner':winners[0],'alternatives':alternatives,'uncertainty':max(0,len(alternatives)-1)}

    def experiment_frontier(self,max_candidates=32):
        grouped={}
        for (relation,left,right),sources in self._edge_sources().items():
            if sources:grouped.setdefault((relation,left),{})[right]=set(sources)
        out=[]
        for (relation,left),by_right in grouped.items():
            if len(by_right)<2:continue
            supports=sorted((len(sources),right) for right,sources in by_right.items());total=sum(x[0] for x in supports);peak=max(x[0] for x in supports)
            ambiguity=(len(supports)*Q)//(len(supports)+1);balance=Q-((peak*Q)//max(1,total));discrimination=max(1,(ambiguity+balance)//2);alternatives=tuple(sorted(right for _support,right in supports))
            out.append(DevelopmentalExperimentV1(relation,left,alternatives,discrimination,total,0))
        out.sort(key=lambda x:(-x.discrimination_q16,-x.evidence,x.relation,x.left))
        return tuple(out[:max(0,int(max_candidates))])

    def replay_hypotheses(self,max_hypotheses=64):
        edges=[(rel,left,right,tuple(sorted(sources))) for (rel,left,right),sources in self._edge_sources().items() if sources];out=[];seen=set()
        for a in edges:
            for b in edges:
                if a[2]!=b[1] or not any(sa!=sb for sa in a[3] for sb in b[3]):continue
                key=((a[0],b[0]),(a[1],a[2],b[2]))
                if key in seen:continue
                seen.add(key);out.append(ReplayHypothesisV1(_id('replay-composition-v1',[list(key[0]),list(key[1])]),key[0],key[1],0))
                if len(out)>=int(max_hypotheses):return tuple(out)
        return tuple(out)

    @property
    def consolidated_count(self):return len(self._consolidated)
    @property
    def episode_count(self):return len(self._episodes)

    def checkpoint(self):
        data=super().checkpoint();data['v3_consolidation']={'withdrawn_sources':sorted(self._withdrawn_sources),'consolidated':[{'relation':x.relation,'left':x.left,'right':x.right,'sources':list(x.sources),'first_tick':x.first_tick,'last_tick':x.last_tick} for x in self._consolidated],'interventions':[{'relation':k[0],'left':k[1],'right':k[2],'sources':sorted(v)} for k,v in sorted(self._intervention_confirmations.items())]};return data

    @classmethod
    def restore(cls,data):
        prior=DevelopmentalPredictiveAmplifierV2.restore(data);out=cls();out.__dict__.update(prior.__dict__);ext=data.get('v3_consolidation',{});out._withdrawn_sources=set(map(int,ext.get('withdrawn_sources',())))
        out._consolidated=[ConsolidatedTransitionV1(int(x['relation']),int(x['left']),int(x['right']),tuple(map(int,x['sources'])),int(x['first_tick']),int(x['last_tick'])) for x in ext.get('consolidated',())];out._intervention_confirmations={}
        for x in ext.get('interventions',()):out._intervention_confirmations[(int(x['relation']),int(x['left']),int(x['right']))]=set(map(int,x.get('sources',())))
        return out
