#!/usr/bin/env python3
"""End-to-end developmental amplifier extensions over V1.

V1 proved the coupled mechanism with explicit combined contacts. V2 splits ordinary
perceptual common-cause admission from later raw-surface attachment so the real organism
contact stream can drive the same learning law without a bespoke training call.

Replay and experiment proposals remain zero-authority. Only authenticated independent
lived contacts may add concept or structured-episode evidence.
"""
from __future__ import annotations
from dataclasses import dataclass

from reference_developmental_predictive_amplifier_v1 import (
    DevelopmentalPredictiveAmplifierV1,
    ReplayHypothesisV1,
    _flat_features,
    _surface_sketch,
    MAX_EXAMPLES,
    MIN_COMMON_FEATURES,
    Q,
)


@dataclass(frozen=True)
class DevelopmentalExperimentV1:
    relation:int
    left:int
    alternatives:tuple[int,...]
    discrimination_q16:int
    evidence:int
    authority:int=0


class DevelopmentalPredictiveAmplifierV2(DevelopmentalPredictiveAmplifierV1):
    """Same resident state with ordinary perception/surface phases and experiment frontier."""

    def observe_percept(self,channels,source,tick,independent=True):
        self._advance(tick);source=int(source);features=_flat_features(channels)
        if source<=0:raise ValueError('amplifier-v2:percept-source')
        if len(features)<MIN_COMMON_FEATURES:return 0
        concept=self._settle_concept(features,source) if independent else 0
        if not concept:
            matches=self._matching_concepts(features)
            if len(matches)==1:concept=matches[0].identity
        row={'source':source,'tick':int(tick),'features':features}
        if not any(x['source']==source and x['features']==features for x in self._examples):
            if len(self._examples)>=MAX_EXAMPLES:self._examples.pop(0)
            self._examples.append(row)
        return int(concept)

    def observe_surface(self,raw,channels,source,tick,independent=True):
        """Attach raw surface only to a concept already supported by current perception."""
        self._advance(tick);source=int(source);raw=tuple(map(int,raw));features=_flat_features(channels)
        if source<=0 or not raw:raise ValueError('amplifier-v2:surface')
        matches=self._matching_concepts(features)
        if len(matches)!=1:return 0
        concept=int(matches[0].identity);sketch=_surface_sketch(raw)
        self._settle_surface(concept,sketch,source)
        row={'concept':concept,'source':source,'tick':int(tick),'sketch':sketch}
        if not any(x['concept']==concept and x['source']==source and x['sketch']==sketch for x in self._surface_examples):
            if len(self._surface_examples)>=MAX_EXAMPLES:self._surface_examples.pop(0)
            self._surface_examples.append(row)
        return concept

    def experiment_frontier(self,max_candidates=32):
        """Rank unresolved lived predictions by expected discrimination, never authority."""
        grouped={}
        for row in self._episodes:
            key=(int(row.relation),int(row.left));by_right=grouped.setdefault(key,{})
            by_right.setdefault(int(row.right),set()).add(int(row.source))
        out=[]
        for (relation,left),by_right in grouped.items():
            if len(by_right)<2:continue
            supports=sorted((len(sources),right) for right,sources in by_right.items())
            total=sum(x[0] for x in supports);peak=max(x[0] for x in supports)
            # Highest when alternatives are both numerous and similarly plausible.
            ambiguity=(len(supports)*Q)//(len(supports)+1)
            balance=Q-((peak*Q)//max(1,total))
            discrimination=max(1,(ambiguity+balance)//2)
            alternatives=tuple(sorted(right for _support,right in supports))
            out.append(DevelopmentalExperimentV1(relation,left,alternatives,discrimination,total,0))
        out.sort(key=lambda x:(-x.discrimination_q16,-x.evidence,x.relation,x.left))
        return tuple(out[:max(0,int(max_candidates))])

    def replay_experiment_frontier(self,max_candidates=32):
        """Expose replay compositions as hypotheses only; they never enter evidence here."""
        return tuple(self.replay_hypotheses(max_candidates))

    @classmethod
    def restore(cls,data):
        prior=DevelopmentalPredictiveAmplifierV1.restore(data)
        out=cls();out.__dict__.update(prior.__dict__);return out
