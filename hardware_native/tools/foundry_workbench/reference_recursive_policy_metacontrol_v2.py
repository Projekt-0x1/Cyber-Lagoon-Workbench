#!/usr/bin/env python3
"""Regime-conditioned self reliability over RecursivePolicyMetacontrolV1.

Global action history is retained only as compatibility/diagnostic summary. When a latent
regime is supplied, commitment scoring and reliability use `(regime, action)` evidence only.
Context-specific writes can be deferred until regime merge/split resolution.
"""
from __future__ import annotations
from reference_recursive_policy_metacontrol_v1 import (
    RecursivePolicyMetacontrolV1,MetacontrolDecisionV1,
    MODE_ACT,MODE_ASK,MODE_OBSERVE,MODE_DEFER,MODE_REVISE,Q,MAX_DECISIONS,_clip,
)

class RecursivePolicyMetacontrolV2(RecursivePolicyMetacontrolV1):
    def __init__(self):
        super().__init__();self._regime_success={};self._regime_failure={}

    @property
    def regime_outcome_count(self):return sum(self._regime_success.values())+sum(self._regime_failure.values())

    def regime_evidence_count(self,action,regime):
        key=(int(regime),int(action));return int(self._regime_success.get(key,0))+int(self._regime_failure.get(key,0))

    def reliability_q16(self,action,regime=0):
        action=int(action);regime=int(regime)
        if regime<=0:return super().reliability_q16(action)
        key=(regime,action);good=int(self._regime_success.get(key,0));bad=int(self._regime_failure.get(key,0))
        return ((good+1)*Q)//(good+bad+2)

    def record_regime_outcome(self,action,success,independent=True,regime=0):
        action=int(action);regime=int(regime)
        if action<=0 or regime<=0 or not independent:return False
        bucket=self._regime_success if bool(success) else self._regime_failure
        key=(regime,action);bucket[key]=int(bucket.get(key,0))+1;return True

    def record_outcome(self,action,success,independent=True,regime=0):
        action=int(action);regime=int(regime)
        if action<=0 or not independent:return False
        # Preserve global summary for compatibility. Regime-specific authority can be
        # written now or deferred by calling record_regime_outcome after split resolution.
        super().record_outcome(action,success,True)
        if regime>0:self.record_regime_outcome(action,success,True,regime)
        return True

    def _branch_score_regime(self,branch,regime):
        reliability=self.reliability_q16(branch.action,regime)
        evidence=(3*_clip(branch.support_q16)+2*_clip(branch.counterfactual_q16)
            +2*_clip(branch.source_quality_q16)+3*reliability)//10
        return max(0,evidence-_clip(branch.consequence_risk_q16)//4-_clip(branch.epistemic_gap_q16)//6)

    def choose(self,branches,resource_q16,controllability_q16,social_quality_q16,tick,
               previous_action=0,regime=0):
        self._advance(tick);resource=_clip(resource_q16);control=_clip(controllability_q16);social=_clip(social_quality_q16)
        branches=tuple(branch for branch in branches if int(branch.action)>0)
        if not branches:return {'mode':MODE_OBSERVE,'selected':0,'decision':0,'alternatives':()}
        scored=sorted(((self._branch_score_regime(branch,int(regime)),int(branch.action),branch)
            for branch in branches),key=lambda row:(-row[0],row[1]))
        top_score,top_action,top_branch=scored[0];second=scored[1][0] if len(scored)>1 else 0
        margin=max(0,top_score-second);uncertainty=_clip(Q-margin+_clip(top_branch.epistemic_gap_q16)//2)
        alternatives=tuple(row[1] for row in scored[:8])
        if resource<Q//5 or control<Q//6:mode=MODE_DEFER;selected=0
        elif len(scored)>1 and (margin<Q//12 or _clip(top_branch.epistemic_gap_q16)>Q//2):
            mode=MODE_ASK if social>=Q//2 else MODE_OBSERVE;selected=0
        elif top_score<9*Q//20:mode=MODE_OBSERVE;selected=0
        else:
            selected=top_action;mode=MODE_REVISE if int(previous_action)>0 and int(previous_action)!=top_action else MODE_ACT
        identity=self._next_decision;self._next_decision+=1
        row=MetacontrolDecisionV1(identity,mode,selected,alternatives,_clip(top_score),uncertainty,
            resource,control,social,int(tick))
        if len(self._decisions)>=MAX_DECISIONS:self._decisions.pop(0)
        self._decisions.append(row)
        return {'mode':mode,'selected':selected,'decision':identity,'alternatives':alternatives,
            'confidence_q16':row.confidence_q16,'uncertainty_q16':uncertainty,'regime':int(regime)}

    def checkpoint(self):
        data=super().checkpoint();data['v2_regime_self']={
            'success':[[r,a,v] for (r,a),v in sorted(self._regime_success.items())],
            'failure':[[r,a,v] for (r,a),v in sorted(self._regime_failure.items())]};return data

    @classmethod
    def restore(cls,data):
        prior=RecursivePolicyMetacontrolV1.restore(data);out=cls();out.__dict__.update(prior.__dict__)
        row=data.get('v2_regime_self',{});out._regime_success={(int(r),int(a)):int(v) for r,a,v in row.get('success',())}
        out._regime_failure={(int(r),int(a)):int(v) for r,a,v in row.get('failure',())};return out
