#!/usr/bin/env python3
"""Embodied recursive metacontrol over zero-authority counterfactual branches.

This owner is not an executive and does not decide truth. It calibrates the organism's own
commitment reliability from independent lived consequences, then arbitrates whether current
resident evidence warrants ACT, ASK, OBSERVE, DEFER, or REVISE. Constructive futures and
social reasons may nominate branches but cannot increment calibration evidence.
"""
from __future__ import annotations
from dataclasses import dataclass

Q=1<<16
MAX_DECISIONS=2048
MODE_ACT=1;MODE_ASK=2;MODE_OBSERVE=3;MODE_DEFER=4;MODE_REVISE=5

def _clip(value,lo=0,hi=Q):return max(lo,min(hi,int(value)))

@dataclass(frozen=True)
class PolicyBranchV1:
    action:int;support_q16:int;counterfactual_q16:int;source_quality_q16:int
    consequence_risk_q16:int;epistemic_gap_q16:int;authority:int=0

@dataclass(frozen=True)
class MetacontrolDecisionV1:
    identity:int;mode:int;selected:int;alternatives:tuple[int,...]
    confidence_q16:int;uncertainty_q16:int;resource_q16:int;controllability_q16:int
    social_quality_q16:int;tick:int

class RecursivePolicyMetacontrolV1:
    """Resident self-calibration and mode arbitration; no semantic/world authority."""
    def __init__(self):
        self._success={};self._failure={};self._decisions=[];self._next_decision=1;self._tick=-1

    def _advance(self,tick):
        tick=int(tick)
        if tick<self._tick:raise ValueError('metacontrol:time-reversal')
        self._tick=tick

    @property
    def outcome_count(self):return sum(self._success.values())+sum(self._failure.values())
    @property
    def decision_count(self):return len(self._decisions)

    def reliability_q16(self,action):
        action=int(action);good=int(self._success.get(action,0));bad=int(self._failure.get(action,0))
        return ((good+1)*Q)//(good+bad+2)

    def record_outcome(self,action,success,independent=True):
        action=int(action)
        if action<=0 or not independent:return False
        bucket=self._success if bool(success) else self._failure
        bucket[action]=int(bucket.get(action,0))+1;return True

    def _branch_score(self,branch):
        reliability=self.reliability_q16(branch.action)
        evidence=(3*_clip(branch.support_q16)+2*_clip(branch.counterfactual_q16)
            +2*_clip(branch.source_quality_q16)+3*reliability)//10
        return max(0,evidence-_clip(branch.consequence_risk_q16)//4-_clip(branch.epistemic_gap_q16)//6)

    def choose(self,branches,resource_q16,controllability_q16,social_quality_q16,tick,previous_action=0):
        self._advance(tick);resource=_clip(resource_q16);control=_clip(controllability_q16);social=_clip(social_quality_q16)
        branches=tuple(branch for branch in branches if int(branch.action)>0)
        if not branches:return {'mode':MODE_OBSERVE,'selected':0,'decision':0,'alternatives':()}
        scored=sorted(((self._branch_score(branch),int(branch.action),branch) for branch in branches),key=lambda row:(-row[0],row[1]))
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
        row=MetacontrolDecisionV1(identity,mode,selected,alternatives,_clip(top_score),uncertainty,resource,control,social,int(tick))
        if len(self._decisions)>=MAX_DECISIONS:self._decisions.pop(0)
        self._decisions.append(row)
        return {'mode':mode,'selected':selected,'decision':identity,'alternatives':alternatives,
            'confidence_q16':row.confidence_q16,'uncertainty_q16':uncertainty}

    def decision(self,identity):return next((row for row in self._decisions if int(row.identity)==int(identity)),None)

    def checkpoint(self):
        return {'schema':1,'tick':self._tick,'next_decision':self._next_decision,
            'success':[[int(a),int(v)] for a,v in sorted(self._success.items())],
            'failure':[[int(a),int(v)] for a,v in sorted(self._failure.items())],
            'decisions':[{'identity':row.identity,'mode':row.mode,'selected':row.selected,'alternatives':list(row.alternatives),
                'confidence_q16':row.confidence_q16,'uncertainty_q16':row.uncertainty_q16,'resource_q16':row.resource_q16,
                'controllability_q16':row.controllability_q16,'social_quality_q16':row.social_quality_q16,'tick':row.tick}
                for row in self._decisions]}

    @classmethod
    def restore(cls,data):
        if int(data.get('schema',0))!=1:raise RuntimeError('metacontrol:checkpoint-schema')
        out=cls();out._tick=int(data.get('tick',-1));out._next_decision=int(data.get('next_decision',1))
        out._success={int(a):int(v) for a,v in data.get('success',())};out._failure={int(a):int(v) for a,v in data.get('failure',())}
        out._decisions=[MetacontrolDecisionV1(int(row['identity']),int(row['mode']),int(row['selected']),tuple(map(int,row.get('alternatives',()))),
            int(row['confidence_q16']),int(row['uncertainty_q16']),int(row['resource_q16']),int(row['controllability_q16']),
            int(row['social_quality_q16']),int(row['tick'])) for row in data.get('decisions',())]
        if len(out._decisions)>MAX_DECISIONS:raise RuntimeError('metacontrol:capacity')
        return out
