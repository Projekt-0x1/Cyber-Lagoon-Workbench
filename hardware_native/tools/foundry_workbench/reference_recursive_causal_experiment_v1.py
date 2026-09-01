#!/usr/bin/env python3
"""Low-risk causal experiment selection over zero-authority cultural hypotheses.

This owner does not learn world truth. The incumbent organism's transition ecology remains
world-learning authority. It calibrates only procedural prediction reliability for the exact
(reason, source, tested action) relation exposed by an independent lived intervention.
"""
from __future__ import annotations
from dataclasses import dataclass

Q=1<<16
MAX_INTERVENTIONS=2048


def _clip(value,lo=0,hi=Q):return max(lo,min(hi,int(value)))


@dataclass(frozen=True)
class ReasonPredictionV1:
    reason:int
    source:int
    predicts_success:bool=True
    authority:int=0


@dataclass(frozen=True)
class CausalInterventionV1:
    identity:int;action:int;decision:int;reasons:tuple[ReasonPredictionV1,...]
    alternatives:tuple[int,...];information_gain_q16:int;tick:int
    settled:bool=False;independent:bool=False;success:bool=False;authority:int=0


class RecursiveCausalExperimentV1:
    def __init__(self):
        self._reason_success={};self._reason_failure={};self._interventions=[]
        self._withdrawn_sources=set();self._next_intervention=1;self._tick=-1

    def _advance(self,tick):
        tick=int(tick)
        if tick<self._tick:raise ValueError('causal-experiment:time-reversal')
        self._tick=tick

    @property
    def reason_outcome_count(self):return sum(self._reason_success.values())+sum(self._reason_failure.values())
    @property
    def intervention_count(self):return len(self._interventions)

    def reason_evidence_count(self,reason,source,action):
        key=(int(reason),int(source),int(action))
        return int(self._reason_success.get(key,0))+int(self._reason_failure.get(key,0))

    def reason_reliability_q16(self,reason,source,action):
        key=(int(reason),int(source),int(action));good=int(self._reason_success.get(key,0));bad=int(self._reason_failure.get(key,0))
        return ((good+1)*Q)//(good+bad+2)

    def reason_calibration(self,action,predictions):
        deltas=[];evidence=0;action=int(action)
        for prediction in predictions:
            if int(prediction.source) in self._withdrawn_sources:continue
            count=self.reason_evidence_count(prediction.reason,prediction.source,action)
            if count<=0:continue
            reliability=self.reason_reliability_q16(prediction.reason,prediction.source,action)
            direction=1 if bool(prediction.predicts_success) else -1
            deltas.append(direction*(int(reliability)-Q//2));evidence+=count
        if not deltas:return 0,0
        return max(-Q//2,min(Q//2,sum(deltas)//len(deltas))),min(Q,evidence*(Q//4))

    def _uncertainty_q16(self,action,predictions):
        rows=[self.reason_reliability_q16(p.reason,p.source,action) for p in predictions
            if int(p.source) not in self._withdrawn_sources]
        if not rows:return Q//2
        return sum(Q-abs(2*value-Q) for value in rows)//len(rows)

    def information_gain_q16(self,branch,all_branches,predictions):
        competitors=[b for b in all_branches if int(b.action)!=int(branch.action)]
        ambiguity=_clip(branch.epistemic_gap_q16)
        competition=(Q-_clip(min(abs(int(branch.support_q16)-int(other.support_q16)) for other in competitors))) if competitors else 0
        return _clip((2*ambiguity+competition+self._uncertainty_q16(int(branch.action),predictions))//4)

    def select_probe(self,branches,reason_map,resource_q16,controllability_q16):
        resource=_clip(resource_q16);control=_clip(controllability_q16)
        if resource<Q//3 or control<Q//2:return 0
        branches=tuple(branches);candidates=[]
        for branch in branches:
            if int(branch.action)<=0 or _clip(branch.consequence_risk_q16)>Q//4 or _clip(branch.epistemic_gap_q16)<Q//3:continue
            predictions=tuple(reason_map.get(int(branch.action),()))
            candidates.append((self.information_gain_q16(branch,branches,predictions),-_clip(branch.consequence_risk_q16),int(branch.action)))
        if not candidates:return 0
        candidates.sort(reverse=True);return int(candidates[0][2]) if int(candidates[0][0])>=Q//3 else 0

    def begin(self,action,decision,predictions,alternatives,information_gain_q16,tick):
        self._advance(tick);action=int(action);decision=int(decision)
        predictions=tuple(p for p in predictions if int(p.reason)>0 and int(p.source)>0 and int(p.source) not in self._withdrawn_sources)
        if action<=0:return 0
        identity=self._next_intervention;self._next_intervention+=1
        row=CausalInterventionV1(identity,action,decision,predictions,tuple(sorted(set(map(int,alternatives)))),_clip(information_gain_q16),int(tick),False,False,False,0)
        if len(self._interventions)>=MAX_INTERVENTIONS:self._interventions.pop(0)
        self._interventions.append(row);return identity

    def settle(self,identity,success,independent,tick):
        self._advance(tick);row=next((x for x in self._interventions if int(x.identity)==int(identity)),None)
        if row is None or row.settled:return False
        independent=bool(independent);success=bool(success)
        revised=CausalInterventionV1(row.identity,row.action,row.decision,row.reasons,row.alternatives,row.information_gain_q16,row.tick,True,independent,success,0)
        self._interventions[self._interventions.index(row)]=revised
        if not independent:return False
        for prediction in row.reasons:
            if int(prediction.source) in self._withdrawn_sources:continue
            matched=(success==bool(prediction.predicts_success));bucket=self._reason_success if matched else self._reason_failure
            key=(int(prediction.reason),int(prediction.source),int(row.action));bucket[key]=int(bucket.get(key,0))+1
        return True

    def intervention(self,identity):return next((x for x in self._interventions if int(x.identity)==int(identity)),None)

    def withdraw_source(self,source):
        source=int(source)
        if source<=0:raise ValueError('causal-experiment:withdraw-source')
        self._withdrawn_sources.add(source);return source

    def checkpoint(self):
        def rows(values):return [[r,s,a,v] for (r,s,a),v in sorted(values.items())]
        return {'schema':2,'tick':self._tick,'next_intervention':self._next_intervention,'withdrawn_sources':sorted(self._withdrawn_sources),
            'reason_success':rows(self._reason_success),'reason_failure':rows(self._reason_failure),
            'interventions':[{'identity':x.identity,'action':x.action,'decision':x.decision,'reasons':[[p.reason,p.source,bool(p.predicts_success),0] for p in x.reasons],
                'alternatives':list(x.alternatives),'information_gain_q16':x.information_gain_q16,'tick':x.tick,'settled':x.settled,
                'independent':x.independent,'success':x.success,'authority':0} for x in self._interventions]}

    @classmethod
    def restore(cls,data):
        schema=int(data.get('schema',0))
        if schema not in (1,2):raise RuntimeError('causal-experiment:checkpoint-schema')
        out=cls();out._tick=int(data.get('tick',-1));out._next_intervention=int(data.get('next_intervention',1));out._withdrawn_sources=set(map(int,data.get('withdrawn_sources',())))
        if schema==2:
            out._reason_success={(int(r),int(s),int(a)):int(v) for r,s,a,v in data.get('reason_success',())}
            out._reason_failure={(int(r),int(s),int(a)):int(v) for r,s,a,v in data.get('reason_failure',())}
        else:
            # V1 lacked action scope; retain history only as non-operative archaeology.
            out._reason_success={};out._reason_failure={}
        out._interventions=[CausalInterventionV1(int(x['identity']),int(x['action']),int(x['decision']),
            tuple(ReasonPredictionV1(int(r),int(s),bool(pred),0) for r,s,pred,_authority in x.get('reasons',())),tuple(map(int,x.get('alternatives',()))),
            int(x['information_gain_q16']),int(x['tick']),bool(x.get('settled',False)),bool(x.get('independent',False)),bool(x.get('success',False)),0)
            for x in data.get('interventions',())]
        if len(out._interventions)>MAX_INTERVENTIONS:raise RuntimeError('causal-experiment:capacity')
        return out
