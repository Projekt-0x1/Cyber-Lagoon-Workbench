#!/usr/bin/env python3
"""Non-compensatory multiaxis affordance competition over metacontrol V2.

V2 checkpoint/self-evidence state is retained. V3 removes weighted branch-score authority.
Current typed branches preserve lived procedure support, recommendation strength, simulation,
source provenance, partner-access applicability, self competence, observed risk, risk ambiguity
and recommendation uncertainty as distinct coordinates.
"""
from __future__ import annotations
from dataclasses import dataclass
from reference_recursive_policy_metacontrol_v2 import RecursivePolicyMetacontrolV2
from reference_recursive_policy_metacontrol_v1 import (
    MetacontrolDecisionV1,MODE_ACT,MODE_ASK,MODE_OBSERVE,MODE_DEFER,MODE_REVISE,
    Q,MAX_DECISIONS,_clip,
)

MAX_FRONTIER=32
COMMIT_RISK_CEILING=Q//2
HIGH_UNCERTAINTY=Q//2
MIN_RESOURCE=Q//5
MIN_CONTROL=Q//6

@dataclass(frozen=True)
class ArbitrationAxesV3:
    action:int
    lived_program_support_q16:int
    recommendation_support_q16:int
    simulation_support_q16:int
    source_quality_q16:int
    access_applicability_q16:int
    self_reliability_q16:int
    consequence_risk_q16:int
    risk_evidence_count:int
    risk_ambiguity_q16:int
    recommendation_uncertainty_q16:int
    authority:int=0

    @property
    def max_uncertainty_q16(self):return max(int(self.risk_ambiguity_q16),int(self.recommendation_uncertainty_q16))

class RecursivePolicyMetacontrolV3(RecursivePolicyMetacontrolV2):
    """Action-mode arbitration without cross-category scalar compensation."""

    def axes(self,branch,control_context=0):
        typed=hasattr(branch,'lived_program_support_q16')
        lived=_clip(getattr(branch,'lived_program_support_q16',getattr(branch,'support_q16',0)))
        recommendation=_clip(getattr(branch,'recommendation_support_q16',Q//2 if typed else Q//2))
        simulation=_clip(getattr(branch,'simulation_support_q16',getattr(branch,'counterfactual_q16',Q//2)))
        source_quality=_clip(getattr(branch,'source_quality_q16',Q//2))
        access=_clip(getattr(branch,'access_applicability_q16',Q//2))
        risk=_clip(getattr(branch,'consequence_risk_q16',0))
        risk_count=max(0,int(getattr(branch,'risk_evidence_count',1 if not typed else 0)))
        risk_ambiguity=_clip(getattr(branch,'risk_ambiguity_q16',0 if not typed else Q//2))
        recommendation_uncertainty=_clip(getattr(branch,'recommendation_uncertainty_q16',getattr(branch,'epistemic_gap_q16',0)))
        return ArbitrationAxesV3(int(branch.action),lived,recommendation,simulation,source_quality,access,
            self.reliability_q16(branch.action,int(control_context)),risk,risk_count,risk_ambiguity,
            recommendation_uncertainty,0)

    @staticmethod
    def _dominates(left,right):
        # No cross-axis exchange rate exists. Dominance requires no worse on every coordinate.
        positive=(
            left.lived_program_support_q16>=right.lived_program_support_q16,
            left.recommendation_support_q16>=right.recommendation_support_q16,
            left.simulation_support_q16>=right.simulation_support_q16,
            left.source_quality_q16>=right.source_quality_q16,
            left.access_applicability_q16>=right.access_applicability_q16,
            left.self_reliability_q16>=right.self_reliability_q16,
        )
        negative=(
            left.consequence_risk_q16<=right.consequence_risk_q16,
            left.risk_ambiguity_q16<=right.risk_ambiguity_q16,
            left.recommendation_uncertainty_q16<=right.recommendation_uncertainty_q16,
        )
        if not all((*positive,*negative)):return False
        return any((
            left.lived_program_support_q16>right.lived_program_support_q16,
            left.recommendation_support_q16>right.recommendation_support_q16,
            left.simulation_support_q16>right.simulation_support_q16,
            left.source_quality_q16>right.source_quality_q16,
            left.access_applicability_q16>right.access_applicability_q16,
            left.self_reliability_q16>right.self_reliability_q16,
            left.consequence_risk_q16<right.consequence_risk_q16,
            left.risk_ambiguity_q16<right.risk_ambiguity_q16,
            left.recommendation_uncertainty_q16<right.recommendation_uncertainty_q16,
        ))

    def pareto_frontier(self,branches,control_context=0,risk_ceiling_q16=COMMIT_RISK_CEILING):
        ceiling=_clip(risk_ceiling_q16);axes=tuple(self.axes(branch,control_context) for branch in branches if int(branch.action)>0)
        admissible=tuple(row for row in axes if row.consequence_risk_q16<=ceiling)
        frontier=[]
        for row in admissible:
            if any(self._dominates(other,row) for other in admissible if other.action!=row.action):continue
            frontier.append(row)
        return tuple(sorted(frontier,key=lambda row:row.action)[:MAX_FRONTIER])

    def choose(self,branches,resource_q16,controllability_q16,social_quality_q16,tick,
               previous_action=0,regime=0):
        self._advance(tick);resource=_clip(resource_q16);control=_clip(controllability_q16);social=_clip(social_quality_q16)
        branches=tuple(branch for branch in branches if int(branch.action)>0)
        if not branches:return {'mode':MODE_OBSERVE,'selected':0,'decision':0,'alternatives':()}
        if resource<MIN_RESOURCE or control<MIN_CONTROL:
            mode=MODE_DEFER;selected=0;alternatives=tuple(sorted({int(branch.action) for branch in branches})[:MAX_FRONTIER])
            confidence=0;uncertainty=max((self.axes(branch,int(regime)).max_uncertainty_q16 for branch in branches),default=0)
        else:
            risk_ceiling=min(COMMIT_RISK_CEILING,max(Q//4,min(resource,control)))
            frontier=self.pareto_frontier(branches,int(regime),risk_ceiling);alternatives=tuple(row.action for row in frontier)
            if not frontier:
                mode=MODE_OBSERVE;selected=0;confidence=0
                uncertainty=max((self.axes(branch,int(regime)).max_uncertainty_q16 for branch in branches),default=0)
            elif len(frontier)>1:
                mode=MODE_ASK if social>=Q//2 else MODE_OBSERVE;selected=0
                # The compatibility confidence field denotes only own-action competence.
                confidence=max(row.self_reliability_q16 for row in frontier)
                uncertainty=max(row.max_uncertainty_q16 for row in frontier)
            else:
                winner=frontier[0];confidence=winner.self_reliability_q16;uncertainty=winner.max_uncertainty_q16
                if uncertainty>HIGH_UNCERTAINTY:
                    mode=MODE_ASK if social>=Q//2 else MODE_OBSERVE;selected=0
                else:
                    selected=winner.action;mode=MODE_REVISE if int(previous_action)>0 and int(previous_action)!=selected else MODE_ACT
        identity=self._next_decision;self._next_decision+=1
        row=MetacontrolDecisionV1(identity,mode,selected,alternatives,_clip(confidence),_clip(uncertainty),resource,control,social,int(tick))
        if len(self._decisions)>=MAX_DECISIONS:self._decisions.pop(0)
        self._decisions.append(row)
        return {'mode':mode,'selected':selected,'decision':identity,'alternatives':alternatives,
            'confidence_q16':row.confidence_q16,'uncertainty_q16':row.uncertainty_q16,
            'regime':int(regime),'pareto_frontier':alternatives}

    @classmethod
    def restore(cls,data):
        old=RecursivePolicyMetacontrolV2.restore(data);out=cls();out.__dict__.update(old.__dict__);return out
