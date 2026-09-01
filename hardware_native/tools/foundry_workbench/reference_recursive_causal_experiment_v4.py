#!/usr/bin/env python3
"""Typed safe-probe frontier without aggregate discrimination/confirmation utilities.

V3 remains the checkpoint/evidence owner. V4 keeps resource/control/risk as hard gates and
represents epistemic/pragmatic probe properties separately. DISCRIMINATE and CONFIRM use
explicit lexicographic priorities over already-safe candidates; unlike categories are never
added into a common utility.
"""
from __future__ import annotations
from dataclasses import dataclass
from reference_recursive_causal_experiment_v3 import RecursiveCausalExperimentV3
from reference_recursive_causal_experiment_v2 import POLICY_DISCRIMINATE,POLICY_CONFIRM
from reference_recursive_causal_experiment_v1 import Q,_clip

@dataclass(frozen=True)
class SafeProbeCandidateV4:
    action:int
    recommendation_uncertainty_q16:int
    epistemic_gap_q16:int
    competitor_closeness_q16:int
    branch_support_q16:int
    simulation_support_q16:int
    risk_q16:int
    reason_count:int
    authority:int=0

class RecursiveCausalExperimentV4(RecursiveCausalExperimentV3):
    """Current safe-probe nomination surface; evidence semantics remain inherited."""

    def _recommendation_uncertainty_world_q16(self,action,recommendations,world_context=0):
        rows=[self.reason_reliability_q16(r.reason,r.source,int(action),int(world_context))
            for r in recommendations if int(r.source) not in self._withdrawn_sources]
        if not rows:return Q//2
        return sum(Q-abs(2*value-Q) for value in rows)//len(rows)

    @staticmethod
    def _competitor_closeness_q16(branch,branches):
        others=[row for row in branches if int(row.action)!=int(branch.action)]
        if not others:return 0
        # This is only closeness in the already-constructed program-support coordinate.
        distance=min(abs(_clip(branch.support_q16)-_clip(other.support_q16)) for other in others)
        return Q-_clip(distance)

    def eligible_probes(self,branches,recommendation_map,resource_q16,controllability_q16,regime=0):
        resource=_clip(resource_q16);control=_clip(controllability_q16)
        if resource<Q//3 or control<Q//2:return ()
        branches=tuple(branches);rows=[]
        for branch in branches:
            action=int(branch.action);risk=_clip(branch.consequence_risk_q16);gap=_clip(branch.epistemic_gap_q16)
            if action<=0 or risk>Q//4 or gap<Q//3:continue
            recommendations=tuple(recommendation_map.get(action,()))
            rows.append(SafeProbeCandidateV4(
                action,
                self._recommendation_uncertainty_world_q16(action,recommendations,int(regime)),
                gap,self._competitor_closeness_q16(branch,branches),
                _clip(branch.support_q16),_clip(branch.counterfactual_q16),risk,
                min(8,len(recommendations)),0))
        return tuple(sorted(rows,key=lambda row:row.action))

    def policy_candidate(self,candidates,policy):
        candidates=tuple(candidates);policy=int(policy)
        if not candidates:return None
        if policy==POLICY_DISCRIMINATE:
            # Epistemic priorities only, then pragmatic risk as a tie-break among already-safe probes.
            return sorted(candidates,key=lambda row:(
                -row.epistemic_gap_q16,
                -row.recommendation_uncertainty_q16,
                -row.competitor_closeness_q16,
                row.risk_q16,row.action))[0]
        if policy==POLICY_CONFIRM:
            # Confirm the strongest current control hypothesis; simulation is secondary.
            # Risk remains a non-compensatory tie-break because every row already passed the hard veto.
            return sorted(candidates,key=lambda row:(
                -row.branch_support_q16,
                -row.simulation_support_q16,
                row.risk_q16,
                row.epistemic_gap_q16,
                row.action))[0]
        return None

    def information_gain_q16(self,branch,all_branches,recommendations,regime=0):
        # Serialized compatibility field only. It now carries one pure epistemic coordinate
        # (current branch gap), never a sum of ambiguity + competition + recommendation state.
        return _clip(branch.epistemic_gap_q16)

    def select_probe(self,branches,recommendation_map,resource_q16,controllability_q16,regime=0):
        candidate=self.policy_candidate(self.eligible_probes(
            branches,recommendation_map,resource_q16,controllability_q16,int(regime)),POLICY_DISCRIMINATE)
        return 0 if candidate is None else int(candidate.action)

    @classmethod
    def restore(cls,data):
        old=RecursiveCausalExperimentV3.restore(data);out=cls();out.__dict__.update(old.__dict__);return out
