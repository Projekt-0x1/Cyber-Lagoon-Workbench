#!/usr/bin/env python3
"""Safe candidate frontier for adaptive experiment-policy mixtures.

V1 remains the owner of action-local procedural-reason calibration and intervention
settlement. V2 adds only a read/nomination surface over branches that already pass the same
present resource, controllability, risk and epistemic-gap vetoes. It does not choose a
policy and does not add evidence.
"""
from __future__ import annotations
from dataclasses import dataclass
from reference_recursive_causal_experiment_v1 import RecursiveCausalExperimentV1,Q,_clip

POLICY_DISCRIMINATE=1
POLICY_CONFIRM=2

@dataclass(frozen=True)
class SafeProbeCandidateV2:
    action:int
    discrimination_q16:int
    confirmation_q16:int
    risk_q16:int
    epistemic_gap_q16:int
    reason_count:int
    authority:int=0

class RecursiveCausalExperimentV2(RecursiveCausalExperimentV1):
    """Enumerates only currently safe probe candidates; world/reason evidence remains V1."""

    def eligible_probes(self,branches,reason_map,resource_q16,controllability_q16):
        resource=_clip(resource_q16);control=_clip(controllability_q16)
        if resource<Q//3 or control<Q//2:return ()
        branches=tuple(branches);rows=[]
        for branch in branches:
            action=int(branch.action)
            risk=_clip(branch.consequence_risk_q16);gap=_clip(branch.epistemic_gap_q16)
            if action<=0 or risk>Q//4 or gap<Q//3:continue
            predictions=tuple(reason_map.get(action,()))
            discrimination=self.information_gain_q16(branch,branches,predictions)
            # Positive testing is only a control priority over an already-safe action.
            # Strong supported/counterfactual branches with lower risk are cheaper to confirm.
            confirmation=_clip((2*_clip(branch.support_q16)+_clip(branch.counterfactual_q16)
                +(Q-risk))//4)
            rows.append(SafeProbeCandidateV2(action,discrimination,confirmation,risk,gap,
                min(8,len(predictions)),0))
        return tuple(sorted(rows,key=lambda row:row.action))

    def policy_candidate(self,candidates,policy):
        candidates=tuple(candidates);policy=int(policy)
        if not candidates:return None
        if policy==POLICY_DISCRIMINATE:
            return sorted(candidates,key=lambda r:(-r.discrimination_q16,r.risk_q16,r.action))[0]
        if policy==POLICY_CONFIRM:
            return sorted(candidates,key=lambda r:(-r.confirmation_q16,r.risk_q16,r.action))[0]
        return None

    def select_probe(self,branches,reason_map,resource_q16,controllability_q16):
        # Compatibility donor behavior remains discriminative. New policy-mixture integration
        # should call eligible_probes() and choose a policy explicitly.
        candidate=self.policy_candidate(self.eligible_probes(
            branches,reason_map,resource_q16,controllability_q16),POLICY_DISCRIMINATE)
        return 0 if candidate is None or candidate.discrimination_q16<Q//3 else int(candidate.action)

    @classmethod
    def restore(cls,data):
        prior=RecursiveCausalExperimentV1.restore(data);out=cls();out.__dict__.update(prior.__dict__);return out
