#!/usr/bin/env python3
"""Typed action-competition branch; unlike causal roles never share one support scalar."""
from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True)
class PolicyBranchV2:
    action:int
    lived_program_support_q16:int
    recommendation_support_q16:int
    simulation_support_q16:int
    source_quality_q16:int
    access_applicability_q16:int
    consequence_risk_q16:int
    risk_evidence_count:int
    risk_ambiguity_q16:int
    recommendation_uncertainty_q16:int
    authority:int=0

    # Narrow compatibility properties for older helper surfaces. Current metacontrol V3
    # reads the typed fields directly when present.
    @property
    def support_q16(self):return int(self.lived_program_support_q16)
    @property
    def counterfactual_q16(self):return int(self.simulation_support_q16)
    @property
    def epistemic_gap_q16(self):return max(int(self.risk_ambiguity_q16),int(self.recommendation_uncertainty_q16))
