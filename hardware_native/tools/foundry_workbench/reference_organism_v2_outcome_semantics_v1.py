#!/usr/bin/env python3
"""Live Adult over non-compensatory control with typed cultural branch construction.

The prior category-safe control authority is preserved exactly in
`reference_organism_v2_noncompensatory_v1.py`. This layer removes the last upstream scalar
candidate collapse: resident culture is upgraded to V3 and emits typed branch coordinates
for lived procedure support, recommendation support, simulation, source quality, partner
shared-access applicability, known risk, risk ambiguity and recommendation uncertainty.
"""
from __future__ import annotations

import reference_organism_v2_noncompensatory_v1 as _prior
for _name in dir(_prior):
    if not _name.startswith('__') and _name not in {'_prior','_BaseReferenceOrganismV2'}:
        globals()[_name]=getattr(_prior,_name)

from reference_recursive_self_culture_control_v3 import RecursiveSelfCultureControlV3,Q as CULTQ
from reference_multiaxis_policy_branch_v2 import PolicyBranchV2

_BaseReferenceOrganismV2=_prior.ReferenceOrganismV2

class ReferenceOrganismV2(_BaseReferenceOrganismV2):
    def __init__(self,population_spec=None):
        super().__init__(population_spec)
        self.recursive_self_culture=RecursiveSelfCultureControlV3.restore(self.recursive_self_culture.checkpoint())

    def _metacontrol_program_branches(self):
        if not self.affordances:return (),(),0,0
        context=self._culture_context();resource,control=self._culture_body_envelope()
        rows=self.recursive_self_culture.program_candidates(
            context,resource,control,self.developmental_source_credibility_q16)
        if not rows:return (),(),resource,control
        futures=self.constructive_self_futures(256)
        world_context,control_context,world_cues,_control_cues,_r,_c=self._infer_contexts()
        recommendation_map=self._experiment_reason_map(rows)
        branches=[]
        for row in rows:
            if not row.get('actions'):continue
            action=int(row['actions'][0])
            if action not in self.affordances:continue
            relevant=[future for future in futures if future.actions and int(future.actions[0])==action]
            positive=sum(1 for future in relevant if future.predicted_outcomes and int(future.predicted_outcomes[0])==3)
            negative=sum(1 for future in relevant if future.predicted_outcomes and int(future.predicted_outcomes[0])==1)
            risk_count=positive+negative
            simulation_support=(CULTQ//2 if risk_count==0 else max(0,min(CULTQ,
                CULTQ//2+((positive-negative)*CULTQ)//(2*risk_count))))
            known_risk=0 if risk_count==0 else (negative*CULTQ)//risk_count
            risk_ambiguity=CULTQ//2 if risk_count==0 else CULTQ//(risk_count+2)
            recommendations=tuple(recommendation_map.get(action,()))
            recommendation_uncertainty=(0 if not recommendations else
                self.recursive_causal_experiment.recommendation_uncertainty_world_q16(
                    action,recommendations,world_context))
            access=CULTQ//2
            if self.partner_present and int(self.partner_source)>0:
                participant=int(self.partner_source)
                recommendation_sources={int(item.source) for item in recommendations}
                supporters=set(map(int,(*row.get('teachers',()),*row.get('confirmations',()))))
                if participant in recommendation_sources or participant in supporters:
                    access=self.recursive_partner_access.access_applicability_q16(
                        participant,world_cues,world_context)
            branches.append(PolicyBranchV2(
                action,
                int(row.get('lived_program_support_q16',0)),
                int(row.get('recommendation_support_q16',CULTQ//2)),
                int(simulation_support),
                int(row.get('source_quality_q16',CULTQ//2)),
                int(access),
                int(known_risk),int(risk_count),int(risk_ambiguity),
                int(recommendation_uncertainty),0))
        self._world_context_current=int(world_context);self._control_context_current=int(control_context)
        return tuple(branches),rows,resource,control

    @classmethod
    def restore(cls,data):
        out=super().restore(data)
        out.recursive_self_culture=RecursiveSelfCultureControlV3.restore(out.recursive_self_culture.checkpoint())
        return out
