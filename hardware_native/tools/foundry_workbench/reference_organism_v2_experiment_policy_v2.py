#!/usr/bin/env python3
"""One continuing Adult with safe-frontier adaptive experiment-policy learning.

The prior learned probe-geometry Adult is preserved in
reference_organism_v2_experiment_strategy_v1.py. Present-state safety remains owned by the
causal-experiment frontier. This layer learns a context-conditional mixture of discriminative
and confirmatory experiment policies over that already-safe set. Policy state has control
priority only and can never create evidence or make an unsafe intervention executable.
"""
from __future__ import annotations

import reference_organism_v2_experiment_strategy_v1 as _prior
for _name in dir(_prior):
    if not _name.startswith('__'):
        globals()[_name]=getattr(_prior,_name)

from reference_recursive_causal_experiment_v2 import (
    RecursiveCausalExperimentV2,POLICY_DISCRIMINATE,POLICY_CONFIRM,Q as POLICYQ,
)
from reference_recursive_experiment_policy_v2 import RecursiveExperimentPolicyV2

_BaseReferenceOrganismV2=_prior.ReferenceOrganismV2
EXPERIMENT_POLICY_SCHEMA=1

class ReferenceOrganismV2(_BaseReferenceOrganismV2):
    def __init__(self,population_spec=None):
        super().__init__(population_spec)
        self.recursive_causal_experiment=RecursiveCausalExperimentV2.restore(self.recursive_causal_experiment.checkpoint())
        self.recursive_experiment_policy=RecursiveExperimentPolicyV2()

    def _eligible_policy_frontier(self,branches,reason_map,resource,control):
        safe=self.recursive_causal_experiment.eligible_probes(branches,reason_map,resource,control)
        if not safe:return ()
        rows=[]
        for candidate in safe:
            branch=next((b for b in branches if int(b.action)==int(candidate.action)),None)
            if branch is None:continue
            predictions=reason_map.get(int(candidate.action),());key=self.recursive_experiment_strategy.structural_key(branch,branches,len(predictions),control)
            if self.recursive_experiment_strategy.permits(key):rows.append(candidate)
        return tuple(rows)

    def _culture_program_nomination(self):
        branches,rows,resource,control=self._metacontrol_program_branches()
        if not branches:
            self._metacontrol_inquiry_alternatives=();self._metacontrol_inquiry_asked=False;self._metacontrol_pending_intervention=0;return None
        prior=self.recursive_metacontrol.decision(self._metacontrol_pending_decision);previous=int(prior.selected) if prior is not None else 0
        result=self.recursive_metacontrol.choose(branches,resource,control,self._metacontrol_social_quality(),int(self._developmental_curriculum_tick),previous)
        self._metacontrol_pending_decision=int(result.get('decision',0));mode=int(result.get('mode',MODE_OBSERVE));selected=int(result.get('selected',0));alternatives=tuple(map(int,result.get('alternatives',())));reason_map=self._experiment_reason_map(rows)
        if mode in (MODE_ASK,MODE_OBSERVE):
            frontier=self._eligible_policy_frontier(branches,reason_map,resource,control)
            if frontier:
                policy,context=self.recursive_experiment_policy.choose_policy(frontier,resource,control)
                if policy and not self.recursive_experiment_policy.permits(policy,context):
                    alternate=POLICY_CONFIRM if policy==POLICY_DISCRIMINATE else POLICY_DISCRIMINATE;policy=alternate if self.recursive_experiment_policy.permits(alternate,context) else 0
                candidate=self.recursive_causal_experiment.policy_candidate(frontier,policy) if policy else None
                if candidate is not None:
                    probe=int(candidate.action);branch=next((b for b in branches if int(b.action)==probe),None);predictions=reason_map.get(probe,())
                    if branch is not None:
                        key=self.recursive_experiment_strategy.structural_key(branch,branches,len(predictions),control);gain=self.recursive_causal_experiment.information_gain_q16(branch,branches,predictions);before_certainty=self.recursive_causal_experiment.reason_calibration(probe,predictions)[1]
                        intervention=self.recursive_causal_experiment.begin(probe,self._metacontrol_pending_decision,predictions,alternatives,gain,int(self._developmental_curriculum_tick))
                        if intervention:
                            self.recursive_experiment_strategy.begin(intervention,key,before_certainty,True);self.recursive_experiment_policy.begin(intervention,policy,context,before_certainty,True);self._metacontrol_pending_intervention=int(intervention);self._metacontrol_inquiry_alternatives=();self._metacontrol_inquiry_asked=False;self._culture_active_program=();return probe
            self._metacontrol_pending_intervention=0
            if mode==MODE_ASK:
                if alternatives!=self._metacontrol_inquiry_alternatives:self._metacontrol_inquiry_asked=False
                self._metacontrol_inquiry_alternatives=alternatives
            else:self._metacontrol_inquiry_alternatives=();self._metacontrol_inquiry_asked=False
            return None
        self._metacontrol_pending_intervention=0;self._metacontrol_inquiry_alternatives=();self._metacontrol_inquiry_asked=False
        if mode==MODE_DEFER or selected<=0:return None
        chosen=next((row for row in rows if row.get('actions') and int(row['actions'][0])==selected),None)
        if chosen is None:return None
        self._culture_active_program=(int(chosen['identity']),0);return selected

    def _culture_episode_from_motor(self,motor,source):
        pending=int(self._metacontrol_pending_intervention);intervention=self.recursive_causal_experiment.intervention(pending) if pending else None
        identity=super()._culture_episode_from_motor(motor,source)
        if intervention is None or motor is None or not getattr(motor,'settled',False):return identity
        if int(intervention.action)!=int(getattr(motor,'action_id',0)):return identity
        after_certainty=self.recursive_causal_experiment.reason_calibration(int(intervention.action),intervention.reasons)[1]
        self.recursive_experiment_policy.settle(pending,after_certainty,bool(getattr(motor,'independent_consequence',False)),True);return identity

    def checkpoint(self):
        data=super().checkpoint();data['recursive_experiment_policy_v2']={'schema':EXPERIMENT_POLICY_SCHEMA,'state':self.recursive_experiment_policy.checkpoint()};return data

    @classmethod
    def restore(cls,data):
        out=super().restore(data);out.recursive_causal_experiment=RecursiveCausalExperimentV2.restore(out.recursive_causal_experiment.checkpoint());row=data.get('recursive_experiment_policy_v2')
        if row is None:out.recursive_experiment_policy=RecursiveExperimentPolicyV2();return out
        if int(row.get('schema',0))!=EXPERIMENT_POLICY_SCHEMA:raise ValueError('organism:experiment-policy-checkpoint')
        out.recursive_experiment_policy=RecursiveExperimentPolicyV2.restore(row['state']);return out

    def _run_recursive_experiment_policy_ratchet(self):
        causal=RecursiveCausalExperimentV2();owner=RecursiveExperimentPolicyV2();A=0xF101;B=0xF202
        high_a=PolicyBranchV1(A,POLICYQ//2,POLICYQ//2,POLICYQ//2,POLICYQ//8,3*POLICYQ//4,0);high_b=PolicyBranchV1(B,5*POLICYQ//8,POLICYQ//2,POLICYQ//2,POLICYQ//8,2*POLICYQ//3,0)
        reasons={A:(ReasonPredictionV1(0xFA11,9501,True,0),),B:(ReasonPredictionV1(0xFA22,9502,True,0),)};frontier=causal.eligible_probes((high_a,high_b),reasons,POLICYQ,POLICYQ);unsafe=causal.eligible_probes((high_a,high_b),reasons,POLICYQ//8,POLICYQ);policy,context=owner.choose_policy(frontier,POLICYQ,POLICYQ);before=owner.evidence_count
        owner.begin(1,POLICY_DISCRIMINATE,context,0,True);after_begin=owner.evidence_count;owner.settle(1,POLICYQ//4,False,True);after_nonindependent=owner.evidence_count
        owner.begin(2,POLICY_DISCRIMINATE,context,0,True);owner.settle(2,0,True,True);owner.begin(3,POLICY_DISCRIMINATE,context,0,True);owner.settle(3,0,True,True);owner.begin(4,POLICY_CONFIRM,context,0,True);owner.settle(4,POLICYQ//4,True,True);owner.begin(5,POLICY_CONFIRM,context,0,True);owner.settle(5,POLICYQ//4,True,True)
        shifted,_=owner.choose_policy(frontier,POLICYQ,POLICYQ);medium_context=owner.context(frontier,5*POLICYQ//8,POLICYQ)
        checks={'safe_frontier_contains_only_presently_admissible_probes':len(frontier)==2 and not unsafe,'high_ambiguity_prior_prefers_discriminative_policy':policy==POLICY_DISCRIMINATE,'policy_begin_creates_zero_evidence':after_begin==before,'nonindependent_outcome_cannot_train_policy':after_nonindependent==before,'realized_diagnosticity_can_shift_policy_mixture':shifted==POLICY_CONFIRM,'policy_competence_is_context_local':owner.evidence_for(POLICY_CONFIRM,medium_context)==0,'policy_learning_cannot_reanimate_resource_veto':not unsafe,'policy_traces_have_zero_authority':all(int(row.authority)==0 for row in owner._traces)}
        cp=owner.checkpoint();restored=RecursiveExperimentPolicyV2.restore(cp);checks['experiment_policy_survives_checkpoint']=restored.checkpoint()==cp
        return checks,sorted(name for name,value in checks.items() if not value)

    def run_frontier_multilingual_curriculum(self):
        receipt=super().run_frontier_multilingual_curriculum();checks,failed=self._run_recursive_experiment_policy_ratchet();merged=dict(receipt.get('checks',{}));merged.update(checks);all_failed=sorted(k for k,v in merged.items() if not v);receipt=dict(receipt);receipt['checks']=merged;receipt['failed']=all_failed;receipt['status']='GREEN' if not all_failed else 'RED';receipt['experiment_policy_evidence']=self.recursive_experiment_policy.evidence_count;receipt['experiment_policy_families']=['DISCRIMINATE','CONFIRM'];receipt['experiment_policy_authority']='CONTROL_PRIORITY_ONLY';return receipt
