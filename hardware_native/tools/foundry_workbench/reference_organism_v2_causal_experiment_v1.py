#!/usr/bin/env python3
"""Recursive metacontrol plus low-risk causal testing in one continuing Adult.

The prior metacontrol Adult is preserved in reference_organism_v2_metacontrol_v1.py.
World-transition learning remains owned by the incumbent organism. This layer adds only a
resident record of which self-model and source-qualified cultural reasons were deliberately
put at risk by a low-risk discriminating probe. Imagined branches and reason predictions
remain authority zero until an independent lived consequence returns.
"""
from __future__ import annotations

import reference_organism_v2_metacontrol_v1 as _prior
for _name in dir(_prior):
    if not _name.startswith('__'):
        globals()[_name]=getattr(_prior,_name)

from reference_recursive_causal_experiment_v1 import RecursiveCausalExperimentV1,ReasonPredictionV1,Q as EXPQ

_BaseReferenceOrganismV2=_prior.ReferenceOrganismV2
CAUSAL_EXPERIMENT_SCHEMA=1

class ReferenceOrganismV2(_BaseReferenceOrganismV2):
    def __init__(self,population_spec=None):
        super().__init__(population_spec);self.recursive_causal_experiment=RecursiveCausalExperimentV1();self._metacontrol_pending_intervention=0

    def _experiment_reason_map(self,rows):
        result={};withdrawn=set(getattr(self.recursive_self_culture,'_withdrawn_sources',()))
        for row in rows:
            if not row.get('actions'):continue
            action=int(row['actions'][0]);program=next((p for p in self.recursive_self_culture._programs if int(p.identity)==int(row['identity'])),None)
            if program is None:continue
            bucket=result.setdefault(action,[])
            for reason_id in program.reason_ids:
                reason=next((r for r in self.recursive_self_culture._reasons if int(r.identity)==int(reason_id) and int(r.source) not in withdrawn),None)
                if reason is None:continue
                prediction=ReasonPredictionV1(int(reason.identity),int(reason.source),True,0)
                if prediction not in bucket:bucket.append(prediction)
        return {action:tuple(rows) for action,rows in result.items()}

    def _metacontrol_program_branches(self):
        branches,rows,resource,control=super()._metacontrol_program_branches()
        if not branches:return branches,rows,resource,control
        reason_map=self._experiment_reason_map(rows);adjusted=[]
        for branch in branches:
            predictions=reason_map.get(int(branch.action),());signed,certainty=self.recursive_causal_experiment.reason_calibration(int(branch.action),predictions)
            if certainty:
                delta=(int(signed)*int(certainty))//(2*EXPQ);support=max(0,min(EXPQ,int(branch.support_q16)+delta));gap=max(0,int(branch.epistemic_gap_q16)-int(certainty)//3)
                branch=PolicyBranchV1(int(branch.action),support,int(branch.counterfactual_q16),int(branch.source_quality_q16),int(branch.consequence_risk_q16),gap,0)
            adjusted.append(branch)
        return tuple(adjusted),rows,resource,control

    def _culture_program_nomination(self):
        branches,rows,resource,control=self._metacontrol_program_branches()
        if not branches:self._metacontrol_inquiry_alternatives=();self._metacontrol_inquiry_asked=False;self._metacontrol_pending_intervention=0;return None
        prior=self.recursive_metacontrol.decision(self._metacontrol_pending_decision);previous=int(prior.selected) if prior is not None else 0
        result=self.recursive_metacontrol.choose(branches,resource,control,self._metacontrol_social_quality(),int(self._developmental_curriculum_tick),previous)
        self._metacontrol_pending_decision=int(result.get('decision',0));mode=int(result.get('mode',MODE_OBSERVE));selected=int(result.get('selected',0));alternatives=tuple(map(int,result.get('alternatives',())));reason_map=self._experiment_reason_map(rows)
        if mode in (MODE_ASK,MODE_OBSERVE):
            probe=self.recursive_causal_experiment.select_probe(branches,reason_map,resource,control)
            if probe>0:
                branch=next((b for b in branches if int(b.action)==probe),None)
                if branch is not None:
                    gain=self.recursive_causal_experiment.information_gain_q16(branch,branches,reason_map.get(probe,()))
                    intervention=self.recursive_causal_experiment.begin(probe,self._metacontrol_pending_decision,reason_map.get(probe,()),alternatives,gain,int(self._developmental_curriculum_tick))
                    if intervention:self._metacontrol_pending_intervention=int(intervention);self._metacontrol_inquiry_alternatives=();self._metacontrol_inquiry_asked=False;self._culture_active_program=();return probe
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
        identity=super()._culture_episode_from_motor(motor,source)
        if motor is None or not getattr(motor,'settled',False):return identity
        intervention=self.recursive_causal_experiment.intervention(self._metacontrol_pending_intervention)
        if intervention is None or int(intervention.action)!=int(getattr(motor,'action_id',0)):return identity
        success=int(getattr(motor,'effect',0))>0;independent=bool(getattr(motor,'independent_consequence',False))
        self.recursive_metacontrol.record_outcome(int(motor.action_id),success,independent)
        self.recursive_causal_experiment.settle(int(intervention.identity),success,independent,max(int(self.recursive_causal_experiment._tick),int(self._developmental_curriculum_tick)))
        self._metacontrol_pending_intervention=0;self._metacontrol_pending_decision=0;return identity

    def contact(self,kind,payload,source,authenticated=True,independent=True):
        payload=tuple(payload)
        if int(kind)==CONTACT_WITHDRAW_SOURCE and len(payload)==1:self.recursive_causal_experiment.withdraw_source(int(payload[0]))
        return super().contact(kind,payload,source,authenticated,independent)

    def reason_predictive_reliability_q16(self,reason,source,action):return self.recursive_causal_experiment.reason_reliability_q16(int(reason),int(source),int(action))

    def checkpoint(self):
        data=super().checkpoint();data['recursive_causal_experiment_v1']={'schema':CAUSAL_EXPERIMENT_SCHEMA,'state':self.recursive_causal_experiment.checkpoint(),'pending_intervention':int(self._metacontrol_pending_intervention)};return data

    @classmethod
    def restore(cls,data):
        out=super().restore(data);row=data.get('recursive_causal_experiment_v1')
        if row is None:return out
        if int(row.get('schema',0))!=CAUSAL_EXPERIMENT_SCHEMA:raise ValueError('organism:causal-experiment-checkpoint')
        out.recursive_causal_experiment=RecursiveCausalExperimentV1.restore(row['state']);out._metacontrol_pending_intervention=int(row.get('pending_intervention',0));return out

    def _run_recursive_causal_experiment_ratchet(self):
        owner=RecursiveCausalExperimentV1();A=0xE101;B=0xE202;R1=0xEA11;R2=0xEA22
        a=PolicyBranchV1(A,EXPQ//2,EXPQ//2,EXPQ//2,EXPQ//8,3*EXPQ//4,0);b=PolicyBranchV1(B,EXPQ//2,EXPQ//2,EXPQ//2,EXPQ//8,EXPQ//2,0)
        reasons={A:(ReasonPredictionV1(R1,9301,True,0),),B:(ReasonPredictionV1(R2,9302,True,0),)};before=owner.reason_outcome_count;baseline_cal=owner.reason_calibration(A,reasons[A])
        probe=owner.select_probe((a,b),reasons,EXPQ,EXPQ);no_resource=owner.select_probe((a,b),reasons,EXPQ//8,EXPQ);high_risk=PolicyBranchV1(A,EXPQ//2,EXPQ//2,EXPQ//2,3*EXPQ//4,3*EXPQ//4,0);no_risky=owner.select_probe((high_risk,b),reasons,EXPQ,EXPQ)
        gain=owner.information_gain_q16(a,(a,b),reasons[A]);first=owner.begin(A,1,reasons[A],(A,B),gain,1);owner.settle(first,True,False,2);after_nonindependent=owner.reason_outcome_count
        second=owner.begin(A,2,reasons[A],(A,B),gain,3);owner.settle(second,True,True,4);after_independent=owner.reason_outcome_count;learned=owner.reason_reliability_q16(R1,9301,A);learned_cal=owner.reason_calibration(A,reasons[A]);other_action_same_reason=owner.reason_reliability_q16(R1,9301,B)
        history_before_withdraw=owner.reason_outcome_count;owner.withdraw_source(9301);third=owner.begin(A,3,reasons[A],(A,B),gain,5);owner.settle(third,False,True,6);withdrawn_cal=owner.reason_calibration(A,reasons[A])
        checks={'high_gap_low_risk_controllable_branch_can_become_probe':probe==A,'resource_gate_prevents_curiosity_action':no_resource==0,'high_consequence_risk_prevents_probe':no_risky!=A,'probe_selection_creates_no_reason_evidence':before==0 and baseline_cal==(0,0),'nonindependent_probe_cannot_calibrate_reason':after_nonindependent==before,'independent_probe_calibrates_only_exposed_reason':after_independent==before+1 and learned>EXPQ//2 and owner.reason_reliability_q16(R2,9302,B)==EXPQ//2,'procedural_reason_reliability_is_action_local':other_action_same_reason==EXPQ//2,'tested_reason_changes_later_deliberative_evidence':learned_cal[0]>0 and learned_cal[1]>0,'withdrawal_preserves_relationship_history_but_deactivates_reason':owner.reason_outcome_count==history_before_withdraw and owner.reason_reliability_q16(R1,9301,A)==learned and withdrawn_cal==(0,0),'experiment_and_reason_predictions_have_zero_authority':a.authority==b.authority==reasons[A][0].authority==0}
        cp=owner.checkpoint();restored=RecursiveCausalExperimentV1.restore(cp);checks['causal_experiment_state_survives_checkpoint']=restored.checkpoint()==cp
        return checks,sorted(name for name,value in checks.items() if not value)

    def run_frontier_multilingual_curriculum(self):
        receipt=super().run_frontier_multilingual_curriculum();checks,failed=self._run_recursive_causal_experiment_ratchet();merged=dict(receipt.get('checks',{}));merged.update(checks);all_failed=sorted(k for k,v in merged.items() if not v);receipt=dict(receipt);receipt['checks']=merged;receipt['failed']=all_failed;receipt['status']='GREEN' if not all_failed else 'RED';receipt['causal_experiment_interventions']=self.recursive_causal_experiment.intervention_count;receipt['causal_experiment_reason_outcomes']=self.recursive_causal_experiment.reason_outcome_count;receipt['causal_experiment_authority']=0;return receipt
