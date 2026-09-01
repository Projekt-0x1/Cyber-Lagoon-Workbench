#!/usr/bin/env python3
"""Causal-experiment Adult plus learned experiment-strategy competence.

The prior causal-experiment Adult is preserved in reference_organism_v2_causal_experiment_v1.py.
Present resource/control/risk/uncertainty safety remains owned by that causal-experiment path.
This layer learns only whether self-selected probe geometries actually reduced procedural
uncertainty after independent lived consequence; learned history may suppress useless probe
strategies but can never make a currently unsafe probe executable.
"""
from __future__ import annotations

import reference_organism_v2_causal_experiment_v1 as _prior
for _name in dir(_prior):
    if not _name.startswith('__'):
        globals()[_name]=getattr(_prior,_name)

from reference_recursive_experiment_strategy_v1 import RecursiveExperimentStrategyV1,Q as STRATQ

_BaseReferenceOrganismV2=_prior.ReferenceOrganismV2
EXPERIMENT_STRATEGY_SCHEMA=1

class ReferenceOrganismV2(_BaseReferenceOrganismV2):
    def __init__(self,population_spec=None):
        super().__init__(population_spec);self.recursive_experiment_strategy=RecursiveExperimentStrategyV1()

    def _culture_program_nomination(self):
        branches,rows,resource,control=self._metacontrol_program_branches()
        if not branches:
            self._metacontrol_inquiry_alternatives=();self._metacontrol_inquiry_asked=False;self._metacontrol_pending_intervention=0;return None
        prior=self.recursive_metacontrol.decision(self._metacontrol_pending_decision);previous=int(prior.selected) if prior is not None else 0
        result=self.recursive_metacontrol.choose(branches,resource,control,self._metacontrol_social_quality(),int(self._developmental_curriculum_tick),previous)
        self._metacontrol_pending_decision=int(result.get('decision',0));mode=int(result.get('mode',MODE_OBSERVE));selected=int(result.get('selected',0));alternatives=tuple(map(int,result.get('alternatives',())))
        reason_map=self._experiment_reason_map(rows)
        if mode in (MODE_ASK,MODE_OBSERVE):
            probe=self.recursive_causal_experiment.select_probe(branches,reason_map,resource,control)
            if probe>0:
                branch=next((b for b in branches if int(b.action)==probe),None);predictions=reason_map.get(probe,())
                if branch is not None:
                    key=self.recursive_experiment_strategy.structural_key(branch,branches,len(predictions),control)
                    if self.recursive_experiment_strategy.permits(key):
                        gain=self.recursive_causal_experiment.information_gain_q16(branch,branches,predictions);before_certainty=self.recursive_causal_experiment.reason_calibration(probe,predictions)[1]
                        intervention=self.recursive_causal_experiment.begin(probe,self._metacontrol_pending_decision,predictions,alternatives,gain,int(self._developmental_curriculum_tick))
                        if intervention:
                            self.recursive_experiment_strategy.begin(intervention,key,before_certainty,True);self._metacontrol_pending_intervention=int(intervention);self._metacontrol_inquiry_alternatives=();self._metacontrol_inquiry_asked=False;self._culture_active_program=();return probe
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
        self.recursive_experiment_strategy.settle(pending,after_certainty,bool(getattr(motor,'independent_consequence',False)),True);return identity

    def checkpoint(self):
        data=super().checkpoint();data['recursive_experiment_strategy_v1']={'schema':EXPERIMENT_STRATEGY_SCHEMA,'state':self.recursive_experiment_strategy.checkpoint()};return data

    @classmethod
    def restore(cls,data):
        out=super().restore(data);row=data.get('recursive_experiment_strategy_v1')
        if row is None:return out
        if int(row.get('schema',0))!=EXPERIMENT_STRATEGY_SCHEMA:raise ValueError('organism:experiment-strategy-checkpoint')
        out.recursive_experiment_strategy=RecursiveExperimentStrategyV1.restore(row['state']);return out

    def _run_recursive_experiment_strategy_ratchet(self):
        owner=RecursiveExperimentStrategyV1();A=0xD101;B=0xD202
        a=PolicyBranchV1(A,STRATQ//2,STRATQ//2,STRATQ//2,STRATQ//8,3*STRATQ//4,0);b=PolicyBranchV1(B,STRATQ//2,STRATQ//2,STRATQ//2,STRATQ//8,STRATQ//2,0)
        key=owner.structural_key(a,(a,b),1,STRATQ);before=owner.evidence_count;owner.begin(1,key,0,True);after_begin=owner.evidence_count
        owner.settle(1,STRATQ//4,False,True);after_nonindependent=owner.evidence_count;owner.begin(2,key,0,False);owner.settle(2,STRATQ//4,True,False);after_forced=owner.evidence_count
        owner.begin(3,key,0,True);owner.settle(3,0,True,True);owner.begin(4,key,0,True);owner.settle(4,0,True,True);suppressed=not owner.permits(key)
        owner.begin(5,key,0,True);owner.settle(5,STRATQ//4,True,True);recovered=owner.permits(key)
        causal=RecursiveCausalExperimentV1();reasons={A:(ReasonPredictionV1(0xDA11,9401,True,0),)};unsafe=causal.select_probe((a,),reasons,STRATQ//8,STRATQ)
        checks={'probe_nomination_creates_zero_strategy_evidence':after_begin==before,'nonindependent_outcome_cannot_train_experiment_strategy':after_nonindependent==before,
          'forced_intervention_cannot_train_self_generated_strategy':after_forced==before,'repeated_zero_diagnosticity_can_suppress_probe_geometry':suppressed,
          'realized_diagnosticity_can_rehabilitate_strategy':recovered and owner.competence_q16(key)>0,'learned_strategy_cannot_override_current_resource_veto':unsafe==0,
          'strategy_trace_authority_is_zero':all(int(row.authority)==0 for row in owner._traces)}
        cp=owner.checkpoint();restored=RecursiveExperimentStrategyV1.restore(cp);checks['experiment_strategy_survives_checkpoint']=restored.checkpoint()==cp
        return checks,sorted(name for name,value in checks.items() if not value)

    def run_frontier_multilingual_curriculum(self):
        receipt=super().run_frontier_multilingual_curriculum();checks,failed=self._run_recursive_experiment_strategy_ratchet();merged=dict(receipt.get('checks',{}));merged.update(checks);all_failed=sorted(k for k,v in merged.items() if not v)
        receipt=dict(receipt);receipt['checks']=merged;receipt['failed']=all_failed;receipt['status']='GREEN' if not all_failed else 'RED';receipt['experiment_strategy_evidence']=self.recursive_experiment_strategy.evidence_count;receipt['experiment_strategy_authority']='CONTROL_PRIORITY_ONLY';return receipt
