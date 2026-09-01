#!/usr/bin/env python3
"""Latent-regime recursive Adult over the adaptive experiment-policy stack.

The prior policy-mixture Adult is preserved in reference_organism_v2_experiment_policy_v2.py.
World truth remains in the incumbent transition ecology. This layer infers latent causal
regimes from bounded resident cues and routes self, reason, experiment-strategy and policy
competence through the resolved regime. Regime splitting occurs only after ordinary world
settlement and before contextual competence evidence is written.
"""
from __future__ import annotations

import reference_organism_v2_experiment_policy_v2 as _prior
for _name in dir(_prior):
    if not _name.startswith('__'):
        globals()[_name]=getattr(_prior,_name)

import reference_organism_v2_causal_experiment_v1 as _causal_lifecycle
import reference_organism_v2_metacontrol_v1 as _metacontrol_branch_base
from reference_recursive_causal_regime_v1 import RecursiveCausalRegimeV1,Q as REGQ
from reference_recursive_policy_metacontrol_v2 import RecursivePolicyMetacontrolV2
from reference_recursive_causal_experiment_v3 import RecursiveCausalExperimentV3
from reference_recursive_experiment_strategy_v2 import RecursiveExperimentStrategyV2
from reference_recursive_experiment_policy_v3 import RecursiveExperimentPolicyV3

_BaseReferenceOrganismV2=_prior.ReferenceOrganismV2
LATENT_REGIME_SCHEMA=1

class ReferenceOrganismV2(_BaseReferenceOrganismV2):
    def __init__(self,population_spec=None):
        super().__init__(population_spec)
        self.recursive_metacontrol=RecursivePolicyMetacontrolV2.restore(self.recursive_metacontrol.checkpoint())
        self.recursive_causal_experiment=RecursiveCausalExperimentV3.restore(self.recursive_causal_experiment.checkpoint())
        self.recursive_experiment_strategy=RecursiveExperimentStrategyV2.restore(self.recursive_experiment_strategy.checkpoint())
        self.recursive_experiment_policy=RecursiveExperimentPolicyV3.restore(self.recursive_experiment_policy.checkpoint())
        self.recursive_causal_regime=RecursiveCausalRegimeV1();self._causal_regime_current=0

    @staticmethod
    def _regime_class(value):
        value=max(0,min(REGQ,int(value)));return 0 if value<REGQ//3 else (1 if value<2*REGQ//3 else 2)

    def _regime_cues(self,state=None,source=0,action=0):
        world=tuple(self.world_state if state is None and self.world_state is not None else (state or ()))
        target=tuple(self.body_target or ());body=tuple(getattr(self,'body_state',()) or ())
        source=int(source or getattr(self,'world_source',0) or getattr(self,'partner_source',0) or 1)
        resource,control=self._culture_body_envelope(source,int(action or 0))
        token=lambda tag,value:_culture_id('causal-regime-cue-v1',[str(tag),int(value)])
        cues=[]
        cues.extend(token('world',x) for x in world[:12]);cues.extend(token('target',x) for x in target[:8]);cues.extend(token('body',x) for x in body[:8])
        cues.append(token('resource-class',self._regime_class(resource)));cues.append(token('control-class',self._regime_class(control)))
        if getattr(self,'partner_present',False) and int(getattr(self,'partner_source',0))>0:
            cues.append(token('partner-present',1));cues.append(token('partner-source',int(self.partner_source)))
        return tuple(cues),int(resource),int(control)

    def _regime_tick(self):
        return max(int(self.recursive_causal_regime._tick),int(getattr(self,'tick_count',0)),int(getattr(self,'_developmental_curriculum_tick',0)))

    def infer_causal_regime(self,state=None,source=0,action=0):
        cues,_resource,_control=self._regime_cues(state,source,action)
        regime=self.recursive_causal_regime.infer(cues,self._regime_tick(),self._causal_regime_current)
        self._causal_regime_current=int(regime);return int(regime)

    def _metacontrol_program_branches(self):
        # Bypass the old global reason-calibration layer. Start from the body-free cultural
        # branches, then add only regime-specific procedural calibration.
        branches,rows,resource,control=_metacontrol_branch_base.ReferenceOrganismV2._metacontrol_program_branches(self)
        if not branches:return branches,rows,resource,control
        regime=self.infer_causal_regime(action=0);reason_map=self._experiment_reason_map(rows);adjusted=[]
        for branch in branches:
            predictions=reason_map.get(int(branch.action),())
            signed,certainty=self.recursive_causal_experiment.reason_calibration(int(branch.action),predictions,regime)
            if certainty:
                delta=(int(signed)*int(certainty))//(2*EXPQ);support=max(0,min(EXPQ,int(branch.support_q16)+delta));gap=max(0,int(branch.epistemic_gap_q16)-int(certainty)//3)
                branch=PolicyBranchV1(int(branch.action),support,int(branch.counterfactual_q16),int(branch.source_quality_q16),int(branch.consequence_risk_q16),gap,0)
            adjusted.append(branch)
        return tuple(adjusted),rows,resource,control

    def _eligible_policy_frontier(self,branches,reason_map,resource,control,regime):
        safe=self.recursive_causal_experiment.eligible_probes(branches,reason_map,resource,control,regime)
        rows=[]
        for candidate in safe:
            branch=next((b for b in branches if int(b.action)==int(candidate.action)),None)
            if branch is None:continue
            predictions=reason_map.get(int(candidate.action),());key=self.recursive_experiment_strategy.structural_key(branch,branches,len(predictions),control)
            if self.recursive_experiment_strategy.permits(key,regime):rows.append(candidate)
        return tuple(rows)

    def _culture_program_nomination(self):
        branches,rows,resource,control=self._metacontrol_program_branches()
        if not branches:
            self._metacontrol_inquiry_alternatives=();self._metacontrol_inquiry_asked=False;self._metacontrol_pending_intervention=0;return None
        regime=int(self._causal_regime_current or self.infer_causal_regime())
        prior=self.recursive_metacontrol.decision(self._metacontrol_pending_decision);previous=int(prior.selected) if prior is not None else 0
        result=self.recursive_metacontrol.choose(branches,resource,control,self._metacontrol_social_quality(),
            int(self._developmental_curriculum_tick),previous,regime)
        self._metacontrol_pending_decision=int(result.get('decision',0));mode=int(result.get('mode',MODE_OBSERVE));selected=int(result.get('selected',0));alternatives=tuple(map(int,result.get('alternatives',())));reason_map=self._experiment_reason_map(rows)
        if mode in (MODE_ASK,MODE_OBSERVE):
            frontier=self._eligible_policy_frontier(branches,reason_map,resource,control,regime)
            if frontier:
                policy,context=self.recursive_experiment_policy.choose_policy(frontier,resource,control,regime)
                if policy and not self.recursive_experiment_policy.permits(policy,context,regime):
                    alternate=POLICY_CONFIRM if policy==POLICY_DISCRIMINATE else POLICY_DISCRIMINATE
                    policy=alternate if self.recursive_experiment_policy.permits(alternate,context,regime) else 0
                candidate=self.recursive_causal_experiment.policy_candidate(frontier,policy) if policy else None
                if candidate is not None:
                    probe=int(candidate.action);branch=next((b for b in branches if int(b.action)==probe),None);predictions=reason_map.get(probe,())
                    if branch is not None:
                        key=self.recursive_experiment_strategy.structural_key(branch,branches,len(predictions),control)
                        gain=self.recursive_causal_experiment.information_gain_q16(branch,branches,predictions,regime)
                        before_certainty=self.recursive_causal_experiment.reason_calibration(probe,predictions,regime)[1]
                        intervention=self.recursive_causal_experiment.begin(probe,self._metacontrol_pending_decision,predictions,alternatives,gain,
                            int(self._developmental_curriculum_tick),regime,True)
                        if intervention:
                            self.recursive_experiment_strategy.begin(intervention,key,before_certainty,True,regime)
                            self.recursive_experiment_policy.begin(intervention,policy,context,before_certainty,True,regime)
                            self._metacontrol_pending_intervention=int(intervention);self._metacontrol_inquiry_alternatives=();self._metacontrol_inquiry_asked=False;self._culture_active_program=();return probe
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
        if motor is None or not getattr(motor,'settled',False):
            return _causal_lifecycle.ReferenceOrganismV2._culture_episode_from_motor(self,motor,source)
        pending=int(self._metacontrol_pending_intervention);intervention=self.recursive_causal_experiment.intervention(pending) if pending else None
        action=int(getattr(motor,'action_id',0));success=int(getattr(motor,'effect',0))>0;independent=bool(getattr(motor,'independent_consequence',False))
        state_before=tuple(getattr(motor,'state_before',()) or ());cues,_resource,_control=self._regime_cues(state_before,source,action)
        proposed=(self.recursive_causal_experiment.intervention_regime(pending) if intervention is not None else int(self._causal_regime_current or self.infer_causal_regime(state_before,source,action)))
        if proposed<=0:proposed=self.infer_causal_regime(state_before,source,action)
        prior_self=self.recursive_metacontrol.reliability_q16(action,proposed);prior_count=self.recursive_metacontrol.regime_evidence_count(action,proposed)

        # Settle the incumbent world/culture/global compatibility path, deliberately
        # bypassing the old global strategy/policy post-processing layers.
        identity=_causal_lifecycle.ReferenceOrganismV2._culture_episode_from_motor(self,motor,source)
        resolved=self.recursive_causal_regime.resolve_after_outcome(proposed,cues,prior_self,success,independent,prior_count,self._regime_tick())
        self._causal_regime_current=int(resolved)
        self.recursive_metacontrol.record_regime_outcome(action,success,independent,resolved)

        if intervention is not None:
            self.recursive_causal_experiment.settle_regime_evidence(pending,resolved)
            after_certainty=self.recursive_causal_experiment.reason_calibration(action,intervention.reasons,resolved)[1]
            same_regime=int(resolved)==int(proposed)
            self.recursive_experiment_strategy.settle(pending,after_certainty,independent,same_regime,resolved)
            self.recursive_experiment_policy.settle(pending,after_certainty,independent,same_regime,resolved)
        return identity

    def reason_predictive_reliability_q16(self,reason,source,action,regime=0):
        regime=int(regime or self._causal_regime_current)
        return self.recursive_causal_experiment.reason_reliability_q16(int(reason),int(source),int(action),regime)

    def self_reliability_q16(self,action,regime=0):
        return self.recursive_metacontrol.reliability_q16(int(action),int(regime or self._causal_regime_current))

    def checkpoint(self):
        data=super().checkpoint();data['latent_causal_regime_v1']={'schema':LATENT_REGIME_SCHEMA,
            'state':self.recursive_causal_regime.checkpoint(),'current':int(self._causal_regime_current)};return data

    @classmethod
    def restore(cls,data):
        out=super().restore(data)
        m=data.get('recursive_metacontrol_v1',{}).get('state');c=data.get('recursive_causal_experiment_v1',{}).get('state')
        s=data.get('recursive_experiment_strategy_v1',{}).get('state');p=data.get('recursive_experiment_policy_v2',{}).get('state')
        if m is not None:out.recursive_metacontrol=RecursivePolicyMetacontrolV2.restore(m)
        if c is not None:out.recursive_causal_experiment=RecursiveCausalExperimentV3.restore(c)
        if s is not None:out.recursive_experiment_strategy=RecursiveExperimentStrategyV2.restore(s)
        if p is not None:out.recursive_experiment_policy=RecursiveExperimentPolicyV3.restore(p)
        row=data.get('latent_causal_regime_v1')
        if row is None:out.recursive_causal_regime=RecursiveCausalRegimeV1();out._causal_regime_current=0;return out
        if int(row.get('schema',0))!=LATENT_REGIME_SCHEMA:raise ValueError('organism:latent-regime-checkpoint')
        out.recursive_causal_regime=RecursiveCausalRegimeV1.restore(row['state']);out._causal_regime_current=int(row.get('current',0));return out

    def _run_latent_causal_regime_ratchet(self):
        regimes=RecursiveCausalRegimeV1();meta=RecursivePolicyMetacontrolV2();causal=RecursiveCausalExperimentV3();A=0xAB11
        cues_a=(101,102,103);r=regimes.infer(cues_a,1,0)
        for tick in range(2,5):
            prior=meta.reliability_q16(A,r);count=meta.regime_evidence_count(A,r);resolved=regimes.resolve_after_outcome(r,cues_a,prior,True,True,count,tick);meta.record_regime_outcome(A,True,True,resolved);r=resolved
        cues_similar=(101,102,103,104);same=regimes.infer(cues_similar,5,r)
        cues_b=(101,102,105,106);proposed=regimes.infer(cues_b,6,r);prior=meta.reliability_q16(A,proposed);count=meta.regime_evidence_count(A,proposed)
        split=regimes.resolve_after_outcome(proposed,cues_b,prior,False,True,count,7);meta.record_regime_outcome(A,False,True,split)
        returned=regimes.infer(cues_a,8,split)
        identical_owner=RecursiveCausalRegimeV1();ident_meta=RecursivePolicyMetacontrolV2();ri=identical_owner.infer(cues_a,1,0)
        for tick in range(2,5):
            p=ident_meta.reliability_q16(A,ri);n=ident_meta.regime_evidence_count(A,ri);ri2=identical_owner.resolve_after_outcome(ri,cues_a,p,True,True,n,tick);ident_meta.record_regime_outcome(A,True,True,ri2);ri=ri2
        before_count=identical_owner.regime_count;p=ident_meta.reliability_q16(A,ri);n=ident_meta.regime_evidence_count(A,ri);same_after_failure=identical_owner.resolve_after_outcome(ri,cues_a,p,False,True,n,5)
        reason=ReasonPredictionV1(0xAB21,9601,True,0);intervention=causal.begin(A,0,(reason,),(A,),REGQ//2,1,r,True);causal.settle(intervention,True,True,2);causal.settle_regime_evidence(intervention,r)
        checks={
          'high_cue_overlap_reuses_existing_regime':same==r,
          'independent_contradiction_with_cue_basis_splits_regime':split!=r and regimes.regime(split).parent==r,
          'regime_specific_self_reliability_does_not_average_contradiction':meta.reliability_q16(A,r)>REGQ//2 and meta.reliability_q16(A,split)<REGQ//2,
          'returning_to_prior_cues_recovers_prior_regime':returned==r,
          'identical_cue_contradiction_updates_same_regime':same_after_failure==ri and identical_owner.regime_count==before_count,
          'reason_evidence_is_regime_local':causal.reason_reliability_q16(reason.reason,reason.source,A,r)>REGQ//2 and causal.reason_reliability_q16(reason.reason,reason.source,A,split)==REGQ//2,
          'regime_and_contextual_evidence_have_zero_cross_owner_world_authority':all(int(x.authority)==0 for x in regimes._regimes),
        }
        cp=(regimes.checkpoint(),meta.checkpoint(),causal.checkpoint());checks['latent_regime_stack_survives_checkpoint']=(RecursiveCausalRegimeV1.restore(cp[0]).checkpoint()==cp[0] and RecursivePolicyMetacontrolV2.restore(cp[1]).checkpoint()==cp[1] and RecursiveCausalExperimentV3.restore(cp[2]).checkpoint()==cp[2])
        return checks,sorted(name for name,value in checks.items() if not value)

    def run_frontier_multilingual_curriculum(self):
        receipt=super().run_frontier_multilingual_curriculum();checks,failed=self._run_latent_causal_regime_ratchet();merged=dict(receipt.get('checks',{}));merged.update(checks);all_failed=sorted(k for k,v in merged.items() if not v)
        receipt=dict(receipt);receipt['checks']=merged;receipt['failed']=all_failed;receipt['status']='GREEN' if not all_failed else 'RED';receipt['latent_causal_regimes']=self.recursive_causal_regime.regime_count;receipt['contextual_self_outcomes']=self.recursive_metacontrol.regime_outcome_count;receipt['contextual_reason_outcomes']=self.recursive_causal_experiment.regime_reason_outcome_count;return receipt
