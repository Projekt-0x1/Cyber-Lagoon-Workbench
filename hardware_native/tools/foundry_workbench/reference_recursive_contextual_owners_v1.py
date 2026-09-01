#!/usr/bin/env python3
"""Semantic adapters separating WORLD_CONTEXT evidence from CONTROL_CONTEXT competence.

Underlying storage retains checkpoint compatibility. Current call sites use explicit causal-role
names. Metacontrol and probe nomination are non-compensatory; contextual evidence may only
settle into a context recorded before outcome; post-hoc context re-keying is forbidden.
"""
from __future__ import annotations
from reference_recursive_policy_metacontrol_v2 import RecursivePolicyMetacontrolV2
from reference_recursive_policy_metacontrol_v3 import RecursivePolicyMetacontrolV3
from reference_recursive_causal_experiment_v3 import RecursiveCausalExperimentV3
from reference_recursive_causal_experiment_v4 import RecursiveCausalExperimentV4
from reference_recursive_experiment_strategy_v1 import RecursiveExperimentStrategyV1
from reference_recursive_experiment_strategy_v2 import RecursiveExperimentStrategyV2
from reference_recursive_experiment_policy_v2 import RecursiveExperimentPolicyV2
from reference_recursive_experiment_policy_v3 import RecursiveExperimentPolicyV3

class ContextualMetacontrolV1(RecursivePolicyMetacontrolV3):
    def self_reliability_q16(self,action,control_context=0):return super().reliability_q16(int(action),int(control_context))
    def record_self_outcome(self,action,success,independent=True,control_context=0):
        if int(control_context)>0:return super().record_regime_outcome(int(action),bool(success),bool(independent),int(control_context))
        return super().record_outcome(int(action),bool(success),bool(independent),0)
    def choose_contextual(self,branches,resource_q16,controllability_q16,social_quality_q16,tick,previous_action=0,control_context=0):return super().choose(branches,resource_q16,controllability_q16,social_quality_q16,tick,previous_action,int(control_context))
    @classmethod
    def restore(cls,data):
        old=RecursivePolicyMetacontrolV2.restore(data);out=cls();out.__dict__.update(old.__dict__);return out

class ContextualCausalExperimentV1(RecursiveCausalExperimentV4):
    def recommendation_reliability_world_q16(self,reason,source,action,world_context=0):return super().reason_reliability_q16(int(reason),int(source),int(action),int(world_context))
    def recommendation_calibration_world(self,action,recommendations,world_context=0):return super().reason_calibration(int(action),recommendations,int(world_context))
    def recommendation_uncertainty_world_q16(self,action,recommendations,world_context=0):return super()._recommendation_uncertainty_world_q16(int(action),recommendations,int(world_context))
    def recommendation_information_gain_world_q16(self,branch,all_branches,recommendations,world_context=0):return super().information_gain_q16(branch,all_branches,recommendations,int(world_context))
    def eligible_world_probes(self,branches,recommendation_map,resource_q16,controllability_q16,world_context=0):return super().eligible_probes(branches,recommendation_map,resource_q16,controllability_q16,int(world_context))
    def begin_world_probe(self,action,decision,recommendations,alternatives,information_gain_q16,tick,world_context=0,defer_world_context=False):return super().begin(action,decision,recommendations,alternatives,information_gain_q16,tick,int(world_context),bool(defer_world_context))
    def intervention_world_context(self,identity):return super().intervention_regime(identity)
    def settle_world_context_evidence(self,identity,resolved_world_context):
        identity=int(identity);resolved=int(resolved_world_context)
        if identity not in self._deferred_regime:return False
        recorded=int(self._intervention_regime.get(identity,0))
        if recorded<=0 or resolved<=0:return False
        return super().settle_regime_evidence(identity,resolved)
    # Historical aliases only; not proposition prediction/truth.
    def reason_reliability_world_q16(self,*args,**kwargs):return self.recommendation_reliability_world_q16(*args,**kwargs)
    def reason_calibration_world(self,*args,**kwargs):return self.recommendation_calibration_world(*args,**kwargs)
    def information_gain_world_q16(self,*args,**kwargs):return self.recommendation_information_gain_world_q16(*args,**kwargs)
    @classmethod
    def restore(cls,data):
        old=RecursiveCausalExperimentV3.restore(data);out=cls();out.__dict__.update(old.__dict__);return out

class ContextualExperimentStrategyV1(RecursiveExperimentStrategyV2):
    def competence_control_q16(self,key,control_context=0):return super().competence_q16(key,int(control_context))
    def permits_control(self,key,control_context=0):return super().permits(key,int(control_context))
    def begin_control(self,intervention,key,before_certainty_q16,self_selected=True,control_context=0):return super().begin(intervention,key,before_certainty_q16,self_selected,int(control_context))
    def settle_control(self,intervention,after_certainty_q16,independent=True,self_selected=True,control_context=0):
        intervention=int(intervention);requested=int(control_context);recorded=int(self._intervention_regime.get(intervention,0))
        if recorded<=0:return RecursiveExperimentStrategyV1.settle(self,intervention,after_certainty_q16,independent,self_selected)
        if requested!=recorded:raise RuntimeError('experiment-strategy:control-context-rekey')
        return super().settle(intervention,after_certainty_q16,independent,self_selected,recorded)
    @classmethod
    def restore(cls,data):
        old=RecursiveExperimentStrategyV2.restore(data);out=cls();out.__dict__.update(old.__dict__);return out

class ContextualExperimentPolicyV1(RecursiveExperimentPolicyV3):
    def choose_control_policy(self,candidates,resource_q16,controllability_q16,control_context=0):return super().choose_policy(candidates,resource_q16,controllability_q16,int(control_context))
    def permits_control_policy(self,policy,context,control_context=0):return super().permits(policy,context,int(control_context))
    def begin_control_policy(self,intervention,policy,context,before_certainty_q16,self_selected=True,control_context=0):return super().begin(intervention,policy,context,before_certainty_q16,self_selected,int(control_context))
    def settle_control_policy(self,intervention,after_certainty_q16,independent=True,self_selected=True,control_context=0):
        intervention=int(intervention);requested=int(control_context);recorded=int(self._intervention_regime.get(intervention,0))
        if recorded<=0:return RecursiveExperimentPolicyV2.settle(self,intervention,after_certainty_q16,independent,self_selected)
        if requested!=recorded:raise RuntimeError('experiment-policy:control-context-rekey')
        return super().settle(intervention,after_certainty_q16,independent,self_selected,recorded)
    @classmethod
    def restore(cls,data):
        old=RecursiveExperimentPolicyV3.restore(data);out=cls();out.__dict__.update(old.__dict__);return out
