#!/usr/bin/env python3
"""Public Adult with explicit WORLD_CONTEXT, CONTROL_CONTEXT, and PARTNER_SHARED_ACCESS.

This layer corrects three category errors in the previous latent-regime stack:
- self reliability is never a world-transition prediction;
- body/control/social context never changes procedural reason truth directly;
- consequence-certified partner access is never promoted to proposition-level belief.
The prior overloaded-context Adult is preserved in reference_organism_v2_overloaded_context_v1.py.
"""
from __future__ import annotations

import reference_organism_v2_overloaded_context_v1 as _prior
for _name in dir(_prior):
    if not _name.startswith('__') and _name not in {'_prior','_BaseReferenceOrganismV2'}:
        globals()[_name]=getattr(_prior,_name)

import reference_organism_v2_causal_experiment_v1 as _causal_lifecycle
import reference_organism_v2_metacontrol_v1 as _metacontrol_branch_base
from reference_recursive_context_partition_v2 import (
    RecursiveContextPartitionV2,WORLD_CONTEXT,CONTROL_CONTEXT,Q as CTXQ,
)
from reference_recursive_contextual_owners_v1 import (
    ContextualMetacontrolV1,ContextualCausalExperimentV1,
    ContextualExperimentStrategyV1,ContextualExperimentPolicyV1,
)
from reference_recursive_partner_access_v2 import RecursivePartnerAccessV2

_BaseReferenceOrganismV2=_prior.ReferenceOrganismV2
CONTEXT_SPLIT_SCHEMA=1

class ReferenceOrganismV2(_BaseReferenceOrganismV2):
    def __init__(self,population_spec=None):
        super().__init__(population_spec)
        self.recursive_metacontrol=ContextualMetacontrolV1.restore(self.recursive_metacontrol.checkpoint())
        self.recursive_causal_experiment=ContextualCausalExperimentV1.restore(self.recursive_causal_experiment.checkpoint())
        self.recursive_experiment_strategy=ContextualExperimentStrategyV1.restore(self.recursive_experiment_strategy.checkpoint())
        self.recursive_experiment_policy=ContextualExperimentPolicyV1.restore(self.recursive_experiment_policy.checkpoint())
        self.recursive_context_partition=RecursiveContextPartitionV2()
        self.recursive_partner_access=RecursivePartnerAccessV2()
        self._world_context_current=0;self._control_context_current=0
        self._pre_outcome_transition_receipts={};self._context_cache=None
        # Old latent-causal-regime state is checkpoint archaeology only from this layer on.
        self._causal_regime_current=0

    @staticmethod
    def _context_class(value):
        value=max(0,min(CTXQ,int(value)))
        return 0 if value<CTXQ//3 else (1 if value<2*CTXQ//3 else 2)

    @staticmethod
    def _context_token(tag,value):return _culture_id('typed-context-v2',[str(tag),int(value)])

    def _world_context_cues(self,state=None):
        world=tuple(self.world_state if state is None and self.world_state is not None else (state or ()))
        return tuple(self._context_token('world-feature',x) for x in world[:16])

    def _control_context_cues(self,world_context,source=0):
        source=int(source or getattr(self,'world_source',0) or getattr(self,'partner_source',0) or 1)
        resource,control=self._culture_body_envelope(source,0)
        cues=[self._context_token('world-context',int(world_context)),
              self._context_token('resource-class',self._context_class(resource)),
              self._context_token('control-class',self._context_class(control))]
        cues.extend(self._context_token('target-feature',x) for x in tuple(self.body_target or ())[:8])
        cues.extend(self._context_token('body-feature',x) for x in tuple(getattr(self,'body_state',()) or ())[:8])
        if getattr(self,'partner_present',False) and int(getattr(self,'partner_source',0))>0:
            cues.append(self._context_token('partner-present',1));cues.append(self._context_token('partner-source',int(self.partner_source)))
        return tuple(cues),int(resource),int(control)

    def _context_tick(self):
        return max(int(self.recursive_context_partition._tick),int(getattr(self,'tick_count',0)),int(getattr(self,'_developmental_curriculum_tick',0)))

    def _infer_contexts(self,state=None,source=0):
        world_cues=self._world_context_cues(state);tick=self._context_tick()
        world_context=self.recursive_context_partition.infer(WORLD_CONTEXT,world_cues,tick,self._world_context_current)
        control_cues,resource,control=self._control_context_cues(world_context,source)
        signature=(tuple(world_cues),tuple(control_cues),int(world_context),int(resource),int(control))
        if self._context_cache is not None and self._context_cache[0]==signature:
            return self._context_cache[1]
        control_context=self.recursive_context_partition.infer(CONTROL_CONTEXT,control_cues,tick,self._control_context_current)
        self._world_context_current=int(world_context);self._control_context_current=int(control_context)
        self._context_cache=(signature,(int(world_context),int(control_context),world_cues,control_cues,int(resource),int(control)))
        return self._context_cache[1]

    def _metacontrol_program_branches(self):
        # Start from body-free cultural evidence; add only WORLD_CONTEXT reason calibration.
        branches,rows,resource,control=_metacontrol_branch_base.ReferenceOrganismV2._metacontrol_program_branches(self)
        if not branches:return branches,rows,resource,control
        world_context,control_context,_wc,_cc,_r,_c=self._infer_contexts()
        reason_map=self._experiment_reason_map(rows);adjusted=[]
        for branch in branches:
            predictions=reason_map.get(int(branch.action),())
            signed,certainty=self.recursive_causal_experiment.reason_calibration_world(int(branch.action),predictions,world_context)
            if certainty:
                delta=(int(signed)*int(certainty))//(2*EXPQ)
                support=max(0,min(EXPQ,int(branch.support_q16)+delta))
                gap=max(0,int(branch.epistemic_gap_q16)-int(certainty)//3)
                branch=PolicyBranchV1(int(branch.action),support,int(branch.counterfactual_q16),int(branch.source_quality_q16),int(branch.consequence_risk_q16),gap,0)
            adjusted.append(branch)
        self._world_context_current=int(world_context);self._control_context_current=int(control_context)
        return tuple(adjusted),rows,resource,control

    def _eligible_policy_frontier(self,branches,reason_map,resource,control,world_context,control_context):
        safe=self.recursive_causal_experiment.eligible_world_probes(branches,reason_map,resource,control,world_context)
        rows=[]
        for candidate in safe:
            branch=next((b for b in branches if int(b.action)==int(candidate.action)),None)
            if branch is None:continue
            predictions=reason_map.get(int(candidate.action),())
            key=self.recursive_experiment_strategy.structural_key(branch,branches,len(predictions),control)
            if self.recursive_experiment_strategy.permits_control(key,control_context):rows.append(candidate)
        return tuple(rows)

    def _culture_program_nomination(self):
        branches,rows,resource,control=self._metacontrol_program_branches()
        if not branches:
            self._metacontrol_inquiry_alternatives=();self._metacontrol_inquiry_asked=False;self._metacontrol_pending_intervention=0;return None
        world_context=int(self._world_context_current);control_context=int(self._control_context_current)
        prior=self.recursive_metacontrol.decision(self._metacontrol_pending_decision);previous=int(prior.selected) if prior is not None else 0
        result=self.recursive_metacontrol.choose_contextual(branches,resource,control,self._metacontrol_social_quality(),
            int(self._developmental_curriculum_tick),previous,control_context)
        self._metacontrol_pending_decision=int(result.get('decision',0));mode=int(result.get('mode',MODE_OBSERVE));selected=int(result.get('selected',0))
        alternatives=tuple(map(int,result.get('alternatives',())));reason_map=self._experiment_reason_map(rows)
        if mode in (MODE_ASK,MODE_OBSERVE):
            frontier=self._eligible_policy_frontier(branches,reason_map,resource,control,world_context,control_context)
            if frontier:
                policy,policy_context=self.recursive_experiment_policy.choose_control_policy(frontier,resource,control,control_context)
                if policy and not self.recursive_experiment_policy.permits_control_policy(policy,policy_context,control_context):
                    alternate=POLICY_CONFIRM if policy==POLICY_DISCRIMINATE else POLICY_DISCRIMINATE
                    policy=alternate if self.recursive_experiment_policy.permits_control_policy(alternate,policy_context,control_context) else 0
                candidate=self.recursive_causal_experiment.policy_candidate(frontier,policy) if policy else None
                if candidate is not None:
                    probe=int(candidate.action);branch=next((b for b in branches if int(b.action)==probe),None);predictions=reason_map.get(probe,())
                    if branch is not None:
                        key=self.recursive_experiment_strategy.structural_key(branch,branches,len(predictions),control)
                        gain=self.recursive_causal_experiment.information_gain_world_q16(branch,branches,predictions,world_context)
                        before_certainty=self.recursive_causal_experiment.reason_calibration_world(probe,predictions,world_context)[1]
                        intervention=self.recursive_causal_experiment.begin_world_probe(probe,self._metacontrol_pending_decision,predictions,alternatives,gain,
                            int(self._developmental_curriculum_tick),world_context,True)
                        if intervention:
                            self.recursive_experiment_strategy.begin_control(intervention,key,before_certainty,True,control_context)
                            self.recursive_experiment_policy.begin_control_policy(intervention,policy,policy_context,before_certainty,True,control_context)
                            self._metacontrol_pending_intervention=int(intervention);self._metacontrol_inquiry_alternatives=();self._metacontrol_inquiry_asked=False
                            self._culture_active_program=();return probe
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

    def contact(self,kind,payload,source,authenticated=True,independent=True):
        payload=tuple(payload);kind=int(kind);source=int(source)
        before_shared=dict(getattr(self,'last_shared_episode_by_partner',{}))
        if kind==CONTACT_MOTOR_CONSEQUENCE and payload:
            ticket=int(payload[0]);motor=next((m for m in self.motor_actions if int(m.ticket)==ticket and not m.settled),None)
            if motor is not None:
                world_context,control_context,world_cues,_cc,_r,_c=self._infer_contexts(tuple(motor.state_before),source)
                prediction=self.cognition.transition(tuple(motor.state_before),int(motor.action_id),2)
                predicted_state=() if prediction is None else tuple(prediction.next_state)
                support=0 if prediction is None else int(prediction.support)
                self._pre_outcome_transition_receipts[ticket]=(int(world_context),int(control_context),tuple(world_cues),predicted_state,support)
        result=super().contact(kind,payload,source,authenticated,independent)
        if kind==CONTACT_WITHDRAW_SOURCE and len(payload)==1:
            self.recursive_partner_access.withdraw_source(int(payload[0]))
        if kind==CONTACT_CONSEQUENCE and bool(independent):
            for partner,episode in dict(getattr(self,'last_shared_episode_by_partner',{})).items():
                if int(before_shared.get(partner,0))==int(episode) or int(episode)<=0 or self.world_state is None:continue
                ep=self._episode_by_id.get(int(episode));world_context,_control_context,world_cues,_cc,_r,_c=self._infer_contexts()
                context=0 if ep is None else int(ep.context)
                self.recursive_partner_access.observe_shared(int(partner),int(episode),context,world_cues,world_context,self._context_tick(),True)
        return result

    def _culture_episode_from_motor(self,motor,source):
        if motor is None or not getattr(motor,'settled',False):
            return _causal_lifecycle.ReferenceOrganismV2._culture_episode_from_motor(self,motor,source)
        pending=int(self._metacontrol_pending_intervention);intervention=self.recursive_causal_experiment.intervention(pending) if pending else None
        ticket=int(getattr(motor,'ticket',0));action=int(getattr(motor,'action_id',0));success=int(getattr(motor,'effect',0))>0
        independent=bool(getattr(motor,'independent_consequence',False));state_before=tuple(getattr(motor,'state_before',()) or ())
        receipt=self._pre_outcome_transition_receipts.get(ticket)
        if receipt is None:
            world_context,control_context,world_cues,_cc,_r,_c=self._infer_contexts(state_before,source);predicted_state=();support=0
        else:
            world_context,control_context,world_cues,predicted_state,support=receipt
        identity=_causal_lifecycle.ReferenceOrganismV2._culture_episode_from_motor(self,motor,source)
        resolved_world=self.recursive_context_partition.resolve_world_after_transition(
            world_context,world_cues,predicted_state,tuple(getattr(motor,'state_after',()) or ()),support,independent,self._context_tick())
        self._world_context_current=int(resolved_world);self._context_cache=None
        # CONTROL_CONTEXT is a performance/control memory index; world transition mismatch
        # cannot itself relabel body/control history. Re-resolve only to bind the resolved world context.
        _wc,resolved_control,_wcu,_cc,_r,_c=self._infer_contexts(state_before,source)
        self._control_context_current=int(resolved_control)
        self.recursive_metacontrol.record_self_outcome(action,success,independent,resolved_control)
        if intervention is not None:
            self.recursive_causal_experiment.settle_world_context_evidence(pending,resolved_world)
            after_certainty=self.recursive_causal_experiment.reason_calibration_world(action,intervention.reasons,resolved_world)[1]
            comparable=int(resolved_world)==int(world_context)
            self.recursive_experiment_strategy.settle_control(pending,after_certainty,independent,comparable,control_context)
            self.recursive_experiment_policy.settle_control_policy(pending,after_certainty,independent,comparable,control_context)
        self._pre_outcome_transition_receipts.pop(ticket,None)
        return identity

    # Historical/global compatibility summaries.
    def reason_predictive_reliability_q16(self,reason,source,action,regime=0):
        return self.recursive_causal_experiment.reason_reliability_world_q16(reason,source,action,int(regime))
    def self_reliability_q16(self,action,regime=0):
        return self.recursive_metacontrol.self_reliability_q16(action,int(regime))

    # Current typed authority surfaces.
    def contextual_reason_predictive_reliability_q16(self,reason,source,action,world_context=None):
        resolved=int(self._world_context_current if world_context is None else world_context)
        return self.recursive_causal_experiment.reason_reliability_world_q16(reason,source,action,resolved)
    def contextual_self_reliability_q16(self,action,control_context=None):
        resolved=int(self._control_context_current if control_context is None else control_context)
        return self.recursive_metacontrol.self_reliability_q16(action,resolved)
    def partner_shared_access_staleness_q16(self,partner):
        world_context,_control_context,world_cues,_cc,_r,_c=self._infer_contexts()
        return self.recursive_partner_access.shared_access_staleness_q16(int(partner),world_cues,world_context)
    def partner_access_applicability_q16(self,partner):
        world_context,_control_context,world_cues,_cc,_r,_c=self._infer_contexts()
        return self.recursive_partner_access.access_applicability_q16(int(partner),world_cues,world_context)

    def checkpoint(self):
        data=super().checkpoint();data['typed_context_partition_v2']={
            'schema':CONTEXT_SPLIT_SCHEMA,'state':self.recursive_context_partition.checkpoint(),
            'world_current':int(self._world_context_current),'control_current':int(self._control_context_current),
            'partner_access':self.recursive_partner_access.checkpoint()};return data

    @classmethod
    def restore(cls,data):
        out=super().restore(data)
        m=data.get('recursive_metacontrol_v1',{}).get('state');c=data.get('recursive_causal_experiment_v1',{}).get('state')
        s=data.get('recursive_experiment_strategy_v1',{}).get('state');p=data.get('recursive_experiment_policy_v2',{}).get('state')
        if m is not None:out.recursive_metacontrol=ContextualMetacontrolV1.restore(m)
        if c is not None:out.recursive_causal_experiment=ContextualCausalExperimentV1.restore(c)
        if s is not None:out.recursive_experiment_strategy=ContextualExperimentStrategyV1.restore(s)
        if p is not None:out.recursive_experiment_policy=ContextualExperimentPolicyV1.restore(p)
        row=data.get('typed_context_partition_v2')
        if row is None:
            out.recursive_context_partition=RecursiveContextPartitionV2();out.recursive_partner_access=RecursivePartnerAccessV2()
            out._world_context_current=0;out._control_context_current=0
        else:
            if int(row.get('schema',0))!=CONTEXT_SPLIT_SCHEMA:raise ValueError('organism:typed-context-checkpoint')
            out.recursive_context_partition=RecursiveContextPartitionV2.restore(row['state'])
            out.recursive_partner_access=RecursivePartnerAccessV2.restore(row.get('partner_access',{'schema':2}))
            out._world_context_current=int(row.get('world_current',0));out._control_context_current=int(row.get('control_current',0))
        out._pre_outcome_transition_receipts={};out._context_cache=None;out._causal_regime_current=0
        return out
