#!/usr/bin/env python3
"""Live Adult with consequence categories separated after every settled motor action.

The prior typed-branch Adult is preserved in `reference_organism_v2_outcome_semantics_v1.py`.
This layer removes `effect > 0 == successful action` from live credit. ACTION_CONTROL_MATCH,
GOAL_ATTAINMENT, PROCEDURE_EXECUTION, and OUTCOME_VALENCE are derived separately.
"""
from __future__ import annotations

import reference_organism_v2_outcome_semantics_v1 as _prior
for _name in dir(_prior):
    if not _name.startswith('__') and _name not in {'_prior','_BaseReferenceOrganismV2'}:
        globals()[_name]=getattr(_prior,_name)

from reference_action_outcome_decomposition_v1 import decompose_action_outcome

_BaseReferenceOrganismV2=_prior.ReferenceOrganismV2
OUTCOME_CATEGORY_SCHEMA=1

class ReferenceOrganismV2(_BaseReferenceOrganismV2):
    def __init__(self,population_spec=None):
        super().__init__(population_spec)
        self._pre_outcome_goal_receipts={}
        self._last_action_outcome_decomposition=None

    def contact(self,kind,payload,source,authenticated=True,independent=True):
        payload=tuple(payload);kind=int(kind)
        if kind==CONTACT_MOTOR_CONSEQUENCE and payload:
            ticket=int(payload[0])
            motor=next((row for row in self.motor_actions if int(row.ticket)==ticket and not row.settled),None)
            if motor is not None:
                # Goal is captured before settlement just like the transition receipt. A later
                # target change cannot rewrite what this action was being evaluated against.
                self._pre_outcome_goal_receipts[ticket]=tuple(self.body_target or ())
        return super().contact(kind,payload,source,authenticated,independent)

    def _outcome_receipt(self,motor,predicted_state,prediction_support,target):
        return decompose_action_outcome(
            motor,predicted_state,prediction_support,target,
            satisfies=self.cognition.satisfies)

    def last_action_outcome_decomposition(self):return self._last_action_outcome_decomposition

    def recommendation_positive_consequence_rate_q16(self,reason,source,action,world_context=0):
        """Pragmatic history only: following recommendation -> positive vs non-positive effect."""
        return self.recursive_causal_experiment.recommendation_reliability_world_q16(
            int(reason),int(source),int(action),int(world_context))

    # Compatibility name retained but explicitly routed to pragmatic consequence history.
    def recommendation_outcome_reliability_q16(self,reason,source,action,world_context=0):
        return self.recommendation_positive_consequence_rate_q16(reason,source,action,world_context)

    def _admit_self_episode_and_execution(self,motor,source):
        """Store valence episode and advance instructed execution independent of reward sign."""
        state_before=tuple(getattr(motor,'state_before',()) or ())
        context=self._culture_context(state_before);action=int(getattr(motor,'action_id',0))
        valence=1 if int(getattr(motor,'effect',0))>0 else (-1 if int(getattr(motor,'effect',0))<0 else 0)
        # SelfCulture's historical 1/2/3 outcome field is valence here, never action success.
        outcome=3 if valence>0 else (1 if valence<0 else 2)
        resource,control=self._culture_body_envelope(source,action)
        social=int(self.partner_source if self.partner_present else 0)
        independent=bool(getattr(motor,'independent_consequence',False))
        tick=self._developmental_advance(max(1,int(self._developmental_stage)),'self-episode-outcome-separated')
        identity=self.recursive_self_culture.observe_episode(
            context,action,outcome,int(source),tick,resource,control,social,independent)

        active=tuple(self._culture_active_program)
        if not active:
            self._culture_reset_execution_evidence();return identity,tick
        pid,cursor=map(int,active)
        program=next((row for row in self.recursive_self_culture._programs if int(row.identity)==pid),None)
        if program is None or cursor>=len(program.actions) or int(program.actions[cursor])!=action:
            self._culture_active_program=();self._culture_reset_execution_evidence();return identity,tick
        self._culture_accumulate_execution_evidence(int(source),independent)
        # A settled expected action is an executed step even when the consequence is aversive.
        if cursor+1<len(program.actions):
            self._culture_active_program=(pid,cursor+1);return identity,tick
        execution_source=int(self._culture_execution_source)
        qualified=bool(self._culture_execution_all_independent and execution_source>0)
        self.recursive_self_culture.confirm_program_execution(pid,execution_source,tick,qualified)
        self._culture_active_program=();self._culture_reset_execution_evidence();return identity,tick

    def _culture_episode_from_motor(self,motor,source):
        if motor is None or not getattr(motor,'settled',False):return 0
        pending=int(self._metacontrol_pending_intervention)
        intervention=self.recursive_causal_experiment.intervention(pending) if pending else None
        ticket=int(getattr(motor,'ticket',0));action=int(getattr(motor,'action_id',0))
        state_before=tuple(getattr(motor,'state_before',()) or ())
        transition=self._pre_outcome_transition_receipts.get(ticket)
        if transition is None:
            world_context,control_context,world_cues,_cc,_r,_c=self._infer_contexts(state_before,source)
            predicted_state=();prediction_support=0
        else:
            world_context,control_context,world_cues,predicted_state,prediction_support=transition
        target=self._pre_outcome_goal_receipts.get(ticket,())
        receipt=self._outcome_receipt(motor,predicted_state,prediction_support,target)
        self._last_action_outcome_decomposition=receipt

        identity,_tick=self._admit_self_episode_and_execution(motor,source)

        # Own-action competence is action-outcome control, never valence or goal luck.
        if receipt.independent and receipt.control_evaluable:
            self.recursive_metacontrol.record_outcome(action,receipt.control_match,True)

        # Recommendation history is explicitly pragmatic consequence association. It does not
        # calibrate proposition truth, action control, or goal attainment.
        if intervention is not None:
            settle_tick=max(int(self.recursive_causal_experiment._tick),int(getattr(self,'_developmental_curriculum_tick',0)))
            positive_consequence=receipt.valence>0
            self.recursive_causal_experiment.settle(pending,positive_consequence,receipt.independent,settle_tick)
            self._metacontrol_pending_intervention=0;self._metacontrol_pending_decision=0

        # External context routing still depends only on supported pre-outcome world prediction.
        resolved_world=self.recursive_context_partition.resolve_world_after_transition(
            int(world_context),world_cues,predicted_state,receipt.actual_state,
            int(prediction_support),receipt.independent,self._context_tick())
        self._world_context_current=int(resolved_world);self._context_cache=None
        _wc,resolved_control,_wcu,_cc,_r,_c=self._infer_contexts(state_before,source)
        self._control_context_current=int(resolved_control)
        if receipt.independent and receipt.control_evaluable:
            self.recursive_metacontrol.record_self_outcome(action,receipt.control_match,True,resolved_control)

        if intervention is not None:
            self.recursive_causal_experiment.settle_world_context_evidence(pending,resolved_world)
            after_certainty=self.recursive_causal_experiment.recommendation_calibration_world(
                action,intervention.reasons,resolved_world)[1]
            comparable=int(resolved_world)==int(world_context)
            self.recursive_experiment_strategy.settle_control(
                pending,after_certainty,receipt.independent,comparable,control_context)
            self.recursive_experiment_policy.settle_control_policy(
                pending,after_certainty,receipt.independent,comparable,control_context)

        self._pre_outcome_transition_receipts.pop(ticket,None)
        self._pre_outcome_goal_receipts.pop(ticket,None)
        return identity

    def checkpoint(self):
        data=super().checkpoint()
        # Decomposition and pre-outcome targets are transient action-local computation. Pending
        # motor actions already carry the durable action identity; do not serialize a shadow log.
        data['outcome_category_v1']={'schema':OUTCOME_CATEGORY_SCHEMA}
        return data

    @classmethod
    def restore(cls,data):
        out=super().restore(data);row=data.get('outcome_category_v1')
        if row is not None and int(row.get('schema',0))!=OUTCOME_CATEGORY_SCHEMA:
            raise ValueError('organism:outcome-category-checkpoint')
        out._pre_outcome_goal_receipts={};out._last_action_outcome_decomposition=None
        return out
