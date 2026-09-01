#!/usr/bin/env python3
"""Public reference Adult over latent causal-regime inference.

The prospective/source-prediction authority is preserved exactly in
reference_organism_v2_prospective_execution_v1.py. This surface keeps historical unqualified
reliability accessors backward-compatible while making contextual competence explicit.
Internal deliberation, memory routing and experiment control remain regime-aware.
"""
from __future__ import annotations

import reference_organism_v2_prospective_execution_v1 as _prior
for _name in dir(_prior):
    if not _name.startswith('__') and _name not in {'_prior','_BaseReferenceOrganismV2'}:
        globals()[_name]=getattr(_prior,_name)

from reference_joint_plan_control_v1 import (JointPlanControlV1,ROLE_SELF,ROLE_PARTNER,STATUS_ACTIVE,STATUS_WAITING_INFO,STATUS_BLOCKED)
from reference_open_joint_role_affordance_v1 import ROLE_SPEAKER_ATOM

_BaseReferenceOrganismV2=_prior.ReferenceOrganismV2
JOINT_PLAN_SCHEMA=1

class ReferenceOrganismV2(_BaseReferenceOrganismV2):
    def __init__(self,population_spec=None):
        super().__init__(population_spec);self.joint_plan_control=JointPlanControlV1();self._joint_ack_ticket=0

    def begin_joint_plan(self,steps,source):
        live={int(row.identity) for row in self.recursive_self_culture._reasons
              if int(row.source) not in self.recursive_self_culture._withdrawn_sources}
        return self.joint_plan_control.begin(self._culture_context(),int(source),tuple(steps),
            int(self._developmental_curriculum_tick),self._known_cultural_actions(),live)

    def joint_plan_set_alternatives(self,actions):
        return self.joint_plan_control.set_alternatives(tuple(actions),int(self._developmental_curriculum_tick))

    def resolve_joint_plan_alternative(self,action,reason_ids,language_factors):
        return self.joint_plan_control.resolve_alternative(int(action),tuple(reason_ids),tuple(language_factors),int(self._developmental_curriculum_tick))

    def observe_joint_partner_action(self,actor,action,success,independent=True):
        return self.joint_plan_control.observe_partner_action(int(actor),int(action),bool(success),bool(independent),int(self._developmental_curriculum_tick))

    def _joint_step_teacher(self,step,plan):
        for reason_id in step.reason_ids:
            row=next((x for x in self.recursive_self_culture._reasons if int(x.identity)==int(reason_id)),None)
            if row is not None and int(row.source) not in self.recursive_self_culture._withdrawn_sources:return int(row.source)
        return int(plan.source)

    def _joint_prepare_self_program(self,step,plan):
        active=tuple(self._culture_active_program)
        if active:
            program=next((x for x in self.recursive_self_culture._programs if int(x.identity)==int(active[0])),None)
            if program is not None and active[1]<len(program.actions) and int(program.actions[active[1]])==int(step.action):return int(program.identity)
        teacher=self._joint_step_teacher(step,plan)
        program=self.compose_cultural_instruction((int(step.action),),teacher,tuple(step.reason_ids),joint_context=self._culture_context())
        return int(program or 0)

    def stage_joint_acknowledgement_surface(self,surface):
        plan=self.joint_plan_control.plan();step=self.joint_plan_control.current_step();surface=tuple(map(int,surface))
        if plan is None or step is None or step.role!=ROLE_SELF or not plan.acknowledgement_pending or not surface:return None
        source=int(self.partner_source if self.partner_present and self.partner_source>0 else self.world_source);channel=int(self.partner_channel if self.partner_present and self.partner_channel>0 else self.communication_channel)
        if source<=0 or channel<=0:return None
        occ=self.population.recruit((0xA11CEAC1,int(step.acknowledgement_factor),int(step.action)));sid=self.next_scene;self.next_scene+=1
        scene=SceneStateV2(sid,channel,int(step.acknowledgement_factor),(int(step.action),),source,occ.identity,True,True);self.pending_scenes.append(scene);self._scene_by_id[int(sid)]=scene
        action=ActionV2(self.next_ticket,self.tick_count,channel,source,surface,occ.identity,sid,int(step.acknowledgement_factor),
            (sid,occ.identity),False,0,surface,False,int(step.acknowledgement_factor),(),(),0,0)
        self._ensure_action_capacity();self.next_ticket+=1;self.actions.append(action);self._index_action(action);self._action_commitments[action.ticket]=self._action_commitment(action);self._joint_ack_ticket=int(action.ticket);return action

    def settle_joint_acknowledgement(self,ticket):
        if int(ticket)<=0 or int(ticket)!=int(self._joint_ack_ticket):return False
        action=next((x for x in self.actions if int(x.ticket)==int(ticket)),None)
        if action is None:return False
        action.settled=True;action.effect=1;self._joint_ack_ticket=0
        return self.joint_plan_control.mark_acknowledged(int(self._developmental_curriculum_tick))

    def _cognitive_tick(self):
        plan=self.joint_plan_control.plan();step=self.joint_plan_control.current_step()
        if plan is not None and step is not None:
            if int(plan.status)==STATUS_WAITING_INFO:
                if self.partner_present and self.partner_source>0:
                    return self._emit_information_request(plan.alternatives)
                return None
            if int(plan.status)==STATUS_BLOCKED:return None
            if int(step.role)==ROLE_PARTNER:return None
            if int(step.role)==ROLE_SELF:
                if bool(plan.acknowledgement_pending):return None
                self._joint_prepare_self_program(step,plan)
        return super()._cognitive_tick()

    def _culture_episode_from_motor(self,motor,source):
        identity=super()._culture_episode_from_motor(motor,source)
        if motor is not None and getattr(motor,'settled',False):
            self.joint_plan_control.settle_self_action(int(getattr(motor,'action_id',0)),int(getattr(motor,'effect',0))>0,
                bool(getattr(motor,'independent_consequence',False)),int(self._developmental_curriculum_tick))
        return identity

    def checkpoint(self):
        data=super().checkpoint();data['joint_plan_control_v1']={'schema':JOINT_PLAN_SCHEMA,'state':self.joint_plan_control.checkpoint(),'ack_ticket':int(self._joint_ack_ticket)};return data

    @classmethod
    def restore(cls,data):
        out=super().restore(data);out.__class__=cls;row=data.get('joint_plan_control_v1')
        if row is None:
            out.joint_plan_control=JointPlanControlV1();out._joint_ack_ticket=0;return out
        if int(row.get('schema',0))!=JOINT_PLAN_SCHEMA:raise ValueError('organism:joint-plan-checkpoint')
        out.joint_plan_control=JointPlanControlV1.restore(row['state']);out._joint_ack_ticket=int(row.get('ack_ticket',0));return out

    def reason_predictive_reliability_q16(self,reason,source,action,regime=0):
        """Compatibility summary unless a regime is explicitly requested."""
        return self.recursive_causal_experiment.reason_reliability_q16(
            int(reason),int(source),int(action),int(regime))

    def contextual_reason_predictive_reliability_q16(self,reason,source,action,regime=None):
        resolved=int(self._causal_regime_current if regime is None else regime)
        return self.recursive_causal_experiment.reason_reliability_q16(
            int(reason),int(source),int(action),resolved)

    def self_reliability_q16(self,action,regime=0):
        """Compatibility summary unless a regime is explicitly requested."""
        return self.recursive_metacontrol.reliability_q16(int(action),int(regime))

    def contextual_self_reliability_q16(self,action,regime=None):
        resolved=int(self._causal_regime_current if regime is None else regime)
        return self.recursive_metacontrol.reliability_q16(int(action),resolved)
