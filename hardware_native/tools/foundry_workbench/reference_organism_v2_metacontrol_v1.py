#!/usr/bin/env python3
"""Withdrawal-safe cultural Adult plus embodied recursive policy metacontrol.

The incumbent organism/culture stack remains the causal owner of world, language, action and
consequence evidence. RecursivePolicyMetacontrolV1 adds a resident commitment boundary:
constructive futures and cultural reasons may nominate zero-authority branches, while the
Adult learns from independent consequences whether to ACT, ASK, OBSERVE, DEFER or REVISE.
"""
from __future__ import annotations

import reference_organism_v2_execution_v3 as _prior
for _name in dir(_prior):
    if not _name.startswith('__'):
        globals()[_name]=getattr(_prior,_name)

from reference_recursive_policy_metacontrol_v1 import (
    RecursivePolicyMetacontrolV1,PolicyBranchV1,
    MODE_ACT,MODE_ASK,MODE_OBSERVE,MODE_DEFER,MODE_REVISE,Q as MCQ,
)

_BaseReferenceOrganismV2=_prior.ReferenceOrganismV2
METACONTROL_SCHEMA=1

class ReferenceOrganismV2(_BaseReferenceOrganismV2):
    def __init__(self,population_spec=None):
        super().__init__(population_spec)
        self.recursive_metacontrol=RecursivePolicyMetacontrolV1()
        self._metacontrol_pending_decision=0
        self._metacontrol_inquiry_alternatives=()
        self._metacontrol_inquiry_asked=False

    def _culture_abort_withdrawn_execution(self,withdrawn_source):
        withdrawn_source=int(withdrawn_source)
        if (self._culture_active_program and self._culture_execution_source>0
                and int(self._culture_execution_source)==withdrawn_source):
            self._culture_active_program=();self._culture_reset_execution_evidence();return True
        return False

    def contact(self,kind,payload,source,authenticated=True,independent=True):
        payload=tuple(payload)
        if int(kind)==CONTACT_WITHDRAW_SOURCE and len(payload)==1:self._culture_abort_withdrawn_execution(int(payload[0]))
        return super().contact(kind,payload,source,authenticated,independent)

    def _metacontrol_social_quality(self):
        if not self.partner_present or int(self.partner_source)<=0:return 0
        credibility=int(self.developmental_source_credibility_q16(int(self.partner_source)))
        return max(0,min(MCQ,(MCQ+credibility)//2))

    def _metacontrol_program_branches(self):
        if not self.affordances:return (),(),0,0
        context=self._culture_context();resource,control=self._culture_body_envelope()
        rows=self.recursive_self_culture.program_candidates(context,resource,control,self.developmental_source_credibility_q16)
        if not rows:return (),(),resource,control
        futures=self.constructive_self_futures(128);branches=[]
        for row in rows[:16]:
            if not row.get('actions'):continue
            action=int(row['actions'][0])
            if action not in self.affordances:continue
            relevant=[future for future in futures if future.actions and int(future.actions[0])==action]
            positive=sum(1 for future in relevant if future.predicted_outcomes and int(future.predicted_outcomes[0])==3)
            negative=sum(1 for future in relevant if future.predicted_outcomes and int(future.predicted_outcomes[0])==1)
            count=positive+negative
            counterfactual=(MCQ//2 if not count else max(0,min(MCQ,MCQ//2+((positive-negative)*MCQ)//(2*count))))
            risk=(0 if not count else (negative*MCQ)//count)
            supporters=tuple(sorted(set((*row.get('teachers',()),*row.get('confirmations',())))))
            if supporters:
                credibility=sum(int(self.developmental_source_credibility_q16(int(src))) for src in supporters)//len(supporters)
                source_quality=max(0,min(MCQ,(MCQ+credibility)//2))
            else:source_quality=MCQ//2
            confirmations=len(row.get('confirmations',()))
            program=next((p for p in self.recursive_self_culture._programs if int(p.identity)==int(row['identity'])),None)
            reason=0
            if program is not None:
                if hasattr(self.recursive_self_culture,'_program_reason_support'):
                    reason=int(self.recursive_self_culture._program_reason_support(program,self.developmental_source_credibility_q16))
                else:
                    reason=int(self.recursive_self_culture.reason_support(program.reason_ids,0))
            support=max(0,min(MCQ,MCQ//2+reason//4+min(MCQ//3,confirmations*(MCQ//6))))
            epistemic_gap=max(0,MCQ-min(MCQ,confirmations*(MCQ//3)+count*(MCQ//8)))
            branches.append(PolicyBranchV1(action,support,counterfactual,source_quality,risk,epistemic_gap,0))
        return tuple(branches),rows,resource,control

    def _culture_program_nomination(self):
        branches,rows,resource,control=self._metacontrol_program_branches()
        if not branches:
            self._metacontrol_inquiry_alternatives=();self._metacontrol_inquiry_asked=False;return None
        prior=self.recursive_metacontrol.decision(self._metacontrol_pending_decision)
        previous=int(prior.selected) if prior is not None else 0
        result=self.recursive_metacontrol.choose(branches,resource,control,self._metacontrol_social_quality(),int(self._developmental_curriculum_tick),previous)
        self._metacontrol_pending_decision=int(result.get('decision',0));mode=int(result.get('mode',MODE_OBSERVE));selected=int(result.get('selected',0))
        alternatives=tuple(map(int,result.get('alternatives',())))
        if mode==MODE_ASK:
            if alternatives!=self._metacontrol_inquiry_alternatives:self._metacontrol_inquiry_asked=False
            self._metacontrol_inquiry_alternatives=alternatives;return None
        self._metacontrol_inquiry_alternatives=();self._metacontrol_inquiry_asked=False
        if mode in (MODE_OBSERVE,MODE_DEFER) or selected<=0:return None
        chosen=next((row for row in rows if row.get('actions') and int(row['actions'][0])==selected),None)
        if chosen is None:return None
        self._culture_active_program=(int(chosen['identity']),0);return selected

    def _cognitive_tick(self):
        action=super()._cognitive_tick()
        if action is not None:return action
        if self._metacontrol_inquiry_alternatives and not self._metacontrol_inquiry_asked:
            inquiry=self._emit_information_request(self._metacontrol_inquiry_alternatives)
            if inquiry is not None:self._metacontrol_inquiry_asked=True;return inquiry
        return None

    def _culture_episode_from_motor(self,motor,source):
        identity=super()._culture_episode_from_motor(motor,source)
        if motor is None or not getattr(motor,'settled',False):return identity
        decision=self.recursive_metacontrol.decision(self._metacontrol_pending_decision)
        if decision is not None and int(decision.selected)==int(getattr(motor,'action_id',0)):
            self.recursive_metacontrol.record_outcome(int(motor.action_id),int(getattr(motor,'effect',0))>0,bool(getattr(motor,'independent_consequence',False)))
            self._metacontrol_pending_decision=0
        return identity

    def checkpoint(self):
        data=super().checkpoint();data['recursive_metacontrol_v1']={'schema':METACONTROL_SCHEMA,'state':self.recursive_metacontrol.checkpoint(),
            'pending_decision':int(self._metacontrol_pending_decision),'inquiry_alternatives':list(map(int,self._metacontrol_inquiry_alternatives)),
            'inquiry_asked':bool(self._metacontrol_inquiry_asked)};return data

    @classmethod
    def restore(cls,data):
        out=super().restore(data);row=data.get('recursive_metacontrol_v1')
        if row is None:return out
        if int(row.get('schema',0))!=METACONTROL_SCHEMA:raise ValueError('organism:metacontrol-checkpoint')
        out.recursive_metacontrol=RecursivePolicyMetacontrolV1.restore(row['state']);out._metacontrol_pending_decision=int(row.get('pending_decision',0))
        out._metacontrol_inquiry_alternatives=tuple(map(int,row.get('inquiry_alternatives',())));out._metacontrol_inquiry_asked=bool(row.get('inquiry_asked',False));return out

    def _run_recursive_policy_metacontrol_ratchet(self):
        owner=RecursivePolicyMetacontrolV1();A=0xC101;B=0xC202;tick=1
        strong=PolicyBranchV1(A,3*MCQ//4,3*MCQ//4,3*MCQ//4,0,MCQ//8,0);weak=PolicyBranchV1(B,MCQ//3,MCQ//3,MCQ//2,MCQ//4,MCQ//2,0)
        before=owner.outcome_count;act=owner.choose((strong,weak),MCQ,MCQ,MCQ//2,tick);tick+=1
        low=owner.choose((strong,weak),MCQ//8,MCQ//8,MCQ//2,tick);tick+=1
        ask_a=PolicyBranchV1(A,MCQ//2,MCQ//2,MCQ//2,0,3*MCQ//4,0);ask_b=PolicyBranchV1(B,MCQ//2,MCQ//2,MCQ//2,0,3*MCQ//4,0)
        ask=owner.choose((ask_a,ask_b),MCQ,MCQ,MCQ,tick);tick+=1;after_choices=owner.outcome_count
        owner.record_outcome(A,True,False);after_nonindependent=owner.outcome_count
        owner.record_outcome(A,True,True);owner.record_outcome(A,True,True);owner.record_outcome(B,False,True)
        rev_a=PolicyBranchV1(A,MCQ//2,MCQ//2,MCQ//2,0,0,0);rev_b=PolicyBranchV1(B,MCQ//2,MCQ//2,MCQ//2,0,0,0)
        calibrated=owner.choose((rev_a,rev_b),MCQ,MCQ,0,tick,previous_action=B);tick+=1
        checks={
          'counterfactual_mode_selection_creates_no_evidence':before==0 and after_choices==before,
          'strong_calibrated_branch_commits_to_act':int(act.get('mode',0))==MODE_ACT and int(act.get('selected',0))==A,
          'same_evidence_defers_under_low_resource_control':int(low.get('mode',0))==MODE_DEFER and int(low.get('selected',0))==0,
          'high_social_quality_turns_high_gap_ambiguity_into_inquiry':int(ask.get('mode',0))==MODE_ASK and int(ask.get('selected',0))==0,
          'nonindependent_consequence_cannot_calibrate_self_model':after_nonindependent==before,
          'independent_lived_outcomes_calibrate_action_reliability':owner.reliability_q16(A)>owner.reliability_q16(B),
          'calibrated_low_gap_evidence_can_change_course_without_feedback':int(calibrated.get('mode',0))==MODE_REVISE and int(calibrated.get('selected',0))==A,
          'imagined_and_social_branches_remain_authority_zero':strong.authority==weak.authority==ask_a.authority==ask_b.authority==rev_a.authority==rev_b.authority==0,
        }
        cp=owner.checkpoint();restored=RecursivePolicyMetacontrolV1.restore(cp);checks['metacontrol_calibration_survives_checkpoint']=restored.checkpoint()==cp
        return checks,sorted(name for name,value in checks.items() if not value)

    def _run_recursive_self_culture_ratchet(self):
        checks,local_failed,future_count=super()._run_recursive_self_culture_ratchet();checks=dict(checks)
        self._culture_active_program=(0xC011,1);self._culture_execution_source=9201;self._culture_execution_all_independent=True
        aborted=self._culture_abort_withdrawn_execution(9201)
        checks['withdrawing_inflight_consequence_source_aborts_execution']=(aborted and not self._culture_active_program and self._culture_execution_source==0 and self._culture_execution_all_independent)
        return checks,sorted(k for k,v in checks.items() if not v),future_count

    def run_frontier_multilingual_curriculum(self):
        receipt=super().run_frontier_multilingual_curriculum();checks,failed=self._run_recursive_policy_metacontrol_ratchet();merged=dict(receipt.get('checks',{}));merged.update(checks)
        all_failed=sorted(k for k,v in merged.items() if not v);receipt=dict(receipt);receipt['checks']=merged;receipt['failed']=all_failed;receipt['status']='GREEN' if not all_failed else 'RED'
        receipt['recursive_metacontrol_decisions']=self.recursive_metacontrol.decision_count;receipt['recursive_metacontrol_outcomes']=self.recursive_metacontrol.outcome_count
        receipt['metacontrol_modes']=['ACT','ASK','OBSERVE','DEFER','REVISE'];return receipt
