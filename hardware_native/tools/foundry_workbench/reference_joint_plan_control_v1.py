#!/usr/bin/env python3
"""Role- and subgoal-qualified joint plans over zero-authority cultural programs.

A joint plan is not a new truth owner.  It records who is expected to perform each already-known
action, which learned reason cohort supports that local subgoal, and which later steps depend on
its completion.  Partner-role observations and self motor consequences are the only events that
advance a cursor.  Alternatives can make one step unresolved; clarification changes intended
coordination, not world evidence.
"""
from __future__ import annotations
from dataclasses import dataclass
import hashlib,json

ROLE_SELF=1
ROLE_PARTNER=2
STATUS_ACTIVE=1
STATUS_WAITING_INFO=2
STATUS_BLOCKED=3
STATUS_COMPLETE=4
MAX_JOINT_PLANS=512
MAX_STEPS=16
MAX_ALTERNATIVES=8


def _id(tag,payload):
    raw=json.dumps([tag,payload],sort_keys=True,separators=(',',':')).encode()
    return int(hashlib.sha256(raw).hexdigest()[:16],16) or 1


@dataclass(frozen=True)
class JointPlanStepV1:
    identity:int
    subgoal:int
    role:int
    actor:int
    action:int
    reason_ids:tuple[int,...]
    language_factors:tuple[int,...]
    acknowledgement_factor:int
    depends_on:tuple[int,...]


@dataclass(frozen=True)
class JointPlanV1:
    identity:int
    context:int
    source:int
    steps:tuple[JointPlanStepV1,...]
    cursor:int
    status:int
    alternatives:tuple[int,...]
    revision:int
    acknowledgement_pending:bool
    authority:int=0


class JointPlanControlV1:
    def __init__(self):self._plans=[];self.active_plan=0;self._tick=-1

    def _advance(self,tick):
        tick=int(tick)
        if tick<self._tick:raise ValueError('joint-plan:time-reversal')
        self._tick=tick

    def _replace(self,row):
        prior=next((x for x in self._plans if int(x.identity)==int(row.identity)),None)
        if prior is None:
            if len(self._plans)>=MAX_JOINT_PLANS:self._plans.pop(0)
            self._plans.append(row)
        else:self._plans[self._plans.index(prior)]=row
        return row

    def plan(self,identity=0):
        identity=int(identity or self.active_plan)
        return next((x for x in self._plans if int(x.identity)==identity),None)

    def current_step(self):
        row=self.plan()
        if row is None or row.status==STATUS_COMPLETE or row.cursor>=len(row.steps):return None
        return row.steps[row.cursor]

    def begin(self,context,source,steps,tick,known_actions=(),live_reasons=()):
        self._advance(tick);context=int(context);source=int(source);known=set(map(int,known_actions));live=set(map(int,live_reasons))
        if min(context,source)<=0 or not steps or len(steps)>MAX_STEPS:return 0
        built=[]
        for index,raw in enumerate(steps):
            if len(raw)!=8:return 0
            subgoal,role,actor,action,reasons,factors,ack_factor,depends=raw
            subgoal=int(subgoal);role=int(role);actor=int(actor);action=int(action);factors=tuple(sorted(set(map(int,factors))));ack_factor=int(ack_factor)
            reasons=tuple(sorted(set(map(int,reasons))));depends=tuple(sorted(set(map(int,depends))))
            if min(subgoal,action)<=0 or not factors or min(factors)<=0 or role not in (ROLE_SELF,ROLE_PARTNER) or action not in known:return 0
            if role==ROLE_SELF and ack_factor<=0:return 0
            if role==ROLE_PARTNER:ack_factor=0
            if role==ROLE_SELF and actor!=0:return 0
            if role==ROLE_PARTNER and actor<=0:return 0
            if any(reason not in live for reason in reasons):return 0
            if any(dep<0 or dep>=index for dep in depends):return 0
            identity=_id('joint-plan-step-v1',[context,index,subgoal,role,actor,action,list(reasons),list(factors),ack_factor,list(depends)])
            built.append(JointPlanStepV1(identity,subgoal,role,actor,action,reasons,factors,ack_factor,depends))
        identity=_id('joint-plan-v1',[context,source,[x.identity for x in built],int(tick)])
        row=JointPlanV1(identity,context,source,tuple(built),0,STATUS_ACTIVE,(),0,bool(built[0].role==ROLE_SELF),0)
        self._replace(row);self.active_plan=identity;return identity

    def set_alternatives(self,actions,tick):
        self._advance(tick);row=self.plan();step=self.current_step();actions=tuple(sorted(set(map(int,actions))))
        if row is None or step is None or len(actions)<2 or len(actions)>MAX_ALTERNATIVES or step.action not in actions:return False
        revised=JointPlanV1(row.identity,row.context,row.source,row.steps,row.cursor,STATUS_WAITING_INFO,actions,row.revision,False,0)
        self._replace(revised);return True

    def resolve_alternative(self,action,reason_ids,language_factors,tick):
        """Clarify intended current action only; this does not assert that it will succeed."""
        self._advance(tick);row=self.plan();step=self.current_step();action=int(action);factors=tuple(sorted(set(map(int,language_factors))))
        if row is None or step is None or row.status!=STATUS_WAITING_INFO or action not in row.alternatives or not factors or min(factors)<=0:return False
        reasons=tuple(sorted(set(map(int,reason_ids))))
        replacement=JointPlanStepV1(_id('joint-plan-step-revision-v1',[step.identity,action,list(reasons),list(factors),row.revision+1]),
            step.subgoal,step.role,step.actor,action,reasons,factors,step.acknowledgement_factor,step.depends_on)
        steps=list(row.steps);steps[row.cursor]=replacement
        revised=JointPlanV1(row.identity,row.context,row.source,tuple(steps),row.cursor,STATUS_ACTIVE,(),row.revision+1,
            bool(step.role==ROLE_SELF),0)
        self._replace(revised);return True

    def revise_self_step(self,action,reason_ids,language_factors,acknowledgement_factor,tick):
        self._advance(tick);row=self.plan();step=self.current_step();action=int(action);factors=tuple(sorted(set(map(int,language_factors))));ack_factor=int(acknowledgement_factor)
        if row is None or step is None or step.role!=ROLE_SELF or action<=0 or not factors or min(factors)<=0 or ack_factor<=0:return False
        reasons=tuple(sorted(set(map(int,reason_ids))))
        replacement=JointPlanStepV1(_id('joint-plan-self-revision-v1',[step.identity,action,list(reasons),list(factors),row.revision+1]),
            step.subgoal,ROLE_SELF,0,action,reasons,factors,ack_factor,step.depends_on)
        steps=list(row.steps);steps[row.cursor]=replacement
        revised=JointPlanV1(row.identity,row.context,row.source,tuple(steps),row.cursor,STATUS_ACTIVE,(),row.revision+1,True,0)
        self._replace(revised);return True

    def mark_acknowledged(self,tick):
        self._advance(tick);row=self.plan();step=self.current_step()
        if row is None or step is None or step.role!=ROLE_SELF or not row.acknowledgement_pending:return False
        self._replace(JointPlanV1(row.identity,row.context,row.source,row.steps,row.cursor,row.status,row.alternatives,row.revision,False,0));return True

    def _advance_cursor(self,row):
        cursor=row.cursor+1
        status=STATUS_COMPLETE if cursor>=len(row.steps) else STATUS_ACTIVE
        ack=bool(status==STATUS_ACTIVE and row.steps[cursor].role==ROLE_SELF)
        revised=JointPlanV1(row.identity,row.context,row.source,row.steps,cursor,status,(),row.revision,ack,0)
        self._replace(revised)
        if status==STATUS_COMPLETE:self.active_plan=0
        return revised

    def settle_self_action(self,action,success,independent,tick):
        self._advance(tick);row=self.plan();step=self.current_step();action=int(action)
        if row is None or step is None or step.role!=ROLE_SELF or action!=step.action or not independent:return False
        if not success:
            self._replace(JointPlanV1(row.identity,row.context,row.source,row.steps,row.cursor,STATUS_BLOCKED,(),row.revision,False,0));return False
        self._advance_cursor(row);return True

    def observe_partner_action(self,actor,action,success,independent,tick):
        self._advance(tick);row=self.plan();step=self.current_step();actor=int(actor);action=int(action)
        if row is None or step is None or step.role!=ROLE_PARTNER or not independent:return False
        if actor!=step.actor or action!=step.action or not success:
            self._replace(JointPlanV1(row.identity,row.context,row.source,row.steps,row.cursor,STATUS_BLOCKED,(),row.revision,False,0));return False
        self._advance_cursor(row);return True

    def invalidate_dependent_suffix(self,failed_step,tick):
        """Block only a suffix that explicitly depends on the failed subgoal index."""
        self._advance(tick);row=self.plan();failed=int(failed_step)
        if row is None or failed<0 or failed>=len(row.steps):return False
        if row.cursor<=failed:
            self._replace(JointPlanV1(row.identity,row.context,row.source,row.steps,row.cursor,STATUS_BLOCKED,(),row.revision,False,0));return True
        return False

    def checkpoint(self):
        return {'schema':1,'tick':self._tick,'active_plan':int(self.active_plan),'plans':[
            {'identity':p.identity,'context':p.context,'source':p.source,'cursor':p.cursor,'status':p.status,
             'alternatives':list(p.alternatives),'revision':p.revision,'acknowledgement_pending':p.acknowledgement_pending,
             'steps':[{'identity':s.identity,'subgoal':s.subgoal,'role':s.role,'actor':s.actor,'action':s.action,
                       'reason_ids':list(s.reason_ids),'language_factors':list(s.language_factors),'acknowledgement_factor':s.acknowledgement_factor,'depends_on':list(s.depends_on)} for s in p.steps]}
            for p in self._plans]}

    @classmethod
    def restore(cls,data):
        if int(data.get('schema',0))!=1:raise RuntimeError('joint-plan:checkpoint-schema')
        out=cls();out._tick=int(data.get('tick',-1));out.active_plan=int(data.get('active_plan',0))
        for p in data.get('plans',()):
            steps=tuple(JointPlanStepV1(int(s['identity']),int(s['subgoal']),int(s['role']),int(s['actor']),int(s['action']),
                tuple(map(int,s.get('reason_ids',()))),tuple(map(int,s.get('language_factors',s.get('language_factor',()) if isinstance(s.get('language_factor',()),(tuple,list)) else (s.get('language_factor',0),)))),int(s.get('acknowledgement_factor',0)),tuple(map(int,s.get('depends_on',())))) for s in p.get('steps',()))
            row=JointPlanV1(int(p['identity']),int(p['context']),int(p['source']),steps,int(p['cursor']),int(p['status']),
                tuple(map(int,p.get('alternatives',()))),int(p.get('revision',0)),bool(p.get('acknowledgement_pending',False)),0)
            out._plans.append(row)
        if len(out._plans)>MAX_JOINT_PLANS or (out.active_plan and out.plan() is None):raise RuntimeError('joint-plan:checkpoint')
        return out
