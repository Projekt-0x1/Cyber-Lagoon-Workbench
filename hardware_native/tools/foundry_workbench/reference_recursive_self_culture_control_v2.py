#!/usr/bin/env python3
"""Sequence-qualified cumulative culture over RecursiveSelfCultureControlV1.

V1 introduced constructive memory, metacognitive review and zero-authority cultural
programs. V2 tightens the causal contract: procedure confirmation comes only from a
completed resident execution, not from temporally unrelated matching fragments; a
procedure may outlive its last teacher only after such lived confirmation; transmitted
reasons are re-authored to the actual retransmitter; reasons are evaluated with their own
source provenance and never receive an unbound privilege to reinforce the incumbent choice.
"""
from __future__ import annotations

from reference_recursive_self_culture_control_v1 import (
    RecursiveSelfCultureControlV1,
    CulturalProgramV1,
    DecisionStateV1,
    Q,
)


class RecursiveSelfCultureControlV2(RecursiveSelfCultureControlV1):
    """Cumulative culture whose persistence is earned by completed lived executions."""

    def __init__(self):
        super().__init__()
        self._program_completions={}  # program -> {source:last_completion_tick}

    def _confirming_sources(self,program):
        rows=self._program_completions.get(int(program.identity),{})
        return tuple(sorted(int(source) for source in rows
            if int(source) not in self._withdrawn_sources))

    def _refresh_program_confirmations(self):
        rows=[]
        for program in self._programs:
            confirmations=self._confirming_sources(program)
            teachers=tuple(sorted(s for s in program.teachers
                if int(s) not in self._withdrawn_sources))
            if not teachers and not confirmations:
                continue
            rows.append(CulturalProgramV1(
                int(program.identity),int(program.context),tuple(program.actions),
                tuple(program.reason_ids),teachers,confirmations,
                tuple(program.withdrawals),0))
        self._programs=rows

    def observe_episode(self,*args,**kwargs):
        identity=super().observe_episode(*args,**kwargs)
        # V1 provisional fragment confirmation is never authoritative in V2.
        self._refresh_program_confirmations()
        return identity

    def confirm_program_execution(self,identity,source,tick,independent=True):
        """Record only a full execution that the organism cursor has already validated."""
        self._advance(tick);identity=int(identity);source=int(source)
        if not independent or identity<=0 or source<=0 or source in self._withdrawn_sources:
            return False
        program=next((row for row in self._programs if int(row.identity)==identity),None)
        if program is None or not program.actions:
            return False
        self._program_completions.setdefault(identity,{})[source]=int(tick)
        self._refresh_program_confirmations()
        return True

    def transmit_program(self,identity):
        self._refresh_program_confirmations()
        program=next((row for row in self._programs if row.identity==int(identity)),None)
        if program is None:
            return ()
        reasons=[]
        for reason_id in program.reason_ids:
            row=next((x for x in self._reasons
                if x.identity==int(reason_id) and x.source not in self._withdrawn_sources),None)
            if row is not None:
                reasons.append((int(row.proposition),int(row.support_q16)))
        return (int(program.context),tuple(program.actions),tuple(reasons),
            tuple(program.teachers),tuple(program.confirmations),
            tuple(program.withdrawals),0)

    def receive_program(self,packet,teacher,joint_context,tick,known_actions=()):
        if len(packet)!=7:
            return 0
        context,actions,reasons,_teachers,_confirmations,_withdrawals,authority=packet
        if int(authority)!=0:
            return 0
        self._advance(tick);local_reasons=[]
        for row in reasons:
            if not isinstance(row,(tuple,list)) or len(row)!=2:
                continue
            proposition,support=row
            reason=self.observe_shared_reason(
                int(proposition),int(teacher),int(joint_context),int(support),int(tick),True)
            if reason:
                local_reasons.append(int(reason))
        return self.compose_instruction(int(context),tuple(map(int,actions)),
            tuple(local_reasons),int(teacher),int(joint_context),int(tick),known_actions)

    def _program_reason_support(self,program,source_credibility):
        total=0;count=0
        for reason_id in set(map(int,program.reason_ids)):
            row=next((x for x in self._reasons
                if int(x.identity)==reason_id and x.source not in self._withdrawn_sources),None)
            if row is None:
                continue
            credibility=max(-Q,min(Q,int(source_credibility(int(row.source)))))
            total+=(int(row.support_q16)*(Q+credibility))//(2*Q);count+=1
        return 0 if not count else max(-Q,min(Q,total//count))

    def program_candidates(self,context,body_resource_q16,controllability_q16,source_credibility):
        self._refresh_program_confirmations()
        context=int(context);resource=max(0,min(Q,int(body_resource_q16)))
        control=max(0,min(Q,int(controllability_q16)));rows=[]
        for program in self._programs:
            if int(program.context)!=context:
                continue
            teachers=tuple(s for s in program.teachers if s not in self._withdrawn_sources)
            confirmations=tuple(s for s in program.confirmations if s not in self._withdrawn_sources)
            support_sources=tuple(sorted(set((*teachers,*confirmations))))
            if not support_sources:
                continue
            cred=sum(max(-Q,min(Q,int(source_credibility(s))))
                for s in support_sources)//len(support_sources)
            reason=self._program_reason_support(program,source_credibility)
            social=max(0,min(Q,(Q+cred)//2));reason_gate=max(Q//4,min(Q,(Q+reason)//2))
            lived=min(Q,(len(confirmations)*Q)//3)
            score=(resource*max(Q//8,control)*max(Q//8,social)*reason_gate)//(Q*Q*Q)
            score=(score*(Q//2+lived//2))//Q
            rows.append({'identity':int(program.identity),'actions':tuple(program.actions),
                'score_q16':int(score),'teachers':teachers,'confirmations':confirmations,
                'reason_ids':tuple(program.reason_ids),'authority':0})
        rows.sort(key=lambda row:(-int(row['score_q16']),-len(row['confirmations']),row['identity']))
        return tuple(rows)

    def review_decision(self,identity,new_evidence_q16,body_resource_q16,
                        controllability_q16,source_credibility_q16=0):
        """Reconsider from resident evidence without an unbound incumbent-choice reason bonus."""
        row=next((x for x in self._decisions if x.identity==int(identity)),None)
        if row is None:
            return {'status':0,'selected':0,'changed':False,'confidence_q16':0}
        evidence=dict(row.evidence_q16)
        for action,value in new_evidence_q16:
            if int(action) in evidence:
                evidence[int(action)]=max(-Q,min(Q,evidence[int(action)]+int(value)))
        resource=max(0,min(Q,int(body_resource_q16)));control=max(0,min(Q,int(controllability_q16)))
        commitment=max(Q//8,(resource*max(Q//8,control))//Q)
        adjusted={action:(value*commitment)//Q for action,value in evidence.items()}
        ranked=sorted(((value,action) for action,value in adjusted.items()),reverse=True)
        top=ranked[0][0];winners=[action for value,action in ranked if value==top]
        selected=winners[0] if len(winners)==1 else 0
        second=max((value for value,action in ranked if action!=selected),default=-Q) if selected else top
        confidence=max(0,min(Q,(top-second+Q)//2)) if selected else 0
        revised=DecisionStateV1(row.identity,row.context,row.alternatives,selected,
            tuple(sorted(evidence.items())),confidence,resource,control,row.reason_ids,
            row.revision+1,self._tick)
        self._decisions[self._decisions.index(row)]=revised
        return {'status':1,'selected':selected,
            'changed':bool(selected and selected!=row.selected),
            'confidence_q16':confidence,'revision':revised.revision}

    def withdraw_source(self,source):
        source=int(source)
        if source<=0:
            raise ValueError('self-culture-v2:withdraw-source')
        self._withdrawn_sources.add(source)
        self._episodes=[x for x in self._episodes if int(x.source)!=source]
        self._reasons=[x for x in self._reasons if int(x.source)!=source]
        for identity,rows in tuple(self._program_completions.items()):
            rows.pop(source,None)
            if not rows:self._program_completions.pop(identity,None)
        programs=[]
        for program in self._programs:
            teachers=tuple(s for s in program.teachers if int(s)!=source)
            withdrawals=tuple(sorted(set((*program.withdrawals,source))))
            programs.append(CulturalProgramV1(int(program.identity),int(program.context),
                tuple(program.actions),tuple(program.reason_ids),teachers,(),withdrawals,0))
        self._programs=programs;self._refresh_program_confirmations();return source

    def checkpoint(self):
        data=super().checkpoint()
        data['v2_program_completions']=[
            {'program':int(identity),'rows':[[int(source),int(tick)]
                for source,tick in sorted(rows.items())]}
            for identity,rows in sorted(self._program_completions.items())]
        return data

    @classmethod
    def restore(cls,data):
        prior=RecursiveSelfCultureControlV1.restore(data)
        out=cls();out.__dict__.update(prior.__dict__);out._program_completions={}
        for row in data.get('v2_program_completions',()):
            identity=int(row['program']);out._program_completions[identity]={
                int(source):int(tick) for source,tick in row.get('rows',())}
        out._refresh_program_confirmations();return out
