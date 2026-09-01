#!/usr/bin/env python3
"""Recursive self-in-culture control for the continuing reference Adult.

This owner is deliberately not an executive. It stores revisable relations among lived
episodes, imagined futures, decisions, instructions and shared reasons. Imagined futures
and cultural programs have zero epistemic authority; they may nominate already-earned
action primitives, but consequence-qualified motor arbitration remains outside this owner.

Mechanistic targets: constructive episodic simulation, post-decision metacognition/change
of mind, rapid instructed task composition, joint/shared reasons, and cumulative cultural
transmission. Body/allostatic context is supplied on every appraisal rather than hidden in
a disembodied global confidence scalar.
"""
from __future__ import annotations
from dataclasses import dataclass
import hashlib, json

Q=1<<16
MAX_EPISODES=4096
MAX_DECISIONS=2048
MAX_PROGRAMS=1024
MAX_REASONS=4096
MAX_FUTURES=256


def _id(tag,payload):
    raw=json.dumps([tag,payload],sort_keys=True,separators=(',',':')).encode()
    return int(hashlib.sha256(raw).hexdigest()[:16],16) or 1


def _clip(v,lo=0,hi=Q):return max(lo,min(hi,int(v)))


@dataclass(frozen=True)
class SelfEpisodeV1:
    identity:int
    context:int
    action:int
    outcome:int
    source:int
    tick:int
    body_resource_q16:int
    controllability_q16:int
    social_context:int


@dataclass(frozen=True)
class ConstructiveFutureV1:
    identity:int
    context:int
    actions:tuple[int,...]
    support_episodes:tuple[int,...]
    predicted_outcomes:tuple[int,...]
    authority:int=0


@dataclass(frozen=True)
class DecisionStateV1:
    identity:int
    context:int
    alternatives:tuple[int,...]
    selected:int
    evidence_q16:tuple[tuple[int,int],...]
    confidence_q16:int
    body_resource_q16:int
    controllability_q16:int
    reason_ids:tuple[int,...]
    revision:int
    tick:int


@dataclass(frozen=True)
class CulturalProgramV1:
    identity:int
    context:int
    actions:tuple[int,...]
    reason_ids:tuple[int,...]
    teachers:tuple[int,...]
    confirmations:tuple[int,...]
    withdrawals:tuple[int,...]
    authority:int=0


@dataclass(frozen=True)
class SharedReasonV1:
    identity:int
    proposition:int
    source:int
    joint_context:int
    support_q16:int
    tick:int


class RecursiveSelfCultureControlV1:
    """Revisable recursive self/culture state with zero-authority simulation."""
    def __init__(self):
        self._episodes=[]
        self._decisions=[]
        self._programs=[]
        self._reasons=[]
        self._withdrawn_sources=set()
        self._tick=-1

    def _advance(self,tick):
        tick=int(tick)
        if tick<self._tick:raise ValueError('self-culture:time-reversal')
        self._tick=tick

    def observe_episode(self,context,action,outcome,source,tick,body_resource_q16,controllability_q16,social_context=0,independent=True):
        self._advance(tick);context=int(context);action=int(action);outcome=int(outcome);source=int(source)
        if not independent or min(context,action,outcome,source)<=0 or source in self._withdrawn_sources:return 0
        payload=[context,action,outcome,source,int(tick),_clip(body_resource_q16),_clip(controllability_q16),int(social_context)]
        identity=_id('self-episode-v1',payload)
        row=SelfEpisodeV1(identity,*payload)
        if row not in self._episodes:
            if len(self._episodes)>=MAX_EPISODES:self._episodes.pop(0)
            self._episodes.append(row)
        # A lived consequence can confirm matching culturally transmitted action fragments.
        updated=[]
        for p in self._programs:
            confirmations=set(p.confirmations)
            if p.context==context and action in p.actions:confirmations.add(source)
            updated.append(CulturalProgramV1(p.identity,p.context,p.actions,p.reason_ids,p.teachers,tuple(sorted(confirmations)),p.withdrawals,0))
        self._programs=updated
        return identity

    def constructive_futures(self,context,max_futures=MAX_FUTURES):
        """Recombine experienced fragments; never create evidence or truth."""
        context=int(context);rows=[e for e in self._episodes if e.context==context and e.source not in self._withdrawn_sources]
        out=[];seen=set()
        # One-step futures preserve experienced action/outcome coupling.
        for e in rows:
            key=((e.action,),(e.outcome,),(e.identity,))
            if key not in seen:
                seen.add(key);out.append(ConstructiveFutureV1(_id('future-v1',[context,list(key[0]),list(key[1]),list(key[2])]),context,key[0],key[2],key[1],0))
        # Two-step constructive recombination uses distinct episodes/sources where possible.
        for a in rows:
            for b in rows:
                if a.identity==b.identity or a.source==b.source:continue
                key=((a.action,b.action),(a.outcome,b.outcome),(a.identity,b.identity))
                if key in seen:continue
                seen.add(key);out.append(ConstructiveFutureV1(_id('future-v1',[context,list(key[0]),list(key[1]),list(key[2])]),context,key[0],key[2],key[1],0))
                if len(out)>=max(0,int(max_futures)):return tuple(out)
        return tuple(out[:max(0,int(max_futures))])

    def observe_shared_reason(self,proposition,source,joint_context,support_q16,tick,independent=True):
        self._advance(tick);proposition=int(proposition);source=int(source);joint_context=int(joint_context)
        if not independent or min(proposition,source,joint_context)<=0 or source in self._withdrawn_sources:return 0
        support=max(-Q,min(Q,int(support_q16)))
        identity=_id('shared-reason-v1',[proposition,source,joint_context])
        row=SharedReasonV1(identity,proposition,source,joint_context,support,int(tick))
        self._reasons=[x for x in self._reasons if x.identity!=identity]
        if len(self._reasons)>=MAX_REASONS:self._reasons.pop(0)
        self._reasons.append(row);return identity

    def reason_support(self,reason_ids,source_credibility_q16=0):
        total=0;count=0
        credibility=max(-Q,min(Q,int(source_credibility_q16)))
        for rid in set(map(int,reason_ids)):
            row=next((x for x in self._reasons if x.identity==rid and x.source not in self._withdrawn_sources),None)
            if row is None:continue
            # Source credibility scales but cannot reverse an independently supported reason.
            weighted=(row.support_q16*(Q+credibility))//(2*Q)
            total+=weighted;count+=1
        return 0 if not count else max(-Q,min(Q,total//count))

    def compose_instruction(self,context,actions,reason_ids,teacher,joint_context,tick,known_actions=()):
        """First-trial task composition from known action primitives only.

        The teacher supplies a sequence over already-earned action identities. This is rapid
        composition, not lexical grounding: unknown primitives are rejected rather than
        becoming true because an instructor named them.
        """
        self._advance(tick);context=int(context);teacher=int(teacher);joint_context=int(joint_context)
        actions=tuple(map(int,actions));known=set(map(int,known_actions));reasons=tuple(sorted(set(map(int,reason_ids))))
        if min(context,teacher,joint_context)<=0 or not actions or teacher in self._withdrawn_sources:return 0
        if known and any(a not in known for a in actions):return 0
        if any(a<=0 for a in actions):return 0
        identity=_id('cultural-program-v1',[context,list(actions),list(reasons),joint_context])
        existing=next((x for x in self._programs if x.identity==identity),None)
        teachers={teacher};confirmations=set();withdrawals=set()
        if existing:
            teachers.update(existing.teachers);confirmations.update(existing.confirmations);withdrawals.update(existing.withdrawals)
            self._programs.remove(existing)
        row=CulturalProgramV1(identity,context,actions,reasons,tuple(sorted(teachers)),tuple(sorted(confirmations)),tuple(sorted(withdrawals)),0)
        if len(self._programs)>=MAX_PROGRAMS:self._programs.pop(0)
        self._programs.append(row);return identity

    def transmit_program(self,identity):
        """Culture packet contains structure/provenance, not motor or truth authority."""
        row=next((x for x in self._programs if x.identity==int(identity)),None)
        if row is None:return ()
        return (row.context,row.actions,row.reason_ids,row.teachers,row.confirmations,row.withdrawals,0)

    def receive_program(self,packet,teacher,joint_context,tick,known_actions=()):
        if len(packet)!=7:return 0
        context,actions,reasons,_teachers,_confirmations,_withdrawals,authority=packet
        if int(authority)!=0:return 0
        return self.compose_instruction(context,actions,reasons,teacher,joint_context,tick,known_actions)

    def program_candidates(self,context,body_resource_q16,controllability_q16,source_credibility):
        """Embodied ranking of provisional programs; returned rows still have authority 0."""
        context=int(context);resource=_clip(body_resource_q16);control=_clip(controllability_q16);rows=[]
        for p in self._programs:
            if p.context!=context:continue
            active_teachers=[s for s in p.teachers if s not in self._withdrawn_sources]
            if not active_teachers:continue
            cred=0
            if active_teachers:
                cred=sum(max(-Q,min(Q,int(source_credibility(s)))) for s in active_teachers)//len(active_teachers)
            reason=self.reason_support(p.reason_ids,cred)
            social=max(0,min(Q,(Q+cred)//2));reason_gate=max(Q//4,min(Q,(Q+reason)//2))
            lived=min(Q,(len(p.confirmations)*Q)//3)
            # No single scalar becomes truth; score is only arbitration priority.
            score=(resource*max(Q//8,control)*max(Q//8,social)*reason_gate)//(Q*Q*Q)
            score=(score*(Q//2+lived//2))//Q
            rows.append({'identity':p.identity,'actions':p.actions,'score_q16':int(score),'teachers':tuple(active_teachers),'confirmations':p.confirmations,'authority':0})
        rows.sort(key=lambda x:(-x['score_q16'],-len(x['confirmations']),x['identity']))
        return tuple(rows)

    def begin_decision(self,context,alternatives,evidence_q16,body_resource_q16,controllability_q16,reason_ids,tick):
        self._advance(tick);context=int(context);alternatives=tuple(sorted(set(map(int,alternatives))))
        evidence={int(a):max(-Q,min(Q,int(v))) for a,v in evidence_q16 if int(a) in alternatives}
        if context<=0 or not alternatives:return 0
        for a in alternatives:evidence.setdefault(a,0)
        ranked=sorted(((evidence[a],a) for a in alternatives),reverse=True);top=ranked[0][0];winners=[a for v,a in ranked if v==top]
        selected=winners[0] if len(winners)==1 else 0
        second=max((v for v,a in ranked if a!=selected),default=-Q) if selected else top
        confidence=_clip((top-second+Q)//2) if selected else 0
        identity=_id('decision-v1',[context,list(alternatives),int(tick),len(self._decisions)])
        row=DecisionStateV1(identity,context,alternatives,selected,tuple(sorted(evidence.items())),confidence,_clip(body_resource_q16),_clip(controllability_q16),tuple(sorted(set(map(int,reason_ids)))),0,int(tick))
        if len(self._decisions)>=MAX_DECISIONS:self._decisions.pop(0)
        self._decisions.append(row);return identity

    def review_decision(self,identity,new_evidence_q16,body_resource_q16,controllability_q16,source_credibility_q16=0):
        """Change of mind from resident evidence; no external feedback is required."""
        row=next((x for x in self._decisions if x.identity==int(identity)),None)
        if row is None:return {'status':0,'selected':0,'changed':False,'confidence_q16':0}
        evidence=dict(row.evidence_q16)
        for a,v in new_evidence_q16:
            if int(a) in evidence:evidence[int(a)]=max(-Q,min(Q,evidence[int(a)]+int(v)))
        resource=_clip(body_resource_q16);control=_clip(controllability_q16)
        reason=self.reason_support(row.reason_ids,source_credibility_q16)
        # Low resource/control reduces commitment, not evidence sign.
        commitment=max(Q//8,(resource*max(Q//8,control))//Q)
        adjusted={a:(v*commitment)//Q for a,v in evidence.items()}
        if reason:
            # Shared reasons modulate willingness to maintain the prior choice, not facts.
            if row.selected in adjusted:adjusted[row.selected]+=(reason*Q)//(4*Q)
        ranked=sorted(((v,a) for a,v in adjusted.items()),reverse=True);top=ranked[0][0];winners=[a for v,a in ranked if v==top];selected=winners[0] if len(winners)==1 else 0
        second=max((v for v,a in ranked if a!=selected),default=-Q) if selected else top
        confidence=_clip((top-second+Q)//2) if selected else 0
        revised=DecisionStateV1(row.identity,row.context,row.alternatives,selected,tuple(sorted(evidence.items())),confidence,resource,control,row.reason_ids,row.revision+1,self._tick)
        self._decisions[self._decisions.index(row)]=revised
        return {'status':1,'selected':selected,'changed':bool(selected and selected!=row.selected),'confidence_q16':confidence,'revision':revised.revision}

    def withdraw_source(self,source):
        source=int(source)
        if source<=0:raise ValueError('self-culture:withdraw-source')
        self._withdrawn_sources.add(source)
        self._episodes=[x for x in self._episodes if x.source!=source]
        self._reasons=[x for x in self._reasons if x.source!=source]
        programs=[]
        for p in self._programs:
            teachers=tuple(s for s in p.teachers if s!=source);confirmations=tuple(s for s in p.confirmations if s!=source);withdrawals=tuple(sorted(set((*p.withdrawals,source))))
            if teachers:programs.append(CulturalProgramV1(p.identity,p.context,p.actions,p.reason_ids,teachers,confirmations,withdrawals,0))
        self._programs=programs

    @property
    def episode_count(self):return len(self._episodes)
    @property
    def program_count(self):return len(self._programs)
    @property
    def decision_count(self):return len(self._decisions)
    @property
    def reason_count(self):return len(self._reasons)

    def checkpoint(self):
        return {'schema':1,'tick':self._tick,'withdrawn_sources':sorted(self._withdrawn_sources),
            'episodes':[x.__dict__ for x in self._episodes],
            'decisions':[{'identity':x.identity,'context':x.context,'alternatives':list(x.alternatives),'selected':x.selected,'evidence':[list(y) for y in x.evidence_q16],'confidence_q16':x.confidence_q16,'body_resource_q16':x.body_resource_q16,'controllability_q16':x.controllability_q16,'reason_ids':list(x.reason_ids),'revision':x.revision,'tick':x.tick} for x in self._decisions],
            'programs':[{'identity':x.identity,'context':x.context,'actions':list(x.actions),'reason_ids':list(x.reason_ids),'teachers':list(x.teachers),'confirmations':list(x.confirmations),'withdrawals':list(x.withdrawals),'authority':0} for x in self._programs],
            'reasons':[x.__dict__ for x in self._reasons]}

    @classmethod
    def restore(cls,data):
        if int(data.get('schema',0))!=1:raise RuntimeError('self-culture:checkpoint-schema')
        out=cls();out._tick=int(data.get('tick',-1));out._withdrawn_sources=set(map(int,data.get('withdrawn_sources',())))
        out._episodes=[SelfEpisodeV1(**{k:int(v) for k,v in x.items()}) for x in data.get('episodes',())]
        out._decisions=[DecisionStateV1(int(x['identity']),int(x['context']),tuple(map(int,x['alternatives'])),int(x['selected']),tuple((int(a),int(v)) for a,v in x['evidence']),int(x['confidence_q16']),int(x['body_resource_q16']),int(x['controllability_q16']),tuple(map(int,x['reason_ids'])),int(x['revision']),int(x['tick'])) for x in data.get('decisions',())]
        out._programs=[CulturalProgramV1(int(x['identity']),int(x['context']),tuple(map(int,x['actions'])),tuple(map(int,x['reason_ids'])),tuple(map(int,x['teachers'])),tuple(map(int,x['confirmations'])),tuple(map(int,x['withdrawals'])),0) for x in data.get('programs',())]
        out._reasons=[SharedReasonV1(**{k:int(v) for k,v in x.items()}) for x in data.get('reasons',())]
        if len(out._episodes)>MAX_EPISODES or len(out._decisions)>MAX_DECISIONS or len(out._programs)>MAX_PROGRAMS or len(out._reasons)>MAX_REASONS:raise RuntimeError('self-culture:capacity')
        return out
