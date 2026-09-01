#!/usr/bin/env python3
"""Bounded persistent temporal multiplexing for authenticated social contact.

This operator owns chronology and pending-contact persistence only.  It has no semantic,
trust, moral, speaker-rank, turn-taking, or public-action authority.
"""
from __future__ import annotations
from dataclasses import dataclass,asdict
import heapq

MAX_PENDING_SOCIAL_CONTACTS=512

@dataclass(frozen=True,order=True)
class SocialStreamContactV1:
    tick:int
    sequence:int
    source:int
    subject:int
    observed_state:int
    action:int
    outcome_q16:int=0
    somatic_q16:int=0
    consequence_present:bool=False
    independent:bool=False

class PersistentMultiAgentSocialStreamV1:
    """Causal-time ordered pending social contacts, checkpointable across overlap."""
    def __init__(self):
        self._pending=[]
        self._next_sequence=1
        self._last_drained_tick=-1
        self.processed_contacts=0  # observer meter; not checkpoint authority

    def admit(self,tick,source,subject,observed_state,action,
              outcome_q16=None,somatic_q16=0,independent=False):
        tick=int(tick);source=int(source);subject=int(subject);observed_state=int(observed_state);action=int(action)
        if tick<0 or min(source,subject,observed_state,action)<=0:raise ValueError('social-stream:contact')
        if tick<self._last_drained_tick:raise ValueError('social-stream:late-contact')
        if len(self._pending)>=MAX_PENDING_SOCIAL_CONTACTS:raise RuntimeError('social-stream:capacity')
        sequence=self._next_sequence;self._next_sequence+=1
        event=SocialStreamContactV1(
            tick,sequence,source,subject,observed_state,action,
            0 if outcome_q16 is None else int(outcome_q16),int(somatic_q16),
            outcome_q16 is not None,bool(independent))
        heapq.heappush(self._pending,event)
        return sequence

    def drain_until(self,adult,tick):
        """Process every due contact once; time/sequence decide order, never semantics."""
        tick=int(tick)
        if tick<self._last_drained_tick:raise ValueError('social-stream:time-reversal')
        processed=[]
        while self._pending and self._pending[0].tick<=tick:
            event=heapq.heappop(self._pending)
            adult.observe_social_source_contact(
                event.subject,event.observed_state,event.action,event.source)
            if event.consequence_present:
                adult.settle_current_social_action_consequence(
                    event.outcome_q16,event.somatic_q16,event.independent)
            processed.append((event.tick,event.sequence,event.source,event.action))
            self.processed_contacts+=1
        self._last_drained_tick=tick
        return tuple(processed)

    @property
    def pending_count(self):return len(self._pending)

    def checkpoint(self):
        return {
            'schema':1,
            'next_sequence':self._next_sequence,
            'last_drained_tick':self._last_drained_tick,
            'pending':[asdict(event) for event in sorted(self._pending)],
        }

    @classmethod
    def restore(cls,data):
        if int(data.get('schema',0))!=1:raise RuntimeError('social-stream:checkpoint-schema')
        stream=cls();stream._next_sequence=int(data.get('next_sequence',0));stream._last_drained_tick=int(data.get('last_drained_tick',-1))
        if stream._next_sequence<1 or stream._last_drained_tick<-1:raise RuntimeError('social-stream:checkpoint-state')
        pending=[];seen=set()
        for row in data.get('pending',()):
            event=SocialStreamContactV1(
                int(row['tick']),int(row['sequence']),int(row['source']),int(row['subject']),
                int(row['observed_state']),int(row['action']),int(row.get('outcome_q16',0)),
                int(row.get('somatic_q16',0)),bool(row.get('consequence_present',False)),bool(row.get('independent',False)))
            if (event.tick<stream._last_drained_tick or event.sequence<=0 or event.sequence in seen
                    or min(event.source,event.subject,event.observed_state,event.action)<=0):
                raise RuntimeError('social-stream:checkpoint-event')
            seen.add(event.sequence);pending.append(event)
        if len(pending)>MAX_PENDING_SOCIAL_CONTACTS:raise RuntimeError('social-stream:checkpoint-capacity')
        if pending and max(e.sequence for e in pending)>=stream._next_sequence:raise RuntimeError('social-stream:checkpoint-sequence')
        stream._pending=pending;heapq.heapify(stream._pending);return stream
