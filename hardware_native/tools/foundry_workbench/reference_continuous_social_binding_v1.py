#!/usr/bin/env python3
"""Continuous overlapping speech binding with source-qualified Merge provenance.

This owner is intentionally narrow. It persists only still-pending acoustic/surface events and
bounded provenance for already-formed recursive Merge roots. It does not own source trust,
reputation polarity, somatic state, action value, turn-taking, or public response policy.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
import heapq

MAX_PENDING_SPEECH=512
MAX_PROVENANCE_ROWS=1024
MAX_RAW_UNITS=4096


@dataclass(frozen=True, order=True)
class SocialSpeechContactV1:
    tick:int
    sequence:int
    speaker:int
    merge_root:int
    raw:tuple[int,...]


class ContinuousSocialBindingV1:
    """Temporal multiplexing + proposition/source provenance, without social authority."""
    def __init__(self):
        self._pending=[]
        self._next_sequence=1
        self._last_drained_tick=-1
        # (merge_root,speaker) -> [count,last_tick]. No wording, trust, valence, or action score.
        self._provenance={}

    def admit(self,tick,speaker,raw,merge_root=0):
        tick=int(tick);speaker=int(speaker);merge_root=int(merge_root)
        raw=tuple(map(int,raw))
        if tick<0 or speaker<=0 or merge_root<0:raise ValueError('continuous-social-binding:contact')
        if tick<self._last_drained_tick:raise ValueError('continuous-social-binding:late-contact')
        if not raw or len(raw)>MAX_RAW_UNITS or any(x<0 or x>255 for x in raw):
            raise ValueError('continuous-social-binding:raw')
        if len(self._pending)>=MAX_PENDING_SPEECH:raise RuntimeError('continuous-social-binding:capacity')
        seq=self._next_sequence;self._next_sequence+=1
        heapq.heappush(self._pending,SocialSpeechContactV1(tick,seq,speaker,merge_root,raw))
        return seq

    def _bind_provenance(self,merge_root,speaker,tick):
        if merge_root<=0:return
        key=(int(merge_root),int(speaker))
        row=self._provenance.get(key)
        if row is None:
            if len(self._provenance)>=MAX_PROVENANCE_ROWS:
                # Deterministic oldest-row eviction; semantic/trust scores never affect retention.
                victim=min(self._provenance,key=lambda k:(self._provenance[k][1],k[0],k[1]))
                del self._provenance[victim]
            self._provenance[key]=[1,int(tick)]
        else:
            row[0]=min(0x7fffffff,int(row[0])+1);row[1]=int(tick)

    def drain_until(self,adult,reputation_factor,tick):
        """Drain by organism time/admission order and route raw speech into existing epistemics."""
        tick=int(tick)
        if tick<self._last_drained_tick:raise ValueError('continuous-social-binding:time-reversal')
        out=[]
        while self._pending and self._pending[0].tick<=tick:
            event=heapq.heappop(self._pending)
            self._bind_provenance(event.merge_root,event.speaker,event.tick)
            accepted=bool(reputation_factor.observe_open_contact(adult,event.raw,event.speaker))
            out.append((event.tick,event.sequence,event.speaker,event.merge_root,accepted))
        self._last_drained_tick=tick
        return tuple(out)

    def sources_for(self,merge_root):
        merge_root=int(merge_root)
        return tuple(sorted((speaker,int(row[0]),int(row[1]))
            for (root,speaker),row in self._provenance.items() if root==merge_root))

    @property
    def pending_count(self):return len(self._pending)

    def checkpoint(self):
        return {
            'schema':1,
            'next_sequence':self._next_sequence,
            'last_drained_tick':self._last_drained_tick,
            'pending':[asdict(e) | {'raw':list(e.raw)} for e in sorted(self._pending)],
            'provenance':[[root,speaker,int(row[0]),int(row[1])]
                for (root,speaker),row in sorted(self._provenance.items())],
        }

    @classmethod
    def restore(cls,data):
        if int(data.get('schema',0))!=1:raise RuntimeError('continuous-social-binding:checkpoint-schema')
        out=cls();out._next_sequence=int(data.get('next_sequence',0));out._last_drained_tick=int(data.get('last_drained_tick',-1))
        if out._next_sequence<1 or out._last_drained_tick<-1:raise RuntimeError('continuous-social-binding:checkpoint-state')
        seen=set();pending=[]
        for row in data.get('pending',()):
            raw=tuple(map(int,row.get('raw',())))
            event=SocialSpeechContactV1(int(row['tick']),int(row['sequence']),int(row['speaker']),int(row.get('merge_root',0)),raw)
            if (event.tick<out._last_drained_tick or event.sequence<=0 or event.sequence in seen or event.speaker<=0
                    or event.merge_root<0 or not raw or len(raw)>MAX_RAW_UNITS or any(x<0 or x>255 for x in raw)):
                raise RuntimeError('continuous-social-binding:checkpoint-event')
            seen.add(event.sequence);pending.append(event)
        if len(pending)>MAX_PENDING_SPEECH or (pending and max(e.sequence for e in pending)>=out._next_sequence):
            raise RuntimeError('continuous-social-binding:checkpoint-capacity')
        provenance={}
        for row in data.get('provenance',()):
            if len(row)!=4:raise RuntimeError('continuous-social-binding:checkpoint-provenance')
            root,speaker,count,last_tick=map(int,row)
            if root<=0 or speaker<=0 or count<=0 or last_tick<0:raise RuntimeError('continuous-social-binding:checkpoint-provenance')
            provenance[(root,speaker)]=[count,last_tick]
        if len(provenance)>MAX_PROVENANCE_ROWS:raise RuntimeError('continuous-social-binding:checkpoint-provenance-capacity')
        out._pending=pending;heapq.heapify(out._pending);out._provenance=provenance
        return out
