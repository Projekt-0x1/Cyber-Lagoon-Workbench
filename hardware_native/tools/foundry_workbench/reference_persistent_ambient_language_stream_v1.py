#!/usr/bin/env python3
"""Bounded semantically-blind public bystander text chronology.

This owner schedules authenticated raw-byte contacts only.  It has no semantic, attention,
truth, trust, platform, toxicity, language, turn-taking, or public-action authority.
"""
from __future__ import annotations
from dataclasses import dataclass,asdict
import hashlib,heapq

MAX_PENDING_AMBIENT_CONTACTS=512
MAX_AMBIENT_CONTACT_BYTES=65536
MAX_PENDING_AMBIENT_BYTES=2*1024*1024


def semantically_blind_index(seed,sequence,count):
    """Choose one pool index without reading any candidate payload bytes."""
    seed=int(seed);sequence=int(sequence);count=int(count)
    if count<=0 or sequence<0:raise ValueError('ambient-stream:sample')
    raw=f'{seed}:{sequence}:{count}'.encode('ascii')
    return int.from_bytes(hashlib.sha256(raw).digest()[:8],'big')%count


@dataclass(frozen=True,order=True)
class AmbientLanguageContactV1:
    tick:int
    sequence:int
    source:int
    raw:tuple[int,...]


class PersistentAmbientLanguageStreamV1:
    """Causal-time ordered pending public text; drained bytes never become a transcript."""
    def __init__(self):
        self._pending=[]
        self._pending_bytes=0
        self._next_sequence=1
        self._last_drained_tick=-1
        self.processed_contacts=0  # observer meter; not checkpoint authority
        self.processed_bytes=0     # observer meter; not checkpoint authority

    @staticmethod
    def _raw(value):
        if isinstance(value,(bytes,bytearray)):raw=tuple(value)
        else:raw=tuple(map(int,value))
        if (not raw or len(raw)>MAX_AMBIENT_CONTACT_BYTES
                or any(x<0 or x>255 for x in raw)):
            raise ValueError('ambient-stream:raw')
        return raw

    def admit(self,tick,source,raw):
        tick=int(tick);source=int(source);raw=self._raw(raw)
        if tick<0 or source<=0:raise ValueError('ambient-stream:contact')
        if tick<self._last_drained_tick:raise ValueError('ambient-stream:late-contact')
        if len(self._pending)>=MAX_PENDING_AMBIENT_CONTACTS:raise RuntimeError('ambient-stream:capacity')
        if self._pending_bytes+len(raw)>MAX_PENDING_AMBIENT_BYTES:raise RuntimeError('ambient-stream:byte-capacity')
        sequence=self._next_sequence;self._next_sequence+=1
        event=AmbientLanguageContactV1(tick,sequence,source,raw)
        heapq.heappush(self._pending,event);self._pending_bytes+=len(raw)
        return sequence

    def admit_from_pool(self,tick,seed,sequence,pool):
        """Transport helper: pool selection is a function of entropy/order/size, never contents."""
        pool=tuple(pool)
        if not pool:raise ValueError('ambient-stream:pool')
        index=semantically_blind_index(seed,sequence,len(pool))
        source,raw=pool[index]
        return index,self.admit(tick,source,raw)

    def drain_until(self,adult,tick):
        """Expose every due post to ambient receptive learning without foregrounding it."""
        tick=int(tick)
        if tick<self._last_drained_tick:raise ValueError('ambient-stream:time-reversal')
        processed=[]
        while self._pending and self._pending[0].tick<=tick:
            capacity=getattr(adult,'ambient_language_capacity_q16',None)
            if callable(capacity) and int(capacity())<=0:
                # Resource pressure belongs to the organism.  Do not drop/replay the due post and
                # do not move the chronology watermark past unprocessed sensory contact.
                self._last_drained_tick=max(self._last_drained_tick,min(tick,self._pending[0].tick))
                return tuple(processed)
            event=heapq.heappop(self._pending);self._pending_bytes-=len(event.raw)
            learned=bool(adult.observe_ambient_language_contact(event.raw,event.source))
            processed.append((event.tick,event.sequence,event.source,len(event.raw),learned))
            self.processed_contacts+=1;self.processed_bytes+=len(event.raw)
        self._last_drained_tick=tick
        return tuple(processed)

    @property
    def pending_count(self):return len(self._pending)
    @property
    def pending_bytes(self):return self._pending_bytes

    def checkpoint(self):
        return {
            'schema':1,'next_sequence':self._next_sequence,'last_drained_tick':self._last_drained_tick,
            'pending_bytes':self._pending_bytes,
            'pending':[{'tick':e.tick,'sequence':e.sequence,'source':e.source,'raw':list(e.raw)}
                       for e in sorted(self._pending)],
        }

    @classmethod
    def restore(cls,data):
        if int(data.get('schema',0))!=1:raise RuntimeError('ambient-stream:checkpoint-schema')
        out=cls();out._next_sequence=int(data.get('next_sequence',0));out._last_drained_tick=int(data.get('last_drained_tick',-1))
        if out._next_sequence<1 or out._last_drained_tick<-1:raise RuntimeError('ambient-stream:checkpoint-state')
        pending=[];seen=set();size=0
        for row in data.get('pending',()):
            event=AmbientLanguageContactV1(int(row['tick']),int(row['sequence']),int(row['source']),cls._raw(row['raw']))
            if (event.tick<out._last_drained_tick or event.sequence<=0 or event.sequence in seen
                    or event.source<=0):raise RuntimeError('ambient-stream:checkpoint-event')
            seen.add(event.sequence);pending.append(event);size+=len(event.raw)
        if len(pending)>MAX_PENDING_AMBIENT_CONTACTS or size>MAX_PENDING_AMBIENT_BYTES:
            raise RuntimeError('ambient-stream:checkpoint-capacity')
        if pending and max(e.sequence for e in pending)>=out._next_sequence:
            raise RuntimeError('ambient-stream:checkpoint-sequence')
        if int(data.get('pending_bytes',size))!=size:raise RuntimeError('ambient-stream:checkpoint-bytes')
        out._pending=pending;heapq.heapify(out._pending);out._pending_bytes=size;return out
