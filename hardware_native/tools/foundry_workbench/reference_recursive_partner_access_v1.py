#!/usr/bin/env python3
"""Partner-local consequence-certified access and perspective divergence.

This is not a proposition-level theory-of-mind store. It records only which bounded causal
cues were jointly established with a partner by positive independent common-ground events.
Perspective divergence is a zero-authority control signal about applicability of that shared
history to the current situation.
"""
from __future__ import annotations
from dataclasses import dataclass

Q=1<<16
MAX_ACCESS_ROWS=4096
MAX_ACCESS_CUES=24

@dataclass(frozen=True)
class PartnerAccessV1:
    identity:int
    partner:int
    episode:int
    context:int
    regime:int
    cues:tuple[int,...]
    tick:int
    authority:int=0

class RecursivePartnerAccessV1:
    def __init__(self):
        self._rows=[];self._withdrawn=set();self._next=1;self._tick=-1

    def _advance(self,tick):
        tick=int(tick)
        if tick<self._tick:raise ValueError('partner-access:time-reversal')
        self._tick=tick

    @staticmethod
    def normalize(cues):return tuple(sorted(set(int(x) for x in cues if int(x)>0)))[:MAX_ACCESS_CUES]

    @staticmethod
    def similarity_q16(left,right):
        left=set(map(int,left));right=set(map(int,right))
        if not left and not right:return Q
        if not left or not right:return 0
        return (2*len(left&right)*Q)//(len(left)+len(right))

    @property
    def evidence_count(self):return len(self._rows)

    def observe_shared(self,partner,episode,context,cues,regime,tick,qualified=True):
        self._advance(tick);partner=int(partner);episode=int(episode);regime=int(regime)
        cues=self.normalize(cues)
        if not qualified or partner<=0 or episode<=0 or partner in self._withdrawn:return 0
        existing=next((r for r in self._rows if r.partner==partner and r.episode==episode),None)
        if existing:return int(existing.identity)
        row=PartnerAccessV1(self._next,partner,episode,int(context),regime,cues,int(tick),0);self._next+=1
        if len(self._rows)>=MAX_ACCESS_ROWS:self._rows.pop(0)
        self._rows.append(row);return int(row.identity)

    def partner_rows(self,partner):return tuple(r for r in self._rows if int(r.partner)==int(partner))

    def perspective_gap_q16(self,partner,current_cues,current_regime=0):
        partner=int(partner);current_regime=int(current_regime);rows=self.partner_rows(partner)
        if partner<=0 or partner in self._withdrawn:return Q
        if not rows:return Q//2  # unknown access, not asserted false belief
        current=self.normalize(current_cues);best=0
        for row in rows:
            similarity=self.similarity_q16(current,row.cues)
            if current_regime>0 and int(row.regime)==current_regime:similarity=min(Q,similarity+Q//8)
            best=max(best,similarity)
        return max(0,Q-best)

    def applicability_q16(self,partner,current_cues,current_regime=0):
        return Q-self.perspective_gap_q16(partner,current_cues,current_regime)

    def withdraw_source(self,source):
        source=int(source)
        if source<=0:raise ValueError('partner-access:withdraw-source')
        self._withdrawn.add(source);return source

    def checkpoint(self):
        return {'schema':1,'tick':self._tick,'next':self._next,'withdrawn':sorted(self._withdrawn),
            'rows':[{'identity':r.identity,'partner':r.partner,'episode':r.episode,'context':r.context,
                'regime':r.regime,'cues':list(r.cues),'tick':r.tick,'authority':0} for r in self._rows]}

    @classmethod
    def restore(cls,data):
        if int(data.get('schema',0))!=1:raise RuntimeError('partner-access:checkpoint-schema')
        out=cls();out._tick=int(data.get('tick',-1));out._next=int(data.get('next',1));out._withdrawn=set(map(int,data.get('withdrawn',())))
        out._rows=[PartnerAccessV1(int(r['identity']),int(r['partner']),int(r['episode']),int(r['context']),int(r.get('regime',0)),tuple(map(int,r.get('cues',()))),int(r['tick']),0) for r in data.get('rows',())]
        if len(out._rows)>MAX_ACCESS_ROWS:raise RuntimeError('partner-access:capacity')
        return out
