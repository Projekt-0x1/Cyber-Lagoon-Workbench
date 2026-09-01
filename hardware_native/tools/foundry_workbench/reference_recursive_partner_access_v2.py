#!/usr/bin/env python3
"""Consequence-certified partner shared access, explicitly not a belief model."""
from __future__ import annotations
from dataclasses import dataclass

Q=1<<16
MAX_ACCESS_ROWS=4096
MAX_ACCESS_CUES=24

@dataclass(frozen=True)
class PartnerSharedAccessV2:
    identity:int
    partner:int
    episode:int
    context:int
    world_context:int
    world_cues:tuple[int,...]
    tick:int
    authority:int=0

class RecursivePartnerAccessV2:
    def __init__(self):self._rows=[];self._withdrawn=set();self._next=1;self._tick=-1
    def _advance(self,tick):
        tick=int(tick)
        if tick<self._tick:raise ValueError('partner-access-v2:time-reversal')
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
    def observe_shared(self,partner,episode,context,world_cues,world_context,tick,qualified=True):
        self._advance(tick);partner=int(partner);episode=int(episode);world_context=int(world_context);world_cues=self.normalize(world_cues)
        if not qualified or partner<=0 or episode<=0 or partner in self._withdrawn:return 0
        old=next((r for r in self._rows if r.partner==partner and r.episode==episode),None)
        if old:return old.identity
        row=PartnerSharedAccessV2(self._next,partner,episode,int(context),world_context,world_cues,int(tick),0);self._next+=1
        if len(self._rows)>=MAX_ACCESS_ROWS:self._rows.pop(0)
        self._rows.append(row);return row.identity
    def partner_rows(self,partner):return tuple(r for r in self._rows if r.partner==int(partner))
    def shared_access_staleness_q16(self,partner,current_world_cues,current_world_context=0):
        partner=int(partner);rows=self.partner_rows(partner);current=self.normalize(current_world_cues);current_world_context=int(current_world_context)
        if partner<=0 or partner in self._withdrawn:return Q
        if not rows:return Q//2
        best=0
        for row in rows:
            sim=self.similarity_q16(current,row.world_cues)
            if current_world_context>0 and row.world_context==current_world_context:sim=min(Q,sim+Q//8)
            best=max(best,sim)
        return max(0,Q-best)
    def access_applicability_q16(self,partner,current_world_cues,current_world_context=0):
        return Q-self.shared_access_staleness_q16(partner,current_world_cues,current_world_context)
    def perspective_gap_q16(self,*_args,**_kwargs):
        raise RuntimeError('partner-access-v2:category-error-access-is-not-perspective')
    def inferred_belief(self,*_args,**_kwargs):
        raise RuntimeError('partner-access-v2:category-error-access-is-not-belief')
    def withdraw_source(self,source):
        source=int(source)
        if source<=0:raise ValueError('partner-access-v2:withdraw-source')
        self._withdrawn.add(source);return source
    def checkpoint(self):
        return {'schema':2,'tick':self._tick,'next':self._next,'withdrawn':sorted(self._withdrawn),
            'rows':[{'identity':r.identity,'partner':r.partner,'episode':r.episode,'context':r.context,'world_context':r.world_context,
                'world_cues':list(r.world_cues),'tick':r.tick,'authority':0} for r in self._rows]}
    @classmethod
    def restore(cls,data):
        schema=int(data.get('schema',0));out=cls()
        if schema==1:
            out._tick=int(data.get('tick',-1));out._next=int(data.get('next',1));out._withdrawn=set(map(int,data.get('withdrawn',())))
            out._rows=[PartnerSharedAccessV2(int(r['identity']),int(r['partner']),int(r['episode']),int(r['context']),int(r.get('regime',0)),tuple(map(int,r.get('cues',()))),int(r['tick']),0) for r in data.get('rows',())]
            return out
        if schema!=2:raise RuntimeError('partner-access-v2:checkpoint-schema')
        out._tick=int(data.get('tick',-1));out._next=int(data.get('next',1));out._withdrawn=set(map(int,data.get('withdrawn',())))
        out._rows=[PartnerSharedAccessV2(int(r['identity']),int(r['partner']),int(r['episode']),int(r['context']),int(r.get('world_context',0)),tuple(map(int,r.get('world_cues',()))),int(r['tick']),0) for r in data.get('rows',())]
        if len(out._rows)>MAX_ACCESS_ROWS:raise RuntimeError('partner-access-v2:capacity')
        return out
