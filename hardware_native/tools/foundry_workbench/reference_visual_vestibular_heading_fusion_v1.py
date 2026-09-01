#!/usr/bin/env python3
"""Bounded reliability-based fusion of independently owned heading coordinates."""
from __future__ import annotations


class VisualVestibularHeadingFusionV1:
    """Fast cue weighting from resident recent dispersion; no fused-state persistence."""
    def __init__(self,window:int=6,min_samples:int=3):
        self.window=int(window);self.min_samples=int(min_samples)
        if not 3<=self.window<=32 or not 2<=self.min_samples<=self.window:
            raise ValueError('heading_fusion:window')
        self.visual=[];self.vestibular=[]

    def _observe(self,rows,value:int):
        value=int(value);rows.append(value)
        if len(rows)>self.window:del rows[:-self.window]
        return value

    def observe_visual(self,value:int):return self._observe(self.visual,value)
    def observe_vestibular(self,value:int):return self._observe(self.vestibular,value)

    @staticmethod
    def _dispersion(rows):
        n=len(rows);total=sum(rows)
        return sum((int(x)*n-total)**2 for x in rows)//max(1,n*n)

    def _weight(self,rows):
        if len(rows)<self.min_samples:return 0
        return 1_000_000//(1+self._dispersion(rows))

    def weights(self):return self._weight(self.visual),self._weight(self.vestibular)

    def fuse(self,current_visual:int,current_vestibular:int):
        visual=int(current_visual);vestibular=int(current_vestibular)
        wv,wb=self.weights()
        if wv<=0 or wb<=0:return None
        return (visual*wv+vestibular*wb+(wv+wb)//2)//(wv+wb)

    def reset_visual(self):self.visual=[]
    def reset_vestibular(self):self.vestibular=[]

    def checkpoint(self):
        return {'schema':1,'window':self.window,'min_samples':self.min_samples,
                'visual':list(self.visual),'vestibular':list(self.vestibular)}

    @classmethod
    def restore(cls,data):
        if int(data.get('schema',0))!=1:raise ValueError('heading_fusion:checkpoint')
        out=cls(int(data['window']),int(data['min_samples']))
        for value in data.get('visual',()):out.observe_visual(int(value))
        for value in data.get('vestibular',()):out.observe_vestibular(int(value))
        return out
