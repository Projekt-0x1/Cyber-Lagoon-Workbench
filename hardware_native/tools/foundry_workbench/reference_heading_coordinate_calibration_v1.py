#!/usr/bin/env python3
"""Bounded learned affine alignment between visual and vestibular heading coordinates."""
from __future__ import annotations


class HeadingCoordinateCalibrationV1:
    def __init__(self,window:int=8,min_support:int=2):
        self.window=int(window);self.min_support=int(min_support);self.pairs=[]
        if self.window<4 or self.min_support<2:raise ValueError('heading_calibration:window')

    def observe_pair(self,visual_coordinate:int,vestibular_angle:int):
        self.pairs.append((int(visual_coordinate),int(vestibular_angle)))
        if len(self.pairs)>self.window:del self.pairs[:-self.window]

    def relation(self):
        grouped={}
        for visual,vestibular in self.pairs:grouped.setdefault(visual,[]).append(vestibular)
        anchors=[]
        for visual,values in grouped.items():
            unique=set(values)
            if len(unique)!=1:return None
            if len(values)>=self.min_support:anchors.append((visual,values[0]))
        if len(anchors)<2:return None
        anchors.sort();x0,y0=anchors[0];x1,y1=anchors[-1]
        den=x1-x0
        if den==0:return None
        num=y1-y0
        for x,y in anchors:
            if (y-y0)*den!=num*(x-x0):return None
        return (x0,y0,num,den)

    def map_visual(self,visual_coordinate:int):
        relation=self.relation()
        if relation is None:return None
        x0,y0,num,den=relation
        numerator=y0*den+num*(int(visual_coordinate)-x0)
        sign=-1 if numerator*den<0 else 1
        return sign*((abs(numerator)+abs(den)//2)//abs(den))

    def checkpoint(self):
        return {'schema':1,'window':self.window,'min_support':self.min_support,'pairs':[list(x) for x in self.pairs]}

    @classmethod
    def restore(cls,data):
        if int(data.get('schema',0))!=1:raise ValueError('heading_calibration:checkpoint')
        out=cls(int(data['window']),int(data['min_support']))
        for pair in data.get('pairs',()):
            if len(pair)!=2:raise ValueError('heading_calibration:checkpoint')
            out.observe_pair(int(pair[0]),int(pair[1]))
        return out
