#!/usr/bin/env python3
"""Learn independent left/right temporal widths around a resident lag center."""
from __future__ import annotations

from reference_heading_coordinate_calibration_v1 import HeadingCoordinateCalibrationV1

MAX_LAG=4
LAG_HISTORY=12
CENTER_QUORUM=3
SIDE_QUORUM=2


class HeadingCalibrationAsymmetricWindowV1:
    def __init__(self):
        self.calibration=HeadingCoordinateCalibrationV1()
        self.lag_history=[]
        self._visual=None;self._vestibular=None
        self.pair_count=0

    @staticmethod
    def _tick(organism):
        tick=int(getattr(organism,'tick_count',-1))
        if tick<0:raise ValueError('heading_asym:organism')
        return tick

    def _counts(self):
        counts={}
        for lag in self.lag_history:counts[lag]=counts.get(lag,0)+1
        return counts

    def learned_center(self):
        counts=self._counts()
        if not counts:return None
        best=max(counts.values());winners=[lag for lag,count in counts.items() if count==best]
        if best<CENTER_QUORUM or len(winners)!=1:return None
        return int(winners[0])

    def widths(self):
        center=self.learned_center()
        if center is None:return None
        counts=self._counts();left=right=0
        for distance in range(1,MAX_LAG+1):
            lag=center-distance
            if abs(lag)>MAX_LAG or counts.get(lag,0)<SIDE_QUORUM:break
            left=distance
        for distance in range(1,MAX_LAG+1):
            lag=center+distance
            if abs(lag)>MAX_LAG or counts.get(lag,0)<SIDE_QUORUM:break
            right=distance
        return left,right

    def _record_lag(self,lag:int):
        lag=int(lag)
        if abs(lag)>MAX_LAG:return False
        self.lag_history.append(lag)
        if len(self.lag_history)>LAG_HISTORY:del self.lag_history[:-LAG_HISTORY]
        return True

    def _settle(self):
        if self._visual is None or self._vestibular is None:return False
        vt,v=self._visual;bt,b=self._vestibular;lag=int(bt-vt)
        if abs(lag)>MAX_LAG:
            if vt<bt:self._visual=None
            else:self._vestibular=None
            return False
        self._record_lag(lag)
        center=self.learned_center();widths=self.widths();accepted=False
        if center is not None and widths is not None:
            left,right=widths;accepted=(center-left)<=lag<=(center+right)
        if accepted:
            self.calibration.observe_pair(v,b);self.pair_count+=1
        self._visual=None;self._vestibular=None
        return bool(accepted)

    def observe_visual(self,organism,value:int):
        tick=self._tick(organism)
        if self._visual is not None and self._visual[0]==tick:raise ValueError('heading_asym:duplicate_visual')
        self._visual=(tick,int(value));return self._settle()

    def observe_vestibular(self,organism,value:int):
        tick=self._tick(organism)
        if self._vestibular is not None and self._vestibular[0]==tick:raise ValueError('heading_asym:duplicate_vestibular')
        self._vestibular=(tick,int(value));return self._settle()

    def map_visual(self,value:int):return self.calibration.map_visual(int(value))
    def relation(self):return self.calibration.relation()

    def checkpoint(self):
        return {'schema':1,'lag_history':list(self.lag_history),'pair_count':self.pair_count,
                'calibration':self.calibration.checkpoint()}

    @classmethod
    def restore(cls,data):
        if int(data.get('schema',0))!=1:raise ValueError('heading_asym:checkpoint')
        out=cls();out.lag_history=[int(x) for x in data.get('lag_history',())]
        if len(out.lag_history)>LAG_HISTORY or any(abs(x)>MAX_LAG for x in out.lag_history):raise ValueError('heading_asym:checkpoint')
        out.pair_count=int(data.get('pair_count',0));out.calibration=HeadingCoordinateCalibrationV1.restore(data['calibration'])
        return out
