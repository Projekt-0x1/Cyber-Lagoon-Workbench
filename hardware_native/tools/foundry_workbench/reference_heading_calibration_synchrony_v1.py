#!/usr/bin/env python3
"""Own visual/vestibular calibration pairing through resident organism time."""
from __future__ import annotations
from reference_heading_coordinate_calibration_v1 import HeadingCoordinateCalibrationV1


class HeadingCalibrationSynchronyV1:
    def __init__(self,calibration=None):
        self.calibration=calibration or HeadingCoordinateCalibrationV1()
        self._visual=None;self._vestibular=None;self._last_pair_tick=-1
        self.pair_count=0

    @staticmethod
    def _tick(organism):
        tick=int(getattr(organism,'tick_count',-1))
        if tick<0:raise ValueError('heading_synchrony:organism')
        return tick

    def _settle(self):
        if self._visual is None or self._vestibular is None:return False
        vt,v=self._visual;bt,b=self._vestibular
        if vt==bt:
            if vt==self._last_pair_tick:raise ValueError('heading_synchrony:duplicate_tick')
            self.calibration.observe_pair(v,b);self.pair_count+=1;self._last_pair_tick=vt
            self._visual=None;self._vestibular=None;return True
        if vt<bt:self._visual=None
        else:self._vestibular=None
        return False

    def observe_visual(self,organism,value:int):
        tick=self._tick(organism)
        if self._visual is not None and self._visual[0]==tick:raise ValueError('heading_synchrony:duplicate_visual')
        self._visual=(tick,int(value));return self._settle()

    def observe_vestibular(self,organism,value:int):
        tick=self._tick(organism)
        if self._vestibular is not None and self._vestibular[0]==tick:raise ValueError('heading_synchrony:duplicate_vestibular')
        self._vestibular=(tick,int(value));return self._settle()

    def map_visual(self,value:int):return self.calibration.map_visual(int(value))
    def relation(self):return self.calibration.relation()

    def checkpoint(self):return {'schema':1,'calibration':self.calibration.checkpoint()}

    @classmethod
    def restore(cls,data):
        if int(data.get('schema',0))!=1:raise ValueError('heading_synchrony:checkpoint')
        return cls(HeadingCoordinateCalibrationV1.restore(data['calibration']))
