#!/usr/bin/env python3
from __future__ import annotations
from reference_authenticated_sensor_timestamp_v1 import AuthenticatedSensorTimestampV1
class SensorClockAlignmentV1:
    def __init__(self,history=8):self.history=int(history);self.pairs=[];self._camera=None;self._imu=None
    @staticmethod
    def _tick(o):
        t=int(getattr(o,'tick_count',-1))
        if t<0:raise ValueError('sensor_clock:organism')
        return t
    def _settle(self):
        if self._camera is None or self._imu is None:return False
        ct,c=self._camera;it,i=self._imu
        if ct==it:
            self.pairs.append((int(c.timestamp_us),int(i.timestamp_us)))
            if len(self.pairs)>self.history:del self.pairs[:-self.history]
            self._camera=self._imu=None;return True
        if ct<it:self._camera=None
        else:self._imu=None
        return False
    @staticmethod
    def _validate(r):
        expected=AuthenticatedSensorTimestampV1.commitment(r.source,r.sequence,r.timestamp_us,r.payload_digest)
        if str(r.commitment)!=expected:raise ValueError('sensor_clock:receipt')
    def observe_camera(self,o,r):self._validate(r);self._camera=(self._tick(o),r);return self._settle()
    def observe_imu(self,o,r):self._validate(r);self._imu=(self._tick(o),r);return self._settle()
    def relation(self):
        rows=sorted(set(self.pairs))
        if len(rows)<3:return None
        x0,y0=rows[0];x1,y1=rows[-1];den=x1-x0
        if den==0:return None
        num=y1-y0
        if any((y-y0)*den!=num*(x-x0) for x,y in rows):return None
        return x0,y0,num,den
    def map_camera(self,timestamp_us):
        r=self.relation()
        if r is None:return None
        x0,y0,num,den=r;n=y0*den+num*(int(timestamp_us)-x0);sign=-1 if n*den<0 else 1
        return sign*((abs(n)+abs(den)//2)//abs(den))
    def checkpoint(self):return {'schema':1,'history':self.history,'pairs':[list(x) for x in self.pairs]}
    @classmethod
    def restore(cls,d):
        if int(d.get('schema',0))!=1:raise ValueError('sensor_clock:checkpoint')
        out=cls(int(d['history']));out.pairs=[(int(a),int(b)) for a,b in d.get('pairs',())];return out
