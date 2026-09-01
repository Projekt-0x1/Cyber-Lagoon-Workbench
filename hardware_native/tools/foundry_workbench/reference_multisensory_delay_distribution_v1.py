#!/usr/bin/env python3
from __future__ import annotations
from reference_authenticated_sensor_timestamp_v1 import AuthenticatedSensorTimestampV1
HISTORY=12;MIN_SAMPLES=5
class MultisensoryDelayDistributionV1:
    def __init__(self):self.history=[]
    @staticmethod
    def _delay(camera_receipt,imu_receipt,clock_alignment):
        for r in (camera_receipt,imu_receipt):
            expected=AuthenticatedSensorTimestampV1.commitment(r.source,r.sequence,r.timestamp_us,r.payload_digest)
            if str(r.commitment)!=expected:raise ValueError('delay_distribution:receipt')
        mapped=clock_alignment.map_camera(int(camera_receipt.timestamp_us))
        if mapped is None:return None
        return int(imu_receipt.timestamp_us)-int(mapped)
    @staticmethod
    def _median(rows):
        rows=sorted(int(x) for x in rows);return rows[(len(rows)-1)//2]
    def observe(self,camera_receipt,imu_receipt,clock_alignment):
        delay=self._delay(camera_receipt,imu_receipt,clock_alignment)
        if delay is None:return False
        self.history.append(delay)
        if len(self.history)>HISTORY:del self.history[:-HISTORY]
        return self.profile() is not None
    def profile(self):
        if len(self.history)<MIN_SAMPLES:return None
        center=self._median(self.history);mad=self._median(abs(x-center) for x in self.history);radius=2*mad
        support=sum(abs(x-center)<=radius for x in self.history)
        if support<=len(self.history)//2:return None
        return center,radius,support
    def accepts(self,camera_receipt,imu_receipt,clock_alignment):
        p=self.profile();delay=self._delay(camera_receipt,imu_receipt,clock_alignment)
        return False if p is None or delay is None else abs(delay-p[0])<=p[1]
    def checkpoint(self):return {'schema':1,'history':list(self.history)}
    @classmethod
    def restore(cls,d):
        if int(d.get('schema',0))!=1:raise ValueError('delay_distribution:checkpoint')
        out=cls();out.history=[int(x) for x in d.get('history',())]
        if len(out.history)>HISTORY:raise ValueError('delay_distribution:checkpoint')
        return out
