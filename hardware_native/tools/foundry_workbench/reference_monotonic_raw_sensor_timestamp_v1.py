#!/usr/bin/env python3
from __future__ import annotations
import time
from reference_authenticated_sensor_timestamp_v1 import AuthenticatedSensorTimestampV1,SensorTimestampReceiptV1

class MonotonicRawSensorTimestampV1:
    def __init__(self):
        self.last_timestamp_ns=0;self.stamped=set()
    @staticmethod
    def _payload_digest(sensor):return AuthenticatedSensorTimestampV1._payload_digest(sensor)
    def stamp(self,sensor):
        source=int(getattr(sensor,'active_source',0));sequence=int(getattr(sensor,'active_sequence',0))
        if source<=0 or sequence<=0:raise ValueError('monotonic_sensor_timestamp:state')
        key=(source,sequence)
        if key in self.stamped:raise ValueError('monotonic_sensor_timestamp:duplicate')
        payload=self._payload_digest(sensor)
        now=int(time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW))
        if now<=self.last_timestamp_ns:now=self.last_timestamp_ns+1
        timestamp_us=max(1,now//1000)
        commitment=AuthenticatedSensorTimestampV1.commitment(source,sequence,timestamp_us,payload)
        self.last_timestamp_ns=now;self.stamped.add(key)
        return SensorTimestampReceiptV1(source,sequence,timestamp_us,payload,commitment)
    def checkpoint(self):
        return {'schema':1,'last_timestamp_ns':self.last_timestamp_ns,'stamped':[list(x) for x in sorted(self.stamped)]}
    @classmethod
    def restore(cls,d):
        if int(d.get('schema',0))!=1:raise ValueError('monotonic_sensor_timestamp:checkpoint')
        out=cls();out.last_timestamp_ns=int(d.get('last_timestamp_ns',0));out.stamped={(int(a),int(b)) for a,b in d.get('stamped',())}
        if out.last_timestamp_ns<0 or any(a<=0 or b<=0 for a,b in out.stamped):raise ValueError('monotonic_sensor_timestamp:checkpoint')
        return out
