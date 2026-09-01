#!/usr/bin/env python3
"""Content-neutral owned social gaze/head-direction trajectory stream boundary."""
from __future__ import annotations
import hashlib
from reference_gesture_sensor_ingress_v1 import GestureSensorIngressV1,MAX_GESTURE_SAMPLES

class GazeSensorIngressV1(GestureSensorIngressV1):
    @staticmethod
    def motion_digest(samples)->str:
        rows=tuple((int(y),int(x)) for y,x in samples)
        if not 3<=len(rows)<=MAX_GESTURE_SAMPLES:raise ValueError('gaze_sensor:shape')
        h=hashlib.sha256(b'gaze-sensor-motion-v1\0')
        for y,x in rows:
            if not -(1<<31)<=y<(1<<31) or not -(1<<31)<=x<(1<<31):raise ValueError('gaze_sensor:coordinate')
            h.update(int(y).to_bytes(4,'little',signed=True));h.update(int(x).to_bytes(4,'little',signed=True))
        h.update(len(rows).to_bytes(2,'little'));return h.hexdigest()
