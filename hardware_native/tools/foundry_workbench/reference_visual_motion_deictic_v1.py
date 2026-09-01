#!/usr/bin/env python3
"""Learn/use deictic spatial profiles from raw pointing motion without host target coordinates."""
from __future__ import annotations
from reference_raw_pointing_motion_v1 import RawPointingMotionV1
from reference_visual_deictic_distance_ecology_v1 import VisualDeicticDistanceEcologyV1

class VisualMotionDeicticV1:
    @staticmethod
    def observe(adult,tracker,ecology,raw,motion,source):
        target=RawPointingMotionV1.target(tracker,motion);ray=RawPointingMotionV1.ray(motion)
        if not target or ray is None:return 0
        row=tracker.active.get(target);oy,ox,_vy,_vx=ray
        return ecology.observe(adult,tracker,raw,int(row[0]),int(row[1]),oy,ox,source)
    @staticmethod
    def resolve(adult,tracker,ecology,raw,motion):
        target=RawPointingMotionV1.target(tracker,motion);ray=RawPointingMotionV1.ray(motion)
        if not target or ray is None:return 0
        row=tracker.active.get(target);oy,ox,_vy,_vx=ray
        resolved=ecology.resolve(adult,tracker,raw,int(row[0]),int(row[1]),oy,ox)
        return target if resolved==target else 0
