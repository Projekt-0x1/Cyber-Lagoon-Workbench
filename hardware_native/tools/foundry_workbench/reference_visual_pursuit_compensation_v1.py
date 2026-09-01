#!/usr/bin/env python3
"""World-relative heading correction from resident motor/reafferent evidence."""
from __future__ import annotations

from reference_visual_heading_center_v1 import SCALE


class VisualPursuitCompensationV1:
    """Read-only correction; pursuit displacement is derived from settled motor state."""
    def __init__(self,max_lag:int=8):
        self.max_lag=int(max_lag)
        if not 1<=self.max_lag<=64:raise ValueError('visual_pursuit:lag')

    def resident_delta(self,organism)->int:
        actions=getattr(organism,'motor_actions',())
        if not actions:return 0
        action=actions[-1]
        if (not bool(getattr(action,'settled',False))
                or not bool(getattr(action,'independent_consequence',False))
                or int(getattr(action,'effect',0))<=0):return 0
        before=tuple(getattr(action,'state_before',()))
        after=tuple(getattr(action,'state_after',()))
        if len(before)!=1 or len(after)!=1:return 0
        if tuple(getattr(organism,'body_state',()))!=after:return 0
        if int(getattr(organism,'body_state_occurrence',0))<=0:return 0
        if int(getattr(organism,'body_state_source',0))!=int(getattr(action,'source',0)):return 0
        age=int(getattr(organism,'tick_count',0))-int(getattr(action,'tick',0))
        if age<0 or age>self.max_lag:return 0
        delta=int(after[0])-int(before[0])
        return delta if -7<=delta<=7 else 0

    def correct(self,retinal_center,organism):
        if retinal_center is None:return None
        cx,cy,sign=map(int,retinal_center)
        delta=self.resident_delta(organism)
        return (cx+delta*SCALE,cy,sign)
