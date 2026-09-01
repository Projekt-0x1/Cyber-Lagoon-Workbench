#!/usr/bin/env python3
"""Stateless head-to-trunk transform over authenticated resident body state."""
from __future__ import annotations

from reference_visual_heading_center_v1 import SCALE
from reference_visual_vestibular_reference_frame_v1 import VisualVestibularReferenceFrameV1

HEAD_NEUTRAL=200


class VisualVestibularBodyFrameV1:
    @staticmethod
    def resident_head_position(organism):
        body=tuple(getattr(organism,'body_state',()))
        source=int(getattr(organism,'body_state_source',0))
        occurrence=int(getattr(organism,'body_state_occurrence',0))
        withdrawn=set(map(int,getattr(organism,'withdrawn_sources',())))
        if len(body)<2 or source<=0 or occurrence<=0 or source in withdrawn:return None
        value=int(body[1])
        if not HEAD_NEUTRAL-16<=value<=HEAD_NEUTRAL+16:return None
        return value-HEAD_NEUTRAL

    @classmethod
    def to_trunk(cls,head_center,organism):
        if head_center is None:return None
        head=cls.resident_head_position(organism)
        if head is None:return None
        cx,cy,sign=map(int,head_center)
        return (cx+head*SCALE,cy,sign)

    @classmethod
    def visual_to_trunk(cls,retinal_center,organism):
        return cls.to_trunk(VisualVestibularReferenceFrameV1.to_head(retinal_center,organism),organism)
