#!/usr/bin/env python3
"""Stateless eye-to-head transform over authenticated resident body state."""
from __future__ import annotations

from reference_visual_heading_center_v1 import SCALE

EYE_NEUTRAL=100


class VisualVestibularReferenceFrameV1:
    """Transform current retinal heading with resident one-dimensional eye/body position."""

    @staticmethod
    def resident_eye_position(organism):
        body=tuple(getattr(organism,'body_state',()))
        source=int(getattr(organism,'body_state_source',0))
        occurrence=int(getattr(organism,'body_state_occurrence',0))
        withdrawn=set(map(int,getattr(organism,'withdrawn_sources',())))
        if len(body)<1 or source<=0 or occurrence<=0 or source in withdrawn:return None
        value=int(body[0])
        if not EYE_NEUTRAL-16<=value<=EYE_NEUTRAL+16:return None
        return value-EYE_NEUTRAL

    @classmethod
    def to_head(cls,retinal_center,organism):
        if retinal_center is None:return None
        eye=cls.resident_eye_position(organism)
        if eye is None:return None
        cx,cy,sign=map(int,retinal_center)
        return (cx+eye*SCALE,cy,sign)
