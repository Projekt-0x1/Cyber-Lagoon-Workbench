#!/usr/bin/env python3
"""Stateless horizontal heading coordinate from a resolved translation residual."""
from __future__ import annotations
import math
from reference_population_v1 import mix64

ANGLE_SCALE=256
HEADING_TAG=0x7E571B


class VestibularTranslationHeadingV1:
    @staticmethod
    def from_resolution(resolution):
        if resolution is None:return None
        residual=tuple(resolution.get('residual',()))
        feature=int(resolution.get('translation_feature',0))
        magnitude=int(resolution.get('translation_magnitude',0))
        if len(residual)!=2 or feature<=0 or magnitude<=0:return None
        lateral,forward=map(int,residual)
        if lateral==0 and forward==0:return None
        angle=math.degrees(math.atan2(lateral,forward))
        return int(round(angle*ANGLE_SCALE))

    @staticmethod
    def token(angle_q8:int):
        value=mix64(HEADING_TAG^mix64(int(angle_q8)&((1<<64)-1)))
        return int(value&((1<<63)-1) or 1)
