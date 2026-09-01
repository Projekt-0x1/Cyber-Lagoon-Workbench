#!/usr/bin/env python3
"""Fuse two raw directional cues using learned opaque-lane reliability."""
from __future__ import annotations
from reference_raw_pointing_motion_v1 import RawPointingMotionV1
class ReliabilityWeightedDirectionalFusionV1:
    @staticmethod
    def resolve(tracker,first_motion,first_lane,second_motion,second_lane,reliability):
        first={e:s for s,e in RawPointingMotionV1.scores(tracker,first_motion)};second={e:s for s,e in RawPointingMotionV1.scores(tracker,second_motion)}
        common=set(first).intersection(second)
        if not common:return 0
        w1=int(reliability.weight_q16(first_lane));w2=int(reliability.weight_q16(second_lane))
        ranked=sorted((first[e]*w1+second[e]*w2,int(e)) for e in common)
        best=ranked[0][0];wins=[e for score,e in ranked if score==best]
        return wins[0] if len(wins)==1 else 0
