#!/usr/bin/env python3
"""Equal-weight fusion of two raw directional social-attention cues."""
from __future__ import annotations
from reference_raw_pointing_motion_v1 import RawPointingMotionV1

class SocialDirectionalCueFusionV1:
    @staticmethod
    def resolve(tracker,first_motion,second_motion):
        first={entity:score for score,entity in RawPointingMotionV1.scores(tracker,first_motion)}
        second={entity:score for score,entity in RawPointingMotionV1.scores(tracker,second_motion)}
        common=set(first).intersection(second)
        if not common:return 0
        ranked=sorted((first[e]+second[e],int(e)) for e in common)
        best=ranked[0][0];wins=[entity for score,entity in ranked if score==best]
        return wins[0] if len(wins)==1 else 0
