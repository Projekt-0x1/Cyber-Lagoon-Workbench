#!/usr/bin/env python3
"""Slow conflict-driven calibration kept separate from fast cue reliability."""
from __future__ import annotations

CALIBRATION_QUORUM=4


class SlowVisualVestibularCalibrationV1:
    """Persistent modality offsets from repeated same-sign cross-modal conflict."""
    def __init__(self):
        self.visual_offset=0;self.vestibular_offset=0
        self.pending_sign=0;self.pending_count=0

    def calibrate_visual(self,value:int)->int:return int(value)+self.visual_offset
    def calibrate_vestibular(self,value:int)->int:return int(value)+self.vestibular_offset

    def observe_pair(self,visual:int,vestibular:int):
        visual=self.calibrate_visual(int(visual));vestibular=self.calibrate_vestibular(int(vestibular))
        diff=vestibular-visual
        if diff==0:
            self.pending_sign=0;self.pending_count=0;return False
        sign=1 if diff>0 else -1
        if sign!=self.pending_sign:self.pending_sign=sign;self.pending_count=1
        else:self.pending_count+=1
        if self.pending_count<CALIBRATION_QUORUM:return False
        self.visual_offset+=sign
        self.vestibular_offset-=2*sign
        self.pending_count=0
        return True

    def lesion_visual(self):self.visual_offset=0
    def lesion_vestibular(self):self.vestibular_offset=0

    def checkpoint(self):
        return {'schema':1,'visual_offset':self.visual_offset,'vestibular_offset':self.vestibular_offset,
                'pending_sign':self.pending_sign,'pending_count':self.pending_count}

    @classmethod
    def restore(cls,data):
        if int(data.get('schema',0))!=1:raise ValueError('slow_multisensory:checkpoint')
        out=cls();out.visual_offset=int(data.get('visual_offset',0));out.vestibular_offset=int(data.get('vestibular_offset',0))
        out.pending_sign=int(data.get('pending_sign',0));out.pending_count=int(data.get('pending_count',0))
        if out.pending_sign not in (-1,0,1) or not 0<=out.pending_count<CALIBRATION_QUORUM:raise ValueError('slow_multisensory:checkpoint')
        return out
