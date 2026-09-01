#!/usr/bin/env python3
"""Content-free local-flow displacement and learned multi-step trajectory features."""
from __future__ import annotations
from reference_population_v1 import mix64
from reference_raw_visual_elements_v1 import RawVisualElementsV1
from reference_unsupervised_perceptual_features_v1 import UnsupervisedPerceptualFeaturesV1

MOTION_TAG=0xF10A01
SCALE=256


class VisualMotionTrajectoryV1:
    """Derive direction/magnitude from raw frames; learn motion-event conjunctions unlabeled."""
    def __init__(self,learner=None):
        self.learner=learner or UnsupervisedPerceptualFeaturesV1()
        self.previous_centroid=None
        self.previous_motion=0

    @staticmethod
    def _token(direction:int,magnitude:int)->int:
        value=mix64(MOTION_TAG^mix64(int(direction)+0x9E3779B97F4A7C15)^mix64(int(magnitude)+17))
        return int(value&((1<<63)-1) or 1)

    @staticmethod
    def active_centroid(frame):
        rows=RawVisualElementsV1._matrix(frame);points=[]
        for y in range(len(rows)-1):
            for x in range(len(rows[0])-1):
                if RawVisualElementsV1._events(rows[y][x],rows[y][x+1],rows[y+1][x],rows[y+1][x+1]):
                    points.append((x,y))
        if not points:return None
        n=len(points)
        return (sum(x for x,_ in points)*SCALE//n,sum(y for _,y in points)*SCALE//n)

    @staticmethod
    def _direction(dx:int,dy:int)->int:
        if dx==0 and dy==0:return 0
        ax,ay=abs(dx),abs(dy)
        if ax>=ay*2:return 1 if dx>0 else 5
        if ay>=ax*2:return 3 if dy>0 else 7
        if dx>0 and dy>0:return 2
        if dx<0 and dy>0:return 4
        if dx<0 and dy<0:return 6
        return 8

    @staticmethod
    def _magnitude(dx:int,dy:int)->int:
        pixels=max(abs(int(dx)),abs(int(dy)))//SCALE
        if pixels<=0:return 0
        return min(7,int(pixels))

    def observe_frame(self,frame,contiguous:bool):
        centroid=self.active_centroid(frame)
        if not contiguous:
            self.previous_centroid=centroid;self.previous_motion=0;return (0,0)
        motion=0;trajectory=0
        if centroid is not None and self.previous_centroid is not None:
            dx=centroid[0]-self.previous_centroid[0];dy=centroid[1]-self.previous_centroid[1]
            direction=self._direction(dx,dy);magnitude=self._magnitude(dx,dy)
            if direction and magnitude:motion=self._token(direction,magnitude)
        if motion and self.previous_motion and motion!=self.previous_motion:
            self.learner.observe_scene((self.previous_motion,motion))
            trajectory=self.learner.feature(self.previous_motion,motion)
        self.previous_centroid=centroid;self.previous_motion=int(motion)
        return int(motion),int(trajectory)

    def gap(self):
        self.previous_centroid=None;self.previous_motion=0

    def trajectory(self,first_motion:int,second_motion:int)->int:
        first_motion=int(first_motion);second_motion=int(second_motion)
        if first_motion<=0 or second_motion<=0 or first_motion==second_motion:return 0
        return int(self.learner.feature(first_motion,second_motion))

    def lesion_trajectory(self,first_motion:int,second_motion:int):
        self.learner.lesion_pair(int(first_motion),int(second_motion))

    def checkpoint(self):return {'schema':1,'learner':self.learner.checkpoint()}

    @classmethod
    def restore(cls,data):
        if int(data.get('schema',0))!=1:raise ValueError('visual_motion:checkpoint')
        return cls(UnsupervisedPerceptualFeaturesV1.restore(data['learner']))
