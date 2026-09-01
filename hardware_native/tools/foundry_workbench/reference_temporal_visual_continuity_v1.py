#!/usr/bin/env python3
"""Transient adjacency adapter over already-learned visual features."""
from __future__ import annotations
from reference_unsupervised_perceptual_features_v1 import UnsupervisedPerceptualFeaturesV1


class TemporalVisualContinuityV1:
    """Learn unlabeled adjacency with one transient previous-frame feature."""
    def __init__(self,learner=None):
        self.learner=learner or UnsupervisedPerceptualFeaturesV1()
        self.previous=()

    @staticmethod
    def _single(features):
        row=tuple(sorted(set(int(x) for x in features if int(x)>0)))
        return row[0] if len(row)==1 else 0

    def observe_features(self,features):
        current=self._single(features)
        learned=0
        if current and self.previous and current!=self.previous[0]:
            self.learner.observe_scene((self.previous[0],current))
            learned=self.learner.feature(self.previous[0],current)
        self.previous=(() if not current else (current,))
        return int(learned)

    def gap(self):
        self.previous=()

    def relation(self,left:int,right:int)->int:
        left=int(left);right=int(right)
        if left<=0 or right<=0 or left==right:return 0
        return int(self.learner.feature(left,right))

    def lesion_relation(self,left:int,right:int):
        self.learner.lesion_pair(int(left),int(right))

    def checkpoint(self):
        return {'schema':1,'learner':self.learner.checkpoint()}

    @classmethod
    def restore(cls,data):
        if int(data.get('schema',0))!=1:raise ValueError('temporal_visual:checkpoint')
        return cls(UnsupervisedPerceptualFeaturesV1.restore(data['learner']))
