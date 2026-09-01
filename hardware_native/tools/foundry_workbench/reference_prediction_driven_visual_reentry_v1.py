#!/usr/bin/env python3
from __future__ import annotations
import time
from reference_recurrent_visual_heading_v1 import RecurrentVisualHeadingV1,RecurrentHeadingOccurrenceV1
from reference_visual_heading_center_v1 import VisualHeadingCenterV1
from reference_visual_vector_prediction_v1 import VisualVectorPredictionV1
class PredictionDrivenVisualReentryV1:
    def __init__(self,predictor=None,recurrent=None):
        self.predictor=predictor or VisualVectorPredictionV1();self.recurrent=recurrent or RecurrentVisualHeadingV1();self.last_residual=None;self.feedforward_count=0;self.reentry_count=0
    def observe_vector(self,o,vector_occurrence,learn=True):
        residual=self.predictor.residual(vector_occurrence);self.last_residual=residual
        rows,shape=vector_occurrence.representation
        direct=VisualHeadingCenterV1.estimate_rows(rows,*shape)
        mismatch=(residual is not None and bool(residual['missing'] or residual['unexpected']))
        if direct is not None and residual is not None and not mismatch:
            self.recurrent.sequence+=1;self.feedforward_count+=1
            result=RecurrentHeadingOccurrenceV1(self.recurrent.sequence,int(o.tick_count),direct,0,int(time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)))
            self.recurrent.current=result;self.recurrent._clear()
        else:
            self.reentry_count+=1;result=self.recurrent.observe_vector(o,vector_occurrence)
        if learn:self.predictor.observe(vector_occurrence)
        return result
    def checkpoint(self):return {'schema':1,'predictor':self.predictor.checkpoint(),'recurrent_sequence':self.recurrent.sequence,'feedforward_count':self.feedforward_count,'reentry_count':self.reentry_count}
    @classmethod
    def restore(cls,d):
        if int(d.get('schema',0))!=1:raise ValueError('prediction_reentry:checkpoint')
        r=RecurrentVisualHeadingV1();r.sequence=int(d.get('recurrent_sequence',0));out=cls(VisualVectorPredictionV1.restore(d['predictor']),r);out.feedforward_count=int(d.get('feedforward_count',0));out.reentry_count=int(d.get('reentry_count',0));return out
