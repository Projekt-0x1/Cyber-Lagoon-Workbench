#!/usr/bin/env python3
from __future__ import annotations
import time
from reference_contextual_visual_prediction_v1 import ContextConditionalVisualPredictionV1
from reference_recurrent_visual_heading_v1 import RecurrentVisualHeadingV1,RecurrentHeadingOccurrenceV1
from reference_visual_heading_center_v1 import VisualHeadingCenterV1
class ContextPredictionDrivenVisualReentryV1:
    def __init__(self,predictor=None,recurrent=None):self.predictor=predictor or ContextConditionalVisualPredictionV1();self.recurrent=recurrent or RecurrentVisualHeadingV1();self.last_residual=None;self.feedforward_count=0;self.reentry_count=0
    def observe_vector(self,o,v,learn=True):
        residual=self.predictor.residual(o,v);self.last_residual=residual;rows,shape=v.representation;direct=VisualHeadingCenterV1.estimate_rows(rows,*shape);mismatch=residual is not None and bool(residual['missing'] or residual['unexpected'])
        if direct is not None and residual is not None and not mismatch:
            self.recurrent.sequence+=1;self.feedforward_count+=1;out=RecurrentHeadingOccurrenceV1(self.recurrent.sequence,int(o.tick_count),direct,0,int(time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)));self.recurrent.current=out;self.recurrent._clear()
        else:self.reentry_count+=1;out=self.recurrent.observe_vector(o,v)
        if learn:self.predictor.observe(o,v)
        return out
    def checkpoint(self):return {'schema':1,'predictor':self.predictor.checkpoint(),'recurrent_sequence':self.recurrent.sequence,'feedforward_count':self.feedforward_count,'reentry_count':self.reentry_count}
    @classmethod
    def restore(cls,d):
        if int(d.get('schema',0))!=1:raise ValueError('context_prediction_reentry:checkpoint')
        r=RecurrentVisualHeadingV1();r.sequence=int(d.get('recurrent_sequence',0));out=cls(ContextConditionalVisualPredictionV1.restore(d['predictor']),r);out.feedforward_count=int(d.get('feedforward_count',0));out.reentry_count=int(d.get('reentry_count',0));return out
