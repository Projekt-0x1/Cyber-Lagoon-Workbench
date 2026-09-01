#!/usr/bin/env python3
"""Bounded central gravity estimate that disambiguates otolith tilt vs translation."""
from __future__ import annotations
import math

from reference_otolith_gravito_inertial_v1 import OtolithGravitoInertialV1

GRAVITY_MAGNITUDE=1000
RESIDUAL_TOLERANCE=4


class TiltTranslationDisambiguatorV1:
    def __init__(self):
        self.orientation_deg=0
        self.canal_source=0
        self.canal_sequence=0
        self.orientation_valid=True

    def observe_canal(self,canal_sensor):
        sample=getattr(canal_sensor,'current_sample',None)
        source=int(getattr(canal_sensor,'active_source',0))
        sequence=int(getattr(canal_sensor,'active_sequence',0))
        if sample is None or source<=0 or sequence<=0:return False
        if self.canal_sequence:
            if source!=self.canal_source or sequence!=self.canal_sequence+1:
                self.orientation_valid=False
                self.canal_source=source;self.canal_sequence=sequence
                return False
        self.orientation_deg=max(-60,min(60,self.orientation_deg+int(sample)))
        self.canal_source=source;self.canal_sequence=sequence
        return bool(self.orientation_valid)

    def predicted_gravity(self):
        if not self.orientation_valid:return None
        angle=math.radians(self.orientation_deg)
        return (int(round(GRAVITY_MAGNITUDE*math.sin(angle))),
                int(round(GRAVITY_MAGNITUDE*math.cos(angle))))

    def resolve(self,otolith_sensor):
        if not self.orientation_valid:return None
        sample=getattr(otolith_sensor,'current_sample',None)
        if sample is None:return None
        gravity=self.predicted_gravity()
        residual=(int(sample[0])-gravity[0],int(sample[1])-gravity[1])
        magnitude=int(round(math.sqrt(residual[0]*residual[0]+residual[1]*residual[1])))
        if magnitude<=RESIDUAL_TOLERANCE:
            return {'translation_feature':0,'translation_magnitude':0,'residual':residual,
                    'gravity':gravity,'orientation_deg':self.orientation_deg}
        feature,_=OtolithGravitoInertialV1.transduce(residual)
        return {'translation_feature':int(feature),'translation_magnitude':magnitude,
                'residual':residual,'gravity':gravity,'orientation_deg':self.orientation_deg}

    def lesion_canal_history(self):
        self.orientation_valid=False

    def checkpoint(self):
        return {'schema':1,'orientation_deg':self.orientation_deg,'canal_source':self.canal_source,
                'canal_sequence':self.canal_sequence,'orientation_valid':self.orientation_valid}

    @classmethod
    def restore(cls,data):
        if int(data.get('schema',0))!=1:raise ValueError('tilt_translation:checkpoint')
        out=cls();out.orientation_deg=int(data.get('orientation_deg',0));out.canal_source=int(data.get('canal_source',0));out.canal_sequence=int(data.get('canal_sequence',0));out.orientation_valid=bool(data.get('orientation_valid',True))
        if not -60<=out.orientation_deg<=60 or out.canal_source<0 or out.canal_sequence<0:raise ValueError('tilt_translation:checkpoint')
        return out
