#!/usr/bin/env python3
"""Stateless opaque 2-D gravito-inertial direction/magnitude transduction."""
from __future__ import annotations
import math
from reference_population_v1 import mix64

OTOLITH_TAG=0x0710117


class OtolithGravitoInertialV1:
    @staticmethod
    def transduce(sample):
        lateral,forward=(int(x) for x in sample)
        if lateral==0 and forward==0:return None
        divisor=math.gcd(abs(lateral),abs(forward)) or 1
        dl=lateral//divisor;df=forward//divisor
        value=mix64(OTOLITH_TAG^mix64((dl+4097)&0xffff)^mix64(((df+4097)&0xffff)*17))
        direction=int(value&((1<<63)-1) or 1)
        magnitude=int(round(math.sqrt(lateral*lateral+forward*forward)))
        return direction,magnitude
