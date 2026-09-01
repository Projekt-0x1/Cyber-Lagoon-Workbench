#!/usr/bin/env python3
"""Strict append: broaden learned causal discourse realizations without revising truth."""
from __future__ import annotations
LIFE_AFTER=('reference_life_extension_causal_depth_plus_v1',)

def build(start):
    from reference_life_function_curriculum_v1 import LifeCurriculumEventV2
    rows=[]
    def add(lane,source=0,payload=()):rows.append(LifeCurriculumEventV2(int(start)+len(rows)+1,lane,int(source),tuple(payload)))
    forms=(
        b'dry soil cracks the surface. In turn, plant roots lose water.',
        b'dry soil cracks the surface. Thus, plant roots lose water.',
    )
    for index,surface in enumerate(forms):
        for witness in range(2):add('authenticated_utterance',0xDA01+index*4+witness,tuple(surface))
    # A second self-contained causal construction is acquired on two other
    # world-certified relations.  Its later use for heater -> warm air is held
    # out: neither that proposition pair nor an alias between the constructions
    # occurs in this developmental contact.
    reformulations=(
        b'plant stomata close because plant roots lose water.',
        b'plant leaves wilt because plant stomata close.',
    )
    for index,surface in enumerate(reformulations):
        for witness in range(2):add('authenticated_utterance',0xDA11+index*4+witness,tuple(surface))
    add('checkpoint_mark',0,('causal_discourse_form_diversity',))
    return tuple(rows)
