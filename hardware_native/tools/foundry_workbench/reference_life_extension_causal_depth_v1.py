#!/usr/bin/env python3
"""Shared-life tail: consequence-owned multi-hop causal depth plus somatic load/recovery."""
from __future__ import annotations
LIFE_AFTER=()

A_EFFECT=0xA104          # warm air dries the soil
A_RIVAL=0xA105           # steady wind closes the vent
A_CRACK=0xA107           # dry soil cracks the surface
A_ROOTS=0xA108           # plant roots lose water

PROPOSITIONS=(
    (A_CRACK,b'dry soil cracks the surface'),
    (A_ROOTS,b'plant roots lose water'),
)

def build(start):
    from reference_life_function_curriculum_v1 import LifeCurriculumEventV2
    rows=[]
    def add(lane,source=0,payload=()):rows.append(LifeCurriculumEventV2(int(start)+len(rows)+1,lane,int(source),tuple(payload)))
    scenes={}
    for atom,surface in PROPOSITIONS:
        for base in (0xE000,0xE800):
            scenes[atom]=int(start)+len(rows)+1
            add('scene',base+atom,(100,atom));add('surface',base+atom,tuple(surface))
    # Existing propositions are referenced by fresh ordinary scene occurrences so this
    # extension depends only on the shared Adult's learned concept/language state.
    for atom in (A_EFFECT,A_RIVAL):
        for base in (0xE100,0xE900):
            scenes[atom]=int(start)+len(rows)+1
            add('scene',base+atom,(100,atom))
            # Current surface is already learned; no new surface evidence is supplied.
    def field(cause,rival,effect,field_source,block_base):
        field_seq=int(start)+len(rows)+1
        add('causal_field',field_source,(scenes[cause],scenes[rival],scenes[effect],256))
        for offset in range(3):
            source=block_base+offset;filler=block_base+0x80+offset
            for returned in (source,source,filler,source):
                add('resident_world_step',returned,(field_seq,scenes[cause],1))
        return field_seq
    depth2=field(A_EFFECT,A_RIVAL,A_CRACK,0xE401,0xE410)
    for source in range(0xE410,0xE413):add('resident_world_step',source,(depth2,scenes[A_EFFECT],1))
    add('checkpoint_mark',0,('causal_depth_2',))
    depth3=field(A_CRACK,A_RIVAL,A_ROOTS,0xE501,0xE510)
    for source in range(0xE510,0xE513):add('resident_world_step',source,(depth3,scenes[A_CRACK],1))
    add('checkpoint_mark',0,('causal_depth_3',))
    # Interoceptive load is content-free organism state. It can veto expensive public
    # elaboration, but it cannot modify causal truth or install language content.
    for offset in range(6):add('body_load',0xE601+offset,(16+offset,1<<15))
    add('checkpoint_mark',0,('causal_depth_loaded',))
    add('quiet',0,(64,));add('checkpoint_mark',0,('causal_depth_recovered',))
    return tuple(rows)
