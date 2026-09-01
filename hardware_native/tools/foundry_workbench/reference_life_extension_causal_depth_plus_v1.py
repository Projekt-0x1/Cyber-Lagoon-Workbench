#!/usr/bin/env python3
"""Append-only shared-life continuation through grounded sibling consequences."""
from __future__ import annotations
LIFE_AFTER=('reference_life_extension_causal_discourse_v1',)
A_RIVAL=0xA105
A_ROOTS=0xA108
A_STOMATA=0xA109
A_WILT=0xA10A
A_GROWTH=0xA10B
A_NEED=0xA10C
A_HUMIDITY=0xA10D
A_HARDEN=0xA10E
A_CONSERVE=0xA10F
PAIR_CONTEXT=0xA110
CONVERGENT_CAUSAL_SOURCES=(0xF7220,0xF7221,0xF7222)
PROPOSITIONS=((A_ROOTS,b'plant roots lose water'),(A_STOMATA,b'plant stomata close'),(A_WILT,b'plant leaves wilt'),(A_GROWTH,b'plant growth slows'),(A_NEED,b'plants need more water'),(A_HUMIDITY,b'the greenhouse needs more humidity'),(A_HARDEN,b'the soil surface hardens'),(A_CONSERVE,b'the leaves conserve moisture'))

def build(start):
    from reference_life_function_curriculum_v1 import LifeCurriculumEventV2
    rows=[]
    def add(lane,source=0,payload=()):rows.append(LifeCurriculumEventV2(int(start)+len(rows)+1,lane,int(source),tuple(payload)))
    scenes={}
    for atom,surface in PROPOSITIONS:
        for base in (0xF000,0xF800):
            scenes[atom]=int(start)+len(rows)+1;add('scene',base+atom,(100,atom));add('surface',base+atom,tuple(surface))
    for atom in (A_RIVAL,0xA104,0xA107):
        for base in (0xF100,0xF900):scenes[atom]=int(start)+len(rows)+1;add('scene',base+atom,(100,atom))
    def settle(cause,rival,effect,field_source,block_base):
        field_seq=int(start)+len(rows)+1;add('causal_field',field_source,(scenes[cause],scenes[rival],scenes[effect],256))
        for offset in range(3):
            source=block_base+offset;filler=block_base+0x80+offset
            for returned in (source,source,filler,source):add('resident_world_step',returned,(field_seq,scenes[cause],1))
        for source in range(block_base,block_base+3):add('resident_world_step',source,(field_seq,scenes[cause],1))
    settle(A_ROOTS,A_RIVAL,A_STOMATA,0xF401,0xF410);add('checkpoint_mark',0,('causal_depth_4',))
    settle(A_STOMATA,A_RIVAL,A_WILT,0xF501,0xF510);add('checkpoint_mark',0,('causal_depth_5',))
    settle(A_WILT,A_RIVAL,A_GROWTH,0xF601,0xF610);add('checkpoint_mark',0,('causal_depth_6',))
    for offset in range(6):add('body_load',0xF701+offset,(32+offset,1<<15))
    add('checkpoint_mark',0,('causal_depth_6_loaded',));add('quiet',0,(64,));add('checkpoint_mark',0,('causal_depth_6_recovered',))
    # Two structurally distinct common-cause pairs provide ordinary language
    # experience for one reusable binary surface factor.  The later target pair
    # is never heard in combined form.
    settle(0xA107,A_RIVAL,A_HARDEN,0xF7010,0xF7020)
    settle(A_STOMATA,A_RIVAL,A_CONSERVE,0xF7110,0xF7120)
    for source,left,right,surface in (
            (0xFA01,A_ROOTS,A_HARDEN,b'plant roots lose water and the soil surface hardens'),
            (0xFA02,A_WILT,A_CONSERVE,b'plant leaves wilt and the leaves conserve moisture')):
        add('relation',source,(PAIR_CONTEXT,scenes[left],scenes[right]))
        add('discourse_surface',source,tuple(surface))
    add('checkpoint_mark',0,('causal_coordination_examples',))
    # Ordinary consequence history closes a held-out convergent path.  No
    # diamond label or future discourse order is exposed to the organism.
    settle(A_HARDEN,A_RIVAL,A_STOMATA,0xF7210,CONVERGENT_CAUSAL_SOURCES[0])
    add('checkpoint_mark',0,('causal_convergent_path',))
    # The same learned cause has another independently settled consequence. This
    # is ordinary life, not a rendered paragraph or an evaluator-owned branch.
    settle(0xA104,A_RIVAL,A_NEED,0xF801,0xF810);add('checkpoint_mark',0,('causal_branch_7',))
    settle(0xA104,A_RIVAL,A_HUMIDITY,0xF901,0xF910);add('checkpoint_mark',0,('causal_siblings_8',))
    return tuple(rows)
