#!/usr/bin/env python3
"""Append-only one-Adult history matrix: partner-local contingency gradients, no state labels."""
from __future__ import annotations
LIFE_AFTER=('reference_life_extension_somatic_appraisal_language_v1',)
Q=1<<16
QUERY=b'warm air dries the soil'
A=0xDA01;B=0xDA02;C=0xDA03

def build(start):
    from reference_life_function_curriculum_v1 import LifeCurriculumEventV2
    rows=[]
    def add(lane,source=0,payload=()):rows.append(LifeCurriculumEventV2(int(start)+len(rows)+1,lane,int(source),tuple(payload)))
    def action(partner,confirmed,feedback):
        contact=int(start)+len(rows)+1;add('authenticated_utterance',partner,tuple(QUERY))
        seq=int(start)+len(rows)+1;add('partner_causal_dialogue_opportunity',partner,(contact,))
        add('causal_dialogue_return',partner+0x100,(seq,Q,0,1 if confirmed else 0))
        # Ordinary linguistic feedback is lived contact, never causal-credit authority.
        add('authenticated_utterance',partner,tuple(feedback))
        return seq
    # A: repeated contingent action success plus matched no-action failure => strong learned control.
    a1=action(A,True,b'that changed what happened');a2=action(A,True,b'that worked again')
    add('causal_dialogue_background',A+0x200,(a2,0))
    # B: first earn control, then three nonconfirming actions degrade current contingency/history.
    b1=action(B,True,b'that changed what happened');b2=action(B,True,b'that worked again')
    add('causal_dialogue_background',B+0x200,(b2,0))
    action(B,False,b'that did not change anything');action(B,False,b'that did not help this time');action(B,False,b'again nothing changed')
    # C: repeated actions without independent confirmation; same causal knowledge, no earned control.
    c1=action(C,False,b'that did not change anything');c2=action(C,False,b'again nothing changed')
    add('causal_dialogue_background',C+0x200,(c2,0))
    add('checkpoint_mark',0,('partner_history_matrix',))
    # Same organism-wide load applies to all partner contexts.
    for offset in range(6):add('body_load',0xDB10+offset,(64+offset,1<<15))
    add('checkpoint_mark',0,('partner_history_matrix_loaded',))
    add('quiet',0,(64,));add('checkpoint_mark',0,('partner_history_matrix_recovered',))
    return tuple(rows)
