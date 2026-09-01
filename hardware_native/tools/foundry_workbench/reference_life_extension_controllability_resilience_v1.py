#!/usr/bin/env python3
"""Append-only learned-control tail: public causal action history modulates later load response."""
from __future__ import annotations
LIFE_AFTER=('reference_life_extension_causal_discourse_forms_v2',)
Q=1<<16
QUERY=b'warm air dries the soil'

def build(start):
    from reference_life_function_curriculum_v1 import LifeCurriculumEventV2
    rows=[]
    def add(lane,source=0,payload=()):rows.append(LifeCurriculumEventV2(int(start)+len(rows)+1,lane,int(source),tuple(payload)))
    # Two public actions with independent returns, then a matched no-action opportunity.
    actions=[]
    for index in range(2):
        contact=int(start)+len(rows)+1;add('authenticated_utterance',0xFA10+index,tuple(QUERY))
        action=int(start)+len(rows)+1;actions.append(action);add('causal_dialogue_opportunity',0xFA10+index,(contact,))
        add('causal_dialogue_return',0xFB10+index,(action,Q,0,1))
    add('causal_dialogue_background',0xFC10,(actions[-1],0))
    add('checkpoint_mark',0,('causal_control_history_learned',))
    # A later expressed action receives no independent confirmation. This lowers
    # current contingency while preserving most of the previously earned slow history.
    contact=int(start)+len(rows)+1;add('authenticated_utterance',0xFA20,tuple(QUERY))
    challenged=int(start)+len(rows)+1;add('causal_dialogue_opportunity',0xFA20,(contact,))
    add('causal_dialogue_return',0xFB20,(challenged,Q,0,0))
    add('checkpoint_mark',0,('causal_control_history_challenged',))
    # Identical content/causal knowledge, now under sustained organism-wide load.
    for offset in range(6):add('body_load',0xFD10+offset,(48+offset,1<<15))
    add('checkpoint_mark',0,('causal_control_resilience_loaded',))
    add('quiet',0,(64,));add('checkpoint_mark',0,('causal_control_resilience_recovered',))
    return tuple(rows)
