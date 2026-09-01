#!/usr/bin/env python3
"""Append-only ordinary dialogue: an open prompt learns a current-construction affordance."""
from __future__ import annotations
LIFE_AFTER=('reference_life_extension_history_matrix_v1',)
QFEATURE=0xB201
PROMPT=b'How are you doing?'

def build(start):
    from reference_life_function_curriculum_v1 import LifeCurriculumEventV2
    from reference_language_mastery_adult_v1 import LanguageMasteryAdultV1
    rows=[]
    def add(lane,source=0,payload=()):rows.append(LifeCurriculumEventV2(int(start)+len(rows)+1,lane,int(source),tuple(payload)))
    prompt_scenes=[]
    for source in (0xDC01,0xDC02):
        seq=int(start)+len(rows)+1;add('scene',source,(100,QFEATURE));add('surface',source,tuple(PROMPT));prompt_scenes.append(seq)
    context=LanguageMasteryAdultV1._somatic_appraisal_language_context()
    f=LanguageMasteryAdultV1._somatic_appraisal_feature
    # These are already-lived state-language combinations. The relation examples differ,
    # so the only stable prediction is prompt -> current construction context.
    states=((f(1,1),f(2,1),f(3,1)),(f(1,0),f(2,0),f(3,0)))
    for index,(prompt,atoms) in enumerate(zip(prompt_scenes,states)):
        state_seq=int(start)+len(rows)+1;add('scene',0xDD10+index,(context,*atoms))
        add('relation',0xDD20+index,(0xB211,prompt,state_seq))
    add('checkpoint_mark',0,('open_state_prompt_grounded',))
    # Re-enter common organism-wide load only after the prompt relation is learned.
    for offset in range(6):add('body_load',0xDE10+offset,(80+offset,1<<15))
    add('checkpoint_mark',0,('open_state_prompt_loaded',))
    add('quiet',0,(64,));add('checkpoint_mark',0,('open_state_prompt_recovered',))
    return tuple(rows)
