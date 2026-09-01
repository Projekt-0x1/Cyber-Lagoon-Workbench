#!/usr/bin/env python3
"""Canonical-life tail: ordinary language names the Adult's own appraisal geometry."""
from __future__ import annotations

LIFE_AFTER=('reference_life_extension_controllability_resilience_v1',)


def build(start):
    from reference_life_function_curriculum_v1 import LifeCurriculumEventV2
    from reference_language_mastery_adult_v1 import LanguageMasteryAdultV1
    rows=[]
    def add(lane,source=0,payload=()):
        rows.append(LifeCurriculumEventV2(int(start)+len(rows)+1,lane,int(source),tuple(payload)))

    context=LanguageMasteryAdultV1._somatic_appraisal_language_context()
    features={
        'positive':LanguageMasteryAdultV1._somatic_appraisal_feature(1,1),
        'negative':LanguageMasteryAdultV1._somatic_appraisal_feature(1,0),
        'control':LanguageMasteryAdultV1._somatic_appraisal_feature(2,1),
        'no_control':LanguageMasteryAdultV1._somatic_appraisal_feature(2,0),
        'loaded':LanguageMasteryAdultV1._somatic_appraisal_feature(3,1),
        'settled':LanguageMasteryAdultV1._somatic_appraisal_feature(3,0),
    }
    surfaces={
        'positive':b'manageable',
        'negative':b'aversive',
        'control':b'I can influence what happens',
        'no_control':b'I have little control over what happens',
        'loaded':b'my body is under strain',
        'settled':b'my body is settled',
    }
    # Ordinary naming contact. Numeric feature identity is the Adult's body/appraisal
    # coordinate; no emotion label, reward head or output phrase is Species law.
    for ordinal,(name,feature) in enumerate(features.items()):
        for witness in range(2):
            source=0xFD80+ordinal*8+witness
            add('scene',source,(100,feature));add('surface',source,tuple(surfaces[name]))

    # Two structurally matched but coordinate-diverse examples induce one reusable
    # three-slot self-state construction. Later mixed combinations are held out.
    examples=(
        ((features['positive'],features['control'],features['loaded']),
         b'My current state feels manageable: I can influence what happens, and my body is under strain.'),
        ((features['negative'],features['no_control'],features['settled']),
         b'My current state feels aversive: I have little control over what happens, and my body is settled.'),
    )
    for witness,(atoms,surface) in enumerate(examples):
        source=0xFE20+witness;add('scene',source,(context,*atoms));add('surface',source,tuple(surface))
    add('checkpoint_mark',0,('somatic_appraisal_language',))
    return tuple(rows)
