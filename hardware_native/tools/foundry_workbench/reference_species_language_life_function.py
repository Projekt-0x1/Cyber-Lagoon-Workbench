#!/usr/bin/env python3
"""Reference lowering from content-free Species laws into language mechanisms.

This is a Workbench backend for the same Genome/Life Function distinction used by
Direct.  It lowers only generic mechanism parameters; curriculum and learned Adult
state remain separate inputs.
"""
from __future__ import annotations

from autotrans_species_ir_v0 import FoundrySpeciesProgramV0
from reference_language_learning_v1 import LearnedSurfaceEcologyV1
from reference_language_mastery_adult_v1 import LanguageMasteryAdultV1

SOURCE_EVIDENCE_LAW = 'source_conditioned_access_evidence'
DEFAULT_MINIMUM_DISTINCT_SOURCES = 2


def source_support_from_species(program: FoundrySpeciesProgramV0) -> int:
    program.validate()
    rows=[law for law in program.laws if law.law==SOURCE_EVIDENCE_LAW]
    if len(rows)!=1:raise ValueError('language_life_function:source_evidence_law')
    params=dict(rows[0].parameters)
    unknown=set(params)-{'minimum_distinct_sources'}
    if unknown:raise ValueError('language_life_function:unknown_parameter')
    value=int(params.get('minimum_distinct_sources',DEFAULT_MINIMUM_DISTINCT_SOURCES))
    if not 1<=value<=16:raise ValueError('language_life_function:minimum_distinct_sources')
    return value


def birth_language_mastery_adult(program: FoundrySpeciesProgramV0) -> LanguageMasteryAdultV1:
    """Birth one blank fast Adult from the content-free Species/Life Function law."""
    adult=LanguageMasteryAdultV1()
    # Birth happens before curriculum/history. Species may provide generic learning
    # machinery and resource law, never learned language/world content.
    adult.language=LearnedSurfaceEcologyV1(source_support_from_species(program))
    return adult
