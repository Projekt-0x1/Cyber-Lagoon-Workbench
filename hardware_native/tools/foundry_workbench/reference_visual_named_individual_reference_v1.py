#!/usr/bin/env python3
"""Resolve an exact learned name against currently visible organism object files."""
from __future__ import annotations
from reference_visual_joint_attention_naming_v1 import VisualJointAttentionNamingV1

class VisualNamedIndividualReferenceV1:
    @staticmethod
    def resolve(adult,tracker,raw):
        units=tuple(int(x) for x in raw)
        visible=tuple(VisualJointAttentionNamingV1.visible_entities(tracker))
        matched=tuple(sorted(
            entity for entity in visible
            if adult.language.lexeme(int(entity))==units))
        return matched[0] if len(matched)==1 else 0
