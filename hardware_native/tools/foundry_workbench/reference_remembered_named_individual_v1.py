#!/usr/bin/env python3
"""Resolve an authoritative learned proper-name surface against durable organism individuals."""
from __future__ import annotations
class RememberedNamedIndividualV1:
    @staticmethod
    def resolve(adult,organism,raw):
        units=tuple(int(x) for x in raw);matches=[]
        for entity in organism.entity_features:
            entity=int(entity)
            if not organism._active_entity_features(entity):continue
            if adult.language.lexeme(entity)==units:matches.append(entity)
        return matches[0] if len(matches)==1 else 0
