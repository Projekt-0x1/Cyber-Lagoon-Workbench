#!/usr/bin/env python3
"""Backend-neutral, content-free species-law IR for the AutoTrans experiment.

This is a compiler boundary, not a mature architecture.  It deliberately stores
only generic organism laws and resource bounds.  Curriculum episodes and trained
Adult state are separate inputs and are rejected if smuggled into law parameters.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from typing import Iterable

SCHEMA = "0x1.autotrans-species-program.v0"

# These names describe forbidden *classes of authored payload*, not runtime words.
_FORBIDDEN_PARAMETER_KEYS = {
    "surface", "text", "token", "word", "sentence", "answer", "expected_output",
    "fact", "entity_name", "transcript", "prompt", "trained_state", "checkpoint",
    "route_identity", "recipe_identity", "network_identity", "adult_state",
}


@dataclass(frozen=True, order=True)
class SpeciesLawV0:
    law: str
    version: int = 1
    parameters: tuple[tuple[str, int], ...] = ()

    def validate(self) -> None:
        if not self.law or self.version <= 0:
            raise ValueError("autotrans:invalid_law")
        seen = set()
        for key, value in self.parameters:
            if not key or key in seen or key in _FORBIDDEN_PARAMETER_KEYS:
                raise ValueError("autotrans:forbidden_or_duplicate_parameter")
            if not isinstance(value, int):
                raise ValueError("autotrans:nonnumeric_parameter")
            seen.add(key)


@dataclass(frozen=True)
class FoundrySpeciesProgramV0:
    laws: tuple[SpeciesLawV0, ...]
    resource_bounds: tuple[tuple[str, int], ...] = ()
    schema: str = SCHEMA

    @staticmethod
    def build(laws: Iterable[SpeciesLawV0], resource_bounds=()) -> "FoundrySpeciesProgramV0":
        normalized = tuple(sorted(laws))
        bounds = tuple(sorted((str(k), int(v)) for k, v in resource_bounds))
        program = FoundrySpeciesProgramV0(normalized, bounds)
        program.validate()
        return program

    def validate(self) -> None:
        if self.schema != SCHEMA or not self.laws:
            raise ValueError("autotrans:species_schema")
        names = set()
        for law in self.laws:
            law.validate()
            if law.law in names:
                raise ValueError("autotrans:duplicate_law")
            names.add(law.law)
        for key, value in self.resource_bounds:
            if not key or key in _FORBIDDEN_PARAMETER_KEYS or value < 0:
                raise ValueError("autotrans:resource_bound")

    def canonical_document(self) -> dict:
        return {
            "schema": self.schema,
            "laws": [
                {"law": row.law, "version": row.version,
                 "parameters": [[k, v] for k, v in row.parameters]}
                for row in self.laws
            ],
            "resource_bounds": [[k, v] for k, v in self.resource_bounds],
        }

    def canonical_bytes(self) -> bytes:
        return json.dumps(
            self.canonical_document(), sort_keys=True, separators=(",", ":")
        ).encode()

    def root(self) -> str:
        return hashlib.sha256(self.canonical_bytes()).hexdigest()


@dataclass(frozen=True)
class BackendLawMappingV0:
    law: str
    backend: str
    lowering: str
    evidence: tuple[str, ...] = field(default_factory=tuple)
    status: str = "UNLOWERED"

    def validate(self) -> None:
        if self.status not in {"MAPPED", "UNLOWERED"}:
            raise ValueError("autotrans:mapping_status")
        if not self.law or not self.backend:
            raise ValueError("autotrans:mapping_identity")
        if self.status == "MAPPED" and (not self.lowering or not self.evidence):
            raise ValueError("autotrans:mapped_without_evidence")


def direct_translation_readiness(
    program: FoundrySpeciesProgramV0, mappings: Iterable[BackendLawMappingV0]
) -> dict:
    rows = tuple(mappings)
    for row in rows:
        row.validate()
    by_law = {row.law: row for row in rows if row.backend == "direct"}
    missing = []
    mapped = []
    for law in program.laws:
        row = by_law.get(law.law)
        if row is None or row.status != "MAPPED":
            missing.append(law.law)
        else:
            mapped.append(law.law)
    return {
        "translation_ready": not missing,
        "mapped": tuple(mapped),
        "unlowered": tuple(missing),
        "species_root": program.root(),
    }
