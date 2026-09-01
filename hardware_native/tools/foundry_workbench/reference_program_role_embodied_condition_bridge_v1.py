#!/usr/bin/env python3
"""Stateless projection of live organism conditions onto Adult program-role leaves."""
from __future__ import annotations

from reference_organism_surface_state_v1 import surface_conditions
from reference_organism_v2 import BODY_STATE_TAG


class ProgramRoleEmbodiedConditionBridgeV1:
    """Mechanical current-state bridge; owns no learned or checkpoint state."""

    @staticmethod
    def conditions(adult, organism, program_identity, base_context=1):
        view = adult.program_role_view(program_identity, base_context)
        body_source = int(getattr(organism, "body_state_source", 0))
        projected = {}
        for leaf, atom in zip(view["leaves"], view["atoms"]):
            rows = tuple(surface_conditions(organism, int(atom), BODY_STATE_TAG, body_source))
            if rows:
                projected[int(leaf)] = rows
        return projected

    @classmethod
    def realize(cls, adult, organism, program_identity, base_context=1):
        return adult.realize_program_role_conditioned(
            program_identity,
            cls.conditions(adult, organism, program_identity, base_context),
            base_context,
        )
