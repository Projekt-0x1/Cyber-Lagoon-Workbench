#!/usr/bin/env python3
"""Stateless reduced contact -> current lived-world response-program competition."""
from __future__ import annotations
from reference_hierarchical_composition_v1 import _identity
from reference_language_mastery_adult_v1 import AdultStateV1
from reference_world_discourse_situation_bridge_v1 import WorldDiscourseSituationBridgeV1

class LivedWorldPragmaticResponseBridgeV1:
    @staticmethod
    def context(organism,query_context,query_atoms):
        world_context=WorldDiscourseSituationBridgeV1.context(organism)
        atoms=tuple(int(x) for x in query_atoms if int(x)!=0)
        if world_context<=0 or int(query_context)<=0 or not atoms:return 0
        return _identity('lived-world-pragmatic-response-context-v1',
                         (world_context,int(query_context),atoms))

    @staticmethod
    def activate_program(adult,organism,query_context,query_atoms,spoken_world_occurrence,
                         state=AdultStateV1()):
        occurrence=int(getattr(organism,'world_state_occurrence',0))
        world_context=WorldDiscourseSituationBridgeV1.context(organism)
        if (occurrence<=0 or occurrence!=int(spoken_world_occurrence)
                or world_context<=0 or adult.last_completed_public_context()!=world_context):
            return 0,0
        context=LivedWorldPragmaticResponseBridgeV1.context(
            organism,query_context,query_atoms)
        if context<=0:return 0,0
        winner=int(adult._select(context,state))
        if not winner:return 0,0
        try:surface=adult.public_surface(winner)
        except (KeyError,RuntimeError):return 0,0
        if not surface:return 0,0
        adult._clear_current_occurrence();adult._current_selection_context=context
        return context,winner
