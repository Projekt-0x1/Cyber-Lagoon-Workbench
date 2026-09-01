#!/usr/bin/env python3
"""Stateless reduced elaboration -> current lived-world discourse-program bridge."""
from __future__ import annotations

from reference_hierarchical_composition_v1 import _identity
from reference_language_mastery_adult_v1 import AdultStateV1
from reference_lived_world_relation_question_bridge_v1 import LivedWorldRelationQuestionBridgeV1
from reference_world_discourse_situation_bridge_v1 import WorldDiscourseSituationBridgeV1


class LivedWorldExplanationElaborationBridgeV1:
    @staticmethod
    def context(organism, prior_context, query_context, query_atoms):
        world_context=WorldDiscourseSituationBridgeV1.context(organism)
        atoms=tuple(int(x) for x in query_atoms if int(x)!=0)
        if world_context<=0 or int(prior_context)<=0 or not atoms:return 0
        return _identity('lived-world-explanation-elaboration-context-v1',
                         (world_context,int(prior_context),atoms))

    @staticmethod
    def activate_program(adult,organism,query_context,query_atoms,spoken_world_occurrence,
                         state=AdultStateV1()):
        occurrence=int(getattr(organism,'world_state_occurrence',0))
        if occurrence<=0 or occurrence!=int(spoken_world_occurrence):return 0,0
        world_ids={int(x.identity) for x in WorldDiscourseSituationBridgeV1.frontier(adult,organism)}
        prior_context=int(adult.last_completed_public_context())
        if not world_ids or prior_context<=0:return 0,0
        prior_program=int(adult._select(prior_context,state))
        prior_leaves=set(LivedWorldRelationQuestionBridgeV1._program_leaves(adult,prior_program)) if prior_program else set()
        if len(prior_leaves)<2 or not prior_leaves.issubset(world_ids):return 0,0
        context=LivedWorldExplanationElaborationBridgeV1.context(
            organism,prior_context,query_context,query_atoms)
        if context<=0:return 0,0
        winner=int(adult._select(context,state))
        if not winner or winner==prior_program:return 0,0
        leaves=set(LivedWorldRelationQuestionBridgeV1._program_leaves(adult,winner))
        if len(leaves)<2 or not leaves.issubset(world_ids):return 0,0
        adult._clear_current_occurrence();adult._current_selection_context=context
        return context,winner
