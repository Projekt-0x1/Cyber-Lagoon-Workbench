#!/usr/bin/env python3
"""Stateless reduced-question -> current lived-world discourse-program bridge."""
from __future__ import annotations

from reference_hierarchical_composition_v1 import _identity
from reference_language_mastery_adult_v1 import AdultStateV1
from reference_world_discourse_situation_bridge_v1 import WorldDiscourseSituationBridgeV1


class LivedWorldRelationQuestionBridgeV1:
    @staticmethod
    def context(organism, query_context, query_atoms):
        world_context=WorldDiscourseSituationBridgeV1.context(organism)
        atoms=tuple(int(x) for x in query_atoms if int(x)!=0)
        if world_context<=0 or int(query_context)<=0 or not atoms:return 0
        return _identity('lived-world-relation-question-context-v1',
                         (world_context,int(query_context),atoms))

    @staticmethod
    def _program_leaves(adult, root):
        stack=[int(root)];seen=set();leaves=set()
        while stack:
            identity=stack.pop()
            if identity in seen:return ()
            seen.add(identity)
            chunk=adult.programs.chunks.get(identity)
            if chunk is not None:
                stack.extend(map(int,chunk.members));continue
            factor=adult.programs.factor(identity)
            if factor is not None and int(factor)<0:leaf=-int(factor)
            elif adult._has_leaf(identity):leaf=identity
            else:return ()
            if not adult._has_leaf(leaf):return ()
            leaves.add(leaf)
        return tuple(sorted(leaves))

    @staticmethod
    def activate_program(adult, organism, query_context, query_atoms, spoken_world_occurrence,
                         state=AdultStateV1()):
        occurrence=int(getattr(organism,'world_state_occurrence',0))
        world_context=WorldDiscourseSituationBridgeV1.context(organism)
        prior_context=int(adult.last_completed_public_context())
        if (occurrence<=0 or occurrence!=int(spoken_world_occurrence)
                or prior_context!=world_context):return 0,0
        frontier=WorldDiscourseSituationBridgeV1.frontier(adult,organism)
        world_ids={int(x.identity) for x in frontier}
        prior_ids={int(x) for x in adult.discourse_credit.candidates(world_context)}
        if not world_ids or prior_ids!=world_ids:return 0,0
        context=LivedWorldRelationQuestionBridgeV1.context(
            organism,query_context,query_atoms)
        if context<=0:return 0,0
        winner=int(adult._select(context,state))
        if not winner:return 0,0
        leaves=set(LivedWorldRelationQuestionBridgeV1._program_leaves(adult,winner))
        if len(leaves)<2 or not leaves.issubset(world_ids):return 0,0
        adult._clear_current_occurrence();adult._current_selection_context=context
        return context,winner
