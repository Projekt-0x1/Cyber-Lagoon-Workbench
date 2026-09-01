#!/usr/bin/env python3
"""Stateless learned-query -> current lived-world proposition intersection."""
from __future__ import annotations

from reference_hierarchical_composition_v1 import _identity
from reference_world_discourse_situation_bridge_v1 import WorldDiscourseSituationBridgeV1


class LivedWorldFollowupBridgeV1:
    @staticmethod
    def leaf_features(adult,leaf):
        identity=int(leaf.identity);template=adult._surface_leaf_family_index.get(identity)
        family=adult._surface_leaf_families.get(template,{}) if template is not None else {}
        features=[]
        for lexeme in family.get(identity,()):
            row=adult.language.historical_lexeme_binding(int(lexeme))
            if row is None:return ()
            feature,_units=row;features.append(int(feature))
        return tuple(features)

    @staticmethod
    def context(organism,query_context,query_atoms):
        world_context=WorldDiscourseSituationBridgeV1.context(organism)
        atoms=tuple(int(x) for x in query_atoms if int(x)!=0)
        if world_context<=0 or int(query_context)<=0 or not atoms:return 0
        return _identity('lived-world-followup-context-v1',(world_context,int(query_context),atoms))

    @staticmethod
    def frontier(adult,organism,query_atoms,spoken_world_occurrence):
        occurrence=int(getattr(organism,'world_state_occurrence',0))
        if occurrence<=0 or occurrence!=int(spoken_world_occurrence):return ()
        query={int(x) for x in query_atoms if int(x)!=0}
        if not query:return ()
        frontier=WorldDiscourseSituationBridgeV1.frontier(adult,organism)
        return tuple(
            leaf for leaf in frontier
            if query.intersection(LivedWorldFollowupBridgeV1.leaf_features(adult,leaf)))

    @staticmethod
    def activate_frontier(adult,organism,query_context,query_atoms,spoken_world_occurrence):
        context=LivedWorldFollowupBridgeV1.context(organism,query_context,query_atoms)
        frontier=LivedWorldFollowupBridgeV1.frontier(
            adult,organism,query_atoms,spoken_world_occurrence)
        if context<=0 or not frontier:return 0,()
        adult._clear_current_occurrence()
        adult._current_selection_context=int(context)
        return int(context),frontier
