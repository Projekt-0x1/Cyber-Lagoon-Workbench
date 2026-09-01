#!/usr/bin/env python3
"""Stateless mechanical bridge from organism-owned world occurrence to Adult situation."""
from __future__ import annotations
from reference_hierarchical_composition_v1 import _identity


class WorldDiscourseSituationBridgeV1:
    """No learned state: gate on live world provenance, derive one stable opaque context."""
    @staticmethod
    def context(organism):
        state=getattr(organism,'world_state',None)
        source=int(getattr(organism,'world_source',0))
        occurrence=int(getattr(organism,'world_state_occurrence',0))
        if state is None or source<=0 or occurrence<=0:return 0
        signature=tuple(map(int,organism.population.signature(tuple(state))))
        if not signature:return 0
        return _identity('world-discourse-situation-v1',signature)

    @classmethod
    def activate(cls,adult,organism):
        context=cls.context(organism)
        if not context:
            adult._clear_current_occurrence()
            return 0
        adult._clear_current_occurrence()
        adult._current_selection_context=int(context)
        return int(context)

    @classmethod
    def frontier(cls,adult,organism,grounding=None):
        """Recruit durable productive propositions entailed by current world constituents."""
        if not cls.context(organism):return ()
        present=set(map(int,(organism.world_state if grounding is None
                             else grounding.resolve_current_world(adult,organism))));rows=[]
        for template_identity,family in sorted(adult._surface_leaf_families.items()):
            binding=adult.language.historical_template_binding(int(template_identity))
            if binding is None:continue
            context,arity,_pieces=binding
            for identity,lexeme_ids in sorted(family.items()):
                if len(lexeme_ids)!=int(arity):continue
                concepts=[]
                for lexeme_identity in lexeme_ids:
                    learned=adult.language.historical_lexeme_binding(int(lexeme_identity))
                    if learned is None:concepts=[];break
                    concepts.append(int(learned[0]))
                if not concepts or not set(concepts).issubset(present):continue
                try:row=adult.leaf(int(context),tuple(concepts))
                except (KeyError,RuntimeError):continue
                if int(row.identity)==int(identity):rows.append(row)
        return tuple(rows)

    @classmethod
    def activate_frontier(cls,adult,organism,grounding=None):
        context=cls.activate(adult,organism)
        return context,(() if not context else cls.frontier(adult,organism,grounding))
