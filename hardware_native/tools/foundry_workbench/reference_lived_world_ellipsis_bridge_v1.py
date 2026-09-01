#!/usr/bin/env python3
"""Stateless immediate-discourse ellipsis over current lived-world proposition matter."""
from __future__ import annotations

from reference_lived_world_followup_bridge_v1 import LivedWorldFollowupBridgeV1
from reference_world_discourse_situation_bridge_v1 import WorldDiscourseSituationBridgeV1


class LivedWorldEllipsisBridgeV1:
    @staticmethod
    def activate_frontier(adult,organism,prior_context,query_atoms,spoken_world_occurrence):
        occurrence=int(getattr(organism,'world_state_occurrence',0))
        if occurrence<=0 or occurrence!=int(spoken_world_occurrence):return 0,()
        world_frontier=WorldDiscourseSituationBridgeV1.frontier(adult,organism)
        world_ids={int(x.identity) for x in world_frontier}
        prior_ids={int(x) for x in adult.discourse_credit.candidates(int(prior_context))}
        if not prior_ids or not prior_ids<world_ids:return 0,()
        query={int(x) for x in query_atoms if int(x)!=0}
        if not query:return 0,()
        target=tuple(
            leaf for leaf in world_frontier
            if query.intersection(LivedWorldFollowupBridgeV1.leaf_features(adult,leaf)))
        target_ids={int(x.identity) for x in target}
        if not target_ids or not target_ids<world_ids:return 0,()
        matches=[]
        for context,members in adult.discourse_credit.context_members.items():
            ids={int(x) for x in members}
            if ids==target_ids:matches.append(int(context))
        if len(matches)!=1:return 0,()
        adult._clear_current_occurrence();adult._current_selection_context=matches[0]
        return matches[0],target
