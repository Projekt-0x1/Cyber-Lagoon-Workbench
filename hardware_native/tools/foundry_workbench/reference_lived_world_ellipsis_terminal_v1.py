#!/usr/bin/env python3
"""Immediate ellipsis-aware continuation over the current lived-world conversation."""
from __future__ import annotations

from reference_language_mastery_contact_adapter_v1 import CONTACT_UTTERANCE
from reference_lived_world_ellipsis_bridge_v1 import LivedWorldEllipsisBridgeV1
from reference_lived_world_followup_bridge_v1 import LivedWorldFollowupBridgeV1
from reference_lived_world_followup_terminal_v1 import externalize


def respond_ellipsis(adult,organism,contact,raw,last_spoken_world_occurrence,channel=0):
    prior_context=int(adult._current_selection_context)
    identity=contact.contact(CONTACT_UTTERANCE,tuple(raw),adult._advance(),channel)
    scene=contact.scenes.get(int(identity)) if int(identity)>0 else None
    if scene is None:return externalize(adult,adult.choose_public_plan())

    # Full learned query has first authority when its exact world×query context
    # already owns consequence-backed discourse matter.
    full_context=LivedWorldFollowupBridgeV1.context(organism,scene.context,scene.atoms)
    full_frontier=LivedWorldFollowupBridgeV1.frontier(
        adult,organism,scene.atoms,last_spoken_world_occurrence)
    if full_context>0 and full_frontier and adult.discourse_credit.candidates(full_context):
        adult._clear_current_occurrence();adult._current_selection_context=int(full_context)
        root=adult.organize_relevant_frontier(full_frontier)
        motor=externalize(adult,root)
        if motor:return motor

    # Otherwise a bare concept may reuse one immediately prior focused situation.
    _context,frontier=LivedWorldEllipsisBridgeV1.activate_frontier(
        adult,organism,prior_context,scene.atoms,last_spoken_world_occurrence)
    if frontier:
        root=adult.organize_relevant_frontier(frontier)
        motor=externalize(adult,root)
        if motor:return motor
    return externalize(adult,adult.choose_public_plan())
