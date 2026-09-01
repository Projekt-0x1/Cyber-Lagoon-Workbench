#!/usr/bin/env python3
"""Continuing body: spontaneous lived-world turn, then learned entity-focused human follow-up."""
from __future__ import annotations

import argparse
import selectors
import sys
from pathlib import Path

from reference_language_mastery_adult_v1 import CompositionWitnessV1
from reference_language_mastery_contact_adapter_v1 import CONTACT_UTTERANCE,LanguageMasteryContactAdapterV1
from reference_language_mastery_terminal_v1 import emit_choice
from reference_lived_world_conversation_terminal_v1 import quiet,restore_session,save_session
from reference_lived_world_followup_bridge_v1 import LivedWorldFollowupBridgeV1


def externalize(adult,root):
    if isinstance(root,CompositionWitnessV1) and int(root.context)<=0:
        context=int(adult._current_selection_context)
        if context>0:
            root=CompositionWitnessV1(
                int(root.identity),context,int(root.template_identity),
                tuple(root.child_identities),int(root.depth),
                tuple(root.surface),tuple(root.pieces))
    return emit_choice(adult,root)


def respond_followup(adult,organism,contact,raw,last_spoken_world_occurrence,channel=0):
    identity=contact.contact(CONTACT_UTTERANCE,tuple(raw),adult._advance(),channel)
    scene=contact.scenes.get(int(identity)) if int(identity)>0 else None
    if scene is not None:
        _context,frontier=LivedWorldFollowupBridgeV1.activate_frontier(
            adult,organism,scene.context,scene.atoms,last_spoken_world_occurrence)
        if frontier:
            root=adult.organize_relevant_frontier(frontier)
            motor=externalize(adult,root)
            if motor:return motor
    return emit_choice(adult,adult.choose_public_plan())


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--resume',required=True,type=Path);parser.add_argument('--idle-ms',type=int,default=25);args=parser.parse_args()
    if not 1<=args.idle_ms<=1000:raise SystemExit('body:idle_ms')
    adult,organism,last=restore_session(args.resume);contact=LanguageMasteryContactAdapterV1(adult)
    readiness=selectors.DefaultSelector();readiness.register(sys.stdin.buffer,selectors.EVENT_READ)
    try:
        while True:
            events=readiness.select(args.idle_ms/1000.0)
            if events:
                raw=sys.stdin.buffer.readline()
                if not raw:break
                motor=respond_followup(adult,organism,contact,raw.rstrip(b'\r\n'),last)
            else:motor,last=quiet(adult,organism,last)
            if motor:sys.stdout.buffer.write(motor+b'\n');sys.stdout.buffer.flush()
    finally:
        readiness.close();save_session(args.resume,adult,organism,last)


if __name__=='__main__':main()
