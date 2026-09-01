#!/usr/bin/env python3
"""Continuing Workbench body that can originate learned discourse from lived world state."""
from __future__ import annotations

import argparse
import json
import os
import selectors
import sys
import tempfile
from pathlib import Path

from reference_language_mastery_adult_v1 import (
    CompositionWitnessV1,
    LanguageMasteryAdultV1,
)
from reference_language_mastery_contact_adapter_v1 import (
    CONTACT_UTTERANCE,
    LanguageMasteryContactAdapterV1,
)
from reference_language_mastery_terminal_v1 import emit_choice
from reference_organism_v2 import ReferenceOrganismV2
from reference_world_discourse_situation_bridge_v1 import WorldDiscourseSituationBridgeV1


def restore_session(path: Path):
    raw=json.loads(path.read_text())
    if not isinstance(raw,dict) or set(raw)!={'adult','organism','last_spoken_world_occurrence'}:
        raise ValueError('world-terminal:checkpoint_shape')
    adult=LanguageMasteryAdultV1.restore(raw['adult'])
    organism=ReferenceOrganismV2.restore(raw['organism'])
    last=max(0,int(raw['last_spoken_world_occurrence']))
    return adult,organism,last


def save_session(path: Path,adult,organism,last_spoken_world_occurrence: int):
    payload={
        'adult':adult.checkpoint(),
        'organism':organism.checkpoint(),
        'last_spoken_world_occurrence':max(0,int(last_spoken_world_occurrence)),
    }
    with tempfile.NamedTemporaryFile('w',dir=path.parent,delete=False) as out:
        json.dump(payload,out,separators=(',',':'),sort_keys=True)
        out.flush();os.fsync(out.fileno());temporary=Path(out.name)
    os.replace(temporary,path)


def respond(adult,contact,raw,channel=0):
    contact.contact(CONTACT_UTTERANCE,tuple(raw),adult._advance(),channel)
    return emit_choice(adult,adult.choose_public_plan())


def quiet(adult,organism,last_spoken_world_occurrence):
    """One resident no-contact opportunity; current world may originate discourse once."""
    occurrence=int(getattr(organism,'world_state_occurrence',0))
    if occurrence>0 and occurrence!=int(last_spoken_world_occurrence):
        _context,frontier=WorldDiscourseSituationBridgeV1.activate_frontier(adult,organism)
        root=adult.organize_relevant_frontier(frontier) if frontier else None
        if isinstance(root,CompositionWitnessV1) and int(root.context)<=0:
            context=int(adult._current_selection_context)
            if context>0:
                root=CompositionWitnessV1(
                    int(root.identity),context,int(root.template_identity),
                    tuple(root.child_identities),int(root.depth),
                    tuple(root.surface),tuple(root.pieces))
        motor=emit_choice(adult,root)
        if motor:return motor,occurrence
    return emit_choice(adult,adult.internal_tick()),int(last_spoken_world_occurrence)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--resume',required=True,type=Path)
    parser.add_argument('--idle-ms',type=int,default=25)
    args=parser.parse_args()
    if not 1<=args.idle_ms<=1000:raise SystemExit('body:idle_ms')
    adult,organism,last=restore_session(args.resume)
    contact=LanguageMasteryContactAdapterV1(adult)
    readiness=selectors.DefaultSelector();readiness.register(sys.stdin.buffer,selectors.EVENT_READ)
    try:
        while True:
            events=readiness.select(args.idle_ms/1000.0)
            if events:
                raw=sys.stdin.buffer.readline()
                if not raw:break
                motor=respond(adult,contact,raw.rstrip(b'\r\n'))
            else:
                motor,last=quiet(adult,organism,last)
            if motor:
                sys.stdout.buffer.write(motor+b'\n');sys.stdout.buffer.flush()
    finally:
        readiness.close();save_session(args.resume,adult,organism,last)


if __name__=='__main__':main()
