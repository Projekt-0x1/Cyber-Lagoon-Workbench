#!/usr/bin/env python3
"""Semantics-free terminal body for one checkpointed Workbench Life."""
from __future__ import annotations

import argparse
import json
import os
import selectors
import sys
import tempfile
from pathlib import Path

from reference_language_mastery_adult_v1 import LanguageMasteryAdultV1
from reference_language_mastery_contact_adapter_v1 import (
    CONTACT_UTTERANCE,
    LanguageMasteryContactAdapterV1,
)


def emit_choice(adult, chosen):
    if not chosen:return b''
    expression=adult.expression(chosen);out=bytearray()
    while (step:=expression.emit()) is not None:
        out.append(step.value)
        if not expression.reafference(step,step.value):
            raise RuntimeError('body:motor_reafference')
    return bytes(out)


def respond(adult, contact, raw, channel=0):
    contact.contact(CONTACT_UTTERANCE,tuple(raw),adult._advance(),channel)
    return emit_choice(adult,adult.choose_public_plan())


def quiet(adult):
    """Advance one resident opportunity without supplying contact or a candidate."""
    return emit_choice(adult,adult.internal_tick())


def save(path, adult):
    with tempfile.NamedTemporaryFile('w',dir=path.parent,delete=False) as out:
        json.dump(adult.checkpoint(),out,separators=(',',':'),sort_keys=True)
        out.flush();os.fsync(out.fileno());temporary=Path(out.name)
    os.replace(temporary,path)


def restore_life(data):
    """Restore the sole Workbench subject; kept lazy for helper-only imports."""
    from reference_life_function_curriculum_v1 import (
        ReferenceLifeFunctionRuntimeV2,canonical_species_program_v2,
    )
    return ReferenceLifeFunctionRuntimeV2.restore(
        canonical_species_program_v2(),data)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--resume',required=True,type=Path)
    parser.add_argument('--idle-ms',type=int,default=25)
    args=parser.parse_args()
    if not 1<=args.idle_ms<=1000:raise SystemExit('body:idle_ms')
    runtime=restore_life(json.loads(args.resume.read_text()))
    readiness=selectors.DefaultSelector();readiness.register(sys.stdin.buffer,selectors.EVENT_READ)
    try:
        while True:
            events=readiness.select(args.idle_ms/1000.0)
            if events:
                raw=sys.stdin.buffer.readline()
                if not raw:break
                motor,_action=runtime.contact_utterance(raw.rstrip(b'\r\n'),1,0)
            else:
                # Readiness supplies physical time only. It cannot inspect or
                # nominate the resident gap, cognition, question, or silence.
                motor=runtime.quiet_public_opportunity(1,0)
            if motor:sys.stdout.buffer.write(motor+b'\n');sys.stdout.buffer.flush()
    finally:
        readiness.close()
        save(args.resume,runtime)


if __name__=='__main__':main()
