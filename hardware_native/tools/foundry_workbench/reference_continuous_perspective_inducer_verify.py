#!/usr/bin/env python3
"""Hostile receipt for continuous recursive perspective induction under social interruption."""
from __future__ import annotations
import json
from reference_continuous_perspective_inducer_v1 import (
    ContinuousPerspectiveInducerV1,K_ASSERT,K_NEGATE,K_QUOTE,K_EMBED,K_IMPERATIVE
)
from reference_cultural_perspective_geometry_v1 import CulturalPerspectiveGeometryV1,Q

A=0xA1;B=0xB1;C=0xC1
CTX=0x9001


def contains_source(g,root,source):
    tag,children=g.node(root)
    if tag in (K_QUOTE,K_EMBED) and children[0]==source:return True
    return any(g.is_node(c) and contains_source(g,c,source) for c in children)


def main():
    g=ContinuousPerspectiveInducerV1()

    # A begins a proposition, quotes B, embeds C under negation. C interrupts independently.
    g.begin(A,K_ASSERT,1);g.emit(A,10,2)
    g.begin(A,K_QUOTE,3,embedded_source=B)
    g.begin(A,K_NEGATE,4);g.emit(A,20,5)
    g.begin(A,K_EMBED,6,embedded_source=C)
    g.begin(C,K_ASSERT,7);g.emit(C,99,8);c_root=g.end(C,9)
    g.emit(A,30,10);g.end(A,11);g.end(A,12);quote_root=g.end(A,13)
    g.emit(A,40,14);a_root=g.end(A,15)

    roots=g.roots_since()
    assert roots[0]==(9,C,c_root) and roots[1]==(15,A,a_root)
    assert contains_source(g,quote_root,B) and contains_source(g,quote_root,C)

    # The acquired recursive operators compose unseen concepts at greater depth without fixtures.
    tick=20;g.begin(B,K_ASSERT,tick)
    for depth,src in enumerate((A,C,A,C,A),1):
        tick+=1;g.begin(B,K_QUOTE if depth%2 else K_EMBED,tick,embedded_source=src)
        tick+=1;g.begin(B,K_NEGATE,tick)
    tick+=1;g.emit(B,0xDEAD,tick)
    for _ in range(10):tick+=1;g.end(B,tick)
    tick+=1;deep_root=g.end(B,tick)
    assert contains_source(g,deep_root,A) and contains_source(g,deep_root,C)

    # Imperative force remains proposition structure. It receives no text-imposed action authority.
    tick+=1;g.begin(A,K_IMPERATIVE,tick);tick+=1;g.emit(A,0xBEEF,tick)
    tick+=1;imperative_root=g.end(A,tick)
    culture=CulturalPerspectiveGeometryV1();culture.observe(CTX,imperative_root,A,tick)
    projected=culture.project(CTX,{A:Q//8},{imperative_root:Q})
    assert projected[0]['proposition_root']==imperative_root
    assert projected[0]['epistemic_q16']==Q//8 and projected[0]['somatic_q16']==Q
    assert 'action' not in projected[0] and 'policy' not in projected[0]

    snap=g.checkpoint();encoded=json.dumps(snap,sort_keys=True,separators=(',',':'))
    assert 'quote' not in encoded.lower() and 'imperative' not in encoded.lower()
    assert ContinuousPerspectiveInducerV1.restore(snap).checkpoint()==snap

    # Quantity: 100,000 continuous events across eight actors compress into bounded DAG/root state.
    scale=ContinuousPerspectiveInducerV1();tick=0
    for i in range(25000):
        speaker=1+(i%8);tick+=1;scale.begin(speaker,K_ASSERT,tick)
        tick+=1;scale.emit(speaker,100+(i%64),tick)
        tick+=1;scale.emit(speaker,200+(i%32),tick)
        tick+=1;scale.end(speaker,tick)
    ss=scale.checkpoint()
    assert tick==100000
    assert len(ss['roots'])<=8192 and len(ss['nodes'])<=65536

    print('FOUNDRY_CONTINUOUS_PERSPECTIVE_INDUCTION_GREEN')
    print('interruption_source_separation=true')
    print('quotation_negation_embedding_recursive=true')
    print('unseen_recursive_depth=5')
    print('imperative_action_authority=false')
    print('scale_stream_events=',tick)
    print('resident_roots=',len(ss['roots']))
    print('resident_nodes=',len(ss['nodes']))


if __name__=='__main__':main()
