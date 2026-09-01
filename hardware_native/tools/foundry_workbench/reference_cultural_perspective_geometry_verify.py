#!/usr/bin/env python3
"""Hostile receipt for cultural perspective geometry under conflicting social curricula."""
from __future__ import annotations
import json
from reference_cultural_perspective_geometry_v1 import CulturalPerspectiveGeometryV1,Q

CTX=0xF101
CALM=0xF201
COUNTER=0xF202
WARN=0xF203
DEVIL=0xFA01
ANGEL=0xFA02
SHADE=0xFA03


def row(projected,prop):
    return next(x for x in projected if x['proposition_root']==prop)


def main():
    g=CulturalPerspectiveGeometryV1()

    # Same continuous social situation; several actors contribute incompatible proposal roots.
    assert g.observe(CTX,COUNTER,DEVIL,10)
    assert g.observe(CTX,CALM,ANGEL,10)
    assert g.observe(CTX,WARN,SHADE,10)

    # Rhetorical repetition raises familiarity only. It must not manufacture source plurality.
    for tick in range(11,1011):assert g.observe(CTX,COUNTER,DEVIL,tick)
    props=g.propositions(CTX)
    counter=next(x for x in props if x[0]==COUNTER)
    assert len(counter[1])==1 and counter[1][0][1]==1001

    # External source epistemics: identical geometry, different speaker reliability.
    weights={DEVIL:Q//16,ANGEL:Q,SHADE:Q//8}
    before=json.dumps(g.checkpoint(),sort_keys=True,separators=(',',':'))
    projected=g.project(CTX,weights,{COUNTER:Q,CALM:0,WARN:Q//2})
    assert row(projected,COUNTER)['epistemic_q16']==Q//16
    assert row(projected,CALM)['epistemic_q16']==Q
    assert row(projected,COUNTER)['familiarity']>row(projected,CALM)['familiarity']
    # Somatic salience is visible but separate; it cannot rewrite epistemic support.
    assert row(projected,COUNTER)['somatic_q16']==Q
    assert json.dumps(g.checkpoint(),sort_keys=True,separators=(',',':'))==before

    # Source recovery changes only external weighting; learned cultural geometry is unchanged.
    recovered=dict(weights);recovered[DEVIL]=Q
    after_recovery=g.project(CTX,recovered,{COUNTER:0})
    assert row(after_recovery,COUNTER)['epistemic_q16']==Q
    assert g.checkpoint()==json.loads(before)

    # Retraction is perspective-local rather than global moral rewriting.
    assert g.retract(CTX,WARN,SHADE)
    remaining=dict(g.propositions(CTX))
    assert WARN not in remaining and CALM in remaining and COUNTER in remaining

    # Rich-curriculum quantity: repeated encounters compress into bounded source/proposition rows.
    scale=CulturalPerspectiveGeometryV1()
    for i in range(100000):
        context=1+(i%64);prop=100+(i%256);speaker=1000+(i%16)
        scale.observe(context,prop,speaker,i)
    snap=scale.checkpoint();encoded=json.dumps(snap,sort_keys=True,separators=(',',':'))
    assert len(snap['rows'])<=8192
    assert 'devil' not in encoded.lower() and 'angel' not in encoded.lower()
    restored=CulturalPerspectiveGeometryV1.restore(snap)
    assert restored.checkpoint()==snap

    print('FOUNDRY_CULTURAL_PERSPECTIVE_GEOMETRY_GREEN')
    print('conflicting_sources=3')
    print('rhetorical_repetitions=1000')
    print('repetition_does_not_create_authority=true')
    print('somatic_epistemic_channels_separate=true')
    print('source_local_revision=true')
    print('scale_events=100000')
    print('resident_rows=',len(snap['rows']))
    print('checkpoint_bytes=',len(encoded))


if __name__=='__main__':main()
