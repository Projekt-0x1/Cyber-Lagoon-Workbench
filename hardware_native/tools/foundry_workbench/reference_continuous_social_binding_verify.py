#!/usr/bin/env python3
"""Hostile structural/scale receipt for continuous social binding.

This verifier exercises overlap, source-qualified Merge provenance, checkpoint replay and
bounded drained-state growth. Full Adult reputation/body arbitration remains covered by the
adjacent open-speech and persistent-social-stream receipts.
"""
from __future__ import annotations
import json
from reference_continuous_social_binding_v1 import ContinuousSocialBindingV1


class FactorProbe:
    def __init__(self):self.calls=[]
    def observe_open_contact(self,adult,raw,speaker):
        self.calls.append((tuple(raw),int(speaker)))
        return adult.accept(int(speaker),tuple(raw))


class AdultProbe:
    def __init__(self,reliable):self.reliable=dict(reliable)
    def accept(self,speaker,raw):return bool(self.reliable.get(int(speaker),False))


def main():
    root=0xCAFE01
    wording=b'do not trust casey'
    adult=AdultProbe({11:True,12:False,13:True})
    factor=FactorProbe();binding=ContinuousSocialBindingV1()

    # Identical wording and Merge proposition from contradictory sources stays source-qualified.
    binding.admit(10,11,wording,root)
    binding.admit(10,12,wording,root)
    binding.admit(10,13,b'casey -- do not trust',root)
    out=binding.drain_until(adult,factor,10)
    assert [x[2] for x in out]==[11,12,13]
    assert [x[4] for x in out]==[True,False,True]
    assert binding.sources_for(root)==((11,1,10),(12,1,10),(13,1,10))

    # Mid-overlap checkpoint must replay exactly, including still-live acoustic occurrences.
    for i in range(7):binding.admit(20+i//3,11+i%3,b'x'+bytes([i]),root+i%2)
    checkpoint=json.loads(json.dumps(binding.checkpoint(),sort_keys=True))
    restored=ContinuousSocialBindingV1.restore(checkpoint)
    f1=FactorProbe();f2=FactorProbe()
    a1=AdultProbe({11:True,12:False,13:True});a2=AdultProbe({11:True,12:False,13:True})
    assert binding.drain_until(a1,f1,30)==restored.drain_until(a2,f2,30)
    assert f1.calls==f2.calls
    assert binding.checkpoint()==restored.checkpoint()

    # Quantity: 4,096 contacts in bounded batches; drained history must not become transcripts.
    scale=ContinuousSocialBindingV1();probe=FactorProbe();adult=AdultProbe({21:True,22:False,23:True,24:False})
    for base in range(0,4096,256):
        for i in range(base,base+256):
            scale.admit(i//4,21+(i%4),b'opaque-'+bytes([i%251]),0xD000+(i%64))
        scale.drain_until(adult,probe,(base+255)//4)
    assert scale.pending_count==0
    snap=scale.checkpoint()
    assert len(snap['provenance'])<=64*4
    assert not snap['pending']
    encoded=json.dumps(snap,sort_keys=True,separators=(',',':'))
    assert 'opaque-' not in encoded

    print('FOUNDRY_CONTINUOUS_SOCIAL_BINDING_GREEN')
    print('identical_wording_source_results=true,false,true')
    print('overlap_sources=3')
    print('scale_contacts=4096')
    print('resident_provenance_rows=',len(snap['provenance']))
    print('pending_after_drain=',scale.pending_count)
    print('checkpoint_bytes=',len(encoded))


if __name__=='__main__':main()
