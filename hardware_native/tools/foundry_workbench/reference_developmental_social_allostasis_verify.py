#!/usr/bin/env python3
"""Hostile factorial receipt for developmental social allostasis."""
from __future__ import annotations
import copy,json
from reference_developmental_social_allostasis_v1 import DevelopmentalSocialAllostasisV1,Q

SRC=0xE101
ACT=0xE201


def trained_arm(recent_betrayal,scarcity,control):
    a=DevelopmentalSocialAllostasisV1()
    if recent_betrayal:
        a.observe_betrayal(SRC,960,Q)
    else:
        a.observe_betrayal(SRC,64,Q)
    if scarcity:
        for _ in range(96):a.observe_scarcity(Q)
    if control:
        for _ in range(4):assert a.observe_controllability(SRC,ACT,Q,True)
    return a


def main():
    # Full 2x2x2x2 intervention: betrayal timing, chronic scarcity, controllability, acute arousal.
    rows=[]
    for recent in (0,1):
        for scarcity in (0,1):
            for control in (0,1):
                adult=trained_arm(recent,scarcity,control)
                durable=json.dumps(adult.checkpoint(),sort_keys=True,separators=(',',':'))
                for acute in (0,1):
                    before=copy.deepcopy(adult.checkpoint())
                    app=adult.appraise(SRC,ACT,1024,Q if acute else 0)
                    after=adult.checkpoint()
                    assert before==after  # acute physiology cannot rewrite developmental learning
                    rows.append((recent,scarcity,control,acute,app))
                assert durable==json.dumps(adult.checkpoint(),sort_keys=True,separators=(',',':'))

    def get(r,s,c,a):
        return next(x[4] for x in rows if x[:4]==(r,s,c,a))

    # Betrayal timing matters with wording/direct world evidence held outside this owner.
    old=get(0,0,0,0);recent=get(1,0,0,0)
    assert recent['betrayal_q16']>old['betrayal_q16']
    assert recent['boundary_drive_q16']>old['boundary_drive_q16']

    # Chronic scarcity raises organism-wide load without changing source betrayal history.
    base=get(1,0,0,0);scarce=get(1,1,0,0)
    assert scarce['betrayal_q16']==base['betrayal_q16']
    assert scarce['arousal_q16']>base['arousal_q16']
    assert scarce['interference_q16']>base['interference_q16']

    # Independently learned controllability reduces interference in the matched source/action.
    noctl=get(1,0,0,0);ctl=get(1,0,1,0)
    assert ctl['control_q16']>0
    assert ctl['interference_q16']<noctl['interference_q16']

    # Acute arousal is transient and can suppress effective control without erasing it.
    calm=get(1,0,1,0);acute=get(1,0,1,1)
    assert acute['control_q16']==calm['control_q16']
    assert acute['effective_control_q16']<calm['effective_control_q16']
    assert acute['interference_q16']>calm['interference_q16']
    assert acute['boundary_drive_q16']>calm['boundary_drive_q16']

    # Repeated scale updates stay bounded by source/action state, not one row per encounter.
    scale=DevelopmentalSocialAllostasisV1()
    for i in range(8192):
        source=1+(i%128);action=1+(i%16)
        if i%7==0:scale.observe_betrayal(source,i,Q//2)
        if i%3==0:scale.observe_scarcity(Q//3)
        if i%5==0:scale.observe_controllability(source,action,Q//2,True)
    snap=scale.checkpoint()
    assert len(snap['betrayal'])<=128
    assert len(snap['control'])<=128*16
    encoded=json.dumps(snap,sort_keys=True,separators=(',',':'))
    assert 'trust' not in encoded and 'insult' not in encoded and 'moral' not in encoded

    print('FOUNDRY_DEVELOPMENTAL_SOCIAL_ALLOSTASIS_GREEN')
    print('factorial_arms=',len(rows))
    print('recent_vs_old_boundary=',recent['boundary_drive_q16'],old['boundary_drive_q16'])
    print('scarcity_interference=',scarce['interference_q16'],base['interference_q16'])
    print('control_interference=',ctl['interference_q16'],noctl['interference_q16'])
    print('acute_interference=',acute['interference_q16'],calm['interference_q16'])
    print('scale_events=8192')
    print('scale_betrayal_rows=',len(snap['betrayal']))
    print('scale_control_rows=',len(snap['control']))
    print('reference_only=true')


if __name__=='__main__':main()
