#!/usr/bin/env python3
"""Developmental social allostasis without semantic or action-value authority.

Sapolsky-style causal depth is represented by distinct timescales: source-local betrayal
history, organism-wide chronic scarcity, learned source/action controllability, and transient
acute arousal.  The owner emits somatic arbitration modifiers only.  It never stores wording,
Merge roots, reputation polarity, moral labels, or public-action values.
"""
from __future__ import annotations

Q=1<<16
MAX_SOURCES=1024
MAX_CONTROL_ROWS=2048
BETRAYAL_DECAY_TICKS=64


def _clip(x,lo=0,hi=Q):
    return max(lo,min(hi,int(x)))


def _decay(value,delta):
    value=_clip(value);delta=max(0,int(delta))
    # Deterministic half-life approximation without floating point.
    halves=delta//BETRAYAL_DECAY_TICKS
    if halves>=16:return 0
    value >>= halves
    rem=delta%BETRAYAL_DECAY_TICKS
    return max(0,value-(value*rem)//(2*BETRAYAL_DECAY_TICKS))


class DevelopmentalSocialAllostasisV1:
    """Learn slow body context; accept acute state only as a transient intervention."""

    def __init__(self):
        self._betrayal={}   # source -> [load_q16,last_tick]
        self._scarcity_q16=0
        self._control={}    # (source,action) -> [success_q16,total_q16]

    def observe_betrayal(self,source,tick,severity_q16=Q):
        source=int(source);tick=int(tick);severity=_clip(severity_q16)
        if source<=0 or tick<0:raise ValueError('developmental-allostasis:betrayal')
        row=self._betrayal.get(source)
        if row is None:
            if len(self._betrayal)>=MAX_SOURCES:raise RuntimeError('developmental-allostasis:source-capacity')
            prior=0
        else:
            if tick<int(row[1]):raise ValueError('developmental-allostasis:time-reversal')
            prior=_decay(row[0],tick-int(row[1]))
        # Salient violations update quickly but remain graded rather than categorical.
        load=_clip(prior + ((Q-prior)*severity)//Q//2)
        self._betrayal[source]=[load,tick]
        return load

    def observe_safe_contact(self,source,tick):
        source=int(source);tick=int(tick)
        if source<=0 or tick<0:raise ValueError('developmental-allostasis:safe-contact')
        row=self._betrayal.get(source)
        if row is None:return 0
        if tick<int(row[1]):raise ValueError('developmental-allostasis:time-reversal')
        load=_decay(row[0],tick-int(row[1]))
        # Safe lived contact is weak extinction, not semantic forgiveness.
        load=(load*7)//8
        self._betrayal[source]=[load,tick]
        return load

    def observe_scarcity(self,severity_q16):
        severity=_clip(severity_q16)
        # Slow allostatic accumulation; one acute episode cannot become chronic context.
        self._scarcity_q16=_clip((self._scarcity_q16*15 + severity)//16)
        return self._scarcity_q16

    def observe_controllability(self,source,action,relief_q16,independent=True):
        source=int(source);action=int(action);relief=_clip(relief_q16)
        if source<=0 or action<=0:raise ValueError('developmental-allostasis:control')
        if not independent:return False
        key=(source,action);row=self._control.get(key)
        if row is None:
            if len(self._control)>=MAX_CONTROL_ROWS:raise RuntimeError('developmental-allostasis:control-capacity')
            row=[0,0];self._control[key]=row
        row[0]=min(0x7fffffff,int(row[0])+relief)
        row[1]=min(0x7fffffff,int(row[1])+Q)
        return True

    def betrayal_load_q16(self,source,tick):
        row=self._betrayal.get(int(source))
        if row is None:return 0
        if int(tick)<int(row[1]):raise ValueError('developmental-allostasis:time-reversal')
        return _decay(row[0],int(tick)-int(row[1]))

    def control_q16(self,source,action):
        row=self._control.get((int(source),int(action)))
        if not row or row[1]<=0:return 0
        return _clip((int(row[0])*Q)//int(row[1]))

    def appraise(self,source,action,tick,acute_arousal_q16=0):
        """Return transient somatic modifiers; learned linguistic/action state is external."""
        betrayal=self.betrayal_load_q16(source,tick)
        scarcity=_clip(self._scarcity_q16)
        acute=_clip(acute_arousal_q16)
        control=self.control_q16(source,action)
        arousal=_clip(acute + betrayal//3 + scarcity//3)
        threat=_clip(betrayal + scarcity//2 + acute//2)
        effective_control=_clip(control - acute//2 - scarcity//4)
        interference=_clip(threat - effective_control//2)
        boundary_drive=_clip(betrayal + acute//2 + scarcity//4 - effective_control//3)
        return {
            'betrayal_q16':betrayal,
            'scarcity_q16':scarcity,
            'acute_q16':acute,
            'control_q16':control,
            'effective_control_q16':effective_control,
            'arousal_q16':arousal,
            'interference_q16':interference,
            'boundary_drive_q16':boundary_drive,
        }

    def checkpoint(self):
        return {
            'schema':1,
            'betrayal':[[int(s),int(v),int(t)] for s,(v,t) in sorted(self._betrayal.items())],
            'scarcity_q16':int(self._scarcity_q16),
            'control':[[int(s),int(a),int(ok),int(total)] for (s,a),(ok,total) in sorted(self._control.items())],
        }

    @classmethod
    def restore(cls,data):
        if int(data.get('schema',0))!=1:raise RuntimeError('developmental-allostasis:checkpoint-schema')
        out=cls();out._scarcity_q16=_clip(data.get('scarcity_q16',0))
        for row in data.get('betrayal',()):
            if len(row)!=3:raise RuntimeError('developmental-allostasis:checkpoint-betrayal')
            source,value,tick=map(int,row)
            if source<=0 or not 0<=value<=Q or tick<0:raise RuntimeError('developmental-allostasis:checkpoint-betrayal')
            out._betrayal[source]=[value,tick]
        for row in data.get('control',()):
            if len(row)!=4:raise RuntimeError('developmental-allostasis:checkpoint-control')
            source,action,ok,total=map(int,row)
            if source<=0 or action<=0 or ok<0 or total<=0 or ok>total:raise RuntimeError('developmental-allostasis:checkpoint-control')
            out._control[(source,action)]=[ok,total]
        if len(out._betrayal)>MAX_SOURCES or len(out._control)>MAX_CONTROL_ROWS:
            raise RuntimeError('developmental-allostasis:checkpoint-capacity')
        return out
