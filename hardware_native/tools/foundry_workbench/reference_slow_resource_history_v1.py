#!/usr/bin/env python3
"""Content-free slow resource-history challenger for Workbench tournaments."""
from __future__ import annotations
from dataclasses import dataclass
Q=1<<16
HISTORY_WINDOW_TICKS=8
SUSTAINED_MIN_CONTACTS=6
LOAD_SAMPLE_CAP_Q16=Q//8
LOAD_CEILING_Q16=Q
CHRONIC_GATE_Q16=Q//2
QUIET_RECOVERY_STEP_Q16=Q//8
LOAD_GAP_RELAX_Q16=3*Q//4

def mul_q16(a,b): return (int(a)*int(b))>>16

@dataclass
class SlowResourceHistoryV1:
    load_accumulator_q16:int=0
    chronic_pressure_q16:int=0
    sustained_contacts:int=0
    contacts:int=0
    last_contact_tick:int=0
    recovery_windows_applied:int=0
    capped_samples:int=0
    recovery_events:int=0
    def _tick(self,tick):
        tick=int(tick)
        if tick<0: raise ValueError('slow_resource:tick')
        if self.contacts and tick<self.last_contact_tick: raise ValueError('slow_resource:chronology')
        return tick
    def advance(self,tick):
        tick=self._tick(tick)
        if not self.contacts:return self.chronic_pressure_q16
        elapsed=tick-self.last_contact_tick
        if elapsed<=HISTORY_WINDOW_TICKS:return self.chronic_pressure_q16
        self.sustained_contacts=0
        windows=elapsed//HISTORY_WINDOW_TICKS
        new=max(0,windows-self.recovery_windows_applied)
        if new:
            before=self.chronic_pressure_q16
            self.chronic_pressure_q16=max(0,self.chronic_pressure_q16-new*QUIET_RECOVERY_STEP_Q16)
            self.recovery_windows_applied=windows
            if self.chronic_pressure_q16!=before:self.recovery_events+=1
        return self.chronic_pressure_q16
    def observe(self,tick,claimed_load_q16):
        tick=self._tick(tick);claimed=max(0,int(claimed_load_q16))
        if self.contacts:
            gap=tick-self.last_contact_tick
            if gap>HISTORY_WINDOW_TICKS:
                self.advance(tick)
                self.load_accumulator_q16=mul_q16(self.load_accumulator_q16,LOAD_GAP_RELAX_Q16)
                self.sustained_contacts=0
        sample=min(claimed,LOAD_SAMPLE_CAP_Q16)
        if claimed>LOAD_SAMPLE_CAP_Q16:self.capped_samples+=1
        self.load_accumulator_q16=min(LOAD_CEILING_Q16,self.load_accumulator_q16+sample)
        if self.contacts and tick-self.last_contact_tick<=HISTORY_WINDOW_TICKS:self.sustained_contacts+=1
        else:self.sustained_contacts=1
        self.contacts+=1;self.last_contact_tick=tick;self.recovery_windows_applied=0
        if self.sustained_contacts>=SUSTAINED_MIN_CONTACTS and self.load_accumulator_q16>=CHRONIC_GATE_Q16:
            self.chronic_pressure_q16=max(self.chronic_pressure_q16,self.load_accumulator_q16)
        return self.chronic_pressure_q16
    def pressure_q16(self):return int(self.chronic_pressure_q16)
    def lesion_history(self):
        self.load_accumulator_q16=0;self.chronic_pressure_q16=0;self.sustained_contacts=0;self.recovery_windows_applied=0
    def checkpoint(self):
        return {'schema':1,'load_accumulator_q16':self.load_accumulator_q16,'chronic_pressure_q16':self.chronic_pressure_q16,'sustained_contacts':self.sustained_contacts,'contacts':self.contacts,'last_contact_tick':self.last_contact_tick,'recovery_windows_applied':self.recovery_windows_applied,'capped_samples':self.capped_samples,'recovery_events':self.recovery_events}
    @classmethod
    def restore(cls,data):
        if int(data.get('schema',0))!=1:raise ValueError('slow_resource:checkpoint_schema')
        s=cls(*(int(data.get(k,-1)) for k in ('load_accumulator_q16','chronic_pressure_q16','sustained_contacts','contacts','last_contact_tick','recovery_windows_applied','capped_samples','recovery_events')))
        if not 0<=s.load_accumulator_q16<=LOAD_CEILING_Q16 or not 0<=s.chronic_pressure_q16<=LOAD_CEILING_Q16 or min(s.sustained_contacts,s.contacts,s.last_contact_tick,s.recovery_windows_applied,s.capped_samples,s.recovery_events)<0 or s.sustained_contacts>s.contacts or (s.contacts==0 and s.last_contact_tick!=0):raise ValueError('slow_resource:checkpoint_state')
        return s
