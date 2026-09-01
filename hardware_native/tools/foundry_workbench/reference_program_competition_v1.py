#!/usr/bin/env python3
"""Reference-only parallel competition over already-earned causal Programs.

Candidate generation is elsewhere. This factor reads learned predictive/control
state and current organism state, then returns unresolved or one unique leader.
It creates no evidence and has no motor authority.
"""
from dataclasses import dataclass
from reference_predictive_credit_profile_v1 import Q

BASE_MARGIN=Q//8
NEG=-(1<<60)

@dataclass(frozen=True)
class ProgramCompetitionStateV1:
    context:int
    resource_pressure_q16:int=0

@dataclass(frozen=True)
class ProgramCompetitionReceiptV1:
    status:int                 # 0 none, 1 leader, 2 unresolved
    leader:int
    runner_up:int
    leader_drive:int
    runner_up_drive:int
    required_margin:int
    considered:int


def _clamp(v): return max(0,min(Q,int(v)))

def prospective_drive(row,bank,state):
    if row is None or not row.control_supported:return NEG
    context=int(state.context)
    score=int(bank.contextual_outcome(row.structure_id,context))
    score+=int(bank.contextual_somatic(row.structure_id,context))
    score+=int(row.accessibility_q16)//4
    score-=int(row.effort_mean_q16)//8
    pressure=_clamp(state.resource_pressure_q16)
    if pressure:
        margin=max(Q//16,Q-pressure)
        score-=(int(row.effort_mean_q16)*pressure)//margin
    return score

def required_competition_margin():
    return BASE_MARGIN

def arbitrate_programs(bank,candidate_ids,state):
    rows=[]
    seen=set()
    for sid in candidate_ids:
        sid=int(sid)
        if sid<=0 or sid in seen:continue
        seen.add(sid)
        row=bank.rows.get(sid)
        drive=prospective_drive(row,bank,state)
        if drive<=NEG//2:continue
        rows.append((drive,sid))
    if not rows:return ProgramCompetitionReceiptV1(0,0,0,0,0,required_competition_margin(),0)
    rows.sort(key=lambda x:(-x[0],x[1]))
    lead,leader=rows[0]
    if len(rows)==1:
        return ProgramCompetitionReceiptV1(1,leader,0,lead,NEG,required_competition_margin(),1)
    runner_drive,runner=rows[1]
    margin=required_competition_margin()
    separation=lead-runner_drive
    if separation<=0 or separation<margin:
        return ProgramCompetitionReceiptV1(2,0,runner,lead,runner_drive,margin,len(rows))
    return ProgramCompetitionReceiptV1(1,leader,runner,lead,runner_drive,margin,len(rows))

@dataclass(frozen=True)
class ProgramBrakeEvidenceV1:
    evidence_identity:int
    action_identity:int
    context:int
    magnitude_q16:int
    tick:int

@dataclass(frozen=True)
class ProgramBrakeReceiptV1:
    decision:int  # 0 unresolved, 1 commit, 2 mismatch brake, 3 malformed evidence
    candidate:int

def apply_program_brake(competition,evidence=None,threshold_q16=Q//2,current_tick=0):
    if competition.status!=1 or competition.leader<=0:return ProgramBrakeReceiptV1(0,0)
    candidate=int(competition.leader)
    if evidence is None:return ProgramBrakeReceiptV1(1,candidate)
    if (evidence.evidence_identity<=0 or evidence.action_identity!=candidate or evidence.context<=0 or
        evidence.tick>int(current_tick) or not 0<=evidence.magnitude_q16<=Q):
        return ProgramBrakeReceiptV1(3,candidate)
    return ProgramBrakeReceiptV1(2 if evidence.magnitude_q16>=_clamp(threshold_q16) else 1,candidate)
