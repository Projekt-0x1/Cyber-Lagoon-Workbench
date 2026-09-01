#!/usr/bin/env python3
"""Transient category-safe decomposition of one settled motor consequence.

No field here owns truth or learning. The receipt only separates already-resident witnesses so
callers cannot substitute reward sign for action control, goal attainment, or procedure
execution.
"""
from __future__ import annotations
from dataclasses import dataclass

MIN_CONTROL_PREDICTION_SUPPORT=2

@dataclass(frozen=True)
class ActionOutcomeDecompositionV1:
    ticket:int
    action:int
    independent:bool
    predicted_state:tuple[int,...]
    prediction_support:int
    actual_state:tuple[int,...]
    target:tuple[int,...]
    control_evaluable:bool
    control_match:bool
    goal_evaluable:bool
    goal_attained:bool
    valence:int
    authority:int=0


def _norm(values):return tuple(sorted(set(int(x) for x in (values or ()) if int(x)!=0)))

def decompose_action_outcome(motor,predicted_state=(),prediction_support=0,target=(),satisfies=None):
    if motor is None or not bool(getattr(motor,'settled',False)):
        raise ValueError('action-outcome:unsettled')
    predicted=_norm(predicted_state);actual=_norm(getattr(motor,'state_after',()) or ());target=_norm(target)
    support=max(0,int(prediction_support));independent=bool(getattr(motor,'independent_consequence',False))
    control_evaluable=bool(predicted and support>=MIN_CONTROL_PREDICTION_SUPPORT)
    control_match=bool(control_evaluable and predicted==actual)
    goal_evaluable=bool(target)
    if goal_evaluable:
        goal_attained=bool(satisfies(actual,target)) if callable(satisfies) else set(target).issubset(set(actual))
    else:goal_attained=False
    effect=int(getattr(motor,'effect',0));valence=1 if effect>0 else (-1 if effect<0 else 0)
    return ActionOutcomeDecompositionV1(
        int(getattr(motor,'ticket',0)),int(getattr(motor,'action_id',0)),independent,
        predicted,support,actual,target,control_evaluable,control_match,goal_evaluable,
        goal_attained,valence,0)
