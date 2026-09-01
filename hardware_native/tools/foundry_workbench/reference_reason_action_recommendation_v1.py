#!/usr/bin/env python3
"""Typed adapter for source-qualified reasons that recommend an action.

`predicts_success` exists only as a checkpoint-compatibility bridge to the older causal-
experiment storage. It means "the recommendation favors taking this action", not that the
reason proposition predicts a world event or is true when the action succeeds.
"""
from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True)
class ReasonActionRecommendationV1:
    reason:int
    source:int
    action:int
    authority:int=0
    @property
    def predicts_success(self):return True
