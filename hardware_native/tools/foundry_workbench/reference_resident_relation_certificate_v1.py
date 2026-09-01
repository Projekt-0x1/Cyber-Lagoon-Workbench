#!/usr/bin/env python3
"""Resident certificate that a public relation program is backed by current evidence."""
from __future__ import annotations
from dataclasses import dataclass
from reference_causal_attribution_ecology_v1 import Refuse

@dataclass(frozen=True)
class ResidentRelationCertificateV1:
    relation_kind:str
    receipt:int
    program:int
    operator:int
    left:int
    right:int
    source_blocks:int
    opened_tick:int

    @staticmethod
    def causal(adult,receipt,program,operator,grounded=None):
        receipt=int(receipt);program=int(program);operator=int(operator)
        resolved=adult.world_causal_learning.resolve(receipt)
        if resolved is None:raise Refuse('relation certificate unresolved causal receipt')
        left,right=map(int,resolved);blocks=int(adult.world_causal_learning.complete_source_blocks(receipt))
        if blocks<3:raise Refuse('relation certificate incomplete causal evidence')
        rows=[row for row in adult.world_causal_learning.current_resolutions() if int(row[4])==receipt]
        if len(rows)!=1:raise Refuse('relation certificate causal receipt not current')
        causes,bound_effect,current_left,current_right,current_receipt,opened=rows[0]
        if int(current_receipt)!=receipt or int(current_left)!=left or int(current_right)!=right or int(bound_effect)!=right:
            raise Refuse('relation certificate causal resolution mismatch')
        chunk=adult.programs.chunks.get(program)
        orientation=0 if grounded is None else int(grounded.orientation(operator))
        expected=((right,left) if orientation>0 else (left,right))
        if grounded is not None and not orientation:raise Refuse('relation certificate operator ungrounded')
        if chunk is None or tuple(map(int,chunk.members))!=expected:
            raise Refuse('relation certificate program relation mismatch')
        try:
            witness=adult._compose_factor(operator,*expected)
        except Exception as exc:
            raise Refuse('relation certificate semantic operator unavailable') from exc
        if int(adult.programs.factor(program) or 0)!=int(witness.template_identity):
            raise Refuse('relation certificate semantic operator template mismatch')
        return ResidentRelationCertificateV1('causal',receipt,program,operator,left,right,blocks,int(opened))
