#!/usr/bin/env python3
"""Transient dependency-driven composition of exact resident reduction definitions."""
from __future__ import annotations
from dataclasses import dataclass
from reference_heterogeneous_exact_relation_algebra_v1 import (
    Q,AFFINE,POLYNOMIAL,SCHUR,BISIMULATION,ExactRelationSourceV1,reduce_exact,
)

LITERAL=0
Q16_REF=1
U32_Q16_REF=2
Q16_SELECTED_REF=3


@dataclass(frozen=True)
class ReductionTermV1:
    kind:int
    value:int=0
    node_identity:int=0
    slot:int=0
    selector_identity:int=0

    @staticmethod
    def literal(value):return ReductionTermV1(LITERAL,int(value),0,0,0)
    @staticmethod
    def q16(node_identity,slot):return ReductionTermV1(Q16_REF,0,int(node_identity),int(slot),0)
    @staticmethod
    def u32_q16(node_identity,slot):return ReductionTermV1(U32_Q16_REF,0,int(node_identity),int(slot),0)
    @staticmethod
    def selected_q16(node_identity,selector_identity,selector_slot=0):
        return ReductionTermV1(Q16_SELECTED_REF,0,int(node_identity),int(selector_slot),int(selector_identity))


@dataclass(frozen=True)
class ExactReductionNodeV1:
    identity:int
    kind:int
    coefficients:tuple[ReductionTermV1,...]=()
    successors:tuple[int,...]=()
    outputs_q16:tuple[int,...]=()
    source_identity:int=0

    def __post_init__(self):
        if int(self.identity)<=0:raise ValueError('reduction-program:identity')
        if int(self.kind) not in (AFFINE,POLYNOMIAL,SCHUR,BISIMULATION):raise ValueError('reduction-program:kind')


@dataclass(frozen=True)
class ExactReductionProgramResultV1:
    witness:object
    evaluated:tuple[tuple[int,object],...]

    def witness_for(self,identity):
        identity=int(identity)
        return next((row for key,row in self.evaluated if key==identity),None)


class AutomaticExactReductionProgramV1:
    """Pure recursive resolver. Dependency order is derived from term incidence."""
    @staticmethod
    def resolve(nodes,output_identity,max_nodes=32):
        nodes=tuple(nodes);output_identity=int(output_identity);max_nodes=int(max_nodes)
        if not 1<=max_nodes<=256 or not nodes or len(nodes)>max_nodes:return None
        index={}
        for node in nodes:
            if int(node.identity) in index:return None
            index[int(node.identity)]=node
        if output_identity not in index:return None
        memo={};visiting=set()

        def evaluate(identity):
            identity=int(identity)
            if identity in memo:return memo[identity]
            if identity in visiting:return None
            node=index.get(identity)
            if node is None:return None
            visiting.add(identity)

            def term_value(term):
                if int(term.kind)==LITERAL:return int(term.value)
                dep=evaluate(term.node_identity)
                if dep is None:return None
                slot=int(term.slot)
                if int(term.kind)==Q16_REF:
                    return int(dep.result_q16[slot]) if 0<=slot<len(dep.result_q16) else None
                if int(term.kind)==U32_Q16_REF:
                    if not 0<=slot<len(dep.result_u32):return None
                    value=int(dep.result_u32[slot])*Q
                    return value if -(1<<31)<=value<(1<<31) else None
                if int(term.kind)==Q16_SELECTED_REF:
                    selector=evaluate(term.selector_identity)
                    if selector is None or not 0<=slot<len(selector.result_u32):return None
                    selected=int(selector.result_u32[slot])
                    return int(dep.result_q16[selected]) if 0<=selected<len(dep.result_q16) else None
                return None

            kind=int(node.kind)
            if kind==BISIMULATION:
                if node.coefficients or not 1<=len(node.successors)<=4 or len(node.outputs_q16)!=len(node.successors):
                    visiting.remove(identity);return None
                successors=tuple(map(int,node.successors))+(0,)*(4-len(node.successors))
                outputs=tuple(map(int,node.outputs_q16))+(0,)*(4-len(node.outputs_q16))
                source=ExactRelationSourceV1(kind,successors=successors,outputs_q16=outputs,state_count=len(node.successors)).sealed()
            else:
                if len(node.coefficients)!=4 or node.successors or node.outputs_q16:
                    visiting.remove(identity);return None
                values=[]
                for term in node.coefficients:
                    value=term_value(term)
                    if value is None:
                        visiting.remove(identity);return None
                    values.append(value)
                source=ExactRelationSourceV1(kind,tuple(values)).sealed()
            if int(node.source_identity) and int(node.source_identity)!=int(source.source_identity):
                visiting.remove(identity);return None
            witness=reduce_exact(source)
            visiting.remove(identity)
            if witness is None:return None
            memo[identity]=witness
            return witness

        witness=evaluate(output_identity)
        if witness is None:return None
        evaluated=tuple(sorted(memo.items()))
        return ExactReductionProgramResultV1(witness,evaluated)
