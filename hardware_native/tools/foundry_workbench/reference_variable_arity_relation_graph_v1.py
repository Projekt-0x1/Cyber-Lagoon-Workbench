#!/usr/bin/env python3
"""Transient exact composition over opaque learned-relation incidence."""
from __future__ import annotations
from dataclasses import dataclass
import hashlib,json
from reference_heterogeneous_exact_relation_algebra_v1 import compose_affine


def _id(payload):
    raw=json.dumps(payload,separators=(',',':'),sort_keys=True).encode()
    return int.from_bytes(hashlib.sha256(b'variable-arity-relation-edge-v1\0'+raw).digest()[:8],'big') or 1


@dataclass(frozen=True)
class RelationEdgeV1:
    left_space:int
    right_space:int
    boundary_q16:tuple[int,int]
    evidence_identity:int

    @classmethod
    def make(cls,left_space,right_space,boundary_q16,evidence_identity):
        left_space=int(left_space);right_space=int(right_space);boundary_q16=tuple(map(int,boundary_q16));evidence_identity=int(evidence_identity)
        if left_space<=0 or right_space<=0 or left_space==right_space or len(boundary_q16)!=2 or evidence_identity<=0:return None
        return cls(left_space,right_space,boundary_q16,evidence_identity)

    @property
    def identity(self):
        return _id((self.left_space,self.right_space,self.boundary_q16,self.evidence_identity))


@dataclass(frozen=True)
class RelationClosureV1:
    left_space:int
    right_space:int
    boundary_q16:tuple[int,int]
    paths:tuple[tuple[int,...],...]


class VariableArityRelationGraphV1:
    """Pure bounded search over current edge incidence; stores no path/cache state."""
    @staticmethod
    def _paths(edges,left_space,right_space,max_depth):
        edges=tuple(sorted(edges,key=lambda e:(e.left_space,e.right_space,e.identity)))
        outgoing={}
        for edge in edges:outgoing.setdefault(int(edge.left_space),[]).append(edge)
        found=[]
        def visit(space,visited,path):
            if len(path)>=max_depth:return
            for edge in outgoing.get(int(space),()):
                nxt=int(edge.right_space)
                if nxt in visited:continue
                next_path=path+(edge,)
                if nxt==int(right_space):found.append(next_path);continue
                visit(nxt,visited|{nxt},next_path)
        visit(int(left_space),{int(left_space)},())
        return tuple(found)

    @staticmethod
    def _compose(path):
        if not path:return None
        boundary=tuple(path[0].boundary_q16)
        for edge in path[1:]:
            boundary=compose_affine(boundary,tuple(edge.boundary_q16))
            if boundary is None:return None
        return boundary

    @classmethod
    def resolve(cls,edges,left_space,right_space,max_depth=6):
        left_space=int(left_space);right_space=int(right_space);max_depth=int(max_depth)
        if left_space<=0 or right_space<=0 or left_space==right_space or not 1<=max_depth<=16:return None
        paths=cls._paths(tuple(edges),left_space,right_space,max_depth)
        if not paths:return None
        rows=[]
        for path in paths:
            boundary=cls._compose(path)
            if boundary is None:return None
            rows.append((boundary,tuple(edge.identity for edge in path)))
        values={row[0] for row in rows}
        if len(values)!=1:return None
        boundary=next(iter(values))
        path_ids=tuple(sorted((row[1] for row in rows),key=lambda x:(len(x),x)))
        return RelationClosureV1(left_space,right_space,boundary,path_ids)
