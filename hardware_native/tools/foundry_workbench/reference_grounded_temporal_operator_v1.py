#!/usr/bin/env python3
"""Ground an opaque connective's argument orientation in lived event succession."""
from __future__ import annotations
from reference_hierarchical_composition_v1 import _identity
from reference_predictive_credit_profile_v1 import Q
MIN_GROUNDING_SOURCES=2
class GroundedTemporalOperatorV1:
    def __init__(self):self.rows={};self.withdrawn=set()
    def observe(self,factor,left,right,event_order,source):
        factor=int(factor);left=int(left);right=int(right);source=int(source)
        if min(factor,left,right,source)<=0:return False
        orientation=1 if event_order.supports(left,right) else -1 if event_order.supports(right,left) else 0
        if not orientation:return False
        self.rows.setdefault(factor,{})[source]=orientation;return True
    def withdraw_source(self,source):self.withdrawn.add(int(source))
    def orientation(self,factor):
        vals=[v for s,v in self.rows.get(int(factor),{}).items() if s not in self.withdrawn];p=sum(v>0 for v in vals);n=sum(v<0 for v in vals)
        if p>=MIN_GROUNDING_SOURCES and n==0:return 1
        if n>=MIN_GROUNDING_SOURCES and p==0:return -1
        return 0
    def materialize(self,adult,event_order,left,right,factor):
        left=int(left);right=int(right);factor=int(factor);orientation=self.orientation(factor)
        if not orientation:return None
        ordered=(left,right) if orientation>0 else (right,left)
        if not event_order.supports(*ordered):return None
        if not adult._has_leaf(left) or not adult._has_leaf(right):return None
        root=adult.compose(factor,left,right)
        context=_identity('grounded-temporal-language-v1',(left,right,factor,orientation))
        chunk=None
        for _ in range(3):chunk=adult.experience_program((left,right),root,Q,context=context,effort_q16=Q//16,controllable=True)
        return chunk
    def checkpoint(self):return {'schema':1,'rows':[{'factor':f,'sources':[[s,v] for s,v in sorted(r.items())]} for f,r in sorted(self.rows.items())],'withdrawn':sorted(self.withdrawn)}
    @classmethod
    def restore(cls,data):
        if int(data.get('schema',0))!=1:raise ValueError('grounded_temporal_operator:checkpoint')
        out=cls()
        for row in data.get('rows',()):
            f=int(row.get('factor',0));vals={int(a):int(b) for a,b in row.get('sources',())}
            if f<=0 or f in out.rows or any(a<=0 or b not in (-1,1) for a,b in vals.items()):raise ValueError('grounded_temporal_operator:row')
            out.rows[f]=vals
        out.withdrawn=set(map(int,data.get('withdrawn',())))
        return out
