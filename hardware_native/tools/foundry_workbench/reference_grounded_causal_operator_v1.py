#!/usr/bin/env python3
"""Source-qualified grounding of an opaque discourse operator to causal argument orientation."""
from __future__ import annotations
from reference_hierarchical_composition_v1 import _identity
from reference_predictive_credit_profile_v1 import Q

MAX_GROUNDED_CAUSAL_OPERATORS=32
MIN_GROUNDING_SOURCES=2

class GroundedCausalOperatorV1:
    def __init__(self):self.rows={};self.withdrawn=set();self.preferred=0
    def observe(self,factor,left,right,learner,receipt,source,adult=None):
        factor=int(factor);left=int(left);right=int(right);source=int(source)
        if factor<=0 or source<=0:return False
        resolved=learner.resolve(int(receipt))
        if resolved is None:return False
        cause,effect=map(int,resolved)
        same=(lambda a,b: adult.leaf_equivalent(a,b)) if adult is not None else (lambda a,b:int(a)==int(b))
        orientation=1 if same(left,effect) and same(right,cause) else -1 if same(left,cause) and same(right,effect) else 0
        if not orientation:return False
        if factor not in self.rows:
            if len(self.rows)>=MAX_GROUNDED_CAUSAL_OPERATORS:return False
            self.rows[factor]={}
        self.rows[factor][source]=orientation
        if self.orientation(factor):self.preferred=factor
        return True
    def withdraw_source(self,source):self.withdrawn.add(int(source))
    def orientation(self,factor):
        votes=[v for s,v in self.rows.get(int(factor),{}).items() if s not in self.withdrawn]
        pos=sum(v>0 for v in votes);neg=sum(v<0 for v in votes)
        if pos>=MIN_GROUNDING_SOURCES and neg==0:return 1
        if neg>=MIN_GROUNDING_SOURCES and pos==0:return -1
        return 0
    def preferred_factor(self):
        """Return the uniquely best-supported live grounded realization.

        Grounding order is chronology, not a surface-style priority.  Public
        realization therefore follows independent live support rather than the
        last factor that happened to cross quorum.  Ties stay unresolved.
        """
        ranked=[]
        for factor,rows in self.rows.items():
            orientation=self.orientation(factor)
            if not orientation:continue
            live=sum(1 for source in rows if source not in self.withdrawn)
            ranked.append((live,int(factor)))
        if not ranked:return 0
        peak=max(row[0] for row in ranked);winners=tuple(factor for support,factor in ranked if support==peak)
        return winners[0] if len(winners)==1 else 0
    def materialize(self,adult,learner,receipt,factor):
        factor=int(factor);resolved=learner.resolve(int(receipt));orientation=self.orientation(factor)
        if resolved is None or not orientation or learner.complete_source_blocks(int(receipt))<3:return None
        cause,effect=map(int,resolved)
        current_cause=adult.current_leaf_for_historical(cause);current_effect=adult.current_leaf_for_historical(effect)
        if current_cause is None or current_effect is None:return None
        children=(current_effect,current_cause) if orientation>0 else (current_cause,current_effect)
        root=adult._compose_factor(factor,*children);child_ids=tuple(int(row.identity) for row in children)
        context=_identity('grounded-causal-language-v1',(cause,effect,factor,orientation))
        chunk=None
        for _ in range(3):chunk=adult.experience_program(child_ids,root,Q,context=context,effort_q16=Q//16,controllable=True)
        return chunk
    def checkpoint(self):
        return {'schema':1,'rows':[{'factor':f,'sources':[[s,v] for s,v in sorted(r.items())]} for f,r in sorted(self.rows.items())],'withdrawn':sorted(self.withdrawn),'preferred':int(self.preferred)}
    @classmethod
    def restore(cls,data):
        if int(data.get('schema',0))!=1:raise ValueError('grounded_causal_operator:checkpoint')
        out=cls()
        for row in data.get('rows',()):
            f=int(row.get('factor',0));vals={int(s):int(v) for s,v in row.get('sources',())}
            if f<=0 or f in out.rows or any(s<=0 or v not in (-1,1) for s,v in vals.items()):raise ValueError('grounded_causal_operator:row')
            out.rows[f]=vals
        out.withdrawn=set(map(int,data.get('withdrawn',())))
        out.preferred=int(data.get('preferred',0))
        if out.preferred and out.preferred not in out.rows:raise ValueError('grounded_causal_operator:preferred')
        return out
