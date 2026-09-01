#!/usr/bin/env python3
"""Ground an opaque event-question construction to lived successor retrieval."""
from __future__ import annotations
MIN_QUESTION_SOURCES=2
MAX_QUESTION_CONTEXTS=32

class GroundedTemporalQuestionV1:
    def __init__(self):self.rows={};self.event_context={};self.withdrawn=set()

    @staticmethod
    def _leaf_context(adult,event_identity):
        identity=int(event_identity);family=adult._surface_leaf_family_index.get(identity)
        if family is None:return 0
        binding=adult.language.historical_template_binding(int(family))
        return 0 if binding is None else int(binding[0])

    def observe(self,query_context,anchor_event,answer_event,event_order,source):
        q=int(query_context);a=int(anchor_event);b=int(answer_event);source=int(source)
        if min(q,a,b,source)<=0 or not event_order.supports(a,b):return False
        if q not in self.rows:
            if len(self.rows)>=MAX_QUESTION_CONTEXTS:return False
            self.rows[q]=set()
        self.rows[q].add(source);return True

    def observe_grounded(self,adult,query_context,anchor_event,answer_event,event_order,source):
        q=int(query_context);a=int(anchor_event);b=int(answer_event);context=self._leaf_context(adult,a)
        if context<=0 or self._leaf_context(adult,b)!=context:return False
        prior=self.event_context.get(q)
        if prior is not None and int(prior)!=context:return False
        if not self.observe(q,a,b,event_order,source):return False
        self.event_context[q]=context;return True

    def supported(self,query_context):
        q=int(query_context)
        return q in self.event_context and sum(1 for s in self.rows.get(q,()) if s not in self.withdrawn)>=MIN_QUESTION_SOURCES

    def answer(self,adult,event_order,query_context,event_context,event_atoms):
        if not self.supported(query_context):return 0
        if int(self.event_context.get(int(query_context),0))!=int(event_context):return 0
        try:anchor=adult.leaf(int(event_context),tuple(map(int,event_atoms)))
        except Exception:return 0
        return int(event_order.successor(anchor.identity))

    def answer_learned(self,adult,event_order,query_context,event_atoms):
        context=int(self.event_context.get(int(query_context),0))
        return self.answer(adult,event_order,query_context,context,event_atoms) if context>0 else 0

    def withdraw_source(self,source):self.withdrawn.add(int(source))

    def checkpoint(self):
        return {'schema':2,'rows':[{'context':q,'event_context':int(self.event_context.get(q,0)),'sources':sorted(s)} for q,s in sorted(self.rows.items())],'withdrawn':sorted(self.withdrawn)}

    @classmethod
    def restore(cls,data):
        schema=int(data.get('schema',0))
        if schema not in (1,2):raise ValueError('grounded_temporal_question:checkpoint')
        out=cls()
        for row in data.get('rows',()):
            q=int(row.get('context',0));sources=set(map(int,row.get('sources',())))
            if q<=0 or q in out.rows or any(s<=0 for s in sources):raise ValueError('grounded_temporal_question:row')
            out.rows[q]=sources
            if schema>=2:
                context=int(row.get('event_context',0))
                if context<=0:raise ValueError('grounded_temporal_question:event_context')
                out.event_context[q]=context
        out.withdrawn=set(map(int,data.get('withdrawn',())))
        return out
