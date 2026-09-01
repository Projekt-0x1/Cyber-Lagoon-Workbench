#!/usr/bin/env python3
"""Ground an opaque event-question construction to intervention-resolved cause retrieval."""
from __future__ import annotations
MIN_QUESTION_SOURCES=2
MAX_QUESTION_CONTEXTS=32

class GroundedCausalQuestionV1:
    def __init__(self):self.rows={};self.event_context={};self.withdrawn=set()

    @staticmethod
    def _leaf_context(adult,event_identity):
        identity=int(event_identity);family=adult._surface_leaf_family_index.get(identity)
        if family is None:return 0
        binding=adult.language.historical_template_binding(int(family))
        return 0 if binding is None else int(binding[0])

    def observe_grounded(self,adult,query_context,effect_event,cause_event,learner,receipt,source):
        q=int(query_context);effect=int(effect_event);cause=int(cause_event);source=int(source)
        if min(q,effect,cause,source)<=0:return False
        if learner.complete_source_blocks(int(receipt))<3 or learner.resolve(int(receipt))!=(cause,effect):return False
        context=self._leaf_context(adult,effect)
        if context<=0 or self._leaf_context(adult,cause)!=context:return False
        prior=self.event_context.get(q)
        if prior is not None and int(prior)!=context:return False
        if q not in self.rows:
            if len(self.rows)>=MAX_QUESTION_CONTEXTS:return False
            self.rows[q]=set()
        self.rows[q].add(source);self.event_context[q]=context;return True

    def supported(self,query_context):
        q=int(query_context)
        return q in self.event_context and sum(1 for s in self.rows.get(q,()) if s not in self.withdrawn)>=MIN_QUESTION_SOURCES

    def answer_learned(self,adult,query_context,event_atoms,causal_models):
        q=int(query_context);context=int(self.event_context.get(q,0))
        if not self.supported(q) or context<=0:return 0
        try:effect=adult.leaf(context,tuple(map(int,event_atoms))).identity
        except Exception:return 0
        causes=[]
        for learner,receipt in causal_models:
            if learner.complete_source_blocks(int(receipt))<3:continue
            resolved=learner.resolve(int(receipt))
            if resolved is not None and int(resolved[1])==int(effect):causes.append(int(resolved[0]))
        causes=sorted(set(causes));return causes[0] if len(causes)==1 else 0

    def withdraw_source(self,source):self.withdrawn.add(int(source))
    def checkpoint(self):return {'schema':1,'rows':[{'context':q,'event_context':self.event_context[q],'sources':sorted(s)} for q,s in sorted(self.rows.items())],'withdrawn':sorted(self.withdrawn)}
    @classmethod
    def restore(cls,data):
        if int(data.get('schema',0))!=1:raise ValueError('grounded_causal_question:checkpoint')
        out=cls()
        for row in data.get('rows',()):
            q=int(row.get('context',0));context=int(row.get('event_context',0));sources=set(map(int,row.get('sources',())))
            if min(q,context)<=0 or q in out.rows or any(s<=0 for s in sources):raise ValueError('grounded_causal_question:row')
            out.rows[q]=sources;out.event_context[q]=context
        out.withdrawn=set(map(int,data.get('withdrawn',())));return out
