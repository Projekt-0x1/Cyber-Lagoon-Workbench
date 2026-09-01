#!/usr/bin/env python3
"""Strict semantic qualification for public multi-proposition language."""
from __future__ import annotations

class CoherentPublicDiscourseV1:
    @staticmethod
    def _template_factor(adult,relation_context):
        template=adult.language.span_template(int(relation_context),2)
        return 0 if template is None else int(template.identity[:15],16)

    @classmethod
    def _causal_edge(cls,adult,operator,members,licenses):
        for grounding,learner,receipt,relation_context in licenses:
            if int(operator)!=cls._template_factor(adult,relation_context):continue
            resolved=learner.resolve(int(receipt));orientation=grounding.orientation(int(relation_context))
            if resolved is None or not orientation or learner.complete_source_blocks(int(receipt))<3:continue
            cause,effect=map(int,resolved);expected=(effect,cause) if orientation>0 else (cause,effect)
            if tuple(map(int,members))==expected:return True
        return False

    @classmethod
    def _temporal_edge(cls,adult,operator,members,licenses):
        left,right=map(int,members)
        for grounding,event_order,relation_context in licenses:
            if int(operator)!=cls._template_factor(adult,relation_context):continue
            orientation=grounding.orientation(int(relation_context))
            if not orientation:continue
            if orientation>0 and event_order.supports(left,right):return True
            if orientation<0 and event_order.supports(right,left):return True
        return False

    @classmethod
    def qualifies(cls,adult,plan,causal_licenses=(),temporal_licenses=()):
        identity=int(getattr(plan,'identity',plan))
        if adult._has_leaf(identity):return True
        seen=set()
        def walk(pid):
            pid=int(pid)
            if adult._has_leaf(pid):return True
            if pid in seen:return False
            chunk=adult.programs.chunks.get(pid)
            if chunk is None or len(chunk.members)!=2:return False
            seen.add(pid)
            try:
                operator=int(adult.programs.factor(pid) or 0);members=tuple(map(int,chunk.members))
                if operator<=0:return False
                if not (cls._causal_edge(adult,operator,members,causal_licenses)
                        or cls._temporal_edge(adult,operator,members,temporal_licenses)):
                    return False
                return all(walk(member) for member in members)
            finally:seen.remove(pid)
        return walk(identity)

    @classmethod
    def externalize(cls,adult,plan,causal_licenses=(),temporal_licenses=()):
        if not cls.qualifies(adult,plan,causal_licenses,temporal_licenses):return b''
        identity=int(getattr(plan,'identity',plan))
        return bytes(adult._leaf_surface(identity)) if adult._has_leaf(identity) else bytes(adult.public_surface(identity))
