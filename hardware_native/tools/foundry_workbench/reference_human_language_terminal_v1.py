#!/usr/bin/env python3
"""Strict human-language ingress: raw contact plus grounded semantics plus coherent public output."""
from __future__ import annotations
from reference_coherent_public_discourse_v1 import CoherentPublicDiscourseV1
from reference_language_mastery_contact_adapter_v1 import CONTACT_UTTERANCE,LanguageMasteryContactAdapterV1

class HumanLanguageSessionV1:
    def __init__(self,adult,event_order=None,temporal_questions=None,causal_questions=None,causal_models=(),causal_licenses=(),temporal_licenses=()):
        self.adult=adult;self.event_order=event_order;self.temporal_questions=temporal_questions;self.causal_questions=causal_questions;self.causal_models=tuple(causal_models)
        self.causal_licenses=tuple(causal_licenses);self.temporal_licenses=tuple(temporal_licenses)

    def _semantic_answer(self,scene):
        if scene is None:return 0
        answers=[];context=int(scene.context);atoms=tuple(map(int,scene.atoms))
        if self.event_order is not None and self.temporal_questions is not None:
            answer=int(self.temporal_questions.answer_learned(self.adult,self.event_order,context,atoms))
            if answer:answers.append(answer)
        if self.causal_questions is not None and self.causal_models:
            answer=int(self.causal_questions.answer_learned(self.adult,context,atoms,self.causal_models))
            if answer:answers.append(answer)
        answers=sorted(set(answers));return answers[0] if len(answers)==1 else 0

    def externalize(self,plan):
        return CoherentPublicDiscourseV1.externalize(
            self.adult,plan,self.causal_licenses,self.temporal_licenses)

    def respond(self,raw,source,channel=0):
        contact=LanguageMasteryContactAdapterV1(self.adult)
        identity=contact.contact(CONTACT_UTTERANCE,tuple(raw),int(source),max(0,int(channel)))
        scene=contact.scenes.get(int(identity)) if int(identity)>0 else None
        semantic=self._semantic_answer(scene)
        if semantic:return self.externalize(semantic)
        plan=self.adult.choose_public_plan()
        return self.externalize(plan) if plan else b''
