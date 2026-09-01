#!/usr/bin/env python3
"""Continuous recursive perspective -> zero-authority nomination -> lived/body arbitration.

This reference bridge consumes roots induced by ContinuousPerspectiveInducerV1. Cultural
exposure may retain any recursive proposition, but only a *top-level* imperative whose payload
resolves to one action concept can nominate that action. The same imperative nested under quote,
embedding, assertion, or negation remains perspective structure and cannot become a command.
Ordinary lived transition evidence owns action value. Developmental allostasis may transiently
change current affordance without rewriting culture or consequence credit.
"""
from __future__ import annotations
from dataclasses import dataclass

from reference_continuous_perspective_inducer_v1 import K_IMPERATIVE
from reference_cultural_perspective_geometry_v1 import CulturalPerspectiveGeometryV1, Q
from reference_developmental_social_allostasis_v1 import DevelopmentalSocialAllostasisV1
from reference_language_action_nomination_v1 import LanguageActionNominationBankV1
from reference_open_language_action_affordance_v1 import OpenLanguageActionAffordanceV1

MAX_ACTION_SEARCH_NODES=128


@dataclass(frozen=True)
class CulturalPragmaticDecisionV1:
    action_identity:int
    status:int
    nomination_identity:int
    available_resource:int
    interference_q16:int
    projected:tuple


class RecursiveCulturalPragmaticsV1:
    """Bind induced recursive social propositions to epistemics, soma, and action competition."""

    def __init__(self):
        self.geometry=CulturalPerspectiveGeometryV1()
        self.allostasis=DevelopmentalSocialAllostasisV1()
        self.nominations=LanguageActionNominationBankV1()
        self.affordances=OpenLanguageActionAffordanceV1()

    @staticmethod
    def _single_atom_under(inducer,root):
        """Return one concept leaf or refuse ambiguous/multi-action imperative payloads."""
        stack=[int(root)];seen=set();atoms=[]
        while stack and len(seen)<MAX_ACTION_SEARCH_NODES:
            current=stack.pop()
            if current in seen:continue
            seen.add(current)
            if not inducer.is_node(current):return 0
            tag,children=inducer.node(current)
            if int(tag)==0:
                if len(children)!=1 or int(children[0])<=0:return 0
                atoms.append(int(children[0]))
                continue
            for child in reversed(children):
                if inducer.is_node(child):stack.append(int(child))
        if stack or len(atoms)!=1:return 0
        return atoms[0]

    def action_from_induced_root(self,inducer,root):
        """Structure-dependent advice role: quotation/embedding cannot inherit imperative force."""
        root=int(root)
        if not inducer.is_node(root):return 0
        tag,children=inducer.node(root)
        if int(tag)!=K_IMPERATIVE or len(children)!=1:return 0
        return self._single_atom_under(inducer,int(children[0]))

    def observe_induced_root(self,inducer,context_root,root,speaker,tick):
        """Assimilate any completed recursive proposition as source-qualified perspective."""
        root=int(root)
        if not inducer.is_node(root):return False
        return self.geometry.observe(int(context_root),root,int(speaker),int(tick))

    def observe_language(self,raw,speaker,tick):
        """Ordinary raw contact may later become an affordance only through observed action."""
        return self.affordances.observe_language(raw,speaker,tick)

    def observe_demonstrated_action(self,action_identity,demonstrator,tick):
        return self.affordances.observe_action(action_identity,demonstrator,tick)

    def nominate_language(self,raw,context_root,tick):
        self.nominations.clear()
        return self.affordances.nominate(self.nominations,raw,context_root,tick)

    def project(self,context_root,source_epistemic_q16=None,somatic_bias_q16=None):
        return self.geometry.project(context_root,source_epistemic_q16,somatic_bias_q16)

    def nominate_context(self,inducer,context_root,tick,source_epistemic_q16=None):
        """Rematerialize current structurally valid advice; nomination authority stays zero."""
        self.nominations.clear();rows=[]
        for row in self.project(context_root,source_epistemic_q16):
            proposition=int(row['proposition_root'])
            action=self.action_from_induced_root(inducer,proposition)
            if action<=0:continue
            nomination=self.nominations.nominate(
                action,proposition,int(context_root),proposition,action,int(context_root),int(tick))
            rows.append((nomination,row))
        return tuple(rows)

    def decide(self,inducer,ecology,current_state,goal,context_root,tick,ordinary_candidates=(),
               source_epistemic_q16=None,base_resource=Q,action_costs=None,
               challenge_source=0,challenge_action=0,acute_arousal_q16=0,control_bank=None):
        nominated=self.nominate_context(inducer,context_root,tick,source_epistemic_q16)
        appraisal={'interference_q16':0}
        if int(challenge_source)>0 and int(challenge_action)>0:
            appraisal=self.allostasis.appraise(
                int(challenge_source),int(challenge_action),int(tick),int(acute_arousal_q16))
        interference=max(0,min(Q,int(appraisal.get('interference_q16',0))))
        available=max(0,int(base_resource)-interference)
        decision=self.nominations.arbitrate(
            ecology,current_state,goal,ordinary_candidates,available,
            {} if action_costs is None else action_costs,control_bank)
        return CulturalPragmaticDecisionV1(
            int(decision.action_identity),int(decision.status),int(decision.nomination_identity),
            available,interference,tuple(row for _nomination,row in nominated))

    def checkpoint(self):
        return {
            'schema':1,
            'geometry':self.geometry.checkpoint(),
            'allostasis':self.allostasis.checkpoint(),
            'nominations':self.nominations.checkpoint(),
            'affordances':self.affordances.checkpoint(),
        }
