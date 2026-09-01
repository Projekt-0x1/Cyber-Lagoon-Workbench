#!/usr/bin/env python3
"""Partner-local utterance surfaces. Not a hierarchy membership tree."""
from __future__ import annotations

from dataclasses import dataclass

from reference_hierarchical_composition_v1 import HierarchicalRefuse, _identity, rematerialize_transient_plan


@dataclass(frozen=True)
class UtteranceLeafV1:
    identity:int
    context:int
    surface:tuple[int,...]
    depth:int=0
    child_identities:tuple=()
    ancestry:tuple=()


class UtteranceBoundaryBankV1:
    def __init__(self):
        self._leaves={};self._scratch={}

    def remember(self, context, resident_identity, surface, persist=True):
        surface=tuple(int(x) for x in surface)
        if not surface or any(x<0 or x>255 for x in surface):raise HierarchicalRefuse('hierarchy:surface')
        ident=_identity('hierarchy-resident-leaf-v1',(int(context),int(resident_identity),surface))
        leaf=UtteranceLeafV1(ident,int(context),surface)
        store=self._leaves if persist else self._scratch
        prior=store.get(ident)
        if prior is not None and prior.surface!=surface:raise RuntimeError('organism:utterance_collision')
        store[ident]=leaf
        return leaf

    def lookup(self, identity):
        identity=int(identity)
        return self._leaves.get(identity) or self._scratch.get(identity)

    def retire(self, identity):
        identity=int(identity)
        return self._leaves.pop(identity,None) is not None or self._scratch.pop(identity,None) is not None

    def checkpoint(self):
        return [{'identity':leaf.identity,'context':leaf.context,'surface':list(leaf.surface)}
                for leaf in sorted(self._leaves.values(),key=lambda row:row.identity)]

    def restore(self, rows):
        self._leaves={}
        for row in rows or ():
            ident=int(row['identity']);surface=tuple(map(int,row.get('surface',())))
            if ident in self._leaves or not surface:raise ValueError('organism:utterance_checkpoint')
            self._leaves[ident]=UtteranceLeafV1(ident,int(row['context']),surface)


def apply_learned_discourse_surface(organism, scene, surface, current_closure):
    if not organism.partner_present or organism.partner_source<=0:return tuple(surface),(),current_closure,0
    partner=int(organism.partner_source)
    prior_ep=organism._episode_by_id.get(int(organism.last_shared_episode_by_partner.get(partner,0)))
    prior_closure=organism.utterances.lookup(int(organism.last_shared_closure_by_partner.get(partner,0)))
    if prior_ep is None or prior_closure is None:return tuple(surface),(),current_closure,0
    links=[l for l in organism._links_from.get(int(prior_ep.scene_identity),())
           if l.active and l.source not in organism.withdrawn_sources and l.right_scene==scene.identity]
    if len(links)!=1:return tuple(surface),(),current_closure,0
    template=organism._span_template_for_partner(int(links[0].relation),2)
    if template is None:return tuple(surface),(),current_closure,0
    try:
        combined_plan,combined_surface=rematerialize_transient_plan(
            organism.language,int(links[0].relation),(prior_closure,current_closure),template,completed_child=0)
    except HierarchicalRefuse:return tuple(surface),(),current_closure,0
    if not combined_surface:return tuple(surface),(),current_closure,0
    sid=int(template.identity[:15],16)
    contributors=(sid,combined_plan.identity,prior_closure.identity,current_closure.identity)
    return tuple(combined_surface),contributors,current_closure,sid


def current_expression_plan(organism, action):
    if action not in organism.actions or getattr(action,'settled',True):return None
    if not action.span_identity or not organism.partner_present or int(organism.partner_source)!=int(action.source):return None
    scene=organism._scene_by_id.get(int(action.scene_identity))
    if scene is None:return None
    partner=int(action.source)
    prior_ep=organism._episode_by_id.get(int(organism.last_shared_episode_by_partner.get(partner,0)))
    prior_closure=organism.utterances.lookup(int(organism.last_shared_closure_by_partner.get(partner,0)))
    current_closure=organism.utterances.lookup(int(action.closure_identity))
    if prior_ep is None or prior_closure is None or current_closure is None:return None
    links=[l for l in organism._links_from.get(int(prior_ep.scene_identity),())
           if l.active and l.source not in organism.withdrawn_sources and l.right_scene==scene.identity]
    if len(links)!=1:return None
    template=organism._span_template_for_partner(int(links[0].relation),2)
    if template is None or int(template.identity[:15],16)!=int(action.span_identity):return None
    try:
        plan,surface=rematerialize_transient_plan(
            organism.language,int(links[0].relation),(prior_closure,current_closure),template,completed_child=0)
    except HierarchicalRefuse:return None
    if (plan.identity not in action.contributors or tuple(surface)!=tuple(action.planned_payload)):
        return None
    return plan
