#!/usr/bin/env python3
from __future__ import annotations
from dataclasses import dataclass
from reference_hierarchical_composition_v1 import HierarchicalRefuse

CONTACT_SCENE=1
CONTACT_SURFACE=2
CONTACT_RELATION=3
CONTACT_DISCOURSE_SURFACE=4
CONTACT_UTTERANCE=5

@dataclass(frozen=True)
class SceneContactV1:
    identity:int
    context:int
    atoms:tuple[int,...]
    source:int

@dataclass(frozen=True)
class RelationContactV1:
    identity:int
    context:int
    scenes:tuple[int,...]
    source:int

class LanguageMasteryContactAdapterV1:
    """Opaque developmental membrane for the fast simulated Adult.

    It transports scene identities, raw bytes, and relation incidence only. It has
    no word/construction/grammar operation and no expected-answer field.
    """
    def __init__(self,adult):
        self.adult=adult;self.next_identity=1;self.current_scene=0;self.current_relation=0
        self.scenes={};self.nested_scenes={};self.relations={}
    def _id(self):
        x=self.next_identity;self.next_identity+=1;return x
    def _reply_binding_trace(self, payload, direct=(), max_depth=4, max_surfaces=64, max_bindings=64):
        """Recover flat bindings plus the learned structural roles actually traversed."""
        root=tuple(int(x) for x in payload);found=[];keys=set();visited=[0];refused=[False];used_roles=set()
        def keep(rows):
            for row in rows:
                if refused[0]:return
                key=(int(row.context),tuple(map(int,row.atoms)),str(row.template_identity),tuple(map(int,getattr(row,'lexical_identities',()))))
                if key in keys:continue
                if len(found)>=int(max_bindings):refused[0]=True;return
                keys.add(key);found.append(row)
        keep(tuple(direct))
        def visit(surface,depth,ancestors):
            if refused[0]:return
            surface=tuple(int(x) for x in surface);visited[0]+=1
            if visited[0]>int(max_surfaces):refused[0]=True;return
            if depth:keep(self.adult.language.invert_surface(surface,max_candidates=max_bindings))
            if refused[0]:return
            spans=self.adult.language.invert_span(surface,max_candidates=32)
            if spans and depth>=int(max_depth):refused[0]=True;return
            for span in spans:
                known_port=self.adult.language.span_reply_port(span.template_identity)
                learned_port=self.adult.structural_reply_port(span.template_identity)
                if known_port is not None and learned_port is None:
                    refused[0]=True;return
                if learned_port is not None:used_roles.add((span.template_identity,int(learned_port)))
                children=enumerate(span.children) if learned_port is None else ((learned_port,span.children[learned_port]),)
                for _port,child in children:
                    child=tuple(int(x) for x in child)
                    if not child or len(child)>=len(surface) or child in ancestors:refused[0]=True;return
                    visit(child,depth+1,ancestors|{child})
                    if refused[0]:return
        visit(root,0,{root})
        if refused[0]:return (),()
        return tuple(found),tuple(sorted(used_roles,key=lambda row:(str(row[0]),row[1])))

    def _reply_bindings(self, payload, direct=(), max_depth=4, max_surfaces=64, max_bindings=64):
        return self._reply_binding_trace(payload,direct,max_depth,max_surfaces,max_bindings)[0]

    def _resource_inhibits_known_span(self,payload,max_depth=4):
        """A temporarily inhibited learned structure must not downgrade to flat fallback."""
        root=tuple(map(int,payload));seen=set()
        def visit(surface,depth):
            surface=tuple(map(int,surface))
            if surface in seen or depth>int(max_depth):return False
            seen.add(surface)
            for span in self.adult.language.invert_span(surface,max_candidates=32):
                known=self.adult.language.span_reply_port(span.template_identity)
                current=self.adult.structural_reply_port(span.template_identity)
                if known is not None and current is None:return True
                children=(span.children if current is None else (span.children[current],))
                if any(visit(child,depth+1) for child in children):return True
            return False
        return visit(root,0)

    def _propose_reply_span_role(self,payload,source):
        pending=set(getattr(self.adult,'_pending_language_competition',()))
        if not pending:return ()
        proposed=[]
        for span in self.adult.language.invert_span(payload):
            if self.adult.language.span_reply_port(span.template_identity) is not None:continue
            matched=[]
            for port,child in enumerate(span.children):
                rows=self._reply_bindings(child,direct=self.adult.language.invert_surface(child),max_depth=3)
                hits={(int(r.context),tuple(map(int,r.atoms))) for r in rows}&pending
                if hits:matched.append((port,hits))
            if len(matched)==1 and len(matched[0][1])==1:proposed.append((span.template_identity,matched[0][0],int(source)))
        return proposed[0] if len(proposed)==1 else ()

    def contact(self,kind,payload,source,channel=0,body_credentials=()):
        kind=int(kind);source=int(source);payload=tuple(int(x) for x in payload)
        channel=max(0,int(channel))
        if kind==CONTACT_SCENE:
            if len(payload)<2:raise ValueError('contact:scene')
            identity=self._id();row=SceneContactV1(identity,int(payload[0]),tuple(payload[1:]),source)
            self.scenes[identity]=row;self.current_scene=identity;return identity
        if kind==CONTACT_SURFACE:
            scene=self.scenes.get(self.current_scene)
            if scene is None or not payload:raise ValueError('contact:surface_without_scene')
            if len(scene.atoms)==1:
                ordered=self.adult._observe_scene_surface(
                    scene.context,scene.atoms,payload,source)
                if len(ordered)>1:
                    self.scenes[scene.identity]=SceneContactV1(
                        scene.identity,scene.context,tuple(ordered),scene.source)
                    return True
                item=self.adult._observe_surface_item(scene.atoms[0],payload,source)
                construction=self.adult._observe_surface_construction(
                    scene.context,scene.atoms,payload,source)
                return bool(item or construction)
            ordered=self.adult._observe_scene_surface(
                scene.context,scene.atoms,payload,source)
            if ordered:
                self.scenes[scene.identity]=SceneContactV1(
                    scene.identity,scene.context,tuple(ordered),scene.source)
            return bool(ordered)
        if kind==CONTACT_RELATION:
            if len(payload)<3:raise ValueError('contact:relation')
            context,*scenes=payload;scenes=tuple(map(int,scenes))
            if len(set(scenes))!=len(scenes) or any(scene not in self.scenes for scene in scenes):raise ValueError('contact:relation_scene')
            identity=self._id();row=RelationContactV1(identity,context,scenes,source)
            self.relations[identity]=row;self.current_relation=identity;return identity
        if kind==CONTACT_DISCOURSE_SURFACE:
            relation=self.relations.get(self.current_relation)
            if relation is None or not payload:raise ValueError('contact:discourse_without_relation')
            scenes=tuple(self.scenes[identity] for identity in relation.scenes)
            children=tuple(self.adult.leaf(scene.context,scene.atoms) for scene in scenes)
            return self.adult._observe_relation_span(relation.context,children,payload,source)
        if kind==CONTACT_UTTERANCE:
            reply_open=self.adult._language_reply_open(channel)
            if not payload:
                self.adult._clear_current_occurrence(preserve_language_reply=reply_open)
                return 0
            bindings=self.adult.language.invert_surface(payload)
            # An actually expressed resident inquiry leaves a compact structural
            # commitment. The next raw contact can close it only by reconstructing
            # exactly one pending binding; the membrane names neither an answer nor
            # a referent. Nonparticipating contact cannot erase or settle it.
            if reply_open:
                settled=self.adult._settle_language_reply(bindings,channel)
                proposed_role=()
                if settled is None:
                    proposed_role=self._propose_reply_span_role(payload,source)
                    reply_bindings,used_roles=self._reply_binding_trace(payload)
                    settled=self.adult._settle_language_reply(reply_bindings,channel)
                    if settled is not None:
                        if proposed_role:self.adult._stage_span_reply_role(*proposed_role)
                        elif used_roles:self.adult._stage_span_reply_roles(used_roles,source)
                if settled is not None:
                    repair_context,(_binding_context,atoms)=settled
                    identity=self._id()
                    self.scenes[identity]=SceneContactV1(
                        identity,int(repair_context),tuple(map(int,atoms)),source)
                    self.current_scene=identity
                    return identity
            # A dependency alias inferred from one encounter is a hypothesis for
            # later consequence/source settlement, not yet an ordinary semantic
            # occurrence.  Otherwise unrelated novel utterances can collapse onto
            # the same missing-slot feature and displace a learned nested closure.
            # Reply settlement above may inspect provisional hypotheses; normal
            # perception below admits only independently acquired lexical rows.
            bindings=tuple(binding for binding in bindings if all(
                any(tuple(map(int,candidate_units))==tuple(map(int,units))
                    for _support,candidate_units,_sources in
                    self.adult.language.lexeme_candidates(int(feature)))
                for lexeme_identity in binding.lexical_identities
                for historical in (self.adult.language.historical_lexeme_binding(
                    int(lexeme_identity)),)
                if historical is not None
                for feature,units in (historical,)))
            # A new ordinary contact ends the prior transient situation. A learned
            # structure that is unavailable under current body/resource state is an
            # inhibition, not evidence that the structure never existed; do not bypass
            # it by falling through to unrestricted nested/flat reconstruction.
            self.adult._clear_current_occurrence(preserve_language_reply=reply_open)
            if self._resource_inhibits_known_span(payload):return 0
            if body_credentials:
                # Recognition must not freeze a learned construction. Re-lived
                # authenticated wording can still revise its world relation use.
                self.adult._observe_open_span_from_known_surfaces(
                    payload,source,body_credentials=body_credentials)
            if len(bindings)>1:
                # One bounded pending relation cannot be overwritten by a second
                # unresolved competition. It remains resident without turning the
                # body into an answer-waiting mode.
                if reply_open:return 0
                return self.adult._activate_language_competition(bindings,channel)
            if len(bindings)==1:
                binding=bindings[0]
                if not binding.atoms or int(binding.context)==0:return 0
                try:
                    self.adult._activate_language_occurrence(
                        binding.context,binding.atoms,preserve_language_reply=reply_open)
                except HierarchicalRefuse:
                    # Recognition does not guarantee that the current hierarchical
                    # resource closure can realize a public occurrence. Preserve its
                    # learned structural scene for causal participation, but keep it
                    # selection-silent instead of crashing or inventing a new leaf.
                    self.adult._clear_current_occurrence(
                        preserve_language_reply=reply_open)
                except RuntimeError as exc:
                    # A uniquely reconstructed one-word lexical occurrence is
                    # perceptible even when it is not itself a productive public
                    # proposition. Keep it transient and selection-silent so
                    # resident discourse can use its learned concept identity.
                    if str(exc)!='adult:construction_not_productive' or len(binding.atoms)!=1:
                        raise
                    self.adult._clear_current_occurrence(
                        preserve_language_reply=reply_open)
                self.adult._current_language_channel=channel
                identity=self._id()
                # The raw bytes are not retained: the learned language ecology has
                # already reduced them to one transient structural binding.
                self.scenes[identity]=SceneContactV1(
                    identity,int(binding.context),tuple(map(int,binding.atoms)),source)
                self.current_scene=identity
                return identity
            closure=self.adult._reconstruct_unique_nested_bindings(payload)
            if closure and not body_credentials:
                # Recognition is not settlement: an unresolved expressed reply must
                # remain open until learned structural-role evidence selects a path.
                if reply_open:return 0
                identity=self._id()
                self.nested_scenes[identity]=tuple(
                    SceneContactV1(identity,int(row.context),tuple(map(int,row.atoms)),source)
                    for row in closure)
                return identity
            # A uniquely completed learned dependency is a more specific account
            # of this raw contact than generic open-span acquisition.  Let that
            # resident competition run first; otherwise the known anchor can cause
            # the open-span path to swallow the unknown port before it is retained.
            inferred=self.adult.language.infer_provisional_dependency_alias(payload,source)
            if inferred is not None:
                feature,units,_context,_target_slot=inferred
                return self.adult.language.lexeme_identity(feature,units)
            # Authenticated transport changes occurrence authority, never truth. It
            # only lets the learned open span participate in the causal learner.
            if not reply_open and self.adult._observe_open_span_from_known_surfaces(
                    payload,source,body_credentials=body_credentials):return 0
            if closure:
                identity=self._id()
                self.nested_scenes[identity]=tuple(
                    SceneContactV1(identity,int(row.context),tuple(map(int,row.atoms)),source)
                    for row in closure)
                return identity
            return 0
        raise ValueError('contact:kind')

    def settle_provisional(self,identity,ticket,effect,independent=True):
        if not independent:return None
        handled,result=self.adult.language.settle_lexeme_identity(
            int(identity),None,int(ticket),int(effect))
        return result if handled else None
