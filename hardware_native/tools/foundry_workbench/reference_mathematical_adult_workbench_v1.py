#!/usr/bin/env python3
"""One heterogeneous fast Workbench Adult: language/cognition + operator factor state."""
from __future__ import annotations
from dataclasses import dataclass
import copy,hashlib,json
from reference_language_mastery_adult_v1 import LanguageMasteryAdultV1
from reference_language_mastery_terminal_v1 import emit_choice
from reference_language_learning_v1 import PIECE_LITERAL,PIECE_PORT
from reference_hierarchical_composition_v1 import _identity as _language_identity
from reference_predictive_credit_profile_v1 import Q
from reference_open_language_action_affordance_v1 import OpenLanguageActionAffordanceV1
from reference_recursive_relation_basis_v1 import RecursiveRelationBasisV1
from reference_resident_variable_depth_endogenous_unfolding_v1 import (
    EndogenousUnfolderV1,PublicActionReceiptV1,ResidentProgramStateV1,_digest,_identity,
)

MIN_BINDING_SOURCES=2
MAX_CAUSAL_DIALOGUE_UPTAKE_ROWS=256
MAX_CAUSAL_DIALOGUE_UPTAKE_EVIDENCE=8
MAX_SETTLED_CAUSAL_ACTION_LINEAGE_ROWS=64

@dataclass(frozen=True)
class RelationBasisPublicReceiptV1:
    relation_identity:int
    ancestry_digest:str
    basis_revision:int
    identity:int

@dataclass(frozen=True)
class CausalDialogueActionReceiptV1:
    programs:tuple[int,...]
    factors:tuple[int,...]
    context:int
    surface_digest:str
    source:int
    channel:int
    episode:int
    born_tick:int
    identity:int

@dataclass(frozen=True)
class CausalContinuationCommitmentV1:
    source:int
    channel:int
    support_context:int
    support_arity:int
    support_template:int
    focus:int
    resolved_receipt:int
    awaiting_action:int
    ready:bool

@dataclass(frozen=True)
class EndogenousInquiryActionReceiptV1:
    context:int
    surface_digest:str
    source:int
    channel:int
    born_tick:int
    identity:int
    obligation_candidates:tuple=()
    obligation_effect:int=0

@dataclass(frozen=True)
class ContextAffordanceActionReceiptV1:
    prompt_space:int
    target_context:int
    surface_digest:str
    source:int
    channel:int
    born_tick:int
    identity:int

class OperatorSurfaceExpressionV1:
    """Transient byte cursor; public action authority exists only after exact reafference."""
    def __init__(self,owner,winner,surface):
        self._owner=owner;self._winner=winner;self._surface=tuple(map(int,surface));self._ordinal=0;self._pending=None;self.receipt=None;self.repairs=0
    def emit(self):
        if self.receipt is not None or self._ordinal>=len(self._surface):return None
        if self._pending is None:self._pending=(int(self._surface[self._ordinal]),int(self._ordinal))
        return self._pending
    def reafference(self,step,observed):
        if self.receipt is not None or self._pending is None or tuple(step)!=tuple(self._pending):return False
        value,ordinal=self._pending
        if int(observed)!=int(value):self.repairs+=1;return False
        self._ordinal=int(ordinal)+1;self._pending=None
        if self._ordinal==len(self._surface):self.receipt=self._owner._commit_operator_public_action(self._winner)
        return True

class RelationBasisSurfaceExpressionV1:
    """Transient realization of one exact relation closure; authority follows exact reafference."""
    def __init__(self,owner,relation_identity,surface,frontier_order=()):
        self._owner=owner;self._relation_identity=int(relation_identity);self._frontier_order=tuple(map(int,frontier_order));self._surface=tuple(map(int,surface));self._ordinal=0;self._pending=None;self.receipt=None;self.repairs=0
    def emit(self):
        if self.receipt is not None or self._ordinal>=len(self._surface):return None
        if self._pending is None:self._pending=(int(self._surface[self._ordinal]),int(self._ordinal))
        return self._pending
    def reafference(self,step,observed):
        if self.receipt is not None or self._pending is None or tuple(step)!=tuple(self._pending):return False
        value,ordinal=self._pending
        if int(observed)!=int(value):self.repairs+=1;return False
        self._ordinal=int(ordinal)+1;self._pending=None
        if self._ordinal==len(self._surface):self.receipt=self._owner._commit_relation_frontier_public_action(self._relation_identity,self._frontier_order)
        return True

class MathematicalWorkbenchAdultV1:
    """Single checkpointed subject; active operator closure is always rematerialized."""
    def __init__(self,language_adult=None,operators=None,relation_basis=None):
        self.language_adult=language_adult if language_adult is not None else LanguageMasteryAdultV1()
        self.operators=operators if operators is not None else ResidentProgramStateV1()
        self.relation_basis=relation_basis if relation_basis is not None else RecursiveRelationBasisV1()
        # Same-individual lifetime language->action affordance state. Raw speech remains
        # transient pending occurrence; durable factors are earned only from later
        # same-source observed action and carry no motor authority by themselves.
        self.language_action_affordances=OpenLanguageActionAffordanceV1()
        self._operator_bindings={};self._operator_binding_withdrawn=set();self._operator_join_context=0;self._operator_join_sources=set();self._relation_language_spaces={};self._relation_edge_language_contexts={}
        self.operator_externalization_enabled=True;self.pending_operator_actions={};self.operator_public_count=0;self.pending_relation_actions={};self.relation_public_count=0
        self.pending_causal_dialogue_actions={};self.recent_causal_dialogue_actions={};self.pending_causal_dialogue_consequences=set();self.causal_dialogue_public_count=0
        self._causal_dialogue_context_by_channel={}
        self._causal_dialogue_uptake_evidence={}
        self._causal_dialogue_dispute_evidence={}
        self._causal_dialogue_formulation_evidence={}
        self._settled_causal_action_lineage={};self._settled_causal_action_lineage_index={}
        self._causal_continuation_commitments={}
        self.pending_endogenous_inquiry_actions={};self.reafferenced_endogenous_inquiry_actions=set();self.endogenous_inquiry_public_count=0
        self.pending_context_affordance_actions={};self.context_affordance_public_count=0
        self._context_affordance_sources={}
        self.last_causal_dialogue_lineage_touches=0
        self.last_causal_dialogue_contact_continuations=()
        self.last_causal_dialogue_continuation_support=()
        self.last_causal_dialogue_embedded_relation_touches=0
        self.last_causal_continuation_frontier_touches=0
        self.last_endogenous_inquiry_touches=0
        self.last_causal_repair_inquiry_touches=0
        self.last_relation_language_touches=0
        # Disposable multi-surface ingress matcher. It indexes only surfaces already
        # licensed by resident relation-space/language state; it has no truth or action
        # authority and is deliberately absent from checkpoint state.
        self._relation_surface_match_epoch=None;self._relation_surface_match_goto=();self._relation_surface_match_fail=();self._relation_surface_match_out=()
        self.last_relation_ingress_touches=0;self.last_relation_ingress_index_surfaces=0;self.last_relation_ingress_index_bytes=0
        self.reset_operator_transient()
    @property
    def language(self):return self.language_adult.language
    def reset_operator_transient(self):self.operator_unfolder=EndogenousUnfolderV1(self.operators)
    def operator_silent_wave(self):return self.operator_unfolder.silent_wave()
    def resident_silent_wave(self):
        self.operator_unfolder.silent_wave();return self.relation_basis.silent_wave()
    def ambient_language_capacity_q16(self):
        return int(self.language_adult.ambient_language_capacity_q16())
    def observe_ambient_language_contact(self,raw,source):
        return bool(self.language_adult.observe_ambient_language_contact(raw,source))
    def operator_run_until_settled(self,max_waves=256):return self.operator_unfolder.run_until_settled(max_waves)
    def operator_winner(self):return self.operator_unfolder.winner()
    def observe_operator_binding(self,node_identity,context,atoms,source):
        node_identity=int(node_identity);context=int(context);atoms=tuple(map(int,atoms));source=int(source)
        if node_identity<=0 or context<=0 or not atoms or source<=0:return False
        self._operator_bindings.setdefault(node_identity,{}).setdefault((context,atoms),set()).add(source);return True
    def observe_operator_join_context(self,context,source):
        context=int(context);source=int(source)
        if context<=0 or source<=0 or self.language.span_template(context,2) is None:return False
        if self._operator_join_context not in (0,context):return False
        self._operator_join_context=context;self._operator_join_sources.add(source);return True
    def withdraw_operator_binding_source(self,source):self._operator_binding_withdrawn.add(int(source))
    def restore_operator_binding_source(self,source):self._operator_binding_withdrawn.discard(int(source))
    def _binding(self,node_identity):
        rows=self._operator_bindings.get(int(node_identity),{})
        live=[]
        for row,sources in rows.items():
            if sum(1 for source in sources if source not in self._operator_binding_withdrawn)>=MIN_BINDING_SOURCES:live.append(row)
        return live[0] if len(live)==1 else None
    def _join_live(self):
        return self._operator_join_context if sum(1 for source in self._operator_join_sources if source not in self._operator_binding_withdrawn)>=MIN_BINDING_SOURCES else 0
    def project_operator_trace(self,trace):
        clauses=[]
        for node_identity in tuple(map(int,trace)):
            row=self._binding(node_identity)
            if row is None:return None
            context,atoms=row
            try:leaf=self.language_adult.leaf(context,atoms)
            except RuntimeError:return None
            clauses.append(tuple(leaf.surface))
        if not clauses:return None
        join=self._join_live()
        if join<=0:return None
        out=clauses[0]
        for clause in clauses[1:]:
            out=self.language.realize_span(join,(out,clause))
            if out is None:return None
        return tuple(out)
    def observe_relation_language_space(self,context,atoms):
        context=int(context);atoms=tuple(map(int,atoms))
        if context<=0 or not atoms:return 0
        # Constructibility is current language evidence; the stable relation-space
        # identity itself contains no surface bytes.
        try:self.language_adult.leaf(context,atoms)
        except RuntimeError:return 0
        sid=_identity('workbench-relation-language-space-v1',(context,atoms))
        prior=self._relation_language_spaces.get(sid)
        if prior is not None and prior!=(context,atoms):raise RuntimeError('mathematical-workbench-adult:relation-space-collision')
        self._relation_language_spaces[sid]=(context,atoms);return sid
    def relation_space_surface(self,space_identity):
        binding=self._relation_language_spaces.get(int(space_identity))
        if binding is None:
            # Compatibility for non-language relation bases used by focused assays.
            if self.language_adult._has_leaf(int(space_identity)):
                return tuple(self.language_adult._leaf_surface(int(space_identity)))
            return None
        try:return tuple(self.language_adult.leaf(binding[0],binding[1]).surface)
        except RuntimeError:return None
    def _relation_surface_matcher(self):
        """Rematerialize a disposable all-occurrence matcher for current relation spaces."""
        epoch=(int(self.relation_basis.revision),int(self.language._support_epoch))
        if self._relation_surface_match_epoch==epoch:return self._relation_surface_match_goto,self._relation_surface_match_fail,self._relation_surface_match_out
        spaces=tuple(sorted(set(
            int(space) for row in self.relation_basis.relations.values()
            for space in (row.left_space,row.right_space))))
        goto=[{}];fail=[0];out=[[]];surface_count=0;surface_bytes=0
        for identity in spaces:
            surface=tuple(self.relation_space_surface(identity) or ())
            if not surface:continue
            surface_count+=1;surface_bytes+=len(surface);state=0
            for value in surface:
                value=int(value)&255
                nxt=goto[state].get(value)
                if nxt is None:
                    nxt=len(goto);goto[state][value]=nxt;goto.append({});fail.append(0);out.append([])
                state=nxt
            out[state].append((int(identity),len(surface)))
        queue=list(goto[0].values());head=0
        while head<len(queue):
            state=queue[head];head+=1
            for value,nxt in goto[state].items():
                queue.append(nxt);fallback=fail[state]
                while fallback and value not in goto[fallback]:fallback=fail[fallback]
                fail[nxt]=goto[fallback].get(value,0)
                if out[fail[nxt]]:out[nxt].extend(out[fail[nxt]])
        self._relation_surface_match_epoch=epoch;self._relation_surface_match_goto=tuple(goto);self._relation_surface_match_fail=tuple(fail);self._relation_surface_match_out=tuple(tuple(rows) for rows in out)
        self.last_relation_ingress_index_surfaces=surface_count;self.last_relation_ingress_index_bytes=surface_bytes
        return self._relation_surface_match_goto,self._relation_surface_match_fail,self._relation_surface_match_out
    def relation_surface_matches(self,raw):
        """Return every current relation-space surface occurrence in one raw-byte pass."""
        raw=bytes(raw);goto,fail,out=self._relation_surface_matcher();state=0;hits=[];self.last_relation_ingress_touches=0
        for end,value in enumerate(raw):
            self.last_relation_ingress_touches+=1
            while state and value not in goto[state]:state=fail[state]
            state=goto[state].get(value,0)
            for identity,width in out[state]:hits.append((end-int(width)+1,end+1,int(identity)))
        return tuple(hits)
    def observe_relation_basis_edge(self,left_space,right_space,boundary_q16,source,language_context=0):
        source=int(source);language_context=int(language_context)
        rid=self.relation_basis.observe_primitive(int(left_space),int(right_space),tuple(map(int,boundary_q16)),source)
        if language_context>0:
            self._relation_edge_language_contexts.setdefault(int(rid),{}).setdefault(language_context,set()).add(source)
        return rid
    def _relation_language_context(self,left_space,right_space):
        candidates=[];withdrawn=self.language._withdrawn;work=max(1,int(self.operators.work_quanta))
        rows=self.relation_basis.active_relations_for_pair(left_space,right_space);self.last_relation_language_touches+=len(rows)
        for rid in rows:
            for context,sources in self._relation_edge_language_contexts.get(int(rid),{}).items():
                live_edge=sum(1 for source in sources if source not in withdrawn)
                template=self.language.span_template(int(context),2)
                if live_edge<MIN_BINDING_SOURCES or template is None:continue
                template_support=int(template.support)
                literal_cost=sum(len(piece.literal) for piece in template.pieces if piece.literal)
                # Relation history and language competence are distinct causes. Do not
                # collapse the stronger one with min(): independently repeated edge
                # evidence must be able to outweigh a cheaper but weaker realization.
                score=live_edge*work*2+template_support*work-literal_cost
                candidates.append((score,live_edge,template_support,-literal_cost,int(context)))
        if not candidates:return 0
        peak=max(row[0] for row in candidates);winners=sorted({row[4] for row in candidates if row[0]==peak})
        return winners[0] if len(winners)==1 else 0
    def relation_language_contexts_for_order(self,order):
        order=tuple(map(int,order))
        return tuple(self._relation_language_context(left,right) for left,right in zip(order,order[1:]))
    def _relation_edge_has_language_history(self,left_space,right_space):
        rows=self.relation_basis.active_relations_for_pair(left_space,right_space);self.last_relation_language_touches+=len(rows)
        return any(self._relation_edge_language_contexts.get(int(rid)) for rid in rows)
    def _public_relation_order(self,order):
        """Resource pressure bounds external elaboration without deleting internal competence."""
        order=tuple(map(int,order))
        if len(order)<2:return ()
        max_clauses=max(2,int(self.operators.work_quanta)*3)
        return order[:min(len(order),max_clauses)]
    def _transient_realize_span(self,context,children):
        """Render learned span pieces into disposable public computation beyond learned-surface caps."""
        children=tuple(tuple(map(int,child)) for child in children);template=self.language.span_template(int(context),len(children))
        if template is None:return None
        try:return self._render_embedded_pieces(template.pieces,children)
        except Exception:return None
    def _render_embedded_pieces(self,pieces,children):
        pieces=tuple(pieces);children=tuple(tuple(map(int,child)) for child in children)
        embedded=self.language_adult._embedded_port_surfaces(pieces,children)
        return tuple(self.language_adult._render_pieces(pieces,embedded))
    def _realize_relation_order(self,order):
        order=tuple(map(int,order));self.last_relation_language_touches=0
        if len(order)<2:return None
        clauses=[]
        for identity in order:
            surface=self.relation_space_surface(identity)
            if not surface:return None
            clauses.append(tuple(surface))
        fallback=self._join_live()
        if fallback<=0:return None
        contexts=self.relation_language_contexts_for_order(order)
        out=clauses[0]
        for index,clause in enumerate(clauses[1:]):
            context=int(contexts[index])
            if context<=0:
                # A relation with no linguistic history may use the old neutral join.
                # A relation with competing learned histories must not hide ambiguity
                # behind that fallback: refusal is the honest public phenotype.
                if self._relation_edge_has_language_history(order[index],order[index+1]):return None
                context=fallback
            out=self._transient_realize_span(context,(out,clause))
            if out is None:return None
        return tuple(out)
    def project_relation_basis(self,relation_identity):
        spaces=self.relation_basis.expand_spaces(int(relation_identity))
        if not spaces or len(spaces)<2:return None
        return self._realize_relation_order(spaces)
    def _commit_relation_basis_public_action(self,relation_identity):
        relation_identity=int(relation_identity)
        if relation_identity not in self.relation_basis.active:return None
        row=self.relation_basis.relations.get(relation_identity)
        if row is None:return None
        ancestry=_digest('recursive-basis-public-ancestry-v1',(relation_identity,row.root_evidence,row.support_paths))
        rid=_identity('recursive-basis-public-receipt-v1',(relation_identity,ancestry,self.relation_basis.revision))
        receipt=RelationBasisPublicReceiptV1(relation_identity,ancestry,self.relation_basis.revision,rid)
        self.pending_relation_actions[rid]=receipt;self.relation_public_count+=1
        return receipt
    def _commit_relation_frontier_public_action(self,relation_identity,frontier_order=()):
        relation_identity=int(relation_identity);frontier_order=tuple(map(int,frontier_order))
        if relation_identity>0:return self._commit_relation_basis_public_action(relation_identity)
        if len(frontier_order)<2:return None
        exact_order,rid=self.relation_basis_order_for_frontier(frontier_order)
        if rid or tuple(exact_order)!=frontier_order:return None
        paths=[];roots=set()
        for left,right in zip(frontier_order,frontier_order[1:]):
            closure=self.relation_basis.resolve(left,right)
            if closure is None or not closure.paths:return None
            path=tuple(closure.paths[0]);paths.append(path)
            for child in path:
                row=self.relation_basis.relations.get(int(child))
                if row is None:return None
                roots.update(map(int,row.root_evidence))
        ancestry=_digest('recursive-basis-frontier-public-ancestry-v1',(frontier_order,tuple(paths),tuple(sorted(roots))))
        receipt_identity=_identity('recursive-basis-frontier-public-receipt-v1',(frontier_order,ancestry,self.relation_basis.revision))
        receipt=RelationBasisPublicReceiptV1(0,ancestry,self.relation_basis.revision,receipt_identity)
        self.pending_relation_actions[receipt_identity]=receipt;self.relation_public_count+=1
        return receipt
    def relation_basis_order_for_frontier(self,identities):
        """Rematerialize one exact ordering from retained abstraction or the live lower graph."""
        matter=tuple(map(int,identities));target=set(matter)
        if len(matter)<2 or len(target)!=len(matter) or any(x<=0 for x in matter):return (),0
        rows=[]
        for rid in sorted(self.relation_basis.active):
            relation=self.relation_basis.relations.get(int(rid))
            if relation is None:continue
            spaces=self.relation_basis.expand_spaces(int(rid))
            if spaces and len(spaces)==len(matter) and set(spaces)==target:
                rows.append((len(spaces),self.relation_basis.generation(rid),int(rid),tuple(spaces)))
        if rows:
            peak=max((n,g) for n,g,_rid,_spaces in rows);w=[row for row in rows if row[:2]==peak]
            if len(w)==1:return w[0][3],w[0][2]
        # Deoptimized path: infer unique endpoints only from active relations whose
        # terminal endpoints lie inside current matter, then ask the exact basis.
        edges=[]
        for rid in sorted(self.relation_basis.active):
            row=self.relation_basis.relations[int(rid)]
            if row.left_space in target and row.right_space in target:edges.append(row)
        lefts={row.left_space for row in edges};rights={row.right_space for row in edges}
        starts=lefts-rights;ends=rights-lefts
        if len(starts)!=1 or len(ends)!=1:return (),0
        closure=self.relation_basis.resolve(next(iter(starts)),next(iter(ends)))
        if closure is None or not closure.paths:return (),0
        orders=[]
        for path in closure.paths:
            out=[]
            for rid in path:
                seq=self.relation_basis.expand_spaces(rid)
                if not seq:out=[];break
                if out and out[-1]==seq[0]:out.extend(seq[1:])
                elif not out:out.extend(seq)
                else:out=[];break
            if len(out)==len(matter) and set(out)==target and tuple(out) not in orders:orders.append(tuple(out))
        return (orders[0],0) if len(orders)==1 else ((),0)
    def relation_basis_for_frontier(self,identities):
        _order,rid=self.relation_basis_order_for_frontier(identities);return int(rid)
    def organize_relation_frontier(self,identities):
        order,_rid=self.relation_basis_order_for_frontier(identities)
        if not order:return None
        return self._realize_relation_order(order)
    def relation_basis_expression_for_frontier(self,identities):
        order,rid=self.relation_basis_order_for_frontier(identities)
        if not order:return None
        public_order=self._public_relation_order(order)
        if not public_order:return None
        # A resource-truncated public action is a lower exact frontier, not a claim
        # that the complete retained relation was outwardly expressed.
        public_rid=rid if tuple(public_order)==tuple(order) else 0
        surface=self._realize_relation_order(public_order)
        if surface is None:return None
        return RelationBasisSurfaceExpressionV1(self,public_rid,surface,() if public_rid>0 else public_order)
    def relation_basis_expression(self):
        relation_identity=int(self.relation_basis.best_derived())
        if relation_identity<=0:return None
        surface=self.project_relation_basis(relation_identity)
        return None if surface is None else RelationBasisSurfaceExpressionV1(self,relation_identity,surface)
    def causal_chain_rows(self,leaf_identity):
        learner=self.language_adult.world_causal_learning
        for receipt in tuple(learner.bindings):
            if learner.complete_source_blocks(receipt)>=3:learner.resolve(receipt)
        rows=tuple(learner.current_resolutions());target=int(leaf_identity)
        incident={};outgoing={};incoming={}
        for row in rows:
            cause,effect=int(row[2]),int(row[3])
            incident.setdefault(cause,[]).append(row);incident.setdefault(effect,[]).append(row)
            outgoing.setdefault(cause,[]).append(row);incoming.setdefault(effect,[]).append(row)
        if target not in incident:return ()
        nodes={target};pending=[target]
        while pending:
            node=pending.pop()
            for row in incident.get(node,()):
                for neighbour in (int(row[2]),int(row[3])):
                    if neighbour not in nodes:nodes.add(neighbour);pending.append(neighbour)
        roots=tuple(node for node in nodes if not incoming.get(node))
        if not roots:return ()
        depth={};visiting=set()
        def remaining(node):
            if node in depth:return depth[node]
            if node in visiting:raise ValueError('causal-component-cycle')
            visiting.add(node);children=tuple(outgoing.get(node,()))
            value=0 if not children else 1+max(remaining(int(row[3])) for row in children)
            visiting.remove(node);depth[node]=value;return value
        try:
            for node in nodes:remaining(node)
        except ValueError:return ()
        # Rematerialize a hierarchy-sensitive motor frontier. Parents precede
        # effects, while consequences of the same cause remain adjacent so an
        # independently learned coordination factor can bind them. This is a
        # disposable breadth frontier, not a stored discourse tree or stage list.
        ordered=[];seen_edges=set();expanded=set()
        pending=list(sorted(roots,key=lambda node:(-depth[node],node)))
        while pending:
            node=int(pending.pop(0))
            if node in expanded:continue
            expanded.add(node)
            children=sorted(outgoing.get(node,()),key=lambda row:(-depth[int(row[3])],int(row[5]),int(row[4])))
            for row in children:
                edge=(int(row[2]),int(row[3]),int(row[4]))
                if edge in seen_edges:continue
                seen_edges.add(edge);ordered.append(row);pending.append(int(row[3]))
        component_rows=sum(1 for row in rows if int(row[2]) in nodes and int(row[3]) in nodes)
        return tuple(ordered) if len(ordered)==component_rows else ()
    def causal_focus_rows(self,leaf_identity):
        """Transient directed ancestry/descendency projection for one lived focus."""
        rows=self.causal_chain_rows(leaf_identity);target=int(leaf_identity)
        if not rows:return ()
        incoming={};outgoing={}
        for row in rows:
            incoming.setdefault(int(row[3]),[]).append(row)
            outgoing.setdefault(int(row[2]),[]).append(row)
        selected=set()
        pending=[target];seen={target}
        while pending:
            node=pending.pop()
            for row in incoming.get(node,()):
                selected.add(int(row[4]));cause=int(row[2])
                if cause not in seen:seen.add(cause);pending.append(cause)
        pending=[target];seen={target}
        while pending:
            node=pending.pop()
            for row in outgoing.get(node,()):
                selected.add(int(row[4]));effect=int(row[3])
                if effect not in seen:seen.add(effect);pending.append(effect)
        return tuple(row for row in rows if int(row[4]) in selected)
    def causal_message_rows(self,leaf_identity):
        """Rematerialize the current occurrence's causal message boundary.

        The learned component may be arbitrarily deep.  Public preparation does
        not dump that component or walk an authored number of levels: the current
        occurrence recruits its direct causes and direct consequences.  A later
        contact can move the occurrence and rematerialize the next boundary.
        """
        target=int(leaf_identity)
        return tuple(row for row in self.causal_focus_rows(target)
                     if target in (int(row[2]),int(row[3])))
    def causal_program_for_row(self,row,factor=0,materialize=True):
        learner=self.language_adult.world_causal_learning;receipt=int(row[4]);factor=int(factor) or int(learner.preferred_factor(self.language_adult))
        if factor<=0:return 0
        cause,effect=int(row[2]),int(row[3]);orientation=int(learner.grounding.orientation(factor));current_cause=self.language_adult.current_leaf_for_historical(cause);current_effect=self.language_adult.current_leaf_for_historical(effect)
        if current_cause is None or current_effect is None or not orientation:return 0
        children=((current_effect,current_cause) if orientation>0 else (current_cause,current_effect));pid=int(self.language_adult.programs.ident(tuple(int(child.identity) for child in children),factor))
        if self.language_adult.programs.factor(pid)!=factor and materialize:
            program=learner.materialize_program(self.language_adult,receipt,factor)
            if program is None:return 0
            pid=int(program.identity)
        if not materialize:
            resolved=learner.resolve(receipt)
            if resolved is None or learner.complete_source_blocks(receipt)<3:return 0
            return pid if learner.language_relation_certificate(self.language_adult,factor,*tuple(int(child.identity) for child in children))[:1]==(1,) else 0
        chunk=self.language_adult.programs.chunks.get(pid);members=tuple(map(int,chunk.members)) if chunk is not None else ()
        return pid if len(members)==2 and learner.language_relation_certificate(self.language_adult,factor,*members)[:1]==(1,) else 0
    def _causal_self_contained_factors(self):
        learner=self.language_adult.world_causal_learning;rows=[]
        for factor,sources in learner.grounding.rows.items():
            pieces=self.language.historical_span_pieces(int(factor));literals=tuple(tuple(piece.literal) for piece in pieces if piece.kind==PIECE_LITERAL);ports=tuple(int(piece.port) for piece in pieces if piece.kind==PIECE_PORT)
            live=sum(1 for source in sources if source not in learner.grounding.withdrawn)
            terminals=sum(value in (ord('.'),ord('?'),ord('!')) for literal in literals for value in literal)
            if not learner.grounding.orientation(int(factor)) or live<2 or not pieces or not literals or not literals[-1] or literals[-1][-1] not in (ord('.'),ord('?'),ord('!')) or terminals!=1 or sorted(ports)!=[0,1]:continue
            rows.append((-live,-sum(map(len,literals)),int(factor)))
        return tuple(factor for _support,_literal,factor in sorted(rows))
    def _causal_opening_factors(self):
        return tuple(factor for factor in self._causal_self_contained_factors()
                     if (self.language.historical_span_pieces(int(factor))
                         and self.language.historical_span_pieces(int(factor))[0].kind==PIECE_LITERAL))
    def _causal_formulation_candidates(self,row,factors,context,channel=0):
        """Compare executable certified forms only after their own control history."""
        ranked=[];credit=self.language_adult.credit;context=int(context)
        for order,factor in enumerate(factors):
            program=int(self.causal_program_for_row(row,int(factor),False))
            if program<=0:continue
            profile=credit.rows.get(program);local=None if profile is None else profile.contexts.get(context)
            supported=bool(local is not None and local.control_supported)
            value=int(credit.contextual_causal_value(program,context)) if supported else 0
            structural=self.causal_dialogue_formulation_support(channel,factor)
            ranked.append((-structural,-value,int(order),int(factor),program))
        return tuple((factor,program) for _structural,_value,_order,factor,program in sorted(ranked))
    def _causal_continuation_factors(self):
        learner=self.language_adult.world_causal_learning;rows=[]
        for factor,sources in learner.grounding.rows.items():
            if learner.grounding.orientation(int(factor))!=-1:continue
            pieces=self.language.historical_span_pieces(int(factor));tail=tuple(pieces[-1].literal) if pieces and pieces[-1].kind==PIECE_LITERAL else ()
            live=sum(1 for source in sources if source not in learner.grounding.withdrawn)
            if not pieces or pieces[0].kind!=PIECE_PORT or int(pieces[0].port)!=0 or live<2 or not tail or tail[-1] not in (ord('.'),ord('?'),ord('!')):continue
            rows.append((-live,-sum(len(piece.literal) for piece in pieces if piece.kind==PIECE_LITERAL),int(factor)))
        return tuple(factor for _support,_literal,factor in sorted(rows))
    @staticmethod
    def _causal_motor_siblings(planned,following_cause=0):
        """Move only the uniquely continuing effect to the motor right edge."""
        planned=tuple(planned);following_cause=int(following_cause)
        matches=tuple(index for index,item in enumerate(planned)
                      if int(item[0][3])==following_cause)
        if len(planned)<2 or len(matches)!=1 or matches[0]==len(planned)-1:return planned
        index=matches[0]
        return (*planned[:index],*planned[index+1:],planned[index])
    def _causal_sibling_surface(self,planned,factor,outer_factor=0,continuation=False):
        """Rematerialize one shared-parent closure from exact certified rows."""
        planned=tuple(planned);factor=int(factor);outer_factor=int(outer_factor)
        if len(planned)<2:return None
        causes={int(item[0][2]) for item in planned}
        if len(causes)!=1 or any(self.causal_program_for_row(item[0],item[2],False)!=int(item[1]) for item in planned):return None
        effects=tuple(int(item[0][3]) for item in planned)
        exact=self.language_adult.common_cause_span_expression(*effects)
        certificates=tuple(self.language_adult.common_cause_span_expression(
            effects[0],effect) for effect in effects[1:])
        if exact:
            if int(exact[0])!=factor:return None
        elif not all(certificates) or any(int(row[0])!=factor for row in certificates):return None
        pieces=self.language.historical_span_pieces(factor)
        if not pieces:return None
        effect_surfaces=tuple(tuple(self.language_adult._leaf_surface(effect)) for effect in effects)
        try:
            if exact:
                coordinated=self._render_embedded_pieces(pieces,effect_surfaces)
            else:
                coordinated=effect_surfaces[0]
                for effect in effect_surfaces[1:]:coordinated=self._render_embedded_pieces(pieces,(coordinated,effect))
            outer_factor=outer_factor or int(planned[0][2]);outer=self.language.historical_span_pieces(outer_factor);orientation=int(self.language_adult.world_causal_learning.grounding.orientation(outer_factor));cause_identity=next(iter(causes));cause=tuple(self.language_adult._leaf_surface(cause_identity))
            certified=tuple(((effect,cause_identity) if orientation>0 else (cause_identity,effect)) for effect in effects)
            if (not outer or not orientation or any(
                    self.language_adult.world_causal_learning.language_relation_certificate(
                        self.language_adult,outer_factor,*children)[:1]!=(1,)
                    for children in certified)):return None
            children=((coordinated,cause) if orientation>0 else (cause,coordinated))
            if continuation:
                if outer[0].kind!=PIECE_PORT or int(outer[0].port)!=0:return None
                outer=tuple(outer[1:])
            return self._render_embedded_pieces(outer,children)
        except Exception:return None
    def _causal_continuation_suffix(self,row,factor):
        learner=self.language_adult.world_causal_learning;cause,effect=int(row[2]),int(row[3])
        if learner.language_relation_certificate(self.language_adult,int(factor),cause,effect)[:1]!=(1,):return None
        pieces=self.language.historical_span_pieces(int(factor))
        if not pieces or pieces[0].kind!=PIECE_PORT or int(pieces[0].port)!=0:return None
        try:return self._render_embedded_pieces(tuple(pieces[1:]),(tuple(self.language_adult._leaf_surface(cause)),tuple(self.language_adult._leaf_surface(effect))))
        except Exception:return None
    @staticmethod
    def _cross_action_continuation_fragment(surface):
        """Consume only the motor boundary already completed by the prior action."""
        surface=tuple(map(int,surface));index=0
        if surface and surface[0] in (ord('.'),ord('?'),ord('!')):
            index=1
            while index<len(surface) and chr(surface[index]).isspace():index+=1
        return surface[index:]
    def _causal_self_contained_surface(self,row,factor):
        learner=self.language_adult.world_causal_learning;cause,effect=int(row[2]),int(row[3]);orientation=int(learner.grounding.orientation(int(factor)))
        children=((effect,cause) if orientation>0 else (cause,effect))
        if learner.language_relation_certificate(self.language_adult,int(factor),*children)[:1]!=(1,) or not orientation:return None
        pieces=self.language.historical_span_pieces(int(factor))
        if not pieces:return None
        try:return self._render_embedded_pieces(pieces,tuple(tuple(self.language_adult._leaf_surface(child)) for child in children))
        except Exception:return None
    @staticmethod
    def _causal_externalization_limit(felt,width):
        width=max(0,int(width))
        if width<=0:return 0
        if felt.interference_q16>=Q//2 or felt.arousal_q16>=3*Q//4 or (felt.valence_q16<=-Q//4 and felt.controllability_q16<Q//2):return 1
        if felt.interference_q16>=Q//4:return min(2,width)
        return width
    def causal_dialogue_uptake_support(self,channel,causal_receipt):
        channel=max(0,int(channel));causal_receipt=int(causal_receipt)
        if channel<=0 or causal_receipt<=0:return 0
        return sum(max(0,int(evidence)) for (row_channel,row_receipt,source),evidence in self._causal_dialogue_uptake_evidence.items()
                   if row_channel==channel and row_receipt==causal_receipt and source not in self.language._withdrawn)
    def causal_dialogue_dispute_support(self,channel,causal_receipt):
        channel=max(0,int(channel));causal_receipt=int(causal_receipt)
        if channel<=0 or causal_receipt<=0:return 0
        return sum(max(0,int(evidence)) for (row_channel,row_receipt,_emitted,_candidate,_effect,source),evidence in self._causal_dialogue_dispute_evidence.items()
                   if row_channel==channel and row_receipt==causal_receipt and source not in self.language._withdrawn)
    def causal_dialogue_formulation_support(self,channel,factor):
        channel=max(0,int(channel));factor=int(factor)
        if channel<=0 or factor<=0:return 0
        support=sum(int(evidence) for (row_channel,row_factor,source),evidence
                    in self._causal_dialogue_formulation_evidence.items()
                    if row_channel==channel and row_factor==factor
                    and source not in self.language._withdrawn)
        return support if abs(support)>=MIN_BINDING_SOURCES else 0

    def _reconcile_causal_dialogue_disputes(self,channel=0):
        """Delete alternative-cause needs only after a unique world settlement."""
        channel=max(0,int(channel));learner=self.language_adult.world_causal_learning;resolved=[]
        for row in learner.current_resolutions():
            cause=self.language_adult.current_leaf_for_historical(int(row[2]));effect=self.language_adult.current_leaf_for_historical(int(row[3]))
            if cause is not None and effect is not None:resolved.append((int(cause.identity),int(effect.identity)))
        removed=0
        for key in tuple(self._causal_dialogue_dispute_evidence):
            row_channel,_receipt,emitted,candidate,effect,_source=key
            if (channel and row_channel!=channel) or emitted==candidate:continue
            causes={cause for cause,row_effect in resolved if self.language_adult.leaf_equivalent(row_effect,effect)}
            if len(causes)==1:self._causal_dialogue_dispute_evidence.pop(key,None);removed+=1
        return removed

    def _causal_action_coordinates(self,receipt):
        emitted=[]
        for pid,factor in zip(receipt.programs,receipt.factors):
            orientation=int(self.language_adult.world_causal_learning.grounding.orientation(int(factor)))
            members=self._causal_program_members(int(pid),int(factor))
            if len(members)!=2 or not orientation:continue
            cause,effect=((members[1],members[0]) if orientation>0 else members)
            bindings=[]
            for causal_receipt,binding in self.language_adult.world_causal_learning.bindings.items():
                row_effect=self.language_adult.current_leaf_for_historical(int(binding.effect))
                row_causes=tuple(self.language_adult.current_leaf_for_historical(int(value)) for value in binding.causes)
                if row_effect is None or any(value is None for value in row_causes) or not self.language_adult.leaf_equivalent(int(row_effect.identity),effect) or not any(self.language_adult.leaf_equivalent(int(value.identity),cause) for value in row_causes):continue
                occurrence=self.language_adult.world_causal_learning.ecology.pending.get(int(causal_receipt));opened=int(occurrence.opened_tick) if occurrence is not None else 0
                bindings.append((opened,int(causal_receipt)))
            if bindings:emitted.append((max(bindings)[1],cause,effect))
        return tuple(sorted(set(emitted)))

    def _reconcile_causal_action_lineage(self):
        """Retire action context whose exact world-evidence receipt is no longer live."""
        learner=self.language_adult.world_causal_learning
        learner._prune_expired_bindings()
        live=set(map(int,learner.bindings));removed=0
        for key in tuple(self._settled_causal_action_lineage):
            if int(key[1]) in live:continue
            self._settled_causal_action_lineage.pop(key,None);removed+=1
            indexed=self._settled_causal_action_lineage_index.get(int(key[0]))
            if indexed is not None:indexed.discard(tuple(map(int,key[1:])))
            if indexed is not None and not indexed:
                self._settled_causal_action_lineage_index.pop(int(key[0]),None)
        return removed

    @staticmethod
    def _causal_action_relative_form_identity():
        """Opaque learned language role; recent action supplies its transient referent."""
        return _identity('workbench-causal-action-relative-form-v1',())

    def _causal_action_leading_coordinate(self,receipt):
        """Recover only the first certified relation of one exact resident action."""
        if receipt is None or not receipt.programs or not receipt.factors:return ()
        factor=int(receipt.factors[0]);orientation=int(
            self.language_adult.world_causal_learning.grounding.orientation(factor))
        members=self._causal_program_members(int(receipt.programs[0]),factor)
        if len(members)!=2 or not orientation:return ()
        cause,effect=((members[1],members[0]) if orientation>0 else members)
        candidates=tuple(row for row in self._causal_action_coordinates(receipt)
                         if self.language_adult.leaf_equivalent(int(row[1]),cause)
                         and self.language_adult.leaf_equivalent(int(row[2]),effect))
        return candidates[0] if len(candidates)==1 else ()

    def causal_continuation_frontier(self,continuations):
        """Reduce certified accepted relations to one topology-owned continuation focus."""
        continuations=tuple((int(receipt),int(effect)) for receipt,effect in continuations)
        self.last_causal_continuation_frontier_touches=0
        if not continuations:return ()
        learner=self.language_adult.world_causal_learning;outgoing={};receipts=set();effects=set()
        for receipt,effect in continuations:
            self.last_causal_continuation_frontier_touches+=1
            binding=learner.bindings.get(receipt);occurrence=learner.ecology.pending.get(receipt)
            if binding is None or occurrence is None or occurrence.result is None or learner.complete_source_blocks(receipt)<3:return ()
            credit={int(slot):(int(n),int(d)) for slot,n,d in occurrence.result.participant_credit};positive=[]
            for slot,cause in zip(binding.slots,binding.causes):
                n,d=credit.get(int(slot),(0,1))
                if d>0 and n>0:positive.append(int(cause))
            if len(positive)!=1:return ()
            cause_leaf=self.language_adult.current_leaf_for_historical(positive[0]);effect_leaf=self.language_adult.current_leaf_for_historical(int(binding.effect))
            if cause_leaf is None or effect_leaf is None:return ()
            current=int(effect_leaf.identity)
            if not self.language_adult.leaf_equivalent(current,effect):return ()
            receipts.add(receipt);effects.add(current);outgoing.setdefault(int(cause_leaf.identity),set()).add(current)
        if len(receipts)!=len(continuations):return ()
        if len(effects)==1:return next(iter(effects)),tuple(sorted(receipts))
        dominated=set()
        for origin in effects:
            pending=list(outgoing.get(origin,()));seen={origin}
            while pending:
                node=int(pending.pop());self.last_causal_continuation_frontier_touches+=1
                if node in seen:continue
                seen.add(node)
                if node in effects:dominated.add(origin);break
                pending.extend(outgoing.get(node,()))
        maxima=effects-dominated
        return (next(iter(maxima)),tuple(sorted(receipts))) if len(maxima)==1 else ()

    def _embedded_certified_causal_acceptances(self,raw,emitted):
        """Find only resident-rematerializable certified propositions inside one contact.

        This is deliberately not a clause splitter. Candidate proposition surfaces
        come only from the exact recent emitted causal coordinates and the Adult's
        learned self-contained relation factors. Raw bytes merely establish that a
        certified learned surface physically occurred in the authenticated contact.
        """
        raw=bytes(raw);emitted=tuple(emitted);self.last_causal_dialogue_embedded_relation_touches=0
        if not raw or not emitted:return ()
        self_contained=self._causal_self_contained_factors();continuations=self._causal_continuation_factors();accepted=set();rows={}
        for causal_receipt,cause,effect in emitted:
            row=((),0,int(cause),int(effect),int(causal_receipt),0);rows[int(causal_receipt)]=row
            for factor in self_contained:
                self.last_causal_dialogue_embedded_relation_touches+=1
                surface=self._causal_self_contained_surface(row,int(factor))
                if surface and bytes(surface) in raw:
                    accepted.add((int(causal_receipt),int(effect)))
                    break
        # Preserve the incumbent all-self-contained batch path exactly. Continuation
        # search exists only for the one-anchor capability that path cannot express.
        if len(accepted)>=2:return tuple(sorted(accepted))
        # A learned continuation construction may omit its first argument locally.
        # That omission receives no semantic authority from bytes: it is bound only
        # when another independently certified occurrence in this same contact has
        # already established an equivalent effect. Iterate the bounded emitted
        # cohort so deeper learned continuation chains can close without a parser.
        pending={int(receipt) for receipt in rows if all(int(receipt)!=item[0] for item in accepted)}
        while pending and accepted:
            predecessor_effects=tuple(effect for _receipt,effect in accepted);progress=False
            for causal_receipt in tuple(sorted(pending)):
                row=rows[causal_receipt];cause=int(row[2])
                if not any(self.language_adult.leaf_equivalent(int(effect),cause) for effect in predecessor_effects):continue
                for factor in continuations:
                    self.last_causal_dialogue_embedded_relation_touches+=1
                    suffix=self._causal_continuation_suffix(row,int(factor))
                    if suffix and bytes(suffix) in raw:
                        accepted.add((causal_receipt,int(row[3])));pending.remove(causal_receipt);progress=True;break
            if not progress:break
        return tuple(sorted(accepted))

    def observe_authenticated_causal_dialogue_contact(self,raw,source,channel=None):
        source=int(source);channel=source if channel is None else max(0,int(channel));raw=tuple(map(int,raw))
        self.last_causal_dialogue_contact_continuations=()
        self.last_causal_dialogue_continuation_support=()
        if source<=0 or channel<=0 or not raw:return 0
        self._reconcile_causal_action_lineage()
        emitted=set(self._settled_causal_action_lineage_index.get(channel,()))
        self.last_causal_dialogue_lineage_touches=len(emitted)
        receipts=tuple(sorted((row for row in self.recent_causal_dialogue_actions.values() if int(row.channel)==channel),key=lambda row:(int(row.born_tick),int(row.identity)),reverse=True));cohort=()
        if receipts:
            latest=receipts[0];cohort=tuple(receipt for receipt in receipts if int(receipt.episode)==int(latest.episode))
            for receipt in cohort:emitted.update(self._causal_action_coordinates(receipt))
        pending_obligations=tuple(r for r in self.pending_endogenous_inquiry_actions.values()
                                  if int(r.channel)==channel and r.obligation_candidates and int(r.obligation_effect)>0)
        eligible_obligation_receipts=None
        if len(pending_obligations)==1:
            obligation=pending_obligations[0];eligible_obligation_receipts=set()
            for (row_channel,causal_receipt,_emitted,candidate,effect,evidence_source),evidence in self._causal_dialogue_dispute_evidence.items():
                if row_channel!=channel or int(evidence)<=0 or int(evidence_source) in self.language._withdrawn:continue
                if (any(self.language_adult.leaf_equivalent(int(candidate),int(bound))
                        for bound in obligation.obligation_candidates)
                        and self.language_adult.leaf_equivalent(int(effect),int(obligation.obligation_effect))):eligible_obligation_receipts.add(int(causal_receipt))
            emitted={row for row in emitted if int(row[0]) in eligible_obligation_receipts}
        emitted=tuple(sorted(emitted))
        if not emitted:return 0
        learner=self.language_adult.world_causal_learning;matched=set();matched_continuations=set();matched_forms=set();disputed=set()
        acted_forms={}
        for receipt in cohort:
            coordinates=self._causal_action_coordinates(receipt)
            for program,acted_factor in zip(receipt.programs,receipt.factors):
                orientation=int(learner.grounding.orientation(int(acted_factor)))
                members=self._causal_program_members(int(program),int(acted_factor))
                if len(members)!=2 or not orientation:continue
                cause,effect=((members[1],members[0]) if orientation>0 else members)
                for coordinate in coordinates:
                    if (self.language_adult.leaf_equivalent(int(coordinate[1]),cause)
                            and self.language_adult.leaf_equivalent(int(coordinate[2]),effect)):
                        acted_forms[tuple(map(int,coordinate))]=int(acted_factor)
        embedded=self._embedded_certified_causal_acceptances(raw,emitted)
        if len(embedded)>=2:
            matched.update(receipt for receipt,_effect in embedded);matched_continuations.update(embedded)
        try:direct_bindings=self.language.invert_surface(raw,max_candidates=16)
        except ValueError:direct_bindings=()
        try:spans=self.language.invert_span(raw,max_candidates=16)
        except ValueError:spans=()
        for span in spans:
            if len(span.children)!=2:continue
            leaves=[]
            for child in span.children:
                try:bindings=self.language.invert_surface(child,max_candidates=8)
                except ValueError:bindings=()
                identities=[]
                for binding in bindings:
                    if binding.atoms and int(binding.context)>0:
                        try:identities.append(int(self.language_adult.leaf(binding.context,binding.atoms).identity))
                        except Exception:pass
                identities=tuple(sorted(set(identities)))
                if len(identities)!=1:leaves=[];break
                leaves.append(identities[0])
            if len(leaves)!=2:continue
            factor=self.language._span_tid(span.template_identity);leaf_set=frozenset(leaves)
            for row in learner.current_resolutions():
                cause=self.language_adult.current_leaf_for_historical(int(row[2]));effect=self.language_adult.current_leaf_for_historical(int(row[3]))
                if cause is None or effect is None or frozenset((int(cause.identity),int(effect.identity)))!=leaf_set:continue
                acted=tuple(item for item in emitted if frozenset(item[1:])==leaf_set)
                if not acted:continue
                if learner.language_relation_certificate(self.language_adult,factor,*leaves)[:1]==(1,):
                    matched.update(int(item[0]) for item in acted)
                    matched_continuations.update((int(item[0]),int(item[2])) for item in acted)
                    matched_forms.update((*tuple(map(int,item)),int(factor)) for item in acted)
                elif learner.language_relation_certificate(self.language_adult,factor,*tuple(reversed(leaves)))[:1]==(1,):disputed.update((int(item[0]),int(row[2]),int(row[2]),int(row[3])) for item in acted)
            for candidate_cause,candidate_effect,_evidence,_sources in learner.current_testimony_resolutions():
                cause=self.language_adult.current_leaf_for_historical(int(candidate_cause));effect=self.language_adult.current_leaf_for_historical(int(candidate_effect))
                if cause is None or effect is None or frozenset((int(cause.identity),int(effect.identity)))!=leaf_set or learner.language_relation_certificate(self.language_adult,factor,*leaves)[:1]!=(4,):continue
                for causal_receipt,emitted_cause,emitted_effect in emitted:
                    if self.language_adult.leaf_equivalent(emitted_effect,int(effect.identity)) and not self.language_adult.leaf_equivalent(emitted_cause,int(cause.identity)):
                        disputed.add((causal_receipt,emitted_cause,int(candidate_cause),int(candidate_effect)))
        for causal_receipt,cause,effect,observed_factor in sorted(matched_forms):
            acted_factor=int(acted_forms.get((causal_receipt,cause,effect),0))
            if acted_factor<=0:continue
            deltas=((acted_factor,1),) if acted_factor==observed_factor else (
                (acted_factor,-1),(observed_factor,1))
            for factor,delta in deltas:
                key=(channel,int(factor),source);prior=int(
                    self._causal_dialogue_formulation_evidence.get(key,0))
                if (key not in self._causal_dialogue_formulation_evidence
                        and len(self._causal_dialogue_formulation_evidence)
                            >=MAX_CAUSAL_DIALOGUE_UPTAKE_ROWS):continue
                after=max(-MAX_CAUSAL_DIALOGUE_UPTAKE_EVIDENCE,
                          min(MAX_CAUSAL_DIALOGUE_UPTAKE_EVIDENCE,prior+int(delta)))
                if after:self._causal_dialogue_formulation_evidence[key]=after
                else:self._causal_dialogue_formulation_evidence.pop(key,None)
        changed=0
        for causal_receipt in sorted(matched):
            for dispute_key in tuple(self._causal_dialogue_dispute_evidence):
                if dispute_key[0]==channel and dispute_key[1]==causal_receipt and dispute_key[-1]==source:self._causal_dialogue_dispute_evidence.pop(dispute_key,None)
            key=(channel,causal_receipt,source);prior=int(self._causal_dialogue_uptake_evidence.get(key,0))
            if key not in self._causal_dialogue_uptake_evidence and len(self._causal_dialogue_uptake_evidence)>=MAX_CAUSAL_DIALOGUE_UPTAKE_ROWS:continue
            after=min(MAX_CAUSAL_DIALOGUE_UPTAKE_EVIDENCE,prior+1);self._causal_dialogue_uptake_evidence[key]=after;changed+=int(after!=prior)
        for causal_receipt,emitted_cause,candidate_cause,effect in sorted(row for row in disputed if row[0] not in matched):
            key=(channel,causal_receipt,emitted_cause,candidate_cause,effect,source);prior=int(self._causal_dialogue_dispute_evidence.get(key,0))
            if key not in self._causal_dialogue_dispute_evidence and len(self._causal_dialogue_dispute_evidence)>=MAX_CAUSAL_DIALOGUE_UPTAKE_ROWS:continue
            self._causal_dialogue_dispute_evidence[key]=min(MAX_CAUSAL_DIALOGUE_UPTAKE_EVIDENCE,prior+1)
        action_relative=()
        if not matched and not disputed and len(cohort)==1:
            coordinate=self._causal_action_leading_coordinate(cohort[0])
            if coordinate and (eligible_obligation_receipts is None
                               or int(coordinate[0]) in eligible_obligation_receipts):
                role=self._causal_action_relative_form_identity()
                learned={tuple(map(int,units)) for support,units,_sources
                         in self.language.lexeme_candidates(role)
                         if int(support)>=MIN_BINDING_SOURCES}
                nested=self.language_adult._reconstruct_unique_nested_bindings(raw)
                nested_role=tuple(binding for binding in nested
                                  if role in tuple(map(int,binding.atoms))
                                  and any((historical is not None
                                           and int(historical[0])==role
                                           and tuple(map(int,historical[1])) in learned)
                                          for lexical_identity in binding.lexical_identities
                                          for historical in (self.language.historical_lexeme_binding(
                                              int(lexical_identity)),)))
                if raw in learned or len(nested_role)==1:
                    action_relative=((int(coordinate[0]),int(coordinate[2])),)
                    if len(nested_role)==1:
                        supported=[]
                        for span in spans:
                            if len(span.children)!=1:continue
                            try:child_bindings=self.language.invert_surface(
                                span.children[0],max_candidates=8)
                            except ValueError:child_bindings=()
                            if any(role in tuple(map(int,binding.atoms))
                                   for binding in child_bindings):
                                supported.append((int(span.context),1,
                                    int(self.language._span_tid(span.template_identity))))
                        if len(set(supported))==1:
                            self.last_causal_dialogue_continuation_support=tuple(supported[0])
                elif (not nested and not any(
                        tuple(map(int,candidate_units))==tuple(map(int,units))
                        for binding in direct_bindings
                        for lexeme_identity in binding.lexical_identities
                        for historical in (self.language.historical_lexeme_binding(
                            int(lexeme_identity)),) if historical is not None
                        for feature,units in (historical,)
                        for _support,candidate_units,_sources
                        in self.language.lexeme_candidates(int(feature)))):
                    self.language_adult.observe_surface_item(role,raw,source)
        tick=int(self.language_adult._tick)
        self.last_causal_dialogue_contact_continuations=tuple(sorted(
            (*row,tick) for row in ((*matched_continuations,*action_relative))
            if row in action_relative or row[0] in matched))
        return changed

    def begin_causal_continuation(self,receipt,support):
        """Persist only the unresolved learned intention, never its future surface."""
        support=tuple(map(int,support));coordinate=self._causal_action_leading_coordinate(receipt)
        if (not isinstance(receipt,CausalDialogueActionReceiptV1)
                or len(support)!=3 or min(support)<=0 or not coordinate
                or len(self._causal_continuation_commitments)>=8
                and int(receipt.channel) not in self._causal_continuation_commitments):return False
        context,arity,template=support
        if self.language.historical_span_pieces(template) is None:return False
        self._causal_continuation_commitments[int(receipt.channel)]=CausalContinuationCommitmentV1(
            int(receipt.source),int(receipt.channel),context,arity,template,
            int(coordinate[2]),int(coordinate[0]),int(receipt.identity),False)
        return True

    def _continuation_support_active(self,commitment):
        template=self.language.span_template(
            int(commitment.support_context),int(commitment.support_arity))
        return bool(template and int(template.tid)==int(commitment.support_template))

    def _settle_causal_continuation_action(self,receipt):
        commitment=self._causal_continuation_commitments.get(int(receipt.channel))
        if commitment is None or int(commitment.awaiting_action)!=int(receipt.identity):return False
        coordinate=self._causal_action_leading_coordinate(receipt)
        if not coordinate:return False
        self._causal_continuation_commitments[int(receipt.channel)]=CausalContinuationCommitmentV1(
            commitment.source,commitment.channel,commitment.support_context,
            commitment.support_arity,commitment.support_template,int(coordinate[2]),
            int(coordinate[0]),0,True)
        return True

    def causal_continuation_work_pending(self,channel=0):
        channels=((int(channel),) if int(channel)>0
                  else tuple(sorted(self._causal_continuation_commitments)))
        return any((row:=self._causal_continuation_commitments.get(candidate)) is not None
                   and row.ready and self._continuation_support_active(row)
                   for candidate in channels)

    def externalize_pending_causal_continuation(self,source,channel=0):
        source=int(source);channel=max(0,int(channel));commitment=self._causal_continuation_commitments.get(channel)
        if (commitment is None or not commitment.ready or source!=commitment.source
                or not self._continuation_support_active(commitment)):return b'',None
        recent_factors=[];cursor=int(commitment.focus);remaining=tuple(
            prior for prior in self.recent_causal_dialogue_actions.values()
            if int(prior.source)==source and int(prior.channel)==channel)
        while remaining:
            candidates=[]
            for prior in remaining:
                coordinate=self._causal_action_leading_coordinate(prior)
                if coordinate and self.language_adult.leaf_equivalent(
                        int(coordinate[2]),cursor):candidates.append((prior,coordinate))
            if len(candidates)!=1:break
            prior,coordinate=candidates[0];recent_factors.extend(map(int,prior.factors))
            cursor=int(coordinate[1]);remaining=tuple(row for row in remaining
                                                     if int(row.identity)!=int(prior.identity))
        surface,receipt=self.externalize_causal_component(
            commitment.focus,source,channel,(commitment.resolved_receipt,),
            commitment.focus if recent_factors else 0,recent_factors,True)
        if receipt is None:
            self._causal_continuation_commitments.pop(channel,None);return b'',None
        self._causal_continuation_commitments[channel]=CausalContinuationCommitmentV1(
            commitment.source,commitment.channel,commitment.support_context,
            commitment.support_arity,commitment.support_template,commitment.focus,
            commitment.resolved_receipt,int(receipt.identity),False)
        return bytes(surface),receipt

    def _causal_discourse_frontier(self,leaf_identity,channel=0,transient_resolved=(),forward_only=False):
        leaf_identity=int(leaf_identity);focus_rows=tuple(self.causal_focus_rows(leaf_identity));channel=max(0,int(channel))
        transient_resolved=frozenset(map(int,transient_resolved))
        if not focus_rows:return (),(),0,None
        state_programs=tuple(self.causal_program_for_row(row,materialize=False) for row in focus_rows)
        if not state_programs or not all(state_programs):return (),(),0,None
        context=self._causal_dialogue_appraisal_context(state_programs,channel)
        appraisal_program=self._causal_expression_credit_program(state_programs[0],context) or state_programs[0]
        felt=self.language_adult.somatic_appraisal(appraisal_program,context)
        rows=tuple(row for row in self.causal_message_rows(leaf_identity)
                   if int(row[4]) not in transient_resolved
                   and (not forward_only or self.language_adult.leaf_equivalent(
                        int(row[2]),leaf_identity))
                   and (channel<=0
                        or self.causal_dialogue_uptake_support(channel,int(row[4]))<MIN_BINDING_SOURCES
                        or self.causal_dialogue_dispute_support(channel,int(row[4]))>0))
        return rows[:self._causal_externalization_limit(felt,len(rows))],state_programs,context,felt

    def compose_causal_component(self,leaf_identity,with_expression_factors=False,channel=0,transient_resolved=(),prior_effect=0,avoid_factors=(),forward_only=False):
        expression_factors=[]
        def done(surface=b'',programs=()):
            result=(bytes(surface),tuple(map(int,programs)))
            return (*result,tuple(expression_factors)) if with_expression_factors else result
        rows,causal_state_programs,appraisal_context,_felt=self._causal_discourse_frontier(
            leaf_identity,channel,transient_resolved,forward_only)
        if not rows or not causal_state_programs:return done()
        # A port-first self-contained form is a complete one-relation motor act;
        # only literal-opening forms can lawfully head a wider coordinated plan.
        self_contained=(self._causal_self_contained_factors() if len(rows)==1
                        else self._causal_opening_factors())
        prior_effect=max(0,int(prior_effect));continuations=self._causal_continuation_factors();planned=[];used=set(map(int,avoid_factors));previous_effect=prior_effect or None
        for row in rows:
            contiguous=previous_effect is not None and int(row[2])==previous_effect;candidates=continuations if contiguous else self_contained;selected=0;program=0
            ordered=self._causal_formulation_candidates(
                row,candidates,appraisal_context,channel)
            ordered=tuple(item for item in ordered if item[0] not in used)+tuple(item for item in ordered if item[0] in used)
            for factor,candidate in ordered:
                if contiguous and self._causal_continuation_suffix(row,factor) is None:continue
                if candidate>0:selected=int(factor);program=int(candidate);break
            if program<=0:break
            planned.append((row,program,selected,contiguous));used.add(selected);previous_effect=int(row[3])
        if len(planned)!=len(rows):return done()
        expression_factors.extend(item[2] for item in planned)
        programs=[];out=();previous_effect=prior_effect or None;index=0
        while index<len(planned):
            row,program,factor,_contiguous=planned[index];siblings=[planned[index]]
            while index+len(siblings)<len(planned) and int(planned[index+len(siblings)][0][2])==int(row[2]):siblings.append(planned[index+len(siblings)])
            next_index=index+len(siblings);following_cause=(int(planned[next_index][0][2]) if next_index<len(planned) else 0)
            motor_siblings=self._causal_motor_siblings(siblings,following_cause)
            contiguous=previous_effect is not None and int(row[2])==previous_effect
            render_factor=int(factor)
            if contiguous:
                ordered_continuations=(int(factor),*(int(candidate) for candidate in continuations
                    if int(candidate)!=int(factor)))
                render_factor=next((int(candidate) for candidate in ordered_continuations
                                    if all(self._causal_continuation_suffix(item[0],candidate) is not None
                                           for item in motor_siblings)),0)
            grouped=None;coordination_factor=0
            if len(siblings)>1:
                candidate=self.language_adult.common_cause_span_expression(
                    *(int(item[0][3]) for item in motor_siblings))
                if not candidate:
                    candidate=self.language_adult.common_cause_span_expression(
                        int(motor_siblings[0][0][3]),int(motor_siblings[1][0][3]))
                if candidate:
                    coordination_factor=int(candidate[0]);grouped=self._causal_sibling_surface(
                        motor_siblings,coordination_factor,render_factor,
                        bool(prior_effect and not out and contiguous))
            if grouped is not None:
                surface=tuple(grouped);programs.extend(int(item[1]) for item in siblings);expression_factors.append(coordination_factor)
                if render_factor!=int(factor):expression_factors.append(render_factor)
                index+=len(siblings);previous_effect=int(motor_siblings[-1][0][3])
            else:
                suffix=self._causal_continuation_suffix(row,render_factor) if contiguous else None
                surface=(tuple(suffix) if suffix is not None else self._causal_self_contained_surface(row,factor))
                if surface is None:return done()
                if suffix is not None and render_factor!=int(factor):expression_factors.append(render_factor)
                programs.append(int(program));index+=1;previous_effect=int(row[3])
            if not out:
                out=(self._cross_action_continuation_fragment(surface)
                     if prior_effect and contiguous else tuple(surface))
                if not out:return done()
                continue
            if contiguous and (grouped is not None or suffix is not None):
                limit=min(len(out),len(surface));overlap=max((width for width in range(1,limit+1) if out[-width:]==surface[:width]),default=0);out=out+tuple(surface[overlap:]);continue
            join=int(self._join_live());merged=self.language.realize_span(join,(out,tuple(surface))) if join>0 else None
            if merged is None:return done()
            out=tuple(merged)
        return done(out,programs)
    def compose_causal_groups(self,leaf_identity,channel=0):
        """Factor one current causal explanation into structural action groups.

        Groups are not stored paragraph objects.  They are disposable projections of
        the exact branch-aware plan already selected by ``compose_causal_component``.
        A new group begins only when the next certified causal row is not continuous
        with the previous row (next cause != previous effect).  The body may render
        completed motor trajectories with spacing, but no newline or paragraph token
        participates in resident cognition.
        """
        full,programs,factors=self.compose_causal_component(
            leaf_identity,with_expression_factors=True,channel=channel)
        if not full or not programs:return ()
        expression_factors=tuple(map(int,factors));causal_factors=expression_factors[:len(programs)]
        if len(causal_factors)!=len(programs):return ()
        rows,causal_state_programs,_appraisal_context,_felt=self._causal_discourse_frontier(leaf_identity,channel)
        if len(rows)!=len(programs) or not causal_state_programs:return ()
        planned=tuple((row,int(program),int(factor)) for row,program,factor in zip(rows,programs,causal_factors))
        partitions=[];current=[];previous_effect=None;previous_cause=None
        for item in planned:
            row=item[0];cause=int(row[2]);effect=int(row[3])
            contiguous=previous_effect is not None and cause==previous_effect
            sibling=False
            if current and previous_cause is not None and cause==previous_cause:
                prior_effect=int(current[-1][0][3])
                certificate=self.language_adult.common_cause_span_expression(prior_effect,effect)
                sibling=bool(certificate and self._causal_sibling_surface((current[-1],item),int(certificate[0])) is not None)
            if current and not (contiguous or sibling):
                partitions.append(tuple(current));current=[]
            current.append(item);previous_effect=effect;previous_cause=cause
        if current:partitions.append(tuple(current))
        groups=[]
        for partition in partitions:
            out=();group_programs=[];previous_effect=None;index=0
            while index<len(partition):
                row,program,factor=partition[index];siblings=[partition[index]]
                while (index+len(siblings)<len(partition)
                       and int(partition[index+len(siblings)][0][2])==int(row[2])):
                    siblings.append(partition[index+len(siblings)])
                grouped=None
                if len(siblings)>1:
                    candidate=self.language_adult.common_cause_span_expression(
                        *(int(item[0][3]) for item in siblings))
                    if not candidate:
                        candidate=self.language_adult.common_cause_span_expression(
                            int(siblings[0][0][3]),int(siblings[1][0][3]))
                    if candidate:grouped=self._causal_sibling_surface(siblings,int(candidate[0]))
                if grouped is not None:
                    contiguous=previous_effect is not None and int(row[2])==previous_effect
                    surface=tuple(grouped);group_programs.extend(int(item[1]) for item in siblings)
                    index+=len(siblings);previous_effect=None
                else:
                    contiguous=previous_effect is not None and int(row[2])==previous_effect
                    suffix=self._causal_continuation_suffix(row,factor) if contiguous else None
                    surface=tuple(suffix) if suffix is not None else self._causal_self_contained_surface(row,factor)
                    if surface is None:return ()
                    group_programs.append(int(program));index+=1;previous_effect=int(row[3])
                if not out:out=tuple(surface);continue
                if contiguous and (grouped is not None or suffix is not None):
                    limit=min(len(out),len(surface));overlap=max(
                        (width for width in range(1,limit+1) if out[-width:]==surface[:width]),default=0)
                    out=out+tuple(surface[overlap:]);continue
                join=int(self._join_live());merged=self.language.realize_span(join,(out,tuple(surface))) if join>0 else None
                if merged is None:return ()
                out=tuple(merged)
            groups.append([out,tuple(group_programs)])
        return tuple((bytes(surface),tuple(programs)) for surface,programs in groups)
    def externalize_causal_groups(self,leaf_identity,source,channel=0):
        """Publish each structural group as its own resident motor trajectory."""
        source=int(source);channel=max(0,int(channel));groups=self.compose_causal_groups(
            leaf_identity,channel)
        if not groups or source<=0:return (),()
        if len(self.pending_causal_dialogue_actions)+len(groups)>8:return (),()
        all_programs=tuple(program for _surface,programs in groups for program in programs)
        _full,selected,factors=self.compose_causal_component(
            leaf_identity,with_expression_factors=True,channel=channel)
        expression_factors=tuple(map(int,factors));causal_factors=expression_factors[:len(selected)];rows,_,_,_=self._causal_discourse_frontier(leaf_identity,channel);rows=tuple(rows[:len(selected)])
        materialized=tuple(self.causal_program_for_row(row,factor) for row,factor in zip(rows,causal_factors))
        if tuple(selected)!=all_programs or materialized!=all_programs:return (),()
        component=self._longest_causal_program_component()
        context_programs=component if all_programs and len(component)>=len(all_programs) else all_programs
        if (not context_programs or len(context_programs)>16
                or len(set(context_programs))!=len(context_programs)
                or any(self.language_adult.programs.factor(pid) is None for pid in context_programs)):
            return (),()
        episode=_identity('causal-dialogue-group-episode-v1',(all_programs,source,channel,int(self.language_adult._tick)+1))
        receipts=[]
        for surface,programs in groups:
            group_factors=(expression_factors if len(groups)==1 and tuple(programs)==tuple(selected)
                           else tuple(int(self.language_adult.programs.factor(program) or 0) for program in programs))
            receipt=self.stage_causal_dialogue_action(
                programs,surface,source,channel,context_programs,group_factors,episode)
            if receipt is None:raise RuntimeError('mathematical-workbench-adult:causal-group-stage')
            receipts.append(receipt)
        return tuple(surface for surface,_programs in groups),tuple(receipts)
    def externalize_causal_component(self,leaf_identity,source,channel=0,transient_resolved=(),prior_effect=0,avoid_factors=(),forward_only=False):
        surface,programs,factors=self.compose_causal_component(
            leaf_identity,with_expression_factors=True,channel=channel,
            transient_resolved=transient_resolved,prior_effect=prior_effect,
            avoid_factors=avoid_factors,forward_only=forward_only)
        expression_factors=tuple(map(int,factors));causal_factors=expression_factors[:len(programs)];rows,_,_,_=self._causal_discourse_frontier(leaf_identity,channel,transient_resolved,forward_only);rows=tuple(rows[:len(programs)])
        materialized=tuple(self.causal_program_for_row(row,factor) for row,factor in zip(rows,causal_factors))
        if materialized!=programs:return b'',None
        component=self._longest_causal_program_component()
        context_programs=component if programs and len(component)>=len(programs) else programs
        receipt=self.stage_causal_dialogue_action(programs,surface,source,channel,context_programs,expression_factors) if surface else None
        return (surface,receipt) if receipt is not None else (b'',None)
    @staticmethod
    def _complete_relation_expression(expression):
        if expression is None:return None,None
        out=[]
        while True:
            step=expression.emit()
            if step is None:break
            value,_ordinal=step;out.append(int(value))
            if not expression.reafference(step,value):raise RuntimeError('mathematical-workbench-adult:relation-basis-reafference')
        return bytes(out),expression.receipt
    def externalize_relation_frontier(self,identities):
        return self._complete_relation_expression(self.relation_basis_expression_for_frontier(identities))
    def externalize_relation_basis(self):
        return self._complete_relation_expression(self.relation_basis_expression())
    def _commit_operator_public_action(self,winner):
        trace_digest=_digest('variable-unfold-trace-v1',tuple(winner.trace))
        receipt_identity=_identity('variable-unfold-public-receipt-v1',(winner.root_identity,winner.witness.identity,trace_digest,self.operators.revision))
        receipt=PublicActionReceiptV1(winner.root_identity,winner.witness.identity,trace_digest,self.operators.revision,receipt_identity)
        self.pending_operator_actions[receipt.identity]=receipt
        self.operator_public_count+=1;return receipt
    def operator_expression(self):
        if not self.operator_externalization_enabled:return None
        winner=self.operator_winner()
        if winner is None or winner.witness is None:return None
        surface=self.project_operator_trace(winner.trace)
        return None if surface is None else OperatorSurfaceExpressionV1(self,winner,surface)
    def externalize_operator(self):
        expression=self.operator_expression()
        if expression is None:return None,None
        out=[]
        while True:
            step=expression.emit()
            if step is None:break
            value,_ordinal=step;out.append(int(value))
            if not expression.reafference(step,value):raise RuntimeError('mathematical-workbench-adult:operator-reafference')
        return bytes(out),expression.receipt
    def settle_operator_consequence(self,receipt,source,effect,independent=True,controllable=True,endogenous=False):
        if not isinstance(receipt,PublicActionReceiptV1) or receipt.identity not in self.pending_operator_actions:return False
        changed=self.operators.settle_consequence(receipt,source,effect,independent,controllable,endogenous)
        if changed:self.pending_operator_actions.pop(receipt.identity,None)
        return bool(changed)
    @staticmethod
    def causal_dialogue_context(programs,channel=0):
        return _identity('causal-dialogue-component-context-v2',(tuple(map(int,programs)),max(0,int(channel))))
    def _causal_program_members(self,program,factor=0):
        """Invert one active causal program identity without persisting its lowering."""
        program=int(program);factor=int(factor);chunk=self.language_adult.programs.chunks.get(program)
        if chunk is not None:return tuple(map(int,chunk.members))
        learner=self.language_adult.world_causal_learning;factors=((factor,) if factor>0 else tuple(learner.grounding.rows));matches=set()
        for row in learner.current_resolutions():
            cause=self.language_adult.current_leaf_for_historical(int(row[2]));effect=self.language_adult.current_leaf_for_historical(int(row[3]))
            if cause is None or effect is None:continue
            for candidate_factor in factors:
                orientation=int(learner.grounding.orientation(int(candidate_factor)))
                if not orientation:continue
                members=((int(effect.identity),int(cause.identity)) if orientation>0 else (int(cause.identity),int(effect.identity)))
                if int(self.language_adult.programs.ident(members,int(candidate_factor)))==program:matches.add(members)
        return next(iter(matches)) if len(matches)==1 else ()
    def stage_causal_dialogue_action(self,programs,surface,source,channel=0,context_programs=(),factors=(),episode=0):
        programs=tuple(map(int,programs));surface=bytes(surface);source=int(source);channel=max(0,int(channel))
        factors=tuple(map(int,factors)) if factors else tuple(int(self.language_adult.programs.factor(pid) or 0) for pid in programs)
        context_programs=tuple(map(int,context_programs)) if context_programs else programs
        if not programs or len(programs)>16 or len(set(programs))!=len(programs) or len(factors)<len(programs) or any(factor<=0 or self.language.historical_span_pieces(factor) is None for factor in factors) or any(len(self._causal_program_members(pid,factor))!=2 for pid,factor in zip(programs,factors)) or not surface or source<=0:return None
        if not context_programs or len(context_programs)>16 or len(set(context_programs))!=len(context_programs):return None
        context=self.causal_dialogue_context(context_programs,channel)
        if channel>0:
            prior=int(self._causal_dialogue_context_by_channel.get(channel,0));profile=self.language_adult.credit.rows.get(programs[0])
            if prior>0 and profile is not None and prior in profile.contexts:context=prior
            self._causal_dialogue_context_by_channel[channel]=int(context)
        start=self.language_adult._advance();end=start+1
        for pid in programs:self.language_adult.credit.observe_use(pid,start,end,1<<12,context)
        self.language_adult._tick=end+1;self.language_adult._select_epoch+=1
        digest=hashlib.sha256(surface).hexdigest();born=int(self.language_adult._tick);episode=max(0,int(episode))
        if episode<=0:episode=_identity('causal-dialogue-single-episode-v1',(programs,context,digest,source,channel,born))
        identity=_identity('causal-dialogue-action-receipt-v2',(programs,factors,context,digest,source,channel,born))
        if identity in self.pending_causal_dialogue_actions or len(self.pending_causal_dialogue_actions)>=8:return None
        receipt=CausalDialogueActionReceiptV1(programs,factors,context,digest,source,channel,episode,born,identity)
        self.pending_causal_dialogue_actions[identity]=receipt;self.causal_dialogue_public_count+=1;return receipt
    def settle_causal_dialogue_return(self,receipt,source,outcome_q16,somatic_q16=0,independent=True,action_succeeded=None):
        if not isinstance(receipt,CausalDialogueActionReceiptV1):return False
        staged=receipt.identity in self.pending_causal_dialogue_actions
        reafferenced=(receipt.identity in self.pending_causal_dialogue_consequences
                      and self.recent_causal_dialogue_actions.get(receipt.identity) is receipt)
        if not (staged or reafferenced):return False
        source=int(source);independent=bool(independent);succeeded=(independent if action_succeeded is None else bool(action_succeeded));tick=self.language_adult._advance()
        if source<=0:return False
        for pid in receipt.programs:
            self.language_adult.credit.observe_return(pid,int(outcome_q16),int(somatic_q16),tick,independent,receipt.context)
            self.language_adult.credit.observe_control(pid,True,independent and succeeded,receipt.context)
        self.language_adult._select_epoch+=1
        if independent and int(receipt.channel)>0:
            for causal_receipt,cause,effect in self._causal_action_coordinates(receipt):
                key=(int(receipt.channel),int(causal_receipt),int(cause),int(effect));coordinate=key[1:]
                self._settled_causal_action_lineage[key]=(int(receipt.episode),int(tick));self._settled_causal_action_lineage_index.setdefault(key[0],set()).add(coordinate)
            while len(self._settled_causal_action_lineage)>MAX_SETTLED_CAUSAL_ACTION_LINEAGE_ROWS:
                victim=min(self._settled_causal_action_lineage.items(),key=lambda item:(int(item[1][1]),item[0]))[0];self._settled_causal_action_lineage.pop(victim,None)
                indexed=self._settled_causal_action_lineage_index.get(victim[0]);indexed.discard(victim[1:]) if indexed is not None else None
                if indexed is not None and not indexed:self._settled_causal_action_lineage_index.pop(victim[0],None)
        if staged:self._archive_causal_dialogue_action(receipt)
        if succeeded:self._settle_causal_continuation_action(receipt)
        self.pending_causal_dialogue_consequences.discard(receipt.identity)
        return True
    def settle_causal_dialogue_reafference(self,receipt,source):
        """Authenticate one emitted act without inventing an external consequence."""
        if (not isinstance(receipt,CausalDialogueActionReceiptV1)
                or receipt.identity not in self.pending_causal_dialogue_actions
                or int(source)!=int(receipt.source)):return False
        self.language_adult._advance();self._archive_causal_dialogue_action(receipt)
        # Own-action reafference proves that these bytes reached the body; it is
        # not evidence that the world accepted the act.  Only the later exact
        # consequence may release a settled continuation.
        self.pending_causal_dialogue_consequences.add(receipt.identity);return True
    def _archive_causal_dialogue_action(self,receipt):
        self.pending_causal_dialogue_actions.pop(receipt.identity,None);self.recent_causal_dialogue_actions[receipt.identity]=receipt
        while len(self.recent_causal_dialogue_actions)>8:
            victim=min(self.recent_causal_dialogue_actions.values(),key=lambda row:(row.born_tick,row.identity));self.recent_causal_dialogue_actions.pop(victim.identity,None);self.pending_causal_dialogue_consequences.discard(victim.identity)
    def observe_causal_dialogue_background(self,receipt_identity,source,outcome_occurs=False):
        receipt=self.recent_causal_dialogue_actions.get(int(receipt_identity));source=int(source)
        if receipt is None or source<=0:return False
        for pid in receipt.programs:self.language_adult.credit.observe_control(pid,False,bool(outcome_occurs),receipt.context)
        self.language_adult._advance();self.language_adult._select_epoch+=1;return True

    def observe_context_affordance(self,prompt_context,prompt_atoms,target_context,source):
        prompt_context=int(prompt_context);prompt_atoms=tuple(map(int,prompt_atoms));target_context=int(target_context);source=int(source)
        if min(prompt_context,target_context,source)<=0 or not prompt_atoms:return False
        prompt_space=self.observe_relation_language_space(prompt_context,prompt_atoms)
        if prompt_space<=0:return False
        self._context_affordance_sources.setdefault((prompt_space,target_context),set()).add(source);return True
    def context_affordance(self,prompt_context,prompt_atoms):
        sid=_identity('workbench-relation-language-space-v1',(int(prompt_context),tuple(map(int,prompt_atoms))))
        candidates=[]
        for (space,target),sources in self._context_affordance_sources.items():
            if space!=sid:continue
            live=sum(1 for source in sources if source not in self.language._withdrawn)
            if live>=MIN_BINDING_SOURCES:candidates.append(int(target))
        live=tuple(sorted(set(candidates)));return live[0] if len(live)==1 else 0
    def _longest_causal_program_component(self):
        learner=self.language_adult.world_causal_learning;nodes=set()
        for row in learner.current_resolutions():nodes.update((int(row[2]),int(row[3])))
        candidates=[]
        for node in nodes:
            rows=self.causal_chain_rows(node)
            if not rows:continue
            programs=tuple(self.causal_program_for_row(row) for row in rows)
            if programs and all(programs):candidates.append((len(programs),programs))
        if not candidates:return ()
        width=max(row[0] for row in candidates);programs={row[1] for row in candidates if row[0]==width}
        return next(iter(programs)) if len(programs)==1 else ()
    def context_relation_affordance(self,prompt_context):
        """Recover a target construction learned across distinct spaces of one context.

        This is a structural relation over ordinary language experience, not a semantic
        question class.  At least two distinct prompt spaces and independent live
        sources must agree before the relation can generalize to a held-out space.
        """
        prompt_context=int(prompt_context);targets={}
        for (space,target),sources in self._context_affordance_sources.items():
            row=self._relation_language_spaces.get(int(space))
            if row is None or int(row[0])!=prompt_context:continue
            live={int(source) for source in sources if source not in self.language._withdrawn}
            if not live:continue
            entry=targets.setdefault(int(target),[set(),set()]);entry[0].add(int(space));entry[1].update(live)
        eligible=tuple(sorted(target for target,(spaces,sources) in targets.items()
                              if len(spaces)>=2 and len(sources)>=MIN_BINDING_SOURCES))
        return eligible[0] if len(eligible)==1 else 0
    def _causal_dialogue_appraisal_context(self,programs,channel):
        programs=tuple(map(int,programs));channel=max(0,int(channel))
        if not programs:return 0
        current=self.causal_dialogue_context(programs,channel)
        if channel<=0:return current
        return int(self._causal_dialogue_context_by_channel.get(channel,0)) or current
    def _causal_expression_credit_programs(self,program,context=0):
        """Rematerialize credited linguistic lowerings of one active causal relation."""
        program=int(program);context=max(0,int(context));profile=self.language_adult.credit.rows.get(program)
        direct=profile is not None and ((context in profile.contexts) if context else bool(profile.contexts))
        members=self._causal_program_members(program)
        if len(members)!=2:return ((program,) if direct else ())
        candidates={program} if direct else set()
        for factor in self.language_adult.world_causal_learning.grounding.rows:
            for ordered in (members,tuple(reversed(members))):
                candidate=int(self.language_adult.programs.ident(ordered,int(factor)));row=self.language_adult.credit.rows.get(candidate)
                if row is not None and ((context in row.contexts) if context else bool(row.contexts)):candidates.add(candidate)
        return tuple(sorted(candidates))
    def _causal_expression_credit_program(self,program,context=0):
        candidates=self._causal_expression_credit_programs(program,context)
        if not candidates:return 0
        context=max(0,int(context));ranked=[]
        for candidate in candidates:
            profile=self.language_adult.credit.rows.get(int(candidate))
            local=None if profile is None or context<=0 else profile.contexts.get(context)
            evidence=(0 if local is None else int(local.outcome_samples),
                      0 if local is None else int(local.control_attempts+local.background_attempts),
                      0 if profile is None else int(profile.exposures),
                      0 if profile is None else int(profile.last_tick))
            ranked.append((*evidence,-int(candidate),int(candidate)))
        return int(max(ranked)[-1])
    def _endogenous_state_inquiry_candidate(self,channel=0):
        """Return a transient surface/bid from a lived state-dependent capability gap."""
        self.last_endogenous_inquiry_touches=0;channel=max(0,int(channel))
        programs=self._longest_causal_program_component()
        if not programs:return None,0
        first=int(programs[0]);context=self._causal_dialogue_appraisal_context(programs,channel);appraisal_program=self._causal_expression_credit_program(first,context) or first
        appraisal=self.language_adult.somatic_appraisal(appraisal_program,context)
        atoms=self.language_adult.somatic_appraisal_atoms(appraisal)
        appraisal_context=self.language_adult._somatic_appraisal_language_context();surface=self.language.realize(appraisal_context,atoms)
        if surface is None:return None,0
        current_depth=self._causal_externalization_limit(appraisal,len(programs));lived_depth=0
        for program in programs:
            self.last_endogenous_inquiry_touches+=1
            profiles=tuple(self.language_adult.credit.rows.get(pid) for pid in self._causal_expression_credit_programs(int(program)))
            if not any(profile is not None and any(
                    bool(local.participated) and (int(local.outcome_samples) or int(local.control_attempts))
                    for local in profile.contexts.values()) for profile in profiles):break
            lived_depth+=1
        gap=lived_depth-current_depth
        if gap<=0:return None,0
        target=self.context_relation_affordance(appraisal_context)
        if target<=0:return None,0
        question=self.language.realize(target,atoms)
        return ((None,0) if question is None else (bytes(question),(gap*Q)//len(programs)))

    def endogenous_state_inquiry(self,channel=0):
        """Rematerialize inquiry when learned self-description aliases stronger lived capability."""
        return self._endogenous_state_inquiry_candidate(channel)[0]

    @staticmethod
    def _causal_inquiry_remaining_work(row):
        coalitions=((),(row.participants[0],),(row.participants[1],),tuple(row.participants));required=set(coalitions);by_source={}
        for evidence in row.evidence:
            if evidence.active:by_source.setdefault(int(evidence.source),set()).add(tuple(evidence.coalition))
        ranked=sorted((len(required&values) for values in by_source.values()),reverse=True)[:3]
        return sum(len(required)-count for count in ranked)+(3-len(ranked))*len(required)

    def _causal_inquiry_efficacy(self,channel,causal_receipt,emitted_cause,effect):
        """Rematerialize expected control from exact lived public-action statistics."""
        channel=max(0,int(channel));coordinate=(int(causal_receipt),int(emitted_cause),int(effect))
        self._reconcile_causal_action_lineage()
        if coordinate not in self._settled_causal_action_lineage_index.get(channel,()):return Q//2
        context=int(self._causal_dialogue_context_by_channel.get(channel,0));attempts=successes=background=background_successes=0
        for pid,profile in self.language_adult.credit.rows.items():
            members=self._causal_program_members(int(pid))
            if len(members)!=2 or not all(any(self.language_adult.leaf_equivalent(int(member),target) for member in members)
                                          for target in (int(emitted_cause),int(effect))):continue
            local=profile.contexts.get(context)
            if local is None:continue
            attempts+=int(local.control_attempts);successes+=int(local.control_successes)
            background+=int(local.background_attempts);background_successes+=int(local.background_successes)
        action_rate=((successes+1)*Q)//(attempts+2)
        background_rate=((background_successes+1)*Q)//(background+2)
        return max(1,min(Q,Q//2+action_rate-background_rate))

    def _causal_repair_inquiry_candidates(self,channel=0):
        """Rematerialize transient bids from unresolved certified causal alternatives."""
        channel=max(0,int(channel));self._reconcile_causal_dialogue_disputes(channel);learner=self.language_adult.world_causal_learning
        rows={}
        for (row_channel,receipt,emitted,candidate,effect,source),evidence in self._causal_dialogue_dispute_evidence.items():
            if row_channel!=channel or int(evidence)<=0 or source in self.language._withdrawn:continue
            rows.setdefault((int(candidate),int(effect)),[]).append((int(receipt),int(emitted)))
        if channel<=0 or not rows:return ()
        out=[];self.last_causal_repair_inquiry_touches=0
        for (candidate,effect),histories in sorted(rows.items()):
            cause_signature=self.language_adult.leaf_signature(candidate);effect_signature=self.language_adult.leaf_signature(effect)
            if cause_signature is None or effect_signature is None:continue
            atoms=tuple((*cause_signature[1],*effect_signature[1]));contexts=set()
            for space,_target in self._context_affordance_sources:
                self.last_causal_repair_inquiry_touches+=1;binding=self._relation_language_spaces.get(int(space))
                if binding is not None and len(binding[1])==len(atoms):contexts.add(int(binding[0]))
            surfaces=set()
            for context in sorted(contexts):
                target=self.context_relation_affordance(int(context))
                if target<=0:continue
                surface=self.language.realize(target,atoms)
                if surface is not None:surfaces.add(bytes(surface))
            if len(surfaces)!=1:continue
            fields=[]
            for receipt,binding in learner.bindings.items():
                current_effect=self.language_adult.current_leaf_for_historical(int(binding.effect));current_causes=tuple(self.language_adult.current_leaf_for_historical(int(value)) for value in binding.causes)
                if current_effect is None or any(value is None for value in current_causes) or not self.language_adult.leaf_equivalent(int(current_effect.identity),effect) or not any(self.language_adult.leaf_equivalent(int(value.identity),candidate) for value in current_causes):continue
                occurrence=learner.ecology.pending.get(int(receipt))
                if occurrence is not None:fields.append((int(occurrence.opened_tick),int(receipt),occurrence))
            if not fields:continue
            _opened,_field,occurrence=max(fields);remaining=max(1,self._causal_inquiry_remaining_work(occurrence));reach=max(1,len(self.causal_focus_rows(candidate)))
            efficacy=max(self._causal_inquiry_efficacy(channel,causal_receipt,emitted,effect)
                         for causal_receipt,emitted in histories)
            out.append((next(iter(surfaces)),(reach*efficacy)//remaining,candidate,effect))
        return tuple(out)

    def _causal_open_field_inquiry_candidates(self,channel=0):
        """Bid on evidence-bearing causal fields without inventing an ambiguity state."""
        channel=max(0,int(channel));learner=self.language_adult.world_causal_learning
        if channel<=0:return ()
        open_fields=learner.current_open_fields();self._reconcile_causal_action_lineage()
        disputed={(int(candidate),int(effect))
                  for (row_channel,_receipt,_emitted,candidate,effect,source),evidence
                  in self._causal_dialogue_dispute_evidence.items()
                  if row_channel==channel and int(evidence)>0 and source not in self.language._withdrawn}
        partner_source=learner.testimony_source(channel);out=[]
        for causes,effect,receipt,_opened in open_fields:
            occurrence=learner.ecology.pending.get(int(receipt))
            current_effect=self.language_adult.current_leaf_for_historical(int(effect))
            if occurrence is None or current_effect is None:continue
            remaining=max(1,self._causal_inquiry_remaining_work(occurrence));guided=learner._guided_testimony(receipt)
            guided_cause=(int(guided[1]) if guided else 0);guided_sources=(tuple(map(int,guided[2])) if guided else ())
            for historical_cause in causes:
                current_cause=self.language_adult.current_leaf_for_historical(int(historical_cause))
                if current_cause is None or (int(current_cause.identity),int(current_effect.identity)) in disputed:continue
                histories=[]
                for causal_receipt,emitted,row_effect in self._settled_causal_action_lineage_index.get(channel,()):
                    if (self.language_adult.leaf_equivalent(int(emitted),int(current_cause.identity))
                            and self.language_adult.leaf_equivalent(int(row_effect),int(current_effect.identity))):
                        histories.append((int(causal_receipt),int(emitted)))
                guided_here=(int(historical_cause)==guided_cause and partner_source in guided_sources)
                if not guided_here and not histories:continue
                cause_signature=self.language_adult.leaf_signature(int(current_cause.identity));effect_signature=self.language_adult.leaf_signature(int(current_effect.identity))
                if cause_signature is None or effect_signature is None:continue
                atoms=tuple((*cause_signature[1],*effect_signature[1]));surfaces=set()
                for space,_target in self._context_affordance_sources:
                    self.last_causal_repair_inquiry_touches+=1;binding=self._relation_language_spaces.get(int(space))
                    if binding is None or len(binding[1])!=len(atoms):continue
                    target=self.context_relation_affordance(int(binding[0]))
                    if target<=0:continue
                    surface=self.language.realize(target,atoms)
                    if surface is not None:surfaces.add(bytes(surface))
                if len(surfaces)!=1:continue
                efficacy=(max(self._causal_inquiry_efficacy(channel,causal_receipt,emitted,int(current_effect.identity))
                              for causal_receipt,emitted in histories) if histories else Q//2)
                source_weight=(Q+min(Q,(len(guided_sources)*Q)//MIN_BINDING_SOURCES) if guided_here else Q)
                reach=max(1,len(self.causal_focus_rows(int(current_cause.identity))))
                out.append((next(iter(surfaces)),(reach*efficacy*source_weight)//(remaining*Q),int(current_cause.identity),int(current_effect.identity)))
        return tuple(out)

    def _causal_inquiry_candidates(self,channel=0):
        """Replace two lossy polar probes with one learned same-field contrast."""
        channel=max(0,int(channel));learner=self.language_adult.world_causal_learning
        base=tuple((*self._causal_repair_inquiry_candidates(channel),
                    *self._causal_open_field_inquiry_candidates(channel)))
        causal_atoms=set()
        for binding in learner.bindings.values():
            for leaf_identity in (*binding.causes,binding.effect):
                signature=self.language_adult.leaf_signature(int(leaf_identity))
                if signature is not None and len(signature[1])==1:causal_atoms.add(int(signature[1][0]))
        source_contexts=set()
        for space,_target in self._context_affordance_sources:
            binding=self._relation_language_spaces.get(int(space))
            if binding is not None and len(binding[1])==3 and all(int(atom) in causal_atoms for atom in binding[1]):source_contexts.add(int(binding[0]))
        grouped=set();contrasts=[]
        for causes,effect,_receipt,_opened in learner.current_open_fields():
            current_effect=self.language_adult.current_leaf_for_historical(int(effect));current_causes=tuple(self.language_adult.current_leaf_for_historical(int(value)) for value in causes)
            if current_effect is None or any(value is None for value in current_causes):continue
            rows={}
            for row in base:
                if not self.language_adult.leaf_equivalent(int(row[3]),int(current_effect.identity)):continue
                if not any(self.language_adult.leaf_equivalent(int(row[2]),int(value.identity)) for value in current_causes):continue
                prior=rows.get(int(row[2]))
                if prior is None or int(row[1])>int(prior[1]):rows[int(row[2])]=row
            if len(rows)!=2:continue
            ordered=tuple(sorted(rows.values(),key=lambda row:(-int(row[1]),bytes(row[0]))));signatures=tuple(self.language_adult.leaf_signature(int(row[2])) for row in ordered);effect_signature=self.language_adult.leaf_signature(int(current_effect.identity))
            if any(signature is None for signature in signatures) or effect_signature is None:continue
            atoms=tuple((*signatures[0][1],*signatures[1][1],*effect_signature[1]));surfaces=set()
            for context in sorted(source_contexts):
                target=self.context_relation_affordance(context)
                if target<=0:continue
                surface=self.language.realize(target,atoms)
                if surface is not None:surfaces.add(bytes(surface))
            if len(surfaces)!=1:continue
            candidates=tuple(int(row[2]) for row in ordered);contrasts.append((next(iter(surfaces)),max(int(row[1]) for row in ordered),candidates,int(current_effect.identity)));grouped.update((int(row[2]),int(row[3])) for row in ordered)
        singles=tuple((bytes(surface),int(bid),(int(candidate),),int(effect))
                      for surface,bid,candidate,effect in base
                      if (int(candidate),int(effect)) not in grouped)
        return (*singles,*contrasts)

    @staticmethod
    def _unique_inquiry_bid(candidates):
        """Return one causally unique winning bid; identical bytes never merge needs."""
        candidates=tuple(candidates)
        if not candidates:return None
        best=max(int(row[1]) for row in candidates);leaders=tuple(row for row in candidates if int(row[1])==best)
        return leaders[0] if len(leaders)==1 else None

    def endogenous_causal_repair_inquiry(self,channel=0):
        """Rematerialize one specific inquiry from a disputed certified binding."""
        winner=self._unique_inquiry_bid(self._causal_repair_inquiry_candidates(channel))
        return None if winner is None else bytes(winner[0])

    def _endogenous_inquiry_candidates(self,channel=0):
        state_surface,state_bid=self._endogenous_state_inquiry_candidate(channel);candidates=[]
        if state_surface:candidates.append((bytes(state_surface),int(state_bid),'state',(),0))
        candidates.extend((bytes(surface),int(bid),'causal',tuple(map(int,bound)),int(effect))
                          for surface,bid,bound,effect in self._causal_inquiry_candidates(channel))
        return tuple(candidates)

    def _endogenous_inquiry_role(self,receipt):
        """Rematerialize the local competition role from receipt coordinates."""
        if self.lexical_hypothesis_for_inquiry(receipt) is not None:return 'lexical'
        return 'causal' if receipt.obligation_candidates or receipt.obligation_effect else 'state'

    def _pending_endogenous_inquiry_roles(self,channel=0):
        channel=max(0,int(channel))
        return {self._endogenous_inquiry_role(receipt) for receipt in
                self.pending_endogenous_inquiry_actions.values()
                if int(receipt.channel)==channel}

    def _available_endogenous_inquiry_candidates(self,channel=0):
        pending=self._pending_endogenous_inquiry_roles(channel)
        return tuple(row for row in self._endogenous_inquiry_candidates(channel)
                     if row[2] not in pending)

    def endogenous_inquiry_work_pending(self,channel=0):
        """Report resident inquiry readiness without granting transport authority."""
        channel=max(0,int(channel))
        return self._unique_inquiry_bid(
            self._available_endogenous_inquiry_candidates(channel)) is not None

    def externalize_endogenous_inquiry(self,source,channel=0):
        """Publish one resident-originated inquiry while transport remains content-free."""
        source=int(source);channel=max(0,int(channel))
        if source<=0:return b'',None
        winner=self._unique_inquiry_bid(
            self._available_endogenous_inquiry_candidates(channel))
        if winner is None:return b'',None
        surface=bytes(winner[0]);obligation_candidates=tuple(map(int,winner[3])) if len(winner)>4 and winner[2]=='causal' else ();obligation_effect=int(winner[4]) if len(winner)>4 and winner[2]=='causal' else 0
        programs=self._longest_causal_program_component()
        if not programs:return b'',None
        context=self._causal_dialogue_appraisal_context(programs,channel);born=int(self.language_adult._advance())
        digest=hashlib.sha256(bytes(surface)).hexdigest();identity=_identity('endogenous-inquiry-action-receipt-v1',(context,digest,source,channel,born,obligation_candidates,obligation_effect))
        if identity in self.pending_endogenous_inquiry_actions or len(self.pending_endogenous_inquiry_actions)>=8:return b'',None
        receipt=EndogenousInquiryActionReceiptV1(context,digest,source,channel,born,identity,obligation_candidates,obligation_effect)
        self.pending_endogenous_inquiry_actions[identity]=receipt;self.endogenous_inquiry_public_count+=1
        return bytes(surface),receipt

    def _lexical_clarification_wrappers(self,child_context,child_surface):
        """Rematerialize unary wrappers learned specifically around this child kind."""
        child_context=int(child_context);child_surface=tuple(map(int,child_surface));rows=[]
        for (context,arity,pieces),sources in self.language._span_sources.items():
            if int(arity)!=1:continue
            live={int(source) for source in sources if source not in self.language._withdrawn}
            if len(live)<self.language.minimum_source_support:continue
            ports=[index for index,piece in enumerate(pieces)
                   if piece.kind==PIECE_PORT and int(piece.port)==0]
            if len(ports)!=1 or any(piece.kind not in (PIECE_LITERAL,PIECE_PORT)
                                    for piece in pieces):continue
            port=ports[0]
            prefix=tuple(value for piece in pieces[:port]
                         for value in piece.literal) if all(
                             piece.kind==PIECE_LITERAL for piece in pieces[:port]) else ()
            suffix=tuple(value for piece in pieces[port+1:]
                         for value in piece.literal) if all(
                             piece.kind==PIECE_LITERAL for piece in pieces[port+1:]) else ()
            expected=_language_identity('adult-receptive-wrapper-v1',(
                1,bool(prefix),prefix,bool(suffix),suffix))
            if int(context)!=int(expected):continue
            try:surface=tuple(self.language_adult._render_pieces(
                pieces,(child_surface,)))
            except Exception:continue
            if surface:rows.append((len(live),int(context),surface))
        return tuple(sorted(rows,key=lambda row:(-row[0],row[1],row[2])))

    def externalize_lexeme_clarification(self,raw,source,channel=0):
        """Retain a bounded ambiguous closure and ask through learned structure."""
        raw=tuple(map(int,raw));source=int(source);channel=max(0,int(channel))
        rows=self.language.provisional_dependency_alias_candidates(raw,source)
        if (len(rows)<2 or source<=0
                or 'lexical' in self._pending_endogenous_inquiry_roles(channel)):
            return b'',None
        shaped={(int(row[2]),int(row[3]),tuple(row[1])) for row in rows}
        if len(shaped)!=1:return b'',None
        candidates=[]
        for feature,units,context,target_slot,bindings,binding_units in rows:
            bindings=tuple(map(int,bindings));binding_units=tuple(binding_units)
            if not 0<=int(target_slot)<len(bindings):continue
            key=(int(feature),tuple(units))
            positive=len(self.language._lexeme_positive.get(key,()))
            counter=len(self.language._lexeme_counter.get(key,()))
            if counter>positive:continue
            surfaces=[];support=0
            for slot,atom in enumerate(bindings):
                if slot==int(target_slot):surfaces.append(tuple(units));continue
                held=binding_units[slot]
                if atom<=0 or held is None:surfaces=[];break
                held=tuple(held);identified=self.language.lexeme(atom)
                if identified is None:surfaces=[];break
                surfaces.append(tuple(identified))
                support+=self.language._active_count(
                    self.language._lexeme_sources.get((int(atom),held),()))
            template=self.language.template(int(context),len(bindings))
            child=(self.language.render_template(template,tuple(surfaces))
                   if template is not None and surfaces else None)
            wrappers=self._lexical_clarification_wrappers(int(context),child or ())
            if not wrappers:continue
            best_wrapper=wrappers[0][0]
            leaders=tuple(row for row in wrappers if row[0]==best_wrapper)
            if len(leaders)!=1:continue
            candidates.append((support,int(feature),tuple(units),leaders[0][2]))
        if not candidates:return b'',None
        best=max(row[0] for row in candidates);leaders=[row for row in candidates if row[0]==best]
        if len(leaders)!=1:return b'',None
        for _support,feature,units,_surface in candidates:
            self.language.observe_naming(feature,units,source)
        _support,feature,units,surface=leaders[0]
        hypothesis=self.language.lexeme_identity(feature,units)
        born=int(self.language_adult._advance());digest=hashlib.sha256(bytes(surface)).hexdigest()
        identity=_identity('endogenous-inquiry-action-receipt-v1',(
            hypothesis,digest,source,channel,born,(hypothesis,),feature))
        receipt=EndogenousInquiryActionReceiptV1(
            hypothesis,digest,source,channel,born,identity,(hypothesis,),feature)
        self.pending_endogenous_inquiry_actions[identity]=receipt
        self.endogenous_inquiry_public_count+=1
        return bytes(surface),receipt

    def externalize_lexeme_hypothesis(self,hypothesis_identity,source,channel=0):
        """Publish one still-losing resident lexical hypothesis for consequence."""
        hypothesis_identity=int(hypothesis_identity);source=int(source);channel=max(0,int(channel))
        row=self.language.lexeme_hypothesis_identity(hypothesis_identity,source)
        if (row is None or source<=0
                or 'lexical' in self._pending_endogenous_inquiry_roles(channel)):
            return b'',None
        feature,units,_evidence_source=row;surface=bytes(units);born=int(self.language_adult._advance())
        candidates=(hypothesis_identity,);context=hypothesis_identity
        digest=hashlib.sha256(surface).hexdigest()
        identity=_identity('endogenous-inquiry-action-receipt-v1',(
            context,digest,source,channel,born,candidates,int(feature)))
        if identity in self.pending_endogenous_inquiry_actions:return b'',None
        receipt=EndogenousInquiryActionReceiptV1(
            context,digest,source,channel,born,identity,candidates,int(feature))
        self.pending_endogenous_inquiry_actions[identity]=receipt
        self.endogenous_inquiry_public_count+=1
        return surface,receipt

    def lexical_hypothesis_for_inquiry(self,receipt):
        if not isinstance(receipt,EndogenousInquiryActionReceiptV1) or len(receipt.obligation_candidates)!=1:return None
        row=self.language.lexeme_hypothesis_identity(
            int(receipt.obligation_candidates[0]),int(receipt.source))
        return (row if row is not None and int(row[0])==int(receipt.obligation_effect)
                else None)

    def settle_lexeme_hypothesis_return(self,receipt,source,effect,independent=True):
        """Let only the exact public hypothesis action qualify its relation."""
        source=int(source);effect=int(effect);row=self.lexical_hypothesis_for_inquiry(receipt)
        if (row is None or source<=0 or effect==0 or not independent
                or int(receipt.identity) not in self.pending_endogenous_inquiry_actions
                ):return False
        handled,_result=self.language.settle_lexeme_identity(
            int(receipt.obligation_candidates[0]),int(receipt.source),
            int(receipt.identity),effect)
        if not handled:return False
        self.language_adult.refresh_lexeme_identity(
            int(receipt.obligation_candidates[0]))
        self.pending_endogenous_inquiry_actions.pop(int(receipt.identity),None)
        self.reafferenced_endogenous_inquiry_actions.discard(int(receipt.identity))
        self.language_adult._advance();return True

    def settle_endogenous_inquiry_motor_return(self,receipt,source,succeeded=True):
        """Record public execution without pretending it answered the inquiry."""
        source=int(source);succeeded=bool(succeeded)
        if (not isinstance(receipt,EndogenousInquiryActionReceiptV1)
                or int(receipt.identity) not in self.pending_endogenous_inquiry_actions
                or source<=0 or source!=int(receipt.channel)):
            return False
        if succeeded:
            if int(receipt.identity) in self.reafferenced_endogenous_inquiry_actions:
                return False
            self.reafferenced_endogenous_inquiry_actions.add(int(receipt.identity))
        else:
            self.pending_endogenous_inquiry_actions.pop(int(receipt.identity),None)
            self.reafferenced_endogenous_inquiry_actions.discard(int(receipt.identity))
        self.language_adult._advance();return True

    def settle_reafferenced_lexeme_hypothesis_contact(self,raw,source,channel=0):
        """Let learned referent structure, not answer words, qualify one inquiry."""
        raw=tuple(map(int,raw));source=int(source);channel=max(0,int(channel))
        pending=tuple(receipt for receipt in self.pending_endogenous_inquiry_actions.values()
                      if int(receipt.channel)==channel
                      and int(receipt.identity) in self.reafferenced_endogenous_inquiry_actions
                      and self.lexical_hypothesis_for_inquiry(receipt) is not None)
        if len(pending)!=1 or not raw or source<=0:return False
        try:rows=self.language.provisional_dependency_alias_candidates(raw,source)
        except ValueError:return False
        features={int(row[0]) for row in rows}
        if len(features)!=1:return False
        receipt=pending[0];observed=next(iter(features))
        effect=1 if observed==int(receipt.obligation_effect) else -1
        return self.settle_lexeme_hypothesis_return(receipt,source,effect,True)

    def _endogenous_inquiry_need_remains(self,receipt):
        """Rematerialize whether this exact prospective information need survives."""
        role=self._endogenous_inquiry_role(receipt);channel=int(receipt.channel)
        if role=='lexical':return True
        if role=='state':
            surface,_bid=self._endogenous_state_inquiry_candidate(channel)
            return bool(surface and hashlib.sha256(bytes(surface)).hexdigest()==receipt.surface_digest)
        for _surface,_bid,_role,candidates,effect in self._endogenous_inquiry_candidates(channel):
            if _role!='causal' or not self.language_adult.leaf_equivalent(
                    int(effect),int(receipt.obligation_effect)):continue
            unmatched=list(map(int,candidates))
            for bound in receipt.obligation_candidates:
                match=next((candidate for candidate in unmatched
                            if self.language_adult.leaf_equivalent(
                                int(candidate),int(bound))),None)
                if match is None:break
                unmatched.remove(match)
            else:
                if not unmatched:return True
        return False

    def settle_endogenous_inquiry_resolution(self,receipt,source):
        """Deactivate only a motor-realized inquiry whose resident need has changed."""
        if (not isinstance(receipt,EndogenousInquiryActionReceiptV1)
                or receipt.identity not in self.pending_endogenous_inquiry_actions
                or int(receipt.identity) not in self.reafferenced_endogenous_inquiry_actions
                or int(source)<=0 or self._endogenous_inquiry_need_remains(receipt)):
            return False
        self.pending_endogenous_inquiry_actions.pop(receipt.identity,None)
        self.reafferenced_endogenous_inquiry_actions.discard(int(receipt.identity))
        self.language_adult._advance();return True

    def current_context_surface(self,target_context,channel=0):
        target_context=int(target_context);channel=max(0,int(channel))
        if target_context==self.language_adult._somatic_appraisal_language_context():
            programs=self._longest_causal_program_component()
            if not programs:return None
            context=self._causal_dialogue_appraisal_context(programs,channel);credited=self._causal_expression_credit_program(programs[0],context)
            felt=self.language_adult.somatic_appraisal(credited or programs[0],context)
            return self.language.realize(target_context,self.language_adult.somatic_appraisal_atoms(felt))
        return None
    def respond_context_affordance(self,prompt_context,prompt_atoms,channel=0):
        target=self.context_affordance(prompt_context,prompt_atoms)
        return None if target<=0 else self.current_context_surface(target,channel)

    def contact_affordance_candidates(self,prompt_context,prompt_atoms,leaf_identity,channel=0,transient_resolved=()):
        """Rematerialize lawful public trajectories without letting the body name one."""
        prompt_context=int(prompt_context);prompt_atoms=tuple(map(int,prompt_atoms));leaf_identity=int(leaf_identity);channel=max(0,int(channel))
        candidates=[];target=self.context_affordance(prompt_context,prompt_atoms)
        context_surface=bytes(self.current_context_surface(target,channel) or b'') if target>0 else b''
        if context_surface:
            space=_identity('workbench-relation-language-space-v1',(prompt_context,prompt_atoms))
            candidates.append(('context',context_surface,space,target,(),()))
        causal_surface,programs,factors=self.compose_causal_component(
            leaf_identity,with_expression_factors=True,channel=channel,
            transient_resolved=transient_resolved)
        if causal_surface:
            candidates.append(('causal',bytes(causal_surface),leaf_identity,0,tuple(programs),tuple(factors)))
        return tuple(candidates)

    def stage_context_affordance_action(self,prompt_space,target_context,surface,source,channel=0):
        prompt_space=int(prompt_space);target_context=int(target_context);surface=bytes(surface);source=int(source);channel=max(0,int(channel))
        binding=self._relation_language_spaces.get(prompt_space)
        if binding is None or self.context_affordance(*binding)!=target_context or bytes(self.current_context_surface(target_context,channel) or b'')!=surface or not surface or source<=0:return None
        born=int(self.language_adult._advance());digest=hashlib.sha256(surface).hexdigest()
        identity=_identity('context-affordance-action-receipt-v1',(prompt_space,target_context,digest,source,channel,born))
        if identity in self.pending_context_affordance_actions or len(self.pending_context_affordance_actions)>=8:return None
        receipt=ContextAffordanceActionReceiptV1(prompt_space,target_context,digest,source,channel,born,identity)
        self.pending_context_affordance_actions[identity]=receipt;self.context_affordance_public_count+=1
        return receipt

    def externalize_contact_affordance(self,prompt_context,prompt_atoms,leaf_identity,source,channel=0,transient_resolved=()):
        candidates=self.contact_affordance_candidates(
            prompt_context,prompt_atoms,leaf_identity,channel,transient_resolved)
        # One longer motor trajectory may already fulfill a shorter candidate in
        # full. Remove only exact suffix-contained trajectories; incomparable
        # alternatives remain a real tie and therefore remain silent.
        leaders=tuple(row for row in candidates if not any(
            other[1]!=row[1] and bytes(other[1]).endswith(bytes(row[1]))
            for other in candidates))
        if len(leaders)!=1:return b'',None
        kind,surface,first,second,_programs,_factors=leaders[0]
        if kind=='causal':
            return self.externalize_causal_component(
                leaf_identity,source,channel,transient_resolved)
        receipt=self.stage_context_affordance_action(first,second,surface,source,channel)
        return (surface,receipt) if receipt is not None else (b'',None)

    def settle_context_affordance_return(self,receipt,source):
        if not isinstance(receipt,ContextAffordanceActionReceiptV1) or receipt.identity not in self.pending_context_affordance_actions or int(source)!=int(receipt.source):return False
        self.pending_context_affordance_actions.pop(receipt.identity,None);self.language_adult._advance();return True

    def checkpoint(self):
        self._reconcile_causal_action_lineage()
        bindings=[]
        for node in sorted(self._operator_bindings):
            for (context,atoms),sources in sorted(self._operator_bindings[node].items()):
                bindings.append({'node':node,'context':context,'atoms':list(atoms),'sources':sorted(sources)})
        edge_language=[{'relation':rid,'context':context,'sources':sorted(sources)} for rid,rows in sorted(self._relation_edge_language_contexts.items()) for context,sources in sorted(rows.items())]
        return {
            'schema':25,'language_adult':self.language_adult.checkpoint(),
            'language_action_affordances':self.language_action_affordances.checkpoint(),
            'operators':self.operators.checkpoint(),'relation_basis':self.relation_basis.checkpoint(),
            'relation_language_spaces':[{'identity':sid,'context':row[0],'atoms':list(row[1])} for sid,row in sorted(self._relation_language_spaces.items())],
            'relation_edge_language_contexts':edge_language,'operator_bindings':bindings,
            'operator_binding_withdrawn':sorted(self._operator_binding_withdrawn),
            'operator_join_context':self._operator_join_context,'operator_join_sources':sorted(self._operator_join_sources),
            'operator_externalization_enabled':bool(self.operator_externalization_enabled),
            'pending_operator_actions':[{'root_identity':r.root_identity,'witness_identity':r.witness_identity,'trace_digest':r.trace_digest,'state_revision':r.state_revision,'identity':r.identity} for r in sorted(self.pending_operator_actions.values(),key=lambda row:row.identity)],
            'pending_relation_actions':[{'relation_identity':r.relation_identity,'ancestry_digest':r.ancestry_digest,'basis_revision':r.basis_revision,'identity':r.identity} for r in sorted(self.pending_relation_actions.values(),key=lambda row:row.identity)],
            'operator_public_count':self.operator_public_count,'relation_public_count':self.relation_public_count,
            'causal_dialogue_public_count':self.causal_dialogue_public_count,
            'causal_dialogue_context_by_channel':[{'channel':channel,'context':context} for channel,context in sorted(self._causal_dialogue_context_by_channel.items())],
            'causal_dialogue_uptake_evidence':[{'channel':channel,'causal_receipt':receipt,'source':source,'evidence':evidence} for (channel,receipt,source),evidence in sorted(self._causal_dialogue_uptake_evidence.items())],
            'causal_dialogue_dispute_evidence':[{'channel':channel,'causal_receipt':receipt,'emitted_cause':emitted,'candidate_cause':candidate,'effect':effect,'source':source,'evidence':evidence} for (channel,receipt,emitted,candidate,effect,source),evidence in sorted(self._causal_dialogue_dispute_evidence.items())],
            'causal_dialogue_formulation_evidence':[{'channel':channel,'factor':factor,'source':source,'evidence':evidence} for (channel,factor,source),evidence in sorted(self._causal_dialogue_formulation_evidence.items())],
            'settled_causal_action_lineage':[{'channel':channel,'episode':episode,'causal_receipt':receipt,'cause':cause,'effect':effect,'settled_tick':tick} for (channel,receipt,cause,effect),(episode,tick) in sorted(self._settled_causal_action_lineage.items())],
            'causal_continuation_commitments':[{'source':r.source,'channel':r.channel,'support_context':r.support_context,'support_arity':r.support_arity,'support_template':r.support_template,'focus':r.focus,'resolved_receipt':r.resolved_receipt,'awaiting_action':r.awaiting_action,'ready':bool(r.ready)} for r in sorted(self._causal_continuation_commitments.values(),key=lambda row:row.channel)],
            'pending_causal_dialogue_actions':[{'programs':list(r.programs),'factors':list(r.factors),'context':r.context,'surface_digest':r.surface_digest,'source':r.source,'channel':r.channel,'episode':r.episode,'born_tick':r.born_tick,'identity':r.identity} for r in sorted(self.pending_causal_dialogue_actions.values(),key=lambda row:row.identity)],
            'recent_causal_dialogue_actions':[{'programs':list(r.programs),'factors':list(r.factors),'context':r.context,'surface_digest':r.surface_digest,'source':r.source,'channel':r.channel,'episode':r.episode,'born_tick':r.born_tick,'identity':r.identity} for r in sorted(self.recent_causal_dialogue_actions.values(),key=lambda row:row.identity)],
            'pending_causal_dialogue_consequences':sorted(self.pending_causal_dialogue_consequences),
            'endogenous_inquiry_public_count':self.endogenous_inquiry_public_count,
            'pending_endogenous_inquiry_actions':[{'context':r.context,'surface_digest':r.surface_digest,'source':r.source,'channel':r.channel,'born_tick':r.born_tick,'identity':r.identity,'obligation_candidates':list(r.obligation_candidates),'obligation_effect':r.obligation_effect} for r in sorted(self.pending_endogenous_inquiry_actions.values(),key=lambda row:row.identity)],
            'reafferenced_endogenous_inquiry_actions':sorted(self.reafferenced_endogenous_inquiry_actions),
            'context_affordance_public_count':self.context_affordance_public_count,
            'pending_context_affordance_actions':[{'prompt_space':r.prompt_space,'target_context':r.target_context,'surface_digest':r.surface_digest,'source':r.source,'channel':r.channel,'born_tick':r.born_tick,'identity':r.identity} for r in sorted(self.pending_context_affordance_actions.values(),key=lambda row:row.identity)],
            'context_affordances':[{'prompt_space':space,'target_context':target,'sources':sorted(sources)} for (space,target),sources in sorted(self._context_affordance_sources.items())]}
    @classmethod
    def restore(cls,data):
        schema=int(data.get('schema',0))
        if schema not in (1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25):raise ValueError('mathematical-workbench-adult:checkpoint-schema')
        language_adult=LanguageMasteryAdultV1.restore(copy.deepcopy(data['language_adult']))
        basis=RecursiveRelationBasisV1.restore(copy.deepcopy(data['relation_basis'])) if schema>=4 and data.get('relation_basis') is not None else RecursiveRelationBasisV1()
        out=cls(language_adult,ResidentProgramStateV1.restore(copy.deepcopy(data['operators'])),basis)
        if schema>=25:
            out.language_action_affordances=OpenLanguageActionAffordanceV1.restore(copy.deepcopy(data['language_action_affordances']))
        if schema>=6:
            for row in data.get('relation_language_spaces',()):
                sid=int(row['identity']);binding=(int(row['context']),tuple(map(int,row['atoms'])))
                expected=_identity('workbench-relation-language-space-v1',binding)
                if sid<=0 or sid!=expected or sid in out._relation_language_spaces:raise ValueError('mathematical-workbench-adult:relation-space')
                out._relation_language_spaces[sid]=binding
        if schema>=7:
            for row in data.get('relation_edge_language_contexts',()):
                rid=int(row.get('relation',0));context=int(row.get('context',0));sources=set(map(int,row.get('sources',())))
                relation=out.relation_basis.relations.get(rid)
                if relation is None or context<=0 or not sources or any(source<=0 for source in sources):raise ValueError('mathematical-workbench-adult:relation-edge-language')
                out._relation_edge_language_contexts.setdefault(rid,{})[context]=sources
        for row in data.get('operator_bindings',()):
            for source in row.get('sources',()):out.observe_operator_binding(row['node'],row['context'],row['atoms'],source)
        out._operator_binding_withdrawn=set(map(int,data.get('operator_binding_withdrawn',())))
        out._operator_join_context=int(data.get('operator_join_context',0));out._operator_join_sources=set(map(int,data.get('operator_join_sources',())))
        out.operator_externalization_enabled=bool(data.get('operator_externalization_enabled',True));out.pending_operator_actions={}
        for row in data.get('pending_operator_actions',()):
            receipt=PublicActionReceiptV1(int(row['root_identity']),int(row['witness_identity']),str(row['trace_digest']),int(row['state_revision']),int(row['identity']));out.pending_operator_actions[receipt.identity]=receipt
        if schema>=4:
            for row in data.get('pending_relation_actions',()):
                receipt=RelationBasisPublicReceiptV1(int(row['relation_identity']),str(row['ancestry_digest']),int(row['basis_revision']),int(row['identity']));out.pending_relation_actions[receipt.identity]=receipt
        out.operator_public_count=int(data.get('operator_public_count',0));out.relation_public_count=int(data.get('relation_public_count',0));out.causal_dialogue_public_count=int(data.get('causal_dialogue_public_count',0))
        if schema>=8:
            def restore_dialogue(rows):
                bank={}
                for row in rows:
                    programs=tuple(map(int,row.get('programs',())));factors=(tuple(map(int,row.get('factors',()))) if schema>=13 else tuple(int(out.language_adult.programs.factor(pid) or 0) for pid in programs));context=int(row.get('context',0));digest=str(row.get('surface_digest',''));source=int(row.get('source',0));channel=int(row.get('channel',0));born=int(row.get('born_tick',0));stored_identity=int(row.get('identity',0));episode=int(row.get('episode',0)) if schema>=15 else stored_identity
                    old_expected=_identity('causal-dialogue-action-receipt-v1',(programs,context,digest,source,channel,born));v2_expected=_identity('causal-dialogue-action-receipt-v2',(programs,factors,context,digest,source,channel,born))
                    expected=(v2_expected if schema>=13 else old_expected)
                    if not programs or any(out.language_adult.programs.factor(pid) is None for pid in programs) or not factors or any(factor<=0 or out.language.historical_span_pieces(factor) is None for factor in factors) or context<=0 or len(digest)!=64 or source<=0 or channel<0 or episode<=0 or born<=0 or stored_identity!=expected or stored_identity in bank:raise ValueError('mathematical-workbench-adult:causal-dialogue-action')
                    bank[stored_identity]=CausalDialogueActionReceiptV1(programs,factors,context,digest,source,channel,episode,born,stored_identity)
                if len(bank)>8:raise ValueError('mathematical-workbench-adult:causal-dialogue-capacity')
                return bank
            out.pending_causal_dialogue_actions=restore_dialogue(data.get('pending_causal_dialogue_actions',()))
            out.recent_causal_dialogue_actions=restore_dialogue(data.get('recent_causal_dialogue_actions',()))
            if set(out.pending_causal_dialogue_actions)&set(out.recent_causal_dialogue_actions):raise ValueError('mathematical-workbench-adult:causal-dialogue-overlap')
            if schema>=21:
                pending=tuple(map(int,data.get('pending_causal_dialogue_consequences',())))
                if (len(pending)!=len(set(pending)) or any(identity not in out.recent_causal_dialogue_actions for identity in pending)):
                    raise ValueError('mathematical-workbench-adult:causal-dialogue-pending-consequence')
                out.pending_causal_dialogue_consequences=set(pending)
        if schema>=10:
            for row in data.get('causal_dialogue_context_by_channel',()):
                channel=int(row.get('channel',0));context=int(row.get('context',0))
                if channel<=0 or context<=0 or channel in out._causal_dialogue_context_by_channel:raise ValueError('mathematical-workbench-adult:causal-dialogue-channel-context')
                out._causal_dialogue_context_by_channel[channel]=context
        elif schema>=8:
            for receipt in sorted((*out.pending_causal_dialogue_actions.values(),*out.recent_causal_dialogue_actions.values()),key=lambda row:(row.born_tick,row.identity)):
                if receipt.channel>0:out._causal_dialogue_context_by_channel[int(receipt.channel)]=int(receipt.context)
        if schema>=9:
            for row in data.get('context_affordances',()):
                space=int(row.get('prompt_space',0));target=int(row.get('target_context',0));sources=set(map(int,row.get('sources',())))
                if space not in out._relation_language_spaces or target<=0 or not sources or any(source<=0 for source in sources):raise ValueError('mathematical-workbench-adult:context-affordance')
                key=(space,target)
                if key in out._context_affordance_sources:raise ValueError('mathematical-workbench-adult:context-affordance-duplicate')
                out._context_affordance_sources[key]=sources
        if schema>=12:
            for row in data.get('causal_dialogue_uptake_evidence',()):
                channel=int(row.get('channel',0));receipt=int(row.get('causal_receipt',0));source=int(row.get('source',0));evidence=int(row.get('evidence',0));key=(channel,receipt,source)
                if min(channel,receipt,source,evidence)<=0 or evidence>MAX_CAUSAL_DIALOGUE_UPTAKE_EVIDENCE or key in out._causal_dialogue_uptake_evidence:raise ValueError('mathematical-workbench-adult:causal-dialogue-uptake')
                out._causal_dialogue_uptake_evidence[key]=evidence
            if len(out._causal_dialogue_uptake_evidence)>MAX_CAUSAL_DIALOGUE_UPTAKE_ROWS:raise ValueError('mathematical-workbench-adult:causal-dialogue-uptake-capacity')
        if schema>=14:
            for row in data.get('causal_dialogue_dispute_evidence',()):
                channel=int(row.get('channel',0));receipt=int(row.get('causal_receipt',0));emitted=int(row.get('emitted_cause',0));candidate=int(row.get('candidate_cause',0));effect=int(row.get('effect',0));source=int(row.get('source',0));evidence=int(row.get('evidence',0));key=(channel,receipt,emitted,candidate,effect,source)
                if schema<16 or min(channel,receipt,emitted,candidate,effect,source,evidence)<=0 or evidence>MAX_CAUSAL_DIALOGUE_UPTAKE_EVIDENCE or key in out._causal_dialogue_dispute_evidence:raise ValueError('mathematical-workbench-adult:causal-dialogue-dispute')
                out._causal_dialogue_dispute_evidence[key]=evidence
            if len(out._causal_dialogue_dispute_evidence)>MAX_CAUSAL_DIALOGUE_UPTAKE_ROWS:raise ValueError('mathematical-workbench-adult:causal-dialogue-dispute-capacity')
        if schema>=24:
            for row in data.get('causal_dialogue_formulation_evidence',()):
                channel=int(row.get('channel',0));factor=int(row.get('factor',0));source=int(row.get('source',0));evidence=int(row.get('evidence',0));key=(channel,factor,source)
                if (min(channel,factor,source)<=0 or evidence==0
                        or abs(evidence)>MAX_CAUSAL_DIALOGUE_UPTAKE_EVIDENCE
                        or key in out._causal_dialogue_formulation_evidence
                        or out.language.historical_span_pieces(factor) is None):
                    raise ValueError('mathematical-workbench-adult:causal-dialogue-formulation')
                out._causal_dialogue_formulation_evidence[key]=evidence
            if len(out._causal_dialogue_formulation_evidence)>MAX_CAUSAL_DIALOGUE_UPTAKE_ROWS:raise ValueError('mathematical-workbench-adult:causal-dialogue-formulation-capacity')
        if schema>=17:
            for row in data.get('settled_causal_action_lineage',()):
                channel=int(row.get('channel',0));episode=int(row.get('episode',0));receipt=int(row.get('causal_receipt',0));cause=int(row.get('cause',0));effect=int(row.get('effect',0));tick=int(row.get('settled_tick',0));key=(channel,receipt,cause,effect)
                if min(channel,episode,receipt,cause,effect,tick)<=0 or key in out._settled_causal_action_lineage or receipt not in out.language_adult.world_causal_learning.bindings:raise ValueError('mathematical-workbench-adult:settled-causal-action-lineage')
                out._settled_causal_action_lineage[key]=(episode,tick);out._settled_causal_action_lineage_index.setdefault(channel,set()).add(key[1:])
            if len(out._settled_causal_action_lineage)>MAX_SETTLED_CAUSAL_ACTION_LINEAGE_ROWS:raise ValueError('mathematical-workbench-adult:settled-causal-action-lineage-capacity')
        if schema>=22:
            for row in data.get('causal_continuation_commitments',()):
                source=int(row.get('source',0));channel=int(row.get('channel',0));context=int(row.get('support_context',0));arity=int(row.get('support_arity',0));template=int(row.get('support_template',0));focus=int(row.get('focus',0));resolved=int(row.get('resolved_receipt',0));awaiting=int(row.get('awaiting_action',0));ready=bool(row.get('ready',False))
                commitment=CausalContinuationCommitmentV1(source,channel,context,arity,template,focus,resolved,awaiting,ready)
                actions={**out.pending_causal_dialogue_actions,**out.recent_causal_dialogue_actions}
                if (min(source,channel,context,arity,template,focus,resolved)<=0
                        or channel in out._causal_continuation_commitments
                        or out.language.historical_span_pieces(template) is None
                        or resolved not in out.language_adult.world_causal_learning.bindings
                        or ready!=(awaiting==0) or awaiting and awaiting not in actions):
                    raise ValueError('mathematical-workbench-adult:causal-continuation-commitment')
                out._causal_continuation_commitments[channel]=commitment
            if len(out._causal_continuation_commitments)>8:raise ValueError('mathematical-workbench-adult:causal-continuation-capacity')
        if schema>=11:
            out.endogenous_inquiry_public_count=int(data.get('endogenous_inquiry_public_count',0))
            if out.endogenous_inquiry_public_count<0:raise ValueError('mathematical-workbench-adult:endogenous-inquiry-count')
            for row in data.get('pending_endogenous_inquiry_actions',()):
                context=int(row.get('context',0));digest=str(row.get('surface_digest',''));source=int(row.get('source',0));channel=int(row.get('channel',0));born=int(row.get('born_tick',0));identity=int(row.get('identity',0))
                candidates=(tuple(map(int,row.get('obligation_candidates',()))) if schema>=20 else ((int(row.get('obligation_candidate',0)),) if schema>=18 and int(row.get('obligation_candidate',0))>0 else ()));effect=int(row.get('obligation_effect',0)) if schema>=18 else 0
                expected=(_identity('endogenous-inquiry-action-receipt-v1',(context,digest,source,channel,born,candidates,effect)) if schema>=20 else (_identity('endogenous-inquiry-action-receipt-v1',(context,digest,source,channel,born,candidates[0],effect)) if schema>=18 else _identity('endogenous-inquiry-action-receipt-v1',(context,digest,source,channel,born))))
                obligation_valid=(not candidates and effect==0) or (0<len(candidates)<=2 and effect>0 and len(set(candidates))==len(candidates) and all(candidate>0 and candidate!=effect for candidate in candidates))
                if context<=0 or len(digest)!=64 or source<=0 or channel<0 or born<=0 or not obligation_valid or identity!=expected or identity in out.pending_endogenous_inquiry_actions:raise ValueError('mathematical-workbench-adult:endogenous-inquiry-action')
                out.pending_endogenous_inquiry_actions[identity]=EndogenousInquiryActionReceiptV1(context,digest,source,channel,born,identity,candidates,effect)
            if schema>=23:
                out.reafferenced_endogenous_inquiry_actions=set(map(
                    int,data.get('reafferenced_endogenous_inquiry_actions',())))
                if (not out.reafferenced_endogenous_inquiry_actions.issubset(
                        out.pending_endogenous_inquiry_actions)
                        or len(out.reafferenced_endogenous_inquiry_actions)>8):
                    raise ValueError('mathematical-workbench-adult:endogenous-inquiry-reafference')
            inquiry_keys={(int(receipt.channel),out._endogenous_inquiry_role(receipt))
                          for receipt in out.pending_endogenous_inquiry_actions.values()}
            if (len(out.pending_endogenous_inquiry_actions)>8
                    or len(inquiry_keys)!=len(out.pending_endogenous_inquiry_actions)):
                raise ValueError('mathematical-workbench-adult:endogenous-inquiry-capacity')
        if schema>=19:
            out.context_affordance_public_count=int(data.get('context_affordance_public_count',0))
            if out.context_affordance_public_count<0:raise ValueError('mathematical-workbench-adult:context-affordance-count')
            for row in data.get('pending_context_affordance_actions',()):
                space=int(row.get('prompt_space',0));target=int(row.get('target_context',0));digest=str(row.get('surface_digest',''));source=int(row.get('source',0));channel=int(row.get('channel',0));born=int(row.get('born_tick',0));identity=int(row.get('identity',0))
                expected=_identity('context-affordance-action-receipt-v1',(space,target,digest,source,channel,born))
                binding=out._relation_language_spaces.get(space)
                if binding is None or out.context_affordance(*binding)!=target or len(digest)!=64 or source<=0 or channel<0 or born<=0 or identity!=expected or identity in out.pending_context_affordance_actions:raise ValueError('mathematical-workbench-adult:context-affordance-action')
                out.pending_context_affordance_actions[identity]=ContextAffordanceActionReceiptV1(space,target,digest,source,channel,born,identity)
            if len(out.pending_context_affordance_actions)>8 or len({r.channel for r in out.pending_context_affordance_actions.values()})!=len(out.pending_context_affordance_actions):raise ValueError('mathematical-workbench-adult:context-affordance-action-capacity')
        out.reset_operator_transient();return out
    def digest(self):return hashlib.sha256(b'mathematical-workbench-adult-v1\0'+json.dumps(self.checkpoint(),sort_keys=True,separators=(',',':')).encode()).hexdigest()
