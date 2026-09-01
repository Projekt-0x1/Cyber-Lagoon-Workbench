#!/usr/bin/env python3
"""Resident hierarchical expression cursor: structural plan -> one motor byte at a time."""
from __future__ import annotations
from dataclasses import asdict,dataclass
from types import SimpleNamespace
import hashlib,json
from reference_hierarchical_composition_v1 import (
    HierarchicalConstructionV1,HierarchicalRefuse,TransientConstructionPlanV1,TransientSequencePlanV1,
    rematerialize_transient_plan,rematerialize_transient_sequence_plan,
)
from reference_language_learning_v1 import PIECE_LITERAL,PIECE_PORT


def _language(owner):
    return owner.language if hasattr(owner,'closure') else owner


def _hierarchy(owner):
    return owner if hasattr(owner,'closure') else None


def language_span_pieces(language, template_identity, context, arity):
    t=language.span_template(int(context),int(arity))
    if t is not None and int(t.identity[:15],16)==int(template_identity):return tuple(t.pieces)
    matches=tuple(t for t in language.span_candidates(int(context),int(arity))
                  if int(t.identity[:15],16)==int(template_identity))
    return tuple(matches[0].pieces) if len(matches)==1 else None

def _span_pieces_held(language, node, hold):
    tid=int(getattr(node,'template_identity',0) or 0)
    kids=getattr(node,'child_identities',())
    if not (tid and kids):return None
    key=(tid,int(node.context),len(kids))
    if hold is not None and key in hold:return hold[key]
    out=language_span_pieces(language,tid,node.context,len(kids))
    if hold is not None:hold[key]=out
    return out

class ExpressionRefuse(ValueError):pass

@dataclass
class ExpressionFrameV1:
    closure_identity:int
    piece_cursor:int=0

@dataclass(frozen=True)
class ExpressionBytePlanV1:
    identity:int
    root_identity:int
    segment_kind:int
    reference_identity:int
    piece_index:int
    byte_index:int
    value:int
    ordinal:int
    attempt:int

SEG_LEAF=1
SEG_LITERAL=2

class IncrementalExpressionV1:
    """No output transcript: only structural stack + current segment cursor."""
    def __init__(self,hierarchy:HierarchicalConstructionV1,root_identity:int,leaves=None):
        self.h=hierarchy;self.leaves=leaves;self.root_identity=int(root_identity)
        if self._child(self.root_identity) is None:raise ExpressionRefuse('expression:root')
        self.stack=[ExpressionFrameV1(self.root_identity,0)]
        self.segment_kind=0;self.reference_identity=0;self.piece_index=0;self.byte_index=0
        self.pending:ExpressionBytePlanV1|None=None
        self.ordinal=0;self.attempt=0;self.repairs=0;self.complete=False
        self.last_plan_touches=0;self.total_plan_touches=0;self.max_stack=1
        self._held_literal=None;self._frame_pieces=None

    def _child(self,identity):
        identity=int(identity)
        if self.h is not None:
            node=self.h.closure(identity)
            if node is not None:return node
        return None if self.leaves is None else self.leaves.lookup(identity)

    def _pieces(self,node):
        return _span_pieces_held(self.h.language,node,self._frame_pieces)

    def _segment_bytes(self):
        if self.segment_kind==SEG_LEAF:
            node=self._child(self.reference_identity)
            surface=() if node is None else (getattr(node,'_leaf_surface',()) or getattr(node,'surface',()))
            if node is None or int(getattr(node,'depth',-1))!=0 or not surface:raise ExpressionRefuse('expression:leaf')
            return tuple(surface)
        if self.segment_kind==SEG_LITERAL:
            if self._held_literal is not None:return self._held_literal
            owner=self._child(self.reference_identity)
            if owner is None or owner.depth==0:raise ExpressionRefuse('expression:template_withdrawn')
            pieces=self._pieces(owner)
            if pieces is None:raise ExpressionRefuse('expression:template_withdrawn')
            if not 0<=self.piece_index<len(pieces):raise ExpressionRefuse('expression:literal')
            piece=pieces[self.piece_index]
            if piece.kind!=PIECE_LITERAL or not piece.literal:raise ExpressionRefuse('expression:literal')
            self._held_literal=tuple(piece.literal);return self._held_literal
        raise ExpressionRefuse('expression:segment')

    def _ensure_segment(self):
        touches=0;self._frame_pieces={}
        try:
            while not self.complete and self.segment_kind==0:
                if not self.stack:self.complete=True;break
                frame=self.stack[-1];node=self._child(frame.closure_identity);touches+=1
                if node is None:raise ExpressionRefuse('expression:closure')
                if node.depth==0:
                    if frame.piece_cursor!=0:self.stack.pop();continue
                    frame.piece_cursor=1;self.segment_kind=SEG_LEAF;self.reference_identity=node.identity;self.piece_index=0;self.byte_index=0;self._held_literal=None;break
                pieces=self._pieces(node)
                if pieces is None:raise ExpressionRefuse('expression:template_withdrawn')
                if frame.piece_cursor>=len(pieces):self.stack.pop();continue
                index=frame.piece_cursor;frame.piece_cursor+=1;piece=pieces[index];touches+=1
                if piece.kind==PIECE_LITERAL:
                    if not piece.literal:continue
                    self.segment_kind=SEG_LITERAL;self.reference_identity=node.identity;self.piece_index=index;self.byte_index=0
                    self._held_literal=tuple(piece.literal);break
                if piece.kind==PIECE_PORT and 0<=piece.port<len(node.child_identities):
                    self.stack.append(ExpressionFrameV1(int(node.child_identities[piece.port]),0));self.max_stack=max(self.max_stack,len(self.stack));continue
                raise ExpressionRefuse('expression:piece')
        finally:
            self._frame_pieces=None
        self.last_plan_touches=touches;self.total_plan_touches+=touches

    @staticmethod
    def _plan_id(root,kind,reference,piece,byte,ordinal,attempt,value):
        body=json.dumps([root,kind,reference,piece,byte,ordinal,attempt,value],separators=(',',':')).encode()
        return int.from_bytes(hashlib.sha256(b'0x1-expression-byte-v1\0'+body).digest()[:8],'big') or 1

    def emit(self):
        if self.pending is not None:raise ExpressionRefuse('expression:pending_reafference')
        self._ensure_segment()
        if self.complete:return None
        segment=self._segment_bytes()
        if not 0<=self.byte_index<len(segment):raise ExpressionRefuse('expression:cursor')
        value=int(segment[self.byte_index]);self.attempt+=1
        plan=ExpressionBytePlanV1(self._plan_id(self.root_identity,self.segment_kind,self.reference_identity,self.piece_index,self.byte_index,self.ordinal,self.attempt,value),self.root_identity,self.segment_kind,self.reference_identity,self.piece_index,self.byte_index,value,self.ordinal,self.attempt)
        self.pending=plan;return plan

    def reafference(self,plan:ExpressionBytePlanV1,actual_value:int):
        if self.pending is None or plan!=self.pending:raise ExpressionRefuse('expression:reafference_identity')
        segment=self._segment_bytes();expected=segment[self.byte_index]
        if plan.value!=expected or plan.byte_index!=self.byte_index:raise ExpressionRefuse('expression:stale_plan')
        self.pending=None
        if int(actual_value)&255!=expected:
            self.repairs+=1;return False
        self.byte_index+=1;self.ordinal+=1;self.attempt=0
        if self.byte_index==len(segment):
            self.segment_kind=0;self.reference_identity=0;self.piece_index=0;self.byte_index=0;self._held_literal=None
        return True

    def checkpoint(self):
        return {'schema':1,'root_identity':self.root_identity,'stack':[asdict(x) for x in self.stack],'segment_kind':self.segment_kind,'reference_identity':self.reference_identity,'piece_index':self.piece_index,'byte_index':self.byte_index,'pending':None if self.pending is None else asdict(self.pending),'ordinal':self.ordinal,'attempt':self.attempt,'repairs':self.repairs,'complete':self.complete,'total_plan_touches':self.total_plan_touches,'max_stack':self.max_stack}

    @classmethod
    def restore(cls,hierarchy,data,leaves=None):
        if set(data)!={'schema','root_identity','stack','segment_kind','reference_identity','piece_index','byte_index','pending','ordinal','attempt','repairs','complete','total_plan_touches','max_stack'} or data['schema']!=1:raise ExpressionRefuse('expression:checkpoint')
        out=cls.__new__(cls);out.h=hierarchy;out.leaves=leaves;out.root_identity=int(data['root_identity'])
        if out._child(out.root_identity) is None:raise ExpressionRefuse('expression:checkpoint_root')
        out.stack=[ExpressionFrameV1(int(x['closure_identity']),int(x['piece_cursor'])) for x in data['stack']]
        out.segment_kind=int(data['segment_kind']);out.reference_identity=int(data['reference_identity']);out.piece_index=int(data['piece_index']);out.byte_index=int(data['byte_index'])
        p=data['pending'];out.pending=None if p is None else ExpressionBytePlanV1(*(int(p[k]) for k in ('identity','root_identity','segment_kind','reference_identity','piece_index','byte_index','value','ordinal','attempt')))
        out.ordinal=int(data['ordinal']);out.attempt=int(data['attempt']);out.repairs=int(data['repairs']);out.complete=bool(data['complete']);out.total_plan_touches=int(data['total_plan_touches']);out.last_plan_touches=0;out.max_stack=int(data['max_stack']);out._held_literal=None;out._frame_pieces=None
        if out.ordinal<0 or out.byte_index<0 or out.max_stack<1:raise ExpressionRefuse('expression:checkpoint_state')
        if out.pending is not None:
            segment=out._segment_bytes()
            if not 0<=out.byte_index<len(segment) or out.pending.value!=segment[out.byte_index] or out.pending.byte_index!=out.byte_index:raise ExpressionRefuse('expression:checkpoint_pending')
        return out


class IncrementalTransientExpressionV1(IncrementalExpressionV1):
    """Incrementally emit one authenticated transient composition plan.

    The root itself never enters a persistent closure table. Learned language
    span pieces plus already-resident child surfaces are enough.
    """
    def __init__(self,owner,plan:TransientConstructionPlanV1,leaves=None,nodes=None,require_live_template=True):
        if not isinstance(plan,TransientConstructionPlanV1):raise ExpressionRefuse('expression:transient_plan')
        self.h=_hierarchy(owner);self.language=_language(owner);self.leaves=leaves
        self.nodes={} if nodes is None else {int(k):v for k,v in dict(nodes).items()}
        self.transient_plan=plan;self.root_identity=int(plan.identity)
        if require_live_template and language_span_pieces(self.language,plan.template_identity,plan.context,len(plan.child_identities)) is None:
            raise ExpressionRefuse('expression:transient_template')
        self.stack=[ExpressionFrameV1(self.root_identity,int(plan.piece_start))]
        self.segment_kind=0;self.reference_identity=0;self.piece_index=0;self.byte_index=0
        self.pending=None;self.ordinal=0;self.attempt=0;self.repairs=0;self.complete=False
        self.last_plan_touches=0;self.total_plan_touches=0;self.max_stack=1
        self._held_literal=None;self._frame_pieces=None

    def _child(self,identity):
        identity=int(identity)
        if identity in self.nodes:return self.nodes[identity]
        if self.h is not None:
            node=self.h.closure(identity)
            if node is not None:return node
        return None if self.leaves is None else self.leaves.lookup(identity)

    def _node(self,identity):
        return self.transient_plan if int(identity)==self.root_identity else self._child(identity)

    def _pieces(self,node):
        return _span_pieces_held(self.language,node,self._frame_pieces)

    def _segment_bytes(self):
        if self.segment_kind==SEG_LEAF:
            node=self._child(self.reference_identity)
            surface=() if node is None else (getattr(node,'_leaf_surface',()) or getattr(node,'surface',()))
            if node is None or int(getattr(node,'depth',-1))!=0 or not surface:raise ExpressionRefuse('expression:leaf')
            return tuple(surface)
        if self.segment_kind==SEG_LITERAL:
            if self._held_literal is not None:return self._held_literal
            owner=self._node(self.reference_identity)
            if owner is None or owner.depth==0:raise ExpressionRefuse('expression:template_withdrawn')
            pieces=self._pieces(owner)
            if pieces is None:raise ExpressionRefuse('expression:template_withdrawn')
            if not 0<=self.piece_index<len(pieces):raise ExpressionRefuse('expression:literal')
            piece=pieces[self.piece_index]
            if piece.kind!=PIECE_LITERAL or not piece.literal:raise ExpressionRefuse('expression:literal')
            self._held_literal=tuple(piece.literal);return self._held_literal
        raise ExpressionRefuse('expression:segment')

    def _ensure_segment(self):
        touches=0;self._frame_pieces={}
        try:
            while not self.complete and self.segment_kind==0:
                if not self.stack:self.complete=True;break
                frame=self.stack[-1];node=self._node(frame.closure_identity);touches+=1
                if node is None:raise ExpressionRefuse('expression:closure')
                if node.depth==0:
                    if frame.piece_cursor!=0:self.stack.pop();self._prune_spent_nodes();continue
                    frame.piece_cursor=1;self.segment_kind=SEG_LEAF;self.reference_identity=node.identity;self.piece_index=0;self.byte_index=0;self._held_literal=None;break
                pieces=self._pieces(node)
                if pieces is None:raise ExpressionRefuse('expression:template_withdrawn')
                if frame.piece_cursor>=len(pieces):self.stack.pop();self._prune_spent_nodes();continue
                index=frame.piece_cursor;frame.piece_cursor+=1;piece=pieces[index];touches+=1
                if piece.kind==PIECE_LITERAL:
                    if not piece.literal:continue
                    self.segment_kind=SEG_LITERAL;self.reference_identity=node.identity;self.piece_index=index;self.byte_index=0
                    self._held_literal=tuple(piece.literal);break
                if piece.kind==PIECE_PORT and 0<=piece.port<len(node.child_identities):
                    self.stack.append(ExpressionFrameV1(int(node.child_identities[piece.port]),0));self.max_stack=max(self.max_stack,len(self.stack));continue
                raise ExpressionRefuse('expression:piece')
        finally:
            self._frame_pieces=None
        self.last_plan_touches=touches;self.total_plan_touches+=touches

    def _collect_descendants(self,identity,keep):
        identity=int(identity)
        if identity in keep:return
        keep.add(identity)
        node=self.nodes.get(identity)
        if node is None:return
        for cid in getattr(node,'child_identities',()):
            self._collect_descendants(int(cid),keep)

    def _prune_spent_nodes(self):
        if not self.nodes:return
        keep=set();keep.add(self.root_identity)
        if self.reference_identity:keep.add(int(self.reference_identity))
        for frame in self.stack:
            keep.add(int(frame.closure_identity))
            node=self._node(frame.closure_identity)
            if node is None:continue
            kids=getattr(node,'child_identities',())
            if not kids:continue
            pieces=self._pieces(node)
            if pieces:
                for i,piece in enumerate(pieces):
                    if i>=int(frame.piece_cursor) and piece.kind==PIECE_PORT and 0<=int(piece.port)<len(kids):
                        self._collect_descendants(int(kids[piece.port]),keep)
            else:
                for cid in kids:
                    self._collect_descendants(int(cid),keep)
        self.nodes={k:v for k,v in self.nodes.items() if int(k) in keep}

    def _resolve(self,identity):
        identity=int(identity)
        if identity in self.nodes:return self.nodes[identity]
        if self.h is not None:
            node=self.h.closure(identity)
            if node is not None:return node
        return None if self.leaves is None else self.leaves.lookup(identity)

    def _bind_rematerialized_nodes(self):
        for node in sorted((n for n in self.nodes.values() if int(getattr(n,'depth',0))>0),key=lambda n:int(n.depth)):
            kids=[]
            for cid in node.child_identities:
                kid=self._resolve(int(cid))
                if kid is None:
                    kids=None;break
                kids.append(kid)
            if kids is None:continue
            try:
                plan,_=rematerialize_transient_plan(self.language,int(node.context),kids,render=False)
            except HierarchicalRefuse:
                continue
            if (int(plan.identity)!=int(node.identity) or int(plan.template_identity)!=int(node.template_identity)
                or int(plan.depth)!=int(node.depth) or tuple(plan.child_identities)!=tuple(node.child_identities)):
                raise ExpressionRefuse('expression:forged_node')
        kids=[]
        for cid in self.transient_plan.child_identities:
            kid=self._resolve(int(cid))
            if kid is None:return
            kids.append(kid)
        completed=0 if int(self.transient_plan.piece_start)>0 else None
        try:
            plan,_surface=rematerialize_transient_plan(self.language,int(self.transient_plan.context),kids,completed_child=completed,render=False)
        except HierarchicalRefuse:
            return
        if (int(plan.identity)!=int(self.transient_plan.identity) or int(plan.template_identity)!=int(self.transient_plan.template_identity)
            or int(plan.piece_start)!=int(self.transient_plan.piece_start)):
            raise ExpressionRefuse('expression:forged_node')

    def checkpoint(self):
        self._prune_spent_nodes()
        body=super().checkpoint();body['schema']=2;body['transient_plan']=asdict(self.transient_plan)
        body['nodes']={str(int(k)):{'identity':int(v.identity),'context':int(v.context),
            'template_identity':int(getattr(v,'template_identity',0) or 0),
            'child_identities':[int(x) for x in getattr(v,'child_identities',())],
            'depth':int(getattr(v,'depth',0))} for k,v in self.nodes.items()}
        return body

    @classmethod
    def restore(cls,hierarchy,data,leaves=None):
        if data.get('schema')!=2 or 'transient_plan' not in data:raise ExpressionRefuse('expression:transient_checkpoint')
        p=data['transient_plan'];plan=TransientConstructionPlanV1(int(p['identity']),int(p['context']),int(p['template_identity']),tuple(map(int,p['child_identities'])),int(p['depth']),int(p.get('piece_start',0)))
        nodes={}
        for key,row in (data.get('nodes') or {}).items():
            node=SimpleNamespace(identity=int(row['identity']),context=int(row['context']),
                                 template_identity=int(row.get('template_identity',0) or 0),
                                 child_identities=tuple(int(x) for x in row.get('child_identities',())),
                                 depth=int(row.get('depth',0)))
            nodes[int(key)]=node
        out=cls(hierarchy,plan,leaves=leaves,nodes=nodes,require_live_template=False)
        out._bind_rematerialized_nodes()
        out.stack=[ExpressionFrameV1(int(x['closure_identity']),int(x['piece_cursor'])) for x in data['stack']]
        out.segment_kind=int(data['segment_kind']);out.reference_identity=int(data['reference_identity']);out.piece_index=int(data['piece_index']);out.byte_index=int(data['byte_index'])
        pending=data['pending'];out.pending=None if pending is None else ExpressionBytePlanV1(*(int(pending[k]) for k in ('identity','root_identity','segment_kind','reference_identity','piece_index','byte_index','value','ordinal','attempt')))
        out.ordinal=int(data['ordinal']);out.attempt=int(data['attempt']);out.repairs=int(data['repairs']);out.complete=bool(data['complete']);out.total_plan_touches=int(data['total_plan_touches']);out.last_plan_touches=0;out.max_stack=int(data['max_stack']);out._held_literal=None;out._frame_pieces=None
        if out.pending is not None:
            segment=out._segment_bytes()
            if not 0<=out.byte_index<len(segment) or out.pending.value!=segment[out.byte_index] or out.pending.byte_index!=out.byte_index:raise ExpressionRefuse('expression:checkpoint_pending')
        return out


class IncrementalTransientSequenceExpressionV1:
    """Stream a recurrent transient sequence without materializing its transcript."""
    def __init__(self,owner,plan:TransientSequencePlanV1,leaves=None,require_live_template=True):
        if not isinstance(plan,TransientSequencePlanV1):raise ExpressionRefuse('expression:sequence_plan')
        self.h=_hierarchy(owner);self.language=_language(owner);self.leaves=leaves
        self.sequence_plan=plan;self.root_identity=int(plan.identity)
        if len(plan.child_identities)<2 or self._child(int(plan.child_identities[0])) is None:raise ExpressionRefuse('expression:sequence_child')
        self.template=self._live_template()
        if self.template is None:
            if require_live_template:raise ExpressionRefuse('expression:sequence_template')
            self.current=None
        else:
            self.current=self._stage(0)
        self.stage_index=0;self.complete=False;self.repairs=0

    def _live_template(self):
        t=self.language.span_template(int(self.sequence_plan.context),2)
        if t is not None and int(t.identity[:15],16)==int(self.sequence_plan.template_identity):return t
        matches=tuple(t for t in self.language.span_candidates(
            int(self.sequence_plan.context),2)
            if int(t.identity[:15],16)==int(self.sequence_plan.template_identity))
        return matches[0] if len(matches)==1 else None

    def _child(self,identity):
        identity=int(identity)
        if self.h is not None:
            node=self.h.closure(identity)
            if node is not None:return node
        return None if self.leaves is None else self.leaves.lookup(identity)

    def _stage(self,index):
        self.template=self._live_template()
        if self.template is None:raise ExpressionRefuse('expression:sequence_template')
        if index==0:return IncrementalExpressionV1(self.h,int(self.sequence_plan.child_identities[0]),leaves=self.leaves)
        left=self._child(int(self.sequence_plan.child_identities[index-1]));right=self._child(int(self.sequence_plan.child_identities[index]))
        if left is None or right is None:raise ExpressionRefuse('expression:sequence_child')
        try:
            plan,_surface=rematerialize_transient_plan(self.language,int(self.sequence_plan.context),(left,right),self.template,completed_child=0,render=False)
        except HierarchicalRefuse as exc:
            raise ExpressionRefuse('expression:sequence_template') from exc
        return IncrementalTransientExpressionV1(self.h if self.h is not None else self.language,plan,leaves=self.leaves)

    def _bind_sequence_plan(self):
        if self.template is None:return
        kids=[self._child(int(i)) for i in self.sequence_plan.child_identities]
        if any(k is None for k in kids):return
        try:
            plan,_surface=rematerialize_transient_sequence_plan(self.language,int(self.sequence_plan.context),kids,self.template,render=False)
        except HierarchicalRefuse:
            return
        if int(plan.identity)!=int(self.sequence_plan.identity) or int(plan.template_identity)!=int(self.sequence_plan.template_identity):
            raise ExpressionRefuse('expression:forged_node')

    def emit(self):
        while not self.complete:
            if self.current is None:raise ExpressionRefuse('expression:sequence_template')
            plan=self.current.emit()
            if plan is not None:return plan
            nxt=self.stage_index+1
            if nxt>=len(self.sequence_plan.child_identities):
                self.complete=True;return None
            self.current=self._stage(nxt)
            self.stage_index=nxt
        return None

    def reafference(self,plan:ExpressionBytePlanV1,actual_value:int):
        ok=self.current.reafference(plan,actual_value)
        self.repairs+=int(not ok);return ok

    def checkpoint(self):
        return {'schema':3,'sequence_plan':asdict(self.sequence_plan),'stage_index':self.stage_index,
                'current':self.current.checkpoint(),'complete':self.complete,'repairs':self.repairs}

    @classmethod
    def restore(cls,hierarchy,data,leaves=None):
        if data.get('schema')!=3 or 'sequence_plan' not in data:raise ExpressionRefuse('expression:sequence_checkpoint')
        p=data['sequence_plan'];plan=TransientSequencePlanV1(int(p['identity']),int(p['context']),int(p['template_identity']),tuple(map(int,p['child_identities'])),int(p['depth']))
        out=cls(hierarchy,plan,leaves=leaves,require_live_template=False);out.stage_index=int(data['stage_index']);out.complete=bool(data['complete']);out.repairs=int(data['repairs'])
        if not 0<=out.stage_index<len(plan.child_identities):raise ExpressionRefuse('expression:sequence_checkpoint_stage')
        if out.stage_index==0:out.current=IncrementalExpressionV1.restore(hierarchy,data['current'],leaves=leaves)
        else:out.current=IncrementalTransientExpressionV1.restore(hierarchy,data['current'],leaves=leaves)
        out._bind_sequence_plan()
        return out
