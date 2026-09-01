#!/usr/bin/env python3
"""Resident-supported variable-depth endogenous mathematical unfolding.

This module is one mathematical factor of the Workbench Adult, not a second Adult
and not a fixture executor. Persistent state holds source-supported opaque reduction
definitions, endogenous needs, body/work history, and delayed consequence state.
Current closure is disposable and reconstructed only while silent work runs.
"""
from __future__ import annotations
from dataclasses import dataclass
import hashlib, json

Q=1<<16; MIN_SOURCE_SUPPORT=2; MAX_ACTIVE_ROOTS=16; MAX_TRACE=128; MAX_WORK_QUANTA=16; MAX_PRIORITY_DELTA=8
LITERAL=0; Q16_REF=1; U32_Q16_REF=2
AFFINE=0; POLYNOMIAL=1; SCHUR=2; BISIMULATION=3

def _digest(tag,obj):
    raw=json.dumps(obj,sort_keys=True,separators=(',',':'),default=list).encode()
    return hashlib.sha256(tag.encode()+b'\0'+raw).hexdigest()
def _identity(tag,obj):return int(_digest(tag,obj)[:16],16) or 1
def qadd(a,b):
    v=int(a)+int(b);return v if -(1<<31)<=v<(1<<31) else None
def qmul(a,b):
    p=int(a)*int(b)
    if p%Q:return None
    v=p//Q;return v if -(1<<31)<=v<(1<<31) else None
def qdiv(a,b):
    b=int(b)
    if not b:return None
    s=int(a)*Q
    if s%b:return None
    v=s//b;return v if -(1<<31)<=v<(1<<31) else None

@dataclass(frozen=True)
class ReductionTermV1:
    kind:int; value:int=0; node_identity:int=0; slot:int=0
    @classmethod
    def literal(cls,value):return cls(LITERAL,int(value),0,0)
    @classmethod
    def q16(cls,node,slot):return cls(Q16_REF,0,int(node),int(slot))
    @classmethod
    def u32_q16(cls,node,slot):return cls(U32_Q16_REF,0,int(node),int(slot))

@dataclass(frozen=True)
class ResidentReductionNodeV1:
    identity:int; kind:int; coefficients:tuple[ReductionTermV1,...]=(); successors:tuple[int,...]=(); outputs_q16:tuple[int,...]=()
    def __post_init__(self):
        if self.identity<=0 or self.kind not in (AFFINE,POLYNOMIAL,SCHUR,BISIMULATION):raise ValueError('variable-unfold:node')
        if self.kind==BISIMULATION:
            if self.coefficients or not 1<=len(self.successors)<=4 or len(self.outputs_q16)!=len(self.successors):raise ValueError('variable-unfold:bisim-shape')
        elif len(self.coefficients)!=4 or self.successors or self.outputs_q16:raise ValueError('variable-unfold:algebra-shape')
        if any(t.kind not in (LITERAL,Q16_REF,U32_Q16_REF) or (t.kind!=LITERAL and t.node_identity<=0) for t in self.coefficients):raise ValueError('variable-unfold:term')
    @property
    def dependencies(self):return tuple(sorted(set(t.node_identity for t in self.coefficients if t.kind!=LITERAL)))
    def row(self):return {'identity':self.identity,'kind':self.kind,'coefficients':[[t.kind,t.value,t.node_identity,t.slot] for t in self.coefficients],'successors':list(self.successors),'outputs_q16':list(self.outputs_q16)}
    @classmethod
    def from_row(cls,row):return cls(int(row['identity']),int(row['kind']),tuple(ReductionTermV1(*map(int,t)) for t in row.get('coefficients',())),tuple(map(int,row.get('successors',()))),tuple(map(int,row.get('outputs_q16',()))))
    @property
    def definition_identity(self):return _identity('resident-reduction-node-definition-v1',self.row())

@dataclass(frozen=True)
class ExactWitnessV1:
    node_identity:int; definition_identity:int; result_q16:tuple[int,...]; result_u32:tuple[int,...]; class_by_state:tuple[int,...]; identity:int
@dataclass(frozen=True)
class EndogenousNeedV1:
    root_identity:int; priority:int; minimum_work:int
    def __post_init__(self):
        if self.root_identity<=0 or not 1<=self.priority<=64 or not 1<=self.minimum_work<=MAX_WORK_QUANTA:raise ValueError('variable-unfold:need')
@dataclass
class ActiveCandidateV1:
    root_identity:int; priority:int; minimum_work:int; memo:dict[int,ExactWitnessV1]; trace:list[int]; operations:int=0; failed:bool=False
    @property
    def complete(self):return self.root_identity in self.memo and not self.failed
    @property
    def witness(self):return self.memo.get(self.root_identity)
@dataclass(frozen=True)
class PublicActionReceiptV1:
    root_identity:int; witness_identity:int; trace_digest:str; state_revision:int; identity:int

class ResidentProgramStateV1:
    """Future-causally-relevant evidence only; no active closure or schedule."""
    def __init__(self,minimum_source_support=MIN_SOURCE_SUPPORT):
        self.minimum_source_support=int(minimum_source_support)
        if not 1<=self.minimum_source_support<=8:raise ValueError('variable-unfold:min-support')
        self._node_rows={};self._need_rows={};self._withdrawn=set();self.work_quanta=4;self.body_contact_identity=0;self.priority_delta={};self.consumed_consequences=set();self.consequence_lineage=[];self.revision=0
    def observe_node(self,node,source):
        source=int(source)
        if not isinstance(node,ResidentReductionNodeV1) or source<=0:raise ValueError('variable-unfold:node-source')
        bucket=self._node_rows.setdefault(node.identity,{});row=bucket.setdefault(node.definition_identity,[node,set()])
        if row[0]!=node:raise ValueError('variable-unfold:definition-collision')
        row[1].add(source);self.revision+=1
    def observe_need(self,need,source):
        source=int(source)
        if not isinstance(need,EndogenousNeedV1) or source<=0:raise ValueError('variable-unfold:need-source')
        self._need_rows.setdefault(need.root_identity,{}).setdefault((need.priority,need.minimum_work),set()).add(source);self.revision+=1
    def body_contact(self,work_quanta,contact_identity,authenticated=True):
        tokens=int(work_quanta);contact_identity=int(contact_identity)
        if not authenticated or contact_identity<=0 or not 1<=tokens<=MAX_WORK_QUANTA:return False
        self.work_quanta=tokens;self.body_contact_identity=contact_identity;self.revision+=1;return True
    def withdraw_source(self,source):
        source=int(source)
        if source>0:self._withdrawn.add(source);self.revision+=1
    def restore_source(self,source):
        source=int(source)
        if source in self._withdrawn:self._withdrawn.remove(source);self.revision+=1
    def _live_count(self,sources):return sum(1 for s in sources if s not in self._withdrawn)
    def node(self,identity):
        rows=self._node_rows.get(int(identity),{});live=[node for node,sources in rows.values() if self._live_count(sources)>=self.minimum_source_support]
        return live[0] if len(live)==1 else None
    def node_support(self,identity):
        node=self.node(identity)
        return 0 if node is None else self._live_count(self._node_rows[int(identity)][node.definition_identity][1])
    def needs(self):
        out=[]
        for root,rows in self._need_rows.items():
            live=[(p,m) for (p,m),sources in rows.items() if self._live_count(sources)>=self.minimum_source_support]
            if len(live)!=1:continue
            p,m=live[0];p=max(1,min(64,p+self.priority_delta.get(root,0)));out.append(EndogenousNeedV1(root,p,m))
        return tuple(sorted(out,key=lambda n:(-n.priority,n.root_identity)))
    def settle_consequence(self,receipt,source,effect,independent=True,controllable=True,endogenous=False):
        """Settle a real resident episode; origin is provenance, never a veto bit."""
        source=int(source);effect=int(effect);origin=1 if bool(endogenous) else 0
        if not isinstance(receipt,PublicActionReceiptV1) or source<=0 or effect==0 or not independent or not controllable:return False
        expected=_identity('variable-unfold-public-receipt-v1',(receipt.root_identity,receipt.witness_identity,receipt.trace_digest,receipt.state_revision))
        if receipt.identity!=expected:return False
        # The causal episode is receipt+source. Flipping the origin label cannot count
        # the same event twice; genuinely new endogenous episodes may still accumulate.
        ev=_identity('variable-unfold-consequence-v1',(receipt.identity,source))
        if ev in self.consumed_consequences:return False
        self.consumed_consequences.add(ev);prior=self.priority_delta.get(receipt.root_identity,0)
        self.priority_delta[receipt.root_identity]=max(-MAX_PRIORITY_DELTA,min(MAX_PRIORITY_DELTA,prior+(1 if effect>0 else -1)))
        self.consequence_lineage.append({'evidence':ev,'receipt':int(receipt.identity),'root':int(receipt.root_identity),'source':source,'effect':1 if effect>0 else -1,'origin':origin,'independent':1,'controllable':1})
        self.revision+=1;return True
    def checkpoint(self):
        nodes=[]
        for identity in sorted(self._node_rows):
            for definition in sorted(self._node_rows[identity]):
                node,sources=self._node_rows[identity][definition];nodes.append({'node':node.row(),'sources':sorted(sources)})
        needs=[]
        for root in sorted(self._need_rows):
            for (p,m),sources in sorted(self._need_rows[root].items()):needs.append({'root':root,'priority':p,'minimum_work':m,'sources':sorted(sources)})
        return {'schema':2,'minimum_source_support':self.minimum_source_support,'nodes':nodes,'needs':needs,'withdrawn':sorted(self._withdrawn),'work_quanta':self.work_quanta,'body_contact_identity':self.body_contact_identity,'priority_delta':[[k,v] for k,v in sorted(self.priority_delta.items())],'consumed_consequences':sorted(self.consumed_consequences),'consequence_lineage':[dict(row) for row in self.consequence_lineage],'revision':self.revision}
    @classmethod
    def restore(cls,data):
        schema=int(data.get('schema',0))
        if schema not in (1,2):raise ValueError('variable-unfold:checkpoint-schema')
        out=cls(int(data['minimum_source_support']))
        for row in data.get('nodes',()):
            node=ResidentReductionNodeV1.from_row(row['node'])
            for source in row.get('sources',()):out.observe_node(node,source)
        for row in data.get('needs',()):
            need=EndogenousNeedV1(int(row['root']),int(row['priority']),int(row['minimum_work']))
            for source in row.get('sources',()):out.observe_need(need,source)
        out._withdrawn=set(map(int,data.get('withdrawn',())));out.work_quanta=int(data['work_quanta']);out.body_contact_identity=int(data.get('body_contact_identity',0));out.priority_delta={int(k):int(v) for k,v in data.get('priority_delta',())};out.consumed_consequences=set(map(int,data.get('consumed_consequences',())));out.consequence_lineage=[dict(row) for row in data.get('consequence_lineage',())] if schema>=2 else [];out.revision=int(data['revision'])
        if schema>=2 and {int(row.get('evidence',0)) for row in out.consequence_lineage}!=out.consumed_consequences:raise ValueError('variable-unfold:consequence-lineage')
        return out
    def digest(self):return _digest('resident-variable-unfold-state-v1',self.checkpoint())

class EndogenousUnfolderV1:
    """Disposable work state; dependency order is derived from current incidence."""
    def __init__(self,state):self.state=state;self.active={};self.waves=0
    @staticmethod
    def _reduce(node,memo):
        def tv(t):
            if t.kind==LITERAL:return t.value
            w=memo.get(t.node_identity)
            if w is None:return None
            if t.kind==Q16_REF:return w.result_q16[t.slot] if 0<=t.slot<len(w.result_q16) else None
            if t.kind==U32_Q16_REF:return w.result_u32[t.slot]*Q if 0<=t.slot<len(w.result_u32) else None
            return None
        if node.kind==BISIMULATION:
            n=len(node.successors)
            if any(not 0<=s<n for s in node.successors):return None
            classes=[]
            for state in range(n):classes.append(next((classes[p] for p in range(state) if node.outputs_q16[p]==node.outputs_q16[state]),state))
            for _ in range(n):
                nxt=[]
                for state in range(n):
                    assigned=next((nxt[p] for p in range(state) if node.outputs_q16[p]==node.outputs_q16[state] and classes[node.successors[p]]==classes[node.successors[state]]),None)
                    nxt.append(max(nxt,default=-1)+1 if assigned is None else assigned)
                if nxt==classes:break
                classes=nxt
            count=max(classes)+1;ru=[0]*count;rq=[0]*count
            for state in range(n):cls=classes[state];ru[cls]=classes[node.successors[state]];rq[cls]=node.outputs_q16[state]
            rq,ru,cb=tuple(rq),tuple(ru),tuple(classes)
        else:
            vals=[tv(t) for t in node.coefficients]
            if any(v is None for v in vals):return None
            a,b,c,d=map(int,vals)
            if node.kind==AFFINE:
                linear=qmul(c,a);scaled=qmul(c,b);offset=None if scaled is None else qadd(scaled,d)
                if linear is None or offset is None:return None
                rq,ru,cb=(linear,offset),(),()
            elif node.kind==POLYNOMIAL:
                a2,ab,b2=qmul(a,a),qmul(a,b),qmul(b,b)
                if None in (a2,ab,b2):return None
                quad=qmul(c,a2);half=qmul(c,ab);const=qmul(c,b2);linear=None if half is None else qadd(half,half);const=None if const is None else qadd(const,d)
                if None in (quad,linear,const):return None
                rq,ru,cb=(quad,linear,const),(),()
            elif node.kind==SCHUR:
                product=qmul(b,c);quot=None if product is None else qdiv(product,Q-d);result=None if quot is None else qadd(a,quot)
                if result is None:return None
                rq,ru,cb=(result,),(),()
            else:return None
        wid=_identity('variable-unfold-witness-v1',(node.identity,node.definition_identity,rq,ru,cb));return ExactWitnessV1(node.identity,node.definition_identity,tuple(rq),tuple(ru),tuple(cb),wid)
    def _activate(self):
        if self.active:return
        needs=[n for n in self.state.needs() if n.minimum_work<=self.state.work_quanta][:MAX_ACTIVE_ROOTS]
        for n in needs:self.active[n.root_identity]=ActiveCandidateV1(n.root_identity,n.priority,n.minimum_work,{},[])
    def _next_node(self,identity,memo,stack):
        if identity in memo:return None,None
        if identity in stack:return None,'cycle'
        node=self.state.node(identity)
        if node is None:return None,'unsupported'
        stack.add(identity)
        for dep in node.dependencies:
            if dep in memo:continue
            nxt,error=self._next_node(dep,memo,stack)
            stack.remove(identity)
            if error:return None,error
            if nxt is not None:return nxt,None
            stack.add(identity)
        stack.remove(identity);return node,None
    def _step(self,c):
        if c.failed or c.complete:return False
        node,error=self._next_node(c.root_identity,c.memo,set())
        if error or node is None:c.failed=True;return False
        witness=self._reduce(node,c.memo)
        if witness is None:c.failed=True;return False
        c.memo[node.identity]=witness;c.trace.append(node.identity);c.operations+=1
        if len(c.trace)>MAX_TRACE:c.failed=True;return False
        return True
    def silent_wave(self):
        self._activate();self.waves+=1
        for _ in range(self.state.work_quanta):
            live=[c for c in self.active.values() if not c.failed and not c.complete]
            if not live:break
            live.sort(key=lambda c:(c.operations,-c.priority,c.root_identity));self._step(live[0])
        return None
    def winner(self):
        complete=[c for c in self.active.values() if c.complete]
        if not complete:return None
        peak=max(c.priority for c in complete);rows=[c for c in complete if c.priority==peak];return rows[0] if len(rows)==1 else None
    def run_until_settled(self,max_waves=256):
        for _ in range(int(max_waves)):
            self.silent_wave();winner=self.winner()
            if winner is not None and not any(not c.failed and not c.complete and c.priority>=winner.priority for c in self.active.values()):return winner
            if self.active and not any(not c.failed and not c.complete for c in self.active.values()):return self.winner()
        return None
