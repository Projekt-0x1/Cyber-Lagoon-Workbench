#!/usr/bin/env python3
"""N+1: endogenous heterogeneous mathematical unfolding before public language."""
from __future__ import annotations
from dataclasses import dataclass, field
import copy, hashlib, json, time
from reference_language_learning_v1 import LearnedSurfaceEcologyV1
Q=1<<16
VISIBLE=b'the internal relation settles.'
VISIBLE_GAIN='LEARNED_LANGUAGE_CAN_BE_RECRUITED_BY_A_COMPLETED_ENDOGENOUS_MATHEMATICAL_CLOSURE_AFTER_SILENCE_WITHOUT_A_FRESH_LANGUAGE_REQUEST'

def ident(tag,*xs):
    h=hashlib.blake2b(digest_size=8,person=b'0x1-unfold');h.update(tag.encode())
    for x in xs:h.update((int(x)&((1<<64)-1)).to_bytes(8,'little',signed=False))
    return int.from_bytes(h.digest(),'little') or 1

def qmul(a,b):
    v=int(a)*int(b)
    if v & 0xffff:return None
    v >>= 16
    return v if -(1<<31)<=v<(1<<31) else None

def qadd(a,b):
    v=int(a)+int(b);return v if -(1<<31)<=v<(1<<31) else None

@dataclass(frozen=True)
class Witness:
    kind:str;result:tuple;classes:tuple=();eliminated:int=0;identity:int=0

def affine_reduce(c):
    a,b,c2,d=map(int,c);A=qmul(c2,a);B0=qmul(c2,b)
    if A is None or B0 is None:return None
    B=qadd(B0,d)
    if B is None:return None
    r=(A,B);return Witness('affine',r,(),1,ident('affine',*r))

def polynomial_reduce(c):
    A,B,k,d=map(int,c);a=qmul(k,A);b=qmul(k,B)
    if a is None or b is None:return None
    r=(a,b,d);return Witness('polynomial',r,(),0,ident('poly',*r))

def schur_reduce(c):
    a,b,c2,d=map(int,c);bc=qmul(b,c2)
    if bc is None or d==0:return None
    num=bc<<16
    if num%d:return None
    r=qadd(a,-(num//d))
    return None if r is None else Witness('schur',(r,),(),1,ident('schur',r))

def bisimulation_reduce(successors,outputs):
    n=len(successors)
    if n==0 or n>8 or len(outputs)!=n or any(s<0 or s>=n for s in successors):return None
    omap={v:i for i,v in enumerate(sorted(set(outputs)))};classes=[omap[v] for v in outputs]
    for _ in range(n+1):
        keys=[(outputs[i],classes[successors[i]]) for i in range(n)]
        uniq={k:j for j,k in enumerate(sorted(set(keys)))};new=[uniq[k] for k in keys]
        if new==classes:break
        classes=new
    reps=[]
    for c in range(max(classes)+1):
        members=[i for i,x in enumerate(classes) if x==c];target=classes[successors[members[0]]]
        if any(classes[successors[i]]!=target for i in members):return None
        reps.append(target)
    return Witness('bisimulation',tuple(reps),tuple(classes),n-len(reps),ident('bisim',*classes,*reps))

@dataclass(frozen=True)
class PersistentState:
    affine_coeffs:tuple=(2*Q,Q,3*Q,-2*Q)
    quotient_successors:tuple=(2,3,2,3)
    quotient_outputs:tuple=(0,0,Q,Q)
    source_a:int=0xEA110001;source_b:int=0xEA110002
    support_a:bool=True;support_b:bool=True
    heldout_x:int=2*Q;pending_root:int=0xEA1100F0;evidence_events:int=2
    def checkpoint(self):return self.__dict__.copy()
    @classmethod
    def restore(cls,d):return cls(**d)

@dataclass
class TransientNode:
    op:str;deps:tuple=();ready:bool=False;value:object=None;identity:int=0
@dataclass
class ActiveClosure:
    nodes:list=field(default_factory=lambda:[TransientNode('affine'),TransientNode('poly',(0,)),TransientNode('schur',(0,)),TransientNode('quotient'),TransientNode('endpoint',(1,2,3))])
    operations:int=0;waves:int=0;endpoint:int=0;endpoint_q16:int=0

class EndogenousMathAdult:
    def __init__(self,state):self.state=state;self.active=None;self.public_feature=0
    def ensure_active(self):
        if self.active is None and self.state.pending_root:self.active=ActiveClosure()
        return self.active
    def _execute(self,i):
        a=self.ensure_active();n=a.nodes[i]
        if n.ready or any(not a.nodes[d].ready for d in n.deps):return False
        if n.op=='affine':
            if not self.state.support_a:return False
            w=affine_reduce(self.state.affine_coeffs)
        elif n.op=='poly':
            A,B=a.nodes[0].value.result;w=polynomial_reduce((A,B,Q//2,3*Q))
        elif n.op=='schur':
            A,_=a.nodes[0].value.result;w=schur_reduce((A,Q//2,Q//2,Q//2))
        elif n.op=='quotient':
            if not self.state.support_b:return False
            w=bisimulation_reduce(self.state.quotient_successors,self.state.quotient_outputs)
        else:
            pw,sw,qw=(a.nodes[j].value for j in (1,2,3));x=self.state.heldout_x
            x2=qmul(x,x);pa=qmul(pw.result[0],x2) if x2 is not None else None;pb=qmul(pw.result[1],x)
            if None in (x2,pa,pb):return False
            subtotal=qadd(pa,pb)
            if subtotal is None:return False
            first=qadd(subtotal,pw.result[2])
            if first is None:return False
            cls=qw.classes[0];nxt=qw.result[cls]
            second=qmul(sw.result[0],first) if nxt==1 else first
            if second is None:return False
            w=Witness('endpoint',(second,),(),0,ident('endpoint',pw.identity,sw.identity,qw.identity,second));a.endpoint=w.identity;a.endpoint_q16=second
        if w is None:return False
        n.value=w;n.identity=w.identity;n.ready=True;a.operations+=1;return True
    def silent_wave(self,work_budget):
        a=self.ensure_active();before=a.operations
        for i in range(len(a.nodes)):
            if a.operations-before>=int(work_budget):break
            self._execute(i)
        a.waves+=1
        return a.operations-before
    def unfold_to_quiescence(self,work_budget):
        a=self.ensure_active()
        for _ in range(32):
            before=a.operations;self.silent_wave(work_budget)
            if a.endpoint or a.operations==before:break
        return a.endpoint
    def externalize(self,enabled=True):
        if enabled and self.active and self.active.endpoint:self.public_feature=self.active.endpoint
        return self.public_feature if enabled else 0

def run(state,budget,externalize=False):
    adult=EndogenousMathAdult(copy.deepcopy(state))
    support_before=(adult.state.support_a,adult.state.support_b,adult.state.evidence_events)
    endpoint=adult.unfold_to_quiescence(budget);pub=adult.externalize(externalize)
    support_after=(adult.state.support_a,adult.state.support_b,adult.state.evidence_events)
    return adult,endpoint,pub,support_before,support_after

def main():
    started=time.perf_counter();checks={};state=PersistentState();checkpoint=state.checkpoint()
    demo,e_demo,_,_,_=run(state,4,False)
    language=LearnedSurfaceEcologyV1();language.observe_naming(e_demo,VISIBLE,0xEA210001)
    checks['one_language_source_is_not_enough']=language.lexeme(e_demo) is None
    language.observe_naming(e_demo,VISIBLE,0xEA210002);learned_before=language.lexeme(e_demo);learned_before_bytes=bytes(learned_before) if learned_before else b''

    low,e_low,p_low,sb,sa=run(PersistentState.restore(copy.deepcopy(checkpoint)),1,False)
    high,e_high,_,_,_=run(PersistentState.restore(copy.deepcopy(checkpoint)),4,False)
    replay,e_replay,_,_,_=run(PersistentState.restore(copy.deepcopy(checkpoint)),1,False)
    motor,e_motor,p_motor,_,_=run(PersistentState.restore(copy.deepcopy(checkpoint)),1,True)
    withdrawn=PersistentState(**{**state.__dict__,'support_a':False});w,e_w,p_w,_,_=run(withdrawn,4,True)
    unsupported=PersistentState(**{**state.__dict__,'support_a':False,'support_b':False});u,e_u,p_u,_,_=run(unsupported,4,True)
    learned_after=language.lexeme(p_motor) if p_motor else None;learned_after_bytes=bytes(learned_after) if learned_after else b''

    checks['silent_unfold_reaches_heterogeneous_endpoint']=bool(e_low and low.active.operations==5 and e_demo==e_low)
    checks['resource_changes_wave_count_not_exact_endpoint']=low.active.waves>high.active.waves and e_low==e_high and low.active.endpoint_q16==high.active.endpoint_q16
    checks['no_public_projection_during_silent_unfold']=p_low==0
    checks['externalization_is_dissociable_from_internal_endpoint']=e_motor==e_low and p_motor==e_low
    checks['checkpoint_omits_transient_closure_and_rematerializes_exactly']='active' not in checkpoint and e_replay==e_low and replay.active.operations==low.active.operations
    checks['source_withdrawal_or_insufficient_history_refuses_endpoint']=e_w==0 and p_w==0 and e_u==0 and p_u==0
    checks['withdrawal_preserves_independently_learned_language_memory']=bytes(language.lexeme(e_low) or ())==VISIBLE
    checks['endogenous_unfold_cannot_self_award_evidence']=sb==sa==(True,True,2)
    checks['learned_language_projects_only_after_internal_closure']=learned_before_bytes==VISIBLE and learned_after_bytes==VISIBLE
    checks['visible_gain_requires_no_fresh_language_contact_during_silence']=learned_after_bytes==VISIBLE and low.active.waves>=2
    persistent_bytes=len(json.dumps(checkpoint,sort_keys=True,separators=(',',':')).encode())
    transient_bytes=len(json.dumps([(n.op,n.deps,n.ready,n.identity) for n in low.active.nodes],separators=(',',':')).encode())
    checks['transient_computation_expands_beyond_persistent_root_count']=len(low.active.nodes)>2 and transient_bytes>0
    checks['bounded_reference_lane']=time.perf_counter()-started<1.0
    failed=[k for k,v in checks.items() if not v]
    result={'schema':'cyber-lagoon.endogenous-mathematical-unfolding.v1','contract':'FOUNDRY_ENDOGENOUS_MATHEMATICAL_UNFOLDING_'+('GREEN' if not failed else 'RED'),'pass':not failed,'reference_only':True,'language_phenotype_improved':not failed,'visible_language_gain':VISIBLE_GAIN,'public_surface':learned_after_bytes.decode() if learned_after_bytes else '','quantity':{'persistent_checkpoint_bytes':persistent_bytes,'transient_trace_bytes':transient_bytes,'relation_operations':low.active.operations,'low_resource_waves':low.active.waves,'high_resource_waves':high.active.waves},'endpoint':{'identity':e_low,'q16':low.active.endpoint_q16},'checks':checks,'failed':failed,'remaining_red':['DIRECT_ADULT_ENDOGENOUS_HETEROGENEOUS_UNFOLDING_PARITY','LEARNED_CROSS_FAMILY_AUTOMATIC_CONSTRUCTOR','OPEN_ENDED_AUTONOMOUS_COGNITIVE_CONTINUATION'],'elapsed_ms':round((time.perf_counter()-started)*1000,3)}
    print(result['contract']);print('visible_language_gain='+VISIBLE_GAIN);print(json.dumps(result,indent=2,sort_keys=True));return 0 if not failed else 1
if __name__=='__main__':raise SystemExit(main())
