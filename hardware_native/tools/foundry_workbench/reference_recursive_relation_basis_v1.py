#!/usr/bin/env python3
"""Persistent exact relation basis whose derived closures re-enter later closure."""
from __future__ import annotations
from dataclasses import dataclass
from collections import deque
import hashlib,json
from reference_heterogeneous_exact_relation_algebra_v1 import compose_affine

MAX_RELATIONS=512
MAX_PATHS=4096
DEFAULT_WORK_BUDGET=256

def _id(tag,value):
    raw=json.dumps(value,sort_keys=True,separators=(',',':'),default=list).encode()
    return int.from_bytes(hashlib.sha256(tag.encode()+b'\0'+raw).digest()[:8],'big') or 1

@dataclass(frozen=True)
class ResidentExactRelationV1:
    left_space:int
    right_space:int
    boundary_q16:tuple[int,int]
    direct_evidence:tuple[int,...]=()
    root_evidence:tuple[int,...]=()
    support_paths:tuple[tuple[int,...],...]=()
    def __post_init__(self):
        if min(int(self.left_space),int(self.right_space))<=0 or int(self.left_space)==int(self.right_space):raise ValueError('recursive-basis:space')
        if len(self.boundary_q16)!=2:raise ValueError('recursive-basis:boundary')
        if any(int(x)<=0 for x in (*self.direct_evidence,*self.root_evidence)):raise ValueError('recursive-basis:evidence')
        if any(not path or any(int(x)<=0 for x in path) for path in self.support_paths):raise ValueError('recursive-basis:path')
    @property
    def identity(self):return _id('resident-exact-relation-v1',(int(self.left_space),int(self.right_space),tuple(map(int,self.boundary_q16))))
    @property
    def derived(self):return bool(self.support_paths)

@dataclass(frozen=True)
class ResidentRelationClosureV1:
    left_space:int
    right_space:int
    boundary_q16:tuple[int,int]
    paths:tuple[tuple[int,...],...]
    relation_touches:int

class RecursiveRelationBasisV1:
    """One executable relation type for lived and self-derived exact boundaries."""
    def __init__(self):
        self.relations={};self.withdrawn_evidence=set();self.active=set();self.ambiguous_pairs=set();self.revision=0
        self.last_touches=0;self.last_path=();self.last_retained=0;self.mutation_touches=0
        self.last_silent_touches=0;self._silent_exhausted_revision=-1
    def _replace(self,relation):
        if relation.identity not in self.relations and len(self.relations)>=MAX_RELATIONS:raise RuntimeError('recursive-basis:capacity')
        self.relations[relation.identity]=relation
    def observe_primitive(self,left,right,boundary_q16,evidence_identity):
        evidence=int(evidence_identity);candidate=ResidentExactRelationV1(int(left),int(right),tuple(map(int,boundary_q16)),(evidence,),(evidence,),())
        old=self.relations.get(candidate.identity)
        if old is not None and evidence in old.direct_evidence:return candidate.identity
        direct={evidence};roots={evidence};paths=()
        if old is not None:direct.update(old.direct_evidence);roots.update(old.root_evidence);paths=old.support_paths
        self._replace(ResidentExactRelationV1(candidate.left_space,candidate.right_space,candidate.boundary_q16,tuple(sorted(direct)),tuple(sorted(roots)),paths))
        self.revision+=1;self._rebuild();return candidate.identity
    def _primitive_active(self,row):return any(int(e) not in self.withdrawn_evidence for e in row.direct_evidence)
    def _base_active(self):
        live={rid for rid,row in self.relations.items() if self._primitive_active(row)};changed=True
        while changed:
            changed=False
            for rid,row in self.relations.items():
                if rid not in live and any(all(int(child) in live for child in path) for path in row.support_paths):live.add(rid);changed=True
        return live
    @staticmethod
    def _compose(path,relations):
        if not path:return None
        boundary=tuple(relations[int(path[0])].boundary_q16)
        for rid in path[1:]:
            boundary=compose_affine(boundary,tuple(relations[int(rid)].boundary_q16))
            if boundary is None:return None
        return boundary
    def _rebuild(self):
        # Active support is causal state; traversal indexes are disposable lookup state.
        # Rebuild them only after mutation so local pair/lhs queries never scan dormant
        # or unrelated active relations during language realization.
        self.active=self._base_active();self._best_paths={};self._pair_boundaries={};self.ambiguous_pairs=set();self.mutation_touches=len(self.active)
        by_left={};by_pair={};spaces=set()
        for rid in sorted(self.active):
            row=self.relations[rid];by_left.setdefault(int(row.left_space),[]).append(int(rid));by_pair.setdefault((int(row.left_space),int(row.right_space)),[]).append(int(rid));spaces|={int(row.left_space),int(row.right_space)}
        self._active_by_left={key:tuple(rows) for key,rows in by_left.items()};self._active_by_pair={key:tuple(rows) for key,rows in by_pair.items()};self._active_spaces=frozenset(spaces)
    def active_relations_for_pair(self,left,right):
        """Disposable O(1) endpoint lookup; identities remain the causal authority."""
        return tuple(self._active_by_pair.get((int(left),int(right)),()))
    def _solve_pair(self,left,right):
        pair=(int(left),int(right))
        if pair in self._pair_boundaries:return self._pair_boundaries[pair]
        outgoing=self._active_by_left;spaces=self._active_spaces
        if pair[0] not in spaces or pair[1] not in spaces:
            self._pair_boundaries[pair]=();self._best_paths[pair]=();return ()
        max_depth=max(1,len(spaces)-1);queue=deque([(pair[0],None,(),frozenset((pair[0],)))])
        shortest={};found={};touches=0;overflow=False
        while queue and not overflow:
            space,boundary,path,visited=queue.popleft()
            if len(path)>=max_depth:continue
            for rid in outgoing.get(space,()):
                touches+=1;row=self.relations[rid];nxt=row.right_space
                if nxt in visited:continue
                composed=tuple(row.boundary_q16) if boundary is None else compose_affine(boundary,tuple(row.boundary_q16))
                if composed is None:continue
                p=path+(rid,);state=(nxt,tuple(composed));prior=shortest.get(state)
                if prior is not None and len(p)>prior:continue
                if prior is None:shortest[state]=len(p)
                if nxt==pair[1]:
                    bucket=found.setdefault(tuple(composed),[])
                    if not bucket or len(p)<len(bucket[0]):bucket[:]=[p]
                    elif len(p)==len(bucket[0]) and p not in bucket:bucket.append(p)
                    if len(found)>1:break
                if len(shortest)>MAX_PATHS:overflow=True;break
                queue.append((nxt,tuple(composed),p,visited|{nxt}))
            if len(found)>1:break
        self.last_touches=touches
        boundaries=tuple(sorted(found))
        if overflow or len(boundaries)>1:
            self.ambiguous_pairs.add(pair);self._pair_boundaries[pair]=boundaries;self._best_paths[pair]=();return boundaries
        paths=tuple(sorted(next(iter(found.values())))) if found else ()
        self._pair_boundaries[pair]=boundaries;self._best_paths[pair]=paths
        return boundaries
    def withdraw_evidence(self,evidence_identity):
        evidence=int(evidence_identity)
        if evidence<=0:return False
        self.withdrawn_evidence.add(evidence);self.revision+=1;self._rebuild();return True
    def restore_evidence(self,evidence_identity):
        evidence=int(evidence_identity)
        if evidence not in self.withdrawn_evidence:return False
        self.withdrawn_evidence.remove(evidence);self.revision+=1;self._rebuild();return True
    def resolve(self,left,right,work_budget=DEFAULT_WORK_BUDGET):
        left=int(left);right=int(right);self.last_touches=0;self.last_path=()
        if min(left,right)<=0 or left==right or int(work_budget)<=0:return None
        boundaries=self._solve_pair(left,right)
        if (left,right) in self.ambiguous_pairs or len(boundaries)!=1:return None
        paths=tuple(self._best_paths.get((left,right),()))
        if not paths:return None
        if len(paths[0])>int(work_budget):self.last_touches=int(work_budget)+1;return None
        self.last_path=paths[0];self.last_touches=len(self.last_path)
        return ResidentRelationClosureV1(left,right,tuple(boundaries[0]),paths,self.last_touches)
    def retain(self,closure):
        if not isinstance(closure,ResidentRelationClosureV1) or not closure.paths:return 0
        paths=tuple(path for path in closure.paths if len(path)>=2 and all(int(rid) in self.active for rid in path))
        if not paths:return 0
        candidate=ResidentExactRelationV1(closure.left_space,closure.right_space,closure.boundary_q16,(),(),paths)
        old=self.relations.get(candidate.identity);direct=set();roots=set();supports=set(paths)
        if old is not None:direct.update(old.direct_evidence);roots.update(old.root_evidence);supports.update(old.support_paths)
        for path in supports:
            for rid in path:
                row=self.relations.get(int(rid))
                if row is None:return 0
                roots.update(row.root_evidence)
        self._replace(ResidentExactRelationV1(candidate.left_space,candidate.right_space,candidate.boundary_q16,tuple(sorted(direct)),tuple(sorted(roots)),tuple(sorted(supports))))
        self.revision+=1;self.last_retained=candidate.identity;self._rebuild();return candidate.identity
    def expand_spaces(self,relation_identity,seen=()):
        rid=int(relation_identity);row=self.relations.get(rid)
        if row is None or rid not in self.active or rid in seen:return None
        live_paths=[p for p in row.support_paths if all(int(x) in self.active for x in p)]
        if not live_paths:return (row.left_space,row.right_space) if self._primitive_active(row) else None
        path=sorted(live_paths,key=lambda p:(len(p),p))[0];out=[]
        for child in path:
            seq=self.expand_spaces(child,seen+(rid,))
            if not seq:return None
            if out and out[-1]==seq[0]:out.extend(seq[1:])
            else:out.extend(seq)
        return tuple(out)
    def expanded_space_paths(self,relation_identity,max_paths=64):
        """Enumerate terminal-space derivations; derived lookup indexes remain non-authoritative."""
        max_paths=max(1,min(int(max_paths),256));memo={};visiting=set()
        def expand(rid):
            rid=int(rid)
            if rid in memo:return memo[rid]
            if rid in visiting:return ()
            row=self.relations.get(rid)
            if row is None or rid not in self.active:return ()
            live=[p for p in row.support_paths if all(int(x) in self.active for x in p)]
            if not live:
                out=((row.left_space,row.right_space),) if self._primitive_active(row) else ()
                memo[rid]=out;return out
            visiting.add(rid);out=[]
            for path in sorted(live,key=lambda p:(len(p),p)):
                partial=((),)
                for child in path:
                    child_paths=expand(child)
                    if not child_paths:partial=();break
                    nxt=[]
                    for prefix in partial:
                        for seq in child_paths:
                            merged=seq if not prefix else prefix+seq[1:] if prefix[-1]==seq[0] else ()
                            if merged and merged not in nxt:nxt.append(merged)
                            if len(nxt)>=max_paths:break
                        if len(nxt)>=max_paths:break
                    partial=tuple(nxt)
                    if not partial:break
                for seq in partial:
                    if seq not in out:out.append(seq)
                    if len(out)>=max_paths:break
                if len(out)>=max_paths:break
            visiting.remove(rid);memo[rid]=tuple(out);return memo[rid]
        return expand(int(relation_identity))

    def generation(self,relation_identity,seen=()):
        rid=int(relation_identity);row=self.relations.get(rid)
        if row is None or rid in seen:return 0
        live=[p for p in row.support_paths if all(int(x) in self.active for x in p)]
        if not live:return 0
        return 1+max(max((self.generation(child,seen+(rid,)) for child in path),default=0) for path in live)
    def silent_wave(self):
        """Compile one closure; unchanged fixed-point revisions do no repeated basis scan."""
        self.last_silent_touches=0
        if self._silent_exhausted_revision==self.revision:return 0
        candidates=[];rows=[self.relations[rid] for rid in sorted(self.active)];by_left={}
        for row in rows:by_left.setdefault(row.left_space,[]).append(row)
        for left in rows:
            for right in by_left.get(left.right_space,()):
                self.last_silent_touches+=1
                if left.left_space==right.right_space:continue
                boundary=compose_affine(left.boundary_q16,right.boundary_q16)
                if boundary is None:continue
                pair=(left.left_space,right.right_space);existing=self._solve_pair(*pair)
                if pair in self.ambiguous_pairs or (existing and tuple(boundary) not in existing):continue
                rid=_id('resident-exact-relation-v1',(left.left_space,right.right_space,tuple(boundary)));current=self.relations.get(rid);path=(left.identity,right.identity)
                if current is not None and path in current.support_paths:continue
                lseq=self.expand_spaces(left.identity);rseq=self.expand_spaces(right.identity)
                if not lseq or not rseq:continue
                span=len(lseq)+len(rseq)-1;candidates.append((-span,left.left_space,right.right_space,path,boundary))
        if not candidates:
            self._silent_exhausted_revision=self.revision;return 0
        _neg,_l,_r,path,boundary=sorted(candidates)[0]
        closure=ResidentRelationClosureV1(self.relations[path[0]].left_space,self.relations[path[-1]].right_space,tuple(boundary),(path,),2)
        return self.retain(closure)
    def remove_derived(self,relation_identity):
        rid=int(relation_identity);row=self.relations.get(rid)
        if row is None or not row.support_paths:return False
        if row.direct_evidence:self.relations[rid]=ResidentExactRelationV1(row.left_space,row.right_space,row.boundary_q16,row.direct_evidence,row.direct_evidence,())
        else:self.relations.pop(rid,None)
        self.revision+=1;self._rebuild();return True
    def best_derived(self):
        rows=[]
        for rid in self.active:
            row=self.relations[rid]
            if not row.derived:continue
            seq=self.expand_spaces(rid)
            if seq:rows.append((len(seq),self.generation(rid),rid))
        if not rows:return 0
        peak=max((n,g) for n,g,_ in rows);w=[rid for n,g,rid in rows if (n,g)==peak]
        return w[0] if len(w)==1 else 0
    def checkpoint(self):
        return {'schema':1,'relations':[{'left':r.left_space,'right':r.right_space,'boundary':list(r.boundary_q16),'direct_evidence':list(r.direct_evidence),'root_evidence':list(r.root_evidence),'support_paths':[list(p) for p in r.support_paths]} for r in sorted(self.relations.values(),key=lambda x:x.identity)],'withdrawn_evidence':sorted(self.withdrawn_evidence),'revision':self.revision}
    @classmethod
    def restore(cls,data):
        if int(data.get('schema',0))!=1:raise ValueError('recursive-basis:checkpoint')
        out=cls()
        for row in data.get('relations',()):
            relation=ResidentExactRelationV1(int(row['left']),int(row['right']),tuple(map(int,row['boundary'])),tuple(map(int,row.get('direct_evidence',()))),tuple(map(int,row.get('root_evidence',()))),tuple(tuple(map(int,p)) for p in row.get('support_paths',())))
            if relation.identity in out.relations:raise ValueError('recursive-basis:duplicate')
            out.relations[relation.identity]=relation
        out.withdrawn_evidence=set(map(int,data.get('withdrawn_evidence',())));out.revision=int(data.get('revision',0))
        for row in out.relations.values():
            if any(any(int(child) not in out.relations for child in path) for path in row.support_paths):raise ValueError('recursive-basis:ancestry')
        out._rebuild();return out
