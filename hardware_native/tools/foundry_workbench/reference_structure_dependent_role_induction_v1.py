#!/usr/bin/env python3
"""Grounded structure-dependent induction from raw surface plus lived referents.

This challenger deliberately contains no language id, parser table, quote/negation/imperative
opcode, expected tree, or response string. It learns two things from ordinary chronology:
(1) reusable raw-surface spans that covary with grounded atoms across independent sources, and
(2) reusable boundary relations that recur between different grounded pairs. Current structure
is then freshly unfolded as a binary hierarchy over grounded anchors and learned relations.

The design is intentionally falsifiable against linear shortcuts: relation factors must transfer
to new atom pairs and deeper compositions, while matched anchor/count foils with the learned
relation in the wrong structural boundary must refuse.
"""
from __future__ import annotations
from dataclasses import dataclass
import hashlib, json

MIN_SPAN=2
MAX_SPAN=24
MIN_ANCHOR=3
MIN_RELATION=2
MAX_EXAMPLES=4096
MAX_FACTORS=2048
MAX_MATCH_TOUCHES=16384


def _h(tag, raw):
    return hashlib.sha256(tag.encode()+b'\0'+bytes(raw)).digest()[:12].hex()


def _id(tag, payload):
    raw=json.dumps([tag,payload],sort_keys=True,separators=(',',':')).encode()
    return int(hashlib.sha256(raw).hexdigest()[:16],16) or 1


def _sketch(raw, min_n=MIN_SPAN, max_n=MAX_SPAN):
    raw=tuple(map(int,raw));rows=[]
    upper=min(max_n,len(raw))
    for n in range(min_n,upper+1):
        for pos in range(0,len(raw)-n+1):
            rows.append((n,_h('span',raw[pos:pos+n])))
    return tuple(dict.fromkeys(rows))


def _common(a,b,min_n):
    common=set(a)&set(b)
    rows=sorted((r for r in common if int(r[0])>=min_n),key=lambda r:(-int(r[0]),r[1]))
    return tuple(rows[:64])


@dataclass(frozen=True)
class GroundedAnchorFactorV1:
    identity:int
    atom:int
    members:tuple
    sources:tuple[int,...]


@dataclass(frozen=True)
class GroundedRelationFactorV1:
    identity:int
    members:tuple
    sources:tuple[int,...]
    pair_witnesses:tuple[tuple[int,int],...]


class StructureDependentRoleInducerV1:
    """Cross-situational constituent and relation learner with no semantic frame labels."""
    def __init__(self):
        self._anchor_examples=[]
        self._relation_examples=[]
        self._anchors=[]
        self._relations=[]
        self._tick=-1
        self.last_match_touches=0

    def _advance(self,tick):
        tick=int(tick)
        if tick<self._tick:raise ValueError('structure-inducer:time-reversal')
        self._tick=tick

    def _settle_anchor(self,atom,source,sketch):
        candidates=[]
        for row in self._anchor_examples:
            if row['atom']!=atom or row['source']==source:continue
            fam=_common(row['sketch'],sketch,MIN_ANCHOR)
            if fam:candidates.append((fam,row))
        if not candidates:return False
        longest=max(f[0][0] for f,_ in candidates)
        fams={tuple(f) for f,_ in candidates if f[0][0]==longest}
        if len(fams)!=1:return False
        family=next(iter(fams));sources={source}|{r['source'] for f,r in candidates if tuple(f)==family}
        existing=next((x for x in self._anchors if x.atom==atom and x.members==family),None)
        if existing:
            idx=self._anchors.index(existing)
            self._anchors[idx]=GroundedAnchorFactorV1(existing.identity,atom,family,tuple(sorted(set(existing.sources)|sources)))
            return True
        if len(self._anchors)>=MAX_FACTORS:return False
        self._anchors.append(GroundedAnchorFactorV1(_id('grounded-anchor-v1',[atom,list(family)]),atom,family,tuple(sorted(sources))))
        return True

    def _settle_relation(self,source,pair,sketch):
        candidates=[]
        for row in self._relation_examples:
            if row['source']==source or row['pair']==pair:continue
            fam=_common(row['sketch'],sketch,MIN_RELATION)
            if fam:candidates.append((fam,row))
        if not candidates:return False
        longest=max(f[0][0] for f,_ in candidates)
        fams={tuple(f) for f,_ in candidates if f[0][0]==longest}
        if len(fams)!=1:return False
        family=next(iter(fams));sources={source};pairs={pair}
        for f,row in candidates:
            if tuple(f)==family:sources.add(row['source']);pairs.add(row['pair'])
        existing=next((x for x in self._relations if x.members==family),None)
        if existing:
            idx=self._relations.index(existing)
            self._relations[idx]=GroundedRelationFactorV1(existing.identity,family,tuple(sorted(set(existing.sources)|sources)),tuple(sorted(set(existing.pair_witnesses)|pairs)))
            return True
        if len(self._relations)>=MAX_FACTORS:return False
        self._relations.append(GroundedRelationFactorV1(_id('grounded-relation-v1',[list(family)]),family,tuple(sorted(sources)),tuple(sorted(pairs))))
        return True

    def observe_singleton(self,raw,atom,source,tick):
        self._advance(tick);raw=tuple(map(int,raw));atom=int(atom);source=int(source)
        if not raw or min(atom,source)<=0:raise ValueError('structure-inducer:singleton')
        sketch=_sketch(raw)
        self._settle_anchor(atom,source,sketch)
        row={'atom':atom,'source':source,'tick':int(tick),'sketch':sketch}
        if not any(x['atom']==atom and x['source']==source and x['sketch']==sketch for x in self._anchor_examples):
            if len(self._anchor_examples)>=MAX_EXAMPLES:self._anchor_examples.pop(0)
            self._anchor_examples.append(row)
        return True

    @staticmethod
    def _occurrences(raw,members):
        raw=tuple(map(int,raw));member=set(members);rows=[]
        lengths=sorted({int(n) for n,_ in members},reverse=True)
        hashes={int(n):{d for nn,d in members if int(nn)==int(n)} for n in lengths}
        for n in lengths:
            for pos in range(0,len(raw)-n+1):
                if _h('span',raw[pos:pos+n]) in hashes[n]:rows.append((pos,pos+n,n))
        return rows

    def _resolve_anchors(self,raw,expected_atoms=()):
        expected=set(map(int,expected_atoms)) if expected_atoms else None
        candidates=[];touches=0
        for factor in self._anchors:
            if expected is not None and factor.atom not in expected:continue
            hits=self._occurrences(raw,factor.members);touches+=max(1,len(factor.members))
            for start,end,n in hits:candidates.append((n,len(factor.sources),factor.atom,factor.identity,start,end))
            if touches>MAX_MATCH_TOUCHES:break
        self.last_match_touches=touches
        candidates.sort(key=lambda r:(-r[0],-r[1],r[4],r[2]))
        chosen=[];used_atoms=set()
        for row in candidates:
            _n,_support,atom,_fid,start,end=row
            if atom in used_atoms or any(not (end<=x[4] or start>=x[5]) for x in chosen):continue
            chosen.append(row);used_atoms.add(atom)
        chosen.sort(key=lambda r:r[4])
        return tuple(chosen)

    def observe_relation_surface(self,raw,grounded_atoms,source,tick):
        self._advance(tick);raw=tuple(map(int,raw));atoms=tuple(map(int,grounded_atoms));source=int(source)
        if len(atoms)<2 or source<=0:raise ValueError('structure-inducer:relation')
        anchors=self._resolve_anchors(raw,atoms)
        if len(anchors)!=len(set(atoms)) or {x[2] for x in anchors}!=set(atoms):return False
        learned=False
        for left,right in zip(anchors,anchors[1:]):
            gap=raw[left[5]:right[4]]
            if len(gap)<MIN_RELATION:continue
            pair=(int(left[2]),int(right[2]));sketch=_sketch(gap,MIN_RELATION,12)
            if self._settle_relation(source,pair,sketch):learned=True
            row={'source':source,'pair':pair,'tick':int(tick),'sketch':sketch}
            if not any(x['source']==source and x['pair']==pair and x['sketch']==sketch for x in self._relation_examples):
                if len(self._relation_examples)>=MAX_EXAMPLES:self._relation_examples.pop(0)
                self._relation_examples.append(row)
        return learned

    def _relation_for_gap(self,gap):
        sketch=set(_sketch(gap,MIN_RELATION,12));scored=[]
        for factor in self._relations:
            common=sketch&set(factor.members)
            longest=max((int(n) for n,_ in common),default=0)
            if longest>=MIN_RELATION:scored.append((longest,len(common),len(factor.pair_witnesses),factor.identity))
        if not scored:return 0
        key=max(x[:3] for x in scored);w={x[3] for x in scored if x[:3]==key}
        return next(iter(w)) if len(w)==1 else 0

    def infer(self,raw,expected_atoms=()):
        raw=tuple(map(int,raw));anchors=self._resolve_anchors(raw,expected_atoms)
        if len(anchors)<2:return {'status':0,'root':0,'depth':0,'anchors':tuple(x[2] for x in anchors),'relations':()}
        relations=[]
        for left,right in zip(anchors,anchors[1:]):
            relation=self._relation_for_gap(raw[left[5]:right[4]])
            if not relation:return {'status':0,'root':0,'depth':0,'anchors':tuple(x[2] for x in anchors),'relations':tuple(relations)}
            relations.append(relation)
        root=int(anchors[0][3]);depth=0
        for relation,anchor in zip(relations,anchors[1:]):
            root=_id('grounded-merge-v1',[relation,root,int(anchor[3])]);depth+=1
        return {'status':1,'root':root,'depth':depth,'anchors':tuple(int(x[2]) for x in anchors),'relations':tuple(map(int,relations))}

    @property
    def anchor_factor_count(self):return len(self._anchors)
    @property
    def relation_factor_count(self):return len(self._relations)

    def checkpoint(self):
        def ex(row):return {'atom':row['atom'],'source':row['source'],'tick':row['tick'],'sketch':[list(x) for x in row['sketch']]}
        def rx(row):return {'source':row['source'],'pair':list(row['pair']),'tick':row['tick'],'sketch':[list(x) for x in row['sketch']]}
        return {'schema':1,'tick':self._tick,'anchor_examples':[ex(x) for x in self._anchor_examples],
            'relation_examples':[rx(x) for x in self._relation_examples],
            'anchors':[{'identity':x.identity,'atom':x.atom,'members':[list(y) for y in x.members],'sources':list(x.sources)} for x in self._anchors],
            'relations':[{'identity':x.identity,'members':[list(y) for y in x.members],'sources':list(x.sources),'pairs':[list(y) for y in x.pair_witnesses]} for x in self._relations]}

    @classmethod
    def restore(cls,data):
        if int(data.get('schema',0))!=1:raise RuntimeError('structure-inducer:checkpoint-schema')
        out=cls();out._tick=int(data.get('tick',-1))
        for r in data.get('anchor_examples',()):out._anchor_examples.append({'atom':int(r['atom']),'source':int(r['source']),'tick':int(r['tick']),'sketch':tuple((int(n),str(d)) for n,d in r['sketch'])})
        for r in data.get('relation_examples',()):out._relation_examples.append({'source':int(r['source']),'pair':tuple(map(int,r['pair'])),'tick':int(r['tick']),'sketch':tuple((int(n),str(d)) for n,d in r['sketch'])})
        for r in data.get('anchors',()):out._anchors.append(GroundedAnchorFactorV1(int(r['identity']),int(r['atom']),tuple((int(n),str(d)) for n,d in r['members']),tuple(map(int,r['sources']))))
        for r in data.get('relations',()):out._relations.append(GroundedRelationFactorV1(int(r['identity']),tuple((int(n),str(d)) for n,d in r['members']),tuple(map(int,r['sources'])),tuple(tuple(map(int,p)) for p in r['pairs'])))
        if len(out._anchor_examples)>MAX_EXAMPLES or len(out._relation_examples)>MAX_EXAMPLES or len(out._anchors)>MAX_FACTORS or len(out._relations)>MAX_FACTORS:raise RuntimeError('structure-inducer:capacity')
        return out
