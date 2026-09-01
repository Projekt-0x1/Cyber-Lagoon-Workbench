#!/usr/bin/env python3
"""Jointly induce entity signatures and reputation relations from grounded open speech.

Grounding comes from nonlinguistic co-presence (`grounded_target`), never a caller-supplied
language span or relation label. Raw fragments are transient. Durable state contains only
bounded hashes of target-specific and relation-specific invariants.
"""
from __future__ import annotations

import hashlib

MIN_BYTES=3
MAX_BYTES=12
MAX_SKETCH=4096
MAX_TARGETS=64
MAX_WITNESSES_PER_TARGET=3
MAX_SIGNATURE_MEMBERS=96
MAX_RELATION_MEMBERS=128
MIN_MEMBER_BYTES=3
MIN_MATCH_MEMBERS=2


def _h(units):
    return hashlib.sha256(bytes(units)).digest()[:12].hex()


def _sketch(raw):
    raw=tuple(map(int,raw));rows=[]
    upper=min(MAX_BYTES,len(raw))
    for n in range(MIN_BYTES,upper+1):
        for i in range(len(raw)-n+1):
            rows.append((n,_h(raw[i:i+n])))
    return tuple(sorted(set(rows),key=lambda x:(-x[0],x[1]))[:MAX_SKETCH])


def _intersection(rows):
    if not rows:return ()
    common=set(rows[0])
    for row in rows[1:]:common&=set(row)
    return tuple(sorted(common,key=lambda x:(-x[0],x[1])))


def _bounded(rows,limit,min_bytes=MIN_MEMBER_BYTES):
    return tuple(x for x in rows if x[0]>=min_bytes)[:limit]


class GroundedOpenSocialFactorV1:
    """Source-local fragment buffers plus bounded joint entity/relation induction."""

    def __init__(self):
        self._live={}
        self._targets={}
        self._relations={}

    @staticmethod
    def _direct_state(world,target):
        correct,wrong=world.testimony_accuracy.get(int(target),(0,0))
        if correct+wrong<=0:return 0
        return 1 if correct>wrong else (-1 if wrong>correct else 0)

    def _mature_target(self,target):
        row=self._targets.get(int(target))
        return bool(row and int(row.get('support',0))>=2 and row.get('invariant'))

    def _rebuild(self):
        mature=[(target,row) for target,row in self._targets.items() if self._mature_target(target)]
        relations={}
        for sign in (-1,1):
            peers=[(target,row) for target,row in mature if int(row['direct'])==sign]
            if len(peers)<2:continue
            # Relation structure is what survives across distinct grounded identities.
            candidates=[]
            for i,(ta,ra) in enumerate(peers):
                for tb,rb in peers[i+1:]:
                    common=_bounded(_intersection((ra['invariant'],rb['invariant'])),MAX_RELATION_MEMBERS)
                    if common:candidates.append(common)
            if not candidates:continue
            family=max(candidates,key=lambda x:(x[0][0],len(x)))
            relations[sign]=family
        self._relations=relations
        # Entity structure is what survives within one grounded identity but not across
        # same-sign identities. This jointly factors "who" from "what is said about them".
        for target,row in mature:
            relation=set(relations.get(int(row['direct']),()))
            own=[x for x in row['invariant'] if x not in relation]
            # Remove members shared with another mature target: they are wrapper/relation noise.
            other=set()
            for ot,orr in mature:
                if ot==target:continue
                other.update(orr['invariant'])
            unique=[x for x in own if x not in other]
            row['entity']=_bounded(tuple(unique),MAX_SIGNATURE_MEMBERS)

    def observe_grounded_utterance(self,adult,raw,grounded_target):
        """Learn from raw speech during nonlinguistically grounded co-presence."""
        target=int(grounded_target);direct=self._direct_state(adult.world_causal_learning,target)
        if target<=0 or direct==0:return False
        sketch=_sketch(raw)
        if not sketch:return False
        row=self._targets.get(target)
        if row is None:
            if len(self._targets)>=MAX_TARGETS:return False
            row={'direct':int(direct),'support':0,'witnesses':[],'invariant':(),'entity':()}
            self._targets[target]=row
        if int(row['direct'])!=int(direct):
            # Direct life changed: reset linguistic factorization rather than silently
            # binding contradictory developmental evidence into one entity signature.
            row={'direct':int(direct),'support':0,'witnesses':[],'invariant':(),'entity':()}
            self._targets[target]=row
        if sketch not in row['witnesses']:
            row['witnesses'].append(sketch)
            row['witnesses']=row['witnesses'][-MAX_WITNESSES_PER_TARGET:]
            row['support']=min(MAX_WITNESSES_PER_TARGET,int(row.get('support',0))+1)
        row['invariant']=_bounded(_intersection(tuple(row['witnesses'])),MAX_SKETCH,MIN_BYTES)
        self._rebuild()
        return True

    @staticmethod
    def _score(present,members):
        overlap=present&set(members)
        return (max((n for n,_ in overlap),default=0),len(overlap))

    def observe_open_utterance(self,adult,raw,speaker):
        """Apply jointly learned entity and relation factors as provisional testimony."""
        speaker=int(speaker)
        if speaker<=0:return False
        present=set(_sketch(raw))
        entities=[]
        for target,row in self._targets.items():
            members=row.get('entity',())
            score=self._score(present,members)
            if score[0]>=MIN_MEMBER_BYTES and score[1]>=MIN_MATCH_MEMBERS:
                entities.append((score,int(target)))
        if not entities:return False
        best_entity=max(score for score,_ in entities)
        targets={target for score,target in entities if score==best_entity}
        if len(targets)!=1:return False
        target=next(iter(targets))
        if target==speaker:return False
        relations=[]
        for sign,members in self._relations.items():
            score=self._score(present,members)
            if score[0]>=MIN_MEMBER_BYTES and score[1]>=MIN_MATCH_MEMBERS:
                relations.append((score,int(sign)))
        if not relations:return False
        best_rel=max(score for score,_ in relations)
        signs={sign for score,sign in relations if score==best_rel}
        if len(signs)!=1:return False
        sign=next(iter(signs))
        return bool(adult.world_causal_learning.observe_reputation_claim(speaker,target,sign>0))

    def admit_fragment(self,adult,physical_source,fragment,speaker=0,grounded_target=0,boundary=False):
        """Keep overlap source-local; arrival order never concatenates different talkers."""
        source=int(physical_source)
        if source<=0:return False
        buf=self._live.setdefault(source,bytearray())
        buf.extend(bytes(fragment))
        if len(buf)>4096:
            del self._live[source]
            return False
        if not boundary:return True
        raw=bytes(buf);del self._live[source]
        if int(grounded_target)>0:
            return self.observe_grounded_utterance(adult,raw,int(grounded_target))
        return self.observe_open_utterance(adult,raw,int(speaker))

    @property
    def live_source_count(self):
        return len(self._live)

    def checkpoint(self):
        return {
            'schema':1,
            'targets':[
                {'target':int(target),'direct':int(row['direct']),'support':int(row.get('support',0)),
                 'invariant':[[int(n),str(d)] for n,d in row.get('invariant',())],
                 'entity':[[int(n),str(d)] for n,d in row.get('entity',())]}
                for target,row in sorted(self._targets.items())
            ],
            'relations':[
                {'direct':int(sign),'members':[[int(n),str(d)] for n,d in members]}
                for sign,members in sorted(self._relations.items())
            ],
        }

    @classmethod
    def restore(cls,data):
        if int(data.get('schema',0))!=1:raise RuntimeError('grounded-open-social:checkpoint-schema')
        out=cls()
        for row in data.get('targets',()):
            target=int(row.get('target',0));direct=int(row.get('direct',0));support=int(row.get('support',0))
            invariant=tuple((int(n),str(d)) for n,d in row.get('invariant',()))
            entity=tuple((int(n),str(d)) for n,d in row.get('entity',()))
            if target<=0 or direct not in (-1,1) or not 0<=support<=MAX_WITNESSES_PER_TARGET or len(invariant)>MAX_SKETCH or len(entity)>MAX_SIGNATURE_MEMBERS:
                raise RuntimeError('grounded-open-social:checkpoint-target')
            if any(not MIN_BYTES<=n<=MAX_BYTES or len(d)!=24 for n,d in invariant+entity):
                raise RuntimeError('grounded-open-social:checkpoint-member')
            out._targets[target]={'direct':direct,'support':support,'witnesses':([invariant] if invariant else []),'invariant':invariant,'entity':entity}
        if len(out._targets)>MAX_TARGETS:raise RuntimeError('grounded-open-social:checkpoint-capacity')
        for row in data.get('relations',()):
            sign=int(row.get('direct',0));members=tuple((int(n),str(d)) for n,d in row.get('members',()))
            if sign not in (-1,1) or len(members)>MAX_RELATION_MEMBERS:
                raise RuntimeError('grounded-open-social:checkpoint-relation')
            if any(not MIN_BYTES<=n<=MAX_BYTES or len(d)!=24 for n,d in members):
                raise RuntimeError('grounded-open-social:checkpoint-member')
            out._relations[sign]=members
        return out
