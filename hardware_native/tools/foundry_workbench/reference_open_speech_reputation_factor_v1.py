#!/usr/bin/env python3
"""Acquire source-reputation relation factors from raw open speech without construction teaching.

Known entity surfaces come only from the resident learned language ecology.  Direct lived
source accuracy supplies the sign for acquisition examples.  The owner retains hashed
surface invariants rather than phrases/transcripts and later emits only provisional,
authenticated-speaker reputation claims through the existing causal-world owner.
"""
from __future__ import annotations

import hashlib

MIN_FACTOR_BYTES=6
MAX_FACTOR_BYTES=32
MAX_SKETCH_HASHES=384
MAX_PENDING_WITNESSES=32
MAX_SETTLED_FACTORS=32
MAX_FACTOR_MEMBERS=128
MIN_DISTINCT_TARGETS=2
MIN_TRANSFER_BYTES=8


def _h(units):
    return hashlib.sha256(bytes(units)).digest()[:12].hex()


class OpenSpeechReputationFactorV1:
    def __init__(self):
        # Pending witness sketches are content hashes only; raw speech is never retained.
        self._pending=[]
        # Settled factor families contain only hashed invariant byte spans.
        self._factors=[]

    @staticmethod
    def _direct_state(world,target):
        correct,wrong=world.testimony_accuracy.get(int(target),(0,0))
        if correct+wrong<=0:return 0
        return 1 if correct>wrong else (-1 if wrong>correct else 0)

    @staticmethod
    def _positions(raw,needle):
        raw=tuple(raw);needle=tuple(needle)
        if not needle or len(needle)>len(raw):return ()
        return tuple(i for i in range(len(raw)-len(needle)+1) if raw[i:i+len(needle)]==needle)

    def _entity_hits(self,adult,raw):
        """Return uniquely occurring learned entity lexemes inside otherwise unparsed speech."""
        raw=tuple(map(int,raw));rows=[]
        ecology=adult.language
        # This is resident learned lexical evidence, not a caller-supplied target/span label.
        for (feature,units),sources in getattr(ecology,'_lexeme_sources',{}).items():
            active=tuple(s for s in sources if s not in getattr(ecology,'_withdrawn',set()))
            if len(set(active))<getattr(ecology,'minimum_source_support',2):continue
            pos=self._positions(raw,units)
            if len(pos)==1:rows.append((int(feature),tuple(units),int(pos[0])))
        # Prefer maximal learned surfaces so a contained shorter lexeme cannot create authority.
        maximal=[]
        for row in rows:
            feature,units,pos=row
            if any(len(other[1])>len(units) and other[2]<=pos and pos+len(units)<=other[2]+len(other[1]) for other in rows):continue
            maximal.append(row)
        return tuple(sorted(set(maximal),key=lambda x:(x[2],-len(x[1]),x[0])))

    @staticmethod
    def _sketch(raw,units,pos):
        """Bounded orientation-neutral structural evidence around the learned entity slot.

        The previous whole-residual bottom-k sketch systematically discarded short relation
        factors such as ``rely on`` whenever a long wrapper supplied more than the sketch
        budget of longer substrings. Keep the local left/right slot neighborhoods instead:
        relation evidence may occur on either side of the entity, and the complete bounded
        family fits without sampling or transcript retention.
        """
        raw=tuple(map(int,raw));units=tuple(map(int,units));pos=int(pos);radius=24
        if pos<0 or pos+len(units)>len(raw):return ()
        sides=(raw[max(0,pos-radius):pos],raw[pos+len(units):min(len(raw),pos+len(units)+radius)])
        rows=[]
        for side in sides:
            upper=min(16,len(side),MAX_FACTOR_BYTES)
            for n in range(MIN_FACTOR_BYTES,upper+1):
                for i in range(len(side)-n+1):rows.append((n,_h(side[i:i+n])))
        # At radius 24 and n=6..16 this is <= 308 raw rows, below the fixed
        # fingerprint budget; no sampling can evict the local relation family.
        return tuple(sorted(set(rows),key=lambda x:(-x[0],x[1]))[:MAX_SKETCH_HASHES])

    @staticmethod
    def _common_family(a,b):
        common=sorted(set(a)&set(b),key=lambda x:(-x[0],x[1]))
        if not common or common[0][0]<MIN_TRANSFER_BYTES:return ()
        # Retain a bounded multi-scale family, not one brittle punctuation-bearing span.
        return tuple(common[:MAX_FACTOR_MEMBERS])

    def _settle(self,sign,target,sketch):
        candidates=[]
        for index,row in enumerate(self._pending):
            if int(row['direct'])!=int(sign) or int(row['target'])==int(target):continue
            family=self._common_family(row['sketch'],sketch)
            if family:candidates.append((family,index,row))
        if not candidates:return False
        longest=max(family[0][0] for family,_i,_row in candidates)
        best=[x for x in candidates if x[0][0][0]==longest]
        fingerprints={tuple(x[0]) for x in best}
        if len(fingerprints)!=1:return False
        family=next(iter(fingerprints))
        if len(self._factors)>=MAX_SETTLED_FACTORS:return False
        self._factors.append({'direct':int(sign),'targets':{int(target),int(best[0][2]['target'])},'members':family})
        used={x[1] for x in best if tuple(x[0])==family}
        self._pending=[row for i,row in enumerate(self._pending) if i not in used]
        return True

    def observe_open_contact(self,adult,raw,speaker=0):
        """Acquire from direct life when available; otherwise bind a learned factor as testimony."""
        raw=tuple(map(int,raw));speaker=int(speaker)
        hits=self._entity_hits(adult,raw)
        if len(hits)!=1:return False
        target,units,pos=hits[0]
        if speaker>0 and speaker==target:return False
        sketch=self._sketch(raw,units,pos)
        if not sketch:return False
        direct=self._direct_state(adult.world_causal_learning,target)
        if direct:
            settled=self._settle(direct,target,sketch)
            if not settled:
                row={'direct':int(direct),'target':int(target),'sketch':sketch}
                duplicate=any(x['direct']==row['direct'] and x['target']==row['target'] and x['sketch']==row['sketch'] for x in self._pending)
                if not duplicate:
                    if len(self._pending)>=MAX_PENDING_WITNESSES:self._pending.pop(0)
                    self._pending.append(row)
            return True
        if speaker<=0:return False
        present=set(sketch);matched=[]
        for row in self._factors:
            overlap=present&set(row['members'])
            score=max((n for n,_digest in overlap),default=0)
            if score>=MIN_TRANSFER_BYTES and len(row['targets'])>=MIN_DISTINCT_TARGETS:matched.append((score,int(row['direct'])))
        if not matched:return False
        best=max(score for score,_sign in matched);signs={sign for score,sign in matched if score==best}
        if len(signs)!=1:return False
        sign=next(iter(signs))
        return bool(adult.world_causal_learning.observe_reputation_claim(speaker,target,sign>0))

    def checkpoint(self):
        return {'schema':1,'factors':[
            {'direct':int(row['direct']),'targets':sorted(map(int,row['targets'])),
             'members':[[int(n),str(digest)] for n,digest in row['members']]}
            for row in self._factors],
            'pending':[
            {'direct':int(row['direct']),'target':int(row['target']),
             'sketch':[[int(n),str(digest)] for n,digest in row['sketch']]}
            for row in self._pending]}

    @classmethod
    def restore(cls,data):
        if int(data.get('schema',0))!=1:raise RuntimeError('open-reputation:checkpoint-schema')
        out=cls()
        for row in data.get('factors',()):
            direct=int(row.get('direct',0));targets=set(map(int,row.get('targets',())));members=tuple((int(n),str(d)) for n,d in row.get('members',()))
            if direct not in (-1,1) or len(targets)<MIN_DISTINCT_TARGETS or len(members)>MAX_FACTOR_MEMBERS or any(not MIN_FACTOR_BYTES<=n<=MAX_FACTOR_BYTES or len(d)!=24 for n,d in members):
                raise RuntimeError('open-reputation:checkpoint-factor')
            out._factors.append({'direct':direct,'targets':targets,'members':members})
        for row in data.get('pending',()):
            direct=int(row.get('direct',0));target=int(row.get('target',0));sketch=tuple((int(n),str(d)) for n,d in row.get('sketch',()))
            if direct not in (-1,1) or target<=0 or len(sketch)>MAX_SKETCH_HASHES or any(not MIN_FACTOR_BYTES<=n<=MAX_FACTOR_BYTES or len(d)!=24 for n,d in sketch):
                raise RuntimeError('open-reputation:checkpoint-pending')
            out._pending.append({'direct':direct,'target':target,'sketch':sketch})
        if len(out._factors)>MAX_SETTLED_FACTORS or len(out._pending)>MAX_PENDING_WITNESSES:raise RuntimeError('open-reputation:checkpoint-capacity')
        return out
