#!/usr/bin/env python3
"""Incremental recursive perspective induction over a continuous multi-party semantic stream.

This reference owner replaces the prior requirement for already-formed proposition Merge roots.
It incrementally composes concept atoms into recursive roots while each authenticated speaker keeps
an independent continuation stack, so interruption cannot steal another speaker's unfinished frame.
Quotation/embedding retain their attributed source as structure. Negation and imperative force are
structural operators only: this owner contains no trust, valence, action value, or response policy.
"""
from __future__ import annotations
from collections import defaultdict

MAX_NODES=65536
MAX_OPEN_PER_SOURCE=64
MAX_ROOTS=8192

K_ASSERT=1
K_NEGATE=2
K_QUOTE=3
K_EMBED=4
K_SEQUENCE=5
K_IMPERATIVE=6


def _mix(tag,*xs):
    h=0x811C9DC5 ^ int(tag)
    for x in xs:
        x=int(x)&0xffffffff
        h^=x;h=(h*0x01000193)&0xffffffff
        h^=(x>>16);h=(h*0x01000193)&0xffffffff
    return h or 1


class ContinuousPerspectiveInducerV1:
    def __init__(self):
        # speaker -> [[frame_kind,embedded_source,[child_roots...]], ...]
        self._open=defaultdict(list)
        # root -> (operator,children). No transcript/token string retention.
        self._nodes={}
        # bounded completed roots: (tick,authenticated_speaker,root)
        self._roots=[]
        self._tick=-1

    def _node(self,tag,children):
        key=(int(tag),tuple(map(int,children)))
        rid=_mix(tag,*children)
        existing=self._nodes.get(rid)
        if existing is not None and existing!=key:
            # Deterministic collision probe; identity is structural, not chronology/source prestige.
            salt=1
            while True:
                rid=_mix(0x7f000000|salt,tag,*children)
                existing=self._nodes.get(rid)
                if existing is None or existing==key:break
                salt+=1
        if rid not in self._nodes:
            if len(self._nodes)>=MAX_NODES:raise RuntimeError('stream-inducer:node-capacity')
            self._nodes[rid]=key
        return rid

    def atom(self,concept_id):
        concept_id=int(concept_id)
        if concept_id<=0:raise ValueError('stream-inducer:atom')
        return self._node(0,(concept_id,))

    def begin(self,speaker,kind,tick,embedded_source=0):
        speaker=int(speaker);kind=int(kind);tick=int(tick);embedded_source=int(embedded_source)
        if speaker<=0 or kind not in (K_ASSERT,K_NEGATE,K_QUOTE,K_EMBED,K_SEQUENCE,K_IMPERATIVE):
            raise ValueError('stream-inducer:begin')
        self._advance(tick)
        stack=self._open[speaker]
        if len(stack)>=MAX_OPEN_PER_SOURCE:raise RuntimeError('stream-inducer:depth')
        if kind in (K_QUOTE,K_EMBED) and embedded_source<=0:
            raise ValueError('stream-inducer:embedded-source')
        stack.append([kind,embedded_source,[]])
        return True

    def emit(self,speaker,concept_id,tick):
        speaker=int(speaker);self._advance(tick)
        stack=self._open.get(speaker)
        if not stack:raise RuntimeError('stream-inducer:no-open-frame')
        stack[-1][2].append(self.atom(concept_id))
        return True

    def attach_root(self,speaker,root,tick):
        speaker=int(speaker);root=int(root);self._advance(tick)
        if root not in self._nodes:raise ValueError('stream-inducer:unknown-root')
        stack=self._open.get(speaker)
        if not stack:raise RuntimeError('stream-inducer:no-open-frame')
        stack[-1][2].append(root)
        return True

    def end(self,speaker,tick):
        speaker=int(speaker);self._advance(tick)
        stack=self._open.get(speaker)
        if not stack:raise RuntimeError('stream-inducer:no-open-frame')
        kind,embedded,children=stack.pop()
        if not children:raise RuntimeError('stream-inducer:empty-frame')

        # Chomskyan recursive composition: binary left-fold Merge, not transcript storage.
        root=children[0]
        for child in children[1:]:root=self._node(K_SEQUENCE,(root,child))
        if kind in (K_QUOTE,K_EMBED):root=self._node(kind,(embedded,root))
        else:root=self._node(kind,(root,))

        if stack:
            stack[-1][2].append(root)
        else:
            if len(self._roots)>=MAX_ROOTS:self._roots.pop(0)
            self._roots.append((self._tick,speaker,root))
        return root

    def _advance(self,tick):
        tick=int(tick)
        if tick<self._tick:raise ValueError('stream-inducer:time-reversal')
        self._tick=tick

    def roots_since(self,min_tick=0):
        return tuple(x for x in self._roots if x[0]>=int(min_tick))

    def node(self,root):
        return self._nodes[int(root)]

    def is_node(self,root):
        return int(root) in self._nodes

    def checkpoint(self):
        if any(self._open.values()):raise RuntimeError('stream-inducer:open-frames')
        return {
            'schema':1,'tick':self._tick,
            'nodes':[[r,t,list(c)] for r,(t,c) in sorted(self._nodes.items())],
            'roots':[[t,s,r] for t,s,r in self._roots],
        }

    @classmethod
    def restore(cls,data):
        if int(data.get('schema',0))!=1:raise RuntimeError('stream-inducer:checkpoint-schema')
        out=cls();out._tick=int(data['tick'])
        for r,t,c in data.get('nodes',()):
            r=int(r);key=(int(t),tuple(map(int,c)))
            if r in out._nodes:raise RuntimeError('stream-inducer:duplicate-node')
            out._nodes[r]=key
        if len(out._nodes)>MAX_NODES:raise RuntimeError('stream-inducer:node-capacity')
        for row in data.get('roots',()):
            if len(row)!=3:raise RuntimeError('stream-inducer:checkpoint-root')
            t,s,r=map(int,row)
            if r not in out._nodes:raise RuntimeError('stream-inducer:dangling-root')
            out._roots.append((t,s,r))
        if len(out._roots)>MAX_ROOTS:raise RuntimeError('stream-inducer:root-capacity')
        return out
