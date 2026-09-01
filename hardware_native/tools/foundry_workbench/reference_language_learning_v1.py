#!/usr/bin/env python3
"""Strict numeric learned surface ecology over opaque resident event slots.

No tokenizer, grammar labels, expected output, prompt, or model call. Lexical
trajectories are learned from one-salient-feature surface contact. Reusable
construction order/scaffolding is induced by exact factorization of those learned
resident trajectories out of later raw surface contact.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json

MAX_SURFACE=512
MAX_TRANSIENT_SURFACE=1<<16
MAX_ATOMS=16
MAX_LEXEMES=4096
MAX_TEMPLATES=1024
MAX_DEPENDENCIES=8192
MIN_SOURCE_SUPPORT=2
PIECE_LITERAL=1
PIECE_PORT=2
INVERSE_LEXEME=1
INVERSE_TEMPLATE=2


def _units(values):
    out=tuple(int(x) for x in values)
    if not out or len(out)>MAX_SURFACE or any(x<0 or x>255 for x in out):raise ValueError('language:surface')
    return out


def _digest(tag,obj):return hashlib.sha256(tag.encode()+b'\0'+json.dumps(obj,sort_keys=True,separators=(',',':')).encode()).hexdigest()

@dataclass(frozen=True)
class SurfacePieceV1:
    kind:int
    port:int=0
    literal:tuple[int,...]=()

@dataclass(frozen=True)
class SurfaceTemplateV1:
    context:int
    arity:int
    pieces:tuple[SurfacePieceV1,...]
    support:int
    sources:tuple[int,...]
    identity:str
    tid:int=field(init=False,repr=False)
    def __post_init__(self):
        object.__setattr__(self,'tid',int(self.identity[:15],16))

@dataclass(frozen=True)
class SpanTemplateV1:
    context:int
    arity:int
    pieces:tuple[SurfacePieceV1,...]
    support:int
    sources:tuple[int,...]
    identity:str
    tid:int=field(init=False,repr=False)
    def __post_init__(self):
        object.__setattr__(self,'tid',int(self.identity[:15],16))

@dataclass(frozen=True)
class SurfaceBindingV1:
    context:int
    atoms:tuple[int,...]
    template_identity:str
    template_sources:tuple[int,...]
    candidate_touches:int
    lexical_identities:tuple[int,...]=()

@dataclass(frozen=True)
class SpanBindingV1:
    context:int
    children:tuple[tuple[int,...],...]
    template_identity:str
    template_sources:tuple[int,...]
    candidate_touches:int

class LearnedSurfaceEcologyV1:
    def __init__(self, minimum_source_support: int = MIN_SOURCE_SUPPORT):
        self.minimum_source_support = int(minimum_source_support)
        if not 1 <= self.minimum_source_support <= 16:
            raise ValueError('language:minimum_source_support')
        self._lexeme_sources:dict[tuple[int,tuple[int,...]],set[int]]={}
        self._lexeme_positive:dict[tuple[int,tuple[int,...]],set[int]]={}
        self._lexeme_counter:dict[tuple[int,tuple[int,...]],set[int]]={}
        self._template_sources:dict[tuple[int,int,tuple[SurfacePieceV1,...]],set[int]]={}
        self._role_template_topologies:dict[tuple[int,int,tuple[SurfacePieceV1,...]],set[tuple[int,int]]]={}
        self._span_sources:dict[tuple[int,int,tuple[SurfacePieceV1,...]],set[int]]={}
        self._span_reply_roles:dict[tuple[int,int],set[int]]={}
        self._span_reply_role_evidence:dict[tuple[int,int,int],int]={}
        self._form_sources:dict[tuple[int,tuple[int,...],tuple[int,...]],set[int]]={}
        self._compat_sources:dict[tuple[int,tuple[int,...]],set[int]]={}
        self._dependency_sources:dict[tuple[int,int,int],set[int]]={}
        self._withdrawn:set[int]=set()
        self._stream_observations:dict[int,set[tuple[tuple[int,...],int]]]={};self.last_segment_touches=0
        self._scene_stream_observations:set[tuple[tuple[int,...],tuple[int,...],int]]=set()
        self._scene_induced_sources:dict[tuple[int,tuple[int,...]],set[int]]={}
        # Rebuildable execution-only incidence. These indexes nominate touched
        # learned state; they are never checkpoint/evidence authority.
        self._lexeme_index:dict[int,set[tuple[int,...]]]={}
        self._template_index:dict[tuple[int,int],set[tuple[SurfacePieceV1,...]]]={}
        self._template_identity_rows:dict[int,tuple[int,int,tuple[SurfacePieceV1,...]]]={}
        self._template_identity_digest:dict[tuple[int,int,tuple[SurfacePieceV1,...]],str]={}
        self._span_index:dict[tuple[int,int],set[tuple[SurfacePieceV1,...]]]={}
        self._span_identity_pieces:dict[int,tuple[SurfacePieceV1,...]]={}
        self._span_identity_digest:dict[tuple[int,int,tuple[SurfacePieceV1,...]],str]={}
        self._form_index:dict[int,set[tuple[tuple[int,...],tuple[int,...]]]]={}
        self._condition_index:dict[int,set[tuple[int,tuple[int,...],tuple[int,...]]]]={}
        self._form_rule_index:dict[tuple[int,...],dict[tuple[int,int,tuple[int,...]],set[tuple[int,tuple[int,...],tuple[int,...]]]]]={}
        self._compat_index:dict[int,set[tuple[int,...]]]={}
        self._dependency_index:dict[int,set[tuple[int,int]]]={}
        self._inverse_form_trie={}
        self._inverse_form_rows:dict[tuple[int,...],set[tuple[int,tuple[int,...]]]]={}
        self._inverse_surface_trie={}
        self._inverse_units_features:dict[tuple[int,...],set[int]]={}
        self._identity_lexemes:dict[int,set[tuple[int,tuple[int,...]]]]={}
        self._lexeme_identity_by_key:dict[tuple[int,tuple[int,...]],int]={}
        self._inverse_template_no_literal:set[tuple[int,int,tuple[SurfacePieceV1,...]]]=set()
        self._inverse_span_trie={}
        self._inverse_span_no_literal:set[tuple[int,int,tuple[SurfacePieceV1,...]]]=set()
        self.last_lookup_touches=0
        self.last_rule_touches=0
        self.last_remote_dependency=False
        self._template_epoch=0
        self._support_epoch=0
        self._cached_template_key=None
        self._cached_template=None
        self._cached_template_epoch=-1
        self._cached_span_key=None
        self._cached_span=None
        self._cached_span_epoch=-1
        self._cached_lexemes={}

    @staticmethod
    def _trie_insert(root,sequence,value):
        node=root
        for unit in tuple(int(x) for x in sequence):node=node.setdefault(unit,{})
        node.setdefault(None,set()).add(value)

    @staticmethod
    def _trie_matches(root,raw,cursor):
        node=root;found=[];touches=0
        for offset in range(cursor,len(raw)):
            touches+=1;node=node.get(int(raw[offset]))
            if node is None:break
            found.extend(node.get(None,()))
        return tuple(found),touches

    def _index_inverse_form(self,feature:int,conditions,units):
        units=tuple(int(x) for x in units);conditions=tuple(int(x) for x in conditions)
        if not units:return
        self._inverse_form_rows.setdefault(units,set()).add((int(feature),conditions))
        self._trie_insert(self._inverse_form_trie,units,units)

    def _index_inverse_lexeme(self,feature:int,units):
        units=tuple(int(x) for x in units)
        if not units:return
        self._inverse_units_features.setdefault(units,set()).add(int(feature))
        self._identity_lexemes.setdefault(self.lexeme_identity(feature,units),set()).add((int(feature),units))
        self._trie_insert(self._inverse_surface_trie,units,(INVERSE_LEXEME,units))

    def _index_template_identity(self,context:int,arity:int,pieces):
        context=int(context);arity=int(arity);pieces=tuple(pieces)
        body={'context':context,'arity':arity,'pieces':[(p.kind,p.port,list(p.literal)) for p in pieces]}
        digest=_digest('surface-template-v1',body);identity=int(digest[:15],16)
        row=(context,arity,pieces);prior=self._template_identity_rows.get(identity)
        if prior is not None and prior!=row:raise ValueError('language:template_identity_collision')
        self._template_identity_rows[identity]=row
        self._template_identity_digest[(context,arity,pieces)]=digest

    def _index_inverse_template(self,context:int,arity:int,pieces):
        key=(int(context),int(arity),tuple(pieces));self._index_template_identity(context,arity,pieces)
        literals=[tuple(piece.literal) for piece in pieces if piece.kind==PIECE_LITERAL and piece.literal]
        if not literals:self._inverse_template_no_literal.add(key);return
        anchor=max(literals,key=lambda units:(len(units),units))
        self._trie_insert(self._inverse_surface_trie,anchor,(INVERSE_TEMPLATE,key))

    def historical_template_binding(self,template_identity:int):
        """Exact historical construction witness by identity, independent of source activity."""
        return self._template_identity_rows.get(int(template_identity))

    def historical_template_pieces(self,template_identity:int):
        row=self.historical_template_binding(template_identity)
        return None if row is None else row[2]

    def historical_lexeme_binding(self,lexeme_identity:int):
        rows=self._identity_lexemes.get(int(lexeme_identity),())
        if len(rows)!=1:return None
        return next(iter(rows))

    def historical_lexeme_units(self,lexeme_identity:int):
        row=self.historical_lexeme_binding(lexeme_identity)
        return None if row is None else row[1]

    @staticmethod
    def span_factor_identity(context:int,arity:int,pieces)->int:
        body={'context':int(context),'arity':int(arity),'pieces':[(p.kind,p.port,list(p.literal)) for p in pieces]}
        return int(_digest('surface-span-v1',body)[:15],16)

    def _index_span_identity(self,context:int,arity:int,pieces):
        pieces=tuple(pieces)
        body={'context':int(context),'arity':int(arity),'pieces':[(p.kind,p.port,list(p.literal)) for p in pieces]}
        digest=_digest('surface-span-v1',body);identity=int(digest[:15],16)
        prior=self._span_identity_pieces.get(identity)
        if prior is not None and prior!=pieces:raise ValueError('language:span_identity_collision')
        self._span_identity_pieces[identity]=pieces
        self._span_identity_digest[(int(context),int(arity),pieces)]=digest

    def _index_inverse_span(self,context:int,arity:int,pieces):
        key=(int(context),int(arity),tuple(pieces));self._index_span_identity(context,arity,pieces)
        literals=[tuple(piece.literal) for piece in pieces if piece.kind==PIECE_LITERAL and piece.literal]
        if not literals:self._inverse_span_no_literal.add(key);return
        anchor=max(literals,key=lambda units:(len(units),units))
        self._trie_insert(self._inverse_span_trie,anchor,key)

    def historical_span_pieces(self,template_identity:int):
        """Exact historical span witness by identity, independent of current source activity."""
        return self._span_identity_pieces.get(int(template_identity))

    def historical_span_identities(self,context:int,arity:int):
        """Derived historical factor identities for one structural family; no checkpoint state."""
        rows={self.span_factor_identity(c,a,pieces)
              for (c,a,pieces) in self._span_sources
              if int(c)==int(context) and int(a)==int(arity)}
        return tuple(sorted(rows))

    def _rebuild_indices(self):
        self._template_epoch+=1;self._support_epoch+=1;self._cached_lexemes.clear()
        self._lexeme_index.clear();self._template_index.clear();self._template_identity_rows.clear();self._template_identity_digest.clear();self._span_index.clear();self._span_identity_pieces.clear();self._span_identity_digest.clear();self._form_index.clear();self._condition_index.clear();self._form_rule_index.clear();self._compat_index.clear();self._dependency_index.clear();self._inverse_form_trie.clear();self._inverse_form_rows.clear();self._inverse_surface_trie.clear();self._inverse_units_features.clear();self._identity_lexemes.clear();self._lexeme_identity_by_key.clear();self._inverse_template_no_literal.clear();self._inverse_span_trie.clear();self._inverse_span_no_literal.clear()
        for feature,units in self._lexeme_sources:
            self._lexeme_index.setdefault(int(feature),set()).add(units);self._index_inverse_lexeme(int(feature),units)
        for context,arity,pieces in self._template_sources:
            self._template_index.setdefault((int(context),int(arity)),set()).add(pieces);self._index_inverse_template(context,arity,pieces)
        for context,arity,pieces in self._span_sources:
            self._span_index.setdefault((int(context),int(arity)),set()).add(pieces);self._index_inverse_span(context,arity,pieces)
        for feature,conditions,units in self._form_sources:
            key=(int(feature),conditions,units);self._form_index.setdefault(int(feature),set()).add((conditions,units))
            for condition in conditions:self._condition_index.setdefault(int(condition),set()).add(key)
            self._index_inverse_form(feature,conditions,units);self._index_form_rule(feature,conditions,units)
        for context,pattern in self._compat_sources:self._compat_index.setdefault(int(context),set()).add(pattern)
        for context,source_slot,target_slot in self._dependency_sources:
            self._dependency_index.setdefault(int(context),set()).add((int(source_slot),int(target_slot)))

    def observe_naming(self,feature:int,surface,source:int):
        units=_units(surface);key=(int(feature),units);self._lexeme_sources.setdefault(key,set()).add(int(source));self._lexeme_index.setdefault(int(feature),set()).add(units);self._index_inverse_lexeme(int(feature),units)
        for conditions,form in self._form_index.get(int(feature),()):self._index_form_rule(int(feature),conditions,form)
        self._cached_lexemes.clear()

    def _active_sources(self,sources):return tuple(sorted(s for s in sources if s not in self._withdrawn))
    def _active_count(self,sources):return sum(1 for s in sources if s not in self._withdrawn)
    def _active_set(self,sources):return {s for s in sources if s not in self._withdrawn}

    def observe_stream_naming(self,feature:int,surface,source:int):
        raw=_units(surface);feature=int(feature);source=int(source);rows=self._stream_observations.setdefault(feature,set())
        rows.add((raw,source))
        active=sorted((r,s) for r,s in rows if s not in self._withdrawn)
        unique_sources={s for _,s in active};self.last_segment_touches=0
        if len(unique_sources)<3:return None
        first=active[0][0];candidates=[];seen=set()
        for n in range(1,min(len(first),64)+1):
            for start in range(0,len(first)-n+1):
                chunk=first[start:start+n]
                if chunk in seen:continue
                seen.add(chunk);contexts=[];ok=True
                for raw_episode,_src in active:
                    positions=self._positions(raw_episode,chunk);self.last_segment_touches+=1
                    if len(positions)!=1:ok=False;break
                    pos=positions[0];left=-1 if pos==0 else raw_episode[pos-1];right=-1 if pos+n==len(raw_episode) else raw_episode[pos+n];contexts.append((left,right))
                if not ok:continue
                if len({x[0] for x in contexts})<3 or len({x[1] for x in contexts})<3:continue
                candidates.append(chunk)
        if not candidates:return None
        longest=max(len(x) for x in candidates);best=sorted(set(x for x in candidates if len(x)==longest))
        if len(best)!=1:raise ValueError('language:stream_segmentation_ambiguous')
        units=best[0];key=(feature,units);self._lexeme_sources.setdefault(key,set()).update(unique_sources);self._lexeme_index.setdefault(feature,set()).add(units);self._index_inverse_lexeme(feature,units);self._cached_lexemes.clear();return units

    def _rebuild_scene_induced_lexemes(self):
        """Settle only uniquely discriminated spans across lived ambiguous scenes."""
        for key,sources in self._scene_induced_sources.items():
            retained=self._lexeme_sources.get(key,set())-set(sources)
            if retained:self._lexeme_sources[key]=retained
            else:self._lexeme_sources.pop(key,None)
        self._scene_induced_sources={};active=tuple(
            row for row in self._scene_stream_observations if row[2] not in self._withdrawn)
        features=tuple(sorted({feature for atoms,_raw,_source in active for feature in atoms}))
        incidence={feature:frozenset((raw,source) for atoms,raw,source in active if feature in atoms)
                   for feature in features}
        for feature in features:
            positives=tuple((raw,source) for atoms,raw,source in active if feature in atoms)
            negatives=tuple((raw,source) for atoms,raw,source in active if feature not in atoms)
            sources={source for _raw,source in positives}
            if len(sources)<max(3,self.minimum_source_support) or not negatives:continue
            # Identical developmental incidence cannot identify which resident
            # referent owns which span. Preserve the competition.
            if any(other!=feature and incidence.get(other)==incidence[feature]
                   for other in features):continue
            seed=min((raw for raw,_source in positives),key=lambda row:(len(row),row))
            candidates=set();self.last_segment_touches=0
            for width in range(1,min(len(seed),64)+1):
                for start in range(len(seed)-width+1):
                    span=seed[start:start+width]
                    if span in candidates:continue
                    self.last_segment_touches+=len(positives)+len(negatives)
                    if (all(len(self._positions(raw,span))==1 for raw,_source in positives)
                            and all(not self._positions(raw,span) for raw,_source in negatives)):
                        candidates.add(span)
            if not candidates:continue
            width=max(map(len,candidates));winners=tuple(sorted(span for span in candidates if len(span)==width))
            if len(winners)!=1:continue
            key=(int(feature),winners[0]);self._lexeme_sources.setdefault(key,set()).update(sources)
            self._scene_induced_sources[key]=set(sources)
        self._rebuild_indices()

    def observe_scene_surface(self,context:int,atoms,surface,source:int):
        """Learn lexical spans and construction ports from one bound world surface.

        ``atoms`` is set-like resident scene participation. Its transport order has
        no language authority; current raw span positions determine port order.
        """
        atoms=tuple(sorted(set(map(int,atoms))));surface=_units(surface);source=int(source)
        if not atoms or len(atoms)>MAX_ATOMS or source<=0:raise ValueError('language:scene_surface')
        induced_features={feature for feature,_units in self._scene_induced_sources}
        if any(self.lexeme(feature) is None or feature in induced_features for feature in atoms):
            self._scene_stream_observations.add((atoms,surface,source))
            self._rebuild_scene_induced_lexemes()
        located=[]
        for feature in atoms:
            matches=[]
            for _support,units,_sources in self.lexeme_candidates(feature):
                positions=self._positions(surface,units)
                if len(positions)==1:matches.append((positions[0],positions[0]+len(units),feature,units))
            if not matches:return 0,(),()
            located.extend(matches)
        located.sort()
        if any(located[i][1]>located[i+1][0] for i in range(len(located)-1)):
            return 0,(),()
        ordered=tuple(row[2] for row in located)
        parts=tuple(row[3] for row in located);factor=0
        if int(context)>0:
            solutions=self._factorize_parts(parts,surface)
            if len(solutions)==1:
                pieces=solutions[0];key=(int(context),len(ordered),pieces)
                self._template_sources.setdefault(key,set()).add(source)
                self._template_index.setdefault((int(context),len(ordered)),set()).add(pieces)
                self._index_inverse_template(int(context),len(ordered),pieces)
                self._template_epoch+=1
                factor=self.construction_factor_identity(int(context),len(ordered),pieces)
                # Repeated participation of one resident referent at distinct
                # surface ports is lived co-reference evidence. Independent
                # scenes must repeat the geometry before the forward relation is live.
                for left in range(len(ordered)):
                    for right in range(left+1,len(ordered)):
                        if ordered[left]!=ordered[right]:continue
                        self.observe_dependency(int(context),left,right,source)
        return int(factor),ordered,tuple(feature for feature in atoms if self.lexeme(feature) is not None)

    def consolidate_scene_streams(self):
        """Discard rematerializable raw episodes once every touched referent settles."""
        if not self._scene_stream_observations:return 0
        features={feature for atoms,_raw,_source in self._scene_stream_observations for feature in atoms}
        if not features or any(self.lexeme(feature) is None for feature in features):return 0
        released=sum(len(raw) for _atoms,raw,_source in self._scene_stream_observations)
        self._scene_stream_observations.clear();self._scene_induced_sources.clear()
        return released

    def lexeme_observations(self,feature:int):
        rows=[];feature=int(feature);units_rows=self._lexeme_index.get(feature,());self.last_lookup_touches=len(units_rows)
        for units in units_rows:
            live=self._active_set(self._lexeme_sources[(feature,units)])
            if live:rows.append((len(live),units,tuple(sorted(live))))
        rows.sort(key=lambda x:(-x[0],x[1]));return rows

    def lexeme_candidates(self,feature:int):
        rows=[]
        for support,units,sources in self.lexeme_observations(feature):
            key=(int(feature),units);positive=len(self._lexeme_positive.get(key,()))
            counter=len(self._lexeme_counter.get(key,()))
            if support>=self.minimum_source_support:rows.append((support,units,sources))
            elif positive>counter:
                # An independently successful resident probe is intervention
                # evidence, not one more passive exposure.  Give it one margin
                # above the passive canalization threshold so action consequence
                # can revise an established name rather than merely tie it.
                rows.append((self.minimum_source_support+positive-counter+1,units,sources))
        rows.sort(key=lambda x:(-x[0],x[1]));return rows

    def provisional_lexemes(self,feature:int):
        rows=[]
        for support,units,sources in self.lexeme_observations(feature):
            key=(int(feature),units);positive=len(self._lexeme_positive.get(key,()))
            counter=len(self._lexeme_counter.get(key,()))
            if support==1 and positive==counter:rows.append((int(feature),units,sources))
        return tuple(rows)

    def settle_provisional(self,feature:int,units,ticket:int,effect:int):
        key=(int(feature),_units(units));ticket=int(ticket);effect=int(effect)
        if key not in self._lexeme_sources or ticket<=0 or effect==0:raise ValueError('language:provisional_settlement')
        positive=self._lexeme_positive.setdefault(key,set());counter=self._lexeme_counter.setdefault(key,set())
        if ticket in positive or ticket in counter:raise ValueError('language:provisional_ticket')
        (positive if effect>0 else counter).add(ticket)
        self._cached_lexemes.clear()
        return key[1] if len(positive)>len(counter) else None

    def lexeme_hypothesis_identity(self,identity:int,source:int|None=None):
        """Resolve one still-losing, non-adverse lexical hypothesis by identity."""
        keys=self._identity_lexemes.get(int(identity),())
        if len(keys)!=1:return None
        feature,units=next(iter(keys));key=(int(feature),units)
        live=self._active_set(self._lexeme_sources[key])
        if (len(live)!=1 or source is not None and live!={int(source)}
                or self.lexeme(feature)==units):return None
        positive=len(self._lexeme_positive.get(key,()))
        counter=len(self._lexeme_counter.get(key,()))
        return (int(feature),units,next(iter(live))) if positive>=counter else None

    def settle_lexeme_identity(self,identity:int,source:int|None,ticket:int,effect:int):
        keys=self._identity_lexemes.get(int(identity),())
        if len(keys)!=1:return False,None
        feature,units=next(iter(keys));live=self._active_set(self._lexeme_sources[(feature,units)])
        if len(live)!=1 or source is not None and live!={int(source)}:return False,None
        return True,self.settle_provisional(feature,units,ticket,effect)

    def _live_lexemes(self,feature:int):
        feature=int(feature)
        for units in self._lexeme_index.get(feature,()):
            sources=self._active_set(self._lexeme_sources[(feature,units)])
            if not sources:continue
            support=len(sources);key=(feature,units)
            positive=len(self._lexeme_positive.get(key,()))
            counter=len(self._lexeme_counter.get(key,()))
            if support>=self.minimum_source_support:score=support
            elif positive>counter:score=self.minimum_source_support+positive-counter+1
            else:continue
            yield score,units,sources

    def lexeme(self,feature:int):
        feature=int(feature)
        if feature in self._cached_lexemes:
            self.last_lookup_touches=0;return self._cached_lexemes[feature]
        indexed=self._lexeme_index.get(feature,())
        self.last_lookup_touches=len(indexed)
        winner=None;peak=-1;ties=0
        for units in indexed:
            n=self._active_count(self._lexeme_sources[(feature,units)])
            if n==0:continue
            if n>=self.minimum_source_support:score=n
            else:
                key=(feature,units);positive=len(self._lexeme_positive.get(key,()));counter=len(self._lexeme_counter.get(key,()))
                if positive>counter:score=self.minimum_source_support+positive-counter+1
                else:continue
            if score>peak:winner=units;peak=score;ties=1
            elif score==peak:ties+=1
        out=None if winner is None or ties!=1 else winner
        self._cached_lexemes[feature]=out;return out

    @staticmethod
    def _positions(surface,needle):
        n=len(needle);return tuple(i for i in range(0,len(surface)-n+1) if surface[i:i+n]==needle)

    def _factorizations(self,atoms:tuple[int,...],surface:tuple[int,...]):
        if not 1<=len(atoms)<=MAX_ATOMS:raise ValueError('language:arity')
        lex=[]
        for atom in atoms:
            options=[];feature=int(atom);indexed=self._lexeme_index.get(feature,())
            self.last_lookup_touches=len(indexed)
            for units in indexed:
                if not self._active_count(self._lexeme_sources[(feature,units)]):continue
                positions=self._positions(surface,units)
                if positions:options.append((units,positions))
            if len(options)!=1:return ()
            lex.append(options[0])
        solutions=[]
        def rec(port,chosen):
            if len(solutions)>1:return
            if port==len(atoms):
                spans=sorted((pos,pos+len(lex[p][0]),p) for p,pos in enumerate(chosen))
                if any(spans[i][1]>spans[i+1][0] for i in range(len(spans)-1)):return
                # Every learned atom must participate once. Remaining bytes are literal scaffold.
                pieces=[];cursor=0
                for start,end,p in spans:
                    if start>cursor:pieces.append(SurfacePieceV1(PIECE_LITERAL,0,surface[cursor:start]))
                    pieces.append(SurfacePieceV1(PIECE_PORT,p,()))
                    cursor=end
                if cursor<len(surface):pieces.append(SurfacePieceV1(PIECE_LITERAL,0,surface[cursor:]))
                solutions.append(tuple(pieces));return
            for pos in lex[port][1]:
                end=pos+len(lex[port][0]);overlap=False
                for prior,p0 in enumerate(chosen):
                    p1=p0+len(lex[prior][0])
                    if not (end<=p0 or pos>=p1):overlap=True;break
                if not overlap:rec(port+1,(*chosen,pos))
        rec(0,())
        return tuple(solutions)

    @staticmethod
    def _factorize_parts(parts,surface):
        parts=tuple(tuple(int(x) for x in part) for part in parts);surface=_units(surface)
        if not parts or len(parts)>MAX_ATOMS or any(not part for part in parts):return ()
        positions=[]
        for part in parts:
            rows=LearnedSurfaceEcologyV1._positions(surface,part)
            if not rows:return ()
            positions.append(rows)
        solutions=[]
        def rec(port,chosen):
            if len(solutions)>1:return
            if port==len(parts):
                spans=sorted((pos,pos+len(parts[p]),p) for p,pos in enumerate(chosen))
                if any(spans[i][1]>spans[i+1][0] for i in range(len(spans)-1)):return
                pieces=[];cursor=0
                for start,end,p in spans:
                    if start>cursor:pieces.append(SurfacePieceV1(PIECE_LITERAL,0,surface[cursor:start]))
                    pieces.append(SurfacePieceV1(PIECE_PORT,p,()));cursor=end
                if cursor<len(surface):pieces.append(SurfacePieceV1(PIECE_LITERAL,0,surface[cursor:]))
                solutions.append(tuple(pieces));return
            for pos in positions[port]:
                end=pos+len(parts[port]);overlap=False
                for prior,p0 in enumerate(chosen):
                    p1=p0+len(parts[prior])
                    if not (end<=p0 or pos>=p1):overlap=True;break
                if not overlap:rec(port+1,(*chosen,pos))
        rec(0,());return tuple(solutions)

    @staticmethod
    def _affix_rule(base,form):
        base=tuple(base);form=tuple(form)
        if form[:len(base)]==base:return (1,0,form[len(base):])
        if base[:len(form)]==form:return (1,len(base)-len(form),())
        if form[-len(base):]==base:return (-1,0,form[:-len(base)])
        if base[-len(form):]==form:return (-1,len(base)-len(form),())
        return None

    @staticmethod
    def _apply_affix_rule(base,rule):
        side,remove,addition=rule;base=tuple(base);remove=int(remove)
        if remove<0 or remove>len(base):return None
        stem=base[:len(base)-remove] if side==1 and remove else base if side==1 else base[remove:] if remove else base
        out=(*stem,*addition) if side==1 else (*addition,*stem)
        return tuple(out) if out and len(out)<=MAX_SURFACE else None

    @staticmethod
    def _edge_overlap(left,right,side:int):
        a=tuple(left) if int(side)<0 else tuple(reversed(left));b=tuple(right) if int(side)<0 else tuple(reversed(right))
        return next((i for i,(x,y) in enumerate(zip(a,b)) if x!=y),min(len(a),len(b)))

    def _index_form_rule(self,feature,conditions,units):
        active=self._active_set(self._form_sources[(int(feature),tuple(conditions),tuple(units))])
        for _support,base,sources in self._live_lexemes(int(feature)):
            if not active.intersection(sources):continue
            rule=self._affix_rule(base,units)
            if rule is not None:self._form_rule_index.setdefault(tuple(conditions),{}).setdefault(rule,set()).add((int(feature),base,tuple(units)))

    def _derived_form_candidates(self,feature,conditions):
        feature=int(feature);current=set(int(x) for x in conditions if int(x)!=0);self.last_rule_touches=0
        if any(set(required).issubset(current) and self._active_count(self._form_sources[(feature,required,units)])
               for required,units in self._form_index.get(feature,())):return []
        bases=tuple(self._live_lexemes(feature))
        if not bases:return []
        rows=[]
        for required,rules in self._form_rule_index.items():
            if not required or not set(required).issubset(current):continue
            for rule,evidence in rules.items():
                for _support,base,base_sources in bases:
                    features={};sources=set()
                    for donor,donor_base,form in evidence:
                        self.last_rule_touches+=1;active=self._active_set(self._form_sources[(donor,required,form)]);shared=active.intersection(base_sources)
                        if len(active)>=self.minimum_source_support and shared:
                            features[int(donor)]=max(features.get(int(donor),0),self._edge_overlap(base,donor_base,rule[0]));sources.update(shared)
                    if len(features)<2:continue
                    # Pack lexicographic (local edge evidence, donor types) into the existing scalar confidence field.
                    locality=sorted(features.values(),reverse=True)[1];confidence=locality*(MAX_LEXEMES+1)+len(features)
                    units=self._apply_affix_rule(base,rule)
                    if units is not None:rows.append((len(required),confidence,required,units,tuple(sorted(sources))))
        return rows

    def observe_form(self,feature:int,conditions,surface,source:int):
        cond=tuple(sorted(set(int(x) for x in conditions if int(x)!=0)));units=_units(surface)
        key=(int(feature),cond,units);self._form_sources.setdefault(key,set()).add(int(source));self._form_index.setdefault(int(feature),set()).add((cond,units))
        for condition in cond:self._condition_index.setdefault(int(condition),set()).add(key)
        self._index_inverse_form(int(feature),cond,units);self._index_form_rule(feature,cond,units)

    def condition_supported(self,condition:int):
        rows=self._condition_index.get(int(condition),());self.last_lookup_touches=len(rows)
        return any(self._active_count(self._form_sources[key])>=self.minimum_source_support for key in rows)

    def form(self,feature:int,conditions=(),require_conditioned:bool=False):
        current=set(int(x) for x in conditions if int(x)!=0)
        feature=int(feature);indexed=self._form_index.get(feature,());self.last_lookup_touches=len(indexed)
        winner=None;best=None;ties=0
        for required,units in indexed:
            if not set(required).issubset(current):continue
            n=self._active_count(self._form_sources[(feature,required,units)])
            if n<self.minimum_source_support:continue
            score=(len(required),n)
            if best is None or score>best:winner=units;best=score;ties=1
            elif score==best:ties+=1
        if best is None:
            derived=self._derived_form_candidates(feature,current)
            if not derived:return None if require_conditioned and current else self.lexeme(feature)
            for row in derived:
                score=(row[0],row[1])
                if best is None or score>best:winner=row[3];best=score;ties=1
                elif score==best:ties+=1
        if ties!=1:return None
        return winner

    def _live_forms(self,feature:int,conditions=(),require_conditioned:bool=False):
        current=set(int(x) for x in conditions if int(x)!=0)
        feature=int(feature);indexed=self._form_index.get(feature,());self.last_lookup_touches=len(indexed)
        any_row=False
        for required,units in indexed:
            if not set(required).issubset(current):continue
            if require_conditioned and not required:continue
            live=self._active_set(self._form_sources[(feature,required,units)])
            n=len(live)
            if n<self.minimum_source_support:continue
            any_row=True
            yield len(required),n,required,units,live
        if not any_row and current:
            yield from self._derived_form_candidates(feature,current)

    def form_candidates(self,feature:int,conditions=(),require_conditioned:bool=False):
        rows=[]
        for spec,n,required,units,active in self._live_forms(feature,conditions,require_conditioned):
            rows.append((spec,n,required,units,tuple(sorted(active))))
        rows.sort(key=lambda x:(-x[0],-x[1],x[2],x[3]));return rows

    def condition_form_candidates(self,conditions=()):
        """Forms independently witnessed on multiple entities for one condition."""
        current=set(int(x) for x in conditions if int(x));keys=set()
        for condition in current:keys.update(self._condition_index.get(condition,()))
        self.last_lookup_touches=len(keys);grouped={}
        for feature,required,units in keys:
            if not required or not set(required).issubset(current):continue
            active=self._active_set(self._form_sources[
                (int(feature),required,units)])
            if len(active)<self.minimum_source_support:continue
            row=grouped.setdefault((required,units),[set(),set(),set()])
            row[0].add(int(feature));row[1].update(active)
            for base in self._lexeme_index.get(int(feature),()):
                if self._active_count(self._lexeme_sources[(int(feature),base)])>=self.minimum_source_support:row[2].add(base)
        rows=[]
        for (required,units),(features,sources,bases) in grouped.items():
            if len(features)<2 or len(bases)<2:continue
            rows.append((len(required),len(sources),required,units,
                         tuple(sorted(sources)),len(features),len(bases)))
        rows.sort(key=lambda x:(-x[0],-x[1],-x[5],-x[6],x[2],x[3]));return rows

    def invert_form_candidates(self,surface,conditions=(),require_conditioned:bool=True):
        raw=_units(surface);current=set(int(x) for x in conditions if int(x)!=0);matched=set();touches=0
        for cursor in range(len(raw)):
            rows,used=self._trie_matches(self._inverse_form_trie,raw,cursor);touches+=used;matched.update(rows)
        out=[]
        for units in matched:
            for feature,required in self._inverse_form_rows.get(units,()):
                touches+=1
                if not set(required).issubset(current):continue
                if require_conditioned and not required:continue
                live=self._active_set(self._form_sources[(int(feature),required,units)])
                if len(live)>=self.minimum_source_support:out.append((int(feature),required,units,tuple(sorted(live))))
        derived=set()
        for cursor in range(len(raw)):
            nominations,used=self._trie_matches(self._inverse_surface_trie,raw,cursor);touches+=used
            for kind,base in nominations:
                if kind!=INVERSE_LEXEME:continue
                for feature in self._inverse_units_features.get(base,()):
                    for _specificity,_support,required,units,sources in self._derived_form_candidates(feature,current):
                        touches+=self.last_rule_touches+1
                        if tuple(raw[cursor:cursor+len(units)])==units:derived.add((int(feature),required,units,sources))
        out.extend(sorted(derived))
        self.last_lookup_touches=touches
        return tuple(sorted(out,key=lambda row:(row[0],row[1],row[2])))

    @staticmethod
    def _equality_pattern(values):
        classes={};next_class=1;out=[]
        for value in tuple(int(x) for x in values):
            if value==0:out.append(0);continue
            if value not in classes:classes[value]=next_class;next_class+=1
            out.append(classes[value])
        return tuple(out)

    def observe_compatibility(self,context:int,port_values,source:int):
        pattern=self._equality_pattern(port_values);self._compat_sources.setdefault((int(context),pattern),set()).add(int(source));self._compat_index.setdefault(int(context),set()).add(pattern);return pattern

    def compatible(self,context:int,port_values):
        pattern=self._equality_pattern(port_values);context=int(context)
        sources=self._compat_sources.get((context,pattern))
        self.last_lookup_touches=0 if sources is None else 1
        return self._active_count(sources or ())>=self.minimum_source_support

    def complete_compatibility(self,context:int,port_values):
        values=tuple(int(x) for x in port_values);context=int(context)
        patterns=self._compat_index.get(context,());self.last_lookup_touches=len(patterns)
        winner=None
        for pattern in patterns:
            if len(pattern)!=len(values) or self._active_count(self._compat_sources[(context,pattern)])<self.minimum_source_support:continue
            classes={};valid=True
            for group,value in zip(pattern,values):
                if group==0:
                    if value:valid=False;break
                elif value:
                    prior=classes.get(group)
                    if prior is not None and prior!=value:valid=False;break
                    classes[group]=value
            required={group for group in pattern if group}
            if not (valid and required==set(classes)):continue
            completed=tuple(0 if group==0 else classes[group] for group in pattern)
            if winner is None:winner=completed
            elif completed!=winner:return None
        return winner

    def observe_dependency(self,context:int,source_slot:int,target_slot:int,source:int):
        context=int(context);source_slot=int(source_slot);target_slot=int(target_slot)
        if context<=0 or source_slot<0 or target_slot<0 or source_slot==target_slot:raise ValueError('language:dependency')
        key=(context,source_slot,target_slot)
        if key not in self._dependency_sources:
            if len(self._dependency_sources)>=MAX_DEPENDENCIES:raise ValueError('language:dependency_bound')
            stack=[target_slot];seen=set()
            while stack:
                slot=stack.pop()
                if slot==source_slot:raise ValueError('language:dependency_cycle')
                if slot in seen:continue
                seen.add(slot);stack.extend(right for left,right in self._dependency_index.get(context,()) if left==slot)
        self._dependency_sources.setdefault(key,set()).add(int(source))
        self._dependency_index.setdefault(context,set()).add((source_slot,target_slot))

    def dependency_supported(self,context:int,source_slot:int,target_slot:int):
        sources=self._dependency_sources.get((int(context),int(source_slot),int(target_slot)),())
        return self._active_count(sources)>=self.minimum_source_support

    def complete_dependencies(self,context:int,port_values):
        values=list(int(x) for x in port_values);context=int(context);rows=self._dependency_index.get(context,())
        self.last_lookup_touches=len(rows);active=[]
        for source_slot,target_slot in rows:
            if source_slot>=len(values) or target_slot>=len(values):continue
            sources=self._dependency_sources[(context,source_slot,target_slot)]
            if self._active_count(sources)>=self.minimum_source_support:active.append((source_slot,target_slot))
        changed=False
        for _ in range(len(values)):
            proposals={}
            for source_slot,target_slot in active:
                value=values[source_slot]
                if not value:continue
                if values[target_slot] and values[target_slot]!=value:return None
                if not values[target_slot]:
                    prior=proposals.get(target_slot)
                    if prior is not None and prior!=value:return None
                    proposals[target_slot]=value
            if not proposals:break
            for target_slot,value in proposals.items():values[target_slot]=value
            changed=True
        return tuple(values) if changed else None

    def realize_conditioned(self,context:int,atoms,port_conditions,compat_values=None):
        atoms=tuple(int(x) for x in atoms);conditions=tuple(tuple(int(y) for y in x) for x in port_conditions)
        if len(atoms)!=len(conditions):return None
        if compat_values is not None and not self.compatible(context,compat_values):return None
        template=self.template(context,len(atoms))
        if template is None:return None
        surfaces=[]
        for atom,cond in zip(atoms,conditions):
            units=self.form(atom,cond,require_conditioned=bool(cond))
            if units is None:return None
            surfaces.append(units)
        out=[]
        for piece in template.pieces:
            if piece.kind==PIECE_LITERAL:out.extend(piece.literal)
            elif piece.kind==PIECE_PORT and 0<=piece.port<len(surfaces):out.extend(surfaces[piece.port])
            else:return None
        return tuple(out) if out and len(out)<=MAX_SURFACE else None

    def observe_span(self,context:int,children,surface,source:int):
        children=tuple(_units(x) for x in children);solutions=self._factorize_parts(children,_units(surface))
        if len(solutions)!=1:return False
        pieces=solutions[0];key=(int(context),len(children),pieces);self._span_sources.setdefault(key,set()).add(int(source));self._span_index.setdefault((int(context),len(children)),set()).add(pieces);self._index_inverse_span(context,len(children),pieces);self._template_epoch+=1;return True

    @staticmethod
    def induced_span_context(pieces):
        pieces=tuple(pieces);body={'arity':sum(1 for p in pieces if p.kind==PIECE_PORT),
            'pieces':[(p.kind,p.port,list(p.literal)) for p in pieces]}
        return int(_digest('induced-discourse-relation-v1',body)[:15],16) or 1

    def observe_induced_span(self,children,surface,source:int):
        children=tuple(_units(x) for x in children);source=int(source)
        if source<=0:return 0
        solutions=self._factorize_parts(children,_units(surface))
        if len(solutions)!=1:return 0
        pieces=solutions[0]
        if not any(piece.kind==PIECE_LITERAL and piece.literal for piece in pieces):return 0
        context=self.induced_span_context(pieces);key=(context,len(children),pieces)
        self._span_sources.setdefault(key,set()).add(source)
        self._span_index.setdefault((context,len(children)),set()).add(pieces)
        self._index_inverse_span(context,len(children),pieces);self._template_epoch+=1
        return context

    @staticmethod
    def _span_tid(template_identity):
        return int(str(template_identity)[:15],16) if isinstance(template_identity,str) else int(template_identity)

    def observe_span_reply_role(self,template_identity,port:int,source:int,evidence_delta:int=1):
        tid=self._span_tid(template_identity);port=int(port);source=int(source);delta=int(evidence_delta)
        pieces=self._span_identity_pieces.get(tid)
        if (pieces is None or source<=0 or port<0 or delta==0
                or port not in {int(p.port) for p in pieces if p.kind==PIECE_PORT}):return False
        self._span_reply_roles.setdefault((tid,port),set()).add(source)
        key=(tid,port,source);prior=self._span_reply_role_evidence.get(key,0)
        self._span_reply_role_evidence[key]=max(-8,min(8,prior+(1 if delta>0 else -1)))
        self._support_epoch+=1;return True

    def span_reply_support(self,template_identity,port:int):
        tid=self._span_tid(template_identity);port=int(port)
        sources=self._span_reply_roles.get((tid,port),())
        return sum(1 for source in sources if source not in self._withdrawn
                   and self._span_reply_role_evidence.get((tid,port,source),1)>0)

    def span_reply_port(self,template_identity):
        tid=self._span_tid(template_identity);live=[]
        for row_tid,port in self._span_reply_roles:
            if row_tid!=tid:continue
            if self.span_reply_support(tid,port)>=self.minimum_source_support:live.append(int(port))
        live=tuple(sorted(set(live)));return live[0] if len(live)==1 else None

    @staticmethod
    def _unfold_literal_ports(pieces,surface,arity:int,fixed=()):
        raw=_units(surface);arity=int(arity);ports=[None]*arity;solutions=[]
        fixed=tuple(fixed) if fixed else (None,)*arity
        if len(fixed)!=arity:return ()
        def rec(piece_index:int,cursor:int):
            if len(solutions)>1:return
            if piece_index==len(pieces):
                if cursor==len(raw) and all(row is not None for row in ports):solutions.append(tuple(ports))
                return
            piece=pieces[piece_index]
            if piece.kind==PIECE_LITERAL:
                literal=piece.literal
                if tuple(raw[cursor:cursor+len(literal)])==literal:rec(piece_index+1,cursor+len(literal))
                return
            if piece.kind!=PIECE_PORT or not 0<=piece.port<arity:return
            prior=ports[piece.port]
            if prior is not None:
                if tuple(raw[cursor:cursor+len(prior)])==prior:rec(piece_index+1,cursor+len(prior))
                return
            required=fixed[piece.port]
            if required is not None:
                required=tuple(required)
                if required and tuple(raw[cursor:cursor+len(required)])==required:
                    ports[piece.port]=required;rec(piece_index+1,cursor+len(required));ports[piece.port]=None
                return
            next_literal=None
            for later in pieces[piece_index+1:]:
                if later.kind==PIECE_LITERAL and later.literal:next_literal=tuple(later.literal);break
                if later.kind==PIECE_PORT:break
            if next_literal is not None:ends=(end for end in LearnedSurfaceEcologyV1._positions(raw,next_literal) if end>cursor)
            elif piece_index==len(pieces)-1:ends=(len(raw),)
            else:ends=range(cursor+1,len(raw)+1)
            for end in ends:
                ports[piece.port]=tuple(raw[cursor:end]);rec(piece_index+1,end);ports[piece.port]=None
                if len(solutions)>1:return
        rec(0,0);return tuple(solutions)

    def invert_span(self,surface,max_candidates:int=32):
        """Unfold current raw higher surface into transient child-surface bindings.

        Span Recipes persist only literal/child-port mathematics and source support.
        Child byte spans exist only in this current computation. Ambiguous splits are
        retained up to the explicit resource bound; overflow refuses atomically.
        """
        raw=_units(surface);max_candidates=int(max_candidates)
        if not 1<=max_candidates<=128:raise ValueError('language:span_inverse_bound')
        nominated=set(self._inverse_span_no_literal);touches=0
        for cursor in range(len(raw)):
            rows,used=self._trie_matches(self._inverse_span_trie,raw,cursor);touches+=used;nominated.update(rows)
        found=[]
        for context,arity,pieces in sorted(nominated,key=lambda row:(row[0],row[1],repr(row[2]))):
            live=self._active_set(self._span_sources[(int(context),int(arity),pieces)]);touches+=1
            if len(live)<self.minimum_source_support:continue
            children=[None]*int(arity);solutions=[];local_limit=max_candidates-len(found)
            def rec(piece_index:int,cursor:int):
                nonlocal touches
                if len(solutions)>local_limit:return
                if piece_index==len(pieces):
                    if cursor==len(raw) and all(x is not None for x in children):
                        candidate=tuple(children)
                        if candidate not in solutions:solutions.append(candidate)
                    return
                piece=pieces[piece_index]
                if piece.kind==PIECE_LITERAL:
                    literal=piece.literal;touches+=1
                    if tuple(raw[cursor:cursor+len(literal)])==literal:rec(piece_index+1,cursor+len(literal))
                    return
                if piece.kind!=PIECE_PORT or not 0<=piece.port<arity:return
                prior=children[piece.port]
                if prior is not None:
                    touches+=1
                    if tuple(raw[cursor:cursor+len(prior)])==prior:rec(piece_index+1,cursor+len(prior))
                    return
                # The child is an exposed boundary value. Enumerate only boundaries
                # that can satisfy the next literal when available; consecutive child
                # ports remain explicitly ambiguous rather than host-segmented.
                next_literal=None
                for later in pieces[piece_index+1:]:
                    if later.kind==PIECE_LITERAL and later.literal:
                        next_literal=tuple(later.literal);break
                    if later.kind==PIECE_PORT:break
                if next_literal is not None:
                    ends=self._positions(raw,next_literal)
                    ends=tuple(end for end in ends if end>cursor)
                elif piece_index==len(pieces)-1:
                    ends=(len(raw),)
                else:
                    ends=tuple(range(cursor+1,len(raw)+1))
                for end in ends:
                    touches+=1
                    if end<=cursor:continue
                    child=tuple(raw[cursor:end]);children[piece.port]=child;rec(piece_index+1,end);children[piece.port]=None
                    if len(solutions)>local_limit:return
            rec(0,0)
            if len(solutions)>local_limit:raise ValueError('language:span_inverse_capacity')
            if solutions:
                ident=_digest('surface-span-v1',{'context':int(context),'arity':int(arity),'pieces':[(p.kind,p.port,list(p.literal)) for p in pieces]})
                active=tuple(sorted(live))
                for child_rows in sorted(solutions):
                    found.append(SpanBindingV1(int(context),child_rows,ident,active,touches))
                    if len(found)>=max_candidates:return tuple(found)
        self.last_lookup_touches=touches
        return tuple(found)

    def span_candidates(self,context:int,arity:int):
        rows=[]
        context=int(context);arity=int(arity);indexed=self._span_index.get((context,arity),());self.last_lookup_touches=len(indexed)
        for pieces in indexed:
            live=self._active_set(self._span_sources[(context,arity,pieces)])
            ctx=context;a=arity
            if len(live)>=self.minimum_source_support:
                ident=_digest('surface-span-v1',{'context':ctx,'arity':a,'pieces':[(p.kind,p.port,list(p.literal)) for p in pieces]})
                rows.append(SpanTemplateV1(ctx,a,pieces,len(live),tuple(sorted(live)),ident))
        rows.sort(key=lambda x:(-x.support,x.identity));return rows

    def span_template(self,context:int,arity:int):
        context=int(context);arity=int(arity);key=(context,arity)
        if key==self._cached_span_key and self._cached_span_epoch==self._template_epoch:
            self.last_lookup_touches=0;return self._cached_span
        indexed=self._span_index.get(key,())
        self.last_lookup_touches=len(indexed)
        winner=None;peak=-1;ties=0
        for pieces in indexed:
            n=self._active_count(self._span_sources[(context,arity,pieces)])
            if n<self.minimum_source_support:continue
            if n>peak:winner=pieces;peak=n;ties=1
            elif n==peak:ties+=1
        if winner is None or ties!=1:
            self._cached_span_key=key;self._cached_span=None;self._cached_span_epoch=self._template_epoch
            return None
        active=tuple(sorted(self._active_set(self._span_sources[(context,arity,winner)])))
        ident=self._span_identity_digest.get((context,arity,winner))
        if ident is None:
            ident=_digest('surface-span-v1',{'context':context,'arity':arity,'pieces':[(p.kind,p.port,list(p.literal)) for p in winner]})
            self._span_identity_digest[(context,arity,winner)]=ident
        out=SpanTemplateV1(context,arity,winner,peak,active,ident)
        self._cached_span_key=key;self._cached_span=out;self._cached_span_epoch=self._template_epoch
        return out

    def realize_span(self,context:int,children):
        children=tuple(tuple(map(int,child)) for child in children)
        if any(not child or len(child)>MAX_TRANSIENT_SURFACE or any(value<0 or value>255 for value in child) for child in children):raise ValueError('language:transient-surface')
        template=self.span_template(context,len(children))
        if template is None:return None
        out=[]
        for piece in template.pieces:
            if piece.kind==PIECE_LITERAL:out.extend(piece.literal)
            elif piece.kind==PIECE_PORT and 0<=piece.port<len(children):out.extend(children[piece.port])
            else:return None
        return tuple(out) if out and len(out)<=MAX_TRANSIENT_SURFACE else None

    def observe_conditioned_construction(self,context:int,atoms,port_conditions,surface,source:int,compat_values=None):
        atoms=tuple(int(x) for x in atoms);conditions=tuple(tuple(int(y) for y in row) for row in port_conditions);surface=_units(surface)
        if len(atoms)!=len(conditions) or not 1<=len(atoms)<=MAX_ATOMS:return False
        if compat_values is not None and not self.compatible(context,compat_values):return False
        parts=[]
        for atom,cond in zip(atoms,conditions):
            units=self.form(atom,cond,require_conditioned=bool(cond))
            if units is None:
                found=tuple(dict.fromkeys(row[3] for row in self._live_forms(atom,cond,require_conditioned=bool(cond)) if self._positions(surface,row[3])))
                if len(found)!=1:return False
                units=found[0]
            parts.append(units)
        solutions=self._factorize_parts(parts,surface)
        if len(solutions)!=1:return False
        pieces=solutions[0];key=(int(context),len(atoms),pieces);self._template_sources.setdefault(key,set()).add(int(source));self._template_index.setdefault((int(context),len(atoms)),set()).add(pieces);self._index_inverse_template(context,len(atoms),pieces);self._template_epoch+=1;return True

    def _observe_remote_conditioned_contact(self,context,atoms,conditions,surface,source,template,fixed):
        cues=[slot for slot,condition in enumerate(conditions) if condition]
        if not cues:return False
        if len(cues)==1:cue=cues[0]
        else:
            active={(left,right) for left,right in self._dependency_index.get(int(context),())
                    if self.dependency_supported(int(context),left,right)}
            frontier=[slot for slot in cues if not any(left==slot and right in cues for left,right in active)]
            if len(frontier)!=1:return False
            cue=frontier[0]
        found=[]
        for target,condition in enumerate(conditions):
            if condition or fixed[target] is None:continue
            trial=list(fixed);trial[target]=None
            solutions=self._unfold_literal_ports(template.pieces,surface,len(atoms),tuple(trial))
            if len(solutions)==1 and solutions[0][target]!=fixed[target]:found.append((target,solutions[0]))
        if len(found)!=1:return False
        target,parts=found[0]
        if any(not units for units in parts):return False
        for slot in cues:
            active={row[3] for row in self._live_forms(int(atoms[slot]),conditions[slot],True)}
            if active and parts[slot] not in active:return False
        for slot in cues:self.observe_form(int(atoms[slot]),conditions[slot],parts[slot],int(source))
        self.observe_form(int(atoms[target]),conditions[cue],parts[target],int(source))
        self.observe_dependency(int(context),cue,target,int(source));self.last_remote_dependency=True;return True

    def observe_conditioned_contact(self,context:int,atoms,port_conditions,surface,source:int,compat_values=None):
        self.last_remote_dependency=False
        atoms=tuple(int(x) for x in atoms);conditions=tuple(tuple(int(y) for y in row if int(y)) for row in port_conditions);surface=_units(surface)
        if len(atoms)!=len(conditions) or not any(conditions):return False
        indexed=self._template_index.get((int(context),len(atoms)),());self.last_lookup_touches=len(indexed)
        source=int(source);owned=None;owned_active=();owned_hits=0
        for pieces in indexed:
            live=self._active_set(self._template_sources[(int(context),len(atoms),pieces)])
            if len(live)<self.minimum_source_support or source not in live:continue
            owned_hits+=1
            if owned_hits>1:return False
            owned=pieces;owned_active=live
        if owned is not None:
            ident=_digest('surface-template-v1',{'context':int(context),'arity':len(atoms),'pieces':[(p.kind,p.port,list(p.literal)) for p in owned]})
            template=SurfaceTemplateV1(int(context),len(atoms),owned,len(owned_active),tuple(sorted(owned_active)),ident)
        else:
            template=self.template(int(context),len(atoms))
        if template is None:return False
        fixed=[]
        for atom,cond in zip(atoms,conditions):
            if cond:fixed.append(None);continue
            hits=0;units=None
            for _score,live,sources in self._live_lexemes(int(atom)):
                if source not in sources:continue
                hits+=1
                if hits>1:return False
                units=live
            fixed.append(units if hits else self.lexeme(int(atom)))
        fixed=tuple(fixed)
        if any(not cond and units is None for cond,units in zip(conditions,fixed)):return False
        solutions=self._unfold_literal_ports(template.pieces,surface,len(atoms),fixed)
        if len(solutions)!=1:return self._observe_remote_conditioned_contact(
            context,atoms,conditions,surface,source,template,fixed)
        parts=solutions[0]
        for slot,(atom,cond,units) in enumerate(zip(atoms,conditions,parts)):
            if not units:return False
            if not cond and fixed[slot]!=units:return False
        for atom,cond,units in zip(atoms,conditions,parts):
            if cond:self.observe_form(int(atom),cond,units,int(source))
        return self.observe_conditioned_construction(context,atoms,conditions,surface,source,compat_values)

    @staticmethod
    def construction_factor_identity(context:int,arity:int,pieces)->int:
        body={'context':int(context),'arity':int(arity),'pieces':[(p.kind,p.port,list(p.literal)) for p in pieces]}
        return int(_digest('surface-template-v1',body)[:15],16)

    def observe_construction_factor(self,context:int,atoms,surface,source:int):
        atoms=tuple(int(x) for x in atoms);surface=_units(surface);solutions=self._factorizations(atoms,surface)
        if len(solutions)!=1:return 0
        pieces=solutions[0];key=(int(context),len(atoms),pieces);self._template_sources.setdefault(key,set()).add(int(source));self._template_index.setdefault((int(context),len(atoms)),set()).add(pieces);self._index_inverse_template(context,len(atoms),pieces);self._template_epoch+=1
        return self.construction_factor_identity(context,len(atoms),pieces)

    def observe_construction(self,context:int,atoms,surface,source:int):
        return bool(self.observe_construction_factor(context,atoms,surface,source))

    def observe_role_construction(self,context:int,atoms,surface,source:int,topology:int):
        """Factor a surface over opaque local relation roles from one lived tree."""
        atoms=tuple(int(x) for x in atoms);surface=_units(surface)
        solutions=self._factorizations(atoms,surface)
        if len(solutions)!=1:return False
        pieces=solutions[0];key=(int(context),len(atoms),pieces)
        self._template_sources.setdefault(key,set()).add(int(source))
        self._template_index.setdefault((int(context),len(atoms)),set()).add(pieces)
        self._index_inverse_template(context,len(atoms),pieces)
        self._role_template_topologies.setdefault(key,set()).add((int(source),int(topology)))
        self._template_epoch+=1
        return True

    def role_context_supported(self,context:int,arity:int):
        """Require independent sources, topology diversity, and real port reordering."""
        hits=0
        for pieces in self._template_index.get((int(context),int(arity)),()):
            key=(int(context),int(arity),pieces)
            ports=[int(piece.port) for piece in pieces if piece.kind==PIECE_PORT]
            if ports==sorted(ports):continue
            n=self._active_count(self._template_sources[key])
            topologies={topology for source,topology in self._role_template_topologies.get(key,())
                        if source not in self._withdrawn}
            if n<self.minimum_source_support or len(topologies)<2:continue
            hits+=1
            if hits>1:return False
        return hits==1

    def template_candidates(self,context:int,arity:int):
        rows=[]
        context=int(context);arity=int(arity);indexed=self._template_index.get((context,arity),());self.last_lookup_touches=len(indexed)
        for pieces in indexed:
            live=self._active_set(self._template_sources[(context,arity,pieces)])
            ctx=context;a=arity
            if len(live)>=self.minimum_source_support:
                ident=_digest('surface-template-v1',{'context':ctx,'arity':a,'pieces':[(p.kind,p.port,list(p.literal)) for p in pieces]})
                rows.append(SurfaceTemplateV1(ctx,a,pieces,len(live),tuple(sorted(live)),ident))
        rows.sort(key=lambda x:(-x.support,x.identity));return rows

    def template(self,context:int,arity:int):
        context=int(context);arity=int(arity);key=(context,arity)
        if key==self._cached_template_key and self._cached_template_epoch==self._template_epoch:
            self.last_lookup_touches=0;return self._cached_template
        indexed=self._template_index.get(key,())
        self.last_lookup_touches=len(indexed)
        winner=None;peak=-1;ties=0
        for pieces in indexed:
            n=self._active_count(self._template_sources[(context,arity,pieces)])
            if n<self.minimum_source_support:continue
            if n>peak:winner=pieces;peak=n;ties=1
            elif n==peak:ties+=1
        if winner is None or ties!=1:
            self._cached_template_key=key;self._cached_template=None;self._cached_template_epoch=self._template_epoch
            return None
        active=tuple(sorted(self._active_set(self._template_sources[(context,arity,winner)])))
        ident=self._template_identity_digest.get((context,arity,winner))
        if ident is None:
            ident=_digest('surface-template-v1',{'context':context,'arity':arity,'pieces':[(p.kind,p.port,list(p.literal)) for p in winner]})
            self._template_identity_digest[(context,arity,winner)]=ident
        out=SurfaceTemplateV1(context,arity,winner,peak,active,ident)
        self._cached_template_key=key;self._cached_template=out;self._cached_template_epoch=self._template_epoch
        return out

    def lexeme_identity(self,feature:int,units)->int:
        feature=int(feature);units=tuple(int(x) for x in units);key=(feature,units)
        ident=self._lexeme_identity_by_key.get(key)
        if ident is not None:return ident
        ident=int(_digest('surface-lexeme-v1',{'feature':feature,'units':list(units)})[:15],16)
        self._lexeme_identity_by_key[key]=ident;return ident

    @staticmethod
    def form_identity(feature:int,conditions,units)->int:
        return int(_digest('surface-form-v1',{'feature':int(feature),'conditions':[int(x) for x in conditions],'units':list(units)})[:15],16)

    @staticmethod
    def condition_form_identity(conditions,units)->int:
        return int(_digest('surface-condition-form-v1',{
            'conditions':[int(x) for x in conditions],
            'units':list(units)})[:15],16)

    @staticmethod
    def render_template(template:SurfaceTemplateV1,surfaces):
        surfaces=tuple(tuple(int(x) for x in row) for row in surfaces);out=[]
        if template is None or len(surfaces)!=template.arity:return None
        for piece in template.pieces:
            if piece.kind==PIECE_LITERAL:out.extend(piece.literal)
            elif piece.kind==PIECE_PORT and 0<=piece.port<len(surfaces):out.extend(surfaces[piece.port])
            else:return None
        return tuple(out) if out and len(out)<=MAX_SURFACE else None

    def realize(self,context:int,atoms):
        atoms=tuple(int(x) for x in atoms);template=self.template(context,len(atoms))
        if template is None:return None
        lex=[]
        for atom in atoms:
            units=self.lexeme(atom)
            if units is None:return None
            lex.append(units)
        out=[]
        for piece in template.pieces:
            if piece.kind==PIECE_LITERAL:out.extend(piece.literal)
            elif piece.kind==PIECE_PORT and 0<=piece.port<len(lex):out.extend(lex[piece.port])
            else:return None
        if not out or len(out)>MAX_SURFACE:return None
        return tuple(out)

    def lexical_features(self,surface):
        """Exact active learned lexical features for one raw surface, without asserting a construction."""
        units=_units(surface);rows=[]
        for feature in self._inverse_units_features.get(units,()):
            key=(int(feature),units);n=self._active_count(self._lexeme_sources.get(key,()))
            positive=len(self._lexeme_positive.get(key,()));counter=len(self._lexeme_counter.get(key,()))
            if n>=self.minimum_source_support or (n==1 and positive>=counter):rows.append(int(feature))
        return tuple(sorted(set(rows)))

    def invert_surface(self,surface,max_candidates:int=64,aliases=None):
        """Unify current raw surface against active learned templates/lexemes.

        Returned bindings are transient observer-neutral computation.  No parse or
        sentence object is persisted.  Ambiguity is preserved by returning every
        bounded candidate; callers must apply their own current resident context.
        """
        raw=_units(surface);max_candidates=int(max_candidates)
        if not 1<=max_candidates<=256:raise ValueError('language:inverse_bound')
        reverse={};touches=0;matched_units=set();nominated=set(self._inverse_template_no_literal)
        for cursor in range(len(raw)):
            rows,used=self._trie_matches(self._inverse_surface_trie,raw,cursor);touches+=used
            for kind,row in rows:
                if kind==INVERSE_LEXEME:matched_units.add(row)
                elif kind==INVERSE_TEMPLATE:nominated.add(row)
        for units in matched_units:
            for feature in self._inverse_units_features.get(units,()):
                touches+=1;n=self._active_count(self._lexeme_sources[(int(feature),units)])
                key=(int(feature),units);positive=len(self._lexeme_positive.get(key,()))
                counter=len(self._lexeme_counter.get(key,()))
                if n>=self.minimum_source_support or (n==1 and positive>=counter):
                    reverse.setdefault(units,set()).add(int(feature))
        if aliases:
            for units,features in aliases.items():
                units=tuple(int(x) for x in units)
                if units and any(tuple(raw[i:i+len(units)])==units for i,x in enumerate(raw) if int(x)==units[0]):
                    touches+=1;reverse.setdefault(units,set()).update(int(f) for f in features)
        ordered_units=tuple(sorted(reverse,key=lambda units:(-len(units),units)))
        found=[]
        for context,arity,pieces in sorted(nominated,key=lambda row:(row[0],row[1],repr(row[2]))):
                live=self._active_set(self._template_sources[(int(context),int(arity),pieces)]);touches+=1
                if len(live)<self.minimum_source_support:continue
                bindings=[None]*int(arity);binding_units=[None]*int(arity);solutions=[];local_limit=max_candidates-len(found)
                def rec(piece_index:int,cursor:int):
                    nonlocal touches
                    if len(solutions)>local_limit:return
                    if piece_index==len(pieces):
                        if cursor==len(raw) and all(x is not None for x in bindings) and all(x is not None for x in binding_units):
                            candidate=(tuple(bindings),tuple(binding_units))
                            if candidate not in solutions:solutions.append(candidate)
                        return
                    piece=pieces[piece_index]
                    if piece.kind==PIECE_LITERAL:
                        literal=piece.literal;touches+=1
                        if tuple(raw[cursor:cursor+len(literal)])==literal:rec(piece_index+1,cursor+len(literal))
                        return
                    if piece.kind!=PIECE_PORT or not 0<=piece.port<arity:return
                    prior=bindings[piece.port]
                    if prior is not None:
                        units=binding_units[piece.port];touches+=1
                        if units is not None and tuple(raw[cursor:cursor+len(units)])==units:rec(piece_index+1,cursor+len(units))
                        return
                    for units in ordered_units:
                        if tuple(raw[cursor:cursor+len(units)])!=units:continue
                        touches+=1
                        for feature in sorted(reverse[units]):
                            bindings[piece.port]=int(feature);binding_units[piece.port]=units
                            rec(piece_index+1,cursor+len(units))
                            bindings[piece.port]=None;binding_units[piece.port]=None
                            if len(solutions)>local_limit:return
                rec(0,0)
                if len(solutions)>local_limit:raise ValueError('language:inverse_capacity')
                if solutions:
                    ident=_digest('surface-template-v1',{'context':int(context),'arity':int(arity),'pieces':[(p.kind,p.port,list(p.literal)) for p in pieces]})
                    active=tuple(sorted(live))
                    for atoms,units_by_port in sorted(solutions):
                        lexical=tuple(self.lexeme_identity(int(feature),units) for feature,units in zip(atoms,units_by_port))
                        found.append(SurfaceBindingV1(int(context),atoms,ident,active,touches,lexical))
                        if len(found)>=max_candidates:return tuple(found)
        self.last_lookup_touches=touches
        return tuple(found)

    def embedded_surface_bindings(self,surface,max_candidates:int=64):
        """Find active learned construction bindings embedded in one raw contact.

        This is the indexed counterpart of repeatedly calling ``invert_surface`` on
        every substring.  It reuses the same live lexeme/template evidence and returns
        transient ``(start,end,binding)`` rows; no parse or sentence state is stored.
        """
        raw=_units(surface);max_candidates=int(max_candidates)
        if not 1<=max_candidates<=256:raise ValueError('language:embedded_inverse_bound')
        reverse={};touches=0;matched_units=set();nominated=set(self._inverse_template_no_literal)
        for cursor in range(len(raw)):
            rows,used=self._trie_matches(self._inverse_surface_trie,raw,cursor);touches+=used
            for kind,row in rows:
                if kind==INVERSE_LEXEME:matched_units.add(row)
                elif kind==INVERSE_TEMPLATE:nominated.add(row)
        for units in matched_units:
            for feature in self._inverse_units_features.get(units,()):
                touches+=1;n=self._active_count(self._lexeme_sources[(int(feature),units)])
                key=(int(feature),units);positive=len(self._lexeme_positive.get(key,()));counter=len(self._lexeme_counter.get(key,()))
                if n>=self.minimum_source_support or (n==1 and positive>=counter):reverse.setdefault(units,set()).add(int(feature))
        ordered_units=tuple(sorted(reverse,key=lambda units:(-len(units),units)))
        found=[]
        for context,arity,pieces in sorted(nominated,key=lambda row:(row[0],row[1],repr(row[2]))):
            live=self._active_set(self._template_sources[(int(context),int(arity),pieces)]);touches+=1
            if len(live)<self.minimum_source_support:continue
            starts=range(len(raw))
            if pieces and pieces[0].kind==PIECE_LITERAL and pieces[0].literal:
                starts=self._positions(raw,pieces[0].literal)
            for start in starts:
                bindings=[None]*int(arity);binding_units=[None]*int(arity);solutions=[];local_limit=max_candidates-len(found)
                def rec(piece_index:int,cursor:int):
                    nonlocal touches
                    if len(solutions)>local_limit:return
                    if piece_index==len(pieces):
                        if cursor>int(start) and all(x is not None for x in bindings) and all(x is not None for x in binding_units):
                            candidate=(int(cursor),tuple(bindings),tuple(binding_units))
                            if candidate not in solutions:solutions.append(candidate)
                        return
                    piece=pieces[piece_index]
                    if piece.kind==PIECE_LITERAL:
                        literal=piece.literal;touches+=1
                        if tuple(raw[cursor:cursor+len(literal)])==literal:rec(piece_index+1,cursor+len(literal))
                        return
                    if piece.kind!=PIECE_PORT or not 0<=piece.port<arity:return
                    prior=bindings[piece.port]
                    if prior is not None:
                        units=binding_units[piece.port];touches+=1
                        if units is not None and tuple(raw[cursor:cursor+len(units)])==units:rec(piece_index+1,cursor+len(units))
                        return
                    for units in ordered_units:
                        if tuple(raw[cursor:cursor+len(units)])!=units:continue
                        touches+=1
                        for feature in sorted(reverse[units]):
                            bindings[piece.port]=int(feature);binding_units[piece.port]=units
                            rec(piece_index+1,cursor+len(units))
                            bindings[piece.port]=None;binding_units[piece.port]=None
                            if len(solutions)>local_limit:return
                rec(0,int(start))
                if len(solutions)>local_limit:raise ValueError('language:embedded_inverse_capacity')
                if solutions:
                    ident=_digest('surface-template-v1',{'context':int(context),'arity':int(arity),'pieces':[(p.kind,p.port,list(p.literal)) for p in pieces]});active=tuple(sorted(live))
                    for end,atoms,units_by_port in sorted(solutions):
                        lexical=tuple(self.lexeme_identity(int(feature),units) for feature,units in zip(atoms,units_by_port))
                        found.append((int(start),int(end),SurfaceBindingV1(int(context),atoms,ident,active,touches,lexical)))
                        if len(found)>=max_candidates:self.last_lookup_touches=touches;return tuple(found)
        self.last_lookup_touches=touches
        return tuple(found)

    def provisional_dependency_alias_candidates(self,surface,source:int,max_candidates:int=32):
        """Discover bounded learned completions for exactly one unknown port.

        Discovery is side-effect free.  Uniqueness is a later competition law,
        so an ambiguous closure remains available for resident information seeking
        instead of being erased inside lexical inversion.
        """
        raw=_units(surface);source=int(source);max_candidates=int(max_candidates)
        if source<=0 or not 1<=max_candidates<=64:raise ValueError('language:provisional_dependency')
        reverse={};matched_units=set();nominated=set(self._inverse_template_no_literal);touches=0
        for cursor in range(len(raw)):
            rows,used=self._trie_matches(self._inverse_surface_trie,raw,cursor);touches+=used
            for kind,row in rows:
                if kind==INVERSE_LEXEME:matched_units.add(row)
                elif kind==INVERSE_TEMPLATE:nominated.add(row)
        for units in matched_units:
            for feature in self._inverse_units_features.get(units,()):
                n=self._active_count(self._lexeme_sources[(int(feature),units)]);touches+=1
                key=(int(feature),units);positive=len(self._lexeme_positive.get(key,()));counter=len(self._lexeme_counter.get(key,()))
                if n>=self.minimum_source_support or (n==1 and positive>=counter):reverse.setdefault(units,set()).add(int(feature))
        ordered_units=tuple(sorted(reverse,key=lambda units:(-len(units),units)))
        candidates=[]
        for context,arity,pieces in sorted(nominated,key=lambda row:(row[0],row[1],repr(row[2]))):
            touches+=1
            if self._active_count(self._template_sources[(int(context),int(arity),pieces)])<self.minimum_source_support or not self._dependency_index.get(int(context)):continue
            bindings=[0]*int(arity);binding_units=[None]*int(arity);unknown=[None]*int(arity)
            def rec(piece_index:int,cursor:int,unknown_port:int=-1):
                nonlocal touches
                if len(candidates)>=max_candidates:return
                if piece_index==len(pieces):
                    if cursor!=len(raw) or unknown_port<0:return
                    completed=self.complete_dependencies(int(context),bindings)
                    if completed is None or completed[unknown_port]==0:return
                    for slot,value in enumerate(bindings):
                        if slot!=unknown_port and value==0:return
                    units=unknown[unknown_port]
                    if not units:return
                    row=(int(completed[unknown_port]),tuple(units),int(context),
                         int(unknown_port),tuple(bindings),tuple(binding_units))
                    if row not in candidates:candidates.append(row)
                    return
                piece=pieces[piece_index]
                if piece.kind==PIECE_LITERAL:
                    literal=piece.literal;touches+=1
                    if tuple(raw[cursor:cursor+len(literal)])==literal:rec(piece_index+1,cursor+len(literal),unknown_port)
                    return
                if piece.kind!=PIECE_PORT or not 0<=piece.port<arity:return
                port=int(piece.port)
                if bindings[port]:
                    units=binding_units[port];touches+=1
                    if units is not None and tuple(raw[cursor:cursor+len(units)])==units:rec(piece_index+1,cursor+len(units),unknown_port)
                    return
                for units in ordered_units:
                    if tuple(raw[cursor:cursor+len(units)])!=units:continue
                    for feature in sorted(reverse[units]):
                        bindings[port]=int(feature);binding_units[port]=units
                        rec(piece_index+1,cursor+len(units),unknown_port)
                        bindings[port]=0;binding_units[port]=None
                if unknown_port>=0:return
                # Unknown port surface is bounded by whatever later learned
                # pieces can match. Try every nonempty split; exact uniqueness is
                # decided only after full template/dependency completion.
                for end in range(cursor+1,len(raw)+1):
                    unknown[port]=tuple(raw[cursor:end])
                    rec(piece_index+1,end,port)
                    unknown[port]=None
            rec(0,0)
        self.last_lookup_touches=touches
        unique=[];seen=set()
        for row in candidates:
            key=(row[0],row[1],row[2],row[3])
            if key not in seen:seen.add(key);unique.append(row)
        return tuple(unique)

    def infer_provisional_dependency_alias(self,surface,source:int,max_candidates:int=32):
        """Commit one uniquely completed learned dependency as a provisional alias."""
        unique=self.provisional_dependency_alias_candidates(
            surface,source,max_candidates)
        if len(unique)!=1:return None
        feature,units,context,target_slot,*_=unique[0]
        self.observe_naming(feature,units,source)
        return feature,units,context,target_slot

    def withdraw_source(self,source:int):self._withdrawn.add(int(source));self._template_epoch+=1;self._support_epoch+=1;self._cached_lexemes.clear()
    def restore_source(self,source:int):self._withdrawn.discard(int(source));self._template_epoch+=1;self._support_epoch+=1;self._cached_lexemes.clear()

    def checkpoint(self):
        return {'schema':9,'minimum_source_support':self.minimum_source_support,'lexemes':[{'feature':f,'units':list(u),'sources':sorted(s)} for (f,u),s in sorted(self._lexeme_sources.items())],
                'lexeme_outcomes':[{'feature':f,'units':list(u),'positive':sorted(self._lexeme_positive.get((f,u),())),'counter':sorted(self._lexeme_counter.get((f,u),()))} for f,u in sorted(set(self._lexeme_positive)|set(self._lexeme_counter))],
                'templates':[{'context':c,'arity':a,'pieces':[{'kind':p.kind,'port':p.port,'literal':list(p.literal)} for p in ps],'sources':sorted(s)} for (c,a,ps),s in self._template_sources.items()],
                'role_topologies':[{'context':c,'arity':a,'pieces':[{'kind':p.kind,'port':p.port,'literal':list(p.literal)} for p in ps],'source_topologies':[list(row) for row in sorted(s)]} for (c,a,ps),s in self._role_template_topologies.items()],
                'spans':[{'context':c,'arity':a,'pieces':[{'kind':p.kind,'port':p.port,'literal':list(p.literal)} for p in ps],'sources':sorted(s)} for (c,a,ps),s in self._span_sources.items()],
                'span_reply_roles':[{'template':tid,'port':port,'sources':sorted(src)} for (tid,port),src in sorted(self._span_reply_roles.items())],
                'span_reply_role_evidence':[{'template':tid,'port':port,'source':source,'evidence':value} for (tid,port,source),value in sorted(self._span_reply_role_evidence.items())],
                'forms':[{'feature':f,'conditions':list(cond),'units':list(units),'sources':sorted(src)} for (f,cond,units),src in self._form_sources.items()],
                'compatibility':[{'context':c,'pattern':list(pattern),'sources':sorted(src)} for (c,pattern),src in self._compat_sources.items()],
                'dependencies':[{'context':c,'source_slot':left,'target_slot':right,'sources':sorted(src)} for (c,left,right),src in self._dependency_sources.items()],
                'streams':[{'feature':f,'raw':list(raw),'source':src} for f,rows in sorted(self._stream_observations.items()) for raw,src in sorted(rows)],
                'scene_streams':[{'atoms':list(atoms),'raw':list(raw),'source':source} for atoms,raw,source in sorted(self._scene_stream_observations)],
                'scene_induced':[{'feature':feature,'units':list(units),'sources':sorted(sources)} for (feature,units),sources in sorted(self._scene_induced_sources.items())],
                'withdrawn':sorted(self._withdrawn)}

    @classmethod
    def restore(cls,d):
        if d.get('schema') not in (1,2,3,4,5,6,7,8,9):raise ValueError('language:checkpoint')
        x=cls(int(d.get('minimum_source_support', MIN_SOURCE_SUPPORT)))
        for row in d['lexemes']:x._lexeme_sources[(int(row['feature']),tuple(row['units']))]=set(map(int,row['sources']))
        for row in d.get('lexeme_outcomes',[]):
            key=(int(row['feature']),tuple(map(int,row['units'])));positive=set(map(int,row.get('positive',())));counter=set(map(int,row.get('counter',())))
            if key not in x._lexeme_sources or positive&counter or any(ticket<=0 for ticket in positive|counter):raise ValueError('language:checkpoint_lexeme_outcomes')
            if positive:x._lexeme_positive[key]=positive
            if counter:x._lexeme_counter[key]=counter
        for row in d['templates']:
            pieces=tuple(SurfacePieceV1(int(p['kind']),int(p['port']),tuple(p['literal'])) for p in row['pieces'])
            x._template_sources[(int(row['context']),int(row['arity']),pieces)]=set(map(int,row['sources']))
        for row in d.get('role_topologies',[]):
            pieces=tuple(SurfacePieceV1(int(p['kind']),int(p['port']),tuple(p['literal'])) for p in row['pieces'])
            key=(int(row['context']),int(row['arity']),pieces)
            if key not in x._template_sources:raise ValueError('language:checkpoint_role_template')
            x._role_template_topologies[key]={tuple(map(int,item)) for item in row['source_topologies']}
        for row in d.get('spans',[]):
            pieces=tuple(SurfacePieceV1(int(p['kind']),int(p['port']),tuple(p['literal'])) for p in row['pieces'])
            x._span_sources[(int(row['context']),int(row['arity']),pieces)]=set(map(int,row['sources']))
        for row in d.get('span_reply_roles',[]):
            key=(int(row['template']),int(row['port']));sources=set(map(int,row['sources']))
            if min(key)<0 or not sources or any(src<=0 for src in sources):raise ValueError('language:checkpoint_span_reply_role')
            x._span_reply_roles[key]=sources
        if int(d.get('schema',1))>=8:
            for row in d.get('span_reply_role_evidence',[]):
                key=(int(row['template']),int(row['port']),int(row['source']));value=int(row['evidence'])
                if key[2]<=0 or not -8<=value<=8 or key[2] not in x._span_reply_roles.get((key[0],key[1]),set()):raise ValueError('language:checkpoint_span_reply_role_evidence')
                x._span_reply_role_evidence[key]=value
        else:
            for (tid,port),sources in x._span_reply_roles.items():
                for source in sources:x._span_reply_role_evidence[(tid,port,source)]=1
        for row in d.get('forms',[]):x._form_sources[(int(row['feature']),tuple(row['conditions']),tuple(row['units']))]=set(map(int,row['sources']))
        for row in d.get('compatibility',[]):x._compat_sources[(int(row['context']),tuple(row['pattern']))]=set(map(int,row['sources']))
        for row in d.get('dependencies',[]):x._dependency_sources[(int(row['context']),int(row['source_slot']),int(row['target_slot']))]=set(map(int,row['sources']))
        for row in d.get('streams',[]):x._stream_observations.setdefault(int(row['feature']),set()).add((tuple(row['raw']),int(row['source'])))
        for row in d.get('scene_streams',[]):x._scene_stream_observations.add((tuple(map(int,row['atoms'])),tuple(map(int,row['raw'])),int(row['source'])))
        for row in d.get('scene_induced',[]):
            key=(int(row['feature']),tuple(map(int,row['units'])));sources=set(map(int,row['sources']))
            if key not in x._lexeme_sources or not sources<=x._lexeme_sources[key]:raise ValueError('language:checkpoint_scene_induced')
            x._scene_induced_sources[key]=sources
        x._withdrawn=set(map(int,d['withdrawn']));x._rebuild_indices()
        for (tid,port) in x._span_reply_roles:
            pieces=x._span_identity_pieces.get(tid)
            if pieces is None or port not in {int(p.port) for p in pieces if p.kind==PIECE_PORT}:raise ValueError('language:checkpoint_span_reply_role')
        return x

    def digest(self):return _digest('learned-surface-ecology-v1',self.checkpoint())
