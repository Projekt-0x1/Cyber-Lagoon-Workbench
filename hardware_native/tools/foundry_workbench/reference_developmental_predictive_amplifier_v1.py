#!/usr/bin/env python3
"""Developmental amplification: multimodal common-cause grounding, prediction, inquiry, replay.

No language id, semantic label, expected concept, reward scalar, or parser category is supplied.
Latent concept identities are earned from recurring multimodal common causes across independent
sources. Raw surfaces attach only after repeated source-diverse contact. Lived structured episodes
then support defeasible completion predictions. Replay composes already-earned relations into
zero-authority hypotheses and can never settle evidence by itself.
"""
from __future__ import annotations
from dataclasses import dataclass
import hashlib, json

MIN_COMMON_FEATURES=2
MIN_SURFACE_BYTES=3
MAX_SURFACE_BYTES=24
MAX_EXAMPLES=4096
MAX_CONCEPTS=1024
MAX_SURFACE_FAMILIES=4096
MAX_EPISODES=4096
Q=1<<16

def _id(tag,payload):
    raw=json.dumps([tag,payload],sort_keys=True,separators=(',',':')).encode()
    return int(hashlib.sha256(raw).hexdigest()[:16],16) or 1

def _hash(tag, raw):
    return hashlib.sha256(tag.encode()+b'\0'+bytes(raw)).digest()[:12].hex()

def _surface_sketch(raw):
    raw=tuple(map(int,raw));rows=[]
    for n in range(MIN_SURFACE_BYTES,min(MAX_SURFACE_BYTES,len(raw))+1):
        for p in range(0,len(raw)-n+1):rows.append((n,_hash('surface',raw[p:p+n])))
    return tuple(dict.fromkeys(rows))

def _tag_feature(channel,feature):return _id('multimodal-feature-v1',[int(channel),int(feature)])

def _flat_features(channels):
    out=[]
    for channel,features in channels:
        channel=int(channel)
        if channel<=0:raise ValueError('amplifier:channel')
        for feature in features:
            feature=int(feature)
            if feature>0:out.append(_tag_feature(channel,feature))
    return tuple(sorted(set(out)))

def _overlap(left,right):
    a=set(left);b=set(right);common=a&b;denom=max(1,min(len(a),len(b)))
    return common,(len(common)*Q)//denom

def _common_surface(a,b):
    return tuple(sorted(set(a)&set(b),key=lambda x:(-int(x[0]),x[1]))[:64])

@dataclass(frozen=True)
class MultimodalConceptV1:
    identity:int
    feature_members:tuple[int,...]
    sources:tuple[int,...]

@dataclass(frozen=True)
class SurfaceFamilyV1:
    concept_identity:int
    members:tuple
    sources:tuple[int,...]

@dataclass(frozen=True)
class StructuredEpisodeV1:
    relation:int
    left:int
    right:int
    source:int
    tick:int

@dataclass(frozen=True)
class ReplayHypothesisV1:
    identity:int
    relations:tuple[int,...]
    anchors:tuple[int,...]
    authority:int=0

class DevelopmentalPredictiveAmplifierV1:
    def __init__(self):
        self._examples=[];self._concepts=[];self._surface_examples=[];self._surface_families=[];self._episodes=[];self._tick=-1;self.last_match_touches=0

    def _advance(self,tick):
        tick=int(tick)
        if tick<self._tick:raise ValueError('amplifier:time-reversal')
        self._tick=tick

    def _matching_concepts(self,features):
        scored=[];touches=0
        for concept in self._concepts:
            touches+=1;common,ratio=_overlap(features,concept.feature_members)
            if len(common)>=MIN_COMMON_FEATURES and ratio>=Q*3//4:scored.append((ratio,len(common),len(concept.sources),concept))
        self.last_match_touches=touches
        if not scored:return ()
        best=max(row[:3] for row in scored)
        return tuple(row[3] for row in scored if row[:3]==best)

    def _settle_concept(self,features,source):
        candidates=[]
        for row in self._examples:
            if int(row['source'])==int(source):continue
            common,ratio=_overlap(features,row['features'])
            if len(common)>=MIN_COMMON_FEATURES and ratio>=Q*3//4:candidates.append((ratio,tuple(sorted(common)),row))
        if not candidates:return 0
        best=max(x[0] for x in candidates);families={x[1] for x in candidates if x[0]==best}
        if len(families)!=1:return 0
        family=next(iter(families));existing=next((x for x in self._concepts if x.feature_members==family),None);sources={int(source)}
        for ratio,f,row in candidates:
            if ratio==best and f==family:sources.add(int(row['source']))
        if existing:
            idx=self._concepts.index(existing);self._concepts[idx]=MultimodalConceptV1(existing.identity,family,tuple(sorted(set(existing.sources)|sources)));return existing.identity
        if len(self._concepts)>=MAX_CONCEPTS:return 0
        identity=_id('multimodal-concept-v1',list(family));self._concepts.append(MultimodalConceptV1(identity,family,tuple(sorted(sources))));return identity

    def _settle_surface(self,concept_identity,sketch,source):
        candidates=[]
        for row in self._surface_examples:
            if row['concept']!=int(concept_identity) or row['source']==int(source):continue
            family=_common_surface(row['sketch'],sketch)
            if family:candidates.append((family,row))
        if not candidates:return False
        longest=max(int(f[0][0]) for f,_ in candidates);families={tuple(f) for f,_ in candidates if int(f[0][0])==longest}
        if len(families)!=1:return False
        family=next(iter(families));sources={int(source)}
        for f,row in candidates:
            if tuple(f)==family:sources.add(int(row['source']))
        existing=next((x for x in self._surface_families if x.concept_identity==int(concept_identity) and x.members==family),None)
        if existing:
            idx=self._surface_families.index(existing);self._surface_families[idx]=SurfaceFamilyV1(existing.concept_identity,existing.members,tuple(sorted(set(existing.sources)|sources)));return True
        if len(self._surface_families)>=MAX_SURFACE_FAMILIES:return False
        self._surface_families.append(SurfaceFamilyV1(int(concept_identity),family,tuple(sorted(sources))));return True

    def observe(self,raw,channels,source,tick,independent=True):
        self._advance(tick);source=int(source);raw=tuple(map(int,raw))
        if source<=0 or not raw:raise ValueError('amplifier:contact')
        features=_flat_features(channels)
        if len(features)<MIN_COMMON_FEATURES:return 0
        concept=self._settle_concept(features,source) if independent else 0
        if not concept:
            matches=self._matching_concepts(features)
            if len(matches)==1:concept=matches[0].identity
        sketch=_surface_sketch(raw)
        if concept:
            self._settle_surface(concept,sketch,source);row={'concept':int(concept),'source':source,'tick':int(tick),'sketch':sketch}
            if not any(x['concept']==row['concept'] and x['source']==source and x['sketch']==sketch for x in self._surface_examples):
                if len(self._surface_examples)>=MAX_EXAMPLES:self._surface_examples.pop(0)
                self._surface_examples.append(row)
        example={'source':source,'tick':int(tick),'features':features}
        if not any(x['source']==source and x['features']==features for x in self._examples):
            if len(self._examples)>=MAX_EXAMPLES:self._examples.pop(0)
            self._examples.append(example)
        return int(concept)

    def ground(self,raw,channels=()):
        raw=tuple(map(int,raw));features=_flat_features(channels) if channels else ();candidates=[];concepts=self._matching_concepts(features) if features else tuple(self._concepts);sketch=set(_surface_sketch(raw))
        for concept in concepts:
            for family in self._surface_families:
                if family.concept_identity!=concept.identity:continue
                common=sketch&set(family.members);longest=max((int(n) for n,_ in common),default=0)
                if longest>=MIN_SURFACE_BYTES:candidates.append((longest,len(common),len(family.sources),concept.identity))
        if not candidates:return ()
        best=max(row[:3] for row in candidates);return tuple(sorted(set(row[3] for row in candidates if row[:3]==best)))

    def perceptual_concepts(self,channels):return tuple(x.identity for x in self._matching_concepts(_flat_features(channels)))

    def observe_structured_episode(self,relations,anchors,source,tick,independent=True):
        self._advance(tick);relations=tuple(map(int,relations));anchors=tuple(map(int,anchors));source=int(source)
        if not independent or source<=0 or len(anchors)<2 or len(relations)!=len(anchors)-1:return False
        added=False
        for relation,left,right in zip(relations,anchors,anchors[1:]):
            if min(relation,left,right)<=0:continue
            row=StructuredEpisodeV1(relation,left,right,source,int(tick))
            if row not in self._episodes:
                if len(self._episodes)>=MAX_EPISODES:self._episodes.pop(0)
                self._episodes.append(row);added=True
        return added

    def predict_completion(self,relation,left):
        relation=int(relation);left=int(left);by_right={}
        for row in self._episodes:
            if row.relation==relation and row.left==left:by_right.setdefault(row.right,set()).add(row.source)
        if not by_right:return {'status':0,'winner':0,'alternatives':(),'uncertainty':0}
        ranked=sorted(((len(srcs),right) for right,srcs in by_right.items()),reverse=True);top=ranked[0][0];winners=sorted(right for support,right in ranked if support==top)
        if len(winners)!=1:return {'status':0,'winner':0,'alternatives':tuple(winners),'uncertainty':len(winners)}
        return {'status':1,'winner':winners[0],'alternatives':tuple(sorted(by_right)),'uncertainty':max(0,len(by_right)-1)}

    def inquiry(self,relation,left):
        prediction=self.predict_completion(relation,left);alternatives=prediction['alternatives']
        if prediction['status']==1 and len(alternatives)<=1:return ()
        return tuple(alternatives) if alternatives else (0,)

    def replay_hypotheses(self,max_hypotheses=64):
        out=[];seen=set()
        for a in self._episodes:
            for b in self._episodes:
                if a.right!=b.left or a.source==b.source:continue
                key=((a.relation,b.relation),(a.left,a.right,b.right))
                if key in seen:continue
                seen.add(key);out.append(ReplayHypothesisV1(_id('replay-composition-v1',[list(key[0]),list(key[1])]),key[0],key[1],0))
                if len(out)>=int(max_hypotheses):return tuple(out)
        return tuple(out)

    @property
    def concept_count(self):return len(self._concepts)
    @property
    def surface_family_count(self):return len(self._surface_families)
    @property
    def episode_count(self):return len(self._episodes)

    def checkpoint(self):
        return {'schema':1,'tick':self._tick,'examples':[{'source':x['source'],'tick':x['tick'],'features':list(x['features'])} for x in self._examples],'concepts':[{'identity':x.identity,'features':list(x.feature_members),'sources':list(x.sources)} for x in self._concepts],'surface_examples':[{'concept':x['concept'],'source':x['source'],'tick':x['tick'],'sketch':[list(y) for y in x['sketch']]} for x in self._surface_examples],'surface_families':[{'concept':x.concept_identity,'members':[list(y) for y in x.members],'sources':list(x.sources)} for x in self._surface_families],'episodes':[x.__dict__ for x in self._episodes]}

    @classmethod
    def restore(cls,data):
        if int(data.get('schema',0))!=1:raise RuntimeError('amplifier:checkpoint-schema')
        out=cls();out._tick=int(data.get('tick',-1));out._examples=[{'source':int(x['source']),'tick':int(x['tick']),'features':tuple(map(int,x['features']))} for x in data.get('examples',())];out._concepts=[MultimodalConceptV1(int(x['identity']),tuple(map(int,x['features'])),tuple(map(int,x['sources']))) for x in data.get('concepts',())];out._surface_examples=[{'concept':int(x['concept']),'source':int(x['source']),'tick':int(x['tick']),'sketch':tuple((int(n),str(d)) for n,d in x['sketch'])} for x in data.get('surface_examples',())];out._surface_families=[SurfaceFamilyV1(int(x['concept']),tuple((int(n),str(d)) for n,d in x['members']),tuple(map(int,x['sources']))) for x in data.get('surface_families',())];out._episodes=[StructuredEpisodeV1(int(x['relation']),int(x['left']),int(x['right']),int(x['source']),int(x['tick'])) for x in data.get('episodes',())]
        if len(out._concepts)>MAX_CONCEPTS or len(out._surface_families)>MAX_SURFACE_FAMILIES or len(out._episodes)>MAX_EPISODES:raise RuntimeError('amplifier:checkpoint-capacity')
        return out
