#!/usr/bin/env python3
"""Packed cold entity-relation authority with rebuildable local incidence.

This is a storage/lowering candidate, not an entity/concept module.  Durable
state is only opaque entity -> relation-id incidence.  A current mental-map
neighborhood is reconstructed transiently from exact relation overlap.
"""
from __future__ import annotations
from array import array
from bisect import bisect_left,bisect_right
import hashlib,struct,time

_MAGIC=b'PER1';_VERSION=1
_HEADER=struct.Struct('<4sB3xII32s')
_ROW=struct.Struct('<QH')
_U64=struct.Struct('<Q')
MAX_FEATURES=64

class PackedEntityRelationsV1:
    def __init__(self,body:bytes=b''):
        self.body=bytes(body)
        self.entity_ids=array('Q');self.offsets=array('I')
        self.feature_keys=array('Q');self.feature_entities=array('Q')
        self.last_entity_touches=0;self.last_feature_touches=0
        if self.body:self._rebuild()

    @classmethod
    def from_mapping(cls,mapping):
        body=bytearray();prior=0
        for entity,features in sorted((int(k),tuple(sorted(set(map(int,v))))) for k,v in mapping.items()):
            if entity<=0 or entity<=prior or not features or len(features)>MAX_FEATURES or any(x<=0 for x in features):
                raise ValueError('packed-entity:row')
            body.extend(_ROW.pack(entity,len(features)))
            for feature in features:body.extend(_U64.pack(feature))
            prior=entity
        return cls(bytes(body))

    def _rebuild(self):
        self.entity_ids=array('Q');self.offsets=array('I');pairs=[];p=0;prior=0
        while p<len(self.body):
            if p+_ROW.size>len(self.body):raise ValueError('packed-entity:body')
            entity,count=_ROW.unpack_from(self.body,p)
            if entity<=prior or count==0 or count>MAX_FEATURES:raise ValueError('packed-entity:body')
            start=p;p+=_ROW.size;need=count*_U64.size
            if p+need>len(self.body):raise ValueError('packed-entity:body')
            features=[]
            for _ in range(count):
                feature=_U64.unpack_from(self.body,p)[0];p+=_U64.size
                if feature<=0:raise ValueError('packed-entity:feature')
                features.append(feature);pairs.append((feature,entity))
            if features!=sorted(set(features)):raise ValueError('packed-entity:feature_order')
            self.entity_ids.append(entity);self.offsets.append(start);prior=entity
        pairs.sort();self.feature_keys=array('Q',(row[0] for row in pairs));self.feature_entities=array('Q',(row[1] for row in pairs))

    def features(self,entity:int):
        entity=int(entity);i=bisect_left(self.entity_ids,entity);self.last_entity_touches=max(1,(len(self.entity_ids).bit_length()))
        if i>=len(self.entity_ids) or self.entity_ids[i]!=entity:return ()
        p=int(self.offsets[i]);stored,count=_ROW.unpack_from(self.body,p);p+=_ROW.size
        if stored!=entity:raise ValueError('packed-entity:index')
        return tuple(_U64.unpack_from(self.body,p+j*_U64.size)[0] for j in range(count))

    def entities_with_feature(self,feature:int):
        feature=int(feature);lo=bisect_left(self.feature_keys,feature);hi=bisect_right(self.feature_keys,feature)
        self.last_feature_touches=hi-lo
        return tuple(int(self.feature_entities[i]) for i in range(lo,hi))

    def overlap(self,entity:int,numerator:int=3,denominator:int=4):
        features=self.features(entity)
        if not features:return ()
        nominated={int(entity)};touches=0
        for feature in features:
            rows=self.entities_with_feature(feature);touches+=len(rows);nominated.update(rows)
        out=[];base=set(features)
        for other in nominated:
            other_features=self.features(other)
            hits=sum(1 for f in other_features if f in base)
            if hits and hits*denominator>=len(features)*numerator:out.append(other)
        self.last_feature_touches=touches
        return tuple(sorted(out))

    @property
    def entity_count(self):return len(self.entity_ids)
    @property
    def relation_count(self):return len(self.feature_keys)
    @property
    def persistent_bytes(self):return _HEADER.size+len(self.body)
    @property
    def runtime_index_payload_bytes(self):
        # Packed/silicon-equivalent payload; Python object overhead is not authority.
        return 8*len(self.entity_ids)+4*len(self.offsets)+8*len(self.feature_keys)+8*len(self.feature_entities)

    def checkpoint(self):
        digest=hashlib.sha256(self.body).digest()
        return _HEADER.pack(_MAGIC,_VERSION,self.entity_count,self.relation_count,digest)+self.body

    @classmethod
    def restore(cls,blob):
        blob=bytes(blob)
        if len(blob)<_HEADER.size:raise ValueError('packed-entity:checkpoint')
        magic,version,nentity,nrelation,digest=_HEADER.unpack_from(blob,0);body=blob[_HEADER.size:]
        if magic!=_MAGIC or version!=_VERSION or hashlib.sha256(body).digest()!=digest:raise ValueError('packed-entity:checkpoint')
        out=cls(body)
        if out.entity_count!=nentity or out.relation_count!=nrelation or out.checkpoint()!=blob:raise ValueError('packed-entity:checkpoint_roundtrip')
        return out


def _mapping(n):return {1_000_000+i:(2_000_000+i*2,2_000_001+i*2) for i in range(n)}

def main():
    started=time.perf_counter();checks={};rows=[]
    for n in (1_000,10_000,50_000,250_000):
        t=time.perf_counter();store=PackedEntityRelationsV1.from_mapping(_mapping(n));build_ms=(time.perf_counter()-t)*1000
        target=1_000_000+n//2;t=time.perf_counter();got=store.overlap(target);lookup_ms=(time.perf_counter()-t)*1000
        rows.append({'entities':n,'relations':store.relation_count,'persistent_bytes':store.persistent_bytes,'runtime_index_payload_bytes':store.runtime_index_payload_bytes,'build_ms':round(build_ms,3),'lookup_ms':round(lookup_ms,6),'candidate_touches':store.last_feature_touches})
        checks[f'local_lookup_{n}']=got==(target,) and store.last_feature_touches==2
    related=_mapping(50_000);related[9_000_000]=(1,2,3,4);related[9_000_001]=(1,2,3,9)
    store=PackedEntityRelationsV1.from_mapping(related);checks['shared_relation_reinstates_related_identity']=store.overlap(9_000_000)==(9_000_000,9_000_001) and store.last_feature_touches==7
    cp=store.checkpoint();restored=PackedEntityRelationsV1.restore(cp)
    checks['checkpoint_exact']=restored.checkpoint()==cp and restored.overlap(9_000_000)==store.overlap(9_000_000)
    bad=bytearray(cp);bad[-1]^=1
    try:PackedEntityRelationsV1.restore(bad);tamper=False
    except ValueError:tamper=True
    checks['tamper_refuses']=tamper
    checks['checkpoint_has_no_runtime_index']=len(cp)==store.persistent_bytes and store.runtime_index_payload_bytes>0
    checks['no_media_or_semantic_object']=not any(hasattr(store,name) for name in ('image','audio','word','sentence','concept','thought','prompt','answer'))
    out={'schema':'agi.reference-packed-entity-relations.v1','pass':all(checks.values()),'checks':checks,'scale':rows,'related_candidate_touches':store.last_feature_touches,'reference_only':True,'physical_direct_parity':'NOT_RUN/RED','graph_flip':False,'claim':'PACKED_RELATIONAL_ENTITY_AUTHORITY_WITH_EPHEMERAL_LOCAL_RECONSTRUCTION_REFERENCE_ONLY','elapsed_ms':round((time.perf_counter()-started)*1000,3)}
    import json;print('FOUNDRY_PACKED_ENTITY_RELATIONS '+('GREEN' if out['pass'] else 'RED')+' cold_relation_rows=1 local_map=1');print(json.dumps(out,indent=2,sort_keys=True));raise SystemExit(0 if out['pass'] else 1)
if __name__=='__main__':main()
