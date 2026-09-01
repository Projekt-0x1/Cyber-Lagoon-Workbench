#!/usr/bin/env python3
from __future__ import annotations
from dataclasses import dataclass, field, asdict
import copy, hashlib, json

# Reference-semantic mirror of the authored DirectGenomeV1 developmental input.
# It intentionally models the meaningful authored fields, not C++ padding/layout.
ABI_CURRENT=2
MAX_TERRITORIES=64
MAX_FIELDS=256
MAX_RULES=1024
DEVELOPMENT_END=8192
PREPARED_LEARNING_END=DEVELOPMENT_END+256
MATTER_BUDGET=1<<30
DEVELOPMENT_SEED=0x53504543

EXTEND,BRANCH,FUSE,RETRACT,MATURE,REPAIR,LONG_TRACT,ENDOGENOUS_SOURCE=range(8)
PHASE_SURFACE=0
PHASE_SEQUENCE=256
PHASE_ASSOCIATION=768
PHASE_CONTROL=1536

@dataclass(frozen=True)
class TerritoryIdentity:
    lineage:int; axis:int; ordinal:int
@dataclass
class Territory:
    identity:TerritoryIdentity; reach:int; chemotype:int; begin_tick:int; flags:int=0
@dataclass
class Field:
    territory:TerritoryIdentity; radius:int; require_value:int; write_value:int; begin_tick:int; end_tick:int; polarity:int=0
@dataclass
class Rule:
    opcode:int; begin_tick:int; end_tick:int; require_value:int=0; write_value:int=0; extent:int=0; child_slot:int=0; branch_count:int=0; field_index:int=-1; minimum_age:int=0; maximum_age:int=0; threshold_q32:int=0; flags:int=0
@dataclass
class Genome:
    abi_version:int=ABI_CURRENT; life_function_version:int=2; development_end_tick:int=DEVELOPMENT_END; matter_budget:int=MATTER_BUDGET; development_seed:int=DEVELOPMENT_SEED
    territories:list[Territory]=field(default_factory=list); fields:list[Field]=field(default_factory=list); rules:list[Rule]=field(default_factory=list)
    parent_root:str=''; delta_root:str=''; verified_pool_frozen:bool=False
    def validate(self):
        if self.abi_version not in (1,2): raise ValueError('genome:abi')
        if len(self.territories)>MAX_TERRITORIES or len(self.fields)>MAX_FIELDS or len(self.rules)>MAX_RULES: raise ValueError('genome:capacity')
        ids=[t.identity for t in self.territories]
        if len(ids)!=len(set(ids)): raise ValueError('genome:duplicate_territory')
        if any(t.reach<=0 for t in self.territories): raise ValueError('genome:zero_reach')
        valid=set(ids)
        if any(f.territory not in valid for f in self.fields): raise ValueError('genome:field_owner')
        if any(r.opcode not in range(8) for r in self.rules): raise ValueError('genome:opcode')
        return True
    def root(self):
        d=asdict(self); d.pop('parent_root'); d.pop('delta_root'); d.pop('verified_pool_frozen')
        return hashlib.sha256(json.dumps(d,sort_keys=True,separators=(',',':')).encode()).hexdigest()

# provenance, territories, reach, degree, population, dense_width, long_tracts,
# begin_tick, competition_enabled, competition_q16, partner-row, competition_share
SPECIES=[
 ('NET00',3,12,12,256,64,8,PHASE_SURFACE,False,0,8,0),
 ('NET01',3,16,10,192,32,12,PHASE_SEQUENCE,False,0,8,0),
 ('NET02',3,24,8,192,32,12,PHASE_SEQUENCE,False,0,8,0),
 ('NET03',2,20,6,160,0,16,PHASE_SEQUENCE,False,0,8,0),
 ('NET04',2,10,16,128,0,10,PHASE_SEQUENCE,False,0,8,0),
 ('NET05',3,40,6,224,48,16,PHASE_ASSOCIATION,False,0,8,0),
 ('NET06',2,32,14,128,0,14,PHASE_CONTROL,False,0,8,0),
 ('NET07',2,24,6,96,0,8,PHASE_CONTROL,False,0,8,0),
 ('NET08',3,16,20,64,0,24,PHASE_SEQUENCE,False,0,14,0),
 ('NET09',3,14,10,96,0,10,PHASE_CONTROL,True,32768,8,5),
 ('NET10',4,10,8,64,16,8,PHASE_ASSOCIATION,False,0,8,0),
 ('NET15',3,14,10,192,32,10,PHASE_SURFACE,False,0,8,0),
 ('NET11',1,28,12,64,0,12,PHASE_ASSOCIATION,False,0,17,0),
 ('NET12',2,48,4,48,0,16,PHASE_ASSOCIATION,False,0,8,0),
 ('NET13',2,56,8,64,0,32,PHASE_ASSOCIATION,False,0,8,0),
 ('NET14',2,32,8,96,0,20,PHASE_SEQUENCE,False,0,8,0),
 ('NET16',1,18,8,64,0,8,PHASE_ASSOCIATION,False,0,17,0),
 ('NET17',1,12,6,48,0,6,PHASE_SURFACE,False,0,12,0),
]

def canonical_species()->Genome:
    g=Genome(); bases=[]; running=0x40
    for row in SPECIES:
        bases.append(running); running+=row[1]
    chemistry=0x40
    for family,row in enumerate(SPECIES):
        name,nterr,reach,degree,pop,dense,longtract,begin,competitive,comp_q16,partner,share=row
        for local in range(nterr):
            tid=TerritoryIdentity(family,0,local)
            g.territories.append(Territory(tid,reach,chemistry,begin))
            flags=1 if competitive else 0
            grow=Rule(EXTEND,begin,DEVELOPMENT_END,chemistry,chemistry,reach,degree,pop,minimum_age=share,threshold_q32=(comp_q16<<16 if competitive else 0),flags=flags)
            g.rules.append(grow)
            if dense:
                q=copy.deepcopy(grow); q.opcode=FUSE; q.branch_count=dense; q.threshold_q32=0; g.rules.append(q)
            recurrent=(longtract+1)//2 if nterr>1 else 0
            cross=longtract-recurrent
            for count,pfam,plocal in ((recurrent,family,(local+1)%nterr if nterr else 0),(cross,partner,local%SPECIES[partner][1])):
                if not count: continue
                fi=len(g.fields)
                g.fields.append(Field(tid,1<<20,bases[pfam]+plocal,1<<16,begin,DEVELOPMENT_END))
                q=copy.deepcopy(grow); q.opcode=LONG_TRACT; q.branch_count=count; q.field_index=fi; q.extent=0; q.minimum_age=0; q.threshold_q32=0; g.rules.append(q)
            if name in ('NET08','NET13'):
                q=copy.deepcopy(grow); q.opcode=REPAIR; q.end_tick=0; q.flags|=2; g.rules.append(q)
            chemistry+=1
    fi=len(g.fields); first=g.territories[0].identity
    g.fields.append(Field(first,1<<20,0,1<<16,DEVELOPMENT_END,PREPARED_LEARNING_END+1,polarity=1))
    g.rules.append(Rule(MATURE,0,0,field_index=fi,minimum_age=DEVELOPMENT_END,maximum_age=PREPARED_LEARNING_END,flags=2))
    g.validate(); return g

def freeze_verified_causes(base:Genome, developmental_rules:list[Rule], manifest_digest:str)->Genome:
    if base.verified_pool_frozen: raise ValueError('genome:verified_pool_already_frozen')
    out=copy.deepcopy(base); out.parent_root=base.root(); out.delta_root=manifest_digest
    out.rules.extend(copy.deepcopy(developmental_rules)); out.verified_pool_frozen=True; out.validate(); return out

if __name__=='__main__':
    g=canonical_species()
    assert len(g.territories)==42 and len(g.fields)==82 and len(g.rules)==148
    print('DIRECT_GENOME_REFERENCE GREEN abi=%d territories=%d fields=%d rules=%d matter=%d root=%s' % (g.abi_version,len(g.territories),len(g.fields),len(g.rules),g.matter_budget,g.root()[:24]))
