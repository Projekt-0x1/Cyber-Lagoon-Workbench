#!/usr/bin/env python3
from __future__ import annotations
from dataclasses import dataclass,asdict
import hashlib,json
from direct_genome_reference import canonical_species,Genome,LONG_TRACT,REPAIR,MATURE

@dataclass
class DevelopmentPlan:
    genome_root:str
    territory_count:int
    field_count:int
    rule_count:int
    long_tract_rule_count:int
    repair_rule_count:int
    mature_rule_count:int
    development_end_tick:int
    matter_budget:int
    family_counts:dict[int,int]
    begin_tick_histogram:dict[int,int]
    plan_digest:str=''


def plan_direct_life_function(genome:Genome)->DevelopmentPlan:
    genome.validate()
    fam={}; phases={}
    for t in genome.territories:
        fam[t.identity.lineage]=fam.get(t.identity.lineage,0)+1
        phases[t.begin_tick]=phases.get(t.begin_tick,0)+1
    p=DevelopmentPlan(
        genome.root(),len(genome.territories),len(genome.fields),len(genome.rules),
        sum(r.opcode==LONG_TRACT for r in genome.rules),
        sum(r.opcode==REPAIR for r in genome.rules),
        sum(r.opcode==MATURE for r in genome.rules),
        genome.development_end_tick,genome.matter_budget,fam,phases)
    body=asdict(p); body['plan_digest']=''
    p.plan_digest=hashlib.sha256(json.dumps(body,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    return p


def reference_birth_spec(genome:Genome|None=None)->dict:
    g=genome or canonical_species(); p=plan_direct_life_function(g)
    # Birth state is a checkpoint ABI. Canonicalize through JSON now so integer
    # mapping keys cannot silently change type on save/restore and alter state identity.
    raw={'schema':'0x1.direct-reference-birth.v1','genome':asdict(g),'development_plan':asdict(p),'external_life_function_detached':True,'resident_development_present':True,'final_connectome_loaded':False}
    return json.loads(json.dumps(raw,sort_keys=True,separators=(',',':')))

if __name__=='__main__':
    g=canonical_species(); p=plan_direct_life_function(g)
    assert (p.territory_count,p.field_count,p.rule_count)==(42,82,148)
    assert p.long_tract_rule_count>0 and p.repair_rule_count>0 and p.mature_rule_count==1
    print('REFERENCE_LIFE_FUNCTION GREEN territories=%d fields=%d rules=%d long_tract_rules=%d repair_rules=%d mature_rules=%d plan=%s' % (p.territory_count,p.field_count,p.rule_count,p.long_tract_rule_count,p.repair_rule_count,p.mature_rule_count,p.plan_digest[:24]))
