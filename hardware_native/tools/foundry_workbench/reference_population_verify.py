#!/usr/bin/env python3
"""Adversarial checks for the strict numeric population substrate."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))

from reference_population_v1 import PopulationBankV1,PopulationSpecV1


def allocated_bytes(bank:PopulationBankV1)->int:
    return bank.numeric_allocation_bytes()


def code_occupancy(bank:PopulationBankV1,feature_count=1024):
    sites=[]
    for feature in range(1,feature_count+1): sites.extend(bank.feature_sites(feature))
    return {'assigned':len(sites),'unique':len(set(sites)),'collision_count':len(sites)-len(set(sites))}


def main():
    started=time.perf_counter();checks={};rows=[]
    sizes=(4096,32768,131072)
    banks=[]
    for n in sizes:
        t=time.perf_counter();b=PopulationBankV1(PopulationSpecV1(n,fanout=2,sites_per_feature=3,eligibility_horizon=4));init_ms=(time.perf_counter()-t)*1000
        banks.append(b);occ=code_occupancy(b)
        rows.append({'sites':n,'edges':b.allocated_edge_count,'allocated_bytes':allocated_bytes(b),'bytes_per_site_equivalent':round(allocated_bytes(b)/n,3),'init_ms':round(init_ms,3),**occ})
    checks['allocation_linear']=all(row['edges']==row['sites']*2 for row in rows) and rows[1]['allocated_bytes']==rows[0]['allocated_bytes']*8 and rows[2]['allocated_bytes']==rows[0]['allocated_bytes']*32
    checks['larger_population_reduces_code_collisions']=rows[0]['collision_count']>rows[1]['collision_count']>=rows[2]['collision_count']

    a=PopulationBankV1(PopulationSpecV1(32768,fanout=3,sites_per_feature=4,eligibility_horizon=3));b=PopulationBankV1(a.spec)
    checks['deterministic_initial_state']=a.digest()==b.digest()
    sig=a.signature((101,202,303));checks['deterministic_sparse_signature']=sig==b.signature((101,202,303)) and 4<=len(sig)<=24
    occ=a.recruit((101,202,303));q=a.quantity_vector(occ,alternatives=2,horizon=7,trajectory=11)
    checks['quantity_vector_real']=q['R']==a.spec.site_count and q['I']==a.allocated_edge_count and q['O']==1 and q['P']==len(occ.sites) and q['E']==len(occ.sites) and q['G']==len(occ.edges) and q['A']==2 and q['H']==7 and q['T']==len(occ.edges) and q['Y']==11 and q['F']==len(occ.sites) and q['C']==a.spec.site_count-len(occ.sites)

    # Independence alone and zero-effect consequence do not earn positive credit.
    zero=PopulationBankV1(a.spec);oz=zero.recruit((11,22));before=zero.digest();z=zero.settle(oz,0,True)
    checks['independence_not_credit']=z=={'credit':0,'revisions':0} and zero.credit_events==0 and zero.revision_events==0
    endogenous=PopulationBankV1(a.spec);oe=endogenous.recruit((11,22));e=endogenous.settle(oe,1,False)
    checks['endogenous_not_credit']=e=={'credit':0,'revisions':0} and endogenous.credit_events==0

    learned=PopulationBankV1(a.spec);ol=learned.recruit((11,22));weights_before=tuple(learned.edge_weight[e] for e in ol.edges);credit=learned.settle(ol,1,True);weights_after=tuple(learned.edge_weight[e] for e in ol.edges)
    checks['independent_causal_learning']=credit['credit']==len(ol.sites) and credit['revisions']==len(ol.edges) and all(y==x+1 for x,y in zip(weights_before,weights_after))

    expired=PopulationBankV1(a.spec);ox=expired.recruit((11,22));[expired.decay() for _ in range(a.spec.eligibility_horizon)];late=expired.settle(ox,1,True)
    checks['expired_eligibility_no_credit']=late=={'credit':0,'revisions':0}

    # Retrieval retains unresolved alternatives rather than storage-order choice.
    mem=PopulationBankV1(a.spec);s1=mem.signature((501,502));s2=mem.signature((501,502));tie=mem.retrieve((501,502),((20,s1),(10,s2)))
    checks['retrieval_tie_preserved']=tie['status']==2 and tie['winner']==0 and tie['alternatives']==2
    unique=mem.retrieve((601,602),((1,mem.signature((601,602))),(2,mem.signature((999,1000)))))
    checks['retrieval_unique_when_supported']=unique['status']==1 and unique['winner']==1

    # Sparse topology is causal: perturb one propagated target and the signature changes.
    lesion=PopulationBankV1(a.spec);features=(7001,7002);base_sig=lesion.signature(features);seed=lesion.feature_sites(features[0])[0];edge=seed*lesion.spec.fanout;old=lesion.edge_target_value(edge);lesion.set_edge_target(edge,(old+137)%lesion.spec.site_count)
    checks['topology_lesion_changes_representation']=lesion.signature(features)!=base_sig

    # Same contact on independently created banks replays exactly through learning.
    left=PopulationBankV1(a.spec);right=PopulationBankV1(a.spec);lo=left.recruit((77,88,99));ro=right.recruit((77,88,99));left.settle(lo,-1,True);right.settle(ro,-1,True)
    checks['population_replay']=left.digest()==right.digest()

    result={'schema':'0x1.reference-population-v1.verify','pass':all(checks.values()),'checks':checks,'scale_rows':rows,'example_quantity_vector':q,'claim':'STRICT_NUMERIC_POPULATION_MECHANISM_NOT_NEURON_SIMULATION_OR_CAPABILITY','elapsed_ms':round((time.perf_counter()-started)*1000,3)}
    print('FOUNDRY_REFERENCE_POPULATION '+('GREEN' if result['pass'] else 'RED')+' scale=131072 causal_sites=1 sparse_incidence=1 ambiguity=1 eligibility=1')
    print(json.dumps(result,indent=2,sort_keys=True));raise SystemExit(0 if result['pass'] else 1)

if __name__=='__main__':main()
