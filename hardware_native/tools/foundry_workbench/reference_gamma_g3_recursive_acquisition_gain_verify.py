#!/usr/bin/env python3
from __future__ import annotations
import json,time
from autotrans_species_ir_v0 import FoundrySpeciesProgramV0,SpeciesLawV0
from reference_language_mastery_contact_adapter_v1 import CONTACT_SCENE,CONTACT_SURFACE,LanguageMasteryContactAdapterV1
from reference_species_language_life_function import birth_language_mastery_adult
from reference_hierarchical_composition_v1 import HierarchicalRefuse

C=9301
JOIN=9302
SEP=b' then '
A1,A2,G1,G2,V1,V2,O1,O2=101,102,201,202,301,302,401,402
NAMES={A1:'careful',A2:'quiet',G1:'engineer',G2:'technician',V1:'tests',V2:'inspects',O1:'sensor',O2:'valve'}
CLAUSES=(
    ((A1,G1,V1,O1),'the careful engineer tests the sensor.'),
    ((A2,G2,V2,O2),'the quiet technician inspects the valve.'),
    ((A1,G2,V2,O1),'the careful technician inspects the sensor.'),
    ((A2,G1,V1,O2),'the quiet engineer tests the valve.'),
)

def species(n):
    return FoundrySpeciesProgramV0.build((
        SpeciesLawV0('authenticated_external_contact'),
        SpeciesLawV0('source_conditioned_access_evidence',parameters=(('minimum_distinct_sources',int(n)),)),
    ))

def contact(adult,scene,text,source):
    c=LanguageMasteryContactAdapterV1(adult)
    c.contact(CONTACT_SCENE,scene,source)
    c.contact(CONTACT_SURFACE,tuple(text.encode()),source)

def refused(adult,left,right):
    try:
        adult.compose(JOIN,left,right)
        return False
    except HierarchicalRefuse:
        return True

def main():
    started=time.perf_counter()
    candidate=birth_language_mastery_adult(species(2))
    conservative=birth_language_mastery_adult(species(3))
    adults=(candidate,conservative)
    for feature,text in NAMES.items():
        for base in (1000,2000,3000):
            for adult in adults:
                contact(adult,(100,feature),text,base+feature)
    for i,(atoms,text) in enumerate(CLAUSES):
        for base in (4000,5000,6000):
            for adult in adults:
                contact(adult,(C,*atoms),text,base+i)
    leaves={adult:[adult.leaf(C,atoms) for atoms,_ in CLAUSES] for adult in adults}
    checks={
        'leaf_competence_equalized':all(bytes(leaves[candidate][i].surface)==bytes(leaves[conservative][i].surface) for i in range(4)),
        'no_join_from_leaf_curriculum':refused(candidate,*leaves[candidate][:2]) and refused(conservative,*leaves[conservative][:2]),
    }
    criterion={'candidate':None,'conservative':None}
    surfaces={}
    for trial,(left_i,right_i) in enumerate(((0,1),(2,3),(1,2)),1):
        for name,adult in (('candidate',candidate),('conservative',conservative)):
            adult.observe_join(JOIN,leaves[adult][left_i],leaves[adult][right_i],7000+trial,SEP)
            if criterion[name] is None and not refused(adult,leaves[adult][3],leaves[adult][0]):
                criterion[name]=trial
        if trial==2:
            first=candidate.compose(JOIN,leaves[candidate][3],leaves[candidate][0])
            nested=candidate.compose(JOIN,first,leaves[candidate][2])
            surfaces['candidate']=bytes(nested.surface).decode()
            checks['candidate_depth_two_after_two_sources']=nested.depth==2 and bytes(nested.surface).count(SEP)==2
            checks['conservative_refuses_after_two_sources']=refused(conservative,leaves[conservative][3],leaves[conservative][0])
    cfirst=conservative.compose(JOIN,leaves[conservative][3],leaves[conservative][0])
    cnested=conservative.compose(JOIN,cfirst,leaves[conservative][2])
    surfaces['conservative']=bytes(cnested.surface).decode()
    checks.update({
        'candidate_criterion_two':criterion['candidate']==2,
        'conservative_criterion_three':criterion['conservative']==3,
        'same_eventual_recursive_surface':surfaces['candidate']==surfaces['conservative'],
        'one_example_recursive_gain':criterion['conservative']-criterion['candidate']==1,
        'heldout_nested_never_demonstrated':surfaces['candidate'] not in [text for _,text in CLAUSES],
        'no_authored_syntax_ontology':all(not hasattr(candidate,n) for n in ('transformer','grammar_rules','syntax_tree','merge_opcode','tokens','target_text')),
        'bounded_fast_path':time.perf_counter()-started<1.0,
    })
    failed=[k for k,v in checks.items() if not v]
    result={
        'contract':'FOUNDRY_GAMMA_G3_RECURSIVE_ACQUISITION_GAIN_GREEN',
        'pass':not failed,
        'reference_only':True,
        'graph_flip':False,
        'visible_language_gain':'RECURSIVE_MULTI_CLAUSE_DISCOURSE_ONE_LIVED_EXAMPLE_EARLIER',
        'candidate_examples_to_criterion':criterion['candidate'],
        'conservative_examples_to_criterion':criterion['conservative'],
        'examples_saved':None if None in criterion.values() else criterion['conservative']-criterion['candidate'],
        'nested_surface':surfaces.get('candidate',''),
        'checks':checks,
        'elapsed_ms':round((time.perf_counter()-started)*1000,3),
        'remaining_red':['DIRECT_RUNTIME_OWNERSHIP','DIRECT_G3_GENOME_LOWERING','PHYSICAL_SIBLING_ACQUISITION_ASSAY','GRAPH_PROMOTION'],
    }
    print(result['contract'] if not failed else 'FOUNDRY_GAMMA_G3_RECURSIVE_ACQUISITION_GAIN_RED '+','.join(failed))
    print(json.dumps(result,indent=2,sort_keys=True))
    raise SystemExit(0 if not failed else 1)

if __name__=='__main__':main()
