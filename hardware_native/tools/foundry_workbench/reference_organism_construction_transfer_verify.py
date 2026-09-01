#!/usr/bin/env python3
from __future__ import annotations
import copy,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
from reference_organism_v2 import *
from reference_population_v1 import PopulationSpecV1
import reference_organism_register_verify as base

def u(text):return tuple(text.encode())
def scene(o,atoms,source):return o.contact(CONTACT_SCENE,(7,base.CTX,4,*atoms),source,True,True)
def surface(o,text,source):return o.contact(CONTACT_SURFACE,u(text),source,True,True)
def settle(o,action,effect,independent=True):return o.contact(CONTACT_CONSEQUENCE,(action.ticket,effect),base.P_FORMAL,True,independent)
def formal(words):return f'the {words[0]} {words[1]} {words[2]} the {words[3]}.'
def terse(words):return f'{words[0]} {words[1]}: {words[2]} {words[3]}.'
def configuration(action):return tuple(row[:3] for row in action.selection_occurrences)

def build(bindings,sets=None):
    o=base.train(ReferenceOrganismV2(PopulationSpecV1(32768,2,4,42,8)))
    if sets is None:
        sets=[(tuple(110+rank*10+i for i in range(4)),tuple(f'{prefix}{i}' for i in range(4))) for rank,prefix in enumerate(('ka','la','ma'))]
    sets=list(sets)
    for rank,(atoms,words) in enumerate(sets):
        for atom,word in zip(atoms,words):
            for repeat in range(2):
                o.contact(CONTACT_SCENE,(7,100,1,atom),50000+rank*100+atom+repeat,True,True);surface(o,word,60000+rank*100+atom+repeat)
    base.partner(o,base.P_FORMAL)
    actions=[]
    for step,index in enumerate(bindings):
        atoms,words=sets[index];base.off(o);scene(o,atoms,70000+step*20);surface(o,formal(words),71000+step*20);base.partner(o,base.P_FORMAL)
        scene(o,atoms,72000+step*20);action=o.tick();assert action is not None;settle(o,action,1);actions.append(action)
        base.off(o);scene(o,atoms,73000+step*20);surface(o,terse(words),74000+step*20);base.partner(o,base.P_FORMAL)
    return o,sets,actions

def heldout(o,sets,source=80000):
    atoms,_words=sets[2];scene(o,atoms,source);return o.tick()

def main():
    started=time.perf_counter();checks={}
    one,one_sets,one_trained=build((0,));one_probe=heldout(one,one_sets)
    checks['one_binding_keeps_generic_language_without_claiming_productivity']=isinstance(one_probe,ActionV2) and one._selection_construction_evidence(one_trained[-1].selection_context,one_trained[-1].template_identity)<2
    repeated,repeated_sets,repeated_trained=build((0,0));repeated_probe=heldout(repeated,repeated_sets)
    checks['repetition_is_not_binding_diversity_but_generic_survives']=isinstance(repeated_probe,ActionV2) and repeated._selection_construction_evidence(repeated_trained[-1].selection_context,repeated_trained[-1].template_identity)<2
    partial_sets=[((110,111,112,113),('ka0','ka1','ka2','ka3')),((120,111,112,113),('la0','ka1','ka2','ka3')),((110,121,112,113),('ka0','la1','ka2','ka3'))]
    partial,partial_sets,partial_trained=build((0,1),partial_sets);partial_probe=heldout(partial,partial_sets)
    checks['one_port_diversity_does_not_license_specialization_other_ports']=isinstance(partial_probe,ActionV2) and partial._selection_construction_evidence(partial_trained[-1].selection_context,partial_trained[-1].template_identity)<2
    o,sets,trained=build((0,1));ctx=trained[-1].selection_context;template=trained[-1].template_identity
    held_cfg=o._selection_configuration(template,tuple(o.language.lexeme_identity(atom,o.language.lexeme(atom)) for atom in sets[2][0]))
    checks['heldout_configuration_has_no_exact_credit']=o._selection_configuration_evidence(ctx,held_cfg)==(0,0)
    checks['two_distinct_consequences_form_construction_evidence']=o._selection_construction_evidence(ctx,template)==2
    checkpoint=o.checkpoint();action=heldout(o,sets);assert action is not None
    checks['heldout_fillers_rebind_same_construction']=action.template_identity==template and action.payload==u(formal(sets[2][1])) and configuration(action)==held_cfg
    checks['heldout_action_is_fresh_actual_network']=action.selection_network_identity not in {a.selection_network_identity for a in trained} and len(action.selection_occurrences)==5 and all(row[3] in action.contributors for row in action.selection_occurrences)
    checks['sparse_construction_nomination']=0<o.last_selection_candidate_touches<=2 and o.last_selection_construction_touches==2 and o.last_selection_construction_touches/o.population.spec.site_count<0.001

    positive=ReferenceOrganismV2.restore(copy.deepcopy(checkpoint));pa=heldout(positive,sets,81000);learned=settle(positive,pa,1,True)
    checks['heldout_credit_requires_returned_consequence']=learned.get('selection_network_updates')==1 and positive._selection_configuration_evidence(ctx,held_cfg)[0]==1
    yoked=ReferenceOrganismV2.restore(copy.deepcopy(checkpoint));ya=heldout(yoked,sets,82000);settle(yoked,ya,1,False)
    checks['yoked_return_cannot_credit_heldout_binding']=yoked._selection_configuration_evidence(ctx,held_cfg)==(0,0)

    negative=ReferenceOrganismV2.restore(copy.deepcopy(checkpoint));na=heldout(negative,sets,83000);settle(negative,na,-1,True);next_action=heldout(negative,sets,83001)
    checks['heldout_counterexample_deopts_construction_transfer']=negative._selection_configuration_evidence(ctx,held_cfg)[0]<0 and (next_action is None or next_action.template_identity!=template)
    withdrawn=ReferenceOrganismV2.restore(copy.deepcopy(checkpoint));withdrawn.contact(CONTACT_WITHDRAW_SOURCE,(base.P_FORMAL,),84000,True,True);checks['source_withdrawal_removes_transfer']=withdrawn._selection_construction_evidence(ctx,template)==0 and heldout(withdrawn,sets,84001) is None

    restored=ReferenceOrganismV2.restore(copy.deepcopy(checkpoint));ra=heldout(restored,sets,85000)
    checks['checkpoint_rebuilds_nomination_index']=ra is not None and ra.payload==u(formal(sets[2][1])) and 'selection_construction_revisions' not in checkpoint
    reversed_order=ReferenceOrganismV2.restore(copy.deepcopy(checkpoint));atoms,words=sets[2];scene(reversed_order,tuple(reversed(atoms)),85500);ordered=reversed_order.tick()
    checks['learned_template_preserves_positional_port_permutation']=ordered is not None and ordered.payload==u(formal(tuple(reversed(words))))
    lesioned=ReferenceOrganismV2.restore(copy.deepcopy(checkpoint));removed=next(row for row in lesioned.selection_configuration_revisions if row.configuration==configuration(trained[0]));lesioned.selection_configuration_revisions.remove(removed);lesioned._rebuild_runtime_indices();lesion_probe=heldout(lesioned,sets,86000)
    checks['one_witness_lesion_removes_specialized_transfer_not_generic_or_exact_survivor']=isinstance(lesion_probe,ActionV2) and lesioned._selection_construction_evidence(ctx,template)<2 and lesioned._selection_configuration_evidence(ctx,configuration(trained[1]))[0]>0 and lesioned._selection_configuration_evidence(ctx,held_cfg)==(0,0)

    ambiguous=ReferenceOrganismV2.restore(copy.deepcopy(checkpoint));base.off(ambiguous);atom=sets[2][0][-1]
    for repeat in range(2):ambiguous.contact(CONTACT_SCENE,(7,100,1,atom),87000+repeat,True,True);surface(ambiguous,'other',87100+repeat)
    base.partner(ambiguous,base.P_FORMAL);checks['construction_credit_does_not_smear_lexical_ambiguity']=heldout(ambiguous,sets,87200) is None

    result={'schema':'agi.reference-organism-construction-transfer.v2','pass':all(checks.values()),'checks':checks,'trained_bindings':2,'heldout_bindings':1,'construction_rows_touched':o.last_selection_construction_touches,'candidate_configurations':o.last_selection_candidate_touches,'resident_sites':o.population.spec.site_count,'persistent_construction_ledger':False,'relation_grounded_argument_order':False,'runtime_llm':False,'expected_output_path':False,'claim':'PORT_LOCAL_CONSEQUENCE_EARNED_CONSTRUCTION_TRANSFER_REFERENCE_ONLY_NOT_RELATION_GROUNDED_ARGUMENT_ORDER_OR_DIRECT_PARITY','elapsed_ms':round((time.perf_counter()-started)*1000,3)}
    print('FOUNDRY_REFERENCE_ORGANISM_CONSTRUCTION_TRANSFER '+('GREEN' if result['pass'] else 'RED')+' port_local=1 distinct_bindings=2 heldout=1 relation_order=0 sparse=1')
    print(json.dumps(result,indent=2,sort_keys=True));raise SystemExit(0 if result['pass'] else 1)

if __name__=='__main__':main()
