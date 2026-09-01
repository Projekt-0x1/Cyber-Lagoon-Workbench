#!/usr/bin/env python3
from __future__ import annotations
import copy,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
from reference_organism_v2 import *
from reference_population_v1 import PopulationSpecV1,mix64
from reference_organism_surface_competition_verify import NAME,CTX,P,train_base,partner,partner_off,scene,surf,sent,objfirst,config,config_value,template_surface,u

def settle(o,a,effect,independent=True):return o.contact(CONTACT_CONSEQUENCE,(a.ticket,effect),P,True,independent)

def configuration_key(action):
    value=0
    for member in config(action):
        for item in member:value=mix64(value^int(item))
    return value

def environment_effect(action,swapped=False):return 1 if bool(configuration_key(action)&1)!=bool(swapped) else -1

def developmental_value(o,action):
    template=next(t for t in o.language.template_candidates(CTX,4) if int(t.identity[:15],16)==action.template_identity)
    lexical=[]
    for atom,candidate in zip((102,202,302,401),action.lexical_identities):
        lexical.append(next(support for support,units,_sources in o.language.lexeme_candidates(atom) if o.language.lexeme_identity(atom,units)==candidate))
    return template.support+sum(lexical)

def build():
    o=ReferenceOrganismV2(PopulationSpecV1(32768,2,4,42,8));train_base(o)
    a=(101,201,301,401);b=(102,202,302,401)
    partner_off(o)
    o.contact(CONTACT_SCENE,(7,NAME,1,401),80001,True,True);surf(o,'device',81001)
    scene(o,a,80002);surf(o,sent(a),81002)
    partner(o);scene(o,a,82001);warm=o.tick();assert warm is not None;settle(o,warm,environment_effect(warm))
    scene(o,b,82002);target=o.tick();assert target is not None;settle(o,target,environment_effect(target))
    target_context=target.selection_context;target_configuration=config(target)
    partner_off(o)
    for j in range(4):scene(o,a,83000+j);surf(o,objfirst(a),84000+j)
    for j in range(3):o.contact(CONTACT_SCENE,(7,NAME,1,401),85000+j,True,True);surf(o,'device',86000+j)
    partner(o)
    return o,a,b,target,target_context,target_configuration

def main():
    started=time.perf_counter();checks={};o,a,b,target,ctx,target_cfg=build()
    templates=o.language.template_candidates(CTX,4);lexemes=o.language.lexeme_candidates(401)
    marginal_template=max(templates,key=lambda t:t.support);marginal_lexeme=max(lexemes,key=lambda row:row[0])
    marginal_surface=o.language.render_template(marginal_template,tuple((marginal_lexeme[1] if atom==401 else o.language.lexeme(atom)) for atom in b))
    checks['marginal_crossing_was_never_credited']=all(row.configuration!=o._selection_configuration(int(marginal_template.identity[:15],16),tuple(o.language.lexeme_identity(atom,(marginal_lexeme[1] if atom==401 else o.language.lexeme(atom))) for atom in b)) for row in o.selection_configuration_revisions)

    nominated=[];effects=[];first=None;earned=None;earned_network=0
    for trial in range(4):
        scene(o,b,87000+trial);action=o.tick();assert action is not None
        if first is None:first=copy.deepcopy(o.checkpoint()),action.ticket
        nominated.append(config(action))
        effect=environment_effect(action);effects.append(effect)
        if effect>0:earned_network=action.selection_network_identity
        settle(o,action,effect)
        if effect>0:earned=action;break
    checks['precommitted_opaque_ecology_nominated_real_trials']=earned is not None and len(nominated)>=2 and len(set(nominated))==len(nominated) and effects[-1]>0 and all(x<0 for x in effects[:-1])
    checks['wrong_crossings_required_independent_counterevidence']=sum(row.counter for row in o.selection_configuration_revisions if row.context==ctx)==len(effects)-1 and all(o._selection_configuration_evidence(ctx,cfg)[0]<0 for cfg in nominated[:-1])
    marginal_total=marginal_template.support+sum(max(rows,key=lambda row:row[0])[0] for rows in (o.language.lexeme_candidates(atom) for atom in b))
    checks['earned_configuration_is_lower_marginal']=earned is not None and developmental_value(o,earned)<marginal_total
    scene(o,b,87500);selected=o.tick();assert selected is not None
    checks['earned_lower_marginal_network_reinstated']=config(selected)==config(earned) and config_value(o,ctx,selected)>0
    checks['transient_network_identity_per_occurrence']=selected.selection_network_identity!=earned_network and earned.selection_network_identity==0 and earned_network!=0 and config(selected)==config(earned)
    checks['sparse_candidate_work']=o.last_selection_candidate_touches==4 and o.last_selection_network_touches<=len(o.selection_configuration_revisions)*o.last_selection_candidate_touches and (o.last_selection_candidate_touches+o.last_selection_network_touches)/o.population.spec.site_count<0.001

    tampered=ReferenceOrganismV2.restore(copy.deepcopy(first[0]));bad=next(a for a in tampered.actions if a.ticket==first[1]);before=len(tampered.selection_configuration_revisions);bad.template_identity^=1;bad.payload=tuple(reversed(bad.payload));bad.selection_network_identity=tampered._selection_network_identity(bad.selection_context,bad.population_occurrence,bad.closure_identity,bad.selection_occurrences)
    try:settle(tampered,bad,-1)
    except ValueError as exc:checks['coherent_action_rewrite_refused_atomically']=str(exc)=='organism:consequence_action_commitment' and not bad.settled and len(tampered.selection_configuration_revisions)==before
    else:checks['coherent_action_rewrite_refused_atomically']=False

    yoked=ReferenceOrganismV2.restore(copy.deepcopy(first[0]));ya=next(a for a in yoked.actions if a.ticket==first[1]);settle(yoked,ya,-1,False);scene(yoked,b,87900);yb=yoked.tick()
    checks['yoked_return_cannot_counter_configuration']=yb is not None and config(yb)==config(ya) and config_value(yoked,ctx,ya)==0

    settle(o,selected,environment_effect(selected));checkpoint=o.checkpoint();restored=ReferenceOrganismV2.restore(copy.deepcopy(checkpoint))
    checks['settled_computational_state_not_checkpoint_authority']=not restored.actions and not restored._action_commitments
    checks['checkpoint_preserves_configuration_recipe_revisions']=restored.digest()==o.digest() and config_value(restored,ctx,target)>0
    corrupted=copy.deepcopy(first[0]);corrupted['action_commitments'][str(first[1]) if str(first[1]) in corrupted['action_commitments'] else first[1]]='0'*64
    try:ReferenceOrganismV2.restore(corrupted)
    except ValueError as exc:checks['corrupt_action_closure_checkpoint_refused']=str(exc)=='organism:checkpoint_action_commitment'
    else:checks['corrupt_action_closure_checkpoint_refused']=False
    withdrawn=ReferenceOrganismV2.restore(copy.deepcopy(first[0]));pending=next(a for a in withdrawn.actions if a.ticket==first[1]);withdrawn.contact(CONTACT_WITHDRAW_SOURCE,(P,),88001,True,True);before=len(withdrawn.selection_configuration_revisions)
    try:settle(withdrawn,pending,environment_effect(pending))
    except ValueError as exc:checks['withdrawal_retires_pending_configuration']=str(exc)=='organism:consequence_ticket' and pending.settled and len(withdrawn.selection_configuration_revisions)==before
    else:checks['withdrawal_retires_pending_configuration']=False

    swapped=ReferenceOrganismV2.restore(copy.deepcopy(first[0]));sa=next(a for a in swapped.actions if a.ticket==first[1]);settle(swapped,sa,environment_effect(sa,True));scene(swapped,b,88900);sb=swapped.tick()
    checks['precommitted_mapping_swap_changes_credit']=sb is not None and config(sb)!=config(earned)

    result={'schema':'agi.reference-organism-selection-network-credit.v2','pass':all(checks.values()),'checks':checks,'candidate_configurations':4,'counterfactual_trials':len(nominated)-1,'resident_sites':o.population.spec.site_count,'touched_candidates':o.last_selection_candidate_touches,'touched_revision_rows':o.last_selection_network_touches,'configuration_revisions':len(o.selection_configuration_revisions),'runtime_llm':False,'environment_mapping':'PRECOMMITTED_OPAQUE_CONFIGURATION_PARITY','reference_only':True,'physical_direct_parity':'NOT_RUN','graph_flip':False,'claim':'CONFIGURATION_LOCAL_CAUSAL_LANGUAGE_SELECTION_REFERENCE_ONLY_NOT_DIRECT_PARITY','elapsed_ms':round((time.perf_counter()-started)*1000,3)}
    print('FOUNDRY_REFERENCE_ORGANISM_SELECTION_NETWORK_CREDIT '+('GREEN' if result['pass'] else 'RED')+' lower_marginal=1 actual_networks=1 independent_counterevidence=1 sparse=1')
    print(json.dumps(result,indent=2,sort_keys=True));raise SystemExit(0 if result['pass'] else 1)

if __name__=='__main__':main()
