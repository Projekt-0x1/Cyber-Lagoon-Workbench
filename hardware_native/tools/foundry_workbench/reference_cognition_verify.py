#!/usr/bin/env python3
from __future__ import annotations
import json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
from reference_cognition_v1 import TransitionEcologyV1
from reference_population_v1 import PopulationBankV1,PopulationSpecV1

def support(e,s,a,n,effect,base):e.observe(s,a,n,effect,base,True);e.observe(s,a,n,effect,base+1,True)

def main():
 t=time.perf_counter();e=TransitionEcologyV1();checks={}
 S0=(10,);S1=(20,);S2=(30,40);ALT=(25,);GOAL=(40,);A1,A2,A3=101,102,103
 support(e,S0,A1,S1,1,1000);support(e,S1,A2,S2,2,1100);support(e,S0,A3,ALT,1,1200);support(e,ALT,A2,S2,1,1300)
 plan=e.plan(S0,GOAL);checks['two_step_plan']=plan.status==1 and plan.actions==(A1,A2) and plan.states==(S0,S1,S2)
 before=e.digest();sim=e.simulate(S0,plan.actions);checks['counterfactual_no_mutation']=sim is not None and sim[0]==(S0,S1,S2) and e.digest()==before
 # Higher accumulated consequence breaks equal-length tie without certifying world truth.
 checks['consequence_ranked_choice']=plan.score>0
 # Equal first-action alternatives remain unresolved.
 tie=TransitionEcologyV1();support(tie,S0,A1,S1,1,2000);support(tie,S1,A2,S2,1,2100);support(tie,S0,A3,ALT,1,2200);support(tie,ALT,A2,S2,1,2300)
 p=tie.plan(S0,GOAL);checks['equal_alternatives_unresolved']=p.status==2 and p.alternatives==2
 # Contradictory equally supported destinations invalidate that action model.
 contradiction=TransitionEcologyV1();support(contradiction,S0,A1,S1,1,3000);support(contradiction,S0,A1,ALT,1,3100)
 checks['contradictory_transition_refuses']=contradiction.transition(S0,A1) is None
 # Source withdrawal can invalidate the preferred route and expose the alternative.
 w=TransitionEcologyV1();support(w,S0,A1,S1,1,4000);support(w,S1,A2,S2,3,4100);support(w,S0,A3,ALT,1,4200);support(w,ALT,A2,S2,1,4300);before_plan=w.plan(S0,GOAL);w.withdraw_source(4101);after=w.plan(S0,GOAL)
 checks['source_withdrawal_replans']=before_plan.status==1 and before_plan.actions[0]==A1 and after.status==1 and after.actions==(A3,A2)
 # Independence is necessary for transition evidence.
 ni=TransitionEcologyV1();ni.observe(S0,A1,S1,5,5000,False);checks['endogenous_prediction_not_world_transition']=ni.transition(S0,A1) is None
 # Population state can carry the same opaque state features; planning is not language state.
 pop=PopulationBankV1(PopulationSpecV1(65536,fanout=2,sites_per_feature=4));o0=pop.recruit(S0);o1=pop.recruit(S1);checks['population_encoded_states']=o0.sites!=o1.sites and pop.quantity_vector(o1)['R']==65536
 cp=e.checkpoint();r=TransitionEcologyV1.restore(cp);checks['checkpoint_replay']=r.digest()==e.digest() and r.plan(S0,GOAL)==plan
 # Opaque ID renaming preserves graph shape/first-plan structure after inverse map.
 shift=90000;ren=TransitionEcologyV1();support(ren,(10+shift,),A1+shift,(20+shift,),1,6000);support(ren,(20+shift,),A2+shift,(30+shift,40+shift),2,6100);rp=ren.plan((10+shift,),(40+shift,));checks['opaque_id_permutation']=rp.status==1 and tuple(x-shift for x in rp.actions)==(A1,A2)
 result={'schema':'0x1.reference-cognition-v1.verify','pass':all(checks.values()),'checks':checks,'plan':{'actions':list(plan.actions),'depth':len(plan.actions),'score':plan.score},'evidence_rows':len(e._evidence),'claim':'LEARNED_COUNTERFACTUAL_PLANNING_NOT_WORLD_ORACLE_OR_HUMAN_REASONING','elapsed_ms':round((time.perf_counter()-t)*1000,3)}
 print('FOUNDRY_REFERENCE_COGNITION '+('GREEN' if result['pass'] else 'RED')+' learned_transitions=1 counterfactual=1 replanning=1 ambiguity=1 language=0')
 print(json.dumps(result,indent=2,sort_keys=True));raise SystemExit(0 if result['pass'] else 1)
if __name__=='__main__':main()
