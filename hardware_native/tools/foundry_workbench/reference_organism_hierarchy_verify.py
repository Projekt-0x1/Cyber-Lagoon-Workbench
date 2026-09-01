#!/usr/bin/env python3
from __future__ import annotations
import copy,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
from types import SimpleNamespace
from reference_organism_v2 import *
from reference_population_v1 import PopulationSpecV1
from reference_organism_discourse_verify import train,scene,partner,settle,u,CTX,REL_NEXT
from reference_hierarchical_composition_v1 import rematerialize_transient_plan,rematerialize_transient_sequence_plan
from reference_incremental_expression_v1 import ExpressionRefuse,IncrementalTransientExpressionV1,IncrementalTransientSequenceExpressionV1

def learn_join(o):
 for left_atoms,right_atoms,src in (((101,201,301,401),(102,202,302,402),71001),((102,201,302,402),(101,202,301,401),71002)):
  left=scene(o,left_atoms,src+100);right=scene(o,right_atoms,src+200);o.contact(CONTACT_SCENE_LINK,(left,right,REL_NEXT),src,True,True);ls=o.language.realize(CTX,left_atoms);rs=o.language.realize(CTX,right_atoms);o.contact(CONTACT_DISCOURSE_SURFACE,(*ls,*u(' Then '),*rs),src,True,True)
 assert o.language.span_template(REL_NEXT,2)
def setup(o,p,base):
 partner(o,p);atoms=((101,202,302,401),(102,201,301,402),(101,201,302,402),(102,202,301,401));ids=[scene(o,row,base+i) for i,row in enumerate(atoms)]
 for i in range(3):o.contact(CONTACT_SCENE_LINK,(ids[i],ids[i+1],REL_NEXT),base+100+i,True,True)
 return ids
def run(o,p,ids,n=4):
 acts=[];closures=[]
 for _ in range(n):
  a=o.tick();assert isinstance(a,ActionV2);c=o.utterances.lookup(a.closure_identity);assert c is not None;acts.append(a);closures.append(c);settle(o,a,p,1)
 return acts,closures
def main():
 t=time.perf_counter();checks={};spec=PopulationSpecV1(65536,fanout=2,sites_per_feature=4,eligibility_horizon=8);P=9901
 o=ReferenceOrganismV2(spec);train(o);learn_join(o);ids=setup(o,P,80000);acts,cs=run(o,P,ids);depths=[x.depth for x in cs]
 clauses=(b'the careful engineer tests the sensor.',b'the quiet technician inspects the valve.',b'the careful engineer inspects the valve.')
 kids=tuple(o.utterances.remember(CTX,88000+i,row,persist=False) for i,row in enumerate(clauses))
 plan,surface=rematerialize_transient_sequence_plan(o.language,REL_NEXT,kids)
 emitted=[];traj=IncrementalTransientSequenceExpressionV1(o.language,plan,leaves=o.utterances)
 while True:
  byte=traj.emit()
  if byte is None:break
  emitted.append(byte.value);assert traj.reafference(byte,byte.value)
 checks['resident_boundaries_are_current_utterances']=depths==[0,0,0,0]
 checks['scene_order']=[a.scene_identity for a in acts]==ids
 checks['incremental_public']=all(bytes(a.payload).startswith(b' Then ') for a in acts[1:])
 checks['heldout_nested_incremental_from_language']=plan.depth>=2 and tuple(emitted)==surface and bytes(surface).count(b' Then ')==2 and traj.complete
 nest=(b'the careful engineer tests the sensor.',b'the quiet technician inspects the valve.',
       b'the careful engineer inspects the valve.',b'the quiet technician tests the sensor.',
       b'the quiet engineer inspects the sensor.')
 nkids=tuple(o.utterances.remember(CTX,89000+i,row,persist=False) for i,row in enumerate(nest))
 span=o.language.span_template(REL_NEXT,2)
 def _node(plan,surface):
  return SimpleNamespace(identity=plan.identity,context=plan.context,template_identity=plan.template_identity,
                         child_identities=plan.child_identities,depth=plan.depth,surface=surface)
 p_ab,s_ab=rematerialize_transient_plan(o.language,REL_NEXT,(nkids[0],nkids[1]),span);n_ab=_node(p_ab,s_ab)
 p_abc,s_abc=rematerialize_transient_plan(o.language,REL_NEXT,(n_ab,nkids[2]),span);n_abc=_node(p_abc,s_abc)
 p_de,s_de=rematerialize_transient_plan(o.language,REL_NEXT,(nkids[3],nkids[4]),span);n_de=_node(p_de,s_de)
 p_top,s_top=rematerialize_transient_plan(o.language,REL_NEXT,(n_abc,n_de),span)
 tree=[];walk=IncrementalTransientExpressionV1(o.language,p_top,leaves=o.utterances,nodes={n_ab.identity:n_ab,n_abc.identity:n_abc,n_de.identity:n_de})
 while True:
  byte=walk.emit()
  if byte is None:break
  tree.append(byte.value);assert walk.reafference(byte,byte.value)
 checks['heldout_balanced_depth_three_incremental']=p_top.depth>=3 and tuple(tree)==s_top and bytes(s_top).count(b' Then ')==4 and walk.complete and not hasattr(o,'hierarchy')
 nodes={n_ab.identity:n_ab,n_abc.identity:n_abc,n_de.identity:n_de}
 mid=IncrementalTransientExpressionV1(o.language,p_top,leaves=o.utterances,nodes=nodes);pre=[]
 for _ in range(40):
  byte=mid.emit();pre.append(byte.value);assert mid.reafference(byte,byte.value)
 mid_cp=copy.deepcopy(mid.checkpoint());mid_blob=json.dumps(mid_cp,separators=(',',':')).encode()
 rest=IncrementalTransientExpressionV1.restore(o.language,mid_cp,leaves=o.utterances);tail=[]
 while True:
  byte=rest.emit()
  if byte is None:break
  tail.append(byte.value);assert rest.reafference(byte,byte.value)
 checks['nested_midstream_restore_exact']=tuple(pre+tail)==s_top and rest.complete and b'the careful engineer' not in mid_blob and 'surface' not in mid_cp
 assert o.utterances.retire(nkids[0].identity)
 def _drain(traj):
  while True:
   byte=traj.emit()
   if byte is None:return
   traj.reafference(byte,byte.value)
 broken=IncrementalTransientExpressionV1(o.language,p_top,leaves=o.utterances,nodes=nodes)
 try:_drain(broken);left_refused=False
 except ExpressionRefuse:left_refused=True
 sib=[];keep=IncrementalTransientExpressionV1(o.language,p_de,leaves=o.utterances)
 while True:
  byte=keep.emit()
  if byte is None:break
  sib.append(byte.value);assert keep.reafference(byte,byte.value)
 checks['forgotten_left_leaf_refuses_full_tree']=left_refused and not broken.complete
 checks['sibling_branch_survives_left_leaf_retire']=tuple(sib)==s_de and keep.complete
 checks['visible_discussion_improvement']=checks['heldout_balanced_depth_three_incremental'] and checks['nested_midstream_restore_exact'] and checks['sibling_branch_survives_left_leaf_retire'] and checks['forgotten_left_leaf_refuses_full_tree'] and checks['incremental_public']
 checks['no_cumulative_resident_transcript']=not hasattr(o,'hierarchy') and 'hierarchy' not in o.checkpoint() and len(o.utterances._leaves)==1 and sum(1 for row in o.shared_episode_relations if row.partner==P and row.closure_identity)==1
 checks['transient_span_participates_without_persistent_pair_closure']=all(a.closure_identity in a.contributors and any(row[0]==PREF_SPAN for row in a.selection_occurrences) for a in acts[1:])
 checks['terminal_silence']=o.tick() is None
 checks['bounded_recurrent_depth']=all(x.depth==0 for x in cs) and not hasattr(o,'hierarchy')
 r0=ReferenceOrganismV2(spec);train(r0);learn_join(r0);rids=setup(r0,P,82000);a0,c0=run(r0,P,rids,2);r=ReferenceOrganismV2.restore(copy.deepcopy(r0.checkpoint()));a=r.tick();cc=r.utterances.lookup(a.closure_identity);checks['checkpoint_continuation']=a.scene_identity==rids[2] and cc is not None and cc.depth==0 and bytes(a.payload).startswith(b' Then ') and not hasattr(r,'hierarchy')
 w=ReferenceOrganismV2(spec);train(w);learn_join(w);w.contact(CONTACT_WITHDRAW_SOURCE,(71002,),90000,True,True);wids=setup(w,9911,84000);wa,wc=run(w,9911,wids);checks['withdrawal_reopens_structure']=all(not bytes(a.payload).startswith(b' Then ') for a in wa[1:]) and [x.depth for x in wc]==[0,0,0,0]
 result={'schema':'0x1.reference-organism-hierarchy.v2','pass':all(checks.values()),'checks':checks,'depths':depths,'closure_count':0,'final_ancestry':len(cs[-1].ancestry),'final_resident_bytes':len(cs[-1].surface),'public_action_bytes':[len(a.payload) for a in acts],'nested_incremental_bytes':len(surface),'nested_incremental_depth':plan.depth,'balanced_incremental_bytes':len(s_top),'balanced_incremental_depth':p_top.depth,'elapsed_ms':round((time.perf_counter()-t)*1000,3),'claim':'BOUNDED_RECURRENT_DISCOURSE_CLOSURE_REFERENCE_NOT_HUMAN_DIALOGUE'}
 print('FOUNDRY_REFERENCE_ORGANISM_HIERARCHY '+('GREEN' if result['pass'] else 'RED')+f" depths={depths} nested={plan.depth} balanced={p_top.depth} incremental=1");print(json.dumps(result,indent=2,sort_keys=True));raise SystemExit(0 if result['pass'] else 1)
if __name__=='__main__':main()
