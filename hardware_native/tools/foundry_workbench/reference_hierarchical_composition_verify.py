#!/usr/bin/env python3
from __future__ import annotations
import json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
from reference_language_learning_v1 import LearnedSurfaceEcologyV1
from types import SimpleNamespace
from reference_hierarchical_composition_v1 import HierarchicalConstructionV1,HierarchicalRefuse,rematerialize_transient_plan

def u(s):return tuple(s.encode())
def main():
 t=time.perf_counter();c={};A1,A2,G1,G2,V1,V2,O1,O2=101,102,201,202,301,302,401,402;CLAUSE=9001;JOIN=9101
 m={A1:'careful',A2:'quiet',G1:'engineer',G2:'technician',V1:'tests',V2:'inspects',O1:'sensor',O2:'valve'};e=LearnedSurfaceEcologyV1()
 for f,text in m.items():e.observe_naming(f,u(text),1000+f);e.observe_naming(f,u(text),2000+f)
 x=(A1,G1,V1,O1);y=(A2,G2,V2,O2);e.observe_construction(CLAUSE,x,u('the careful engineer tests the sensor.'),3001);e.observe_construction(CLAUSE,y,u('the quiet technician inspects the valve.'),3002);assert e.template(CLAUSE,4)
 h=HierarchicalConstructionV1(e);leaves=(h.leaf(CLAUSE,x),h.leaf(CLAUSE,y),h.leaf(CLAUSE,(A2,G1,V1,O2)),h.leaf(CLAUSE,(A1,G2,V2,O1)))
 c['leaf_ids']=len({z.identity for z in leaves})==4
 c['first_demo_not_enough']=h.observe(JOIN,leaves[:2],(*leaves[0].surface,32,*leaves[1].surface),5001) and e.span_template(JOIN,2) is None
 c['second_demo_induces']=h.observe(JOIN,leaves[1:3],(*leaves[1].surface,32,*leaves[2].surface),5002) and e.span_template(JOIN,2) is not None
 held=h.compose(JOIN,leaves[2:4]);inner=h.compose(JOIN,leaves[:2]);mid=h.compose(JOIN,(inner,leaves[2]));outer=h.compose(JOIN,(mid,leaves[3]));c['depth3_untrained']=outer.depth==3;c['heldout_pair']=held.depth==1
 c['compose_does_not_copy_span_into_witness_table']=not hasattr(h,'_template_witnesses') and 'templates' not in h.checkpoint()
 c['visible_discussion_improvement']=c['compose_does_not_copy_span_into_witness_table'] and c['depth3_untrained']
 left=h.compose(JOIN,(h.compose(JOIN,leaves[:2]),leaves[2]));right=h.compose(JOIN,(leaves[0],h.compose(JOIN,leaves[1:3])));c['same_bytes_different_structure']=left.surface==right.surface and left.identity!=right.identity
 e.withdraw_source(5002)
 try:h.compose(JOIN,(leaves[3],leaves[0]));refused=False
 except HierarchicalRefuse:refused=True
 c['withdrawal_refuses']=refused;e.restore_source(5002)
 e2=LearnedSurfaceEcologyV1.restore(e.checkpoint());h2=HierarchicalConstructionV1.restore(e2,h.checkpoint());c['checkpoint']=h2.digest()==h.digest() and h2.closure(outer.identity)==outer
 q=h.quantity_vector(outer);c['bounded_width']=q['current_child_width']==2 and q['current_depth']==3
 expected=tuple(outer.surface);retired=h.retire_unreferenced_composites()
 def _wrap(plan,surface):
  return SimpleNamespace(identity=plan.identity,context=plan.context,template_identity=plan.template_identity,child_identities=plan.child_identities,depth=plan.depth,surface=surface)
 p_ab,s_ab=rematerialize_transient_plan(e,JOIN,(leaves[0],leaves[1]));n_ab=_wrap(p_ab,s_ab)
 p_abc,s_abc=rematerialize_transient_plan(e,JOIN,(n_ab,leaves[2]));n_abc=_wrap(p_abc,s_abc)
 p_out,s_out=rematerialize_transient_plan(e,JOIN,(n_abc,leaves[3]))
 c['retired_composites_still_rematerialize']=retired>=4 and s_out==expected and h.closure(outer.identity) is None and all(h.closure(leaf.identity) is not None for leaf in leaves) and all(node.depth==0 for node in h._closures.values())
 c['visible_discussion_improvement']=c['retired_composites_still_rematerialize'] and c['compose_does_not_copy_span_into_witness_table']
 r={'schema':'0x1.reference-hierarchical-composition.v1','pass':all(c.values()),'checks':c,'quantity':q,'outer_bytes':len(expected),'retired_composites':retired,'same_surface_ids':[left.identity,right.identity],'elapsed_ms':round((time.perf_counter()-t)*1000,3),'claim':'HIERARCHICAL_CONSTRUCTION_REFERENCE_ONLY'}
 print('FOUNDRY_HIERARCHICAL_COMPOSITION '+('GREEN' if r['pass'] else 'RED')+f" depth=3 closures={h.closure_count} retired={retired}");print(json.dumps(r,indent=2,sort_keys=True));raise SystemExit(0 if r['pass'] else 1)
if __name__=='__main__':main()
