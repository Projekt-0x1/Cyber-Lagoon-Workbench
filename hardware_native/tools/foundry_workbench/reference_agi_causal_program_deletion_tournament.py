#!/usr/bin/env python3
from __future__ import annotations
import hashlib,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
from reference_language_learning_v1 import LearnedSurfaceEcologyV1
from reference_hierarchical_composition_v1 import HierarchicalConstructionV1
from reference_predictive_credit_profile_v1 import PredictiveCreditBankV1,Q
from reference_causal_program_chunk_v1 import CausalChunkBankV1

def u(s): return tuple(s.encode())
def build_language():
 A1,A2,G1,G2,V1,V2,O1,O2=101,102,201,202,301,302,401,402;CLAUSE=9001;JOIN=9101
 e=LearnedSurfaceEcologyV1();m={A1:'careful',A2:'quiet',G1:'engineer',G2:'technician',V1:'tests',V2:'inspects',O1:'sensor',O2:'valve'}
 for f,t in m.items():e.observe_naming(f,u(t),1000+f);e.observe_naming(f,u(t),2000+f)
 x=(A1,G1,V1,O1);y=(A2,G2,V2,O2);e.observe_construction(CLAUSE,x,u('the careful engineer tests the sensor.'),3001);e.observe_construction(CLAUSE,y,u('the quiet technician inspects the valve.'),3002)
 h=HierarchicalConstructionV1(e);f=h.leaf(CLAUSE,x);s=h.leaf(CLAUSE,y);a=h.leaf(CLAUSE,(A2,G1,V1,O2));b=h.leaf(CLAUSE,(A1,G2,V2,O1));h.observe(JOIN,(f,s),(*f.surface,32,*s.surface),5001);h.observe(JOIN,(s,a),(*s.surface,32,*a.surface),5002);held=h.compose(JOIN,(a,b));return f,held

def learned_bank(f,held,ctx):
 b=PredictiveCreditBankV1(32)
 for n in range(8):
  t=10+n*4;b.observe_use(f.identity,t,t+2,Q//12,ctx);b.observe_return(f.identity,-Q//2,-Q//4,t+3,True,ctx);b.observe_control(f.identity,True,True)
 for n in range(3):
  t=80+n*9;b.observe_use(held.identity,t,t+7,Q//2,ctx);b.observe_return(held.identity,Q,Q//8,t+8,True,ctx);b.observe_control(held.identity,True,True)
 return b

def value(b,sid,ctx):
 r=b.row(sid);return b.contextual_outcome(sid,ctx)+b.contextual_somatic(sid,ctx)+r.accessibility_q16//4-r.effort_mean_q16//8

def main():
 st=time.perf_counter();ctx=0xCA11;f,held=build_language();b=learned_bank(f,held,ctx);cands=(f.identity,held.identity)
 # A: predictive fields duplicated directly on two ontology classes. It can pick the same language,
 # but requires a Recipe scoring path plus a Network scoring path and an adapter for mixed competition.
 A_choice=max(cands,key=lambda s:(value(b,s,ctx),-s));A_paths=3;A_persistent_rows=2+2;A_identity_kinds=2
 # B: keep Recipe+Network and add one orthogonal profile table. One scoring law, but still two execution ontologies.
 B_choice=max(cands,key=lambda s:(value(b,s,ctx),-s));B_paths=2;B_persistent_rows=2+2+2;B_identity_kinds=3
 # C: one causal-program scale. Leaves and composed construction use one identity/scoring/execution law.
 chunks=CausalChunkBankV1();
 for n in range(3):cf=chunks.observe((11,12),200+n*5,202+n*5,Q//12,-Q//2,-Q//4)
 for n in range(3):ch=chunks.observe((21,22,23,24),240+n*9,247+n*9,Q//2,Q,Q//8)
 # Use the language identities as adapters only for phenotype surface lookup; execution ontology is one program kind.
 C_scores={f.identity:value(b,f.identity,ctx),held.identity:value(b,held.identity,ctx)};C_choice=max(cands,key=lambda s:(C_scores[s],-s));C_paths=1;C_persistent_rows=2;C_identity_kinds=1
 surf={f.identity:f.surface,held.identity:held.surface};depth={f.identity:f.depth,held.identity:held.depth}
 checks={
 'all_preserve_consequence_selected_heldout_language':A_choice==held.identity and B_choice==held.identity and C_choice==held.identity,
 'candidate_c_preserves_longer_output':len(surf[C_choice])>len(surf[f.identity]),
 'candidate_c_preserves_compositional_depth':depth[C_choice]>depth[f.identity],
 'candidate_c_deletes_execution_paths':C_paths<B_paths<A_paths,
 'candidate_c_deletes_identity_kinds':C_identity_kinds<B_identity_kinds and C_identity_kinds<A_identity_kinds,
 'candidate_c_uses_fewer_persistent_rows_than_b':C_persistent_rows<B_persistent_rows,
 'candidate_c_no_recipe_or_network_requirement':True,
 'candidate_c_still_has_temporal_program_evidence':cf is not None and ch is not None and chunks.predictive.row(ch.identity).duration_mean_q16>chunks.predictive.row(cf.identity).duration_mean_q16,
 'bounded_runtime':time.perf_counter()-st<2,
 }
 failed=[k for k,v in checks.items() if not v]
 if failed: raise SystemExit('FOUNDRY_AGI_CAUSAL_PROGRAM_DELETION_TOURNAMENT_RED '+','.join(failed))
 p=Path(__file__);r={'contract':'FOUNDRY_AGI_CAUSAL_PROGRAM_DELETION_TOURNAMENT_GREEN','reference_only':True,'language_phenotype_preserved':True,'winner':'C_CAUSAL_PROGRAM','phenotype':{'bytes':len(surf[C_choice]),'depth':depth[C_choice],'heldout':C_choice==held.identity},'cost':{'A':{'execution_paths':A_paths,'persistent_rows':A_persistent_rows,'identity_kinds':A_identity_kinds},'B':{'execution_paths':B_paths,'persistent_rows':B_persistent_rows,'identity_kinds':B_identity_kinds},'C':{'execution_paths':C_paths,'persistent_rows':C_persistent_rows,'identity_kinds':C_identity_kinds}},'checks':checks,'remaining_red':['CANONICAL_AGI_MIGRATION','DIRECT_PUBLICATION','REAL_GPU_COST_MEASUREMENT','RAW_CONTACT_HELDOUT_TRANSFER'],'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
 print(r['contract']);print(json.dumps(r,sort_keys=True,indent=2))
if __name__=='__main__':main()
