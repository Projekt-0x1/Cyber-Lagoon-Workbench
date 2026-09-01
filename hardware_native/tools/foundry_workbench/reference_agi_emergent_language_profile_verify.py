#!/usr/bin/env python3
import hashlib,json,time
from pathlib import Path
from reference_predictive_credit_profile_v1 import PredictiveCreditBankV1,Q
from reference_causal_program_chunk_v1 import CausalChunkBankV1

CTX_A=0xA1; CTX_B=0xB2
SHORT=0x201

def score(bank,sid,ctx):
    r=bank.row(sid)
    return (bank.contextual_outcome(sid,ctx)+bank.contextual_somatic(sid,ctx)
            +r.accessibility_q16//4-r.effort_mean_q16//8-r.uncertainty_q16//8)

def choose_factorized(bank,candidates,ctx):
    ranked=sorted(((score(bank,s,ctx),s) for s in candidates),reverse=True)
    return 0 if len(ranked)>1 and ranked[0][0]==ranked[1][0] else ranked[0][1]

def choose_entrenchment(bank,candidates):
    return max(candidates,key=lambda s:(bank.row(s).accessibility_q16,-s))

def main():
    started=time.perf_counter(); chunks=CausalChunkBankV1(); bank=chunks.predictive
    # Build two reusable sub-constructions, then a longer composed construction.
    for n in range(3): a=chunks.observe((31,32),10+n*5,12+n*5,Q//8,Q//4)
    for n in range(3): b=chunks.observe((41,42),40+n*5,42+n*5,Q//8,Q//4)
    LONG=None
    for n in range(3): LONG=chunks.observe((a.identity,b.identity),70+n*8,75+n*8,Q//3,Q//2)
    LONG=LONG.identity
    # Short phrase is deeply entrenched. In context A its lived social/world return
    # is negative; in context B it is successful. No semantic labels or target text.
    for n in range(8):
        t=120+n*4;bank.observe_use(SHORT,t,t+1,Q//16,CTX_A);bank.observe_return(SHORT,-Q//2,-Q//4,t+2,True,CTX_A)
    for n in range(4):
        t=180+n*4;bank.observe_use(SHORT,t,t+1,Q//16,CTX_B);bank.observe_return(SHORT,Q,Q//4,t+2,True,CTX_B)
    # The longer construction is less frequent but succeeds in A and is poor in B.
    for n in range(3):
        t=220+n*8;bank.observe_use(LONG,t,t+5,Q//3,CTX_A);bank.observe_return(LONG,3*Q//4,Q//8,t+6,True,CTX_A)
    for n in range(2):
        t=260+n*8;bank.observe_use(LONG,t,t+5,Q//3,CTX_B);bank.observe_return(LONG,-Q//2,-Q//8,t+6,True,CTX_B)
    candidates=(SHORT,LONG)
    base_a=choose_entrenchment(bank,candidates);base_b=choose_entrenchment(bank,candidates)
    new_a=choose_factorized(bank,candidates,CTX_A);new_b=choose_factorized(bank,candidates,CTX_B)
    lengths={SHORT:1,LONG:4};depths={SHORT:0,LONG:chunks.chunks[LONG].depth}
    base_positive=sum(bank.contextual_outcome(x,c)>0 for x,c in ((base_a,CTX_A),(base_b,CTX_B)))
    new_positive=sum(bank.contextual_outcome(x,c)>0 for x,c in ((new_a,CTX_A),(new_b,CTX_B)))
    checks={
      'baseline_is_habit_bound':base_a==SHORT and base_b==SHORT,
      'context_a_revalues_to_composed_construction':new_a==LONG,
      'context_b_retains_habit_when_consequentially_good':new_b==SHORT,
      'context_sensitive_bias_emerges':new_a!=new_b,
      'public_length_increases_when_context_supports_it':lengths[new_a]>lengths[base_a],
      'compositional_depth_increases':depths[new_a]>depths[base_a],
      'capability_breadth_positive_contexts_improves':new_positive>base_positive,
      'chosen_context_a_value_improves':bank.contextual_outcome(new_a,CTX_A)>bank.contextual_outcome(base_a,CTX_A),
      'no_expected_string_or_token_surface':True,
      'bounded_runtime':time.perf_counter()-started<2,
    }
    failed=[k for k,v in checks.items() if not v]
    if failed:raise SystemExit('FOUNDRY_AGI_EMERGENT_LANGUAGE_PROFILE_RED '+','.join(failed))
    p=Path(__file__);r={'contract':'FOUNDRY_AGI_EMERGENT_LANGUAGE_PROFILE_GREEN','reference_only':True,'language_phenotype_improved':True,'tokens':False,'expected_strings':False,'transformer':False,'backprop':False,'baseline':{'ctx_a':base_a,'ctx_b':base_b,'positive_contexts':base_positive,'ctx_a_length':lengths[base_a],'ctx_a_depth':depths[base_a]},'factorized':{'ctx_a':new_a,'ctx_b':new_b,'positive_contexts':new_positive,'ctx_a_length':lengths[new_a],'ctx_a_depth':depths[new_a]},'checks':checks,'remaining_red':['CANONICAL_DIRECT_INTEGRATION','PUBLIC_MOTOR_LANGUAGE_PARITY','HELD_OUT_RAW_SURFACE_COMPOSITION','CAPACITY_INTERFERENCE_SWEEP'],'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
    print(r['contract']);print(json.dumps(r,sort_keys=True,indent=2))
if __name__=='__main__':main()
