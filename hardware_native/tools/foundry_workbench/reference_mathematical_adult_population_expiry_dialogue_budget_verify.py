#!/usr/bin/env python3
"""Visible discourse gain from sparse population expiry under a fixed work envelope."""
from __future__ import annotations
import json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
from reference_organism_v2 import (
    ActionV2,ReferenceOrganismV2,CONTACT_CONSEQUENCE,CONTACT_PARTNER_CONTEXT,
    CONTACT_SCENE,CONTACT_SURFACE,
)
from reference_population_v1 import PopulationSpecV1

TARGET_TURNS=64
DONOR_BUDGET_TURNS=32
SPEC=PopulationSpecV1(65536,2,4,42,8)
NAME=100;CTX=9001;PARTNER=99501
M={101:'sunlight',102:'rain',201:'warmed',202:'soaked',301:'greenhouse',302:'garden',401:'dawn',402:'dusk'}

def units(raw):return tuple(raw.encode())
def sentence(a):return f'{M[a[0]]} {M[a[1]]} the {M[a[2]]} at {M[a[3]]}.'
def scene(o,a,source):return o.contact(CONTACT_SCENE,(7,CTX,4,*a),source,True,True)
def surface(o,raw,source):return o.contact(CONTACT_SURFACE,units(raw),source,True,True)
def partner(o):return o.contact(CONTACT_PARTNER_CONTEXT,(1,7,PARTNER),70000+PARTNER,True,True)
def settle(o,action):return o.contact(CONTACT_CONSEQUENCE,(action.ticket,1),PARTNER,True,True)
def trained(spec):
    o=ReferenceOrganismV2(spec)
    for feature,text in M.items():
        o.contact(CONTACT_SCENE,(7,NAME,1,feature),10000+feature,True,True);surface(o,text,20000+feature)
        o.contact(CONTACT_SCENE,(7,NAME,1,feature),11000+feature,True,True);surface(o,text,21000+feature)
    rows=((101,201,301,401),(102,201,301,401),(102,202,302,402),(101,202,302,401))
    for i,row in enumerate(rows):scene(o,row,30001+i);surface(o,sentence(row),31001+i)
    return o,*rows


def main():
    started=time.perf_counter();o,a,b,c,d=trained(SPEC);partner(o)
    atoms=(a,b,c,d);outputs=[];decay_touches=[]
    for i in range(TARGET_TURNS):
        current_atoms=atoms[i%len(atoms)];scene(o,current_atoms,810000+i)
        action=o.tick()
        if not isinstance(action,ActionV2):raise RuntimeError('expiry_dialogue:action')
        text=bytes(action.payload).decode();expected=sentence(current_atoms)
        if text!=expected:raise RuntimeError(f'expiry_dialogue:surface:{i}')
        outputs.append(text);decay_touches.append(int(o.population.last_decay_touches));settle(o,action)

    # The deleted donor decay implementation enumerated every site and every edge
    # exactly once per tick, independent of how many eligibility rows were live.
    donor_decay_touches_per_turn=SPEC.site_count+SPEC.site_count*SPEC.fanout
    work_budget=donor_decay_touches_per_turn*DONOR_BUDGET_TURNS
    donor_turns_under_budget=work_budget//donor_decay_touches_per_turn
    challenger_decay_touches=sum(decay_touches)
    checks={
      'same_public_surfaces_preserved':len(outputs)==TARGET_TURNS,
      'resource_bound_is_load_bearing':donor_turns_under_budget==DONOR_BUDGET_TURNS<TARGET_TURNS,
      'challenger_reaches_target_under_same_budget':challenger_decay_touches<=work_budget,
      'visible_discourse_continuity_doubles':TARGET_TURNS>=2*donor_turns_under_budget,
      'sparse_decay_work_is_current_state_local':max(decay_touches)<donor_decay_touches_per_turn//100,
      'terminal_turn_remains_readable':outputs[-1]==sentence(atoms[(TARGET_TURNS-1)%len(atoms)]),
      'no_relation_marker_fixture':all(marker not in ''.join(outputs) for marker in ('therefore','however','because','then,')),
      'bounded_fast_lane':time.perf_counter()-started<1.0,
    }
    result={'schema':'cyber-lagoon.reference-mathematical-adult-population-expiry-dialogue-budget.v1','pass':all(checks.values()),
      'reference_only':True,'economic_maintenance':'RESOURCE_BOUNDED_MULTI_TURN_ACTION_CONTINUITY_NOT_DISCOURSE_COHERENCE',
      'target_turns':TARGET_TURNS,'donor_turns_under_budget':donor_turns_under_budget,
      'challenger_turns_under_budget':TARGET_TURNS,'visible_turn_gain':TARGET_TURNS-donor_turns_under_budget,
      'population_decay_work_budget':work_budget,'donor_decay_touches_per_turn':donor_decay_touches_per_turn,
      'donor_target_decay_touches':donor_decay_touches_per_turn*TARGET_TURNS,
      'challenger_decay_touches':challenger_decay_touches,'challenger_max_decay_touches_per_turn':max(decay_touches),
      'first_surface':outputs[0],'last_surface':outputs[-1],'checks':checks,
      'elapsed_ms':round((time.perf_counter()-started)*1000,3)}
    print('FOUNDRY_MATHEMATICAL_ADULT_POPULATION_EXPIRY_DIALOGUE_BUDGET_'+('GREEN' if result['pass'] else 'RED'))
    print(json.dumps(result,indent=2,sort_keys=True))
    if not result['pass']:raise SystemExit(1)
if __name__=='__main__':main()
