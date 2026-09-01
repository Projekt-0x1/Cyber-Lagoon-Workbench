#!/usr/bin/env python3
"""Hostile R0 for lexical spans and port order learned from lived scene incidence."""
from __future__ import annotations

import copy,json,time

from reference_language_learning_v1 import LearnedSurfaceEcologyV1

A,B,C,D,E,F=101,102,301,302,501,502
CLAUSE=700


def u(text):return tuple(text.encode())


EPISODES=(
    ((A,C),'bright sunlight reaches the greenhouse.',1001),
    ((D,A),'the soil changes under bright sunlight.',1002),
    ((B,A),'warm air follows bright sunlight.',1003),
    ((C,B),'the greenhouse holds warm air.',1004),
    ((B,D),'warm air moves above the soil.',1005),
    ((D,C),'the soil borders the greenhouse.',1006),
)


def train(ecology,episodes=EPISODES):
    for atoms,raw,source in episodes:
        ecology.observe_scene_surface(0,atoms,u(raw),source)


def main():
    started=time.perf_counter();e=LearnedSurfaceEcologyV1()
    train(e)
    lexical={feature:bytes(e.lexeme(feature) or ()) for feature in (A,B,C,D)}

    # Observer order is intentionally reversed. Surface incidence must own ports.
    first=e.observe_scene_surface(CLAUSE,(C,A),u('bright sunlight changes the greenhouse.'),2001)
    second=e.observe_scene_surface(CLAUSE,(D,B),u('warm air changes the soil.'),2002)
    heldout=bytes(e.realize(CLAUSE,(A,D)) or ())

    # Sapolsky developmental-history control: same current contact after loss of
    # contrast cannot acquire a mapping that the full prior life acquired.
    lesioned=LearnedSurfaceEcologyV1()
    train(lesioned,tuple(row for row in EPISODES if A not in row[0])+
          (((A,C),'bright sunlight reaches the greenhouse.',1001),))

    # Two referents with identical developmental incidence must compete forever;
    # the host may not assign either raw span to either resident identity.
    confounded=LearnedSurfaceEcologyV1()
    for source,raw in ((3001,'alpha beta appears.'),(3002,'we saw alpha beta.'),(3003,'again: alpha beta!')):
        confounded.observe_scene_surface(0,(E,F),u(raw),source)

    checkpoint=e.checkpoint();restored=LearnedSurfaceEcologyV1.restore(copy.deepcopy(checkpoint))
    before_withdraw=restored.lexeme(A)
    restored.withdraw_source(1001);restored.withdraw_source(1002);restored.withdraw_source(1003)
    after_withdraw=restored.lexeme(A)
    checks={
        'raw_scene_incidence_learns_readable_lexical_spans':lexical=={
            A:b'bright sunlight',B:b'warm air',C:b'the greenhouse',D:b'the soil'},
        'observer_scene_order_does_not_own_construction_ports':bool(first[0]) and bool(second[0]),
        'heldout_composition_is_readable':heldout==b'bright sunlight changes the soil.',
        'contrast_lesion_prevents_same_current_contact_settlement':lesioned.lexeme(A) is None,
        'incidence_confound_remains_unsettled':confounded.lexeme(E) is None and confounded.lexeme(F) is None,
        'checkpoint_retains_future_causal_math':LearnedSurfaceEcologyV1.restore(copy.deepcopy(checkpoint)).realize(CLAUSE,(A,D))==u('bright sunlight changes the soil.'),
        'source_history_revises_future_availability':before_withdraw==u('bright sunlight') and after_withdraw is None,
        'bounded_fast_path':time.perf_counter()-started<1.0,
    }
    failed=[key for key,value in checks.items() if not value]
    result={'schema':'cyber-lagoon.cross-situational-constituent-induction.v1',
            'pass':not failed,'reference_only':True,'lexical':{str(k):v.decode() for k,v in lexical.items()},
            'heldout':heldout.decode(errors='replace'),'checks':checks,
            'checkpoint_bytes':len(json.dumps(checkpoint,sort_keys=True,separators=(',',':')).encode()),
            'elapsed_ms':round((time.perf_counter()-started)*1000,3)}
    print('FOUNDRY_CROSS_SITUATIONAL_CONSTITUENT_INDUCTION_'+('GREEN' if not failed else 'RED'))
    print(json.dumps(result,indent=2,sort_keys=True))
    if failed:raise SystemExit(1)


if __name__=='__main__':main()
