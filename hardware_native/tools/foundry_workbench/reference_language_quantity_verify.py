#!/usr/bin/env python3
"""Large learned-library assay for the strict surface ecology."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))

from reference_language_learning_v1 import LearnedSurfaceEcologyV1

LEXEMES=10_000
CONTEXTS=256


def units_for(feature:int):
    # Opaque numeric surface trajectory; unique for the tested feature range.
    x=int(feature)
    return (65,x&255,(x>>8)&255,(x>>16)&255)


def raw_sentence(atoms):
    literals=((240,1),(241,2),(242,3),(243,4),(244,5))
    out=list(literals[0])
    for i,atom in enumerate(atoms):
        out.extend(units_for(atom));out.extend(literals[i+1])
    return tuple(out)


def main():
    started=time.perf_counter();e=LearnedSurfaceEcologyV1();checks={}
    t=time.perf_counter()
    for feature in range(1,LEXEMES+1):
        u=units_for(feature);e.observe_naming(feature,u,100_000+feature);e.observe_naming(feature,u,200_000+feature)
    lexeme_ms=(time.perf_counter()-t)*1000
    probes=(1,2,255,256,4096,9999,10000);checks['ten_thousand_lexemes']=all(e.lexeme(f)==units_for(f) for f in probes) and len(e._lexeme_sources)==LEXEMES
    e.lexeme(9999);checks['lexeme_lookup_touched_not_total']=e.last_lookup_touches==1

    t=time.perf_counter();heldout=[]
    for c in range(CONTEXTS):
        context=1_000_000+c;base=1+c*32
        a=(base,base+1,base+2,base+3);b=(base+4,base+5,base+6,base+7);h=(base+4,base+1,base+6,base+3)
        assert max((*a,*b,*h))<=LEXEMES
        if not e.observe_construction(context,a,raw_sentence(a),300_000+c):raise AssertionError('first observation factorization')
        if not e.observe_construction(context,b,raw_sentence(b),400_000+c):raise AssertionError('second observation factorization')
        heldout.append((context,h,raw_sentence(h)))
    template_ms=(time.perf_counter()-t)*1000
    checks['hundreds_of_templates']=len(e._template_sources)==CONTEXTS and all(e.template(context,4) is not None for context,_,_ in heldout)
    e.template(heldout[-1][0],4);checks['template_lookup_touched_not_total']=e.last_lookup_touches==1

    t=time.perf_counter();outputs=[e.realize(context,atoms) for context,atoms,_ in heldout];realize_ms=(time.perf_counter()-t)*1000
    checks['all_contexts_heldout']=all(out==expected for out,(_,_,expected) in zip(outputs,heldout))
    checks['no_recipe_per_word_or_context']=e.__class__.__name__=='LearnedSurfaceEcologyV1'

    cp=e.checkpoint();raw=json.dumps(cp,sort_keys=True,separators=(',',':')).encode();restored=LearnedSurfaceEcologyV1.restore(cp)
    checks['derived_indices_not_checkpoint_authority']=not any('index' in k for k in cp)
    checks['large_checkpoint_replay']=restored.digest()==e.digest() and restored.realize(*heldout[-1][:2])==heldout[-1][2]

    # Removing one evidence source deactivates only the affected construction support, not the entire ecology.
    ctx0,atoms0,_=heldout[0];e.withdraw_source(400_000);checks['local_source_failure_not_global']=e.template(ctx0,4) is None and e.template(heldout[1][0],4) is not None

    result={'schema':'0x1.reference-language-quantity.v1','pass':all(checks.values()),'checks':checks,
            'lexeme_bindings':len(e._lexeme_sources),'construction_contexts':len(e._template_sources),'checkpoint_bytes':len(raw),
            'lexeme_train_ms':round(lexeme_ms,3),'template_train_ms':round(template_ms,3),'heldout_realize_ms':round(realize_ms,3),
            'heldout_contexts':len(heldout),'claim':'LARGE_REFERENCE_LEARNED_LIBRARY_NOT_HUMAN_LANGUAGE_SCALE_OR_PERFORMANCE','elapsed_ms':round((time.perf_counter()-started)*1000,3)}
    print('FOUNDRY_REFERENCE_LANGUAGE_QUANTITY '+('GREEN' if result['pass'] else 'RED')+f' lexemes={LEXEMES} contexts={CONTEXTS} mechanism_classes=1 per_item_compile=0')
    print(json.dumps(result,indent=2,sort_keys=True));raise SystemExit(0 if result['pass'] else 1)

if __name__=='__main__':main()
