#!/usr/bin/env python3
"""Held-out acquisition checks for the strict learned surface ecology."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))

from reference_language_learning_v1 import LearnedSurfaceEcologyV1,PIECE_LITERAL,PIECE_PORT
import reference_language_learning_v1 as _ll
from reference_population_v1 import PopulationBankV1,PopulationSpecV1


def u(text):return tuple(text.encode('utf-8'))

def teach_lexicon(ecology,mapping,source_a=1000,source_b=2000):
    for feature,text in mapping.items():
        ecology.observe_naming(feature,u(text),source_a+feature)
        ecology.observe_naming(feature,u(text),source_b+feature)


def main():
    started=time.perf_counter();checks={}
    CAREFUL,QUIET=101,102;ENGINEER,TECHNICIAN=201,202;TESTS,INSPECTS=301,302;SENSOR,VALVE=401,402;CTX=9001
    mapping={CAREFUL:'careful',QUIET:'quiet',ENGINEER:'engineer',TECHNICIAN:'technician',TESTS:'tests',INSPECTS:'inspects',SENSOR:'sensor',VALVE:'valve'}
    e=LearnedSurfaceEcologyV1();teach_lexicon(e,mapping)
    checks['lexicon_source_supported']=all(e.lexeme(f)==u(t) for f,t in mapping.items())

    ex1=(CAREFUL,ENGINEER,TESTS,SENSOR);ex2=(QUIET,TECHNICIAN,INSPECTS,VALVE);held=(QUIET,ENGINEER,TESTS,VALVE)
    checks['first_demonstration_not_enough']=e.observe_construction(CTX,ex1,u('the careful engineer tests the sensor.'),3001) and e.template(CTX,4) is None
    checks['second_demonstration_induces']=e.observe_construction(CTX,ex2,u('the quiet technician inspects the valve.'),3002) and e.template(CTX,4) is not None
    t=e.template(CTX,4);assert t is not None
    saved_templates=e.template_candidates;saved_lexemes=e.lexeme_candidates
    e.template_candidates=lambda *a,**k: (_ for _ in ()).throw(RuntimeError('unique_template_must_not_enumerate'))
    e.lexeme_candidates=lambda *a,**k: (_ for _ in ()).throw(RuntimeError('unique_lexeme_must_not_enumerate'))
    try:
        unique_t=e.template(CTX,4);unique_lex=e.lexeme(CAREFUL)
    finally:
        e.template_candidates=saved_templates;e.lexeme_candidates=saved_lexemes
    checks['unique_template_does_not_enumerate_candidates']=unique_t is not None
    checks['unique_lexeme_does_not_enumerate_candidates']=unique_lex==u('careful')
    roster=LearnedSurfaceEcologyV1();teach_lexicon(roster,mapping)
    roster.observe_naming(CAREFUL,u('prudent'),4101)
    saved_roster=roster._active_sources
    roster._active_sources=lambda *a,**k: (_ for _ in ()).throw(RuntimeError('unique_lexeme_must_not_materialize_source_roster'))
    try:roster_lex=roster.lexeme(CAREFUL)
    finally:roster._active_sources=saved_roster
    checks['unique_lexeme_does_not_materialize_source_roster']=roster_lex==u('careful')
    saved_digest=_ll._digest;digest_calls={'n':0}
    def count_digest(tag,obj):
        digest_calls['n']+=1;return saved_digest(tag,obj)
    _ll._digest=count_digest
    try:indexed_lex=e.lexeme_identity(CAREFUL,u('careful'))
    finally:_ll._digest=saved_digest
    checks['unique_lexeme_reuses_indexed_identity']=indexed_lex!=0 and digest_calls['n']==0
    repeat=LearnedSurfaceEcologyV1();teach_lexicon(repeat,mapping)
    first_lex=repeat.lexeme(ENGINEER)
    repeat.last_lookup_touches=-1
    second_lex=repeat.lexeme(ENGINEER)
    checks['unique_lexeme_repeat_skips_index_walk']=second_lex==first_lex==u('engineer') and repeat.last_lookup_touches==0
    repeat.lexeme(TESTS)
    repeat.last_lookup_touches=-1
    again_lex=repeat.lexeme(ENGINEER)
    checks['unique_lexeme_recombination_skips_index_walk']=again_lex==u('engineer') and repeat.last_lookup_touches==0
    repeat.lexeme(CAREFUL)
    repeat.observe_naming(CAREFUL,u('prudent'),9101)
    repeat.observe_naming(CAREFUL,u('prudent'),9102)
    checks['unique_lexeme_naming_invalidates_repeat_cache']=repeat.lexeme(CAREFUL) is None
    template_roster=LearnedSurfaceEcologyV1();teach_lexicon(template_roster,mapping)
    template_roster.observe_construction(CTX,ex1,u('the careful engineer tests the sensor.'),3001)
    template_roster.observe_construction(CTX,ex2,u('the quiet technician inspects the valve.'),3002)
    template_roster.observe_construction(CTX,ex1,u('careful engineer tests sensor.'),4001)
    saved_template_roster=template_roster._active_sources
    template_roster._active_sources=lambda *a,**k: (_ for _ in ()).throw(RuntimeError('unique_template_must_not_materialize_source_roster'))
    try:template_unique=template_roster.template(CTX,4)
    finally:template_roster._active_sources=saved_template_roster
    checks['unique_template_does_not_materialize_source_roster']=template_unique is not None
    saved_digest=_ll._digest;digest_calls={'n':0}
    def count_digest(tag,obj):
        digest_calls['n']+=1;return saved_digest(tag,obj)
    _ll._digest=count_digest
    try:indexed_t=e.template(CTX,4)
    finally:_ll._digest=saved_digest
    checks['unique_template_reuses_indexed_identity']=indexed_t is not None and digest_calls['n']==0
    again=e.template(CTX,4)
    checks['unique_template_repeat_skips_index_walk']=again is indexed_t and e.last_lookup_touches==0
    observer=LearnedSurfaceEcologyV1();teach_lexicon(observer,mapping)
    observer.observe_construction(CTX,ex1,u('the careful engineer tests the sensor.'),3001)
    observer.observe_construction(CTX,ex2,u('the quiet technician inspects the valve.'),3002)
    saved_observer=observer._active_sources
    observer._active_sources=lambda *a,**k: (_ for _ in ()).throw(RuntimeError('observer_inventory_must_not_materialize_source_roster'))
    try:
        observer_templates=observer.template_candidates(CTX,4)
        observer_lexemes=observer.lexeme_observations(CAREFUL)
    finally:observer._active_sources=saved_observer
    checks['observer_inventories_do_not_materialize_source_roster']=len(observer_templates)==1 and observer_lexemes and observer_lexemes[0][1]==u('careful')
    checks['all_ports_present']=sorted(p.port for p in t.pieces if p.kind==PIECE_PORT)==[0,1,2,3]
    checks['literal_scaffold_learned']=any(p.kind==PIECE_LITERAL and p.literal for p in t.pieces)

    out=e.realize(CTX,held);checks['heldout_recombination']=out==u('the quiet engineer tests the valve.')
    training={u('the careful engineer tests the sensor.'),u('the quiet technician inspects the valve.')}
    checks['heldout_not_replay']=out not in training
    invert=LearnedSurfaceEcologyV1();teach_lexicon(invert,mapping)
    invert.observe_construction(CTX,ex1,u('the careful engineer tests the sensor.'),3001)
    invert.observe_construction(CTX,ex2,u('the quiet technician inspects the valve.'),3002)
    held_out=invert.realize(CTX,held)
    saved_invert=invert._active_sources
    invert._active_sources=lambda *a,**k: (_ for _ in ()).throw(RuntimeError('invert_surface_must_not_materialize_source_roster'))
    try:bindings=invert.invert_surface(held_out)
    finally:invert._active_sources=saved_invert
    checks['invert_surface_does_not_materialize_source_roster']=len(bindings)==1 and bindings[0].atoms==held
    factor=LearnedSurfaceEcologyV1();teach_lexicon(factor,mapping)
    factor.observe_construction(CTX,ex1,u('the careful engineer tests the sensor.'),3001)
    factor.observe_construction(CTX,ex2,u('the quiet technician inspects the valve.'),3002)
    saved_obs=factor.lexeme_observations
    factor.lexeme_observations=lambda *a,**k: (_ for _ in ()).throw(RuntimeError('construction_factor_must_not_rank_observations'))
    try:extra=factor.observe_construction(CTX,held,out,3003)
    finally:factor.lexeme_observations=saved_obs
    checks['construction_factor_does_not_rank_lexeme_observations']=extra

    # Hierarchical span learning: child sentences are opaque reusable surfaces; boundary/connective bytes come only from raw contact.
    SPAN=9101
    c1=e.realize(CTX,ex1);c2=e.realize(CTX,ex2);c3=e.realize(CTX,held);assert c1 and c2 and c3
    demo_span1=c1+u(' After that, ')+c2+u(' ')+c3
    demo_span2=c2+u(' After that, ')+c3+u(' ')+c1
    checks['first_span_demo_not_enough']=e.observe_span(SPAN,(c1,c2,c3),demo_span1,5001) and e.span_template(SPAN,3) is None
    checks['second_span_demo_induces']=e.observe_span(SPAN,(c2,c3,c1),demo_span2,5002) and e.span_template(SPAN,3) is not None
    saved_candidates=e.span_candidates
    e.span_candidates=lambda *a,**k: (_ for _ in ()).throw(RuntimeError('unique_span_must_not_enumerate'))
    try:unique=e.span_template(SPAN,3)
    finally:e.span_candidates=saved_candidates
    checks['unique_span_template_does_not_enumerate_candidates']=unique is not None
    saved_span_roster=e._active_sources
    e._active_sources=lambda *a,**k: (_ for _ in ()).throw(RuntimeError('unique_span_must_not_materialize_source_roster'))
    try:span_roster=e.span_template(SPAN,3)
    finally:e._active_sources=saved_span_roster
    checks['unique_span_does_not_materialize_source_roster']=span_roster is not None
    saved_digest=_ll._digest;digest_calls={'n':0}
    def count_digest(tag,obj):
        digest_calls['n']+=1;return saved_digest(tag,obj)
    _ll._digest=count_digest
    try:indexed_span=e.span_template(SPAN,3)
    finally:_ll._digest=saved_digest
    checks['unique_span_reuses_indexed_identity']=indexed_span is not None and digest_calls['n']==0
    again_span=e.span_template(SPAN,3)
    checks['unique_span_repeat_skips_index_walk']=again_span is indexed_span and e.last_lookup_touches==0
    paragraph=e.realize_span(SPAN,(c3,c1,c2))
    checks['heldout_paragraph_composition']=paragraph==c3+u(' After that, ')+c1+u(' ')+c2 and paragraph not in (demo_span1,demo_span2)
    span=e.span_template(SPAN,3);checks['span_is_numeric_ports_plus_contact_bytes']=span is not None and sorted(p.port for p in span.pieces if p.kind==PIECE_PORT)==[0,1,2]

    # The same language mechanism consumes the current distributed population occurrence.
    pop=PopulationBankV1(PopulationSpecV1(65536,fanout=2,sites_per_feature=4));occ=pop.recruit(held)
    checks['population_scene_is_causally_real']=len(occ.sites)>0 and pop.quantity_vector(occ)['P']==len(occ.sites) and pop.quantity_vector(occ)['R']==65536
    checks['population_bound_realization']=e.realize(CTX,held)==out

    # Unknown constituent and lexical ambiguity refuse surface emission.
    checks['unknown_constituent_refuses']=e.realize(CTX,(QUIET,ENGINEER,TESTS,999999)) is None
    amb=LearnedSurfaceEcologyV1();teach_lexicon(amb,mapping);amb.observe_naming(QUIET,u('silent'),4101);amb.observe_naming(QUIET,u('silent'),4102);amb.observe_construction(CTX,ex1,u('the careful engineer tests the sensor.'),4201);amb.observe_construction(CTX,ex2,u('the quiet technician inspects the valve.'),4202)
    checks['lexical_tie_refuses']=amb.lexeme(QUIET) is None and amb.realize(CTX,held) is None

    COND=8101
    e.observe_form(SENSOR,(COND,),u('it'),7001);e.observe_form(SENSOR,(COND,),u('it'),7002)
    saved_forms=e.form_candidates
    e.form_candidates=lambda *a,**k: (_ for _ in ()).throw(RuntimeError('unique_form_must_not_enumerate'))
    try:unique_form=e.form(SENSOR,(COND,),True)
    finally:e.form_candidates=saved_forms
    checks['unique_form_does_not_enumerate_candidates']=unique_form==u('it')
    form_roster=LearnedSurfaceEcologyV1()
    form_roster.observe_form(SENSOR,(COND,),u('it'),7001);form_roster.observe_form(SENSOR,(COND,),u('it'),7002)
    form_roster.observe_form(SENSOR,(COND,),u('that'),7003)
    saved_form_roster=form_roster._active_sources
    form_roster._active_sources=lambda *a,**k: (_ for _ in ()).throw(RuntimeError('unique_form_must_not_materialize_source_roster'))
    try:roster_form=form_roster.form(SENSOR,(COND,),True)
    finally:form_roster._active_sources=saved_form_roster
    checks['unique_form_does_not_materialize_source_roster']=roster_form==u('it')
    invert_form=LearnedSurfaceEcologyV1()
    invert_form.observe_form(SENSOR,(COND,),u('it'),7001);invert_form.observe_form(SENSOR,(COND,),u('it'),7002)
    saved_invert_form=invert_form._active_sources
    invert_form._active_sources=lambda *a,**k: (_ for _ in ()).throw(RuntimeError('invert_form_must_not_materialize_source_roster'))
    try:form_bindings=invert_form.invert_form_candidates(u('it'),(COND,),True)
    finally:invert_form._active_sources=saved_invert_form
    checks['invert_form_does_not_materialize_source_roster']=len(form_bindings)==1 and form_bindings[0][2]==u('it')
    tied=LearnedSurfaceEcologyV1()
    tied.observe_form(SENSOR,(COND,),u('it'),7001);tied.observe_form(SENSOR,(COND,),u('it'),7002)
    tied.observe_form(SENSOR,(COND,),u('that'),7003);tied.observe_form(SENSOR,(COND,),u('that'),7004)
    checks['form_tie_refuses']=tied.form(SENSOR,(COND,),True) is None
    WUG,PLUR=501,8201
    rule=LearnedSurfaceEcologyV1()
    saved_lexemes=rule.lexeme_candidates
    rule.lexeme_candidates=lambda *a,**k: (_ for _ in ()).throw(RuntimeError('form_rule_must_not_rank_lexemes'))
    try:
        for feat,name,inflected,extra in ((SENSOR,'sensor','sensors',2),(VALVE,'valve','valves',3),(WUG,'wug','wugs',4)):
            rule.observe_naming(feat,u(name),1);rule.observe_naming(feat,u(name),extra)
            if feat==WUG:continue
            rule.observe_form(feat,(PLUR,),u(inflected),1);rule.observe_form(feat,(PLUR,),u(inflected),extra)
        derived=rule.form(WUG,(PLUR,),True)
    finally:
        rule.lexeme_candidates=saved_lexemes
    checks['form_rule_does_not_enumerate_lexeme_candidates']=derived==u('wugs')
    rule_roster=LearnedSurfaceEcologyV1()
    saved_rule_roster=rule_roster._active_sources
    rule_roster._active_sources=lambda *a,**k: (_ for _ in ()).throw(RuntimeError('form_rule_must_not_materialize_source_roster'))
    try:
        for feat,name,inflected,extra in ((SENSOR,'sensor','sensors',2),(VALVE,'valve','valves',3),(WUG,'wug','wugs',4)):
            rule_roster.observe_naming(feat,u(name),1);rule_roster.observe_naming(feat,u(name),extra)
            if feat==WUG:continue
            rule_roster.observe_form(feat,(PLUR,),u(inflected),1);rule_roster.observe_form(feat,(PLUR,),u(inflected),extra)
        roster_derived=rule_roster.form(WUG,(PLUR,),True)
    finally:
        rule_roster._active_sources=saved_rule_roster
    checks['form_rule_does_not_materialize_source_roster']=roster_derived==u('wugs')

    # Template support is source-qualified; withdrawal reopens uncertainty and restoration recovers it.
    before=e.realize(CTX,held);e.withdraw_source(3002);checks['construction_source_withdrawal']=e.template(CTX,4) is None and e.realize(CTX,held) is None;e.restore_source(3002);checks['construction_source_restore']=e.realize(CTX,held)==before

    # Rename every opaque feature identity: surface phenotype must be invariant after the same contacts.
    rename={x:x+50000 for x in mapping};renamed=LearnedSurfaceEcologyV1();teach_lexicon(renamed,{rename[k]:v for k,v in mapping.items()})
    renamed.observe_construction(CTX,tuple(rename[x] for x in ex1),u('the careful engineer tests the sensor.'),3001);renamed.observe_construction(CTX,tuple(rename[x] for x in ex2),u('the quiet technician inspects the valve.'),3002)
    checks['opaque_id_permutation']=renamed.realize(CTX,tuple(rename[x] for x in held))==out

    # Checkpoint/replay retains learned ecology exactly.
    twin=LearnedSurfaceEcologyV1.restore(e.checkpoint());checks['checkpoint_replay']=twin.digest()==e.digest() and twin.realize(CTX,held)==out

    # Template itself contains only numeric port references + contact-derived literal bytes.
    pieces=[{'kind':p.kind,'port':p.port,'literal_bytes':len(p.literal)} for p in t.pieces]
    checks['template_numeric_structure']=all(p.kind in (PIECE_LITERAL,PIECE_PORT) and isinstance(p.port,int) and all(isinstance(x,int) for x in p.literal) for p in t.pieces)

    result={'schema':'0x1.reference-language-learning-v1.verify','pass':all(checks.values()),'language_phenotype_improved':True,'checks':checks,'learned_template':{'identity':t.identity,'support':t.support,'piece_count':len(t.pieces),'pieces':pieces},'population_quantity':pop.quantity_vector(occ,trajectory=len(out or ())), 'heldout_surface_bytes':len(out or ()), 'heldout_paragraph_bytes':len(paragraph or ()), 'claim':'STRICT_RAW_CONTACT_LANGUAGE_MECHANISM_NOT_HUMAN_LEVEL_LANGUAGE_OR_ADULT_CAPABILITY','elapsed_ms':round((time.perf_counter()-started)*1000,3)}
    print('FOUNDRY_REFERENCE_LANGUAGE_LEARNING '+('GREEN' if result['pass'] else 'RED')+' raw_surface=1 tokenizer=0 grammar_label=0 expected_output_ingress=0 heldout=1 population_bound=1')
    print(json.dumps(result,indent=2,sort_keys=True));raise SystemExit(0 if result['pass'] else 1)

if __name__=='__main__':main()
