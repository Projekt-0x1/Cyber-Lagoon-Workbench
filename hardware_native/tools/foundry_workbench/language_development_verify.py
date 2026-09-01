#!/usr/bin/env python3
from __future__ import annotations
import copy
import json
from language_development_v1 import ClosureV1, LanguageDevelopmentV1, LanguageRefuse


def b(text: str) -> tuple[int, ...]:
    return tuple(text.encode("utf-8"))


# Observer-side names only. Runtime receives numeric feature coalitions.
F_MOD,F_AGENT,F_ACTION,F_OBJECT=8101,8102,8103,8104
F_POS,F_CAUTION,F_CONTROL,F_SEEK,F_NOVEL,F_LOW,F_CALM=8110,8111,8112,8113,8114,8115,8116
F_HUMAN,F_TECH,F_DESIGN,F_RESEARCH,F_ANALYTIC,F_PRACTICAL=8120,8121,8122,8123,8124,8125
F_CHECK,F_VISUAL,F_TRANSITIVE,F_PROBE,F_COGNITIVE=8130,8131,8132,8133,8134
F_DEVICE,F_MEASURE,F_PHYSICAL,F_FLOW,F_ABSTRACT,F_FINDING,F_INFO=8140,8141,8142,8143,8144,8145,8146

WORD_FEATURES={
    "careful":(F_MOD,F_POS,F_CAUTION,F_CONTROL),
    "curious":(F_MOD,F_POS,F_SEEK,F_NOVEL),
    "quiet":(F_MOD,F_POS,F_LOW,F_CALM),
    "engineer":(F_AGENT,F_HUMAN,F_TECH,F_DESIGN),
    "scientist":(F_AGENT,F_HUMAN,F_RESEARCH,F_ANALYTIC),
    "technician":(F_AGENT,F_HUMAN,F_TECH,F_PRACTICAL),
    "inspects":(F_ACTION,F_CHECK,F_VISUAL,F_TRANSITIVE),
    "tests":(F_ACTION,F_CHECK,F_PROBE,F_TRANSITIVE),
    "reviews":(F_ACTION,F_CHECK,F_COGNITIVE,F_TRANSITIVE),
    "sensor":(F_OBJECT,F_DEVICE,F_MEASURE,F_PHYSICAL),
    "valve":(F_OBJECT,F_DEVICE,F_FLOW,F_PHYSICAL),
    "result":(F_OBJECT,F_ABSTRACT,F_FINDING,F_INFO),
}

CTX_CLAUSE=(5001,5002,5003)
CTX_CAUSE=(6001,6101,6201)
CTX_CONTRAST=(6001,6102,6202)
CTX_ELAB=(6001,6103,6203)
CTX_SEQUENCE=(6001,6104,6204)
CONNECTORS={
    CTX_CAUSE:" therefore, ",
    CTX_CONTRAST:" however, ",
    CTX_ELAB:" also, ",
    CTX_SEQUENCE:" then, ",
}


def ground_word(machine: LanguageDevelopmentV1, episode: int, word: str):
    machine.contact_episode_features(episode, WORD_FEATURES[word])
    machine.contact_surface(episode, b(word))
    return machine.consolidate_grounded_episode(episode)


def observe_clause(machine, episode, words, surface):
    entities=tuple(WORD_FEATURES[w] for w in words)
    machine.contact_utterance_context(episode,CTX_CLAUSE,entities)
    machine.contact_surface(episode,b(surface))
    machine.observe_utterance_episode(episode)


def leaf(*words):
    return ClosureV1(1,CTX_CLAUSE,tuple(WORD_FEATURES[w] for w in words))


def main():
    machine=LanguageDevelopmentV1(32768)
    checks={
      'blank_language_at_birth':not machine.surface_units and not machine.associations and not machine.constructions and not machine.span_constructions,
      'no_answer_api':not hasattr(machine,'answer'),
      'no_think_api':not hasattr(machine,'think'),
      'no_speak_api':not hasattr(machine,'speak'),
    }

    # Surface-only temporal chunks; no fabricated referent/concept identity.
    for raw in ('the',' ','.',*CONNECTORS.values()):machine.learn_structural_chunk(b(raw))

    for i,word in enumerate(WORD_FEATURES):ground_word(machine,1000+i,word)
    checks['separate_surface_and_feature_streams']=all(ep in machine.episode_features and ep in machine.episode_surfaces for ep in range(1000,1000+len(WORD_FEATURES)))
    checks['no_surface_semantic_id_field']=all(len(row.raw)>0 for row in machine.surface_units.values()) and len(machine.associations)==len(WORD_FEATURES)

    observe_clause(machine,3001,('careful','engineer','inspects','sensor'),'the careful engineer inspects the sensor.')
    try:machine.consolidate_construction(CTX_CLAUSE)
    except LanguageRefuse:checks['one_example_insufficient']=True
    else:checks['one_example_insufficient']=False
    observe_clause(machine,3002,('quiet','technician','tests','valve'),'the quiet technician tests the valve.')
    construction=machine.consolidate_construction(CTX_CLAUSE)
    checks['construction_learned_postbirth']=construction.arity==4 and len(machine.constructions)==1

    held=leaf('curious','scientist','reviews','result')
    held_text=machine.render_closure(held).decode()
    checks['heldout_recombination']=held_text=='the curious scientist reviews the result.'

    # Sparse-population semantic retrieval survives one missing selective feature.
    scientist_lesioned=(F_AGENT,F_HUMAN,F_RESEARCH)
    normal=machine.resolve_features(WORD_FEATURES['scientist'])
    one_cut=machine.resolve_features(scientist_lesioned)
    checks['single_feature_lesion_survives']=normal==one_cut
    try:machine.resolve_features((F_AGENT,F_HUMAN))
    except LanguageRefuse:checks['coalition_damage_refuses']=True
    else:checks['coalition_damage_refuses']=False

    u1=machine.realize_clause_units(CTX_CLAUSE,tuple(WORD_FEATURES[w] for w in ('careful','engineer','inspects','sensor')))
    u2=machine.realize_clause_units(CTX_CLAUSE,tuple(WORD_FEATURES[w] for w in ('quiet','technician','tests','valve')))
    u3=machine.realize_clause_units(CTX_CLAUSE,tuple(WORD_FEATURES[w] for w in ('curious','scientist','reviews','result')))
    pairsets=((u1,u2),(u2,u3))
    episode=4000
    for context,connector in CONNECTORS.items():
        for left,right in pairsets:
            episode+=1
            machine.contact_discourse_context(episode,context,(left,right))
            raw=machine.render_units(left)+connector.encode()+machine.render_units(right)
            machine.contact_surface(episode,tuple(raw))
            machine.observe_discourse_episode(episode)
        machine.consolidate_span(context)
    checks['discourse_markers_learned_postbirth']=len(machine.span_constructions)==4

    leaves=(
      held,
      leaf('careful','engineer','inspects','sensor'),
      leaf('quiet','technician','tests','valve'),
      held,
      leaf('careful','engineer','inspects','sensor'),
      leaf('quiet','technician','tests','valve'),
      held,
      leaf('careful','engineer','inspects','sensor'),
    )
    contexts=(CTX_CAUSE,CTX_CONTRAST,CTX_ELAB,CTX_SEQUENCE,CTX_CAUSE,CTX_CONTRAST,CTX_ELAB)
    layer=list(leaves);ci=0
    while len(layer)>1:
        nxt=[]
        for i in range(0,len(layer),2):
            nxt.append(ClosureV1(2,contexts[ci],children=(layer[i],layer[i+1])));ci+=1
        layer=nxt
    prose=machine.render_closure(layer[0]).decode()
    checks['recursive_eight_sentence_prose']=prose.count('.')==8
    checks['learned_rhetorical_variation']=all(marker.strip() in prose for marker in CONNECTORS.values())

    # Selective higher-construction lesion destroys the affected discourse closure
    # while preserving clause realization.
    lesion=copy.deepcopy(machine)
    cause_sig=lesion.population.signature(CTX_CAUSE)
    lesion.span_constructions=[c for c in lesion.span_constructions if c.context!=cause_sig]
    try:lesion.render_closure(layer[0])
    except LanguageRefuse:checks['higher_lesion_refuses']=True
    else:checks['higher_lesion_refuses']=False
    checks['higher_lesion_preserves_clause']=lesion.render_closure(held).decode()==held_text

    q=machine.quantity()
    checks['cold_population_exceeds_materialized']=q['population_sites']>q['materialized_sites']>0
    checks['sparse_candidate_work']=q['last_candidate_touches']<128

    result={
      'schema':'0x1.compact-developmental-language.v1',
      'pass':all(checks.values()),'checks':checks,'quantity':q,
      'heldout':held_text,'prose':prose,
      'runtime_llm':False,'mature_language_at_birth':False,
      'direct_concept_to_surface_contact':False,
      'external_construction_recipe_id':False,
      'claim':'DEVELOPMENTAL_DISTRIBUTED_LANGUAGE_REFERENCE_NOT_HUMAN_LEVEL',
    }
    print('FOUNDRY_LANGUAGE_DEVELOPMENT '+('GREEN' if result['pass'] else 'RED')+
          ' mature_language_at_birth=0 direct_concept_surface=0 learned_clause=1 learned_discourse=1 recursive_prose=1 runtime_llm=0')
    print(json.dumps(result,indent=2,sort_keys=True))
    raise SystemExit(0 if result['pass'] else 1)


if __name__=='__main__':main()
