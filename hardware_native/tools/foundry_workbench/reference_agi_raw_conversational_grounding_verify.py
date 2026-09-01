#!/usr/bin/env python3
"""Raw lexical teaching changes later composition across checkpoint/restart."""
from __future__ import annotations

import copy
import hashlib
import json
import time
from pathlib import Path

from reference_language_mastery_adult_v1 import LanguageMasteryAdultV1
from reference_language_mastery_contact_adapter_v1 import (
    CONTACT_SCENE,CONTACT_SURFACE,CONTACT_UTTERANCE,
    LanguageMasteryContactAdapterV1,
)

COREFERENCE_CONTACT,CLAUSE=0xD3F,0xCA11
SENSOR,VALVE=401,402
CAREFUL,QUIET,ENGINEER,TECHNICIAN,TESTS,INSPECTS=601,602,701,702,801,802


def scene(contact,context,atoms,surface,source):
    assert contact.contact(CONTACT_SCENE,(context,*atoms),source)
    return contact.contact(CONTACT_SURFACE,surface,source)


def prepared(coreference_examples=2):
    adult=LanguageMasteryAdultV1();contact=LanguageMasteryContactAdapterV1(adult)
    # Two forms are independently grounded against each referent. The primary
    # form has one more source, keeping ordinary production determinate.
    for feature,primary,alternate,base in (
            (SENSOR,b'sensor',b'gauge',1000),(VALVE,b'valve',b'tap',2000)):
        scene(contact,0,(feature,),primary,base+1)
        scene(contact,0,(feature,),primary,base+2)
        scene(contact,0,(feature,),alternate,base+11)
        scene(contact,0,(feature,),alternate,base+12)
        scene(contact,0,(feature,),primary,base+3)
    words=((CAREFUL,b'careful',2300),(QUIET,b'quiet',2400),
           (ENGINEER,b'engineer',2500),(TECHNICIAN,b'technician',2600),
           (TESTS,b'tests',2700),(INSPECTS,b'inspects',2800))
    for feature,surface,base in words:
        scene(contact,0,(feature,),surface,base+1)
        scene(contact,0,(feature,),surface,base+2)
    # Each surface contains two independently grounded forms participating in
    # one resident referent. Repetition with a different referent induces the
    # carrier and forward port co-reference; no observer installs a role.
    if coreference_examples>=1:
        scene(contact,COREFERENCE_CONTACT,(SENSOR,),b'sensor aka gauge',3001)
    if coreference_examples>=2:
        scene(contact,COREFERENCE_CONTACT,(VALVE,),b'valve aka tap',3002)
    scene(contact,CLAUSE,(CAREFUL,ENGINEER,TESTS,SENSOR),
          b'the careful engineer tests the sensor.',3201)
    scene(contact,CLAUSE,(QUIET,TECHNICIAN,INSPECTS,VALVE),
          b'the quiet technician inspects the valve.',3202)
    return adult,contact


def main():
    started=time.perf_counter();adult,contact=prepared();checks={}
    heldout=(CAREFUL,TECHNICIAN,TESTS,SENSOR)
    before=bytes(adult.leaf(CLAUSE,heldout).surface)

    # The body provides only raw bytes and provenance. The learned carrier and
    # dependency competition nominate one revisable lexical hypothesis.
    identity=contact.contact(CONTACT_UTTERANCE,b'sensor aka dax',4001)
    checks['raw_contact_creates_one_resident_provisional_hypothesis']=(
        identity>0 and any(bytes(row[1])==b'dax'
                           for row in adult.language.provisional_lexemes(SENSOR)))

    # This was the old RED: the adapter-local identity map disappeared here.
    # Resume a fresh membrane over the same learned Adult and settle only through
    # the resident lexeme identity reconstructed from future-relevant state.
    checkpoint=adult.checkpoint();restored=LanguageMasteryAdultV1.restore(
        copy.deepcopy(checkpoint));resumed=LanguageMasteryContactAdapterV1(restored)
    yoked=LanguageMasteryAdultV1.restore(copy.deepcopy(checkpoint))
    checks['yoked_restart_keeps_hypothesis_but_cannot_promote_it']=(
        any(bytes(row[1])==b'dax' for row in yoked.language.provisional_lexemes(SENSOR))
        and bytes(yoked.leaf(CLAUSE,heldout).surface)==before)
    # Consequence is not a curriculum rung.  Let independently ticketed returns
    # compete until the resident public winner changes, with a safety bound
    # derived from the live evidence it must overcome rather than a fixed count.
    incumbent_support=max(row[0] for row in restored.language.lexeme_observations(SENSOR))
    returns=[];ticket=5001
    while bytes(restored.language.lexeme(SENSOR) or ())!=b'dax' and len(returns)<=incumbent_support:
        returns.append(resumed.settle_provisional(identity,ticket,+1,True));ticket+=1
    returns=tuple(returns)
    after=bytes(restored.leaf(CLAUSE,heldout).surface)
    checks['independent_returns_after_restart_change_heldout_composition']=(
        all(bytes(row or ())==b'dax' for row in returns)
        and before==b'the careful technician tests the sensor.'
        and after==b'the careful technician tests the dax.')
    recognized=resumed.contact(CONTACT_UTTERANCE,after,5101)
    checks['learned_form_reenters_raw_comprehension_after_restart']=(
        recognized>0 and resumed.scenes[recognized].atoms==heldout)
    checks['adapter_no_longer_persists_observer_alias_map']=(
        not hasattr(contact,'provisional_aliases') and
        'sensor aka dax' not in json.dumps(checkpoint,sort_keys=True))

    # Chomsky falsifier: the changed public string is an unheard recombination,
    # not replay of either clause exemplar or of the teaching contact.
    checks['structure_not_sentence_replay_drives_public_gain']=(
        after not in (b'the careful engineer tests the sensor.',
                      b'the quiet technician inspects the valve.',
                      b'sensor aka dax'))

    # Sapolsky destructive audit: identical current contact changes with prior
    # development, source availability, contingent return, and counterhistory.
    underdeveloped,under_contact=prepared(1)
    checks['developmental_history_changes_same_current_contact']=(
        under_contact.contact(CONTACT_UTTERANCE,b'sensor aka zorp',6001)==0)
    withdrawn,withdrawn_contact=prepared()
    withdrawn.language.withdraw_source(3001);withdrawn.language.withdraw_source(3002)
    checks['source_withdrawal_changes_same_current_contact']=(
        withdrawn_contact.contact(CONTACT_UTTERANCE,b'sensor aka zorp',6001)==0)
    countered=LanguageMasteryAdultV1.restore(copy.deepcopy(restored.checkpoint()))
    counter_contact=LanguageMasteryContactAdapterV1(countered)
    counter_returns=[]
    while (bytes(countered.language.lexeme(SENSOR) or ())!=b'sensor' and
           len(counter_returns)<=len(returns)):
        counter_returns.append(counter_contact.settle_provisional(identity,ticket,-1,True))
        ticket+=1
    checks['counterhistory_reopens_word_choice_without_erasing_evidence']=(
        bytes(countered.leaf(CLAUSE,heldout).surface)==before and
        any(bytes(row[1])==b'dax' for row in countered.language.lexeme_observations(SENSOR)))

    ambiguous,ambiguous_contact=prepared();OTHER=499
    # Hostile intervention creates a second lawful owner for the anchor without
    # changing the acquired construction. The capability path above remains body-only.
    ambiguous.language.observe_naming(OTHER,b'sensor',7001)
    ambiguous.language.observe_naming(OTHER,b'sensor',7002)
    ambiguous_contact.contact(CONTACT_UTTERANCE,b'sensor aka blicket',7003)
    checks['referent_competition_refuses_ambiguous_anchor']=(
        all(bytes(row[1])!=b'blicket'
            for feature in (SENSOR,OTHER)
            for row in ambiguous.language.lexeme_observations(feature)))
    checks['bounded_fast_path']=time.perf_counter()-started<1.0

    failed=[name for name,passed in checks.items() if not passed]
    result={
        'schema':'cyber-lagoon.raw-conversational-grounding.v3',
        'contract':'FOUNDRY_AGI_RAW_CONVERSATIONAL_GROUNDING_'+
                   ('GREEN' if not failed else 'RED'),
        'pass':not failed,'reference_only':True,'runtime_llm':False,
        'language_phenotype_improved':not failed,
        'visible_before':before.decode(),'visible_after':after.decode(),
        'checkpoint_restart_between_teaching_and_consequence':True,
        'consequence_competition':{'positive_returns':len(returns),
                                   'counter_returns':len(counter_returns),
                                   'incumbent_live_support':incumbent_support},
        'checks':checks,'failed':failed,
        'remaining_red':['NEW_REFERENT_NOT_ONLY_NEW_FORM',
                         'LEARNED_LEXICAL_CLARIFICATION_QUESTION',
                         'DIRECT_PARITY','BROAD_HUMAN_DIALOGUE'],
        'elapsed_ms':round((time.perf_counter()-started)*1000,3),
        'sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    print(result['contract']);print(json.dumps(result,indent=2,sort_keys=True))
    raise SystemExit(0 if not failed else 1)


if __name__=='__main__':main()
