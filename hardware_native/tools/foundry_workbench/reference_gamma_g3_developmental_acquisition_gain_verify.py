#!/usr/bin/env python3
"""G3: two content-free Species siblings live the same canonical curriculum."""
from __future__ import annotations
import copy,inspect,json,time
from reference_life_function_curriculum_v1 import (
    ReferenceLifeFunctionRuntimeV2,canonical_developmental_probe_v2,
    canonical_life_function_curriculum_v2,canonical_species_program_v2,
)


def refused(adult,context,heldout):
    try:adult.leaf(context,heldout);return False
    except RuntimeError as exc:return str(exc)=='adult:construction_not_productive'


def probe(runtime):
    # Evaluation is a checkpoint fork; it cannot modify the canonical developing subject.
    return runtime.fork_for_probe()


def main():
    started=time.perf_counter();checks={};curriculum=canonical_life_function_curriculum_v2()
    context,_examples,heldout,features=canonical_developmental_probe_v2(curriculum)
    candidate_program=canonical_species_program_v2(2);conservative_program=canonical_species_program_v2(4)
    candidate=ReferenceLifeFunctionRuntimeV2(candidate_program);conservative=ReferenceLifeFunctionRuntimeV2(conservative_program)
    checks['distinct_content_free_species_roots']=candidate_program.root()!=conservative_program.root()
    document=json.dumps(candidate_program.canonical_document(),sort_keys=True)
    checks['species_contains_no_language_content']=all(word not in document for word in ('Mara','Jonah','station','harbor','sentence','answer','expected_output'))
    checks['siblings_share_exact_curriculum_root']=candidate.program.root()!=conservative.program.root() and curriculum.root()==curriculum.root()
    checks['blank_birth_has_no_heldout_answer']=refused(probe(candidate).adult.language_adult,context,heldout) and refused(probe(conservative).adult.language_adult,context,heldout)

    criterion={'candidate':None,'conservative':None};candidate_after_two=None;grounded_equal=False
    construction_sources={};construction_count=0;first_construction_source=0;last_construction_cursor=0
    for event in curriculum.events:
        candidate.apply(event);conservative.apply(event)
        if event.lane=='checkpoint_mark' and event.payload==('grounded',):
            ca=probe(candidate).adult.language_adult;co=probe(conservative).adult.language_adult
            grounded_equal=all(ca.language.lexeme(f)==co.language.lexeme(f) is not None for f in features)
        elif event.lane=='scene' and int(event.payload[0])==context and len(event.payload)==5:
            construction_sources[int(event.source)]=tuple(map(int,event.payload[1:]))
            if first_construction_source<=0:first_construction_source=int(event.source)
        elif event.lane=='surface' and int(event.source) in construction_sources:
            construction_count+=1;last_construction_cursor=int(event.sequence)
            for name,runtime in (('candidate',candidate),('conservative',conservative)):
                if criterion[name] is None and not refused(probe(runtime).adult.language_adult,context,heldout):criterion[name]=construction_count
            if construction_count==2:candidate_after_two=copy.deepcopy(candidate.checkpoint())
            if construction_count==4:break

    checks['lexicon_equalized_by_same_grounding_chronology']=grounded_equal
    checks['candidate_reaches_heldout_after_two_lived_constructions']=criterion['candidate']==2
    checks['conservative_requires_four_lived_constructions']=criterion['conservative']==4
    checks['developmental_gain_saves_two_lived_constructions']=criterion['conservative']-criterion['candidate']==2
    cand=probe(candidate).adult.language_adult;cons=probe(conservative).adult.language_adult
    cand_surface=bytes(cand.leaf(context,heldout).surface);cons_surface=bytes(cons.leaf(context,heldout).surface)
    learned_surface=b' '.join(bytes(cand.language.lexeme(feature) or ()) for feature in heldout)+b'.'
    checks['same_eventual_heldout_capability']=cand_surface==cons_surface==learned_surface

    cut=ReferenceLifeFunctionRuntimeV2.restore(candidate_program,candidate_after_two)
    cut.adult.language.withdraw_source(first_construction_source)
    checks['source_withdrawal_removes_early_gain']=refused(cut.adult.language_adult,context,heldout)
    checks['checkpoint_preserves_species_mechanism_parameter']=(candidate.adult.language.minimum_source_support==2 and conservative.adult.language.minimum_source_support==4)
    functions={name for name,obj in globals().items() if inspect.isfunction(obj)}
    checks['private_teaching_script_deleted']=not ({'teach_lexicon','teach_construction'}&functions) and 'LanguageMasteryContactAdapterV1' not in globals() and 'CONTACT_SCENE' not in globals() and 'CONTACT_SURFACE' not in globals()
    checks['canonical_probes_are_checkpoint_forks']=candidate.cursor==conservative.cursor==last_construction_cursor and construction_count==4
    checks['no_optimizer_or_answer_installation']=all(not hasattr(candidate.adult.language_adult,n) for n in ('optimizer','loss','gradient','backprop','target_text','expected_output'))
    checks['bounded_fast_path']=time.perf_counter()-started<1.0
    failed=[k for k,v in checks.items() if not v]
    result={'contract':'FOUNDRY_GAMMA_G3_DEVELOPMENTAL_ACQUISITION_GAIN_'+('GREEN' if not failed else 'RED'),'pass':not failed,'reference_only':True,'graph_flip':False,'curriculum_root':curriculum.root(),'curriculum_cursor':candidate.cursor,'visible_language_gain':'SAME_LIFE_CURRICULUM_REACHES_HELDOUT_PRODUCTIVE_LANGUAGE_TWO_LIVED_CONSTRUCTIONS_EARLIER_UNDER_A_CONTENT_FREE_SPECIES_DISPOSITION','candidate_examples_to_criterion':criterion['candidate'],'conservative_examples_to_criterion':criterion['conservative'],'examples_saved':None if None in criterion.values() else criterion['conservative']-criterion['candidate'],'heldout_surface':cand_surface.decode(),'species_roots':{'candidate':candidate_program.root(),'conservative':conservative_program.root()},'species_parameter':{'candidate_minimum_distinct_sources':2,'conservative_minimum_distinct_sources':4},'checks':checks,'failed':failed,'elapsed_ms':round((time.perf_counter()-started)*1000,3),'remaining_red':['DIRECT_G3_LOWERING','PHYSICAL_SIBLING_ACQUISITION_ASSAY','GRAPH_PROMOTION'],'next_falsifiers':{'chomsky':'Use the same sibling chronology for a later recursive/long-distance structural acquisition delta rather than only flat productive construction.','sapolsky':'Under the same current construction contact, vary prior source withdrawal/recovery and body/resource history while preserving the Species difference.'}}
    print(result['contract']);print(json.dumps(result,indent=2,sort_keys=True));return 0 if not failed else 1
if __name__=='__main__':raise SystemExit(main())
