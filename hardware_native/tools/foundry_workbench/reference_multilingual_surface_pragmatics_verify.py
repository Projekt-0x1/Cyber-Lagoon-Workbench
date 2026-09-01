#!/usr/bin/env python3
"""Focused component matrix for open-speech pragmatic boundary induction.

This is intentionally not a developmental-curriculum authority. The canonical same-life
proof lives in reference_life_function_curriculum_verify.py.
"""
from __future__ import annotations
import json

from reference_cultural_perspective_geometry_v1 import Q
from reference_developmental_social_allostasis_v1 import DevelopmentalSocialAllostasisV1
from reference_recursive_cultural_pragmatics_v1 import RecursiveCulturalPragmaticsV1
from reference_multilingual_surface_pragmatics_v1 import LANGUAGES,MultilingualSurfacePragmaticsV1

COUNTER=0xE301; INSULT=0xE302; ROOM=0xE201
SOURCE=0xE401; QUOTED_SOURCE=0xE402
FORMS={
    "en":(("counter",COUNTER),("idiot",INSULT)),
    "de":(("kontere",COUNTER),("idiot",INSULT)),
    "ru":(("ответь",COUNTER),("идиот",INSULT)),
    "ja":(("反撃",COUNTER),("ばか",INSULT)),
    "zh":(("反击",COUNTER),("笨蛋",INSULT)),
    "mixed":(("counter",COUNTER),("clown",INSULT)),
}
DIRECT={
    "en":"Counter!",
    "de":"Kontere!",
    "ru":"Ответь!",
    "ja":"反撃しろ！",
    "zh":"反击！",
    "mixed":"Bro, counter jetzt!",
}
QUOTED={
    "en":"Dana said: \"Counter!\"",
    "de":"Dana sagte: „Kontere!“",
    "ru":"Дана сказала: «Ответь!»",
    "ja":"ダナは「反撃しろ！」と言った。",
    "zh":"达娜说：“反击！”",
    "mixed":"Dana so: \"Bro, counter jetzt!\"",
}
INSULTS={
    "en":"You are an idiot.",
    "de":"Du bist ein Idiot.",
    "ru":"Ты идиот.",
    "ja":"お前はばかだ。",
    "zh":"你是笨蛋。",
    "mixed":"Bro, absolute clown.",
}


def matrix(values):
    return set(values)==set(LANGUAGES)


def main():
    checks={};rows={}
    parser=MultilingualSurfacePragmaticsV1(FORMS);bridge=RecursiveCulturalPragmaticsV1()

    direct={}
    for i,language in enumerate(LANGUAGES):
        emission=parser.feed(100+i,language,DIRECT[language],10+i*10,True)
        direct[language]=emission
        assert bridge.observe_induced_root(parser.inducer,ROOM,emission.root,100+i,emission.tick)
    direct_roots={x.root for x in direct.values()}
    checks["same_concept_same_recursive_root_across_multilingual_matrix"]=(
        matrix(direct) and len(direct_roots)==1
        and all(x.concept_identity==COUNTER and x.force=="imperative" for x in direct.values())
        and all(bridge.action_from_induced_root(parser.inducer,x.root)==COUNTER for x in direct.values()))
    rows["direct_root"]=next(iter(direct_roots))

    quoted={}
    for i,language in enumerate(LANGUAGES):
        emission=parser.feed(200+i,language,QUOTED[language],100+i*10,True,QUOTED_SOURCE)
        quoted[language]=emission
        assert bridge.observe_induced_root(parser.inducer,ROOM,emission.root,200+i,emission.tick)
    quote_roots={x.root for x in quoted.values()}
    checks["quoted_imperative_loses_top_level_force_across_multilingual_matrix"]=(
        matrix(quoted) and len(quote_roots)==1 and quote_roots!=direct_roots
        and all(x.concept_identity==COUNTER and x.force=="quoted_imperative" for x in quoted.values())
        and all(bridge.action_from_induced_root(parser.inducer,x.root)==0 for x in quoted.values()))
    rows["quoted_root"]=next(iter(quote_roots))

    interruption=[]
    for i,language in enumerate(LANGUAGES):
        p=MultilingualSurfacePragmaticsV1(FORMS);speaker_a=300+i;speaker_b=400+i
        text=DIRECT[language];cut=max(1,len(text)//2)
        assert p.feed(speaker_a,language,text[:cut],1000+i*20,False) is None
        b=p.feed(speaker_b,language,text,1001+i*20,True)
        a=p.feed(speaker_a,language,text[cut:],1002+i*20,True)
        roots=p.inducer.roots_since()
        interruption.append(p.pending_speakers()==() and roots[-2][1]==speaker_b and roots[-1][1]==speaker_a
                            and a.root==b.root and a.concept_identity==b.concept_identity==COUNTER)
    checks["speaker_local_interruption_preserves_same_boundary_across_multilingual_matrix"]=(
        len(interruption)==len(LANGUAGES) and all(interruption))

    body={}
    for i,language in enumerate(LANGUAGES):
        p=MultilingualSurfacePragmaticsV1(FORMS)
        emission=p.feed(500+i,language,INSULTS[language],2000+i*10,True)
        soma=DevelopmentalSocialAllostasisV1();tick=3000
        soma.observe_betrayal(SOURCE,tick,Q)
        for _ in range(4):soma.observe_controllability(SOURCE,emission.concept_identity,Q,True)
        loaded=soma.appraise(SOURCE,emission.concept_identity,tick+1,Q)
        recovered=soma.appraise(SOURCE,emission.concept_identity,tick+180,0)
        body[language]=(emission.root,emission.concept_identity,int(loaded["interference_q16"]),
                        int(recovered["interference_q16"]))
    body_signatures={(v[1],v[2],v[3]) for v in body.values()}
    checks["same_grounded_insult_drives_same_somatic_transition_across_multilingual_matrix"]=(
        matrix(body) and len({v[0] for v in body.values()})==1 and len(body_signatures)==1
        and all(v[1]==INSULT for v in body.values()))
    rows["body_signature"]=next(iter(body_signatures))

    scale=MultilingualSurfacePragmaticsV1(FORMS);scale_bridge=RecursiveCulturalPragmaticsV1()
    for i in range(100000):
        language=LANGUAGES[i%len(LANGUAGES)]
        emission=scale.feed(600+(i%len(LANGUAGES)),language,DIRECT[language],4000+i,True)
        scale_bridge.observe_induced_root(scale.inducer,1+(i%64),emission.root,600+(i%len(LANGUAGES)),4000+i)
    snap={"surface":scale.checkpoint(),"bridge":scale_bridge.checkpoint()}
    blob=json.dumps(snap,sort_keys=True,separators=(",",":"),ensure_ascii=False).casefold()
    forbidden=tuple(x.casefold() for x in list(DIRECT.values())+list(QUOTED.values())+list(INSULTS.values()))
    checks["quantity_100k_open_surface_contacts_is_bounded_nontranscript_across_matrix"]=(
        len(scale.inducer.roots_since())<=8192 and len(snap["bridge"]["geometry"]["rows"])<=64*3
        and len(blob)<1000000 and not any(x in blob for x in forbidden))

    failed=[k for k,v in checks.items() if not v]
    result={
        "schema":"cyber-lagoon.multilingual-surface-pragmatics-component.v2",
        "contract":"FOUNDRY_MULTILINGUAL_SURFACE_PRAGMATICS_COMPONENT_"+("GREEN" if not failed else "RED"),
        "pass":not failed,"reference_only":True,"runtime_llm":False,"novel_synthesis":True,
        "languages":list(LANGUAGES),"checks":checks,"failed":failed,
        "invariants":rows,
        "scale":{"open_surface_contacts":100000,"resident_roots":len(scale.inducer.roots_since()),
                 "perspective_rows":len(snap["bridge"]["geometry"]["rows"]),"checkpoint_bytes":len(blob)},
        "remaining_red":["RAW_ACOUSTIC_DIARIZATION_AND_PROSODY","LEARNED_SURFACE_LEXICON_ACQUISITION",
                         "DIRECT_MULTIPARTY_SOCIAL_PARITY","BROAD_HUMAN_PARTY_DIALOGUE"],
        "claim_boundary":"Focused component falsifier only; not a same-life curriculum receipt and not empirical proof of Chomsky, Sapolsky, or AGI.",
    }
    print(result["contract"]);print(json.dumps(result,indent=2,sort_keys=True,ensure_ascii=False));return 0 if not failed else 1

if __name__=="__main__":raise SystemExit(main())
