#!/usr/bin/env python3
"""Bounded multilingual surface-speech -> recursive pragmatic structure transducer.

This reference layer consumes UTF-8 utterance chunks rather than pre-supplied frame-kind events.
Language-specific surface forms are grounded into shared concept identities by an explicit lexicon,
while quotation and imperative force are induced from the current utterance shape. Completed roots
are delegated to ContinuousPerspectiveInducerV1, so all downstream cultural, epistemic, somatic,
and action arbitration operates on language-invariant recursive structure rather than transcript
strings. Pending text exists only while a speaker has an unfinished utterance and is never
checkpointed.
"""
from __future__ import annotations
from dataclasses import dataclass
import re
import unicodedata

from reference_continuous_perspective_inducer_v1 import (
    ContinuousPerspectiveInducerV1, K_ASSERT, K_QUOTE, K_IMPERATIVE
)

LANGUAGES=("en","de","ru","ja","zh","mixed")
MAX_PENDING_CHARS=512
MAX_FORMS_PER_LANGUAGE=256

QUOTE_PAIRS=(("\"","\""),("“","”"),("„","“"),("«","»"),("「","」"),("『","』"))
JA_IMPERATIVE_SUFFIXES=("しろ","せよ","して","してください","なさい")
ZH_IMPERATIVE_PREFIXES=("请","請","给我","給我")
ZH_IMPERATIVE_SUFFIXES=("吧","！","!")


def _norm(text):
    text=unicodedata.normalize("NFKC",str(text)).casefold().strip()
    return re.sub(r"\s+"," ",text)


@dataclass(frozen=True)
class SurfaceEmissionV1:
    root:int
    concept_identity:int
    force:str
    language:str
    speaker:int
    tick:int


class MultilingualSurfacePragmaticsV1:
    """Derive recursive force from open surface chunks; retain no completed transcript."""

    def __init__(self,concept_forms):
        self.inducer=ContinuousPerspectiveInducerV1()
        self._forms={}
        for language,rows in dict(concept_forms).items():
            language=str(language)
            if language not in LANGUAGES:raise ValueError("surface-pragmatics:language")
            if len(rows)>MAX_FORMS_PER_LANGUAGE:raise RuntimeError("surface-pragmatics:form-capacity")
            normalized=[]
            for surface,concept in rows:
                surface=_norm(surface);concept=int(concept)
                if not surface or concept<=0:raise ValueError("surface-pragmatics:form")
                normalized.append((surface,concept))
            normalized.sort(key=lambda x:(-len(x[0]),x[0],x[1]))
            self._forms[language]=tuple(normalized)
        if set(self._forms)!=set(LANGUAGES):raise ValueError("surface-pragmatics:language-matrix")
        self._pending={}

    @staticmethod
    def _quote_span(text):
        best=None
        for opener,closer in QUOTE_PAIRS:
            start=text.find(opener)
            if start<0:continue
            end=text.find(closer,start+len(opener))
            if end<0:continue
            candidate=(start,end+len(closer),text[start+len(opener):end])
            if best is None or candidate[0]<best[0]:best=candidate
        return best

    def _resolve_concept(self,language,text):
        normalized=_norm(text)
        matches=[]
        for surface,concept in self._forms[language]:
            if surface in normalized:matches.append((len(surface),surface,concept))
        if not matches:return 0
        matches.sort(reverse=True)
        best_len=matches[0][0]
        concepts={concept for length,_surface,concept in matches if length==best_len}
        if len(concepts)!=1:raise RuntimeError("surface-pragmatics:ambiguous-concept")
        return concepts.pop()

    def _is_imperative(self,language,text,concept):
        if int(concept)<=0:return False
        normalized=_norm(text)
        if language in ("en","de","ru","mixed"):
            return normalized.endswith("!") or normalized.startswith(("please ","bitte ","пожалуйста ","bro ","ey "))
        if language=="ja":
            bare=normalized.rstrip("。.!！?？")
            return normalized.endswith(("!","！")) or any(bare.endswith(x) for x in JA_IMPERATIVE_SUFFIXES)
        if language=="zh":
            return (normalized.endswith(("!","！")) or normalized.startswith(ZH_IMPERATIVE_PREFIXES)
                    or normalized.rstrip("。.!！?？").endswith(ZH_IMPERATIVE_SUFFIXES))
        return False

    def _emit_root(self,speaker,language,text,tick,attributed_source):
        speaker=int(speaker);tick=int(tick);attributed_source=int(attributed_source or speaker)
        span=self._quote_span(text)
        if span is not None:
            _start,_end,inner=span
            concept=self._resolve_concept(language,inner)
            if concept<=0:raise RuntimeError("surface-pragmatics:ungrounded-quote")
            # Recursive decomposition belongs to one admitted external occurrence.  Internal
            # structure depth must not fabricate future organism time and invalidate a peer
            # speaker's chronologically later-but-overlapping contact.
            self.inducer.begin(speaker,K_ASSERT,tick)
            self.inducer.begin(speaker,K_QUOTE,tick,embedded_source=attributed_source)
            if self._is_imperative(language,inner,concept):
                self.inducer.begin(speaker,K_IMPERATIVE,tick)
                self.inducer.emit(speaker,concept,tick)
                self.inducer.end(speaker,tick)
                force="quoted_imperative"
            else:
                self.inducer.begin(speaker,K_ASSERT,tick)
                self.inducer.emit(speaker,concept,tick)
                self.inducer.end(speaker,tick)
                force="quoted_assertion"
            self.inducer.end(speaker,tick)
            root=self.inducer.end(speaker,tick)
            return SurfaceEmissionV1(root,concept,force,language,speaker,tick)

        concept=self._resolve_concept(language,text)
        if concept<=0:raise RuntimeError("surface-pragmatics:ungrounded-surface")
        kind=K_IMPERATIVE if self._is_imperative(language,text,concept) else K_ASSERT
        self.inducer.begin(speaker,kind,tick)
        self.inducer.emit(speaker,concept,tick)
        root=self.inducer.end(speaker,tick)
        force="imperative" if kind==K_IMPERATIVE else "assertion"
        return SurfaceEmissionV1(root,concept,force,language,speaker,tick)

    def feed(self,speaker,language,chunk,tick,final=True,attributed_source=0):
        """Consume one speaker-local UTF-8 chunk; interruptions cannot consume another buffer."""
        speaker=int(speaker);language=str(language);tick=int(tick)
        if speaker<=0 or language not in LANGUAGES:raise ValueError("surface-pragmatics:feed")
        previous=self._pending.get(speaker)
        if previous is not None and previous[0]!=language:raise RuntimeError("surface-pragmatics:language-switch-open")
        text=(previous[1] if previous else "")+str(chunk)
        if len(text)>MAX_PENDING_CHARS:raise RuntimeError("surface-pragmatics:pending-capacity")
        if not final:
            self._pending[speaker]=(language,text,int(attributed_source or (previous[2] if previous else 0)))
            return None
        self._pending.pop(speaker,None)
        return self._emit_root(speaker,language,text,tick,attributed_source or (previous[2] if previous else 0))

    def pending_speakers(self):
        return tuple(sorted(self._pending))

    def checkpoint(self):
        if self._pending:raise RuntimeError("surface-pragmatics:open-utterances")
        return {"schema":1,"inducer":self.inducer.checkpoint()}
