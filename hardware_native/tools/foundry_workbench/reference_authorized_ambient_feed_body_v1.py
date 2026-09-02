#!/usr/bin/env python3
"""Authorized external-feed body for semantically-blind ambient Life contact.

Provider acquisition is deliberately outside cognition.  This membrane accepts only an already-
authorized candidate pool of opaque ``(source, raw_bytes)`` contacts, chooses one index from entropy,
Life sequence and pool size, then records the selected bytes through ordinary ambient Life events.
It owns no provider parser, topic filter, toxicity score, language label, trust bit or action policy.
"""
from __future__ import annotations

import hashlib
import secrets

from reference_life_function_curriculum_v1 import LifeCurriculumEventV2
from reference_persistent_ambient_language_stream_v1 import (
    PersistentAmbientLanguageStreamV1,semantically_blind_index,
)

MAX_AUTHORIZED_AMBIENT_POOL=512
MAX_AUTHORIZED_AMBIENT_POOL_BYTES=8*1024*1024


def _candidate_pool(pool):
    rows=[];total_bytes=0
    for source,raw in tuple(pool):
        source=int(source);raw=PersistentAmbientLanguageStreamV1._raw(raw)
        if source<=0:raise ValueError('ambient-feed-body:source')
        rows.append((source,raw));total_bytes+=len(raw)
        if len(rows)>MAX_AUTHORIZED_AMBIENT_POOL:raise RuntimeError('ambient-feed-body:pool-capacity')
        if total_bytes>MAX_AUTHORIZED_AMBIENT_POOL_BYTES:raise RuntimeError('ambient-feed-body:pool-byte-capacity')
    if not rows:raise ValueError('ambient-feed-body:pool')
    return tuple(rows)


def _pool_commitment(rows):
    """Bind the exact ordered candidate pool without retaining it in organism state."""
    digest=hashlib.sha256()
    for source,raw in rows:
        source_bytes=str(int(source)).encode('ascii');payload=bytes(raw)
        digest.update(len(source_bytes).to_bytes(4,'big'));digest.update(source_bytes)
        digest.update(len(payload).to_bytes(4,'big'));digest.update(payload)
    return digest.hexdigest()


class AuthorizedAmbientFeedBodyV1:
    """Transport-only sampler. Calling cadence belongs to the external body/world loop."""

    @staticmethod
    def choose_index(entropy,life_sequence,pool_count):
        return semantically_blind_index(int(entropy),int(life_sequence),int(pool_count))

    @classmethod
    def admit_one(cls,runtime,tick,pool,entropy=None):
        rows=_candidate_pool(pool);tick=int(tick);life_sequence=int(runtime.cursor)+1
        if tick<0:raise ValueError('ambient-feed-body:tick')
        entropy=secrets.randbits(128) if entropy is None else int(entropy)
        index=cls.choose_index(entropy,life_sequence,len(rows));source,raw=rows[index]
        event=LifeCurriculumEventV2(life_sequence,'ambient_social_post',source,(tick,*raw))
        runtime.apply(event)
        return {
            'index':int(index),'source':int(source),'bytes':len(raw),
            'sha256':hashlib.sha256(bytes(raw)).hexdigest(),
            'life_sequence':life_sequence,'tick':tick,'entropy':entropy,
            'pool_count':len(rows),'pool_sha256':_pool_commitment(rows),
        }

    @staticmethod
    def drain(runtime,tick):
        tick=int(tick)
        event=LifeCurriculumEventV2(int(runtime.cursor)+1,'ambient_social_drain',0,(tick,))
        return runtime.apply(event)

    @classmethod
    def pump(cls,runtime,tick,pool,entropy=None):
        receipt=cls.admit_one(runtime,tick,pool,entropy)
        processed=cls.drain(runtime,tick)
        return receipt,processed
