#!/usr/bin/env python3
"""Transient Networks-of-Networks closure for source epistemics × social allostasis.

No learned state lives here. Recursive/social provenance, testimony authority, developmental
body history, controllability and language/action credit remain owned by their incumbent
mechanisms. This closure only rematerializes their current interaction.
"""
from __future__ import annotations
from dataclasses import dataclass
from reference_hierarchical_composition_v1 import _identity

Q=1<<16


def _clip(value):return max(0,min(Q,int(value)))
def _band(value):
    value=_clip(value)
    return 0 if value<Q//4 else (1 if value<Q//2 else 2)


@dataclass(frozen=True)
class SourceQualifiedSocialClosureV1:
    merge_roots:tuple[int,...]
    target:int
    epistemic_state:int
    credible_positive:int
    credible_negative:int
    recipient_source:int
    boundary_drive_q16:int
    interference_q16:int
    arousal_q16:int
    effective_control_q16:int

    @property
    def credible_conflict(self):return bool(self.credible_positive and self.credible_negative)

    @property
    def response_context(self):
        # Opaque composition coordinate. No source identity, word, moral label or answer text
        # becomes the context; current source epistemics and body geometry do.
        return _identity('source-qualified-social-allostatic-closure-v1',(
            int(self.epistemic_state),int(self.credible_conflict),
            _band(self.boundary_drive_q16),_band(self.interference_q16)))


class SourceQualifiedSocialAllostaticClosureV1:
    """Stateless current closure over existing resident owners."""

    @staticmethod
    def evaluate(world,binding,allostasis,merge_roots,target,action,tick,acute_arousal_q16=0):
        roots=(int(merge_roots),) if isinstance(merge_roots,int) else tuple(map(int,merge_roots))
        roots=tuple(sorted(set(roots)));target=int(target);action=int(action);tick=int(tick)
        if not roots or min(*roots,target,action)<=0 or tick<0:raise ValueError('social-allostatic-closure:coordinate')
        # A live room may contain several structurally distinct propositions. Preserve
        # those roots and union only their authenticated source provenance; opposing
        # claims must not be collapsed into one fake proposition identity.
        merged={}
        for root in roots:
            for source,count,last_tick in binding.sources_for(root):
                prior=merged.get(int(source),(0,-1))
                merged[int(source)]=(int(prior[0])+int(count),max(int(prior[1]),int(last_tick)))
        provenance=tuple((source,count,last_tick) for source,(count,last_tick) in sorted(merged.items()))
        if not provenance:
            return SourceQualifiedSocialClosureV1(roots,target,
                int(world.testimony_reliability_state(target)),0,0,0,0,0,0,0)

        acute_by_source=(acute_arousal_q16 if isinstance(acute_arousal_q16,dict) else None)
        positive=negative=0;appraisals=[]
        for source,_count,_last_tick in provenance:
            source=int(source)
            claim=world.reputation_claims.get((source,target))
            if claim is not None and world.reputation_reliable(source):
                if bool(claim):positive+=1
                else:negative+=1
            acute=(acute_by_source.get(source,0) if acute_by_source is not None else acute_arousal_q16)
            appraisal=allostasis.appraise(source,action,tick,acute)
            appraisals.append((source,appraisal))

        # Recipient attention is social/body arbitration, not testimony authority. A source
        # can demand boundary attention through lived adverse history without gaining a vote.
        recipient,appraisal=max(appraisals,key=lambda row:(
            int(row[1]['boundary_drive_q16']),int(row[1]['arousal_q16']),-int(row[0])))
        return SourceQualifiedSocialClosureV1(
            roots,target,int(world.testimony_reliability_state(target)),
            positive,negative,int(recipient),
            _clip(appraisal['boundary_drive_q16']),_clip(appraisal['interference_q16']),
            _clip(appraisal['arousal_q16']),_clip(appraisal['effective_control_q16']))
