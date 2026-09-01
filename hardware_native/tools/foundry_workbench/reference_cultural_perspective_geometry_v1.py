#!/usr/bin/env python3
"""Source-qualified cultural perspective geometry over already-formed recursive Merge roots.

This owner assimilates curriculum exposure without turning quoted norms, repeated rhetoric,
source identity, or somatic state into truth/action authority. Durable state is bounded
(context_root, proposition_root, speaker) attestation geometry only.
"""
from __future__ import annotations

Q=1<<16
MAX_ATTESTATIONS=8192
MAX_SOURCES_PER_PROPOSITION=256


def _clip(x,lo=0,hi=Q):
    return max(lo,min(hi,int(x)))


class CulturalPerspectiveGeometryV1:
    def __init__(self):
        # (context_root, proposition_root, speaker) -> [exposures,last_tick]
        self._rows={}

    def observe(self,context_root,proposition_root,speaker,tick):
        context_root=int(context_root);proposition_root=int(proposition_root)
        speaker=int(speaker);tick=int(tick)
        if min(context_root,proposition_root,speaker)<=0 or tick<0:
            raise ValueError('cultural-perspective:observation')
        key=(context_root,proposition_root,speaker)
        row=self._rows.get(key)
        if row is None:
            if len(self._rows)>=MAX_ATTESTATIONS:
                # Retention is chronology-only; no trust, valence, or moral score may choose victims.
                victim=min(self._rows,key=lambda k:(self._rows[k][1],k[0],k[1],k[2]))
                del self._rows[victim]
            self._rows[key]=[1,tick]
        else:
            if tick<int(row[1]):raise ValueError('cultural-perspective:time-reversal')
            row[0]=min(0x7fffffff,int(row[0])+1);row[1]=tick
        return True

    def retract(self,context_root,proposition_root,speaker):
        """A source can revise its own attestation; this never erases other perspectives."""
        return self._rows.pop((int(context_root),int(proposition_root),int(speaker)),None) is not None

    def propositions(self,context_root):
        """Return source-separated curriculum geometry; exposure count is familiarity, not authority."""
        context_root=int(context_root);grouped={}
        for (ctx,prop,speaker),(count,last_tick) in self._rows.items():
            if ctx!=context_root:continue
            grouped.setdefault(prop,[]).append((speaker,int(count),int(last_tick)))
        return tuple((prop,tuple(sorted(rows))) for prop,rows in sorted(grouped.items()))

    def project(self,context_root,source_epistemic_q16=None,somatic_bias_q16=None):
        """Interface Merge to external epistemics/body state without collapsing their channels.

        Each source contributes at most once to epistemic support regardless of repetition.
        Somatic bias is an external transient coordinate keyed by proposition root; it is reported
        separately and never modifies durable geometry or epistemic support.
        """
        source_epistemic_q16={} if source_epistemic_q16 is None else dict(source_epistemic_q16)
        somatic_bias_q16={} if somatic_bias_q16 is None else dict(somatic_bias_q16)
        out=[]
        for proposition,sources in self.propositions(context_root):
            if len(sources)>MAX_SOURCES_PER_PROPOSITION:
                sources=sources[:MAX_SOURCES_PER_PROPOSITION]
            weights=[_clip(source_epistemic_q16.get(speaker,0)) for speaker,_count,_tick in sources]
            epistemic=min(Q,sum(weights)//max(1,len(weights))) if weights else 0
            familiarity=sum(count for _speaker,count,_tick in sources)
            out.append({
                'proposition_root':int(proposition),
                'sources':tuple(int(s) for s,_c,_t in sources),
                'source_count':len(sources),
                'familiarity':int(familiarity),
                'epistemic_q16':int(epistemic),
                'somatic_q16':_clip(somatic_bias_q16.get(int(proposition),0)),
            })
        return tuple(sorted(out,key=lambda r:r['proposition_root']))

    def checkpoint(self):
        return {
            'schema':1,
            'rows':[[int(c),int(p),int(s),int(v[0]),int(v[1])]
                    for (c,p,s),v in sorted(self._rows.items())],
        }

    @classmethod
    def restore(cls,data):
        if int(data.get('schema',0))!=1:raise RuntimeError('cultural-perspective:checkpoint-schema')
        out=cls()
        for row in data.get('rows',()):
            if len(row)!=5:raise RuntimeError('cultural-perspective:checkpoint-row')
            context,prop,speaker,count,tick=map(int,row)
            if min(context,prop,speaker,count)<=0 or tick<0:
                raise RuntimeError('cultural-perspective:checkpoint-row')
            key=(context,prop,speaker)
            if key in out._rows:raise RuntimeError('cultural-perspective:checkpoint-duplicate')
            out._rows[key]=[count,tick]
        if len(out._rows)>MAX_ATTESTATIONS:raise RuntimeError('cultural-perspective:checkpoint-capacity')
        return out
