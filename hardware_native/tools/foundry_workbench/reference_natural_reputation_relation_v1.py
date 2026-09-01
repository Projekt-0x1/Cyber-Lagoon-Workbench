#!/usr/bin/env python3
"""Learn language-relation polarity from direct source life, then bind new reputation claims.

The relation owner never parses trust words and never receives a reliability polarity at
inference.  It only consumes incumbent learned surface/span bindings plus direct lived
source calibration and hands provisional claims to OpenWorldCausalLearningV1.
"""
from __future__ import annotations

MAX_REPUTATION_LANGUAGE_RELATIONS=32
MAX_REPUTATION_RELATION_EXAMPLES=128
MAX_REPUTATION_BINDING_DEPTH=3
MAX_REPUTATION_FRAGMENT_STREAMS=32
MAX_REPUTATION_FRAGMENT_BYTES=128
REPUTATION_FRAGMENT_HORIZON=16
MIN_RELATION_TARGETS=2


class NaturalReputationRelationV1:
    def __init__(self):
        # context -> target -> direct sign {-1,+1}; raw surfaces are never stored persistently.
        self._examples={}
        # Current incomplete acoustic/surface streams only. Keyed by authenticated
        # speaker + opaque stream occurrence so interjecting speakers cannot splice.
        # This frontier is deliberately omitted from checkpoint state.
        self._fragment_streams={}

    @staticmethod
    def _direct_state(world,target):
        correct,wrong=world.testimony_accuracy.get(int(target),(0,0))
        if correct+wrong<=0:return 0
        return 1 if correct>wrong else (-1 if wrong>correct else 0)

    def relation_polarity(self,context):
        rows=self._examples.get(int(context),{})
        positive=sum(1 for value in rows.values() if int(value)>0)
        negative=sum(1 for value in rows.values() if int(value)<0)
        if positive>=MIN_RELATION_TARGETS and negative==0:return True
        if negative>=MIN_RELATION_TARGETS and positive==0:return False
        return None

    def _prune_fragment_streams(self,tick):
        tick=int(tick);cutoff=tick-REPUTATION_FRAGMENT_HORIZON
        for key,row in tuple(self._fragment_streams.items()):
            if int(row[1])<cutoff:self._fragment_streams.pop(key,None)

    def _append_fragment(self,speaker,stream_identity,raw,tick):
        speaker=int(speaker);stream_identity=int(stream_identity);tick=int(tick);raw=tuple(map(int,raw))
        if min(speaker,stream_identity)<=0 or not raw:return ()
        self._prune_fragment_streams(tick);key=(speaker,stream_identity)
        prior=self._fragment_streams.get(key,((),tick))[0];combined=tuple(prior)+raw
        if len(combined)>MAX_REPUTATION_FRAGMENT_BYTES:
            self._fragment_streams.pop(key,None);return ()
        if key not in self._fragment_streams and len(self._fragment_streams)>=MAX_REPUTATION_FRAGMENT_STREAMS:
            oldest=min(self._fragment_streams,key=lambda k:(int(self._fragment_streams[k][1]),k))
            self._fragment_streams.pop(oldest,None)
        self._fragment_streams[key]=(combined,tick);return combined

    def pending_fragment_streams(self,tick=0):
        if int(tick)>0:self._prune_fragment_streams(tick)
        return tuple((speaker,stream,len(raw),last_tick)
                     for (speaker,stream),(raw,last_tick) in sorted(self._fragment_streams.items()))

    def close_fragmented_surface(self,raw,speaker,stream_identity,tick,final=False):
        """Close one source-local raw stream without assigning construction semantics."""
        combined=self._append_fragment(speaker,stream_identity,raw,tick)
        if not combined:return ()
        if not final:return ()
        self._fragment_streams.pop((int(speaker),int(stream_identity)),None)
        return combined

    def observe_fragmented_example(self,adult,context,target,raw,speaker,stream_identity,tick,final=False):
        """Acquire one construction/relation example from source-local interrupted surface contact."""
        combined=self._append_fragment(speaker,stream_identity,raw,tick)
        if not combined:return (False,False)
        key=(int(speaker),int(stream_identity))
        if not final:return (False,False)
        self._fragment_streams.pop(key,None)
        context=int(context);target=int(target)
        learned=bool(adult.observe_surface_construction(context,(target,),combined,int(speaker)))
        direct=self._direct_state(adult.world_causal_learning,target)
        calibrated=bool(learned and direct and self._admit_direct_example(context,target,direct))
        return learned,calibrated

    def observe_fragmented_claim(self,adult,raw,speaker,stream_identity,tick,final=False):
        """Resolve one interrupted mature claim only when accumulated learned structure is unique."""
        combined=self._append_fragment(speaker,stream_identity,raw,tick)
        if not combined:return ((),False)
        key=(int(speaker),int(stream_identity));binding=self.natural_claim_binding(adult,combined)
        if binding:
            self._fragment_streams.pop(key,None)
            _context,target,polarity=binding
            if int(speaker)==int(target):return ((),False)
            claimed=bool(adult.world_causal_learning.observe_reputation_claim(int(speaker),int(target),bool(polarity)))
            return (binding,claimed)
        if final:self._fragment_streams.pop(key,None)
        return ((),False)

    def _all_bindings(self,adult,raw,depth=0):
        raw=tuple(map(int,raw));out=[]
        try:
            for row in adult.language.invert_surface(raw,max_candidates=16):
                if len(row.atoms)==1:
                    out.append((int(row.context),int(row.atoms[0]),depth))
        except Exception:
            pass
        if depth>=MAX_REPUTATION_BINDING_DEPTH:return tuple(out)
        try:spans=adult.language.invert_span(raw,max_candidates=16)
        except Exception:spans=()
        # A generic no-literal split factor can expose arbitrary substrings and must
        # not become semantic embedding authority. Recurse only through promoted
        # span factors that carry learned literal boundary structure. This remains
        # construction-neutral: no reputation word, wrapper context, or port is named.
        structured=[];template_counts={}
        for span in spans:
            template_counts[span.template_identity]=template_counts.get(span.template_identity,0)+1
        for span in spans:
            # An acquired structural template must select one decomposition of this
            # occurrence. A template that can split the same bytes many ways is
            # unresolved syntax, not authority to expose whichever substring helps.
            if template_counts.get(span.template_identity,0)!=1:continue
            try:factor=int(adult.language._span_tid(span.template_identity))
            except Exception:continue
            earned=False
            for (context,arity,pieces),sources in adult.language._span_sources.items():
                try:
                    same=int(adult.language.span_factor_identity(context,arity,pieces))==factor
                    active=len(adult.language._active_sources(sources))
                except Exception:
                    continue
                if (same and active>=adult.language.minimum_source_support and
                        any(getattr(piece,'kind',0)==1 and tuple(getattr(piece,'literal',())) for piece in pieces)):
                    earned=True;break
            if earned:structured.append(span)
        for span in structured:
            for child in span.children:
                out.extend(self._all_bindings(adult,child,depth+1))
        # Multiple structural paths to the same learned relation/target are one candidate.
        return tuple(sorted(set(out),key=lambda row:(row[2],row[0],row[1])))

    def _admit_direct_example(self,context,target,direct):
        context=int(context);target=int(target);direct=int(direct)
        if min(context,target)<=0 or direct not in (-1,1):return False
        # Once multiple lived exemplars have established a relation's polarity,
        # later true/false uses are testimony about speakers, not new evidence
        # about what the learned relation means.
        if self.relation_polarity(context) is not None:return False
        if context not in self._examples and len(self._examples)>=MAX_REPUTATION_LANGUAGE_RELATIONS:return False
        rows=self._examples.setdefault(context,{})
        total=sum(len(x) for x in self._examples.values())
        if target not in rows and total>=MAX_REPUTATION_RELATION_EXAMPLES:return False
        prior=rows.get(target)
        if prior is not None and int(prior)!=direct:return False
        rows[target]=direct;return True

    def calibrate_from_direct_life(self,adult,raw):
        """Let directly calibrated targets teach the polarity of one learned relation."""
        candidates=[]
        for context,target,depth in self._all_bindings(adult,raw):
            direct=self._direct_state(adult.world_causal_learning,target)
            if direct:candidates.append((context,target,direct,depth))
        if not candidates:return False
        nearest=min(depth for _context,_target,_direct,depth in candidates)
        unique={(context,target,direct) for context,target,direct,depth in candidates if depth==nearest}
        if len(unique)!=1:return False
        return self._admit_direct_example(*next(iter(unique)))

    def natural_claim_binding(self,adult,raw):
        """Return the unique learned reputation relation/target justified by structure."""
        candidates=[]
        for context,target,depth in self._all_bindings(adult,raw):
            polarity=self.relation_polarity(context)
            if polarity is not None:candidates.append((context,target,bool(polarity),depth))
        semantic={(context,target,polarity) for context,target,polarity,_depth in candidates}
        return () if len(semantic)!=1 else tuple(next(iter(semantic)))

    def observe_natural_claim(self,adult,raw,speaker):
        """Bind raw/nested learned structure to one provisional source-qualified claim."""
        speaker=int(speaker)
        if speaker<=0:return False
        binding=self.natural_claim_binding(adult,raw)
        if not binding:return False
        _context,target,polarity=binding
        if speaker==target:return False
        return bool(adult.world_causal_learning.observe_reputation_claim(speaker,target,polarity))

    def checkpoint(self):
        return {'schema':1,'relations':[
            {'context':context,'examples':[{'target':target,'direct':direct} for target,direct in sorted(rows.items())]}
            for context,rows in sorted(self._examples.items())]}

    @classmethod
    def restore(cls,data):
        if int(data.get('schema',0))!=1:raise RuntimeError('natural-reputation:checkpoint-schema')
        out=cls();seen_targets=0
        for relation in data.get('relations',()):
            context=int(relation.get('context',0))
            if context<=0 or context in out._examples:raise RuntimeError('natural-reputation:checkpoint-relation')
            rows={}
            for row in relation.get('examples',()):
                target=int(row.get('target',0));direct=int(row.get('direct',0))
                if target<=0 or direct not in (-1,1) or target in rows:raise RuntimeError('natural-reputation:checkpoint-example')
                rows[target]=direct;seen_targets+=1
            if not rows:raise RuntimeError('natural-reputation:checkpoint-empty')
            out._examples[context]=rows
        if len(out._examples)>MAX_REPUTATION_LANGUAGE_RELATIONS or seen_targets>MAX_REPUTATION_RELATION_EXAMPLES:
            raise RuntimeError('natural-reputation:checkpoint-capacity')
        return out
