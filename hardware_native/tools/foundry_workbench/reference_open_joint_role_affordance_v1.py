#!/usr/bin/env python3
"""Learn speaker-vs-other action roles from raw speech -> observed actor chronology.

The learner receives no pronoun labels, language ids, role strings, or expected actions.  A raw
utterance is attributed only after a later observed action names both the prior speaker and the
physical actor.  `speaker==actor` and `speaker!=actor` are the only role evidence.  Structural
surface/action learning is delegated to the same open affordance learner used by ordinary
instruction, with a composite role/action identity that is decoded only at this boundary.
"""
from __future__ import annotations
import hashlib,json
from reference_open_language_action_affordance_v1 import OpenLanguageActionAffordanceV1,MIN_TRANSFER_BYTES

ROLE_SPEAKER=1
ROLE_OTHER=2
ROLE_SHIFT=32
ROLE_MASK=(1<<ROLE_SHIFT)-1
ROLE_SPEAKER_ATOM=0xA11CE001
ROLE_OTHER_ATOM=0xA11CE002

def _ack_identity(cohort):
    raw=json.dumps(['joint-ack-role-context-v1',list(map(int,cohort))],sort_keys=True,separators=(',',':')).encode()
    return int(hashlib.sha256(raw).hexdigest()[:16],16) or 1


def _composite(role,action):
    role=int(role);action=int(action)
    if role not in (ROLE_SPEAKER,ROLE_OTHER) or action<=0 or action>ROLE_MASK:return 0
    return (role<<ROLE_SHIFT)|action


def _decode(value):
    value=int(value);role=value>>ROLE_SHIFT;action=value&ROLE_MASK
    return (role,action) if role in (ROLE_SPEAKER,ROLE_OTHER) and action>0 else (0,0)


class OpenJointRoleAffordanceV1:
    """Role/action construction learner with bounded structural state and no language router."""
    def __init__(self):self._inner=OpenLanguageActionAffordanceV1();self._raw_pending={};self.last_settled=();self._ack_contexts={};self._ack_surfaces={}

    def observe_language(self,raw,speaker,tick):
        speaker=int(speaker);tick=int(tick);raw=tuple(map(int,raw))
        ok=self._inner.observe_language(raw,speaker,tick)
        if ok:self._raw_pending[speaker]=(tick,raw)
        return ok

    def observe_actor_action(self,speaker,actor,action,tick):
        speaker=int(speaker);actor=int(actor);action=int(action);tick=int(tick)
        if min(speaker,actor,action)<=0:return False
        role=ROLE_SPEAKER if speaker==actor else ROLE_OTHER
        pending=self._raw_pending.pop(speaker,None)
        ok=self._inner.observe_action(_composite(role,action),speaker,tick)
        raw=() if pending is None else tuple(pending[1])
        target=_composite(role,action)
        factors=(tuple(map(int,self._inner.supporting_factors(raw,target))) if ok and raw and target else ())
        self.last_settled=(role,action,raw,factors) if ok else ()
        if ok and role==ROLE_SPEAKER and raw:
            sketch=self._inner._sketch(raw);target=_composite(ROLE_SPEAKER,action)
            cohort=tuple(sorted({int(example['source']) for example in self._inner._examples
                                 if int(example['action'])==int(target) and tuple(example['sketch'])==tuple(sketch)}))
            if len(cohort)>=2:
                ack_id=_ack_identity(cohort)
                row=self._ack_contexts.setdefault((int(action),int(ack_id)),{'sources':set(map(int,cohort)),'sketch':sketch})
                row['sources'].update(map(int,cohort))
                surf=self._ack_surfaces.setdefault((int(ack_id),int(action),tuple(raw)),set());surf.update(map(int,cohort))
                self.last_settled=(role,action,raw,(int(ack_id),))
        return ok

    def _left_boundary_supported(self,raw,composite):
        current=self._inner._sketch(tuple(map(int,raw)));by_key={}
        for member in current:
            if len(member)==3 and member[0]=='P' and int(member[1])>=MIN_TRANSFER_BYTES:
                by_key.setdefault(tuple(member[:3]),[]).append((0,'P'))
            elif len(member)==5 and member[0]=='I' and int(member[1])>=MIN_TRANSFER_BYTES:
                by_key.setdefault(tuple(member[:3]),[]).append((int(member[3]),'I'))
        for example in self._inner._examples:
            if int(example['action'])!=int(composite):continue
            for member in example['sketch']:
                key=tuple(member[:3])
                for left,kind in by_key.get(key,()):
                    if len(member)==3 and member[0]=='P':return True
                    if len(member)==5 and member[0]=='I' and left<=int(member[3]):return True
        return False

    def candidates(self,raw):
        rows=[]
        for composite,factors in self._inner.candidates(raw):
            role,action=_decode(composite)
            if role and action and self._left_boundary_supported(raw,composite):
                rows.append((role,action,tuple(map(int,factors))))
        if rows:return tuple(rows)
        # Joint-role variants may append newly acquired cooperative material after an
        # already consequence-grounded role/action constituent. Preserve the learned
        # left boundary (so an unseen outer report/quote cannot inherit force), while
        # allowing only right-flank growth around a >= transfer-width interior anchor.
        current=self._inner._sketch(tuple(map(int,raw)));internals={}
        for member in current:
            if len(member)==5 and member[0]=='I':internals.setdefault(tuple(member[:3]),[]).append((int(member[3]),int(member[4])))
        scored=[]
        for factor in self._inner._factors:
            if len(set(map(int,factor.sources)))<2:continue
            role,action=_decode(factor.action_identity)
            if not role or not action:continue
            hits=[]
            for member in factor.members:
                if len(member)!=5 or member[0]!='I' or int(member[1])<MIN_TRANSFER_BYTES:continue
                key=tuple(member[:3]);max_left,max_right=int(member[3]),int(member[4])
                if any(left<=max_left and right>max_right for left,right in internals.get(key,())):
                    hits.append(int(member[1]))
            if hits:scored.append((max(hits),len(hits),len(set(map(int,factor.sources))),int(role),int(action),int(factor.identity)))
        if not scored:return ()
        peak=max(row[:3] for row in scored);semantic={(role,action) for span,hits,sources,role,action,_factor in scored if (span,hits,sources)==peak}
        if len(semantic)!=1:return ()
        role,action=next(iter(semantic));factors=tuple(sorted(factor for span,hits,sources,r,a,factor in scored if (span,hits,sources)==peak and (r,a)==(role,action)))
        return ((int(role),int(action),factors),)

    def factor_sources(self,factor):
        factor=int(factor);row=next((x for x in self._inner._factors if int(x.identity)==factor),None)
        return () if row is None else tuple(map(int,row.sources))

    def role_action_sources(self,raw,role,action):
        # Recover the strongest developmental source cohort for this held-out role/action form.
        current=self._inner._sketch(tuple(map(int,raw)));target=_composite(role,action);rows=[]
        if not current or not target:return ()
        for example in self._inner._examples:
            if int(example['action'])!=int(target):continue
            family=self._inner._common_family(tuple(example['sketch']),current)
            if not family:continue
            span=max((int(x[1]) for x in family),default=0);edge=len({x[0] for x in family if x[0] in ('P','S')})
            rows.append((span,edge,len(family),int(example['source'])))
        if not rows:return ()
        peak=max(x[:3] for x in rows)
        return tuple(sorted({source for span,edge,width,source in rows if (span,edge,width)==peak}))

    def acknowledgement_factor(self,raw,action,source=0):
        # Proposal structure establishes OTHER-role/action; shared developmental speakers link
        # that role alternant to SPEAKER-role acknowledgement without a language/register label.
        current=self._inner._sketch(tuple(map(int,raw)));action=int(action);source=int(source)
        cohort=set(self.role_action_sources(raw,ROLE_OTHER,action))
        if not current:return 0
        current_keys={tuple(x[:3]) for x in current};rows=[]
        for (row_action,factor),row in self._ack_contexts.items():
            if int(row_action)!=action or len(row['sources'])<2:continue
            overlap=len(cohort&set(map(int,row['sources'])))
            partner=int(source in set(map(int,row['sources']))) if source>0 else 0
            if not overlap and not partner:continue
            common=current_keys&{tuple(x[:3]) for x in row['sketch']}
            structural=max((int(x[1]) for x in common),default=0)
            rows.append((partner,overlap,structural,len(common),len(row['sources']),int(factor)))
        if not rows:return 0
        best=max(x[:5] for x in rows);leaders=[factor for partner,overlap,structural,hits,sources,factor in rows if (partner,overlap,structural,hits,sources)==best]
        return leaders[0] if len(leaders)==1 else 0

    def acknowledgement_surface(self,context,action,source=0):
        context=int(context);action=int(action);source=int(source);rows=[]
        for (ctx,act,raw),sources in self._ack_surfaces.items():
            if int(ctx)!=context or int(act)!=action or len(sources)<2:continue
            partner=int(source in sources) if source>0 else 0
            rows.append((partner,len(sources),tuple(raw)))
        if not rows:return ()
        peak=max((p,n) for p,n,_ in rows);leaders={raw for p,n,raw in rows if (p,n)==peak}
        return tuple(next(iter(leaders))) if len(leaders)==1 else ()

    def paired_speaker_factor(self,factor,action):
        # Pair role alternants from shared developmental provenance, never a language id.
        factor=int(factor);action=int(action);target=next((x for x in self._inner._factors if int(x.identity)==factor),None)
        if target is None:return 0
        target_sources=set(map(int,target.sources));target_members={tuple(x[:3]) for x in target.members}
        rows=[]
        for row in self._inner._factors:
            role,row_action=_decode(row.action_identity)
            if role!=ROLE_SPEAKER or row_action!=action:continue
            overlap=len(target_sources&set(map(int,row.sources)))
            shared=max((int(x[1]) for x in row.members if tuple(x[:3]) in target_members),default=0)
            if overlap:rows.append((overlap,shared,int(row.identity)))
        if not rows:return 0
        peak=max((a,b) for a,b,_ in rows);leaders=[identity for a,b,identity in rows if (a,b)==peak]
        return leaders[0] if len(leaders)==1 else 0

    @property
    def factor_count(self):return self._inner.factor_count

    def checkpoint(self):return {'schema':1,'inner':self._inner.checkpoint(),'ack_contexts':[{'action':a,'factor':f,'sources':sorted(row['sources']),'sketch':[list(x) for x in row['sketch']]} for (a,f),row in sorted(self._ack_contexts.items())],'ack_surfaces':[{'context':c,'action':a,'raw':list(raw),'sources':sorted(src)} for (c,a,raw),src in sorted(self._ack_surfaces.items())]}

    @classmethod
    def restore(cls,data):
        if int(data.get('schema',0))!=1:raise RuntimeError('joint-role-affordance:checkpoint-schema')
        out=cls();out._inner=OpenLanguageActionAffordanceV1.restore(data.get('inner',{}))
        for row in data.get('ack_contexts',()):
            key=(int(row['action']),int(row['factor']));out._ack_contexts[key]={'sources':set(map(int,row.get('sources',()))),'sketch':tuple(tuple(x) for x in row.get('sketch',()))}
        for row in data.get('ack_surfaces',()):
            out._ack_surfaces[(int(row['context']),int(row['action']),tuple(map(int,row['raw'])))]=set(map(int,row.get('sources',())))
        return out
