#!/usr/bin/env python3
"""Learn zero-authority action affordances from raw language -> observed action chronology.

No language id, translation table, imperative/quote category, expected action, or response policy
exists here.  A source-local raw linguistic occurrence may become evidence for an action only when
that same authenticated actor subsequently performs the action inside the short causal horizon.
Settled hashed surface families can later nominate actions into the ordinary resident action bank;
all motor authority remains in independently lived transition/consequence evidence.
"""
from __future__ import annotations
from dataclasses import dataclass
import hashlib,json

MAX_PENDING_SOURCES=128
MAX_EXAMPLES=1024
MAX_FACTORS=512
ACTION_HORIZON=8
MIN_FACTOR_BYTES=5
MAX_FACTOR_BYTES=24
MIN_TRANSFER_BYTES=9
MAX_FACTOR_MEMBERS=96


def _hash(tag,units):
    return hashlib.sha256(tag.encode()+b'\0'+bytes(units)).digest()[:12].hex()


def _identity(tag,values):
    raw=json.dumps([tag,*values],sort_keys=True,separators=(',',':')).encode()
    return int(hashlib.sha256(raw).hexdigest()[:16],16) or 1


@dataclass(frozen=True)
class OpenLanguageActionFactorV1:
    identity:int
    action_identity:int
    members:tuple
    sources:tuple[int,...]


class OpenLanguageActionAffordanceV1:
    """Generic cross-situational raw-surface/action learner."""

    def __init__(self):
        # Pending raw speech is transient occurrence state and never checkpointed.
        self._pending={}
        # Action-linked developmental witnesses persist only as hashed structural sketches.
        self._examples=[]
        self._factors=[]
        # Rebuildable execution cache. Only new developmental examples advance this
        # revision; the cache is never serialized and carries no action authority.
        self._examples_revision=0
        self._structural_inventory_cache=None
        self.last_match_touches=0
        self.last_candidate_mode='none'

    @staticmethod
    def _sketch(raw):
        raw=tuple(map(int,raw))
        if not raw:return ()
        rows=[]
        # Beginning-sensitive evidence prevents an unseen quotation/wrapper from inheriting
        # an interior actionable phrase merely because the bytes occur somewhere inside it.
        upper=min(MAX_FACTOR_BYTES,len(raw))
        for n in range(MIN_FACTOR_BYTES,upper+1):
            rows.append(('P',n,_hash('prefix',raw[:n])))
        # A bounded right edge helps distinguish otherwise equal left prefixes when learned.
        for n in range(MIN_FACTOR_BYTES,min(12,len(raw))+1):
            rows.append(('S',n,_hash('suffix',raw[-n:])))
        # Interior constituents carry their flank geometry. This is not a parser label:
        # it is a raw structural receipt that allows a stable action constituent to
        # survive typological movement while bounding how much unseen material may wrap it.
        # Keep short-to-medium constituent spans so predicate/action evidence can be
        # discovered without assuming whitespace, script, morphology or a language-specific
        # tokenizer. UTF-8 byte spans are observation matter; their learned recurrence and
        # consequence provenance, not byte identity itself, determines structural use.
        interior_widths=range(MIN_FACTOR_BYTES,upper+1)
        for n in interior_widths:
            for start in range(0,len(raw)-n+1):
                rows.append(('I',n,_hash('interior',raw[start:start+n]),start,len(raw)-(start+n)))
        return tuple(rows)

    @staticmethod
    def _common_family(left,right):
        edge_left={row for row in left if len(row)==3};edge_right={row for row in right if len(row)==3}
        common=edge_left&edge_right
        prefixes=sorted((row for row in common if row[0]=='P'),key=lambda x:(-x[1],x[2]))
        suffixes=sorted((row for row in common if row[0]=='S'),key=lambda x:(-x[1],x[2]))
        prefix=max((int(row[1]) for row in prefixes),default=0)
        suffix=max((int(row[1]) for row in suffixes),default=0)
        # Typologically neutral edge evidence: an English-like stable opening may
        # carry the factor alone, while German-like variable material may intervene
        # between a shorter lexical onset and a stable right constituent.
        if prefix>=MIN_TRANSFER_BYTES or (prefix>=MIN_FACTOR_BYTES and suffix>=MIN_TRANSFER_BYTES):
            return tuple((prefixes+suffixes)[:MAX_FACTOR_MEMBERS])
        # If both edges vary, discover a shared raw constituent and retain only its
        # learned flank envelope. This is the Merge-like invariant: the constituent
        # survives displacement, while arbitrary new outer embedding is refused.
        li={row[:3]:row[3:] for row in left if len(row)==5 and row[0]=='I'}
        ri={row[:3]:row[3:] for row in right if len(row)==5 and row[0]=='I'}
        keys=set(li)&set(ri)
        longest=max((int(key[1]) for key in keys),default=0)
        if longest<MIN_TRANSFER_BYTES:return ()
        rows=[]
        for key in sorted((key for key in keys if int(key[1])==longest),key=lambda x:x[2]):
            ll,lr=map(int,li[key]);rl,rr=map(int,ri[key])
            rows.append((*key,max(ll,rl,MIN_TRANSFER_BYTES),max(lr,rr,MIN_TRANSFER_BYTES)))
        return tuple(rows[:MAX_FACTOR_MEMBERS])

    def _prune_pending(self,tick):
        cutoff=int(tick)-ACTION_HORIZON
        for source,row in tuple(self._pending.items()):
            if int(row[0])<cutoff:self._pending.pop(source,None)

    def observe_language(self,raw,speaker,tick):
        speaker=int(speaker);tick=int(tick);raw=tuple(map(int,raw))
        if speaker<=0 or tick<0 or not raw:raise ValueError('open-language-action:language')
        self._prune_pending(tick)
        sketch=self._sketch(raw)
        if not sketch:return False
        if speaker not in self._pending and len(self._pending)>=MAX_PENDING_SOURCES:
            victim=min(self._pending,key=lambda s:(self._pending[s][0],s));self._pending.pop(victim,None)
        self._pending[speaker]=(tick,sketch)
        return True

    def _settle(self,action,source,sketch):
        candidates=[]
        for row in self._examples:
            if int(row['action'])!=int(action) or int(row['source'])==int(source):continue
            family=self._common_family(row['sketch'],sketch)
            if family:candidates.append((family,row))
        if not candidates:return False
        def family_key(family):
            edge_kinds={member[0] for member in family if member[0] in ('P','S')}
            span=max((int(member[1]) for member in family),default=0)
            support=sum(1 for member in family if member[0] in ('P','S','I'))
            return (len(edge_kinds),span,support)
        # Do not force developmental hypotheses through one early winner. Keep the
        # strongest boundary topology and the strongest invariant span; later raw
        # contact decides which network actually rematerializes. This is bounded
        # Networks-of-Networks competition, not a language/register classifier.
        edge_peak=max((family_key(family)[0],family_key(family)[2],family_key(family)[1]) for family,_row in candidates)
        span_peak=max((family_key(family)[1],family_key(family)[0],family_key(family)[2]) for family,_row in candidates)
        selected=[]
        for family,row in candidates:
            key=family_key(family)
            if (key[0],key[2],key[1])==edge_peak or (key[1],key[0],key[2])==span_peak:
                selected.append((tuple(family),row))
        families=[]
        for family,_row in selected:
            if family not in families:families.append(family)
        changed=False
        for family in families[:4]:
            sources={int(source)}|{int(row['source']) for fam,row in selected if tuple(fam)==family}
            existing=next((f for f in self._factors if f.action_identity==int(action) and f.members==family),None)
            if existing is not None:
                merged=tuple(sorted(set(existing.sources)|sources));idx=self._factors.index(existing)
                self._factors[idx]=OpenLanguageActionFactorV1(existing.identity,existing.action_identity,existing.members,merged);changed=True;continue
            if len(self._factors)>=MAX_FACTORS:continue
            ident=_identity('open-language-action-factor-v1',[int(action),[list(member) for member in family]])
            self._factors.append(OpenLanguageActionFactorV1(ident,int(action),family,tuple(sorted(sources))));changed=True
        return changed

    def observe_action(self,action_identity,demonstrator,tick):
        """Bind a real observed action to that demonstrator's recent raw speech occurrence."""
        action=int(action_identity);source=int(demonstrator);tick=int(tick)
        if min(action,source)<=0 or tick<0:raise ValueError('open-language-action:action')
        self._prune_pending(tick)
        pending=self._pending.pop(source,None)
        if pending is None:return False
        speech_tick,sketch=pending
        if tick<int(speech_tick) or tick-int(speech_tick)>ACTION_HORIZON:return False
        self._settle(action,source,sketch)
        row={'action':action,'source':source,'tick':tick,'sketch':sketch}
        duplicate=any(x['action']==action and x['source']==source and x['sketch']==sketch for x in self._examples)
        if not duplicate:
            if len(self._examples)>=MAX_EXAMPLES:self._examples.pop(0)
            self._examples.append(row);self._examples_revision+=1;self._structural_inventory_cache=None
        return True

    @staticmethod
    def _sketch_position(row):
        if len(row)!=5:return 'edge'
        left,right=int(row[3]),int(row[4])
        if left==0:return 'left'
        if right==0:return 'right'
        return 'middle'

    def _structural_inventory(self):
        """Induce language-neutral action anchors and productive construction matter."""
        held=self._structural_inventory_cache
        if held is not None and int(held[0])==int(self._examples_revision):return held[1]
        examples=tuple(self._examples)
        key_actions={};key_sources={};key_positions={}
        frame_actions={};frame_sources={}
        for row in examples:
            action=int(row['action']);source=int(row['source']);sketch=tuple(row['sketch'])
            seen=set()
            for member in sketch:
                key=tuple(member[:3])
                if key in seen:continue
                seen.add(key);key_actions.setdefault(key,set()).add(action)
                key_sources.setdefault((key,action),set()).add(source)
                key_positions.setdefault((key,action),set()).add(self._sketch_position(member))
                if member[0] in ('P','S'):
                    frame_actions.setdefault(key,set()).add(action)
                    frame_sources.setdefault(key,set()).add(source)
        anchors={}
        for key,actions in key_actions.items():
            if len(actions)!=1:continue
            action=next(iter(actions))
            if len(key_sources.get((key,action),()))>=2:
                anchors.setdefault(action,set()).add(key)
        # A construction frame is not an action label: it must recur across different
        # actions and independent sources. Keep only sufficiently informative edge matter.
        frames={key for key,actions in frame_actions.items()
                if len(actions)>=2 and len(frame_sources.get(key,()))>=2
                and int(key[1])>=MIN_FACTOR_BYTES}
        result=(anchors,frames,key_positions)
        self._structural_inventory_cache=(int(self._examples_revision),result)
        return result

    def _has_contrastive_rival(self,action):
        """Return true when another action shares substantial non-action structure."""
        action=int(action);own=tuple(row for row in self._examples if int(row['action'])==action)
        rivals=tuple(row for row in self._examples if int(row['action'])!=action)
        for left in own:
            left_keys={tuple(member[:3]) for member in left['sketch'] if member[0]=='I'}
            for right in rivals:
                common=left_keys&{tuple(member[:3]) for member in right['sketch'] if member[0]=='I'}
                if max((int(key[1]) for key in common),default=0)>=max(12,MIN_TRANSFER_BYTES):
                    return True
        return False

    def structural_candidates(self,raw):
        """Compose consequence-earned constituent and construction evidence.

        This path has no language id and no tokenizer. It operates over the same hashed
        prefix/suffix/interior structural matter for whitespace, non-whitespace and mixed
        scripts. One constituent can nominate an action only when it is action-specific
        across independent sources; a construction frame can license transfer only when
        it is shared across different actions. Both are required for held-out force.
        """
        current=self._sketch(tuple(map(int,raw)))
        if not current or len(self._examples)<4:return ()
        anchors,frames,positions=self._structural_inventory()
        if not anchors or not frames:return ()
        current_by_key={}
        for member in current:current_by_key.setdefault(tuple(member[:3]),[]).append(member)
        matched_frames=tuple(key for key in frames if key in current_by_key)
        if not matched_frames:return ()
        # Preserve learned boundary dependencies. If a suffix frame was repeatedly
        # observed together with a productive prefix frame, an outer wrapper that
        # displaces that prefix cannot inherit top-level force from the suffix alone.
        guarded=[]
        for frame in matched_frames:
            if frame[0]!='S':
                guarded.append(frame);continue
            companion_actions={}
            for row in self._examples:
                keys={tuple(member[:3]) for member in row['sketch']}
                if frame not in keys:continue
                for key in keys:
                    if key in frames and key[0]=='P':
                        companion_actions.setdefault(key,set()).add(int(row['action']))
            companions={key for key,actions in companion_actions.items() if len(actions)>=2}
            if companions and not any(key in current_by_key for key in companions):continue
            guarded.append(frame)
        matched_frames=tuple(guarded)
        if not matched_frames:return ()
        prefix_frames=tuple(key for key in matched_frames if key[0]=='P')
        if prefix_frames:matched_frames=prefix_frames
        frame_peak=max(int(key[1]) for key in matched_frames)
        matched_frames=tuple(key for key in matched_frames if int(key[1])==frame_peak)
        scored=[]
        for action,action_anchors in anchors.items():
            compatible=[]
            for key in action_anchors:
                for member in current_by_key.get(key,()):
                    if self._sketch_position(member) in positions.get((key,action),()):
                        compatible.append(key);break
            if not compatible:continue
            # Action specificity and two-source support already exclude generic residue.
            # Do not compare this utterance against the longest byte span ever learned
            # for the same action in another language/register: UTF-8 width and lexical
            # length are surface geometry, not a universal constituent hierarchy.
            anchor_peak=max(int(key[1]) for key in compatible)
            if not prefix_frames:
                if frame_peak<MIN_TRANSFER_BYTES or anchor_peak<MIN_TRANSFER_BYTES:continue
            elif anchor_peak<=MIN_TRANSFER_BYTES and not self._has_contrastive_rival(action):continue
            strongest=tuple(sorted(key for key in compatible if int(key[1])==anchor_peak))
            factor=_identity('open-language-action-structural-factor-v2',[
                int(action),[list(key) for key in matched_frames],[list(key) for key in strongest]])
            scored.append((frame_peak,anchor_peak,len(strongest),int(action),int(factor)))
        if not scored:return ()
        peak=max((fw,aw,hits) for fw,aw,hits,_action,_factor in scored)
        winners={action for fw,aw,hits,action,_factor in scored if (fw,aw,hits)==peak}
        if len(winners)!=1:return ()
        action=next(iter(winners));factors=tuple(sorted(factor for fw,aw,hits,a,factor in scored
                                                        if a==action and (fw,aw,hits)==peak))
        return ((int(action),factors),)

    def _has_productive_structural_frame(self,raw):
        current=self._sketch(tuple(map(int,raw)))
        if not current:return False
        _anchors,frames,_positions=self._structural_inventory();current_keys={tuple(member[:3]) for member in current}
        for frame in frames:
            if frame not in current_keys:continue
            if frame[0]!='S':return True
            companions=set()
            for row in self._examples:
                keys={tuple(member[:3]) for member in row['sketch']}
                if frame not in keys:continue
                companions.update(key for key in keys if key in frames and key[0]=='P')
            if not companions or any(key in current_keys for key in companions):return True
        return False

    def _structural_frame_strength(self,raw):
        current=self._sketch(tuple(map(int,raw)))
        if not current:return (0,0)
        _anchors,frames,_positions=self._structural_inventory();keys={tuple(member[:3]) for member in current}
        prefix=max((int(key[1]) for key in frames if key[0]=='P' and key in keys),default=0)
        suffix=max((int(key[1]) for key in frames if key[0]=='S' and key in keys),default=0)
        return int(prefix),int(suffix)

    def _embedded_scope_owns_action(self,raw,action):
        raw=tuple(map(int,raw));full_prefix,full_suffix=self._structural_frame_strength(raw)
        if full_prefix>0 or full_suffix<=0:return False
        # A child can own top-level action force only if it begins with an already
        # productive prefix frame. Prefilter suffix children by those learned hashes
        # before invoking structural reconstruction; this preserves the exact scope
        # law while avoiding history-wide analysis for impossible child boundaries.
        _anchors,frames,_positions=self._structural_inventory()
        prefix_frames={key for key in frames if key[0]=='P'}
        if not prefix_frames:return False
        for start in range(1,max(1,len(raw)-MIN_FACTOR_BYTES+1)):
            child=raw[start:];upper=min(MAX_FACTOR_BYTES,len(child));has_prefix=False
            for n in range(MIN_FACTOR_BYTES,upper+1):
                if ('P',n,_hash('prefix',child[:n])) in prefix_frames:
                    has_prefix=True;break
            if not has_prefix:continue
            rows=self.structural_candidates(child)
            if len(rows)==1 and int(rows[0][0])==int(action):return True
        return False

    def _target_surface_factors(self,current,action):
        edges={row for row in current if len(row)==3};internals={};scored=[]
        for row in current:
            if len(row)==5 and row[0]=='I':internals.setdefault(row[:3],[]).append((int(row[3]),int(row[4])))
        for factor in self._factors:
            if int(factor.action_identity)!=int(action):continue
            edge_members={row for row in factor.members if len(row)==3};matched=edges&edge_members
            prefix=max((int(row[1]) for row in matched if row[0]=='P'),default=0);suffix=max((int(row[1]) for row in matched if row[0]=='S'),default=0)
            interior=0;interior_hits=0
            for member in factor.members:
                if len(member)!=5 or member[0]!='I':continue
                key=member[:3];max_left,max_right=int(member[3]),int(member[4])
                if any(left<=max_left and right<=max_right for left,right in internals.get(key,())):interior=max(interior,int(member[1]));interior_hits+=1
            if prefix>=MIN_TRANSFER_BYTES or (prefix>=MIN_FACTOR_BYTES and suffix>=MIN_TRANSFER_BYTES) or interior>=MIN_TRANSFER_BYTES:
                scored.append((max(prefix,suffix,interior),min(prefix,suffix),len(matched)+interior_hits,int(factor.identity)))
        if not scored:return ()
        peak=max(row[:3] for row in scored)
        return tuple(sorted(identity for primary,secondary,hits,identity in scored if (primary,secondary,hits)==peak))

    def supporting_factors(self,raw,action_identity):
        """Rematerialize language support for an independently observed target action.

        This is post-consequence evidence lookup, not action recognition: the physical
        action identity is already authoritative, so unrelated action candidates need
        not compete merely to recover the supporting structural factor.
        """
        raw=tuple(map(int,raw));action=int(action_identity);current=self._sketch(raw)
        if not current or action<=0:return ()
        if any(int(row['action'])==action and tuple(row['sketch'])==tuple(current) for row in self._examples):
            return self._target_surface_factors(current,action)
        anchors,frames,positions=self._structural_inventory();current_by_key={}
        for member in current:current_by_key.setdefault(tuple(member[:3]),[]).append(member)
        matched_frames=tuple(key for key in frames if key in current_by_key)
        if matched_frames:
            guarded=[]
            for frame in matched_frames:
                if frame[0]!='S':guarded.append(frame);continue
                companions=set()
                for row in self._examples:
                    keys={tuple(member[:3]) for member in row['sketch']}
                    if frame not in keys:continue
                    companions.update(key for key in keys if key in frames and key[0]=='P')
                if companions and not any(key in current_by_key for key in companions):continue
                guarded.append(frame)
            matched_frames=tuple(guarded);prefix_frames=tuple(key for key in matched_frames if key[0]=='P')
            if prefix_frames:matched_frames=prefix_frames
        action_anchors=anchors.get(action,set())
        compatible=[]
        for key in action_anchors:
            for member in current_by_key.get(key,()):
                if self._sketch_position(member) in positions.get((key,action),()):compatible.append(key);break
        if matched_frames and compatible:
            frame_peak=max(int(key[1]) for key in matched_frames);matched_frames=tuple(key for key in matched_frames if int(key[1])==frame_peak)
            anchor_peak=max(int(key[1]) for key in compatible)
            if anchor_peak>MIN_TRANSFER_BYTES or self._has_contrastive_rival(action):
                strongest=tuple(sorted(key for key in compatible if int(key[1])==anchor_peak))
                return (_identity('open-language-action-structural-factor-v2',[
                    action,[list(key) for key in matched_frames],[list(key) for key in strongest]]),)
        if self._has_productive_structural_frame(raw):return ()
        return self._target_surface_factors(current,action)

    def candidates(self,raw):
        structural=self.structural_candidates(raw)
        if structural:
            action=int(structural[0][0]) if len(structural)==1 else 0
            if (action>0 and not self.witnessed_action_surface(raw,action)
                    and self._embedded_scope_owns_action(raw,action)):
                self.last_candidate_mode='structural-embedded-refuse';return ()
            self.last_candidate_mode='structural';return structural
        # Once the organism has induced a productive construction frame, failure to
        # bind a predicate inside that frame is a structural refusal. A fuzzy raw-span
        # fallback may not overrule the hierarchy with accidental lexical overlap.
        if self._has_productive_structural_frame(raw):
            self.last_candidate_mode='structural-refuse';return ()
        self.last_candidate_mode='surface-family'
        sketch_rows=self._sketch(tuple(map(int,raw)));edges={row for row in sketch_rows if len(row)==3}
        internals={}
        for row in sketch_rows:
            if len(row)==5 and row[0]=='I':internals.setdefault(row[:3],[]).append((int(row[3]),int(row[4])))
        scored=[];touches=0
        for factor in self._factors:
            touches+=1;edge_members={row for row in factor.members if len(row)==3};matched=edges&edge_members
            prefix=max((int(row[1]) for row in matched if row[0]=='P'),default=0)
            suffix=max((int(row[1]) for row in matched if row[0]=='S'),default=0)
            interior=0;interior_hits=0
            for member in factor.members:
                if len(member)!=5 or member[0]!='I':continue
                key=member[:3];max_left,max_right=int(member[3]),int(member[4])
                if any(left<=max_left and right<=max_right for left,right in internals.get(key,())):
                    interior=max(interior,int(member[1]));interior_hits+=1
            edge_ok=(prefix>=MIN_TRANSFER_BYTES or
                     (prefix>=MIN_FACTOR_BYTES and suffix>=MIN_TRANSFER_BYTES))
            if edge_ok or interior>=MIN_TRANSFER_BYTES:
                scored.append((max(prefix,suffix,interior),min(prefix,suffix),len(matched)+interior_hits,
                               factor.action_identity,factor.identity))
        self.last_match_touches=touches
        if not scored:return ()
        best_key=max((primary,secondary,matches) for primary,secondary,matches,_a,_i in scored)
        actions={a for primary,secondary,matches,a,_i in scored if (primary,secondary,matches)==best_key}
        if len(actions)!=1:return ()
        action=next(iter(actions))
        factor_ids=tuple(sorted(i for primary,secondary,matches,a,i in scored
                                if (primary,secondary,matches)==best_key and a==action))
        return ((int(action),factor_ids),)

    def nominate(self,bank,raw,context_signature,tick):
        """Rematerialize zero-authority nominations through the incumbent action bank."""
        rows=[]
        for action,factors in self.candidates(raw):
            factor=int(factors[0]);rows.append(bank.nominate(
                action,factor,int(context_signature),factor,action,int(context_signature),int(tick)))
        return tuple(rows)

    def witnessed_action_surface(self,raw,action_identity):
        """Whether this exact structural sketch was itself followed by this action."""
        sketch=self._sketch(tuple(map(int,raw)));action=int(action_identity)
        return bool(sketch and any(int(row['action'])==action and tuple(row['sketch'])==tuple(sketch)
                                   for row in self._examples))

    @property
    def pending_count(self):return len(self._pending)
    @property
    def factor_count(self):return len(self._factors)

    def checkpoint(self):
        return {'schema':2,'examples':[
            {'action':int(r['action']),'source':int(r['source']),'tick':int(r['tick']),
             'sketch':[list(member) for member in r['sketch']]}
            for r in self._examples],
            'factors':[{'identity':f.identity,'action':f.action_identity,'sources':list(f.sources),
                        'members':[list(member) for member in f.members]}
                       for f in self._factors]}

    @classmethod
    def restore(cls,data):
        schema=int(data.get('schema',0))
        if schema not in (1,2):raise RuntimeError('open-language-action:checkpoint-schema')
        out=cls()
        def member(raw):
            if len(raw)==3:return (str(raw[0]),int(raw[1]),str(raw[2]))
            if len(raw)==5:return (str(raw[0]),int(raw[1]),str(raw[2]),int(raw[3]),int(raw[4]))
            raise RuntimeError('open-language-action:checkpoint-member')
        for r in data.get('examples',()):
            sketch=tuple(member(row) for row in r.get('sketch',()))
            action=int(r.get('action',0));source=int(r.get('source',0));tick=int(r.get('tick',-1))
            if min(action,source)<=0 or tick<0:raise RuntimeError('open-language-action:checkpoint-example')
            out._examples.append({'action':action,'source':source,'tick':tick,'sketch':sketch})
        out._examples_revision=(1 if out._examples else 0);out._structural_inventory_cache=None
        for r in data.get('factors',()):
            members=tuple(member(row) for row in r.get('members',()))
            f=OpenLanguageActionFactorV1(int(r.get('identity',0)),int(r.get('action',0)),members,tuple(map(int,r.get('sources',()))))
            if min(f.identity,f.action_identity)<=0 or len(f.sources)<2:raise RuntimeError('open-language-action:checkpoint-factor')
            out._factors.append(f)
        if len(out._examples)>MAX_EXAMPLES or len(out._factors)>MAX_FACTORS:raise RuntimeError('open-language-action:checkpoint-capacity')
        return out
