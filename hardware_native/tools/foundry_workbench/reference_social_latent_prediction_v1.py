#!/usr/bin/env python3
"""Opaque social latent-state inference for fast Adult language experiments.

Hidden-state hypotheses are induced only from observable partner-history signatures.
They are prediction conveniences, not truth, mind labels, or world authority.
"""
from __future__ import annotations
from dataclasses import dataclass
import hashlib,json


def _id(tag,values):
    raw=json.dumps([int(x) for x in values],separators=(',',':')).encode()
    return int.from_bytes(hashlib.sha256(tag.encode()+b'\0'+raw).digest()[:8],'little') or 1


@dataclass(frozen=True)
class SocialLatentHypothesisV1:
    identity:int
    features:tuple[int,...]
    support:int


class SocialLatentPredictionV1:
    """Sparse nearest-evidence latent hypotheses over another agent's observations."""
    def __init__(self,min_support=2):
        self.min_support=int(min_support)
        self._feature_sets={}
        self._feature_ids={}
        self._feature_index={}
        self._action_returns={}
        # Derived lookup for local resident action competition.  The learned
        # return rows are causal state; this index is rematerialized on restore.
        self._action_index={}
        # Unsettled action tickets are future-relevant eligibility, not dialogue
        # turns.  Only an independently returned consequence can consume one.
        self._pending_actions={}
        self._withdrawn=set()
        self.last_touches=0
        self._infer_epoch=0
        self._cached_infer_observed=None
        self._cached_infer=0
        self._cached_infer_epoch=-1
        self.last_choose_touches=0
        self._return_epoch=0
        self._cached_choose_key=None
        self._cached_choose=0
        self._cached_choose_epoch=-1

    def observe_history(self,features,source):
        features=tuple(sorted(set(int(x) for x in features if int(x))))
        if not features:return 0
        if features not in self._feature_sets:
            self._feature_sets[features]=set()
            self._feature_ids[features]=_id('social-latent-v1',features)
            for feature in features:self._feature_index.setdefault(feature,set()).add(features)
        self._feature_sets[features].add(int(source));self._infer_epoch+=1
        return self._feature_ids[features]

    def withdraw_source(self,source):
        self._withdrawn.add(int(source));self._infer_epoch+=1;self._return_epoch+=1

    def hypotheses(self):
        rows=[]
        for features,sources in self._feature_sets.items():
            support=len(sources-self._withdrawn)
            if support>=self.min_support:rows.append(SocialLatentHypothesisV1(self._feature_ids[features],features,support))
        rows.sort(key=lambda x:x.identity);return tuple(rows)

    def _rank(self,observed_features):
        observed=frozenset(int(x) for x in observed_features if int(x))
        self.last_touches=0;nominated=set()
        for feature in observed:nominated.update(self._feature_index.get(feature,()))
        known_observed={feature for feature in observed if feature in self._feature_index}
        rows=[]
        for features in nominated:
            sources=self._feature_sets.get(features)
            if sources is None:continue
            support=len(sources-self._withdrawn)
            if support<self.min_support:continue
            self.last_touches+=1;target=set(features);overlap=len(known_observed&target)
            missing=len(target-known_observed);extra=len(known_observed-target)
            score=4*overlap-2*missing-extra
            rows.append((score,support,self._feature_ids[features]))
        rows.sort(reverse=True)
        return observed,rows

    def infer(self,observed_features):
        observed=frozenset(int(x) for x in observed_features if int(x))
        if observed==self._cached_infer_observed and self._cached_infer_epoch==self._infer_epoch:
            self.last_touches=0;return self._cached_infer
        observed,rows=self._rank(observed)
        tied=len(rows)>1 and rows[0][:2]==rows[1][:2]
        out=0 if not rows or rows[0][0]<=0 or tied else rows[0][2]
        self._cached_infer_observed=observed;self._cached_infer=out;self._cached_infer_epoch=self._infer_epoch
        return out

    def competing_state(self,observed_features):
        """Rematerialize the best live alternative closure for this observation."""
        _observed,rows=self._rank(observed_features)
        if not rows or rows[0][0]<=0:return 0,()
        best=rows[0][:2]
        alternatives=tuple(sorted(row[2] for row in rows if row[:2]==best))
        state=alternatives[0] if len(alternatives)==1 else _id('social-competition-v1',alternatives)
        return state,alternatives

    def _record_action_return(self,state_identity,action_identity,outcome,source):
        state_identity=int(state_identity);action_identity=int(action_identity);source=int(source)
        if min(state_identity,action_identity,source)<=0 or source in self._withdrawn:return False
        key=(state_identity,action_identity);rows=self._action_returns.setdefault(key,{})
        total,count=rows.get(source,(0,0));rows[source]=(total+int(outcome),count+1)
        self._action_index.setdefault(state_identity,set()).add(action_identity)
        self._return_epoch+=1;return True

    def observe_action_return(self,hypothesis_identity,action_identity,outcome,source,independent=True):
        if not independent:return False
        return self._record_action_return(hypothesis_identity,action_identity,outcome,source)

    def begin_action(self,state_identity,action_identity,ticket):
        state_identity=int(state_identity);action_identity=int(action_identity);ticket=int(ticket)
        if min(state_identity,action_identity,ticket)<=0 or ticket in self._pending_actions:return False
        self._pending_actions[ticket]=(state_identity,action_identity);return True

    def settle_action(self,ticket,outcome,source,independent=True):
        ticket=int(ticket);pending=self._pending_actions.get(ticket)
        if pending is None or not independent:return False
        if not self._record_action_return(pending[0],pending[1],outcome,source):return False
        self._pending_actions.pop(ticket);return True

    def expected_return(self,hypothesis_identity,action_identity):
        rows=self._action_returns.get((int(hypothesis_identity),int(action_identity)),{})
        total=count=0
        for source,(source_sum,source_count) in rows.items():
            if source in self._withdrawn:continue
            total+=source_sum;count+=source_count
        return 0 if not count else total/count

    def choose(self,hypothesis_identity,actions):
        if not hypothesis_identity:return 0
        actions=tuple(int(action) for action in actions)
        key=(int(hypothesis_identity),actions)
        if key==self._cached_choose_key and self._cached_choose_epoch==self._return_epoch:
            self.last_choose_touches=0;return self._cached_choose
        winner=0;best=None;ties=0;self.last_choose_touches=0
        for action in actions:
            self.last_choose_touches+=1
            score=self.expected_return(hypothesis_identity,action)
            if best is None or score>best:best=score;winner=action;ties=1
            elif score==best:ties+=1
        out=0 if best is None or best<=0 or ties!=1 else winner
        self._cached_choose_key=key;self._cached_choose=out;self._cached_choose_epoch=self._return_epoch
        return out

    def choose_resident(self,state_identity):
        return self.choose(int(state_identity),tuple(sorted(self._action_index.get(int(state_identity),()))))

    def resident_choice(self,observed_features):
        state,alternatives=self.competing_state(observed_features)
        return self.choose_resident(state),state,alternatives

    def checkpoint(self):
        return {'schema':2,'min_support':self.min_support,
                'feature_sets':[{'features':list(features),'sources':sorted(sources)}
                                for features,sources in sorted(self._feature_sets.items())],
                'action_returns':[{'hypothesis':key[0],'action':key[1],
                                   'sources':[{'identity':source,'total':value[0],'count':value[1]}
                                              for source,value in sorted(rows.items())]}
                                  for key,rows in sorted(self._action_returns.items())],
                'pending_actions':[{'ticket':ticket,'state':row[0],'action':row[1]}
                                   for ticket,row in sorted(self._pending_actions.items())],
                'withdrawn':sorted(self._withdrawn)}

    @classmethod
    def restore(cls,data):
        if int(data.get('schema',0))!=2:raise ValueError('social_latent:checkpoint_schema')
        bank=cls(int(data.get('min_support',0)))
        if bank.min_support<=0:raise ValueError('social_latent:checkpoint_support')
        for row in data.get('feature_sets',()):
            features=tuple(map(int,row.get('features',())));sources=set(map(int,row.get('sources',())))
            if (not features or features!=tuple(sorted(set(features))) or features in bank._feature_sets or
                    any(x<=0 for x in features) or any(x<=0 for x in sources)):
                raise ValueError('social_latent:checkpoint_features')
            bank._feature_sets[features]=sources
            bank._feature_ids[features]=_id('social-latent-v1',features)
            for feature in features:bank._feature_index.setdefault(feature,set()).add(features)
        for row in data.get('action_returns',()):
            key=(int(row.get('hypothesis',0)),int(row.get('action',0)))
            if min(key)<=0 or key in bank._action_returns:raise ValueError('social_latent:checkpoint_action')
            sources={}
            for value in row.get('sources',()):
                source=int(value.get('identity',0));total=int(value.get('total',0));count=int(value.get('count',0))
                if source<=0 or source in sources or count<=0:raise ValueError('social_latent:checkpoint_return')
                sources[source]=(total,count)
            if not sources:raise ValueError('social_latent:checkpoint_return')
            bank._action_returns[key]=sources
            bank._action_index.setdefault(key[0],set()).add(key[1])
        for row in data.get('pending_actions',()):
            ticket=int(row.get('ticket',0));state=int(row.get('state',0));action=int(row.get('action',0))
            if min(ticket,state,action)<=0 or ticket in bank._pending_actions:raise ValueError('social_latent:checkpoint_pending')
            bank._pending_actions[ticket]=(state,action)
        bank._withdrawn=set(map(int,data.get('withdrawn',())))
        if any(x<=0 for x in bank._withdrawn):raise ValueError('social_latent:checkpoint_withdrawn')
        return bank
