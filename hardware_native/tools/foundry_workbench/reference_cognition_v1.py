#!/usr/bin/env python3
"""Strict numeric learned transition/counterfactual planning ecology."""
from __future__ import annotations
from dataclasses import dataclass
import hashlib,json

MIN_SOURCE_SUPPORT=2
MAX_STATE_FEATURES=32
MAX_PLAN_DEPTH=8
MAX_EVIDENCE=16384
MAX_PROSPECTIVE_RECIPES=1024
MAX_PROSPECTIVE_REHEARSALS=65535
MAX_PROSPECTIVE_EXPERTS=1024
MAX_PROSPECTIVE_INTENTIONS=64
PROSPECTIVE_TTL_TICKS=8
EXPERT_MIN_COMPLETIONS=3
EXPERT_PROBATION_PASSES=2


def _state(values):
    out=tuple(sorted(set(int(x) for x in values if int(x)!=0)))
    if not out or len(out)>MAX_STATE_FEATURES:raise ValueError('cognition:state')
    return out

def _digest(tag,obj):return hashlib.sha256(tag.encode()+b'\0'+json.dumps(obj,sort_keys=True,separators=(',',':')).encode()).hexdigest()

@dataclass(frozen=True)
class TransitionEdgeV1:
    state:tuple[int,...]
    action:int
    next_state:tuple[int,...]
    effect:int
    support:int
    sources:tuple[int,...]

@dataclass(frozen=True)
class PlanV1:
    status:int                 # 0 none, 1 unique, 2 unresolved alternatives
    actions:tuple[int,...]
    states:tuple[tuple[int,...],...]
    score:int
    alternatives:int
    alternative_actions:tuple[int,...]=()
    recipe_identity:int=0
    shadow_credit:int=0
    cue_exact:int=1
    effects:tuple[int,...]=()
    sources:tuple[int,...]=()
    intention_identity:int=0

@dataclass
class ProspectiveRecipeV1:
    identity:int
    start:tuple[int,...]
    goal:tuple[int,...]
    actions:tuple[int,...]
    states:tuple[tuple[int,...],...]
    effects:tuple[int,...]
    sources:tuple[int,...]
    shadow_credit:int
    shadow_counter:int=0
    rehearsals:int=1
    counter_sources:tuple[int,...]=()
    created_tick:int=0
    expires_tick:int=PROSPECTIVE_TTL_TICKS

@dataclass
class ProspectiveExpertRecipeV1:
    """Compact reusable frontier-selection relation with rematerialization witness."""
    identity:int
    start:tuple[int,...]
    goal:tuple[int,...]
    actions:tuple[int,...]
    states:tuple[tuple[int,...],...]
    context_signature:int
    witness_recipes:tuple[int,...]
    completion_sources:tuple[int,...]
    completion_count:int
    witness_digest:str
    frontier_revision:int
    probation_passes:int=0
    active:int=0
    uses:int=0
    deoptimizations:int=0

@dataclass
class ProspectiveIntentionV1:
    """One unfinished route bound by resident action, not a continuously active sentence."""
    identity:int
    recipe_identity:int
    goal:tuple[int,...]
    actions:tuple[int,...]
    states:tuple[tuple[int,...],...]
    effects:tuple[int,...]
    sources:tuple[int,...]
    cursor:int
    created_tick:int
    last_retrieved_tick:int=0
    retrievals:int=0

class TransitionEcologyV1:
    def __init__(self,resident_authority=None):
        self.__resident_authority=resident_authority
        self._evidence:dict[tuple[tuple[int,...],int,tuple[int,...],int],set[int]]={}
        self._evidence_by_state_action:dict[tuple[tuple[int,...],int],set[tuple]]={}
        self._incoming_by_state:dict[tuple[int,...],set[tuple]]={}
        self._actions_by_state:dict[tuple[int,...],set[int]]={}
        self._reconcile_frontier:set[tuple[int,...]]=set()
        self._reconcile_priority:dict[tuple[int,...],int]={}
        self._withdrawn:set[int]=set()
        self._prospective_recipes:dict[tuple[tuple[int,...],tuple[int,...]],ProspectiveRecipeV1]={}
        self._prospective_experts:dict[tuple[tuple[int,...],tuple[int,...],int],ProspectiveExpertRecipeV1]={}
        self._prospective_intentions:dict[int,ProspectiveIntentionV1]={}
        self._intention_ids_by_atom:dict[int,set[int]]={}
        self._expert_completion_sources:dict[tuple[tuple[int,...],tuple[int,...],tuple[int,...],int],set[int]]={}
        self._expert_completion_counts:dict[tuple[tuple[int,...],tuple[int,...],tuple[int,...],int],int]={}
        self._frontier_revision=0
        self.last_plan_touches=0
        self.last_reconcile_touches=0
        self.last_reconcile_attempts=0
        self.last_intention_touches=0

    def observe(self,state,action:int,next_state,effect:int,source:int,independent:bool=True):
        if not independent:return False
        if len(self._evidence)>=MAX_EVIDENCE and (tuple(state),int(action),tuple(next_state),int(effect)) not in self._evidence:raise ValueError('cognition:evidence_bound')
        state=_state(state);next_state=_state(next_state);action=int(action);effect=int(effect);source=int(source)
        key=(state,action,next_state,effect);bucket=self._evidence.setdefault(key,set());new_source=source not in bucket;bucket.add(source)
        if new_source:self._frontier_revision+=1
        self._evidence_by_state_action.setdefault((state,action),set()).add(key)
        self._incoming_by_state.setdefault(next_state,set()).add(key)
        self._actions_by_state.setdefault(state,set()).add(action)
        if new_source:self._reconcile_frontier.update((state,next_state))
        alternatives=self._evidence_by_state_action.get((state,action),())
        if new_source and len(alternatives)>1:
            # A contradictory actual return raises only scheduling eligibility.
            # Locally reachable unambiguous alternatives become the first
            # target-free reconciliation candidates; no evidence or credit is added.
            for candidate_action in self._actions_by_state.get(state,()):
                candidate=self.transition(state,candidate_action,1)
                if candidate is None:continue
                self._reconcile_frontier.add(candidate.next_state)
                self._reconcile_priority[candidate.next_state]=(
                    self._reconcile_priority.get(candidate.next_state,0)+1)
        invalidated=[]
        for recipe_key,recipe in self._prospective_recipes.items():
            for index,(prior,planned_action,expected) in enumerate(zip(recipe.states,recipe.actions,recipe.states[1:])):
                expected_effect=recipe.effects[index]
                reinforced=(prior==state and planned_action==action and expected==next_state
                            and expected_effect==effect and new_source)
                if reinforced:
                    invalidated.append(recipe_key);break
                contradicted=(prior==state and planned_action==action and
                              (expected!=next_state or (expected_effect>0) != (effect>0)))
                if contradicted and source not in recipe.counter_sources:
                    recipe.shadow_counter+=1
                    recipe.counter_sources=tuple(sorted((*recipe.counter_sources,source)))
                    break
        for recipe_key in invalidated:self._prospective_recipes.pop(recipe_key,None)
        self._settle_prospective_intentions(state,action,next_state,effect)
        return True

    def _reconcile_local(self,resident_authority,current_tick:int,budget:int=1):
        """Try bounded target-free two-hop condensation around recent contact."""
        if resident_authority is None or resident_authority is not self.__resident_authority:
            raise ValueError('cognition:prospective_authority')
        budget=max(1,min(int(budget),64));built=attempts=touches=0
        ordered=sorted(tuple(self._reconcile_frontier),
                       key=lambda middle:(-self._reconcile_priority.get(middle,0),middle))
        for middle in ordered:
            incoming=sorted(self._incoming_by_state.get(middle,()))
            outgoing=[]
            for action in sorted(self._actions_by_state.get(middle,())):
                outgoing.extend(sorted(self._evidence_by_state_action.get((middle,action),())))
            touches+=len(incoming)+len(outgoing)
            candidates=[]
            for left in incoming:
                if not self._active_sources(self._evidence.get(left,())):continue
                for right in outgoing:
                    if not self._active_sources(self._evidence.get(right,())):continue
                    if left[0]!=right[2]:candidates.append((left[0],right[2]))
            self._reconcile_frontier.discard(middle)
            self._reconcile_priority.pop(middle,None)
            for start,goal in sorted(set(candidates)):
                attempts+=1
                plan=self._condense_prospective(start,goal,resident_authority,current_tick=current_tick)
                built+=int(bool(plan.recipe_identity))
                if attempts>=budget:break
            if attempts>=budget:break
        self.last_reconcile_touches=touches;self.last_reconcile_attempts=attempts
        return built

    def withdraw_source(self,source:int):
        source=int(source)
        if source not in self._withdrawn:self._frontier_revision+=1
        self._withdrawn.add(source)
    def restore_source(self,source:int):
        source=int(source)
        if source in self._withdrawn:self._frontier_revision+=1
        self._withdrawn.discard(source)
    def _active_sources(self,sources):return tuple(sorted(s for s in sources if s not in self._withdrawn))

    def edges(self,min_support:int=MIN_SOURCE_SUPPORT):
        min_support=max(1,int(min_support))
        rows=[]
        for (state,action,nxt,effect),sources in self._evidence.items():
            active=self._active_sources(sources)
            if len(active)>=min_support:rows.append(TransitionEdgeV1(state,action,nxt,effect,len(active),active))
        return tuple(sorted(rows,key=lambda x:(x.state,x.action,x.next_state,x.effect)))

    def transition(self,state,action:int,min_support:int=MIN_SOURCE_SUPPORT):
        state=_state(state);action=int(action);min_support=max(1,int(min_support));rows=[]
        for key in self._evidence_by_state_action.get((state,action),()):
            sources=self._evidence.get(key)
            if sources is None:continue
            active=self._active_sources(sources)
            if len(active)>=min_support:
                rows.append(TransitionEdgeV1(key[0],key[1],key[2],key[3],len(active),active))
        if not rows:return None
        peak=max(e.support for e in rows);w=[e for e in rows if e.support==peak]
        # Same next state can have multiple effect observations; combine only if destination is unique.
        destinations=sorted(set(e.next_state for e in w))
        if len(destinations)!=1:return None
        dest=destinations[0];selected=[e for e in w if e.next_state==dest]
        # Effect is consequence/utility evidence, not truth. Use support-weighted signed sum only for ranking.
        effect=sum(e.effect*e.support for e in selected)
        support=sum(e.support for e in selected);sources=tuple(sorted(set(s for e in selected for s in e.sources)))
        return TransitionEdgeV1(state,int(action),dest,effect,support,sources)

    @staticmethod
    def satisfies(state,goal):return set(_state(goal)).issubset(set(_state(state)))

    def _search_plan(self,start,goal,max_depth=MAX_PLAN_DEPTH,min_support:int=MIN_SOURCE_SUPPORT):
        start=_state(start);goal=_state(goal);max_depth=max(1,min(int(max_depth),MAX_PLAN_DEPTH))
        self.last_plan_touches=0
        if self.satisfies(start,goal):return PlanV1(1,(),(start,),0,1,())
        frontier=[(start,(),(start,),0)];solutions=[];seen_depth={start:0}
        for depth in range(max_depth):
            nxt_frontier=[]
            for state,actions,states,score in frontier:
                for action in sorted(self._actions_by_state.get(state,())):
                    self.last_plan_touches+=1
                    edge=self.transition(state,action,min_support)
                    if edge is None:continue
                    na=actions+(action,);ns=states+(edge.next_state,);sc=score+edge.effect
                    if self.satisfies(edge.next_state,goal):solutions.append((len(na),-sc,na,ns,sc));continue
                    nd=depth+1
                    if seen_depth.get(edge.next_state,nd+1)<nd:continue
                    seen_depth[edge.next_state]=nd;nxt_frontier.append((edge.next_state,na,ns,sc))
            if solutions:break
            frontier=nxt_frontier
        if not solutions:return PlanV1(0,(),(start,),0,0,())
        solutions.sort(key=lambda x:(x[0],x[1],x[2]));best_depth,best_neg=solutions[0][:2];best=[x for x in solutions if x[0]==best_depth and x[1]==best_neg]
        first_actions=sorted(set(x[2][0] for x in best if x[2]))
        if len(first_actions)!=1:return PlanV1(2,(),(start,),-best_neg,len(first_actions),tuple(first_actions))
        # Multiple paths sharing the same first action are safe to begin; keep deterministic best continuation.
        chosen=best[0];return PlanV1(1,chosen[2],chosen[3],chosen[4],len(best),(chosen[2][0],) if chosen[2] else ())

    def prospective_frontier(self,start,goal,max_depth=MAX_PLAN_DEPTH,max_candidates=8,
                             current_tick=None,depth_slack:int=0):
        """Return a bounded population of one-source prospective paths.

        ``depth_slack`` keeps additional simple paths beyond the first solution
        depth. These are endogenous candidate Networks, not authoritative plans.
        Actual transition consequences determine their score; equal score preserves
        ambiguity. No candidate is persisted or credited merely by appearing here.
        """
        start=_state(start);goal=_state(goal);max_depth=max(1,min(int(max_depth),MAX_PLAN_DEPTH))
        max_candidates=max(1,min(int(max_candidates),64));depth_slack=max(0,min(int(depth_slack),MAX_PLAN_DEPTH-1));self.last_plan_touches=0
        if self.satisfies(start,goal):return (PlanV1(1,(),(start,),0,1,()),)
        frontier=[(start,(),(start,),0,(),())];solutions=[];first_solution_depth=None
        for depth in range(max_depth):
            nxt=[]
            for state,actions,states,score,effects,sources in frontier:
                for action in sorted(self._actions_by_state.get(state,())):
                    self.last_plan_touches+=1;edge=self.transition(state,action,1)
                    if edge is None or edge.next_state in states:continue
                    na=actions+(action,);ns=states+(edge.next_state,);sc=score+edge.effect
                    ne=effects+(edge.effect,);src=tuple(sorted(set((*sources,*edge.sources))))
                    if self.satisfies(edge.next_state,goal):
                        solutions.append((na,ns,sc,ne,src))
                        if first_solution_depth is None:first_solution_depth=len(na)
                        continue
                    if first_solution_depth is None or len(na)<first_solution_depth+depth_slack:
                        nxt.append((edge.next_state,na,ns,sc,ne,src))
            if first_solution_depth is not None and depth+1>=first_solution_depth+depth_slack:break
            frontier=nxt
        rows=[];seen=set()
        for actions,states,score,effects,sources in sorted(solutions,key=lambda x:(-x[2],x[0],x[1])):
            key=(actions,states)
            if key in seen:continue
            seen.add(key)
            identity=int(_digest('prospective-recipe-v1',[list(start),list(goal),list(actions),
                [list(x) for x in states],list(effects),list(sources)])[:15],16) or 1
            support=min((self.transition(s,a,1).support for s,a in zip(states,actions)),default=0)
            rows.append(PlanV1(1,actions,states,score,len(solutions),
                               (actions[0],) if actions else (),identity,support))
            if len(rows)>=max_candidates:break
        return tuple(rows)

    @staticmethod
    def _expert_witness_digest(start,goal,context_signature,frontier):
        return _digest('prospective-expert-witness-v1',[
            list(_state(start)),list(_state(goal)),int(context_signature),
            [[int(row.recipe_identity),list(row.actions),[list(x) for x in row.states],int(row.score)]
             for row in frontier]
        ])

    def record_expert_completion(self,start,goal,actions,states,source:int,context_signature:int):
        """Nominate only after repeated independent successful whole-route returns."""
        start=_state(start);goal=_state(goal);actions=tuple(map(int,actions));states=tuple(_state(x) for x in states)
        source=int(source);context_signature=int(context_signature)
        if (source<=0 or source in self._withdrawn or not actions or len(states)!=len(actions)+1
                or states[0]!=start or states[-1]!=goal):return None
        expert_key=(start,goal,context_signature);prior=self._prospective_experts.get(expert_key)
        if prior is not None:return prior
        key=(start,goal,actions,context_signature);sources=self._expert_completion_sources.setdefault(key,set());sources.add(source)
        count=self._expert_completion_counts.get(key,0)+1;self._expert_completion_counts[key]=count
        if count<EXPERT_MIN_COMPLETIONS:return None
        frontier=self.prospective_frontier(start,goal,max_candidates=64,depth_slack=2)
        exact=[row for row in frontier if row.actions==actions and row.states==states]
        if len(exact)!=1:return None
        peak=max((row.score*1024)//max(1,len(row.actions)) for row in frontier)
        winners=[row for row in frontier if (row.score*1024)//max(1,len(row.actions))==peak]
        if len(winners)!=1 or winners[0].actions!=actions:return None
        if len(self._prospective_experts)>=MAX_PROSPECTIVE_EXPERTS:raise ValueError('cognition:prospective_expert_bound')
        witness=tuple(int(row.recipe_identity) for row in frontier)
        wd=self._expert_witness_digest(start,goal,context_signature,frontier)
        identity=int(_digest('prospective-expert-recipe-v1',[
            list(start),list(goal),list(actions),[list(x) for x in states],context_signature,
            list(witness),sorted(sources),count,wd,self._frontier_revision])[:15],16) or 1
        expert=ProspectiveExpertRecipeV1(identity,start,goal,actions,states,context_signature,
            witness,tuple(sorted(sources)),count,wd,self._frontier_revision)
        self._prospective_experts[expert_key]=expert
        return expert

    def probation_expert(self,start,goal,context_signature:int,selected:PlanV1,frontier):
        """Earn activation only through exact shadow equivalence to live frontier selection."""
        key=(_state(start),_state(goal),int(context_signature));expert=self._prospective_experts.get(key)
        if expert is None or expert.active:return expert
        frontier=tuple(frontier)
        # A revision mismatch is already a completed deoptimization event in
        # ``active_expert``. Do not count the same challenge again during fallback
        # probation; a new expert revision must be nominated from later recurrence.
        if expert.frontier_revision!=self._frontier_revision:return expert
        live_digest=self._expert_witness_digest(key[0],key[1],key[2],frontier)
        equivalent=(expert.witness_digest==live_digest
                    and selected.status==1 and selected.actions==expert.actions and selected.states==expert.states)
        if equivalent:
            expert.probation_passes+=1
            if expert.probation_passes>=EXPERT_PROBATION_PASSES:expert.active=1
        else:
            expert.probation_passes=0;expert.deoptimizations+=1
        return expert

    def active_expert(self,start,goal,context_signature:int):
        """Fast reusable relation; stale source/evidence revision deoptimizes to frontier."""
        self.last_plan_touches=0
        key=(_state(start),_state(goal),int(context_signature));expert=self._prospective_experts.get(key)
        if expert is None or not expert.active:return None
        if expert.frontier_revision!=self._frontier_revision:
            expert.active=0;expert.probation_passes=0;expert.deoptimizations+=1;return None
        for state,action,expected in zip(expert.states,expert.actions,expert.states[1:]):
            self.last_plan_touches+=1;edge=self.transition(state,action,1)
            if edge is None or edge.next_state!=expected:
                expert.active=0;expert.probation_passes=0;expert.deoptimizations+=1;return None
        expert.uses+=1
        score=sum((self.transition(state,action,1).effect for state,action in zip(expert.states,expert.actions)),0)
        return PlanV1(1,expert.actions,expert.states,score,1,(expert.actions[0],),expert.identity,len(expert.completion_sources))

    @staticmethod
    def _intention_identity(recipe_identity,goal,actions,states,effects,sources,created_tick):
        return int(_digest('prospective-intention-v1',[
            int(recipe_identity),list(goal),list(actions),[list(x) for x in states],
            list(effects),list(sources),int(created_tick)])[:15],16) or 1

    def _unindex_intention(self,intention):
        if intention.cursor>=len(intention.actions):return
        for atom in intention.states[intention.cursor]:
            bucket=self._intention_ids_by_atom.get(int(atom))
            if bucket is None:continue
            bucket.discard(intention.identity)
            if not bucket:self._intention_ids_by_atom.pop(int(atom),None)

    def _index_intention(self,intention):
        if intention.cursor>=len(intention.actions):return
        for atom in intention.states[intention.cursor]:
            self._intention_ids_by_atom.setdefault(int(atom),set()).add(intention.identity)

    def _retire_intention(self,identity):
        intention=self._prospective_intentions.pop(int(identity),None)
        if intention is not None:self._unindex_intention(intention)

    def _retain_prospective_intention(self,plan:PlanV1,resident_authority,current_tick:int):
        """Bind a selected learned route as one future-relevant unfinished occurrence."""
        if resident_authority is None or resident_authority is not self.__resident_authority:
            raise ValueError('cognition:prospective_authority')
        if plan.status!=1 or plan.recipe_identity<=0 or len(plan.actions)<2:return None
        effects=[];sources=set()
        for state,action,expected in zip(plan.states,plan.actions,plan.states[1:]):
            edge=self.transition(state,action,1)
            if edge is None or edge.next_state!=expected:return None
            effects.append(edge.effect);sources.update(edge.sources)
        created_tick=max(0,int(current_tick));goal=_state(plan.states[-1])
        actions=tuple(map(int,plan.actions));states=tuple(_state(x) for x in plan.states)
        effects=tuple(map(int,effects));sources=tuple(sorted(sources))
        identity=self._intention_identity(plan.recipe_identity,goal,actions,states,effects,sources,created_tick)
        same=[row for row in self._prospective_intentions.values()
              if row.recipe_identity==int(plan.recipe_identity) and row.actions==actions
              and row.states==states and row.cursor<len(row.actions)]
        if len(same)>1:raise ValueError('cognition:prospective_intention_duplicate')
        if same:return same[0]
        if len(self._prospective_intentions)>=MAX_PROSPECTIVE_INTENTIONS:
            raise ValueError('cognition:prospective_intention_bound')
        intention=ProspectiveIntentionV1(identity,int(plan.recipe_identity),goal,actions,
            states,effects,sources,0,created_tick)
        self._prospective_intentions[identity]=intention;self._index_intention(intention)
        return intention

    def _settle_prospective_intentions(self,state,action,next_state,effect):
        """Advance or retire only intentions whose current bound edge participated."""
        state=_state(state);next_state=_state(next_state);action=int(action);effect=int(effect)
        candidate_ids=set()
        for atom in state:candidate_ids.update(self._intention_ids_by_atom.get(int(atom),()))
        for identity in sorted(candidate_ids):
            row=self._prospective_intentions.get(identity)
            if row is None or row.cursor>=len(row.actions):continue
            cursor=row.cursor
            if row.states[cursor]!=state or row.actions[cursor]!=action:continue
            expected_state=row.states[cursor+1];expected_effect=row.effects[cursor]
            if expected_state!=next_state or (expected_effect>0)!=(effect>0):
                self._retire_intention(identity);continue
            self._unindex_intention(row);row.cursor+=1
            if row.cursor>=len(row.actions) or self.satisfies(next_state,row.goal):
                self._prospective_intentions.pop(identity,None)
            else:self._index_intention(row)

    def reactivate_intention(self,current_state,goal,resident_authority,current_tick:int):
        """Cue-trigger one unfinished route; partial cues can inform speech, not action."""
        if resident_authority is None or resident_authority is not self.__resident_authority:
            raise ValueError('cognition:prospective_authority')
        current=_state(current_state);goal=_state(goal);candidate_ids=set();self.last_intention_touches=0
        for atom in current:candidate_ids.update(self._intention_ids_by_atom.get(int(atom),()))
        candidates=[]
        for identity in sorted(candidate_ids):
            self.last_intention_touches+=1;row=self._prospective_intentions.get(identity)
            if row is None or row.goal!=goal or row.cursor>=len(row.actions):continue
            cue=row.states[row.cursor]
            if not set(cue).issubset(current) or any(source in self._withdrawn for source in row.sources):continue
            actions=row.actions[row.cursor:];states=row.states[row.cursor:];score=0;valid=True
            for state,action,expected in zip(states,actions,states[1:]):
                self.last_intention_touches+=1;edge=self.transition(state,action,1)
                if edge is None or edge.next_state!=expected:valid=False;break
                score+=edge.effect
            if valid:candidates.append((row,actions,states,score,int(cue==current)))
        if not candidates:return PlanV1(0,(),(current,),0,0,())
        peak=max((row[4],row[3]) for row in candidates)
        winners=[row for row in candidates if (row[4],row[3])==peak]
        if len(winners)!=1:
            first=tuple(sorted(set(row[1][0] for row in winners if row[1])))
            return PlanV1(2,(),(current,),0,len(winners),first,0,0,0)
        row,actions,states,score,exact=winners[0]
        row.last_retrieved_tick=max(0,int(current_tick));row.retrievals+=1
        return PlanV1(1,actions,states,score,1,(actions[0],),row.recipe_identity,
                      max(1,len(row.sources)),exact,row.effects[row.cursor:],row.sources,
                      row.identity)

    def _shadow_plan(self,start,goal,current_tick=None):
        start=_state(start);goal=_state(goal);candidates=[];self.last_plan_touches=0
        for recipe in self._prospective_recipes.values():
            if recipe.goal!=goal or recipe.shadow_credit<=recipe.shadow_counter:continue
            if current_tick is not None and int(current_tick)>recipe.expires_tick:continue
            if any(source in self._withdrawn for source in recipe.sources):continue
            positions=[i for i,state in enumerate(recipe.states[:-1]) if state==start]
            self.last_plan_touches+=len(recipe.states)
            if len(positions)!=1:continue
            begin=positions[0];actions=recipe.actions[begin:];states=recipe.states[begin:]
            if not actions or len(states)!=len(actions)+1:continue
            score=0;valid=True
            for state,action,expected in zip(states,actions,states[1:]):
                self.last_plan_touches+=1;edge=self.transition(state,action,1)
                if edge is None or edge.next_state!=expected:valid=False;break
                score+=edge.effect
            if valid:candidates.append((recipe,actions,states,score))
        if not candidates:return None
        first_actions=tuple(sorted(set(row[1][0] for row in candidates)))
        if len(candidates)!=1:
            return PlanV1(2,(),(start,),0,len(first_actions),first_actions)
        recipe,actions,states,score=candidates[0]
        return PlanV1(1,actions,states,score,1,(actions[0],),recipe.identity,
                      recipe.shadow_credit-recipe.shadow_counter)

    def plan(self,start,goal,max_depth=MAX_PLAN_DEPTH,current_tick=None):
        authoritative=self._search_plan(start,goal,max_depth)
        if authoritative.status!=0:return authoritative
        shadow=self._shadow_plan(start,goal,current_tick)
        return shadow if shadow is not None else authoritative

    def _condense_prospective(self,start,goal,resident_authority,max_depth=MAX_PLAN_DEPTH,current_tick=0):
        """Condense one unique receipt-backed prospective path without evidence.

        A one-source transition is eligible for shadow composition but remains
        absent from authoritative ``edges()``. Repeating the same internal route
        increments only a rehearsal counter; it cannot self-confirm its credit.
        """
        if resident_authority is None or resident_authority is not self.__resident_authority:
            raise ValueError('cognition:prospective_authority')
        plan=self._search_plan(start,goal,max_depth,1)
        if plan.status!=1 or len(plan.actions)<2:return plan
        effects=[];sources=set()
        for state,action in zip(plan.states,plan.actions):
            edge=self.transition(state,action,1)
            if edge is None:return PlanV1(0,(),(_state(start),),0,0,())
            effects.append(edge.effect);sources.update(edge.sources)
        key=(_state(start),_state(goal));identity=int(_digest(
            'prospective-recipe-v1',[list(key[0]),list(key[1]),list(plan.actions),
                                     [list(x) for x in plan.states],effects,sorted(sources)])[:15],16) or 1
        prior=self._prospective_recipes.get(key)
        if prior is not None and prior.identity==identity:
            prior.rehearsals=min(MAX_PROSPECTIVE_REHEARSALS,prior.rehearsals+1);recipe=prior
        else:
            if len(self._prospective_recipes)>=MAX_PROSPECTIVE_RECIPES:
                raise ValueError('cognition:prospective_bound')
            created_tick=max(0,int(current_tick))
            recipe=ProspectiveRecipeV1(identity,key[0],key[1],plan.actions,plan.states,
                                       tuple(effects),tuple(sorted(sources)),
                                       min(self.transition(s,a,1).support for s,a in zip(plan.states,plan.actions)),
                                       created_tick=created_tick,
                                       expires_tick=created_tick+PROSPECTIVE_TTL_TICKS)
            self._prospective_recipes[key]=recipe
        return PlanV1(1,recipe.actions,recipe.states,plan.score,1,
                      (recipe.actions[0],),recipe.identity,
                      recipe.shadow_credit-recipe.shadow_counter)

    def simulate(self,start,actions):
        state=_state(start);trace=[state];score=0
        for action in tuple(int(x) for x in actions):
            edge=self.transition(state,action)
            if edge is None:return None
            state=edge.next_state;trace.append(state);score+=edge.effect
        return tuple(trace),score

    def checkpoint(self):
        return {'schema':2,
                'evidence':[{'state':list(s),'action':a,'next_state':list(n),'effect':e,'sources':sorted(src)} for (s,a,n,e),src in sorted(self._evidence.items())],
                'withdrawn':sorted(self._withdrawn),'frontier_revision':self._frontier_revision,
                'reconcile_frontier':[list(state) for state in sorted(self._reconcile_frontier)],
                'reconcile_priority':[[list(state),priority] for state,priority in sorted(self._reconcile_priority.items())],
                'prospective_recipes':[{'identity':r.identity,'start':list(r.start),'goal':list(r.goal),'actions':list(r.actions),'states':[list(x) for x in r.states],'effects':list(r.effects),'sources':list(r.sources),'shadow_credit':r.shadow_credit,'shadow_counter':r.shadow_counter,'rehearsals':r.rehearsals,'counter_sources':list(r.counter_sources),'created_tick':r.created_tick,'expires_tick':r.expires_tick} for _key,r in sorted(self._prospective_recipes.items())],
                'expert_completion_sources':[{'start':list(k[0]),'goal':list(k[1]),'actions':list(k[2]),'context_signature':k[3],'sources':sorted(v),'count':self._expert_completion_counts.get(k,0)} for k,v in sorted(self._expert_completion_sources.items())],
                'prospective_experts':[{'identity':r.identity,'start':list(r.start),'goal':list(r.goal),'actions':list(r.actions),'states':[list(x) for x in r.states],'context_signature':r.context_signature,'witness_recipes':list(r.witness_recipes),'completion_sources':list(r.completion_sources),'completion_count':r.completion_count,'witness_digest':r.witness_digest,'frontier_revision':r.frontier_revision,'probation_passes':r.probation_passes,'active':r.active,'uses':r.uses,'deoptimizations':r.deoptimizations} for _key,r in sorted(self._prospective_experts.items())],
                'prospective_intentions':[{'identity':r.identity,'recipe_identity':r.recipe_identity,'goal':list(r.goal),'actions':list(r.actions),'states':[list(x) for x in r.states],'effects':list(r.effects),'sources':list(r.sources),'cursor':r.cursor,'created_tick':r.created_tick,'last_retrieved_tick':r.last_retrieved_tick,'retrievals':r.retrievals} for _identity,r in sorted(self._prospective_intentions.items())]}
    @classmethod
    def restore(cls,d,resident_authority=None):
        if d.get('schema')!=2:raise ValueError('cognition:checkpoint')
        x=cls(resident_authority)
        for row in d['evidence']:
            key=(_state(row['state']),int(row['action']),_state(row['next_state']),int(row['effect']))
            x._evidence[key]=set(map(int,row['sources']))
            x._evidence_by_state_action.setdefault((key[0],key[1]),set()).add(key)
            x._incoming_by_state.setdefault(key[2],set()).add(key)
            x._actions_by_state.setdefault(key[0],set()).add(key[1])
        x._withdrawn=set(map(int,d['withdrawn']))
        minimum_revision=sum(len(sources) for sources in x._evidence.values())
        x._frontier_revision=int(d.get('frontier_revision',minimum_revision))
        if x._frontier_revision<minimum_revision:raise ValueError('cognition:frontier_revision')
        x._reconcile_frontier=set(_state(state) for state in d.get('reconcile_frontier',()))
        x._reconcile_priority={_state(row[0]):int(row[1]) for row in d.get('reconcile_priority',())}
        if (any(priority<=0 for priority in x._reconcile_priority.values())
                or not set(x._reconcile_priority).issubset(x._reconcile_frontier)):
            raise ValueError('cognition:reconcile_priority')
        rows=d.get('prospective_recipes',())
        if len(rows)>MAX_PROSPECTIVE_RECIPES:raise ValueError('cognition:prospective_bound')
        for row in rows:
            start=_state(row['start']);goal=_state(row['goal']);actions=tuple(map(int,row['actions']))
            states=tuple(_state(state) for state in row['states']);effects=tuple(map(int,row['effects']))
            sources=tuple(sorted(set(map(int,row['sources']))));counter_sources=tuple(sorted(set(map(int,row.get('counter_sources',())))))
            rehearsals=int(row.get('rehearsals',1));shadow_counter=int(row.get('shadow_counter',0))
            created_tick=int(row.get('created_tick',0));expires_tick=int(row.get('expires_tick',created_tick+PROSPECTIVE_TTL_TICKS))
            if (not 2<=len(actions)<=MAX_PLAN_DEPTH or len(states)!=len(actions)+1
                    or len(effects)!=len(actions) or states[0]!=start or states[-1]!=goal
                    or any(action<=0 for action in actions) or not sources
                    or not 1<=rehearsals<=MAX_PROSPECTIVE_REHEARSALS
                    or created_tick<0 or expires_tick!=created_tick+PROSPECTIVE_TTL_TICKS):
                raise ValueError('cognition:prospective_checkpoint')
            edge_sources=[]
            for state,action,next_state,effect in zip(states,actions,states[1:],effects):
                found=x._evidence.get((state,action,next_state,effect))
                if not found:raise ValueError('cognition:prospective_checkpoint')
                edge_sources.append(set(found))
            expected_sources=tuple(sorted(set().union(*edge_sources)))
            expected_support=min(len(found) for found in edge_sources)
            expected_identity=int(_digest('prospective-recipe-v1',[list(start),list(goal),list(actions),
                [list(state) for state in states],list(effects),list(expected_sources)])[:15],16) or 1
            expected_counters=set()
            for (state,action,next_state,effect),observed_sources in x._evidence.items():
                for prior,planned_action,expected,expected_effect in zip(states,actions,states[1:],effects):
                    if prior==state and planned_action==action and (expected!=next_state or (expected_effect>0)!=(effect>0)):
                        expected_counters.update(observed_sources);break
            if (sources!=expected_sources or int(row['shadow_credit'])!=expected_support
                    or int(row['identity'])!=expected_identity
                    or counter_sources!=tuple(sorted(expected_counters))
                    or shadow_counter!=len(counter_sources)):
                raise ValueError('cognition:prospective_checkpoint')
            recipe=ProspectiveRecipeV1(expected_identity,start,goal,actions,states,effects,sources,expected_support,shadow_counter,rehearsals,counter_sources,created_tick,expires_tick)
            if (recipe.start,recipe.goal) in x._prospective_recipes:raise ValueError('cognition:prospective_checkpoint')
            x._prospective_recipes[(recipe.start,recipe.goal)]=recipe
        for row in d.get('expert_completion_sources',()):
            key=(_state(row['start']),_state(row['goal']),tuple(map(int,row['actions'])),int(row['context_signature']))
            sources=set(map(int,row['sources']));count=int(row.get('count',len(sources)))
            if (not sources or any(source<=0 for source in sources) or key in x._expert_completion_sources
                    or count<len(sources) or count<=0):
                raise ValueError('cognition:prospective_expert_completion_checkpoint')
            x._expert_completion_sources[key]=sources;x._expert_completion_counts[key]=count
        expert_rows=d.get('prospective_experts',())
        if len(expert_rows)>MAX_PROSPECTIVE_EXPERTS:raise ValueError('cognition:prospective_expert_bound')
        for row in expert_rows:
            start=_state(row['start']);goal=_state(row['goal']);actions=tuple(map(int,row['actions']))
            states=tuple(_state(state) for state in row['states']);context_signature=int(row['context_signature'])
            witness=tuple(map(int,row['witness_recipes']));completion_sources=tuple(sorted(set(map(int,row['completion_sources']))))
            completion_count=int(row.get('completion_count',len(completion_sources)))
            frontier_revision=int(row['frontier_revision']);probation=int(row.get('probation_passes',0));active=int(row.get('active',0));uses=int(row.get('uses',0));deopts=int(row.get('deoptimizations',0))
            completion_key=(start,goal,actions,context_signature);known=x._expert_completion_sources.get(completion_key,set());known_count=x._expert_completion_counts.get(completion_key,0)
            if (not actions or len(states)!=len(actions)+1 or states[0]!=start or states[-1]!=goal
                    or completion_sources!=tuple(sorted(known)) or completion_count!=known_count or completion_count<EXPERT_MIN_COMPLETIONS
                    or not witness or frontier_revision<=0 or frontier_revision>x._frontier_revision
                    or not 0<=probation<=EXPERT_PROBATION_PASSES or active not in (0,1) or uses<0 or deopts<0):
                raise ValueError('cognition:prospective_expert_checkpoint')
            if frontier_revision==x._frontier_revision:
                frontier=x.prospective_frontier(start,goal,max_candidates=64,depth_slack=2)
                expected_witness=tuple(int(candidate.recipe_identity) for candidate in frontier)
                expected_digest=x._expert_witness_digest(start,goal,context_signature,frontier)
                if witness!=expected_witness or str(row['witness_digest'])!=expected_digest:
                    raise ValueError('cognition:prospective_expert_checkpoint')
            else:
                expected_digest=str(row['witness_digest'])
                if active:raise ValueError('cognition:prospective_expert_stale_active')
            expected_identity=int(_digest('prospective-expert-recipe-v1',[
                list(start),list(goal),list(actions),[list(state) for state in states],context_signature,
                list(witness),list(completion_sources),completion_count,expected_digest,frontier_revision])[:15],16) or 1
            if int(row['identity'])!=expected_identity:raise ValueError('cognition:prospective_expert_checkpoint')
            expert=ProspectiveExpertRecipeV1(expected_identity,start,goal,actions,states,context_signature,witness,
                completion_sources,completion_count,expected_digest,frontier_revision,probation,active,uses,deopts)
            key=(start,goal,context_signature)
            if key in x._prospective_experts:raise ValueError('cognition:prospective_expert_checkpoint')
            x._prospective_experts[key]=expert
        intention_rows=d.get('prospective_intentions',())
        if len(intention_rows)>MAX_PROSPECTIVE_INTENTIONS:raise ValueError('cognition:prospective_intention_bound')
        for row in intention_rows:
            recipe_identity=int(row['recipe_identity']);goal=_state(row['goal'])
            actions=tuple(map(int,row['actions']));states=tuple(_state(state) for state in row['states'])
            effects=tuple(map(int,row['effects']));sources=tuple(sorted(set(map(int,row['sources']))))
            cursor=int(row['cursor']);created_tick=int(row['created_tick'])
            last_retrieved_tick=int(row.get('last_retrieved_tick',0));retrievals=int(row.get('retrievals',0))
            expected_identity=x._intention_identity(recipe_identity,goal,actions,states,effects,sources,created_tick)
            if (recipe_identity<=0 or not 2<=len(actions)<=MAX_PLAN_DEPTH
                    or len(states)!=len(actions)+1 or len(effects)!=len(actions)
                    or states[-1]!=goal or not sources or any(action<=0 for action in actions)
                    or not 0<=cursor<len(actions) or created_tick<0 or last_retrieved_tick<0 or retrievals<0
                    or int(row['identity'])!=expected_identity or expected_identity in x._prospective_intentions):
                raise ValueError('cognition:prospective_intention_checkpoint')
            for state,action,next_state,effect in zip(states,actions,states[1:],effects):
                found=[key for key in x._evidence_by_state_action.get((state,action),())
                       if key[2]==next_state and ((key[3]>0)==(effect>0))]
                if not found:raise ValueError('cognition:prospective_intention_checkpoint')
            intention=ProspectiveIntentionV1(expected_identity,recipe_identity,goal,actions,states,
                effects,sources,cursor,created_tick,last_retrieved_tick,retrievals)
            x._prospective_intentions[expected_identity]=intention;x._index_intention(intention)
        return x
    def digest(self):return _digest('transition-ecology-v1',self.checkpoint())
