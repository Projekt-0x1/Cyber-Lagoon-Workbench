from dataclasses import dataclass, field
Q=1<<16; PROFILE_CAPACITY=128; SUCCESSOR_CAPACITY=8; ACCESS_GAIN=Q//8; ACCESS_DECAY=Q//32; CONTROL_SUCCESS_QUORUM=2; BACKGROUND_OPPORTUNITY_QUORUM=1; CONTROL_HISTORY_STEP=Q//8

def ema(old,sample,count): return int(sample) if count<=1 else old+(int(sample)-old)//count
@dataclass
class Succ: identity:int; count:int=0; mean_gap_q16:int=0
@dataclass
class ContextStat:
    outcome_samples:int=0; outcome_mean_q16:int=0; somatic_mean_q16:int=0; participated:bool=False
    control_attempts:int=0; control_successes:int=0; background_attempts:int=0; background_successes:int=0; control_history_q16:int=0
    successors:dict=field(default_factory=dict)
    @property
    def controllability_q16(self):
        action_rate=0 if self.control_attempts<=0 else (self.control_successes*Q)//self.control_attempts
        background_rate=0 if self.background_attempts<=0 else (self.background_successes*Q)//self.background_attempts
        return max(0,action_rate-background_rate)
    @property
    def control_ready(self):return self.control_successes>=CONTROL_SUCCESS_QUORUM and self.background_attempts>=BACKGROUND_OPPORTUNITY_QUORUM and self.controllability_q16>=Q//2
    @property
    def control_supported(self):return self.control_ready or self.control_history_q16>=Q//2
    def expected_successor(self,min_count=1):return unique_successor(self.successors,min_count)
def unique_successor(successors,min_count=1):
    if not successors:return 0
    winner=0;best=-1;ties=0
    for row in successors.values():
        if row.count>best:best=row.count;winner=row.identity;ties=1
        elif row.count==best:ties+=1
    return 0 if ties!=1 or best<max(1,int(min_count)) else winner
@dataclass
class Profile:
    structure_id:int; exposures:int=0; accessibility_q16:int=0; outcome_samples:int=0; outcome_mean_q16:int=0
    duration_mean_q16:int=0; duration_abs_error_q16:int=0
    effort_mean_q16:int=0; effort_abs_error_q16:int=0
    prediction_error_q16:int=0; uncertainty_q16:int=0; somatic_mean_q16:int=0
    control_attempts:int=0; control_successes:int=0; background_attempts:int=0; background_successes:int=0; control_history_q16:int=0
    last_context:int=0; last_tick:int=0; successors:dict=field(default_factory=dict); contexts:dict=field(default_factory=dict)
    @property
    def controllability_q16(self):
        action_rate=0 if self.control_attempts<=0 else (self.control_successes*Q)//self.control_attempts
        background_rate=0 if self.background_attempts<=0 else (self.background_successes*Q)//self.background_attempts
        return max(0,action_rate-background_rate)
    @property
    def control_ready(self):return self.control_successes>=CONTROL_SUCCESS_QUORUM and self.background_attempts>=BACKGROUND_OPPORTUNITY_QUORUM and self.controllability_q16>=Q//2
    @property
    def control_supported(self):return self.control_ready or self.control_history_q16>=Q//2
    def expected_successor(self):
        return unique_successor(self.successors)
class PredictiveCreditBankV1:
    def __init__(self,capacity=PROFILE_CAPACITY):
        self.capacity=capacity; self.rows={}; self.capacity_refusals=0
        self.evictions=0; self.last_evicted=0; self.context_members={}
    @staticmethod
    def _retention_key(row):
        # Preserve learned consequence (positive or negative) and demonstrated
        # controllability before mere repetition. Among equally weak rows, old
        # low-exposure matter yields first. No reward-sign preference is encoded.
        return (1 if row.outcome_samples>0 else 0,
                1 if row.control_history_q16>0 else 0,
                1 if row.control_successes>0 else 0,
                row.outcome_samples,row.control_history_q16,row.control_successes,row.exposures,row.last_tick,
                -row.structure_id)
    def row(self,sid):
        sid=int(sid)
        if sid<=0: raise ValueError('structure_id')
        if sid in self.rows:return self.rows[sid]
        if len(self.rows)>=self.capacity:
            victim=min(self.rows.values(),key=self._retention_key)
            for context in tuple(victim.contexts):
                members=self.context_members.get(context)
                if members is None:continue
                members.discard(victim.structure_id)
                if not members:del self.context_members[context]
            self.last_evicted=victim.structure_id;del self.rows[victim.structure_id]
            self.evictions+=1
        self.rows[sid]=Profile(sid);return self.rows[sid]
    def observe_use(self,sid,start,end,effort_q16,context=0):
        if end<start:raise ValueError('chronology')
        r=self.row(sid);r.exposures+=1;r.accessibility_q16=min(Q,r.accessibility_q16+ACCESS_GAIN)
        d=(end-start)*Q;pd=r.duration_mean_q16;r.duration_mean_q16=ema(r.duration_mean_q16,d,r.exposures);r.duration_abs_error_q16=ema(r.duration_abs_error_q16,abs(d-pd),r.exposures)
        e=max(0,int(effort_q16));pe=r.effort_mean_q16;r.effort_mean_q16=ema(r.effort_mean_q16,e,r.exposures);r.effort_abs_error_q16=ema(r.effort_abs_error_q16,abs(e-pe),r.exposures)
        context=int(context);r.last_context=context;r.last_tick=int(end);cs=r.contexts.setdefault(context,ContextStat());cs.participated=True;self.context_members.setdefault(context,set()).add(r.structure_id);self._unc(r);return r
    def candidates(self,context):
        """Rematerialized local participants; never persistent selection authority."""
        members=self.context_members.get(int(context))
        return members if members else ()
    def observe_return(self,sid,outcome_q16,somatic_q16,tick,independent=True,context=0):
        r=self.row(sid)
        if not independent:return r
        pred=r.outcome_mean_q16;r.outcome_samples+=1;r.outcome_mean_q16=ema(r.outcome_mean_q16,outcome_q16,r.outcome_samples);r.prediction_error_q16=ema(r.prediction_error_q16,abs(outcome_q16-pred),r.outcome_samples)
        r.somatic_mean_q16=ema(r.somatic_mean_q16,somatic_q16,r.outcome_samples);r.last_context=int(context);r.last_tick=max(r.last_tick,int(tick));cs=r.contexts.setdefault(int(context),ContextStat());cs.outcome_samples+=1;cs.outcome_mean_q16=ema(cs.outcome_mean_q16,outcome_q16,cs.outcome_samples);cs.somatic_mean_q16=ema(cs.somatic_mean_q16,somatic_q16,cs.outcome_samples);self._unc(r);return r
    @staticmethod
    def _revise_control(row,public_action,independent_return):
        was_ready=row.control_ready
        if public_action:
            row.control_attempts+=1
            if independent_return:row.control_successes+=1
        else:
            row.background_attempts+=1
            if independent_return:row.background_successes+=1
        now_ready=row.control_ready
        if now_ready and not was_ready:row.control_history_q16=Q
        elif now_ready and public_action and independent_return:row.control_history_q16=min(Q,row.control_history_q16+CONTROL_HISTORY_STEP)
        elif public_action!=independent_return:row.control_history_q16=max(0,row.control_history_q16-CONTROL_HISTORY_STEP)
    def observe_control(self,sid,public_action,independent_return,context=None):
        r=self.row(sid)
        cs=None
        if context is not None:
            cs=r.contexts.get(int(context))
            if cs is None or not cs.participated:raise ValueError('credit:context_control_without_participation')
        self._revise_control(r,bool(public_action),bool(independent_return))
        if cs is not None:
            self._revise_control(cs,bool(public_action),bool(independent_return))
        return r
    def observe_successor(self,sid,successor,gap,context=None):
        if successor<=0 or gap<0:raise ValueError('successor')
        r=self.row(sid);rows=r.successors
        if context is not None:
            cs=r.contexts.get(int(context))
            if cs is None or not cs.participated:raise ValueError('credit:context_successor_without_participation')
            rows=cs.successors
        before=unique_successor(rows);s=rows.get(successor)
        if s is None:
            if len(rows)>=SUCCESSOR_CAPACITY:
                victim=min(rows.values(),key=lambda x:(x.count,-x.identity));del rows[victim.identity]
            s=Succ(successor);rows[successor]=s
        s.count+=1;s.mean_gap_q16=ema(s.mean_gap_q16,gap*Q,s.count);mismatch=Q if before not in (0,successor) else 0;r.prediction_error_q16=(7*r.prediction_error_q16+mismatch)//8;self._unc(r);return r
    def contextual_expected_successor(self,sid,context,min_count=1):
        cs=self.row(sid).contexts.get(int(context))
        return 0 if cs is None else cs.expected_successor(min_count)
    def disuse(self,tick):
        for r in self.rows.values():
            if tick>r.last_tick:r.accessibility_q16=max(0,r.accessibility_q16-(tick-r.last_tick)*ACCESS_DECAY)
    def _unc(self,r):
        disp=min(Q,(r.duration_abs_error_q16+r.effort_abs_error_q16)//2);total=sum(x.count for x in r.successors.values());amb=0 if not total else ((total-max(x.count for x in r.successors.values()))*Q)//total;r.uncertainty_q16=max(r.uncertainty_q16,disp,amb)
    def contextual_outcome(self,sid,context):
        r=self.row(sid);c=r.contexts.get(int(context));return r.outcome_mean_q16 if c is None or c.outcome_samples==0 else c.outcome_mean_q16
    def contextual_somatic(self,sid,context):
        r=self.row(sid);c=r.contexts.get(int(context));return r.somatic_mean_q16 if c is None or c.outcome_samples==0 else c.somatic_mean_q16
    def contextual_causal_value(self,sid,context):
        """Price learned consequence by the action's current unique efficacy."""
        r=self.row(sid);c=r.contexts.get(int(context))
        if c is None:return 0
        value=self.contextual_outcome(sid,context)+self.contextual_somatic(sid,context)
        # Fast current contingency and slower earned control history are distinct
        # timescales.  Disconfirmation already decays history in observe_control;
        # retaining its larger surviving value preserves learned coping without
        # letting raw consequence bypass causal evidence.
        efficacy=max(c.controllability_q16,c.control_history_q16)
        magnitude=(abs(value)*efficacy)//Q
        return -magnitude if value<0 else magnitude
    def contextual_control_supported(self,sid,context):
        c=self.row(sid).contexts.get(int(context));return bool(c is not None and c.participated and c.control_supported)
    def contextual_action_supported(self,sid,context):
        c=self.row(sid).contexts.get(int(context))
        if c is None or not c.participated:return False
        if c.background_attempts==0:return c.control_successes>=CONTROL_SUCCESS_QUORUM and c.outcome_samples>0
        return c.control_supported
    def snapshot(self):
        return tuple((k,self.rows[k].exposures,self.rows[k].accessibility_q16,self.rows[k].outcome_mean_q16,self.rows[k].duration_mean_q16,self.rows[k].effort_mean_q16,self.rows[k].prediction_error_q16,self.rows[k].uncertainty_q16,self.rows[k].somatic_mean_q16,self.rows[k].control_attempts,self.rows[k].control_successes,self.rows[k].background_attempts,self.rows[k].background_successes,self.rows[k].controllability_q16,self.rows[k].control_history_q16,tuple((s.identity,s.count,s.mean_gap_q16) for s in sorted(self.rows[k].successors.values(),key=lambda x:x.identity)),tuple((c,cs.outcome_samples,cs.outcome_mean_q16,cs.somatic_mean_q16,cs.participated,cs.control_attempts,cs.control_successes,cs.background_attempts,cs.background_successes,cs.controllability_q16,cs.control_history_q16,tuple((s.identity,s.count,s.mean_gap_q16) for s in sorted(cs.successors.values(),key=lambda x:x.identity))) for c,cs in sorted(self.rows[k].contexts.items()))) for k in sorted(self.rows)),self.capacity_refusals
    def checkpoint(self):
        return {'schema':5,'capacity':self.capacity,'capacity_refusals':self.capacity_refusals,
                'evictions':self.evictions,'last_evicted':self.last_evicted,'rows':[
          {'identity':sid,'exposures':r.exposures,'accessibility':r.accessibility_q16,
           'outcome_samples':r.outcome_samples,'outcome':r.outcome_mean_q16,
           'duration':r.duration_mean_q16,'duration_error':r.duration_abs_error_q16,
           'effort':r.effort_mean_q16,'effort_error':r.effort_abs_error_q16,
           'prediction_error':r.prediction_error_q16,'uncertainty':r.uncertainty_q16,
           'somatic':r.somatic_mean_q16,'control_attempts':r.control_attempts,
           'control_successes':r.control_successes,'background_attempts':r.background_attempts,
           'background_successes':r.background_successes,'control_history':r.control_history_q16,
           'last_context':r.last_context,
           'last_tick':r.last_tick,
           'successors':[{'identity':s.identity,'count':s.count,'gap':s.mean_gap_q16}
                         for s in sorted(r.successors.values(),key=lambda x:x.identity)],
           'contexts':[{'identity':c,'outcome_samples':cs.outcome_samples,
                        'outcome':cs.outcome_mean_q16,'somatic':cs.somatic_mean_q16,
                        'participated':bool(cs.participated),
                        'control_attempts':cs.control_attempts,'control_successes':cs.control_successes,
                        'background_attempts':cs.background_attempts,'background_successes':cs.background_successes,
                        'control_history':cs.control_history_q16,
                        'successors':[{'identity':s.identity,'count':s.count,'gap':s.mean_gap_q16}
                                      for s in sorted(cs.successors.values(),key=lambda x:x.identity)]}
                       for c,cs in sorted(r.contexts.items())]}
          for sid,r in sorted(self.rows.items())]}
    @classmethod
    def restore(cls,data):
        schema=int(data.get('schema',0))
        if schema!=5:raise ValueError('credit:checkpoint_schema')
        capacity=int(data.get('capacity',0));rows=data.get('rows',())
        if capacity<=0 or len(rows)>capacity:raise ValueError('credit:checkpoint_capacity')
        bank=cls(capacity);seen=set()
        bank.capacity_refusals=int(data.get('capacity_refusals',0));bank.evictions=int(data.get('evictions',0));bank.last_evicted=int(data.get('last_evicted',0))
        if min(bank.capacity_refusals,bank.evictions,bank.last_evicted)<0:raise ValueError('credit:checkpoint_counter')
        for item in rows:
            sid=int(item.get('identity',0))
            if sid<=0 or sid in seen:raise ValueError('credit:checkpoint_identity')
            seen.add(sid);r=Profile(sid)
            r.exposures=int(item.get('exposures',0));r.accessibility_q16=int(item.get('accessibility',0));r.outcome_samples=int(item.get('outcome_samples',0));r.outcome_mean_q16=int(item.get('outcome',0))
            r.duration_mean_q16=int(item.get('duration',0));r.duration_abs_error_q16=int(item.get('duration_error',0));r.effort_mean_q16=int(item.get('effort',0));r.effort_abs_error_q16=int(item.get('effort_error',0))
            r.prediction_error_q16=int(item.get('prediction_error',0));r.uncertainty_q16=int(item.get('uncertainty',0));r.somatic_mean_q16=int(item.get('somatic',0));r.control_attempts=int(item.get('control_attempts',0));r.control_successes=int(item.get('control_successes',0));r.background_attempts=int(item.get('background_attempts',0));r.background_successes=int(item.get('background_successes',0));r.control_history_q16=int(item.get('control_history',0));r.last_context=int(item.get('last_context',0));r.last_tick=int(item.get('last_tick',0))
            if (min(r.exposures,r.outcome_samples,r.duration_mean_q16,r.duration_abs_error_q16,
                    r.effort_mean_q16,r.effort_abs_error_q16,r.prediction_error_q16,
                    r.uncertainty_q16,r.control_attempts,r.control_successes,r.background_attempts,r.background_successes,r.last_tick)<0 or
                    not 0<=r.accessibility_q16<=Q or not 0<=r.control_history_q16<=Q or r.control_successes>r.control_attempts or r.background_successes>r.background_attempts):
                raise ValueError('credit:checkpoint_profile')
            for value in item.get('successors',()):
                ident=int(value.get('identity',0));count=int(value.get('count',0));gap=int(value.get('gap',0))
                if ident<=0 or ident in r.successors or count<=0 or gap<0 or len(r.successors)>=SUCCESSOR_CAPACITY:raise ValueError('credit:checkpoint_successor')
                r.successors[ident]=Succ(ident,count,gap)
            for value in item.get('contexts',()):
                context=int(value.get('identity',0));samples=int(value.get('outcome_samples',0))
                if context in r.contexts or samples<0:raise ValueError('credit:checkpoint_context')
                cs=ContextStat(samples,int(value.get('outcome',0)),int(value.get('somatic',0)),bool(value.get('participated',False)),
                               int(value.get('control_attempts',0)),int(value.get('control_successes',0)),
                               int(value.get('background_attempts',0)),int(value.get('background_successes',0)),
                               int(value.get('control_history',0)))
                if (min(cs.control_attempts,cs.control_successes,cs.background_attempts,cs.background_successes)<0
                        or cs.control_successes>cs.control_attempts or cs.background_successes>cs.background_attempts
                        or not 0<=cs.control_history_q16<=Q):raise ValueError('credit:checkpoint_context_control')
                for successor in value.get('successors',()):
                    ident=int(successor.get('identity',0));count=int(successor.get('count',0));gap=int(successor.get('gap',0))
                    if ident<=0 or ident in cs.successors or count<=0 or gap<0 or len(cs.successors)>=SUCCESSOR_CAPACITY:raise ValueError('credit:checkpoint_context_successor')
                    cs.successors[ident]=Succ(ident,count,gap)
                r.contexts[context]=cs
                if cs.participated:bank.context_members.setdefault(context,set()).add(sid)
            bank.rows[sid]=r
        return bank
