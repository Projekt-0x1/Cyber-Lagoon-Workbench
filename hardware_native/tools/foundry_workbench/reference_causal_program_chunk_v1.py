from dataclasses import dataclass
from reference_predictive_credit_profile_v1 import PredictiveCreditBankV1,Q
CHUNK_QUORUM=3
@dataclass(frozen=True)
class Chunk:
    identity:int; members:tuple; depth:int; operator:int=0
class CausalChunkBankV1:
    def __init__(self):self.chunks={};self.counts={};self.factors={};self.predictive=PredictiveCreditBankV1(64)
    def bind_factor(self,ident,factor):
        ident=int(ident);factor=int(factor)
        if factor==0:raise RuntimeError('chunk:factor_zero')
        prior=self.factors.get(ident)
        if prior is not None and int(prior)!=factor:raise RuntimeError('chunk:factor_collision')
        self.factors[ident]=factor;return factor
    def factor(self,ident):return self.factors.get(int(ident))
    def factor_checkpoint(self):
        return {'schema':1,'factors':[{'program':k,'factor':int(v)} for k,v in sorted(self.factors.items())]}
    def restore_factor_checkpoint(self,data):
        if int(data.get('schema',0))!=1:raise RuntimeError('chunk:factor_checkpoint_schema')
        factors={}
        for row in data.get('factors',()):
            ident=int(row['program']);factor=int(row.get('factor',0))
            if ident in factors or factor==0:raise RuntimeError('chunk:factor_checkpoint_row')
            factors[ident]=factor
        self.factors=factors
    @staticmethod
    def _count_key(operator,members):
        operator=int(operator);members=tuple(map(int,members))
        return members if operator==0 else (operator,members)
    @staticmethod
    def _count_parts(key):
        if len(key)==2 and isinstance(key[1],tuple):return int(key[0]),tuple(map(int,key[1]))
        return 0,tuple(map(int,key))
    def checkpoint(self):
        """Learned chunk closure; expression factors remain surface-state authority."""
        count_rows=[]
        for key,value in sorted(self.counts.items(),key=lambda item:repr(item[0])):
            operator,members=self._count_parts(key)
            count_rows.append({'operator':operator,'members':list(members),'count':value})
        return {'schema':2,
                'chunks':[{'identity':k,'members':list(v.members),'depth':v.depth,'operator':v.operator}
                          for k,v in sorted(self.chunks.items())],
                'counts':count_rows,
                'predictive':self.predictive.checkpoint()}
    @classmethod
    def restore(cls,data):
        schema=int(data.get('schema',0))
        if schema not in (1,2):raise RuntimeError('chunk:checkpoint_schema')
        bank=cls();chunks={};counts={}
        for row in data.get('chunks',()):
            identity=int(row.get('identity',0));members=tuple(map(int,row.get('members',())))
            depth=int(row.get('depth',0));operator=int(row.get('operator',0)) if schema>=2 else 0
            if (identity<=0 or identity in chunks or not members or
                    cls.ident(members,operator)!=identity or depth<=0):
                raise RuntimeError('chunk:checkpoint_chunk')
            chunks[identity]=Chunk(identity,members,depth,operator)
        for identity,chunk in chunks.items():
            expected=1+max((chunks[m].depth if m in chunks else 0) for m in chunk.members)
            if chunk.depth!=expected:raise RuntimeError('chunk:checkpoint_depth')
        for row in data.get('counts',()):
            members=tuple(map(int,row.get('members',())));count=int(row.get('count',0))
            operator=int(row.get('operator',0)) if schema>=2 else 0;key=cls._count_key(operator,members)
            if (not members or key in counts or cls.ident(members,operator) in chunks or
                    not 0<count<CHUNK_QUORUM):raise RuntimeError('chunk:checkpoint_count')
            counts[key]=count
        predictive=PredictiveCreditBankV1.restore(data.get('predictive',{}))
        if any(identity not in chunks for identity in predictive.rows):
            raise RuntimeError('chunk:checkpoint_predictive')
        bank.chunks=chunks;bank.counts=counts;bank.predictive=predictive
        return bank
    @staticmethod
    def ident(members,operator=0):
        h=1469598103934665603
        operator=int(operator)
        if operator:
            h=((h^0x4F50455241544F52)*1099511628211)&((1<<64)-1)
            h=((h^operator)*1099511628211)&((1<<64)-1)
        for x in members:h=((h^int(x))*1099511628211)&((1<<64)-1)
        return h or 1
    def observe(self,members,start,end,effort_q16,outcome_q16,somatic_q16=0,successor=0,operator=0):
        members=tuple(map(int,members));operator=int(operator);key=self._count_key(operator,members);ident=self.ident(members,operator)
        if ident not in self.chunks:
            count=self.counts.get(key,0)+1
            if count>=CHUNK_QUORUM:
                depth=1+max((self.chunks[m].depth if m in self.chunks else 0) for m in members)
                self.chunks[ident]=Chunk(ident,members,depth,operator);self.counts.pop(key,None)
            else:self.counts[key]=count
        if ident in self.chunks:
            self.predictive.observe_use(ident,start,end,effort_q16)
            self.predictive.observe_return(ident,outcome_q16,somatic_q16,end+1,True)
            if successor:self.predictive.observe_successor(ident,successor,1)
        return self.chunks.get(ident)
    def executable(self,ident):
        p=self.predictive.row(ident)
        return ident in self.chunks and p.accessibility_q16>0 and p.outcome_samples>0 and p.outcome_mean_q16>0
    def decision_depth(self,ident):return 1 if self.executable(ident) else len(self.chunks[ident].members)
    def devalue(self,ident,outcome_q16):self.predictive.observe_return(ident,outcome_q16,-Q//4,self.predictive.row(ident).last_tick+1,True)
