from dataclasses import dataclass, field
Q=1<<16
@dataclass
class Row:
    action:int=0; success:int=0; background:int=0; background_success:int=0; value:int=0
    def control(self):
        if self.action<2 or self.background<1:return False
        ar=self.success/self.action;br=self.background_success/self.background
        return ar-br>=0.5
@dataclass
class Program:
    identity:int; surface:bytes; outcomes:dict=field(default_factory=dict)
class OutcomeSpecificActionCreditV1:
    def __init__(self):self.programs={}
    def add(self,pid,surface):self.programs[pid]=Program(int(pid),bytes(surface));return self.programs[pid]
    def action_return(self,pid,outcome,value=Q):
        p=self.programs[pid];r=p.outcomes.setdefault(int(outcome),Row());r.action+=1;r.success+=value>0;r.value=value
    def background(self,outcome,occurs):
        for p in self.programs.values():
            # A free outcome creates a counterfactual row but does not overwrite
            # the program's distinct learned outcome rows.
            r=p.outcomes.setdefault(int(outcome),Row());r.background+=1;r.background_success+=bool(occurs)
    def background_non_event(self,pid,outcome):
        r=self.programs[pid].outcomes.setdefault(int(outcome),Row());r.background+=1
    def choose(self):
        supported=[]
        for p in self.programs.values():
            best=max((r.value for r in p.outcomes.values() if r.control()),default=None)
            if best is not None:supported.append((best,p.identity))
        if not supported:return 0
        m=max(x[0] for x in supported);ids=[pid for value,pid in supported if value==m]
        return ids[0] if len(ids)==1 else 0
class GenericOutcomePredecessor(OutcomeSpecificActionCreditV1):
    def background(self,outcome,occurs):
        # Rejected predecessor: a free positive event degrades every currently
        # learned outcome as if all consequences were interchangeable.
        for p in self.programs.values():
            for r in p.outcomes.values():r.background+=1;r.background_success+=bool(occurs)
