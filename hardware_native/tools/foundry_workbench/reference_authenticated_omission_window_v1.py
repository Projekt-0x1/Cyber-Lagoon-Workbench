from dataclasses import dataclass

@dataclass(frozen=True)
class ExpectedOutcomeV1:
    source_identity:int
    context_signature:int
    horizon_tick:int
    expected_surface:str
    omitted_surface:str

@dataclass(frozen=True)
class ExternalProgressV1:
    source_identity:int
    context_signature:int
    resident_tick:int
    independent_external:bool=True

class AuthenticatedOmissionWindowV1:
    def __init__(self, expected:ExpectedOutcomeV1):
        self.expected=expected
        self.omitted=False
        self.closure_tick=0
    def advance_internal(self, resident_tick:int):
        # Processor chronology is not evidence about the world.
        return self.omitted
    def contact(self, progress:ExternalProgressV1):
        e=self.expected
        if (not progress.independent_external or
            progress.source_identity!=e.source_identity or
            progress.context_signature!=e.context_signature or
            progress.resident_tick<=e.horizon_tick):
            return False
        if self.omitted:
            return False
        self.omitted=True; self.closure_tick=progress.resident_tick
        return True
    def discuss(self):
        return self.expected.omitted_surface if self.omitted else self.expected.expected_surface

class TickOnlyOmissionPredecessor(AuthenticatedOmissionWindowV1):
    def advance_internal(self,resident_tick:int):
        if resident_tick>self.expected.horizon_tick:
            self.omitted=True; self.closure_tick=resident_tick
        return self.omitted
