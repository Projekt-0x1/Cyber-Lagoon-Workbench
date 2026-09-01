#!/usr/bin/env python3
"""Preserved public Adult before WORLD_CONTEXT / CONTROL_CONTEXT separation."""
from __future__ import annotations

import reference_organism_v2_latent_regime_v1 as _prior
for _name in dir(_prior):
    if not _name.startswith('__') and _name not in {'_prior','_BaseReferenceOrganismV2'}:
        globals()[_name]=getattr(_prior,_name)

_BaseReferenceOrganismV2=_prior.ReferenceOrganismV2

class ReferenceOrganismV2(_BaseReferenceOrganismV2):
    def reason_predictive_reliability_q16(self,reason,source,action,regime=0):
        return self.recursive_causal_experiment.reason_reliability_q16(int(reason),int(source),int(action),int(regime))
    def contextual_reason_predictive_reliability_q16(self,reason,source,action,regime=None):
        resolved=int(self._causal_regime_current if regime is None else regime)
        return self.recursive_causal_experiment.reason_reliability_q16(int(reason),int(source),int(action),resolved)
    def self_reliability_q16(self,action,regime=0):
        return self.recursive_metacontrol.reliability_q16(int(action),int(regime))
    def contextual_self_reliability_q16(self,action,regime=None):
        resolved=int(self._causal_regime_current if regime is None else regime)
        return self.recursive_metacontrol.reliability_q16(int(action),resolved)
