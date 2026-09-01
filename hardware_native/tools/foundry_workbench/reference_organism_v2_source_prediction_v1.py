#!/usr/bin/env python3
"""Live Adult with typed source-prediction calibration over separated action outcomes.

The prior outcome-category Adult is preserved in `reference_organism_v2_consequence_split_v1.py`.
Current source epistemics ignore historical reward-contaminated SourceCalibrationV2 rows.
Only explicit grounded source predictions may update the live source calibration table;
action-only recommendations and their valence remain separate pragmatic relations.
"""
from __future__ import annotations

import reference_organism_v2_consequence_split_v1 as _prior
for _name in dir(_prior):
    if not _name.startswith('__') and _name not in {'_prior','_BaseReferenceOrganismV2'}:
        globals()[_name]=getattr(_prior,_name)

from reference_source_prediction_calibration_v1 import SourcePredictionCalibrationV1

_BaseReferenceOrganismV2=_prior.ReferenceOrganismV2
SOURCE_PREDICTION_SCHEMA=1

class ReferenceOrganismV2(_BaseReferenceOrganismV2):
    def __init__(self,population_spec=None):
        super().__init__(population_spec)
        self.source_prediction_calibration=SourcePredictionCalibrationV1()
        self._source_prediction_settlement_plan={}

    def _source_calibration(self,source:int,context:int):
        """Current epistemic source relation: explicit prediction accuracy only."""
        return self.source_prediction_calibration.calibration(int(source),int(context))

    def developmental_source_credibility_q16(self,source:int):
        """Compatibility name; returns only current-target prediction quality, never reputation."""
        context=int(self._source_context_signature())
        if context<=0:return 0
        return self.source_prediction_calibration.quality_q16(int(source),context)

    def _revise_source_calibration(self,source:int,context:int,effect:int):
        """Intercept core source settlement; action-only advice cannot become epistemic credit."""
        key=(int(source),int(context))
        explicit=bool(self._source_prediction_settlement_plan.get(key,False))
        if not explicit:return 0
        direction=int(effect)
        if direction==0:return 0
        wrote=self.source_prediction_calibration.observe(
            int(source),int(context),direction>0,True)
        return self.source_prediction_calibration.evidence_for(source,context) if wrote else 0

    def contact(self,kind,payload,source,authenticated=True,independent=True):
        payload=tuple(payload);kind=int(kind)
        plan={}
        if kind==CONTACT_MOTOR_CONSEQUENCE and payload:
            ticket=int(payload[0])
            motor=next((row for row in self.motor_actions if int(row.ticket)==ticket and not row.settled),None)
            if motor is not None and int(getattr(motor,'source_context',0))>0:
                lookup={int(row.identity):row for row in self.source_assertions}
                for assertion_id in tuple(getattr(motor,'source_assertion_ids',()) or ()):
                    row=lookup.get(int(assertion_id))
                    if row is None:continue
                    plan[(int(row.source),int(motor.source_context))]=bool(tuple(row.predicted_state))
        previous=self._source_prediction_settlement_plan
        self._source_prediction_settlement_plan=plan
        try:return super().contact(kind,payload,source,authenticated,independent)
        finally:self._source_prediction_settlement_plan=previous

    def checkpoint(self):
        data=super().checkpoint();data['source_prediction_calibration_v1']={
            'schema':SOURCE_PREDICTION_SCHEMA,'state':self.source_prediction_calibration.checkpoint()};return data

    @classmethod
    def restore(cls,data):
        out=super().restore(data);row=data.get('source_prediction_calibration_v1')
        if row is None:
            # Do not migrate legacy SourceCalibrationV2 because it may contain reward-based
            # recommendation updates that are not identifiable post hoc.
            out.source_prediction_calibration=SourcePredictionCalibrationV1()
        else:
            if int(row.get('schema',0))!=SOURCE_PREDICTION_SCHEMA:raise ValueError('organism:source-prediction-checkpoint')
            out.source_prediction_calibration=SourcePredictionCalibrationV1.restore(row['state'])
        out._source_prediction_settlement_plan={};return out
