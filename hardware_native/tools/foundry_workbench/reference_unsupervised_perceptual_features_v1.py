#!/usr/bin/env python3
"""Exposure-only discovery of recurrent opaque raw-element conjunction features."""
from __future__ import annotations
from itertools import combinations
from reference_population_v1 import PopulationBankV1,PopulationRecruitmentEcologyV1,PopulationSpecV1,mix64

PAIR_TAG=0xF34701
FEATURE_QUORUM=4


class UnsupervisedPerceptualFeaturesV1:
    """Learn conjunction support from exposure only; no labels, reward or category state."""
    def __init__(self,site_count=32768):
        self.population=PopulationBankV1(PopulationSpecV1(int(site_count),2,3,8,8))
        self.last_touches=0

    @staticmethod
    def pair_token(left:int,right:int)->int:
        a,b=sorted((int(left),int(right)))
        if a<=0 or b<=0 or a==b:raise ValueError('perceptual_feature:pair')
        value=mix64(a^mix64(b^0x9E3779B97F4A7C15))&((1<<63)-1)
        return int(value or 1)

    def _token_sites(self,left:int,right:int):
        token=self.pair_token(left,right);seed=self.population.feature_sites(token);prop=[]
        for site in seed:prop.append(self.population.edge_target_value(site*self.population.spec.fanout))
        return tuple(sorted(set((*seed,*prop))))

    def _feature_signature(self,left:int,right:int):
        token=self.pair_token(left,right)
        return self.population.signature((PAIR_TAG,token))

    def observe_scene(self,elements):
        values=tuple(sorted(set(int(x) for x in elements if int(x)>0)))
        if len(values)<2:return 0
        touched=0
        for left,right in combinations(values,2):
            self.population.prepare((PAIR_TAG,self.pair_token(left,right)));touched+=1
        self.last_touches=touched
        return touched

    def support(self,left:int,right:int)->int:
        sites=self._token_sites(left,right);self.last_touches=len(sites)
        return min((int(self.population.support[site]) for site in sites),default=0)

    def feature(self,left:int,right:int)->int:
        if self.support(left,right)<FEATURE_QUORUM:return 0
        return PopulationRecruitmentEcologyV1.morphology_identity(self._feature_signature(left,right))

    def features_from_patches(self,patches):
        out=[]
        for patch in patches:
            row=tuple(int(x) for x in patch)
            if len(row)!=2:raise ValueError('perceptual_feature:patch')
            feature=self.feature(row[0],row[1])
            if feature:out.append(int(feature))
        return tuple(dict.fromkeys(out))

    def lesion_pair(self,left:int,right:int):
        for site in self._token_sites(left,right):self.population.support[site]=0

    def checkpoint(self):return {'schema':1,'population':self.population.checkpoint()}

    @classmethod
    def restore(cls,data):
        if int(data.get('schema',0))!=1:raise ValueError('perceptual_feature:checkpoint')
        out=cls();out.population=PopulationBankV1.restore(data['population']);return out
