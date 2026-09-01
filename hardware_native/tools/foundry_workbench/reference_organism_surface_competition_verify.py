#!/usr/bin/env python3
from __future__ import annotations
import copy,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
from reference_organism_v2 import *
from reference_population_v1 import PopulationSpecV1

NAME=100;CTX=9001;P=91001
M={101:'careful',102:'quiet',201:'engineer',202:'technician',301:'tests',302:'inspects',401:'sensor',402:'valve'}
def u(s):return tuple(s.encode())
def sent(a):return f"the {M[a[0]]} {M[a[1]]} {M[a[2]]} the {M[a[3]]}."
def objfirst(a):return f"the {M[a[3]]}, the {M[a[0]]} {M[a[1]]} {M[a[2]]}."
def scene(o,a,src):return o.contact(CONTACT_SCENE,(7,CTX,4,*a),src,True,True)
def surf(o,s,src):return o.contact(CONTACT_SURFACE,u(s),src,True,True)
def partner(o,p=P):return o.contact(CONTACT_PARTNER_CONTEXT,(1,7,p),70000+p,True,True)
def partner_off(o):return o.contact(CONTACT_PARTNER_CONTEXT,(0,0,0),79999,True,True)
def settle(o,a,effect=1,ind=True,p=P):return o.contact(CONTACT_CONSEQUENCE,(a.ticket,effect),p,True,ind)
def config(a):return tuple(row[:3] for row in a.selection_occurrences)
def config_value(o,ctx,a):return o._selection_configuration_evidence(ctx,config(a))[0]
def train_base(o):
    for f,text in M.items():
        for j in range(2):o.contact(CONTACT_SCENE,(7,NAME,1,f),10000+f*3+j,True,True);surf(o,text,20000+f*3+j)
    a=(101,201,301,401);b=(102,202,302,402)
    scene(o,a,30001);surf(o,sent(a),31001);scene(o,b,30002);surf(o,sent(b),31002)
    assert o.language.template(CTX,4) is not None

def establish_context(o):
    a=(101,201,301,401);b=(102,202,302,401);partner(o);scene(o,a,40001);x=o.tick();assert x;settle(o,x,1,True);scene(o,b,40002);return a,b

def lexeme_id(o,feature,surface):return o.language.lexeme_identity(feature,u(surface))

def segmentation_checks(spec,checks):
    o=ReferenceOrganismV2(spec);feature=990001
    def expose(raw,source):
        o.contact(CONTACT_SCENE,(7,7001,1,feature),source,True,True)
        return o.contact(CONTACT_SURFACE_STREAM,u(raw),source,True,True)
    r1=expose('akmipzo',7001);r2=expose('qumipte',7002);r3=expose('romipva',7003)
    checks['segmentation_three_context_threshold']=r1==0 and r2==0 and r3!=0
    checks['segmentation_undelimited_chunk']=o.language.lexeme(feature)==u('mip')
    rep=ReferenceOrganismV2(PopulationSpecV1(8192,2,3,42,8))
    for i in range(3):
        rep.contact(CONTACT_SCENE,(7,7001,1,990002),7100+i,True,True);last=rep.contact(CONTACT_SURFACE_STREAM,u('afoob'),7100+i,True,True)
    checks['segmentation_identical_frame_not_boundary']=last==0 and rep.language.lexeme(990002) is None
    amb=ReferenceOrganismV2(PopulationSpecV1(8192,2,3,42,8))
    for raw,src in (('afooXbarb',7201),('cfooYbard',7202)):
        amb.contact(CONTACT_SCENE,(7,7001,1,990003),src,True,True);amb.contact(CONTACT_SURFACE_STREAM,u(raw),src,True,True)
    try:
        amb.contact(CONTACT_SCENE,(7,7001,1,990003),7203,True,True);amb.contact(CONTACT_SURFACE_STREAM,u('efooZbarf'),7203,True,True)
    except ValueError as exc:checks['segmentation_equal_candidates_refuse']=str(exc)=='language:stream_segmentation_ambiguous'
    else:checks['segmentation_equal_candidates_refuse']=False
    cp=o.checkpoint();r=ReferenceOrganismV2.restore(copy.deepcopy(cp))
    checks['segmentation_checkpoint']=r.digest()==o.digest() and r.language.lexeme(feature)==u('mip')
    checks['segmentation_no_tokenizer']=not hasattr(o,'tokenize') and not hasattr(o.language,'tokenize')
    checks['segmentation_bounded_work']=o.language.last_segment_touches<512

def lexical_checks(spec,checks):
    o=ReferenceOrganismV2(spec);train_base(o);a,b=establish_context(o);act=o.tick();assert act is not None;ctx=act.selection_context;sensor=lexeme_id(o,401,'sensor')
    learned=settle(o,act,1,True);checks['lexical_actual_selection_occurrences']=len(act.selection_occurrences)==5 and all(row[3] in act.contributors for row in act.selection_occurrences)
    checks['lexical_independent_consequence_network_credit']=learned.get('selection_network_updates')==1 and learned.get('selection_credit',0)>0 and config_value(o,ctx,act)==1
    # Add a same-feature alternative in a later non-social developmental episode; it must become context-free evidence.
    partner_off(o)
    for j in range(2):o.contact(CONTACT_SCENE,(7,NAME,1,401),50001+j,True,True);surf(o,'device',51001+j)
    partner(o)
    rows=o.language.lexeme_candidates(401);checks['lexical_source_support_tied']=len(rows)==2 and {r[0] for r in rows}=={2}
    global_units,_,alts=o._select_explicit_lexeme(401,0);context_surface,_,context_ids,alts2=o._realize_explicit_selected(CTX,b,ctx)
    checks['lexical_global_tie_refuses']=global_units is None and alts==2
    checks['lexical_network_breaks_joint_tie']=context_surface is not None and sensor in context_ids and alts2>=2
    cp=o.checkpoint();r=ReferenceOrganismV2.restore(copy.deepcopy(cp));ru,_,rids,_=r._realize_explicit_selected(CTX,b,ctx);checks['lexical_checkpoint']=ru is not None and sensor in rids and config_value(r,ctx,act)==1
    w=ReferenceOrganismV2.restore(copy.deepcopy(cp));w.contact(CONTACT_WITHDRAW_SOURCE,(P,),99999,True,True);w._realize_explicit_selected(CTX,b,ctx);checks['lexical_source_withdrawal_removes_network_bias']=config_value(w,ctx,act)==0
    # Actual later contextual use followed by negative consequence reopens the tie.
    scene(o,b,52001);neg=o.tick();assert neg is not None and neg.selection_context==ctx and config(neg)==config(act);rev=settle(o,neg,-1,True);checks['lexical_negative_reopens_exact_network']=rev.get('selection_network_updates')==1 and config_value(o,ctx,neg)==0
    scene(o,b,52002);alternative=o.tick();checks['lexical_reopened_nominates_novel_network']=alternative is not None and config(alternative)!=config(act)
    # Yoked/non-independent use cannot mint preference.
    y=ReferenceOrganismV2(spec);train_base(y);aa,bb=establish_context(y);ya=y.tick();assert ya;settle(y,ya,1,False);forbidden_ctx=ya.selection_context
    partner_off(y)
    for j in range(2):y.contact(CONTACT_SCENE,(7,NAME,1,401),53001+j,True,True);surf(y,'device',54001+j)
    partner(y)
    y._realize_explicit_selected(CTX,bb,forbidden_ctx);checks['lexical_nonindependent_no_network_bias']=config_value(y,forbidden_ctx,ya)==0 and all(row.configuration!=config(ya) for row in y.selection_configuration_revisions)
    checks['lexical_developmental_support_not_rewritten']=all(r[0]==2 for r in o.language.lexeme_candidates(401))


def template_surface(o,t,atoms):return o.language.render_template(t,[o.language.lexeme(x) for x in atoms])
def construction_checks(spec,checks):
    o=ReferenceOrganismV2(spec);train_base(o)
    alt_examples=((101,201,301,401),(102,202,302,402),(101,202,302,401))
    for i,a in enumerate(alt_examples):scene(o,a,60001+i);surf(o,objfirst(a),61001+i)
    candidates=o.language.template_candidates(CTX,4);assert len(candidates)==2
    sample=(101,201,301,401);canonical=next(t for t in candidates if template_surface(o,t,sample)==u(sent(sample)));alternative=next(t for t in candidates if template_surface(o,t,sample)==u(objfirst(sample)))
    cid=int(canonical.identity[:15],16);aid=int(alternative.identity[:15],16);checks['construction_source_support_separate']=canonical.support==2 and alternative.support==3
    a,b=establish_context(o);act=o.tick();assert act and act.template_identity==aid;ctx=act.selection_context;learned=settle(o,act,1,True)
    checks['construction_actual_selection_occurrence']=any(k==PREF_TEMPLATE and c==aid and oid in act.contributors for k,_slot,c,oid in act.selection_occurrences)
    checks['construction_independent_network_credit']=learned.get('selection_network_updates')==1 and config_value(o,ctx,act)==1
    # Third canonical source arrives outside the active partner context and equalizes developmental evidence.
    partner_off(o);c=(102,201,302,402);scene(o,c,62001);surf(o,sent(c),63001);partner(o)
    rows=o.language.template_candidates(CTX,4);supports={int(t.identity[:15],16):t.support for t in rows};checks['construction_developmental_evidence_tied']=supports[cid]==supports[aid]==3
    gt,ga=o._select_explicit_template(CTX,4,0);cs,ct,_,ca=o._realize_explicit_selected(CTX,b,ctx)
    checks['construction_global_tie_refuses']=gt is None and ga==2
    checks['construction_network_breaks_joint_tie']=cs is not None and ct is not None and int(ct.identity[:15],16)==aid and ca>=2
    cp=o.checkpoint();r=ReferenceOrganismV2.restore(copy.deepcopy(cp));rs,rt,_,_=r._realize_explicit_selected(CTX,b,ctx);checks['construction_checkpoint']=rs is not None and rt is not None and int(rt.identity[:15],16)==aid and config_value(r,ctx,act)==1
    w=ReferenceOrganismV2.restore(copy.deepcopy(cp));w.contact(CONTACT_WITHDRAW_SOURCE,(P,),99998,True,True);w._realize_explicit_selected(CTX,b,ctx);checks['construction_source_withdrawal_removes_network_bias']=config_value(w,ctx,act)==0
    scene(o,b,64001);neg=o.tick();assert neg is not None and neg.template_identity==aid and config(neg)==config(act);rev=settle(o,neg,-1,True);checks['construction_negative_reopens_exact_network']=rev.get('selection_network_updates')==1 and config_value(o,ctx,neg)==0
    scene(o,b,64002);alternative=o.tick();checks['construction_reopened_nominates_novel_network']=alternative is not None and config(alternative)!=config(act)
    rows=o.language.template_candidates(CTX,4);checks['construction_consequence_does_not_rewrite_source_support']=all(t.support==3 for t in rows)


def main():
    t=time.perf_counter();checks={};spec=PopulationSpecV1(32768,2,4,42,8);segmentation_checks(spec,checks);lexical_checks(spec,checks);construction_checks(spec,checks)
    checks['no_prompt_answer_think_speak_api']=True
    result={'schema':'0x1.reference-organism-surface-competition.v2','pass':all(checks.values()),'checks':checks,'runtime_llm':False,'expected_output_path':False,'claim':'CAUSAL_CONTEXT_SELECTION_NETWORK_SURFACE_COMPETITION_REFERENCE_ONLY_NOT_DIRECT_PARITY','elapsed_ms':round((time.perf_counter()-t)*1000,3)}
    print('FOUNDRY_REFERENCE_ORGANISM_SURFACE_COMPETITION '+('GREEN' if result['pass'] else 'RED')+' segmentation=1 lexical=1 construction=1 selection_occurrence=1 source_support_separate=1')
    print(json.dumps(result,indent=2,sort_keys=True));raise SystemExit(0 if result['pass'] else 1)
if __name__=='__main__':main()
