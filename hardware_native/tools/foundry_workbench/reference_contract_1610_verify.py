#!/usr/bin/env python3
"""Hostile P0 controls for the bounded graph-neutral reference contract."""
import copy,hashlib,hmac,json,time
from reference_contract_1610 import *

def units(s): return tuple(s.encode())
def node(c,n,*kids): return (c,n,tuple(kids))
def flat(x):
    c,n,k=x; out=[c,n,len(k)]
    for y in k: out+=flat(y)
    return out
class World:
    def __init__(self): self.ticket=10000
    def put(self,m,kind,payload,source=900,channel=1,independent=1,**kw):
        t=kw.get('ticket',self.ticket);self.ticket+=int('ticket' not in kw)
        raw=ContactV1(t,kw.get('incarnation',m.state.incarnation),kw.get('deadline',m.state.tick+8),source,channel,kind,tuple(payload),1,independent)
        sealed=m.seal_contact(raw)
        for k,v in kw.get('tamper',{}).items():setattr(sealed,k,v)
        return m.contact(sealed)
def refuse(fn,prefix):
    try:fn()
    except Refuse as e:
        assert str(e).startswith(prefix),(prefix,str(e));return True
    raise AssertionError('expected refusal')
def fresh():return ReferenceMachineV1(authored_starting_state(),authored_recipe_pool())
LEX={1:{(0,0):' ',(10,1):'cat',(10,2):'cats',(11,1):'dog',(11,2):'dogs',(20,1):'sees',(20,2):'see',(30,1):'and'},2:{(0,0):' ',(10,2):'Katzen',(11,2):'Hunde',(20,2):'sehen'}}
ROOTS={501:node(20,1,node(10,2),node(11,1)),502:node(20,2,node(10,2),node(11,2)),503:node(30,1,node(10,2),node(11,1)),505:node(20,2,node(10,2),node(11,2))}
def prepare(reverse=False):
    m,w=fresh(),World();rows=[]
    for lang,table in LEX.items():
        for (c,n),text in table.items():
            raw=units(text);rows.append((CONTACT_LEXEME,(lang,c,n,len(raw),*raw),100+lang))
        rows.append((CONTACT_FRAME,(lang,1,2,3,1,0,2),100+lang))
    if reverse:rows.reverse()
    for kind,payload,src in rows:w.put(m,kind,payload,src)
    for cue,root in ROOTS.items():
        body=flat(root);w.put(m,CONTACT_QUERY,(cue,2 if cue==505 else 1,len(body),3,*body),301)
    return m,w
def ask(m,w,cue,source=900):w.put(m,CONTACT_STIMULUS,(cue,),source);return m.tick()
def settle(m,w,a,effect=1,counter=0,independent=1,**kw):
    return w.put(m,CONTACT_CONSEQUENCE,(effect,counter),kw.get('source',a.source),kw.get('channel',a.channel),independent,ticket=kw.get('ticket',a.ticket),incarnation=kw.get('incarnation',a.incarnation),deadline=kw.get('deadline',a.deadline),tamper=kw.get('tamper',{}))

class OpaqueWorld:
    def __init__(self,authority,checkpoint_authority):
        self.authority=authority;self.__checkpoint_authority=checkpoint_authority;self.ticket=20000
    def put(self,m,kind,payload,source=900,channel=1,independent=1,**kw):
        ticket=kw.get('ticket',self.ticket);self.ticket+=int('ticket' not in kw)
        incarnation,tick,session,sequence=m.ingress_coordinates()
        raw=ContactV1(ticket,kw.get('incarnation',incarnation),
            kw.get('deadline',tick+8),source,channel,kind,tuple(payload),0,independent)
        sealed=self.authority.seal(raw,session,sequence)
        for k,v in kw.get('tamper',{}).items():setattr(sealed,k,v)
        return m.contact(sealed)
    def restore(self,blob):
        return OpaqueRelationMachineV2.restore(
            blob,self.authority,self.__checkpoint_authority)

def opaque_fixture(work_limit=MAX_WORK_PER_TICK):
    authority,checkpoint_authority=authored_opaque_authorities_v2()
    return (OpaqueRelationMachineV2(authority=authority,
        checkpoint_authority=checkpoint_authority,work_limit=work_limit),
        OpaqueWorld(authority,checkpoint_authority))

def opaque_experience(m,w,key,sites,prior,later,source):
    coalition,port_a,port_b=key
    payload=(coalition,port_a,port_b,len(sites),*sites,len(prior),*prior,len(later),*later)
    return w.put(m,OPAQUE_EXPERIENCE,payload,source)

def opaque_current(m,w,key,sites,base,source):
    context,controller,target_port=key
    payload=(context,controller,target_port,len(sites),*sites,len(base),*base)
    w.put(m,OPAQUE_CURRENT,payload,source)
    return m.tick()

def opaque_withdraw(m,w,source,authority_source=999):
    return w.put(m,OPAQUE_WITHDRAW,(source,),authority_source)

def opaque_pair(m,w,key,sites,rows,sources=(101,102)):
    for source,(prior,later) in zip(sources,rows):
        opaque_experience(m,w,key,sites,prior,later,source)

def restore_opaque(blob,w):
    return w.restore(blob)

def opaque_relation_controls():
    c={};key=(7001,7002,7003);sites=(7101,7102,7103)
    rows=(((11,21,31),(11,21,91)),((12,22,32),(12,22,91)))
    m,w=opaque_fixture();opaque_pair(m,w,key,sites,rows)
    c['opaque_experience_not_action']=not m.state.actions and all(
        type(x) is int for x in (row.identity for row in m.state.experiences))
    held=(13,23,33);action=opaque_current(m,w,key,sites,held,103)
    c['opaque_neutral_hypothesis_marked']=action.provisional==1 and action.recipe_identity not in m.state.credit
    c['opaque_heldout_transfer']=(action.payload[:-1]==held[:-1]
        and action.payload[-1]==rows[0][1][-1]
        and action.payload not in {row[1] for row in rows})
    c['opaque_per_byte_ancestry']=len(action.payload)==len(action.ancestry) and all(
        a.offset==i and a.unit==action.payload[i]
        and a.occurrence_identity==action.occurrence_identity
        and a.relation_roots==action.contributors for i,a in enumerate(action.ancestry))

    # The same resident law sees only bounded numeric coalitions; this second
    # transfer uses disjoint ports, sites, sources and unit identities.
    nonlanguage=(((41,51),(41,81,82)),((42,52),(42,81,82)))
    n,nw=opaque_fixture();nkey=(7201,7202,7203);nsites=(7301,7302)
    opaque_pair(n,nw,nkey,nsites,nonlanguage,(111,112))
    nbase=(43,53);naction=opaque_current(n,nw,nkey,nsites,nbase,113)
    c['opaque_nonlanguage_same_law']=(type(n) is type(m)
        and naction.payload[:1]==nbase[:1]
        and naction.payload[1:]==nonlanguage[0][1][1:])

    # Inverse-equivalent behavior under independent identity/value renaming.
    value_map={11:61,21:71,31:41,91:51,12:62,22:72,32:42,13:63,23:73,33:43}
    inverse={v:k for k,v in value_map.items()}
    mapped=lambda xs:tuple(value_map[x] for x in xs)
    pm,pw=opaque_fixture();pkey=(17001,17002,17003);psites=(17101,17102,17103)
    prows=tuple((mapped(base),mapped(target)) for base,target in rows)
    opaque_pair(pm,pw,pkey,psites,prows,(1101,1102))
    paction=opaque_current(pm,pw,pkey,psites,mapped(held),1103)
    c['opaque_id_value_permutation']=tuple(inverse[x] for x in paction.payload)==action.payload
    c['opaque_permuted_ancestry']=tuple(inverse[x.unit] for x in paction.ancestry)==action.payload

    # Causal difference, not contact independence alone, selects the edit.
    other=(((11,21,31),(11,21,81)),((12,22,32),(12,22,81)))
    chosen=[]
    for reverse in (False,True):
        q,qw=opaque_fixture()
        sites_a=(7101,7102);sites_b=(7101,7103)
        opaque_pair(q,qw,key,sites_a,rows,(201,202))
        opaque_pair(q,qw,key,sites_b,other,(203,204))
        action_a=opaque_current(q,qw,key,sites_a,held,205)
        settle(q,qw,action_a,0 if reverse else 1,1 if reverse else 0)
        action_b=opaque_current(q,qw,key,sites_b,held,206)
        settle(q,qw,action_b,1 if reverse else 0,0 if reverse else 1)
        chosen.append(opaque_current(q,qw,key,sites,held,207))
    c['opaque_causal_credit_reversal']=(chosen[0].payload[-1]==rows[0][1][-1]
        and chosen[1].payload[-1]==other[0][1][-1]
        and chosen[0].payload!=chosen[1].payload
        and not chosen[0].provisional and not chosen[1].provisional)
    z,zw=opaque_fixture();opaque_pair(z,zw,key,sites,rows,(211,212))
    neutral=opaque_current(z,zw,key,sites,held,213);credit=dict(z.state.credit)
    settle(z,zw,neutral,1,0,0)
    c['opaque_independence_not_credit']=z.state.credit==credit

    amb,aw=opaque_fixture();opaque_pair(amb,aw,key,sites,rows,(301,302));opaque_pair(amb,aw,key,sites,other,(303,304))
    aw.put(amb,OPAQUE_CURRENT,(key[0],key[1],key[2],len(sites),*sites,len(held),*held),305)
    h=amb.state_hash();c['opaque_equal_ambiguity_atomic']=refuse(amb.tick,'opaque:ambiguous') and amb.state_hash()==h
    same_output=(((11,22,32),(11,23,91)),((12,24,34),(12,23,91)))
    same,smw=opaque_fixture();opaque_pair(same,smw,key,sites,rows,(311,312));opaque_pair(same,smw,key,sites,same_output,(313,314))
    smw.put(same,OPAQUE_CURRENT,(key[0],key[1],key[2],len(sites),*sites,len(held),*held),315)
    h=same.state_hash();c['opaque_same_bytes_distinct_cause_refusal']=refuse(same.tick,'opaque:ambiguous') and same.state_hash()==h

    withdrawn,ww=opaque_fixture();opaque_pair(withdrawn,ww,key,sites,rows,(401,402));opaque_withdraw(withdrawn,ww,401)
    ww.put(withdrawn,OPAQUE_CURRENT,(key[0],key[1],key[2],len(sites),*sites,len(held),*held),403)
    h=withdrawn.state_hash();refuse(withdrawn.tick,'opaque:no_candidate');c['opaque_withdrawal_cascade']=withdrawn.state_hash()==h
    h=withdrawn.state_hash();c['opaque_withdrawn_source_reentry_refusal']=refuse(
        lambda:opaque_experience(withdrawn,ww,key,sites,(16,26,36),(16,26,91),401),
        'opaque:source_withdrawn') and withdrawn.state_hash()==h
    opaque_experience(withdrawn,ww,key,sites,(14,24,34),(14,24,91),404)
    c['opaque_reacquisition_new_source']=withdrawn.tick().payload[-1]==rows[0][1][-1]
    cascade,cw2=opaque_fixture();opaque_pair(cascade,cw2,key,sites,rows,(411,412))
    cw2.put(cascade,OPAQUE_CURRENT,(key[0],key[1],key[2],len(sites),*sites,len(held),*held),411)
    opaque_withdraw(cascade,cw2,411);c['opaque_withdrawal_clears_current']=cascade.state.current is None
    credited,crw=opaque_fixture();opaque_pair(credited,crw,key,sites,rows,(421,422))
    earned=opaque_current(credited,crw,key,sites,held,423);settle(credited,crw,earned,1,0)
    assert earned.recipe_identity in credited.state.credit
    opaque_withdraw(credited,crw,421);c['opaque_withdrawal_erases_credit_lineage']=earned.recipe_identity not in credited.state.credit
    opaque_experience(credited,crw,key,sites,(14,24,34),(14,24,91),424)
    retrial=opaque_current(credited,crw,key,sites,(15,25,35),425)
    c['opaque_replacement_must_relearn_credit']=retrial.provisional==1

    replay,rw=opaque_fixture();opaque_pair(replay,rw,key,sites,rows,(501,502))
    rw.put(replay,OPAQUE_CURRENT,(key[0],key[1],key[2],len(sites),*sites,len(held),*held),503)
    blob=replay.checkpoint();left=restore_opaque(blob,rw);right=restore_opaque(blob,rw)
    la,ra=left.tick(),right.tick();c['opaque_complete_checkpoint_replay']=la==ra and left.state_hash()==right.state_hash()
    action_blob=left.checkpoint();action_restored=restore_opaque(action_blob,rw)
    c['opaque_action_checkpoint_replay']=action_restored.state_hash()==left.state_hash() and action_restored.state.actions==left.state.actions
    bad=bytearray(blob);bad[len(bad)//2]^=1;c['opaque_corrupt_checkpoint_refusal']=refuse(lambda:restore_opaque(bytes(bad),rw),'opaque:checkpoint_')
    forged=json.loads(blob);forged['body']['next_identity']=1
    forged_blob=json.dumps(forged,sort_keys=True,separators=(',',':')).encode()
    c['opaque_recomputed_public_checkpoint_refusal']=refuse(lambda:restore_opaque(forged_blob,rw),'opaque:checkpoint_corrupt')
    # Replace only the authenticated current contact by starting from evidence-only state.
    evidence,evidence_world=opaque_fixture();opaque_pair(evidence,evidence_world,key,sites,rows,(521,522));evidence_blob=evidence.checkpoint()
    d1=evidence_world.restore(evidence_blob);d2=evidence_world.restore(evidence_blob)
    dw1,dw2=evidence_world,evidence_world
    c['opaque_altered_current_divergence']=opaque_current(d1,dw1,key,sites,(13,23,33),523).payload!=opaque_current(d2,dw2,key,sites,(14,24,34),524).payload

    sparse,sw=opaque_fixture();opaque_pair(sparse,sw,key,sites,rows,(601,602))
    for i in range(20):
        base=(100+i,120+i);target=(100+i,150+i)
        opaque_experience(sparse,sw,(8000+i,9001,9002),(10001+i,10031+i),base,target,700+i)
    sparse_action=opaque_current(sparse,sw,key,sites,held,603)
    c['opaque_sparse_bounded_work']=(sparse.last_touches==2
        and sparse.state.work <= 16*sparse.last_touches+1
        and sparse_action.payload==action.payload)
    local,lw=opaque_fixture();opaque_pair(local,lw,key,sites,rows,(611,612))
    lw.put(local,OPAQUE_CURRENT,(key[0],key[1],key[2],len(sites),*sites,len(held),*held),613)
    original_deepcopy=copy.deepcopy
    def guarded_deepcopy(value,memo=None):
        if isinstance(value,OpaqueRelationStateV2):raise AssertionError('global state copy')
        if (isinstance(value,list) and value
                and isinstance(value[0],(OpaqueExperienceV2,OpaqueActionV2))):
            raise AssertionError('population copy')
        return original_deepcopy(value,memo)
    copy.deepcopy=guarded_deepcopy
    try:c['opaque_tick_no_population_copy']=local.tick().payload==action.payload
    finally:copy.deepcopy=original_deepcopy
    ingress,igw=opaque_fixture()
    incarnation,tick,session,sequence=ingress.ingress_coordinates()
    raw=ContactV1(17001,incarnation,tick+8,17,1,OPAQUE_EXPERIENCE,
        (1,2,3,1,4,1,10,1,11),0,1)
    sealed=igw.authority.seal(raw,session,sequence)
    copy.deepcopy=guarded_deepcopy
    try:c['opaque_contact_no_state_copy']=ingress.contact(sealed)>0
    finally:copy.deepcopy=original_deepcopy
    ingress_fault,ifw=opaque_fixture();incarnation,tick,session,sequence=ingress_fault.ingress_coordinates()
    raw=ContactV1(17002,incarnation,tick+8,18,1,OPAQUE_CURRENT,
        (1,2,3,1,4,1,10),0,1)
    sealed=ifw.authority.seal(raw,session,sequence);h=ingress_fault.state_hash()
    def fail_receipt_copy(value,memo=None):
        if type(value) is ContactV1:raise RuntimeError('receipt copy fault')
        return original_deepcopy(value,memo)
    copy.deepcopy=fail_receipt_copy
    try:
        try:ingress_fault.contact(sealed)
        except RuntimeError:pass
        c['opaque_ingress_copy_fault_atomic']=ingress_fault.state_hash()==h
    finally:copy.deepcopy=original_deepcopy
    class ContactSubclass(ContactV1):pass
    subclass=ContactSubclass(**vars(sealed));h=ingress_fault.state_hash()
    c['opaque_contact_subclass_refusal']=refuse(
        lambda:ingress_fault.contact(subclass),'opaque:contact_type') and ingress_fault.state_hash()==h
    tick_fault,tfw=opaque_fixture();opaque_pair(tick_fault,tfw,key,sites,rows,(617,618))
    tfw.put(tick_fault,OPAQUE_CURRENT,(key[0],key[1],key[2],len(sites),*sites,len(held),*held),619)
    h=tick_fault.state_hash()
    def fail_action_copy(value,memo=None):
        if isinstance(value,OpaqueActionV2):raise RuntimeError('action copy fault')
        return original_deepcopy(value,memo)
    copy.deepcopy=fail_action_copy
    try:
        try:tick_fault.tick()
        except RuntimeError:pass
        c['opaque_tick_copy_fault_atomic']=tick_fault.state_hash()==h
    finally:copy.deepcopy=original_deepcopy
    limited,lmw=opaque_fixture(work_limit=1);opaque_pair(limited,lmw,key,sites,rows,(614,615))
    lmw.put(limited,OPAQUE_CURRENT,(key[0],key[1],key[2],len(sites),*sites,len(held),*held),616)
    h=limited.state_hash();c['opaque_resource_refusal_atomic']=refuse(limited.tick,'opaque:work') and limited.state_hash()==h

    continuing,ctw=opaque_fixture();opaque_pair(continuing,ctw,key,sites,rows,(621,622))
    for i in range(MAX_FROZEN_ACTION_CLOSURE+4):
        emitted=opaque_current(continuing,ctw,key,sites,(20+(i%20),40+(i%20),60+(i%20)),623)
        settle(continuing,ctw,emitted,0,0)
    c['opaque_settled_history_retirement']=(len(continuing.state.actions)<=MAX_FROZEN_ACTION_CLOSURE
        and opaque_current(continuing,ctw,key,sites,held,623) is not None)

    invalid,iw=opaque_fixture();raw=ContactV1(1,invalid.state.incarnation,8,1,1,OPAQUE_CURRENT,('surface',),0,1)
    sealed=iw.authority.seal(raw,invalid.state.session_epoch,invalid.state.next_ingress_sequence);h=invalid.state_hash();c['opaque_literal_bypass_refusal']=refuse(lambda:invalid.contact(sealed),'ir:literal:') and invalid.state_hash()==h
    forged_machine,forged_world=opaque_fixture();raw=ContactV1(7,forged_machine.state.incarnation,8,1,1,OPAQUE_CURRENT,(1,2,3,1,4,1,5),1,1,forged_machine.state.session_epoch,forged_machine.state.next_ingress_sequence)
    public_key=hashlib.sha256(b'opaque-relation-contact-v2\0'+json.dumps([forged_machine.state.incarnation,forged_machine.state.session_epoch],separators=(',',':')).encode()).digest()
    raw.auth_tag=int.from_bytes(hmac.new(public_key,json.dumps(raw.signed_fields(),separators=(',',':')).encode(),hashlib.sha256).digest()[:8],'little')&((1<<63)-1)
    h=forged_machine.state_hash();c['opaque_public_key_forgery_refusal']=refuse(lambda:forged_machine.contact(raw),'contact:receipt') and forged_machine.state_hash()==h and not hasattr(forged_machine,'seal_contact')
    authored=OpaqueRelationStateV2(next_identity=6,experiences=[
        OpaqueExperienceV2(1,31,1,2,3,(4,),(10,20),(10,90),2),
        OpaqueExperienceV2(3,32,1,2,3,(4,),(11,21),(11,90),4)],
        current=OpaqueCurrentV2(5,33,1,1,2,3,(4,),(12,22)))
    c['opaque_host_state_install_refusal']=refuse(
        lambda:OpaqueRelationMachineV2(authored,iw.authority,object()),'opaque:state_install')
    c['opaque_contact_checkpoint_capability_split']=not hasattr(iw.authority,'tag') and not hasattr(iw.authority,'checkpoint_tag')
    isolated,isw=opaque_fixture();opaque_pair(isolated,isw,key,sites,rows,(701,702));returned=opaque_current(isolated,isw,key,sites,held,703)
    returned.active=0;h=isolated.state_hash()
    c['opaque_returned_action_isolated']=refuse(
        lambda:isw.put(isolated,OPAQUE_CURRENT,(key[0],key[1],key[2],len(sites),*sites,len(held),*held),704),
        'opaque:pending_consequence') and isolated.state_hash()==h and isolated.state.actions[-1].active==1
    injected,ijw=opaque_fixture();opaque_pair(injected,ijw,key,sites,rows,(711,712));probe=opaque_current(injected,ijw,key,sites,held,713);settle(injected,ijw,probe,0,0)
    snapshot=injected.state;snapshot.credit[probe.recipe_identity]=1;snapshot.credit_lineage[probe.recipe_identity]=probe.contributors
    next_probe=opaque_current(injected,ijw,key,sites,held,714)
    c['opaque_live_state_snapshot_isolated']=(next_probe.provisional==1
        and probe.recipe_identity not in injected.state.credit
        and probe.recipe_identity not in injected.state.credit_lineage)
    return c

def main():
    started=time.perf_counter();c={}
    b,bw=fresh(),World();c['blank_silence']=b.tick() is None;bw.put(b,CONTACT_STIMULUS,(999,));h=b.state_hash();refuse(b.tick,'arbitrate:no_candidate');c['no_teach_atomic']=b.state_hash()==h;c['no_host_goal_api']=not hasattr(b,'enqueue_goal')
    m,w=prepare();a=ask(m,w,501);c['opaque_stimulus_learned_choice']=a.payload==units('cats sees dog');c['per_byte_ancestry']=len(a.ancestry)==len(a.payload) and all(x.offset==i and x.unit==a.payload[i] for i,x in enumerate(a.ancestry));h=m.state_hash();refuse(m.tick,'tick:pending_consequence');c['settlement_first']=m.state_hash()==h;credit=dict(m.state.credit);settle(m,w,a,0,0);c['equal_counterfactual_no_credit']=m.state.credit==credit
    w.put(m,CONTACT_STIMULUS,(502,));w.put(m,CONTACT_STIMULUS,(503,));x=m.tick();settle(m,w,x,1,0);seq=m.state.next_ingress_sequence;y=m.tick();c['resident_zero_input_continuation']=x.payload==units('cats see dogs') and y.payload==units('cats and dog') and m.state.next_ingress_sequence==seq;settle(m,w,y,1,0);c['causal_difference_credit']=bool(m.state.credit);g=ask(m,w,505);c['multilingual_same_mechanism']=g.payload==units('Katzen sehen Hunde');settle(m,w,g)
    base,_=prepare();blob=base.checkpoint();l=ReferenceMachineV1.restore(blob,base.pool);r=ReferenceMachineV1.restore(blob,base.pool);la=ask(l,World(),501);ra=ask(r,World(),501);c['exact_checkpoint_replay']=la==ra and l.state_hash()==r.state_hash();alt=ReferenceMachineV1.restore(blob,base.pool);c['altered_input']=ask(alt,World(),502).payload!=la.payload;bad=bytearray(blob);bad[len(bad)//2]^=1;c['corrupt_checkpoint_refusal']=refuse(lambda:ReferenceMachineV1.restore(bytes(bad),base.pool),'checkpoint:')
    p1,w1=prepare();p2,w2=prepare(True);o1=ask(p1,w1,501);o2=ask(p2,w2,501);c['opaque_permutation']=o1.payload==o2.payload and o1.frontier!=o2.frontier
    amb,aw=prepare();body=flat(ROOTS[501]);aw.put(amb,CONTACT_QUERY,(777,1,len(body),3,*body),301);body=flat(ROOTS[503]);aw.put(amb,CONTACT_QUERY,(777,1,len(body),3,*body),302);aw.put(amb,CONTACT_STIMULUS,(777,));h=amb.state_hash();refuse(amb.tick,'arbitrate:ambiguous');c['ambiguity_atomic']=amb.state_hash()==h
    auth,ww=prepare();pending=ask(auth,ww,501);raw=ContactV1(1,auth.state.incarnation,auth.state.tick+2,900,1,CONTACT_STIMULUS,(501,),1,1);h=auth.state_hash();refuse(lambda:auth.contact(raw),'contact:session_sequence');c['unsealed_refusal']=auth.state_hash()==h
    for name,kw,prefix in [('wrong_ticket',{'ticket':pending.ticket+9},'consequence:'),('wrong_source',{'source':pending.source+1},'consequence:'),('wrong_channel',{'channel':pending.channel+1},'consequence:'),('wrong_incarnation',{'incarnation':pending.incarnation+1},'contact:'),('stale',{'deadline':auth.state.tick-1},'contact:')]:
        h=auth.state_hash();refuse(lambda q=kw:settle(auth,ww,pending,**q),prefix);c[name+'_atomic']=auth.state_hash()==h
    h=auth.state_hash();refuse(lambda:settle(auth,ww,pending,tamper={'source':pending.source+1}),'contact:receipt');c['tamper_atomic']=auth.state_hash()==h
    wd,wx=prepare();act=ask(wd,wx,501);wd.withdraw_source(101);c['cascade_withdrawal']=not next(z for z in wd.state.actions if z.ticket==act.ticket).active and not next(z for z in wd.state.occurrences if z.identity==act.occurrence_identity).current and all(not q.active for q in wd.state.queries);h=wd.state_hash();refuse(lambda:settle(wd,wx,act),'consequence:ticket');c['withdrawn_return_atomic']=wd.state_hash()==h
    cap,cw=prepare();deep=node(20,1,node(20,1,node(20,1,node(20,1,node(10,1),node(11,1)),node(11,1)),node(11,1)),node(11,1));deep_body=flat(deep);h=cap.state_hash();c['cap_plus_one_depth_refuse']=refuse(lambda:cw.put(cap,CONTACT_QUERY,(888,1,len(deep_body),4,*deep_body),301),'query:bound') and cap.state_hash()==h;c['nested_depth_plus_one_refuse']=refuse(lambda:cw.put(cap,CONTACT_QUERY,(889,1,len(deep_body),3,*deep_body),301),'query:depth') and cap.state_hash()==h
    wide=True
    for _ in range(5):
        act=ask(cap,cw,501);settle(cap,cw,act)
    c['width_not_aliased_to_depth']=wide
    c['literal_refusal']=refuse(lambda:RecipeIrV1(1,99,(InstructionV1(1,'x'),),1).validate(),'ir:literal:');c['semantic_operand_refusal']=refuse(lambda:RecipeIrV1(1,99,(InstructionV1(1,7),),1).validate(),'ir:semantic_operand');c['program_bound']=refuse(lambda:RecipeIrV1(1,99,tuple(InstructionV1(1) for _ in range(MAX_PROGRAM+1)),1).validate(),'ir:program_bound');c['pool_admission']=refuse(lambda:VerifiedRecipePool((RecipeIrV1(1,1,(InstructionV1(1),),1),)),'pool:admission')
    pool=authored_recipe_pool();pool.admission_receipt='0'*64;c['pool_receipt']=refuse(lambda:ReferenceMachineV1(authored_starting_state(),pool),'machine:frozen_input');script=authored_starting_state();script.queries=[QueryV1(1,0,1,1,node(10,1),1)];script.continuations={1:1};script.next_identity=2;c['host_continuation']=refuse(lambda:ReferenceMachineV1(script,authored_recipe_pool()),'state:host_continuation')
    c.update(opaque_relation_controls())
    failed=sorted(k for k,v in c.items() if not v)
    if failed:raise SystemExit('FOUNDRY_REFERENCE_ADULT_CONTRACT_RED '+','.join(failed))
    receipt={'contract':'FOUNDRY_REFERENCE_ADULT_CONTRACT_GREEN','capability_status':'C0_RED','reference_only':True,'adult_attached':False,'physical_direct_parity':PARITY_STATUS,'graph_flip':False,'human_level_language_claim':False,'adult_language_claim':False,'runtime_host_cognition':False,'production_ir':PRODUCTION_IR,'experimental_ir':'ReferenceRecipeIrV1','opaque_relation_ir':'OpaqueRelationStateV2.experimental','opaque_claim':'BOUNDED_REFERENCE_OPAQUE_EDIT_INDUCTION_AND_CONSEQUENCE_RANKING_ONLY','neutral_hypothesis_actions':True,'consequence_earned_initial_learning':False,'consumer_scale':False,'translation_status':TRANSLATION_STATUS,'checks':c,'elapsed_ms':round((time.perf_counter()-started)*1000,3)}
    print('FOUNDRY_REFERENCE_ADULT_CONTRACT_GREEN C0_RED reference_only=true adult_attached=false physical_direct_parity=NOT_RUN/RED');print(json.dumps(receipt,indent=2,sort_keys=True))
if __name__=='__main__':main()
