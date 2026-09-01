#!/usr/bin/env python3
from __future__ import annotations
import hashlib,http.client,json,os,select,subprocess,sys,tempfile,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))

from reference_language_mastery_adult_v1 import LanguageMasteryAdultV1,AdultStateV1
from reference_language_mastery_contact_adapter_v1 import LanguageMasteryContactAdapterV1,CONTACT_UTTERANCE
from reference_predictive_credit_profile_v1 import Q


def emit_all(adult,pid):
    expr=adult.expression(pid);out=[]
    while True:
        plan=expr.emit()
        if plan is None:break
        out.append(plan.value)
        if not expr.reafference(plan,plan.value):raise AssertionError('chat:reafference')
        if len(out)>10000:raise AssertionError('chat:nontermination')
    return tuple(out),expr


def settle_internal(adult,state=AdultStateV1(),limit=128):
    """Workbench observer: advance opaque resident work to action/quiescence."""
    for ticks in range(1,limit+1):
        if not adult.internal_work_pending():return 0,ticks-1
        chosen=adult.internal_tick(state)
        if chosen:return chosen,ticks
    raise AssertionError('chat:internal_nonconvergence')


def main():
    started=time.perf_counter();adult=LanguageMasteryAdultV1();contact=LanguageMasteryContactAdapterV1(adult)
    CLAUSE=9001;JOIN=9101;DEF=0xD3F
    A1,A2,G1,G2,V1,V2,O1,O2,LABEL_A,LABEL_B=101,102,201,202,301,302,401,402,501,502
    names={A1:'careful',A2:'quiet',G1:'engineer',G2:'technician',V1:'tests',V2:'inspects',O1:'sensor',O2:'valve',LABEL_A:'gauge',LABEL_B:'tap'}
    for concept,text in names.items():
        raw=tuple(text.encode());adult.observe_surface_item(concept,raw,1000+concept);adult.observe_surface_item(concept,raw,2000+concept)
    x=(A1,G1,V1,O1);y=(A2,G2,V2,O2)
    assert adult.observe_surface_construction(CLAUSE,x,tuple(b'the careful engineer tests the sensor.'),3001)
    assert adult.observe_surface_construction(CLAUSE,y,tuple(b'the quiet technician inspects the valve.'),3002)
    l1=adult.leaf(CLAUSE,x);l2=adult.leaf(CLAUSE,y)

    # Mid-conversation declarative lexical teaching. Earlier experience has
    # learned a generic two-port surface relation and a directional dependency;
    # the novel alias itself enters only as raw bytes, with no new concept ID.
    assert adult.observe_surface_construction(DEF,(O1,LABEL_A),tuple(b'sensor aka gauge'),3901)
    assert adult.observe_surface_construction(DEF,(O2,LABEL_B),tuple(b'valve aka tap'),3902)
    adult.language.observe_dependency(DEF,0,1,3911);adult.language.observe_dependency(DEF,0,1,3912)
    alias_identity=contact.contact(CONTACT_UTTERANCE,tuple(b'sensor aka dax'),4001)
    assert alias_identity!=0 and contact.settle_provisional(alias_identity,4100,+1,False) is None
    assert all(row[1]!=tuple(b'dax') for row in adult.language.lexeme_candidates(O1))
    assert contact.settle_provisional(alias_identity,4101,+1,True)==tuple(b'dax')
    assert contact.settle_provisional(alias_identity,4102,+1,True)==tuple(b'dax')
    taught_heldout=adult.leaf(CLAUSE,(A1,G2,V1,O1))
    assert b'dax' in bytes(taught_heldout.surface)
    raw_scene=contact.contact(CONTACT_UTTERANCE,tuple(taught_heldout.surface),4501)
    assert raw_scene!=0
    raw_binding=contact.scenes[raw_scene]
    raw_taught=adult.leaf(raw_binding.context,raw_binding.atoms)
    assert raw_taught.identity==taught_heldout.identity

    l3=adult.leaf(CLAUSE,(A2,G1,V1,O2));l4=adult.leaf(CLAUSE,(A1,G2,V2,O1))
    assert adult.observe_join(JOIN,l1,l2,5001)
    assert adult.observe_join(JOIN,l2,l3,5002)
    pair_left=adult.compose(JOIN,raw_taught,l2)
    pair_right=adult.compose(JOIN,l3,l4)

    p_left=p_right=None
    for _ in range(3):
        p_left=adult.experience_program((taught_heldout.identity,l2.identity),pair_left,Q//2,Q//16,0xD001,Q//8,True)
        p_right=adult.experience_program((l3.identity,l4.identity),pair_right,Q//2,Q//16,0xD001,Q//8,True)
    recursive=adult.compose(JOIN,p_left.identity,p_right.identity)
    p_recursive=None
    for _ in range(3):p_recursive=adult.experience_program((p_left.identity,p_right.identity),recursive,3*Q//4,Q//8,0xD001,Q//3,True)
    SHORT=0xD100;adult.experience_atomic_program(SHORT,l1,Q//2,Q//16,0xD001,Q//16,True)
    for _ in range(5):
        adult.experience_choice(p_recursive.identity,3*Q//4,Q//8,0xD001,Q//3,6,True)
        adult.experience_choice(SHORT,Q//4,0,0xD001,Q//16,1,True)
    deep_choice=adult.choose(AdultStateV1())
    urgent_choice=adult.choose(AdultStateV1(urgency_q16=Q,pressure_q16=Q))

    # Conversational correction is ordinary consequence revision, not supervised
    # answer replacement. A yoked correction cannot move preference; independent
    # corrective consequences can.
    CORR=0xC011;P_OLD=0xC101;P_NEW=0xC102
    adult.experience_atomic_program(P_OLD,taught_heldout,Q//2,Q//16,CORR,Q//8,True)
    adult.experience_atomic_program(P_NEW,l4,Q//4,0,CORR,Q//8,True)
    for _ in range(3):adult.experience_choice(P_OLD,Q//2,0,CORR,Q//8,2,True)
    for _ in range(2):adult.experience_choice(P_NEW,Q//4,0,CORR,Q//8,2,True)
    before_correction=adult.choose(AdultStateV1())
    adult.credit.observe_return(P_OLD,-8*Q,0,adult._tick+1,False,CORR)
    after_yoked=adult.choose(AdultStateV1())
    for _ in range(6):
        adult.experience_choice(P_OLD,-Q,-Q//8,CORR,Q//8,2,True)
        adult.experience_choice(P_NEW,3*Q//4,Q//16,CORR,Q//8,2,True)
    after_correction=adult.choose(AdultStateV1())

    # Learned partner-sensitive pragmatic language in the same continuing Adult.
    SAY=0x5101;EXPLICIT=0x5102;COMPACT=0x5103;ASK=0x5104
    for concept,text in ((EXPLICIT,'the sensor moved to locker two.'),(COMPACT,"it's there."),(ASK,'where do you think it is?')):
        raw=tuple(text.encode());adult.observe_surface_item(concept,raw,10000+concept);adult.observe_surface_item(concept,raw,20000+concept)
        assert adult.observe_surface_construction(SAY,(concept,),raw,30000+concept)
    explicit=adult.leaf(SAY,(EXPLICIT,));compact=adult.leaf(SAY,(COMPACT,));ask=adult.leaf(SAY,(ASK,))
    P_EXPLICIT=0xE101;P_COMPACT=0xE102;P_ASK=0xE103
    OBJ=0x7001;SA=0x7101;SB=0x7102;SU=0x7103;AA=0x7201;AB=0x7202;PA=0x7301;PB=0x7302;PU=0x7303
    for n in range(3):
        adult.observe_social_contact(PA,OBJ,SA,AB,0x8000+n);adult.observe_social_behavior(PA,OBJ,AA,True)
        adult.observe_social_contact(PB,OBJ,SB,AB,0x8100+n);adult.observe_social_behavior(PB,OBJ,AB,True)
    for n,action in enumerate((AA,AB,AA,AB)):
        adult.observe_social_contact(PU,OBJ,SU,AB,0x8200+n);adult.observe_social_behavior(PU,OBJ,action,True)
    for cycle in range(4):
        for index,(pid,root,outcome,agent,state,effort,duration) in enumerate((
            (P_EXPLICIT,explicit,3*Q//4,PA,SA,Q//5,3),(P_COMPACT,compact,-Q//2,PA,SA,Q//16,1),(P_ASK,ask,-Q//4,PA,SA,Q//8,2),
            (P_COMPACT,compact,3*Q//4,PB,SB,Q//16,1),(P_EXPLICIT,explicit,Q//8,PB,SB,Q//5,3),(P_ASK,ask,-Q//4,PB,SB,Q//8,2),
            (P_ASK,ask,3*Q//4,PU,SU,Q//8,2),(P_COMPACT,compact,-Q//4,PU,SU,Q//16,1),(P_EXPLICIT,explicit,-Q//4,PU,SU,Q//5,3))):
            adult.observe_social_contact(agent,OBJ,state,AB,0x880000+cycle*16+index)
            adult.experience_atomic_program(pid,root,Q//4,0,None,effort,True)
            adult.experience_choice(pid,outcome,Q//16 if outcome>0 else -Q//16,None,effort,duration,True)
    adult.observe_social_contact(PA,OBJ,SA,AB,0x8301);stale_choice=adult.choose(AdultStateV1())
    adult.observe_social_contact(PB,OBJ,SB,AB,0x8302);current_choice=adult.choose(AdultStateV1())
    adult.observe_social_contact(PU,OBJ,SU,AB,0x8303);unknown_choice=adult.choose(AdultStateV1())

    # Two independently learned referents share one surface.  Contact preserves
    # both lawful structural inversions; it does not let the membrane pick one.
    # A clarification program becomes useful only through returned consequence
    # in that sparse competition, then may emerge on one later contact-free tick.
    for concept,source in ((O1,0x8A01),(O1,0x8A02),(O2,0x8A03),(O2,0x8A04)):
        adult.observe_surface_item(concept,tuple(b'probe'),source)
    ambiguous_probe=tuple(b'the careful engineer tests the probe.')
    pre_gap_checkpoint=json.loads(json.dumps(adult.checkpoint()))
    training_gap=contact.contact(CONTACT_UTTERANCE,ambiguous_probe,0x8A10)
    training_gap_context=adult._current_selection_context
    training_gap_first_choice=adult.choose(AdultStateV1())
    adult.experience_choice(P_ASK,Q//8,0,None,Q//8,2,True)
    adult.experience_choice(P_ASK,Q//8,0,None,Q//8,2,True)
    weak_gap=contact.contact(CONTACT_UTTERANCE,ambiguous_probe,0x8A11)
    weak_gap_first_choice=adult.choose(AdultStateV1())
    weak_gap_question,weak_gap_ticks=settle_internal(adult)
    contact.contact(CONTACT_UTTERANCE,ambiguous_probe,0x8A12)
    for _ in range(3):
        adult.experience_choice(P_ASK,3*Q//4,Q//16,None,Q//8,2,True)
    learned_gap=contact.contact(CONTACT_UTTERANCE,ambiguous_probe,0x8A13)
    learned_gap_first_choice=adult.choose(AdultStateV1())
    pending_checkpoint=json.loads(json.dumps(adult.checkpoint()))
    learned_gap_question,learned_gap_ticks=settle_internal(adult)
    pending_resumed=LanguageMasteryAdultV1.restore(pending_checkpoint)
    pending_resumed_first_choice=pending_resumed.choose(AdultStateV1())
    pending_resumed_question,pending_resumed_ticks=settle_internal(pending_resumed)
    pending_resumed_extra_tick=pending_resumed.internal_tick(AdultStateV1())
    pending_lesion=json.loads(json.dumps(pending_checkpoint))
    for row in pending_lesion['selection_credit']['rows']:
        for context_row in row['contexts']:
            if int(context_row['identity'])==training_gap_context:
                context_row['participated']=False
    pending_lesioned=LanguageMasteryAdultV1.restore(pending_lesion)
    pending_lesion_question,pending_lesion_ticks=settle_internal(pending_lesioned)

    # Destructive causal audit: identical current ambiguity must not collapse
    # development, controllability, recent consequence, and current state into
    # one universal transition.
    naive_gap=LanguageMasteryAdultV1.restore(pre_gap_checkpoint)
    naive_contact=LanguageMasteryContactAdapterV1(naive_gap)
    naive_contact.contact(CONTACT_UTTERANCE,ambiguous_probe,0x8A20)
    naive_question,naive_ticks=settle_internal(naive_gap)
    yoked_gap=LanguageMasteryAdultV1.restore(pre_gap_checkpoint)
    yoked_contact=LanguageMasteryContactAdapterV1(yoked_gap)
    yoked_contact.contact(CONTACT_UTTERANCE,ambiguous_probe,0x8A21)
    for _ in range(4):
        yoked_gap.experience_choice(P_ASK,3*Q//4,Q//16,None,Q//8,2,False)
    yoked_contact.contact(CONTACT_UTTERANCE,ambiguous_probe,0x8A22)
    yoked_question,yoked_ticks=settle_internal(yoked_gap)
    discouraged=LanguageMasteryAdultV1.restore(pending_checkpoint)
    for _ in range(4):
        discouraged.experience_choice(P_ASK,-Q,-Q//8,None,Q//8,2,True)
    discouraged_contact=LanguageMasteryContactAdapterV1(discouraged)
    discouraged_contact.contact(CONTACT_UTTERANCE,ambiguous_probe,0x8A23)
    discouraged_question,discouraged_ticks=settle_internal(discouraged)
    pressured=LanguageMasteryAdultV1.restore(pending_checkpoint)
    pressured_question,pressured_ticks=settle_internal(
        pressured,AdultStateV1(urgency_q16=Q,pressure_q16=Q))

    # Long-horizon pressure at fixed capacities. Fill the language-selection bank
    # with consequence-cold one-use programs; learned consequence-backed chat
    # programs must survive by retention merit rather than capacity growth.
    credit_capacity=adult.credit.capacity;evictions_before=adult.credit.evictions
    filler=0xF00000
    for n in range(credit_capacity+64):
        pid=filler+n
        adult.programs.bind_factor(pid,adult.programs.factor(SHORT))
        tick=adult._advance();adult.credit.observe_use(pid,tick,tick+1,Q//64,0xF000+n);adult._tick=tick+2
    language_evictions=adult.credit.evictions-evictions_before
    post_deep=adult._probe_choice(0xD001,AdultStateV1())
    post_corrected=adult._probe_choice(CORR,AdultStateV1())
    adult.observe_social_contact(PA,OBJ,SA,AB,0x8311);post_stale=adult.choose(AdultStateV1())
    adult.observe_social_contact(PB,OBJ,SB,AB,0x8312);post_current=adult.choose(AdultStateV1())
    adult.observe_social_contact(PU,OBJ,SU,AB,0x8313);post_unknown=adult.choose(AdultStateV1())

    # Saturate other-referenced prediction with unrelated agents as well. A useful
    # focal social regularity must survive or the shared retention law is wrong.
    focal_before=adult.predict_social_action(PA,OBJ);social_evictions_before=adult.prospection.predictive.evictions
    for n in range(adult.prospection.predictive.capacity+64):
        agent=0x740000+n;subject=0x750000+n;state=0x760000+n
        adult.observe_social_contact(agent,subject,state,0xA00000+n,0x900000+n)
        adult.observe_social_behavior(agent,subject,0xA00000+n,True)
    social_evictions=adult.prospection.predictive.evictions-social_evictions_before
    focal_after=adult.predict_social_action(PA,OBJ)
    adult.observe_social_contact(PA,OBJ,SA,AB,0x8321);post_social_stale=adult.choose(AdultStateV1())
    adult.observe_social_contact(PB,OBJ,SB,AB,0x8322);post_social_current=adult.choose(AdultStateV1())
    adult.observe_social_contact(PU,OBJ,SU,AB,0x8323);post_social_unknown=adult.choose(AdultStateV1())

    # A recognized utterance must itself reinstate the learned current
    # occurrence. The body supplies raw bytes only; it cannot name the
    # situation or candidate response. Bind one response to the stable learned
    # prompt occurrence, then require a fresh process-like restore and raw
    # contact to recover that choice.
    P_CONTACT_REPLY=0xE104
    for _ in range(3):
        adult.experience_atomic_program(
            P_CONTACT_REPLY,l4,3*Q//4,Q//16,taught_heldout.identity,Q//8,True)

    # A terminal stop is loss of active occurrence, not rebirth. Preserve learned
    # connectivity/sufficient statistics, then require fresh body contact to
    # reconstruct the current situation before any public action can be selected.
    checkpoint=json.loads(json.dumps(adult.checkpoint()))
    resumed=LanguageMasteryAdultV1.restore(checkpoint)
    checkpoint_exact=resumed.checkpoint()==checkpoint
    resumed_without_contact=resumed.choose(AdultStateV1())
    resumed_contact=LanguageMasteryContactAdapterV1(resumed)
    resumed_raw_scene=resumed_contact.contact(
        CONTACT_UTTERANCE,tuple(taught_heldout.surface),0x8330)
    resumed_raw_choice=resumed.choose(AdultStateV1())
    claude_bin=os.environ.get('AGI_CLAUDE_BIN','')
    claude_cli=claude_cli_resumed=claude_cli_quiet=None
    with tempfile.TemporaryDirectory() as directory:
        checkpoint_path=Path(directory)/'adult.json'
        checkpoint_path.write_text(json.dumps(checkpoint))
        terminal=Path(__file__).with_name('reference_language_mastery_terminal_v1.py')
        quiet_process_outputs=[]
        for idle_ms in (1,37):
            quiet_checkpoint_path=Path(directory)/f'quiet-adult-{idle_ms}.json'
            quiet_checkpoint_path.write_text(json.dumps(pending_checkpoint))
            quiet_process=subprocess.Popen(
                (sys.executable,str(terminal),'--resume',str(quiet_checkpoint_path),
                 '--idle-ms',str(idle_ms)),stdin=subprocess.PIPE,stdout=subprocess.PIPE)
            try:
                quiet_ready=select.select((quiet_process.stdout,),(),(),0.5)[0]
                quiet_process_outputs.append(
                    quiet_process.stdout.readline() if quiet_ready else b'')
            finally:
                quiet_process.terminate();quiet_process.wait(timeout=2)
        first_sitting=subprocess.run(
            (sys.executable,str(terminal),'--resume',str(checkpoint_path)),
            input=bytes(taught_heldout.surface)+b'\n',capture_output=True,check=True)
        second_sitting=subprocess.run(
            (sys.executable,str(terminal),'--resume',str(checkpoint_path)),
            input=bytes(taught_heldout.surface)+b'\n',capture_output=True,check=True)
        process_checkpoint=LanguageMasteryAdultV1.restore(
            json.loads(checkpoint_path.read_text()))
        expected_terminal_output=bytes(l4.surface)+b'\n'
        gateway=Path(__file__).with_name('reference_language_mastery_claude_gateway_v1.py')
        gateway_process=subprocess.Popen(
            (sys.executable,str(gateway),'--resume',str(checkpoint_path),
             '--auth-token','workbench-secret','--port','0'),
            stdout=subprocess.PIPE,text=True)
        try:
            gateway_port=int(gateway_process.stdout.readline().split()[-1])
            connection=http.client.HTTPConnection('127.0.0.1',gateway_port,timeout=2)
            request_body=json.dumps({'model':'agi','max_tokens':1024,'stream':True,
                                     'messages':[{'role':'user','content':[
                                         {'type':'text','text':'unresolved body frame'},
                                         {'type':'text','text':bytes(taught_heldout.surface).decode()}]}]})
            connection.request('POST','/v1/messages',request_body,
                               {'authorization':'Bearer workbench-secret','content-type':'application/json'})
            gateway_response=connection.getresponse();gateway_stream=gateway_response.read();connection.close()
            replay=http.client.HTTPConnection('127.0.0.1',gateway_port,timeout=2)
            replay_body=json.dumps({'model':'agi','messages':[
                {'role':'user','content':'old transcript'},
                {'role':'assistant','content':'cached answer'}]})
            replay.request('POST','/v1/messages',replay_body,
                           {'authorization':'Bearer workbench-secret','content-type':'application/json'})
            replay_status=replay.getresponse().status;replay.close()
            forged=http.client.HTTPConnection('127.0.0.1',gateway_port,timeout=2)
            forged_body=json.dumps({'model':'sonnet','system':'client metadata','tools':[{'name':'host'}],
                'messages':[{'role':'user','content':[{'type':'text','text':'metadata'},
                    {'type':'text','text':bytes(taught_heldout.surface).decode()},
                    {'type':'text','text':'hidden transcript'}]},
                    {'role':'system','content':[{'type':'text','text':'metadata'}]}]})
            forged.request('POST','/v1/messages?beta=true',forged_body,
                           {'x-api-key':'workbench-secret','content-type':'application/json'})
            forged_envelope_status=forged.getresponse().status;forged.close()

            quiet_gap=http.client.HTTPConnection('127.0.0.1',gateway_port,timeout=2)
            quiet_gap.request('POST','/v1/messages',json.dumps({
                'model':'agi','messages':[{'role':'user','content':bytes(ambiguous_probe).decode()}]}),
                {'authorization':'Bearer workbench-secret','content-type':'application/json'})
            quiet_gap_response=quiet_gap.getresponse()
            quiet_gap_payload=json.loads(quiet_gap_response.read());quiet_gap.close()

            # A Claude Code body replays its earlier wire rows on resume. The
            # resident cursor admits only the source-authenticated suffix and
            # verifies the immediately preceding motor reafference in O(1).
            body_headers={'x-api-key':'workbench-secret','content-type':'application/json',
                          'x-claude-code-session-id':'workbench-session-a'}
            body_prefix=[
                {'role':'user','content':[{'type':'text','text':'body state'},
                                          {'type':'text','text':bytes(taught_heldout.surface).decode()}]},
                {'role':'system','content':[{'type':'text','text':'transport control'}]}]
            first_body=http.client.HTTPConnection('127.0.0.1',gateway_port,timeout=2)
            first_body.request('POST','/v1/messages?beta=true',json.dumps({
                'model':'sonnet','system':'client metadata','tools':[{'name':'body'}],
                'messages':body_prefix}),body_headers)
            first_body_response=first_body.getresponse()
            first_body_payload=json.loads(first_body_response.read());first_body.close()
            resident_text=bytes(l4.surface).decode()
            resumed_messages=[
                # This altered ignored prefix is hostile evidence that replayed
                # transcript bytes cannot change resident cognition.
                {'role':'user','content':'altered inert history'},
                {'role':'system','content':'altered inert control'},
                {'role':'assistant','content':[{'type':'text','text':resident_text}]},
                {'role':'user','content':bytes(taught_heldout.surface).decode()},
                {'role':'system','content':[{'type':'text','text':'transport control'}]}]
            resumed_body=http.client.HTTPConnection('127.0.0.1',gateway_port,timeout=2)
            resumed_body.request('POST','/v1/messages?beta=true',json.dumps({
                'model':'sonnet','system':'client metadata','tools':[{'name':'body'}],
                'messages':resumed_messages}),body_headers)
            resumed_body_response=resumed_body.getresponse()
            resumed_body_payload=json.loads(resumed_body_response.read());resumed_body.close()
            tampered_body=http.client.HTTPConnection('127.0.0.1',gateway_port,timeout=2)
            tampered_body.request('POST','/v1/messages?beta=true',json.dumps({
                'model':'sonnet','system':'client metadata','tools':[{'name':'body'}],
                'messages':resumed_messages+[
                    {'role':'assistant','content':[{'type':'text','text':'forged motor'}]},
                    {'role':'user','content':bytes(taught_heldout.surface).decode()},
                    {'role':'system','content':[{'type':'text','text':'transport control'}]}]}),
                body_headers)
            tampered_reafference_status=tampered_body.getresponse().status;tampered_body.close()
            duplicate_body=http.client.HTTPConnection('127.0.0.1',gateway_port,timeout=2)
            duplicate_body.request('POST','/v1/messages?beta=true',json.dumps({
                'model':'sonnet','system':'client metadata','tools':[{'name':'body'}],
                'messages':resumed_messages}),body_headers)
            duplicate_body_status=duplicate_body.getresponse().status;duplicate_body.close()
            if claude_bin:
                claude_env=os.environ.copy()
                claude_env.update({
                    'ANTHROPIC_BASE_URL':f'http://127.0.0.1:{gateway_port}',
                    'ANTHROPIC_AUTH_TOKEN':'workbench-secret',
                    'ANTHROPIC_API_KEY':'workbench-secret',
                    'CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC':'1',
                    'CLAUDE_CONFIG_DIR':str(Path(directory)/'claude-config'),
                })
                claude_cli=subprocess.run(
                    (claude_bin,'-p','--model','sonnet','--max-turns','1',
                     '--output-format','json',
                     bytes(taught_heldout.surface).decode()),
                    capture_output=True,timeout=20,env=claude_env)
                claude_first=json.loads(claude_cli.stdout)
                claude_cli_resumed=subprocess.run(
                    (claude_bin,'-p','--resume',claude_first['session_id'],
                     '--model','sonnet','--max-turns','1','--output-format','json',
                     bytes(taught_heldout.surface).decode()),
                    capture_output=True,timeout=20,env=claude_env)
                claude_cli_quiet=subprocess.run(
                    (claude_bin,'-p','--model','sonnet','--max-turns','1',
                     '--output-format','json',bytes(ambiguous_probe).decode()),
                    capture_output=True,timeout=20,env=claude_env)
        finally:
            gateway_process.terminate();gateway_process.wait(timeout=2)
        gateway_checkpoint_text=checkpoint_path.read_text()
        gateway_checkpoint=LanguageMasteryAdultV1.restore(json.loads(gateway_checkpoint_text))
    resumed.observe_social_contact(PA,OBJ,SA,AB,0x8331);resumed_stale=resumed.choose(AdultStateV1())
    resumed.observe_social_contact(PB,OBJ,SB,AB,0x8332);resumed_current=resumed.choose(AdultStateV1())
    resumed.observe_social_contact(PU,OBJ,SU,AB,0x8333);resumed_unknown=resumed.choose(AdultStateV1())
    resumed_recursive=resumed.public_surface(p_recursive.identity)
    lesion=json.loads(json.dumps(checkpoint))
    for row in lesion['selection_credit']['rows']:
        for context_row in row['contexts']:context_row['participated']=False
    lesioned=LanguageMasteryAdultV1.restore(lesion)
    lesioned.observe_social_contact(PA,OBJ,SA,AB,0x8334)
    connectivity_lesion_choice=lesioned.choose(AdultStateV1())
    checkpoint_keys=json.dumps(checkpoint,sort_keys=True)

    emitted,expr=emit_all(adult,p_recursive.identity)
    recursive_surface=adult.public_surface(p_recursive.identity)

    # Raw text ambiguity remains unresolved: adding an independently learned alias
    # for the same surface creates multiple lawful inversions, so the membrane must
    # not choose one on behalf of the Adult.
    DAX_ALT=498
    adult.observe_surface_item(DAX_ALT,tuple(b'dax'),4601);adult.observe_surface_item(DAX_ALT,tuple(b'dax'),4602)
    ambiguous_raw_scene=contact.contact(CONTACT_UTTERANCE,tuple(taught_heldout.surface),4603)
    stale_choice_after_ambiguous_contact=adult.choose(AdultStateV1())
    unlearned_ambiguity_after_quiet_tick,unlearned_ambiguity_ticks=settle_internal(adult)
    try:
        adult.choose((SHORT,p_recursive.identity),0xD001,AdultStateV1())
        host_candidate_nomination_refused=False
    except TypeError:
        host_candidate_nomination_refused=True
    try:
        adult.choose(0xD001,AdultStateV1())
        host_context_nomination_refused=False
    except TypeError:
        host_context_nomination_refused=True
    local_candidate_peak=max(map(len,adult.credit.context_members.values()))
    checks={
      'conversation_teaches_new_word_into_heldout_sentence':b'dax' in bytes(taught_heldout.surface) and taught_heldout.surface not in (l1.surface,l2.surface),
      'raw_utterance_reconstructs_internal_clause_without_concept_ids':raw_taught.identity==taught_heldout.identity,
      'ambiguous_raw_utterance_is_not_host_resolved':ambiguous_raw_scene!=0 and stale_choice_after_ambiguous_contact==0,
      'newly_taught_language_participates_in_recursive_answer':bytes(taught_heldout.surface) in bytes(recursive_surface),
      'consequence_supports_deeper_longer_answer':deep_choice==p_recursive.identity and len(recursive_surface)>len(l1.surface) and adult.program_depth(deep_choice)>=2,
      'urgency_shortens_same_adult_answer':urgent_choice==SHORT,
      'yoked_correction_does_not_rewrite_answer_preference':before_correction==P_OLD and after_yoked==P_OLD,
      'independent_correction_changes_later_answer':after_correction==P_NEW,
      'stale_partner_history_elicits_explicit_language':stale_choice==P_EXPLICIT,
      'current_shared_history_elicits_compact_language':current_choice==P_COMPACT,
      'social_ambiguity_elicits_clarification':unknown_choice==P_ASK,
      'fixed_language_capacity_survives_eviction_pressure':language_evictions>0 and len(adult.credit.rows)==credit_capacity and post_deep==p_recursive.identity and post_corrected==P_NEW and post_stale==P_EXPLICIT and post_current==P_COMPACT and post_unknown==P_ASK,
      'partner_trace_survives_social_capacity_pressure':social_evictions>0 and adult.prospection.trace_evictions>0 and len(adult.prospection._traces)==adult.prospection.trace_capacity and focal_before==AA and focal_after==AA and post_social_stale==P_EXPLICIT and post_social_current==P_COMPACT,
      'restart_preserves_same_learned_language_and_revision':checkpoint_exact and resumed_recursive==recursive_surface and b'dax' in bytes(resumed_recursive) and resumed._probe_choice(CORR,AdultStateV1())==P_NEW,
      'restart_preserves_partner_sensitive_future_behavior':resumed_stale==post_social_stale and resumed_current==post_social_current and resumed_unknown==post_social_unknown and resumed.predict_social_action(PA,OBJ)==AA,
      'restart_drops_active_occurrence_until_fresh_contact':resumed_without_contact==0,
      'raw_contact_reinstates_resident_occurrence_after_restart':resumed_raw_scene!=0 and resumed_raw_choice==P_CONTACT_REPLY,
      'two_process_terminal_resumes_same_learned_adult':first_sitting.stdout==expected_terminal_output and second_sitting.stdout==expected_terminal_output and process_checkpoint.public_surface(P_CONTACT_REPLY)==l4.surface,
      'claude_code_messages_wire_carries_resident_expression':gateway_response.status==200 and b'event: message_start' in gateway_stream and bytes(l4.surface) in gateway_stream and b'event: message_stop' in gateway_stream,
      'claude_code_gateway_preserves_ordered_text_frames':gateway_response.status==200 and bytes(l4.surface) in gateway_stream,
      'claude_code_gateway_refuses_transcript_mind':replay_status==422,
      'claude_code_gateway_refuses_forged_extra_contact':forged_envelope_status==422,
      'first_ambiguous_evaluation_is_voluntarily_silent':training_gap!=0 and training_gap_first_choice==0 and weak_gap!=0 and weak_gap_first_choice==0 and learned_gap!=0 and learned_gap_first_choice==0 and pending_resumed_first_choice==0,
      'history_changes_settling_latency_without_phase_ladder':weak_gap_question==P_ASK and learned_gap_question==P_ASK and 1<=learned_gap_ticks<weak_gap_ticks,
      'same_ambiguity_requires_lived_inquiry_history':naive_question==0,
      'yoked_information_does_not_author_inquiry':yoked_question==0,
      'recent_adverse_consequence_suppresses_same_inquiry':discouraged_question==0,
      'current_body_state_changes_same_gap_trajectory':pressured_question!=learned_gap_question or pressured_ticks!=learned_gap_ticks,
      'checkpoint_preserves_recurrent_trajectory':pending_resumed_question==P_ASK and pending_resumed_ticks==learned_gap_ticks and pending_resumed_extra_tick==0,
      'live_terminal_originates_clarification_without_stdin':quiet_process_outputs==[bytes(ask.surface)+b'\n']*2,
      'host_timeout_cadence_does_not_choose_question':len(set(quiet_process_outputs))==1,
      'pending_gap_survives_checkpoint_without_raw_wording':pending_checkpoint['pending_language_competition'] and bytes(ambiguous_probe).decode() not in json.dumps(pending_checkpoint),
      'participating_credit_is_required_for_quiet_clarification':pending_lesion_question==0,
      'claude_body_schedules_resident_quiet_clarification':quiet_gap_response.status==200 and quiet_gap_payload['content'][0]['text']==bytes(ask.surface).decode(),
      'resident_body_cursor_admits_only_new_suffix':first_body_response.status==200 and resumed_body_response.status==200 and first_body_payload['content'][0]['text']==resident_text and resumed_body_payload['content'][0]['text']==resident_text,
      'replayed_prefix_is_causally_inert':resumed_body_response.status==200,
      'resident_body_cursor_refuses_forged_reafference':tampered_reafference_status==422,
      'resident_body_cursor_refuses_duplicate_boundary':duplicate_body_status==422,
      'body_cursor_survives_checkpoint_without_transcript':gateway_checkpoint.body_ingress_cursor('workbench-session-a')[0]==5 and 'altered inert history' not in gateway_checkpoint_text,
      'unrewarded_ambiguity_remains_silent_after_recurrence':unlearned_ambiguity_after_quiet_tick==0,
      'learned_participation_connectivity_is_causally_required':connectivity_lesion_choice==0,
      'derived_lookup_and_active_context_are_not_persisted':'context_members' not in checkpoint_keys and 'feature_index' not in checkpoint_keys and 'current_selection_context' not in checkpoint_keys,
      'transport_cannot_nominate_public_candidates':host_candidate_nomination_refused,
      'transport_cannot_nominate_current_context':host_context_nomination_refused,
      'resident_selection_is_context_local_under_pressure':local_candidate_peak<=4 and len(adult.credit.rows)==credit_capacity,
      'recursive_public_expression_is_incremental_and_exact':emitted==recursive_surface and not hasattr(expr,'payload') and not hasattr(expr,'surface'),
      'one_continuing_adult':True,
      'no_expected_answer_token_or_llm_runtime':True,
      'bounded_fast_path':time.perf_counter()-started<(30.0 if claude_bin else 1.0),
    }
    if claude_bin:
        checks['literal_claude_code_process_uses_resident_adult']=(
            claude_cli is not None and claude_cli.returncode==0
            and json.loads(claude_cli.stdout)['result']==bytes(l4.surface).decode())
        checks['literal_claude_code_resume_uses_same_resident_adult']=(
            claude_cli_resumed is not None and claude_cli_resumed.returncode==0
            and json.loads(claude_cli_resumed.stdout)['result']==bytes(l4.surface).decode())
        checks['literal_claude_code_waits_for_resident_clarification']=(
            claude_cli_quiet is not None and claude_cli_quiet.returncode==0
            and json.loads(claude_cli_quiet.stdout)['result']==bytes(ask.surface).decode())
    failed=[k for k,v in checks.items() if not v]
    if failed:raise SystemExit('FOUNDRY_AGI_CHAT_SESSION_RED '+','.join(failed)+' restart='+repr((resumed_stale,resumed_current,resumed_unknown,resumed.predict_social_action(PA,OBJ),P_EXPLICIT,P_COMPACT,P_ASK,AA))+' settling='+repr((weak_gap_ticks,learned_gap_ticks,pending_resumed_ticks,pending_lesion_ticks,unlearned_ambiguity_ticks,quiet_gap_response.status,quiet_gap_payload))+' claude='+repr(None if claude_cli is None else (claude_cli.returncode,claude_cli.stdout[-1000:],claude_cli.stderr[-1000:],None if claude_cli_resumed is None else (claude_cli_resumed.returncode,claude_cli_resumed.stdout[-1000:],claude_cli_resumed.stderr[-1000:]))))
    p=Path(__file__);result={'contract':'FOUNDRY_AGI_CHAT_SESSION_GREEN','reference_only':True,'usable_chat_phenotype_improved':True,'checks':checks,
      'conversation':{'taught':bytes(taught_heldout.surface).decode(),'deep_bytes':len(recursive_surface),'deep_depth':adult.program_depth(p_recursive.identity),'corrected':bytes(adult.public_surface(after_correction)).decode(),'stale_partner':bytes(adult.public_surface(stale_choice)).decode(),'current_partner':bytes(adult.public_surface(current_choice)).decode(),'ambiguous_partner':bytes(adult.public_surface(unknown_choice)).decode()},
      'continuity':{'checkpoint_schema':checkpoint['schema'],'active_occurrence_persisted':False,'resumed_without_contact':resumed_without_contact,'connectivity_lesion_choice':connectivity_lesion_choice},
      'selection':{'host_candidate_nominations':0,'host_context_nominations':0,'context_candidate_peak':local_candidate_peak,'resident_rows':len(adult.credit.rows),'weak_gap_ticks':weak_gap_ticks,'learned_gap_ticks':learned_gap_ticks,'lesion_ticks':pending_lesion_ticks,'unlearned_ticks':unlearned_ambiguity_ticks,'naive_ticks':naive_ticks,'yoked_ticks':yoked_ticks,'discouraged_ticks':discouraged_ticks,'pressured_ticks':pressured_ticks},
      'tokens':False,'transformer':False,'backprop':False,'expected_output':False,'runtime_llm':False,'elapsed_ms':round((time.perf_counter()-started)*1000,3),'remaining_red':(['DIRECT_CONTINUING_ADULT_PARITY'] if claude_bin else ['DIRECT_CONTINUING_ADULT_PARITY','CLAUDE_CODE_HEADLESS_BINARY_EXECUTION']),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
    print(result['contract']);print(json.dumps(result,sort_keys=True,indent=2))

if __name__=='__main__':main()
