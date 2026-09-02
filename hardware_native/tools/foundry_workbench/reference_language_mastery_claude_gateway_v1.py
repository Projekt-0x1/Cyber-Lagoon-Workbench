#!/usr/bin/env python3
"""Text-only Anthropic Messages body for one checkpointed Workbench Adult."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
import time
from http.server import BaseHTTPRequestHandler,HTTPServer
from pathlib import Path

from reference_authorized_ambient_feed_body_v1 import AuthorizedAmbientFeedBodyV1,MAX_AUTHORIZED_AMBIENT_POOL
from reference_language_learning_v1 import MAX_SURFACE
from reference_life_function_curriculum_v1 import (
    ReferenceLifeFunctionRuntimeV2,canonical_species_program_v2,
)

BODY_ACTION_TOOL='agi_body_action'
CLAUDE_MCP_BODY_ACTION_TOOL='mcp__agi_body__agi_body_action'
BODY_ACTION_TOOLS=(BODY_ACTION_TOOL,CLAUDE_MCP_BODY_ACTION_TOOL)
# Claude Code retries empty/whitespace-only Anthropic turns.  WORD JOINER is a
# non-rendering transport frame: it closes the physical turn without inventing
# a resident utterance or becoming contact on the next append-only boundary.
CLAUDE_SILENCE_FRAME='\u2060'
BODY_ACTION_INPUT_SCHEMA={'type':'object','properties':{'surface':{'type':'string'}},
                          'required':['surface']}
CONTACT_IDENTITY_HEADER='x-agi-contact-identity'


def authorized_ambient_pool(request):
    """Decode an authorized provider-neutral pool; the collector never supplies entropy."""
    if not isinstance(request,dict) or set(request)!={'candidates'}:raise ValueError('ambient-body:envelope')
    candidates=request['candidates']
    if not isinstance(candidates,list) or not candidates or len(candidates)>MAX_AUTHORIZED_AMBIENT_POOL:raise ValueError('ambient-body:candidates')
    pool=[]
    for row in candidates:
        if (not isinstance(row,dict) or set(row)!={'source','text'} or type(row['source']) is not int
                or row['source']<=0 or not isinstance(row['text'],str) or not row['text']):
            raise ValueError('ambient-body:candidate')
        pool.append((int(row['source']),row['text'].encode('utf-8')))
    return tuple(pool)


def text_frames(content):
    if isinstance(content,str):return (content.encode(),)
    if isinstance(content,dict) and content.get('type')=='text' and isinstance(content.get('text'),str):
        return (content['text'].encode(),)
    if (isinstance(content,list) and content and
            all(isinstance(block,dict) and block.get('type')=='text' and
                isinstance(block.get('text'),str) for block in content)):
        return tuple(block['text'].encode() for block in content)
    shape=[block.get('type') if isinstance(block,dict) else type(block).__name__
           for block in content] if isinstance(content,list) else type(content).__name__
    raise ValueError('body:nontext_contact:'+repr(shape))


def _tool_reafference(content):
    if not (isinstance(content,list) and len(content)==1):return None
    block=content[0]
    if (not isinstance(block,dict) or block.get('type')!='tool_use'
            or block.get('name') not in BODY_ACTION_TOOLS or not isinstance(block.get('id'),str)
            or not isinstance(block.get('input'),dict)
            or set(block['input'])!={'surface'} or not isinstance(block['input']['surface'],str)):
        return None
    frame=json.dumps(block,separators=(',',':'),sort_keys=True).encode()
    return frame,block['input']['surface'].encode(),block['id']


def assistant_reafference(content):
    tool=_tool_reafference(content)
    if tool is not None:return (tool[0],),tool[1],tool[2]
    frames=text_frames(content)
    return frames,b''.join(frames),''


def _tool_result_frames(content,tool_id):
    if not (isinstance(content,list) and content):raise ValueError('body:tool_result')
    block=content[0]
    if (not isinstance(block,dict) or block.get('type')!='tool_result'
            or block.get('tool_use_id')!=tool_id
            or not isinstance(block.get('is_error',False),bool)):
        raise ValueError('body:tool_result')
    raw=block.get('content','')
    result_frames=(() if raw=='' else text_frames(raw))
    trailing=content[1:]
    dialogue_frames=(text_frames(trailing) if trailing else ())
    return (*result_frames,*dialogue_frames),bool(block.get('is_error',False)), \
        (False,)*len(result_frames)+(True,)*len(dialogue_frames)


def reafference_commitment(frames):
    digest=hashlib.sha256(b'CYBER_LAGOON_BODY_REAFFERENCE_V1\0')
    for raw in frames:
        digest.update(len(raw).to_bytes(8,'big'));digest.update(raw)
    return digest.hexdigest()


def body_source_identity(source):
    if not source:return 0
    digest=hashlib.sha256(b'CYBER_LAGOON_BODY_SOURCE_V1\0'+source.encode()).digest()
    return int.from_bytes(digest[:8],'big') or 1


def save_runtime(path,runtime):
    """Atomically checkpoint the one continuing Life runtime."""
    with tempfile.NamedTemporaryFile('w',dir=path.parent,delete=False) as out:
        json.dump(runtime.checkpoint(),out,separators=(',',':'),sort_keys=True)
        out.flush();os.fsync(out.fileno());temporary=Path(out.name)
    os.replace(temporary,path)


def resident_contact(runtime,frames,channel,observe_causal_dialogue=False):
    """Page opaque body bytes into the one resident Life without choosing content."""
    channel=int(channel)
    frames=tuple(frames)
    observe=(tuple(bool(value) for value in observe_causal_dialogue)
             if isinstance(observe_causal_dialogue,(tuple,list)) else
             (bool(observe_causal_dialogue),)*len(frames))
    if len(observe)!=len(frames):raise ValueError('body:dialogue_frame_mask')
    out=[]
    for raw,is_dialogue in zip(frames,observe):
        for start in range(0,len(raw),MAX_SURFACE):
            page=raw[start:start+MAX_SURFACE]
            if is_dialogue:
                runtime.adult.observe_authenticated_causal_dialogue_contact(
                    page,channel,channel)
            out.append(runtime.contact_utterance(page,channel,channel)[0])
    return b''.join(out)


def settle_reafferenced_contact(runtime,surface,channel):
    """Settle one exact motor occurrence without fabricating its consequence."""
    digest=hashlib.sha256(bytes(surface)).hexdigest();channel=int(channel)
    causal=tuple(row for row in
                  runtime.adult.pending_causal_dialogue_actions.values()
                  if int(row.source)==channel and int(row.channel)==channel
                  and row.surface_digest==digest)
    contextual=tuple(row for row in
                     runtime.adult.pending_context_affordance_actions.values()
                     if int(row.source)==channel and int(row.channel)==channel
                     and row.surface_digest==digest)
    inquiries=tuple(row for row in
                    runtime.adult.pending_endogenous_inquiry_actions.values()
                    if int(row.source)==channel and int(row.channel)==channel
                    and row.surface_digest==digest)
    pending=(*causal,*contextual,*inquiries)
    # ponytail: a body turn containing multiple public acts remains physically
    # authenticated but uncredited until ingress owns their ordered identities.
    if len(pending)!=1:return False
    settled=(runtime.adult.settle_causal_dialogue_reafference(
        causal[0],channel) if causal else runtime.settle_contact_consequence(
            contextual[0].identity,channel,0,0,True) if contextual else True)
    if not settled:return False
    return 'causal' if causal else 'context' if contextual else 'inquiry'


def settled_causal_common_ground(runtime,channel):
    """Recover the resident receipt hidden behind Claude's silence frame."""
    channel=int(channel)
    return any(int(row.source)==channel and int(row.channel)==channel
               and int(row.identity) not in runtime.adult.pending_causal_dialogue_actions
               for row in runtime.adult.recent_causal_dialogue_actions.values())


def _tool_id(identity):return 'toolu_agi_'+format(int(identity),'x')


def resident_body_action(runtime,surface,channel,tools):
    offered=tuple(tool['name'] for tool in tools if isinstance(tool,dict)
                  and tool.get('name') in BODY_ACTION_TOOLS
                  and tool.get('input_schema')==BODY_ACTION_INPUT_SCHEMA)
    if len(offered)!=1:return None
    digest=hashlib.sha256(bytes(surface)).hexdigest();channel=int(channel)
    rows=tuple(row for row in runtime.adult.pending_causal_dialogue_actions.values()
               if int(row.source)==channel and int(row.channel)==channel
               and row.surface_digest==digest)
    rows=(*rows,*(row for row in runtime.adult.pending_endogenous_inquiry_actions.values()
                  if int(row.source)==channel and int(row.channel)==channel
                  and row.surface_digest==digest))
    if len(rows)!=1:return None
    return {'type':'tool_use','id':_tool_id(rows[0].identity),'name':offered[0],
            'input':{'surface':bytes(surface).decode('utf-8')}}


def settle_tool_result(runtime,tool_id,channel,is_error):
    prefix='toolu_agi_'
    if not isinstance(tool_id,str) or not tool_id.startswith(prefix):return False
    try:identity=int(tool_id[len(prefix):],16)
    except ValueError:return False
    outcome=-(1<<16) if bool(is_error) else (1<<16)
    receipt=runtime.adult.recent_causal_dialogue_actions.get(identity)
    if (receipt is not None and int(receipt.source)==int(channel)
            and int(receipt.channel)==int(channel)):
        return runtime.adult.settle_causal_dialogue_return(
            receipt,channel,outcome,0,True,not bool(is_error))
    inquiry=runtime.adult.pending_endogenous_inquiry_actions.get(identity)
    return bool(inquiry is not None and int(inquiry.channel)==int(channel)
                and runtime.adult.settle_endogenous_inquiry_motor_return(
                    inquiry,channel,not bool(is_error)))


def claude_contact_event(messages,cursor,expected_reafference):
    """Decode one exact append-only body event without replaying its prefix."""
    cursor=int(cursor)
    if cursor<0 or cursor>len(messages):raise ValueError('body:claude_event_boundary')
    suffix=messages[cursor:]
    if not suffix:raise ValueError('body:claude_duplicate_boundary')
    continuing=bool(expected_reafference)
    reafferenced_surface=None;tool_id=''
    if continuing:
        own=suffix[0]
        if own.get('role')!='assistant':raise ValueError('body:claude_reafference')
        own_frames,reafferenced_surface,tool_id=assistant_reafference(own.get('content'))
        observed=reafference_commitment(own_frames)
        if observed!=expected_reafference:raise ValueError('body:claude_reafference')
        suffix=suffix[1:]
    elif cursor:
        raise ValueError('body:claude_event_boundary')

    user_rows=[]
    for row in suffix:
        role=row.get('role')
        if role=='user':user_rows.append(row)
        elif role!='system':raise ValueError('body:claude_event_boundary')
    if not user_rows:raise ValueError('body:claude_no_contact')

    frames=[];dialogue_mask=[];tool_result=None
    for position,row in enumerate(user_rows):
        if tool_id and position==0:
            row_frames,is_error,row_dialogue=_tool_result_frames(
                row.get('content'),tool_id)
            tool_result=(tool_id,is_error)
        else:
            row_frames=text_frames(row.get('content'))
            row_dialogue=(True,)*len(row_frames)
        # Claude Code's first physical user envelope prepends body-state text.
        # Its final block is new exafferent contact. This codec distinction is
        # source/protocol history, never a cognitive phase or content label.
        if not continuing and position==0:
            if len(row_frames)<2:raise ValueError('body:claude_initial_boundary')
            row_frames=row_frames[-1:]
            row_dialogue=row_dialogue[-1:]
        frames.extend(row_frames)
        dialogue_mask.extend(row_dialogue)
    if tool_id and tool_result is None:raise ValueError('body:tool_result')
    return tuple(frames),len(messages),reafferenced_surface,tool_result, \
        tuple(dialogue_mask)


def claude_contact_suffix(messages,cursor,expected_reafference):
    """Text-only compatibility surface; tool results require the body event path."""
    frames,next_sequence,_surface,tool_result,_dialogue_mask=claude_contact_event(
        messages,cursor,expected_reafference)
    if tool_result is not None:raise ValueError('body:nontext_contact:tool_result')
    return frames,next_sequence


class AdultMessagesHandler(BaseHTTPRequestHandler):
    protocol_version='HTTP/1.1'

    def log_message(self,*_):pass

    def _json(self,status,payload):
        body=json.dumps(payload,separators=(',',':')).encode()
        self.send_response(status);self.send_header('content-type','application/json')
        self.send_header('content-length',str(len(body)));self.send_header('connection','close')
        self.end_headers();self.wfile.write(body);self.close_connection=True

    def do_POST(self):
        if os.environ.get('AGI_GATEWAY_TRACE'):
            print('AGI_GATEWAY_REQUEST',self.path,
                  'authorization='+str(bool(self.headers.get('authorization'))),
                  'x_api_key='+str(bool(self.headers.get('x-api-key'))),file=sys.stderr,flush=True)
        route=self.path.split('?',1)[0]
        if route not in ('/v1/messages','/v1/ambient'):return self._json(404,{'type':'error','error':{'type':'not_found_error','message':'not found'}})
        bearer=self.headers.get('authorization')=='Bearer '+self.server.auth_token
        api_key=self.headers.get('x-api-key')==self.server.auth_token
        if not (bearer or api_key):
            return self._json(401,{'type':'error','error':{'type':'authentication_error','message':'unauthorized'}})
        try:
            size=int(self.headers.get('content-length','0'))
            if size<=0 or size>1_048_576:raise ValueError('body:frame_size')
            request=json.loads(self.rfile.read(size))
            if route=='/v1/ambient':
                pool=authorized_ambient_pool(request);tick=int(self.server.runtime.adult.language_adult._tick)
                receipt,processed=AuthorizedAmbientFeedBodyV1.pump(self.server.runtime,tick,pool,entropy=None)
                save_runtime(self.server.checkpoint,self.server.runtime)
                return self._json(200,{'type':'ambient_receipt','selection':receipt,
                                      'processed':[list(row) for row in processed],
                                      'pending_count':int(self.server.runtime.ambient_stream.pending_count)})
            messages=request.get('messages')
            if os.environ.get('AGI_GATEWAY_TRACE'):
                print('AGI_GATEWAY_SHAPE',
                      'session='+repr(self.headers.get('x-claude-code-session-id')),
                      'headers='+repr(sorted(name.lower() for name in self.headers)),
                      'request_keys='+repr(sorted(request)),
                      'metadata_keys='+repr(sorted(request.get('metadata',{})))
                          if isinstance(request.get('metadata'),dict) else 'metadata_keys=[]',
                      'messages='+str(len(messages) if isinstance(messages,list) else -1),
                      'roles='+repr([row.get('role') for row in messages] if isinstance(messages,list) else []),
                      'content_types='+repr([[part.get('type') for part in row.get('content',())]
                          if isinstance(row.get('content'),list) else type(row.get('content')).__name__
                          for row in messages] if isinstance(messages,list) else []),
                      'system='+str(bool(request.get('system'))),
                      'tools='+str(len(request.get('tools',()))),file=sys.stderr,flush=True)
            if not isinstance(messages,list):raise ValueError('body:transcript_replay_refused')
            if self.server.runtime.ambient_stream.pending_count:
                AuthorizedAmbientFeedBodyV1.drain(self.server.runtime,int(self.server.runtime.adult.language_adult._tick))
            claude_source=self.headers.get('x-claude-code-session-id')
            next_sequence=0;reafferenced_surface=None;tool_result=None
            if claude_source:
                brain=self.server.runtime.adult.language_adult
                cursor,expected_reafference=brain.body_ingress_cursor(claude_source)
                if not (bool(request.get('system')) and bool(request.get('tools'))):
                    raise ValueError('body:claude_envelope')
                (frames,next_sequence,reafferenced_surface,tool_result,
                 dialogue_mask)=claude_contact_event(
                    messages,cursor,expected_reafference)
            elif len(messages)==1 and messages[0].get('role')=='user':
                frames=text_frames(messages[0].get('content'))
                dialogue_mask=(True,)*len(frames)
            else:raise ValueError('body:transcript_replay_refused')
            # Every user text block is transported in bounded ordered physical
            # pages. The gateway does not inspect or choose among framing blocks.
            # Transport sessions own append-only replay boundaries; they are not
            # social identities.  One authenticated body therefore retains one
            # continuing contact provenance across CLI restarts unless that body
            # explicitly authenticates a different contact at its membrane.
            contact_identity=self.headers.get(CONTACT_IDENTITY_HEADER)
            channel=body_source_identity(contact_identity or self.server.body_identity)
            reafferent_action=False
            if reafferenced_surface is not None:
                reafferent_action=settle_reafferenced_contact(
                    self.server.runtime,reafferenced_surface,channel)
                if (not reafferent_action
                        and reafferenced_surface==CLAUDE_SILENCE_FRAME.encode()
                        and settled_causal_common_ground(self.server.runtime,channel)):
                    reafferent_action='causal'
            if tool_result is not None:
                if reafferent_action not in ('causal','inquiry') or not settle_tool_result(
                        self.server.runtime,tool_result[0],channel,tool_result[1]):
                    raise ValueError('body:tool_result')
            # Ordinary authenticated partner contact always reaches the
            # resident common-ground competition. The Adult—not adjacency at
            # this membrane—decides whether any earlier same-session action is
            # relevant. Tool-result frames remain consequence provenance only.
            motor=resident_contact(
                self.server.runtime,frames,channel,dialogue_mask)
            for _ in range(128):
                if motor or not self.server.runtime.internal_work_pending(channel):break
                # Keep the request physically open across bounded no-contact
                # intervals. The host supplies time/compute only; the resident
                # Adult decides convergence depth, action, or silence.
                time.sleep(self.server.idle_seconds)
                motor=self.server.runtime.quiet_public_opportunity(channel,channel)
            else:raise ValueError('body:resident_work_bound')
            body_action=resident_body_action(
                self.server.runtime,motor,channel,request.get('tools',())) if motor else None
            # Claude Code retries an empty assistant message by appending its own
            # synthetic visibility request inside the current user row. That
            # client-generated text is body machinery, not exafferent contact.
            # Every Claude Code session therefore receives the existing invisible
            # silence frame when the Adult emits no motor; actual resident motor
            # surfaces remain ordinary text.
            wire_text=(motor.decode('utf-8') if motor else
                       CLAUDE_SILENCE_FRAME if claude_source else '')
            content=([body_action] if body_action is not None
                     else [{'type':'text','text':wire_text}])
            stop_reason='tool_use' if body_action is not None else 'end_turn'
            if claude_source:
                reafference_frames=((json.dumps(body_action,separators=(',',':'),sort_keys=True).encode(),)
                                     if body_action is not None else (wire_text.encode(),))
                self.server.runtime.adult.language_adult.settle_body_ingress(
                    claude_source,next_sequence,reafference_commitment(reafference_frames),0)
            save_runtime(self.server.checkpoint,self.server.runtime)
        except (KeyError,TypeError,ValueError,UnicodeError,json.JSONDecodeError) as exc:
            print('AGI_CLAUDE_GATEWAY_REFUSED',str(exc),file=sys.stderr,flush=True)
            return self._json(422,{'type':'error','error':{'type':'invalid_request_error','message':str(exc)}})
        model=str(request.get('model','agi'));message_id='msg_agi_'+format(
            self.server.runtime.adult.language_adult._tick,'x')
        if request.get('stream'):
            if body_action is None:
                block_start={'type':'text','text':''};delta={'type':'text_delta','text':content[0]['text']}
            else:
                block_start={**body_action,'input':{}};delta={'type':'input_json_delta','partial_json':json.dumps(body_action['input'],separators=(',',':'))}
            events=(
                ('message_start',{'type':'message_start','message':{'id':message_id,'type':'message','role':'assistant','content':[],'model':model,'stop_reason':None,'stop_sequence':None,'usage':{'input_tokens':0,'output_tokens':0}}}),
                ('content_block_start',{'type':'content_block_start','index':0,'content_block':block_start}),
                ('content_block_delta',{'type':'content_block_delta','index':0,'delta':delta}),
                ('content_block_stop',{'type':'content_block_stop','index':0}),
                ('message_delta',{'type':'message_delta','delta':{'stop_reason':stop_reason,'stop_sequence':None},'usage':{'output_tokens':0}}),
                ('message_stop',{'type':'message_stop'}))
            body=b''.join(('event: '+event+'\ndata: '+json.dumps(data,separators=(',',':'))+'\n\n').encode() for event,data in events)
            self.send_response(200);self.send_header('content-type','text/event-stream')
            self.send_header('content-length',str(len(body)));self.send_header('connection','close')
            self.end_headers();self.wfile.write(body);self.close_connection=True
            return
        self._json(200,{'id':message_id,'type':'message','role':'assistant','content':content,
                        'model':model,'stop_reason':stop_reason,'stop_sequence':None,
                        'usage':{'input_tokens':0,'output_tokens':0}})


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--resume',required=True,type=Path)
    parser.add_argument('--auth-token',required=True);parser.add_argument('--port',type=int,default=0)
    parser.add_argument('--idle-ms',type=int,default=25)
    args=parser.parse_args();server=HTTPServer(('127.0.0.1',args.port),AdultMessagesHandler)
    if not 1<=args.idle_ms<=1000:raise SystemExit('body:idle_ms')
    server.runtime=ReferenceLifeFunctionRuntimeV2.restore(
        canonical_species_program_v2(),json.loads(args.resume.read_text()))
    server.checkpoint=args.resume;server.auth_token=args.auth_token
    server.body_identity=args.auth_token
    server.idle_seconds=args.idle_ms/1000.0
    print('AGI_CLAUDE_GATEWAY',server.server_port,flush=True);server.serve_forever()


if __name__=='__main__':main()
