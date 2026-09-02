#!/usr/bin/env python3
"""HTTP-level gate for authorized ambient pools on the existing localhost Adult gateway."""
from __future__ import annotations

import http.client
import json
import tempfile
import threading
import time
from http.server import HTTPServer
from pathlib import Path

from reference_language_mastery_claude_gateway_v1 import AdultMessagesHandler,save_runtime
from reference_life_function_curriculum_v1 import ReferenceLifeFunctionRuntimeV2,canonical_species_program_v2

TOKEN='ambient-gateway-test-token'


class Gateway:
    def __init__(self,directory):
        self.path=Path(directory)/'adult.json';self.server=None;self.thread=None
    def start(self):
        runtime=ReferenceLifeFunctionRuntimeV2(canonical_species_program_v2())
        save_runtime(self.path,runtime)
        self.server=HTTPServer(('127.0.0.1',0),AdultMessagesHandler)
        self.server.runtime=runtime;self.server.checkpoint=self.path;self.server.auth_token=TOKEN
        self.server.body_identity=TOKEN;self.server.idle_seconds=0.001
        self.thread=threading.Thread(target=self.server.serve_forever,daemon=True);self.thread.start();return self
    def stop(self):
        if self.server is not None:self.server.shutdown();self.server.server_close()
        if self.thread is not None:self.thread.join(timeout=2)
    def post(self,path,payload,auth=True):
        connection=http.client.HTTPConnection('127.0.0.1',self.server.server_port,timeout=10)
        headers={'content-type':'application/json'}
        if auth:headers['authorization']='Bearer '+TOKEN
        body=json.dumps(payload,separators=(',',':')).encode()
        connection.request('POST',path,body=body,headers=headers);response=connection.getresponse()
        raw=response.read();connection.close()
        return response.status,json.loads(raw)


def main():
    started=time.perf_counter();checks={}
    candidates=[
        {'source':0x51001,'text':'first authorized ambient candidate'},
        {'source':0x51002,'text':'second authorized ambient candidate'},
        {'source':0x51003,'text':'third authorized ambient candidate'},
    ]
    with tempfile.TemporaryDirectory(prefix='ambient-gateway-') as directory:
        body=Gateway(directory).start()
        try:
            status,_=body.post('/v1/ambient',{'candidates':candidates},auth=False)
            checks['ambient_route_requires_existing_gateway_authentication']=status==401
            status,refused=body.post('/v1/ambient',{'candidates':candidates,'entropy':7})
            checks['collector_cannot_supply_or_search_entropy']=status==422 and refused['error']['message']=='ambient-body:envelope'
            status,result=body.post('/v1/ambient',{'candidates':candidates})
            selection=result.get('selection',{});wire=json.dumps(result,sort_keys=True)
            checks['authenticated_pool_uses_server_entropy_and_returns_auditable_selection_receipt']=(
                status==200 and result.get('type')=='ambient_receipt'
                and type(selection.get('entropy')) is int and selection.get('pool_count')==len(candidates)
                and 0<=int(selection.get('index',-1))<len(candidates)
                and len(str(selection.get('pool_sha256','')))==64 and len(str(selection.get('sha256','')))==64)
            checks['gateway_response_does_not_echo_candidate_text_or_pool']=(
                all(row['text'] not in wire for row in candidates) and 'candidates' not in result)
            checks['selected_post_and_drain_are_two_append_only_events_on_same_runtime']=(
                body.server.runtime.cursor==2 and len(result.get('processed',()))==1
                and body.server.runtime.ambient_stream.pending_count==0)
            stored=json.loads(body.path.read_text())
            restored=ReferenceLifeFunctionRuntimeV2.restore(canonical_species_program_v2(),stored)
            checks['ambient_http_contact_persists_exact_same_life_checkpoint']=(
                restored.checkpoint()==body.server.runtime.checkpoint())
        finally:body.stop()
    failed=sorted(name for name,value in checks.items() if not value)
    result={'schema':'cyber-lagoon.authorized-ambient-gateway.v1','contract':'FOUNDRY_AUTHORIZED_AMBIENT_GATEWAY_'+('GREEN' if not failed else 'RED'),'pass':not failed,'reference_only':True,'runtime_llm':False,'checks':checks,'failed':failed,'elapsed_ms':round((time.perf_counter()-started)*1000,3)}
    print(result['contract']);print(json.dumps(result,indent=2,sort_keys=True));return 0 if not failed else 1


if __name__=='__main__':raise SystemExit(main())
