#!/usr/bin/env python3
"""Graph-native Foundry control surface.

Foundry may inspect the constitutional frontier, bind reference evidence to the
owning issue, and demote contradicted claims. RED->GREEN promotion is deliberately
not exposed here; it remains owned by the node's named production contract.
"""
from __future__ import annotations
import argparse, hashlib, json, re, subprocess, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
GRAPH=ROOT/'docs/constitutional_dependency_graph.json'
sys.path.insert(0,str(ROOT/'tools'))
import validate_constitutional_dependency_graph as gv

def load():
 d=json.loads(GRAPH.read_text());idx=gv._node_index(d);gv._validate_issues(d,idx);return d,idx

def cmd_frontier(a):
 d,idx=load();ready=set(gv.eligible_red_nodes(d));rows=[]
 for k in sorted(idx):
  n=idx[k]
  if n['status']!='RED':continue
  text=(k+' '+n['law']).lower()
  if a.match and a.match.lower() not in text:continue
  rows.append({'id':k,'eligible':k in ready,'status':n['status'],'issue':(n.get('work') or {}).get('issue'),'contract':(n.get('work') or {}).get('contract'),'red_requirements':[r for r in n['requires'] if idx[r]['status']!='GREEN']})
 print(json.dumps({'schema':'0x1.foundry-frontier.v1','rows':rows},indent=2,sort_keys=True))

def cmd_node(a):
 _,idx=load()
 if a.node not in idx:raise SystemExit('unknown node')
 print(json.dumps(idx[a.node],indent=2,sort_keys=True))

def cmd_audit(a):
 _,idx=load();node=idx['b.deterministic_blank_birth'];files=(node.get('work') or {}).get('files',())
 pats=(re.compile(r'DirectIndividualMicrostateV1'),re.compile(r'salt_lo|salt_hi'),re.compile(r'microstate_root'),re.compile(r'--xi-(?:lo|hi)'))
 hits=[]
 for rel in files:
  p=ROOT/rel
  if not p.is_file():continue
  for no,line in enumerate(p.read_text(errors='replace').splitlines(),1):
   if any(x.search(line) for x in pats):hits.append({'path':rel,'line':no,'text':line.strip()[:200]})
 acknowledged=(node['status']=='RED')
 ok=(not hits and node['status'] in ('RED','GREEN')) or (bool(hits) and acknowledged)
 result={'schema':'0x1.foundry-constitutional-audit.v1','pass':ok,'node':node['id'],'node_status':node['status'],'forbidden_seed_hit_count':len(hits),'examples':hits if a.verbose else hits[:8],'drift_acknowledged':bool(hits) and acknowledged,'ready_for_reverify':not hits and node['status']=='RED'}
 print(json.dumps(result,indent=2,sort_keys=True))
 raise SystemExit(0 if ok else 1)

def cmd_demote(a):
 d,idx=load()
 if a.node not in idx:raise SystemExit('unknown node')
 if not re.fullmatch(r'#\d+',a.issue):raise SystemExit('--issue must be #<number>')
 n=idx[a.node]
 if not n.get('work'):raise SystemExit('node has no work block')
 old=n['work']['issue'];n['status']='RED';n['work']['issue']=a.issue
 if old in d['issues']:
  d['issues'][old]=[x for x in d['issues'][old] if x!=a.node]
  if not d['issues'][old]:del d['issues'][old]
 d['issues'].setdefault(a.issue,[])
 if a.node not in d['issues'][a.issue]:d['issues'][a.issue].append(a.node)
 d['issues'][a.issue].sort();gv._node_index(d);gv._validate_issues(d,gv._node_index(d))
 if not a.write:
  print(json.dumps({'write':False,'node':a.node,'status':'RED','old_issue':old,'issue':a.issue},indent=2));return
 GRAPH.write_text(json.dumps(d,indent=2)+'\n');print(json.dumps({'write':True,'node':a.node,'status':'RED','issue':a.issue},indent=2))

def cmd_reference(a):
 _,idx=load()
 if a.node not in idx:raise SystemExit('unknown node')
 n=idx[a.node];issue=(n.get('work') or {}).get('issue')
 if not issue:raise SystemExit('node has no issue')
 receipt=None
 if a.receipt:
  p=Path(a.receipt)
  if not p.is_file():raise SystemExit('receipt missing')
  b=p.read_bytes();receipt=(str(p),hashlib.sha256(b).hexdigest(),len(b))
 body=['Foundry reference evidence (no capability promotion):','',f'- node: `{a.node}` (graph status `{n["status"]}`)',f'- summary: {a.summary}']
 if a.command:body.append(f'- command: `{a.command}`')
 if receipt:body.extend((f'- receipt: `{receipt[0]}`',f'- sha256: `{receipt[1]}`',f'- bytes: `{receipt[2]}`'))
 body.extend(('',"The node remains RED until its named production contract earns GREEN."));text='\n'.join(body)
 if not a.post:print(text);return
 subprocess.run(['gh','issue','comment',issue[1:],'--body',text],cwd=ROOT,check=True);print(json.dumps({'posted':True,'node':a.node,'issue':issue},indent=2))

def main():
 p=argparse.ArgumentParser();sub=p.add_subparsers(dest='cmd',required=True)
 q=sub.add_parser('frontier');q.add_argument('--match',default='');q.set_defaults(fn=cmd_frontier)
 q=sub.add_parser('node');q.add_argument('node');q.set_defaults(fn=cmd_node)
 q=sub.add_parser('audit');q.add_argument('--verbose',action='store_true');q.set_defaults(fn=cmd_audit)
 q=sub.add_parser('demote');q.add_argument('node');q.add_argument('--issue',required=True);q.add_argument('--write',action='store_true');q.set_defaults(fn=cmd_demote)
 q=sub.add_parser('reference');q.add_argument('node');q.add_argument('--summary',required=True);q.add_argument('--command',default='');q.add_argument('--receipt');q.add_argument('--post',action='store_true');q.set_defaults(fn=cmd_reference)
 a=p.parse_args();a.fn(a)
if __name__=='__main__':main()
