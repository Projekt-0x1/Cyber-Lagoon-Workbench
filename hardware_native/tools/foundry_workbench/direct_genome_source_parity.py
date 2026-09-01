#!/usr/bin/env python3
from __future__ import annotations
import re,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
import direct_genome_reference as ref

ROOT=Path(__file__).resolve().parents[3]
HEADER=ROOT/'hardware_native/src/hardware_native/direct_network_genome.cuh'

def uint_const(src,name):
    m=re.search(rf'\b{name}\s*=\s*([0-9]+)u',src)
    if not m: raise SystemExit(f'DIRECT_GENOME_SOURCE_PARITY RED missing:{name}')
    return int(m.group(1))

def main():
    s=HEADER.read_text()
    checks={
      'abi_v1':uint_const(s,'kDirectGenomeAbiV1')==1,
      'abi_v2':uint_const(s,'kDirectGenomeAbiV2')==ref.ABI_CURRENT,
      'max_territories':uint_const(s,'kDirectMaxTerritoriesV1')==ref.MAX_TERRITORIES,
      'max_fields':uint_const(s,'kDirectMaxFieldsV1')==ref.MAX_FIELDS,
      'max_rules':uint_const(s,'kDirectMaxRulesV1')==ref.MAX_RULES,
      'opcode_count':uint_const(s,'kDirectRuleOpcodeCountV1')==8,
    }
    expected={'extend':ref.EXTEND,'branch':ref.BRANCH,'fuse':ref.FUSE,'retract':ref.RETRACT,'mature':ref.MATURE,'repair':ref.REPAIR,'long_tract':ref.LONG_TRACT,'endogenous_source':ref.ENDOGENOUS_SOURCE}
    for name,value in expected.items():
        m=re.search(rf'\b{name}\s*=\s*([0-9]+)u',s)
        checks['opcode_'+name]=bool(m) and int(m.group(1))==value
    # The reference must model the same three authored arrays, not a mature brain blob.
    checks['container_shape']=all(x in s for x in ('DirectTerritorySpecV1 territories[kDirectMaxTerritoriesV1]','DirectFieldSpecV1 fields[kDirectMaxFieldsV1]','DirectRuleSpecV1 rules[kDirectMaxRulesV1]'))
    ok=all(checks.values())
    print('DIRECT_GENOME_SOURCE_PARITY '+('GREEN' if ok else 'RED')+' '+ ' '.join(f'{k}={int(v)}' for k,v in checks.items()))
    raise SystemExit(0 if ok else 1)
if __name__=='__main__':main()
