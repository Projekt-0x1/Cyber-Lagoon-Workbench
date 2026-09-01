#!/usr/bin/env python3
from __future__ import annotations

import copy
import json
from reference_nonlocal_port_dependency_v1 import NonlocalPortDependencyV1


def u(text):
    return tuple(text.encode())


def learned(port_a=11, port_b=29):
    law = NonlocalPortDependencyV1()
    law.observe(101, 501, 1, 700, port_a, port_b, (3, 5, 8), u("dax"), u("daxes"))
    law.observe(102, 502, 2, 700, port_a, port_b, (5, 8, 13), u("mip"), u("mipes"))
    return law


def main():
    law = learned()
    result = law.apply(700, 11, 29, (5, 8, 21), u("nov"))
    checks = {
        "nonce_transfer": result is not None and bytes(result.units) == b"noves",
        "distinct_action_contributor": result is not None and result.recipe_identity > 0,
        "per_byte_ancestry": result is not None and len(result.byte_ancestry) == len(result.units)
                              and all(row == (101, 102) for row in result.byte_ancestry),
        "controller_lesion": law.apply(700, 11, 29, (3, 21), u("nov")) is None,
        "remote_sham": law.apply(700, 11, 29, (5, 8, 999), u("nov")) is not None,
        "role_swap_refusal": law.apply(700, 29, 11, (5, 8), u("nov")) is None,
    }
    one = NonlocalPortDependencyV1()
    one.observe(201, 601, 1, 700, 11, 29, (5, 8), u("dax"), u("daxes"))
    one.observe(202, 601, 2, 700, 11, 29, (5, 8), u("mip"), u("mipes"))
    checks["single_source_repetition_refusal"] = one.apply(700, 11, 29, (5, 8), u("nov")) is None

    withdrawn = learned(); withdrawn.withdraw_source(501)
    checks["source_withdrawal"] = withdrawn.apply(700, 11, 29, (5, 8), u("nov")) is None

    ambiguous = learned()
    ambiguous.observe(301, 503, 3, 700, 11, 29, (5, 8), u("dax"), u("daxum"))
    ambiguous.observe(302, 504, 4, 700, 11, 29, (5, 8), u("mip"), u("mipum"))
    checks["equal_alternative_refusal"] = ambiguous.apply(700, 11, 29, (5, 8), u("nov")) is None

    replay = NonlocalPortDependencyV1.restore(law.checkpoint())
    checks["checkpoint_exact"] = replay.checkpoint() == law.checkpoint() and replay.apply(
        700, 11, 29, (5, 8), u("nov")) == result

    permuted = learned(407, 103)
    changed = permuted.apply(700, 407, 103, (5, 8), u("nov"))
    checks["opaque_port_permutation"] = changed is not None and changed.units == result.units

    try:
        NonlocalPortDependencyV1.restore(json.dumps({"schema": 1, "observations": [],
                                                     "expected_output": [1]}))
        checks["literal_bypass_refusal"] = False
    except Exception:
        checks["literal_bypass_refusal"] = True

    receipt = {
        "schema": "0x1.reference-nonlocal-port-dependency.v1",
        "pass": all(checks.values()), "checks": checks,
        "reference_only": True, "adult_attached": False, "graph_flip": False,
        "physical_direct_parity": "NOT_RUN/RED", "runtime_llm": False,
        "claim": "GENERIC_NONLOCAL_PORT_DEPENDENCY_REFERENCE_PROPERTY_ONLY",
    }
    print("FOUNDRY_NONLOCAL_PORT_DEPENDENCY " + ("GREEN" if receipt["pass"] else "RED"))
    print(json.dumps(receipt, indent=2, sort_keys=True))
    raise SystemExit(0 if receipt["pass"] else 1)


if __name__ == "__main__":
    main()
