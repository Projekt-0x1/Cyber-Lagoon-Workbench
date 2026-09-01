#!/usr/bin/env python3
"""Bounded numeric relation IR unfolded from compact resident Recipes."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json


IR_VERSION = 0x52524931
MAX_INPUTS = 8
MAX_CONSTANTS = 8
MAX_INSTRUCTIONS = 8
MAX_BINDINGS = 512
MAX_BINDING_SOURCE_REFS = 4096
MAX_EXECUTION_WORK = 8192
OP_HALT = 0
OP_REQUIRE_INPUT_CONSTANT = 1
OP_REQUIRE_UNIQUE_BINDING = 2
OP_ACCUMULATE_PRODUCT = 3


class ResidentRelationIrRefuse(RuntimeError):
    pass


def _canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _identity(domain: bytes, values) -> int:
    value = int.from_bytes(hashlib.sha256(
        domain + b"\0" + _canonical(values)).digest()[:8], "little") & ((1 << 63) - 1)
    return value or 1


def _numeric(value, path="$"):
    if isinstance(value, bool) or not isinstance(value, (int, tuple, list)):
        raise ResidentRelationIrRefuse(f"relation_ir:type:{path}")
    if isinstance(value, int):
        if not -(1 << 63) <= value < (1 << 63):
            raise ResidentRelationIrRefuse(f"relation_ir:integer:{path}")
        return
    if len(value) > MAX_BINDINGS:
        raise ResidentRelationIrRefuse(f"relation_ir:bound:{path}")
    for index, row in enumerate(value):
        _numeric(row, f"{path}[{index}]")


@dataclass(frozen=True)
class ResidentRelationInstructionV1:
    opcode: int
    first: int
    second: int


@dataclass(frozen=True)
class ResidentRelationIrProgramV1:
    identity: int
    version: int
    input_arity: int
    constants: tuple[int, ...]
    instructions: tuple[ResidentRelationInstructionV1, ...]
    witness_roots: tuple[int, ...]
    source_roots: tuple[int, ...]


@dataclass(frozen=True)
class ResidentRelationBindingV1:
    root: int
    value: int
    source_roots: tuple[int, ...]


@dataclass(frozen=True)
class ResidentRelationFrameV1:
    inputs: tuple[int, ...]
    bindings: tuple[ResidentRelationBindingV1, ...]


@dataclass(frozen=True)
class ResidentRelationResultV1:
    identity: int
    program: int
    value: int
    work: int
    binding_roots: tuple[int, ...]
    source_roots: tuple[int, ...]


def _program_values(program):
    return (program.version, program.input_arity, program.constants,
            tuple((row.opcode, row.first, row.second)
                  for row in program.instructions),
            program.witness_roots, program.source_roots)


def unfold_bound_relation_v1(span_recipe: int, mapping_root: int, credit: int,
                             witness_roots, source_roots):
    constants = (int(span_recipe), int(mapping_root), int(credit), 4)
    instructions = (
        ResidentRelationInstructionV1(OP_REQUIRE_INPUT_CONSTANT, 0, 0),
        ResidentRelationInstructionV1(OP_REQUIRE_INPUT_CONSTANT, 1, 1),
        ResidentRelationInstructionV1(OP_REQUIRE_UNIQUE_BINDING, 2, 0),
        ResidentRelationInstructionV1(OP_ACCUMULATE_PRODUCT, 2, 3),
        ResidentRelationInstructionV1(OP_HALT, 0, 0),
    )
    roots = tuple(sorted(map(int, witness_roots)))
    sources = tuple(sorted(map(int, source_roots)))
    shell = ResidentRelationIrProgramV1(
        0, IR_VERSION, 3, constants, instructions, roots, sources)
    program = ResidentRelationIrProgramV1(
        _identity(b"resident-relation-ir-program-v1", _program_values(shell)),
        shell.version, shell.input_arity, shell.constants, shell.instructions,
        shell.witness_roots, shell.source_roots)
    validate_program_v1(program)
    return program


def validate_program_v1(program):
    if not isinstance(program, ResidentRelationIrProgramV1):
        raise ResidentRelationIrRefuse("relation_ir:program_type")
    if (not isinstance(program.constants, tuple)
            or not isinstance(program.instructions, tuple)
            or not isinstance(program.witness_roots, tuple)
            or not isinstance(program.source_roots, tuple)
            or not all(isinstance(row, ResidentRelationInstructionV1)
                       for row in program.instructions)):
        raise ResidentRelationIrRefuse("relation_ir:instruction_type")
    _numeric((program.identity, program.version, program.input_arity,
              program.constants,
              tuple((row.opcode, row.first, row.second)
                    for row in program.instructions),
              program.witness_roots, program.source_roots))
    expected_ops = (
        (OP_REQUIRE_INPUT_CONSTANT, 0, 0),
        (OP_REQUIRE_INPUT_CONSTANT, 1, 1),
        (OP_REQUIRE_UNIQUE_BINDING, 2, 0),
        (OP_ACCUMULATE_PRODUCT, 2, 3),
        (OP_HALT, 0, 0),
    )
    if (program.version != IR_VERSION or program.input_arity != 3
            or len(program.constants) != 4
            or program.constants[0] <= 0 or program.constants[1] <= 0
            or program.constants[2] <= 0 or program.constants[3] != 4
            or not 2 <= len(program.witness_roots) <= MAX_BINDINGS
            or not program.source_roots
            or tuple((row.opcode, row.first, row.second)
                     for row in program.instructions) != expected_ops
            or len(set(program.witness_roots)) != len(program.witness_roots)
            or len(set(program.source_roots)) != len(program.source_roots)
            or tuple(sorted(program.witness_roots)) != program.witness_roots
            or tuple(sorted(program.source_roots)) != program.source_roots
            or program.identity != _identity(
                b"resident-relation-ir-program-v1", _program_values(program))):
        raise ResidentRelationIrRefuse("relation_ir:program")
def execute_relation_ir_v1(program, frame):
    validate_program_v1(program)
    if (not isinstance(frame, ResidentRelationFrameV1)
            or not isinstance(frame.inputs, tuple)
            or not isinstance(frame.bindings, tuple)
            or not all(isinstance(row, ResidentRelationBindingV1)
                       for row in frame.bindings)):
        raise ResidentRelationIrRefuse("relation_ir:frame_type")
    if (len(frame.inputs) != program.input_arity
            or len(frame.bindings) > MAX_BINDINGS
            or sum(len(row.source_roots) for row in frame.bindings)
            > MAX_BINDING_SOURCE_REFS):
        raise ResidentRelationIrRefuse("relation_ir:frame")
    _numeric((frame.inputs, tuple((row.root, row.value, row.source_roots)
                                  for row in frame.bindings)))
    bindings = {row.root: row for row in frame.bindings}
    if (len(bindings) != len(frame.bindings)
            or set(bindings) != set(program.witness_roots)
            or any(not row.source_roots for row in frame.bindings)
            or any(not set(row.source_roots).issubset(program.source_roots)
                   for row in frame.bindings)
            or {source for row in frame.bindings for source in row.source_roots}
            != set(program.source_roots)):
        raise ResidentRelationIrRefuse("relation_ir:binding")
    work = (16 + len(program.constants) + len(program.instructions)
            + len(program.witness_roots) + len(program.source_roots)
            + len(frame.inputs) + 3 * len(frame.bindings)
            + sum(len(row.source_roots) for row in frame.bindings))
    if work > MAX_EXECUTION_WORK:
        raise ResidentRelationIrRefuse("relation_ir:work")
    value, used = 0, ()
    for row in program.instructions:
        if row.opcode == OP_HALT:
            break
        if row.opcode == OP_REQUIRE_INPUT_CONSTANT:
            if frame.inputs[row.first] != program.constants[row.second]:
                value = 0; used = (); break
        elif row.opcode == OP_REQUIRE_UNIQUE_BINDING:
            selected = [bindings[root] for root in program.witness_roots if root in bindings]
            values = {binding.value for binding in selected}
            if (len(selected) != len(program.witness_roots) or len(values) != 1
                    or frame.inputs[row.first] not in values):
                value = 0; used = (); break
            used = tuple(sorted(binding.root for binding in selected))
        elif row.opcode == OP_ACCUMULATE_PRODUCT:
            value += program.constants[row.first] * program.constants[row.second]
    sources = tuple(sorted({source for root in used
                            for source in bindings[root].source_roots}))
    result_values = (program.identity, value, work, used, sources)
    return ResidentRelationResultV1(
        _identity(b"resident-relation-ir-result-v1", result_values),
        program.identity, value, work, used, sources)
