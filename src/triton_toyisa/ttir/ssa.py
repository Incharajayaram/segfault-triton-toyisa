from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class TypeExpr:
    text: str

@dataclass
class Loc:
    name: str
    line: int | None = None
    col: int | None = None

@dataclass
class SsaValue:
    id: str
    type: TypeExpr
    # defining_op will be set during graph construction
    defining_op: Optional['Operation'] = None

@dataclass
class Operation:
    name: str
    results: list[SsaValue]
    operands: list[SsaValue]
    attrs: dict[str, str]
    regions: list['Region']
    loc: Loc | None

@dataclass
class Block:
    args: list[SsaValue]
    ops: list[Operation]

@dataclass
class Region:
    blocks: list[Block]

@dataclass
class Module:
    ops: list[Operation]
    loc_table: dict[str, Loc]
    source_path: str
