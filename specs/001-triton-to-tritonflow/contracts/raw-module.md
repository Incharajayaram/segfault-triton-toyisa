# Contract: `RawModule` — the parser / IR-model seam

**Modules**: `ttir/parser.py` (producer), `ttir/to_ir.py` (consumer)
**Owners**: Track A (parser) produces it; Track B (IR) consumes it.
**Why this exists**: parsing and IR modelling are different skills and, in this team, different people. This
is the interface that lets them work at the same time without editing the same file. It is **frozen before
lunch on day 1** and it is the first item on the freeze list.
**Requirements served**: FR-001, FR-002, FR-003, FR-019.

## The division of labour, precisely

| Concern | Owner | Why |
|---|---|---|
| Lexing, tokens, line/col tracking | Track A (parser) | text-level |
| Attribute syntax, nested dicts, `dense<…>` | Track A | text-level |
| `#loc` table syntax, inline `loc(…)` syntax | Track A | text-level |
| Region/block/brace nesting *as text* | Track A | text-level |
| Error recovery: unterminated input, bad token, truncated module | Track A | text-level |
| **Type text → `TypeExpr`** (shape, dtype, address space) | Track B (IR) | semantic |
| **SSA name → `SsaValue`** resolution, value numbering | Track B | semantic |
| **Multi-result binding** (`%acc_25:3` → three values) | Track B | semantic |
| Region nesting → `Region`/`Block` objects, block args | Track B | semantic |
| `loc` resolution against the table | Track B | semantic |
| Duplicate SSA name, operand arity mismatch, undefined operand | Track B | structural-invalidity, not syntax |

**The rule**: Track A's output contains **strings, not semantics**. Types stay text. Attribute values stay
text. Operands and results stay SSA names as written. Track A never decides what a type *means*; Track B never
looks at a character of source.

## Interface

```python
# ttir/parser.py  (Track A)
parse_raw(text: str, *, source_path: str = "<string>") -> RawModule

# ttir/to_ir.py   (Track B)
build_ir(raw: RawModule, *, raise_on_invalid: bool = False) -> ParseResult
```

### Structures (frozen day 1, field definitions only)

```python
@dataclass(frozen=True)
class RawOp:
    name: str                          # "tt.load", "scf.for", "arith.muli" — verbatim
    results: list[str | None]          # SSA names as written: ["%acc"], or ["%a", None] for a discarded result
    operands: list[str]                # SSA names as written: ["%a_ptrs_34"]
    attrs: dict[str, str]              # values as written: {"inputPrecision": "tf32"}
    result_types: list[str]            # type text, unresolved: ["tensor<64x64xf32>"]
    operand_types: list[str]           # type text, unresolved
    regions: list[RawRegion]
    loc: RawLoc | None
    line: int
    col: int

@dataclass(frozen=True)
class RawRegion:
    blocks: list[RawBlock]

@dataclass(frozen=True)
class RawBlock:
    args: list[tuple[str, str]]        # (SSA name, type text)
    ops: list[RawOp]                   # includes the terminator as the last element
    terminator_index: int

@dataclass(frozen=True)
class RawLoc:
    name: str
    line: int | None = None
    col: int | None = None

@dataclass(frozen=True)
class RawModule:
    ops: list[RawOp]                   # top level
    loc_table: dict[str, RawLoc]
    source_path: str
    triton_version: str | None
    diagnostics: list[ParseDiagnostic] # non-empty only for *syntactic* failures
```

## Postconditions

1. **Total, no exceptions.** `parse_raw` returns for any string. It never raises and never loops. The only way
   it reports a problem is `diagnostics`.
2. **Text fidelity.** `RawOp.name`, SSA names, type text and attribute values are byte-for-byte what was read.
   No normalisation happens here — normalisation is Track B's job, and doing it in two places is how two
   dialects appear.
3. **Spans are recorded.** Every `RawOp` carries `line`/`col` of its first token. Diagnostics that name a
   position must be accurate to the line, and to the column where it is cheap. This is what makes
   `PARSE_UNSUPPORTED(line, col)` honest instead of decorative.
4. **Structure is preserved as text-level nesting.** A `RawRegion` per `{ … }`. A nested `scf.if` inside an
   `scf.for` produces nested `RawRegion`s. Flattening here would destroy the one thing Track B needs.
5. **Terminator is not special-cased.** It is an operation in the block with its index recorded. Track A does
   not know that `scf.yield` is a terminator; Track B does.
6. **Multi-result arity is carried, not judged.** `%acc_25:3` produces one `RawOp` whose `results` has three
   entries — with placeholder names if the text omits them (`["%acc_25", "%acc_25#1", "%acc_25#2"]`). Whether
   a *use* has the right arity is an invalid-IR error and belongs to Track B (EC-005).
7. **Deterministic.** Same bytes → equal `RawModule`. No set iteration, no dict-ordering dependence.
8. **`diagnostics` is empty on success.** A non-empty `diagnostics` means the module is unusable: Track B
   refuses it and returns the diagnostic rather than a partial IR (FR-001).

## The two-layer failure route

Both layers report the same external marker, and the contract fixes which layer owns which class, so that
neither owner is tempted to fix the other's bug or to blame it:

| Failure | Layer | Example |
|---|---|---|
| Unterminated attribute dict / brace | A (syntax) | `{a = 1` at EOF |
| Unknown token where a grammar position expects one | A (syntax) | truncated module (EC-025) |
| Deep nesting beyond a stated limit | A (syntax) | 200-deep regions (EC-027) |
| Type text that is not a valid type | B (semantic) | `tensor<64x64x???>` |
| Operand references a value that does not exist | B (semantic) | `%typo` |
| Duplicate SSA name for two definitions | B (semantic) | EC-026 |
| Multi-result used with the wrong arity | B (semantic) | EC-005 |
| `loc` key missing from the table | B (semantic) | `loc(#loc99)` with no `#loc99` |

```python
ParseDiagnostic(kind="PARSE_UNSUPPORTED", line=…, col=…, expected=…, found=…, snippet=…, layer="syntax"|"ir")
```

The `layer` field exists so a failing test immediately names the owning track.

## Tests

- **Track A's contract test** (`tests/contract/test_raw_module.py`): the four fixtures produce a `RawModule`
  with the recorded op counts; type text and attribute values are verbatim; spans are correct; 30+ malformed
  inputs produce `diagnostics` with accurate lines and no exception.
- **Track B's contract test** (`tests/contract/test_to_ir.py`): `build_ir` on a `RawModule` produces a `Module`
  whose `SsaValue`s resolve, whose multi-result ops bind three values, whose regions nest, and whose `loc`s
  bind to names. Built **by hand** on day 1 (no parser needed), then re-run against Track A's real output the
  moment it lands.
- **The seam test** (`tests/integration/test_parser_to_ir.py`, owned by Track B, run from day 2): 
  `build_ir(parse_raw(fixture))` equals the structure recorded in `data-model.md` §2 — op counts, region
  nesting, multi-result arity, `loc` names recovered.
- **Both-input rule** (from `parallelism.md` Discipline 2): Track B's contract test is parametrised
  `["handbuilt", "real"]`. The `real` row is `xfail` until Track A lands, and a hard failure after that. By end
  of day 3 no contract test may run only on a hand-built input.

## Why this layer is worth its hour

Without it, the parser is one 1.8-day file that one person owns, and the critical path's first link is a
single point of failure and a single point of idleness. With it, the front end is two halves that proceed
simultaneously, the specialists keep their own files, and the error classes are assigned rather than
negotiated at integration time. The cost is one frozen dataclass set and the discipline of keeping text as
text in Track A.
