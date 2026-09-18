# Contract: `ttir` parser

**Module**: `src/tritonflow/ttir/` (`lexer.py`, `parser.py`, `ssa.py`, `graph.py`)
**Consumers**: every downstream stage. **Producer**: frozen fixtures and `torch.compile`.
**Requirements**: FR-001, FR-002, FR-003, FR-004, FR-019.

## Interface

```python
parse_module(text: str, *, source_path: str = "<string>") -> ParseResult

class ParseResult:            # exactly one of the two is set
    module: Module | None
    diagnostic: ParseDiagnostic | None      # kind == "PARSE_UNSUPPORTED"

def parse_file(path: str | os.PathLike[str]) -> ParseResult
def build_def_use(module: Module) -> DefUseGraph
def walk_region(region: Region) -> Iterator[Operation]      # region-aware, depth-first
def topo_within_region(region: Region) -> list[Operation]   # per-region only, never across
```

`ParseDiagnostic` = `{kind, line, col, expected, found, snippet}`.

## Preconditions

- `text` is a `ttir` module emitted by the pinned Triton, or arbitrary/adversarial text.
- Nothing else. The parser must not require a live Triton install (FR-034).

## Postconditions

1. **Total.** For any input string, `parse_module` returns a `ParseResult`. It never raises, and it never
   returns `module=None` and `diagnostic=None` (FR-001).
2. **Multi-result binding.** `Operation.results` contains *every* result. The Tier-1 loop is
   `%acc_25:3 = scf.for …` — three results, not one (FR-002).
3. **Loc preservation.** `loc("a_ptrs")` yields `Operation.loc.name == "a_ptrs"`, and the module keeps the
   `#loc` table. Malformed loc syntax degrades to `loc=None`; it does not fail the parse.
4. **Region preservation.** Nested regions remain nested. `walk_region` descends into them; nothing flattens
   structure (FR-019).
5. **Determinism.** The same bytes produce the same structure and the same ordering, across processes and
   `PYTHONHASHSEED` values (FR-004).
6. **Traversal is local.** `topo_within_region` never moves an operation out of its region. A `tt.load` inside
   `scf.for` stays inside it (T-4, D10).

## Failure modes

| Input | Required behaviour |
|---|---|
| Unknown op name in a known grammar position | parse succeeds, op retained verbatim (lowering will later mark it `UNSUPPORTED`) |
| Unterminated attribute dict, stray `{`, bad `#loc` | `PARSE_UNSUPPORTED` with line and column; no partial module |
| Multi-result op with mismatched arity on use | `PARSE_UNSUPPORTED` |
| Empty module / empty function | valid `Module`, zero operations |
| Enormous input (>10 MB) | either parses or returns `PARSE_UNSUPPORTED`; no unbounded memory growth |

## Tests

- **Contract**: all four fixtures parse; `diagnostic is None`; op counts match the recorded snapshot.
- **Unit**: `%acc_25:3` binds 3 results. `loc("a_ptrs")` recovered. Nested `scf.for` walks depth-first.
- **Determinism**: parse each fixture 3× in separate processes; assert byte-identical `repr` of the module.
- **Negative**: `tests/unit/test_parser_negative.py` — 30+ malformed inputs, each asserting
  `PARSE_UNSUPPORTED` and no exception (feeds SC-008).
