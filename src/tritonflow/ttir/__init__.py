"""ttir: the front end.

`lexer` and `parser` are Track A's (text-level, `RawModule`); `ssa`, `to_ir` and
`graph` are Track B's (semantic, `Module`). The seam between them is
`specs/001-triton-to-tritonflow/contracts/raw-module.md`.

No module in here may import Triton or torch (enforced by
`tools/check_test_map.py` law 3).
"""
