from dataclasses import dataclass

from triton_toyisa.ttir.lexer import Lexer, Token


@dataclass(frozen=True)
class RawLoc:
    name: str
    line: int | None = None
    col: int | None = None


@dataclass(frozen=True)
class RawBlock:
    args: list[tuple[str, str]]
    ops: list["RawOp"]
    terminator_index: int


@dataclass(frozen=True)
class RawRegion:
    blocks: list[RawBlock]


@dataclass(frozen=True)
class RawOp:
    name: str
    results: list[str | None]
    operands: list[str]
    attrs: dict[str, str]
    result_types: list[str]
    operand_types: list[str]
    regions: list[RawRegion]
    loc: RawLoc | None
    line: int
    col: int


@dataclass(frozen=True)
class ParseDiagnostic:
    kind: str
    line: int | None
    col: int | None
    expected: str | None
    found: str | None
    snippet: str | None
    layer: str


@dataclass(frozen=True)
class RawModule:
    ops: list[RawOp]
    loc_table: dict[str, RawLoc]
    source_path: str
    triton_version: str | None
    diagnostics: list[ParseDiagnostic]


class Parser:
    def __init__(self, text: str, source_path: str):
        self.lexer = Lexer(text)
        self.tokens = list(self.lexer.tokenize())
        self.pos = 0
        self.source_path = source_path
        self.diagnostics = []
        self.loc_table = {}
        self.depth = 0

    def peek(self) -> Token | None:
        if self.pos < len(self.tokens):
            return self.tokens[self.pos]
        return None

    def advance(self) -> Token | None:
        if self.pos < len(self.tokens):
            t = self.tokens[self.pos]
            self.pos += 1
            return t
        return None

    def error(self, msg: str, t: Token | None = None):
        if not t:
            t = self.peek()
        line = t.line if t else None
        col = t.col if t else None
        found = t.value if t else "EOF"
        self.diagnostics.append(
            ParseDiagnostic(
                kind="PARSE_UNSUPPORTED",
                line=line,
                col=col,
                expected=None,
                found=found,
                snippet=msg,
                layer="syntax",
            )
        )

    def skip_newlines(self):
        while self.peek() and self.peek().kind == "NEWLINE":
            self.advance()

    def parse_module(self) -> RawModule:
        ops = []

        while self.peek():
            self.skip_newlines()
            if not self.peek():
                break

            # Check for loc table definitions
            if self.peek().kind == "LOC_ID" and self.peek().value.startswith("#loc"):
                self.parse_loc_def_line()
                continue

            # Parse normal ops
            op = self.parse_op()
            if op:
                ops.append(op)
            else:
                # If parse_op failed, advance to avoid infinite loop
                self.advance()

        return RawModule(
            ops=ops,
            loc_table=self.loc_table,
            source_path=self.source_path,
            triton_version=None,
            diagnostics=self.diagnostics,
        )

    def parse_loc_def_line(self):
        # #loc1 = loc("path":1:2)
        loc_id = self.advance()
        line_tokens = []
        while self.peek() and self.peek().kind != "NEWLINE":
            line_tokens.append(self.advance())

        # Parse it
        if (
            len(line_tokens) >= 3
            and line_tokens[0].value == "="
            and line_tokens[1].value == "loc"
            and line_tokens[2].value == "("
        ):
            if (
                len(line_tokens) >= 8
                and line_tokens[3].kind == "STRING"
                and line_tokens[4].value == ":"
            ):
                self.loc_table[loc_id.value] = RawLoc(
                    name=loc_id.value, line=int(line_tokens[5].value), col=int(line_tokens[7].value)
                )
            else:
                self.loc_table[loc_id.value] = RawLoc(name=loc_id.value)

    def parse_op(self) -> RawOp | None:
        self.skip_newlines()
        start_tok = self.peek()
        if not start_tok:
            return None
        if start_tok.value == "}":
            return None

        if self.depth > 200:
            self.error("Nesting limit exceeded", start_tok)
            return None

        line_tokens = []
        has_region = False

        # Read the operation definition (up to NEWLINE or region '{')
        paren_depth = 0
        brace_depth = 0
        bracket_depth = 0

        while self.peek():
            t = self.peek()

            if t.value == "(":
                paren_depth += 1
            elif t.value == ")":
                paren_depth -= 1
            elif t.value == "[":
                bracket_depth += 1
            elif t.value == "]":
                bracket_depth -= 1
            elif t.value == "{":
                brace_depth += 1
            elif t.value == "}":
                brace_depth -= 1

            if t.kind == "NEWLINE":
                if paren_depth == 0 and brace_depth == 0 and bracket_depth == 0:
                    self.advance()
                    break
                else:
                    # Inside a nested structure, newline is just a token
                    pass

            # Check if this '{' starts a region.
            # A region '{' in Triton is usually the very last token on the line.
            if t.value == "{" and paren_depth == 0 and bracket_depth == 0 and brace_depth == 1:
                # Peek ahead to see if it's the end of the line
                next_t = self.tokens[self.pos + 1] if self.pos + 1 < len(self.tokens) else None
                if not next_t or next_t.kind == "NEWLINE":
                    has_region = True
                    self.advance()  # consume '{'
                    brace_depth -= 1
                    # consume the newline too if present
                    if self.peek() and self.peek().kind == "NEWLINE":
                        self.advance()
                    break

            if t.kind == "ERROR":
                self.error(f"Unknown token: {t.value}", t)

            line_tokens.append(self.advance())

        if brace_depth > 0 or paren_depth > 0 or bracket_depth > 0:
            self.error("Unterminated attribute dict or tuple", start_tok)

        if not line_tokens:
            return None

        # 1. Parse results and name
        results = []
        name_tok = None

        idx = 0
        while idx < len(line_tokens):
            if line_tokens[idx].value == "=":
                idx += 1
                break
            idx += 1

        if idx < len(line_tokens) and "=" in [t.value for t in line_tokens]:
            # We have results
            res_idx = 0
            while res_idx < idx - 1:
                t = line_tokens[res_idx]
                if t.kind == "SSA_ID":
                    if ":" in t.value:
                        base, count = t.value.split(":")
                        for i in range(int(count)):
                            results.append(f"{base}#{i}")
                    else:
                        results.append(t.value)
                res_idx += 1

            name_tok = line_tokens[idx]
            idx += 1
        else:
            # No results, first token is name
            idx = 0
            name_tok = line_tokens[idx]
            idx += 1

        if not name_tok or (name_tok.kind not in ("BARE_ID", "STRING")):
            self.error("Expected operation name", start_tok)
            return None

        # 2. Extract operands (any SSA_ID in the remainder that isn't inside a nested loc)
        operands = []
        paren_depth = 0
        loc_tok = None

        # To find the trailing loc, we look from the right
        # loc ( #loc1 )
        right_idx = len(line_tokens) - 1
        if (
            right_idx >= 3
            and line_tokens[right_idx].value == ")"
            and line_tokens[right_idx - 2].value == "("
            and line_tokens[right_idx - 3].value == "loc"
        ):
                loc_tok = line_tokens[right_idx - 1]
                line_tokens = line_tokens[: right_idx - 3]  # truncate the loc

        for i in range(idx, len(line_tokens)):
            t = line_tokens[i]
            if t.value == "(":
                paren_depth += 1
            elif t.value == ")":
                paren_depth -= 1
            elif t.kind == "SSA_ID":
                operands.append(t.value)

        # 3. Parse Region if present
        regions = []
        if has_region:
            self.depth += 1
            region_ops = []
            while self.peek():
                self.skip_newlines()
                if self.peek() and self.peek().value == "}":
                    break
                # loc defs can appear inside regions
                if self.peek().kind == "LOC_ID" and self.peek().value.startswith("#loc"):
                    self.parse_loc_def_line()
                    continue

                child = self.parse_op()
                if child:
                    region_ops.append(child)
                else:
                    self.advance()

            end_brace = None
            if self.peek() and self.peek().value == "}":
                end_brace = self.advance()
            self.depth -= 1

            # Read trailing loc if present after region
            while self.peek() and self.peek().kind != "NEWLINE":
                t = self.advance()
                if t.value == "loc" and self.peek() and self.peek().value == "(":
                    self.advance()  # (
                    lt = None
                    if self.peek() and self.peek().kind == "LOC_ID":
                        lt = self.advance()
                    if lt:
                        loc_tok = lt
                    self.advance()  # )

            if not end_brace:
                # If we couldn't find the closing brace, we hit EOF (or some other error).
                # Report on EOF if we are at the end, else on the current token.
                self.error("Expected '}' to close region", None if not self.peek() else self.peek())

            regions.append(
                RawRegion(
                    blocks=[
                        RawBlock(
                            args=[],
                            ops=region_ops,
                            terminator_index=len(region_ops) - 1 if region_ops else 0,
                        )
                    ]
                )
            )

        return RawOp(
            name=name_tok.value,
            results=results,
            operands=operands,
            attrs={},  # Intentionally left blank for this level of parser
            result_types=[],
            operand_types=[],
            regions=regions,
            loc=RawLoc(name=loc_tok.value) if loc_tok else None,
            line=name_tok.line,
            col=name_tok.col,
        )


def parse_raw(text: str, *, source_path: str = "<string>") -> RawModule:
    parser = Parser(text, source_path)
    return parser.parse_module()
