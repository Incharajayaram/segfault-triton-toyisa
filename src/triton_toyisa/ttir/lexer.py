import re
from collections.abc import Iterator
from dataclasses import dataclass


@dataclass(frozen=True)
class Token:
    kind: str       # e.g., 'SSA_ID', 'BARE_ID', 'STRING', 'NUMBER', 'PUNCT', 'NEWLINE'
    value: str      # The actual text matched
    line: int
    col: int

class Lexer:
    TOKEN_REGEX = re.compile(
        r'(?P<COMMENT>//[^\n]*(?:\n|$))|'
        r'(?P<STRING>"[^"\\]*(?:\\.[^"\\]*)*")|'
        r'(?P<SSA_ID>%[A-Za-z0-9_]+(?::\d+)?(?:#\d+)?)|'
        r'(?P<SYM_ID>@[A-Za-z_][A-Za-z0-9_]*)|'
        r'(?P<LOC_ID>#[A-Za-z_][A-Za-z0-9_]*)|'
        r'(?P<BARE_ID>[a-zA-Z_!][a-zA-Z0-9_$.]*)|'
        r'(?P<NUMBER>[-+]?[0-9]+(?:\.[0-9]+)?(?:[eE][-+]?[0-9]+)?)|'
        r'(?P<PUNCT>->|==|!=|<=|>=|[=:,<>{}\(\)\[\]+*?/#|-])|'
        r'(?P<NEWLINE>\n)|'
        r'(?P<WHITESPACE>[ \t\r\f\v]+)|'
        r'(?P<MISMATCH>.)'
    )

    def __init__(self, text: str):
        # Strip BOM if present
        self.text = text.lstrip('\ufeff').lstrip('\xef\xbb\xbf')

    def tokenize(self) -> Iterator[Token]:
        line_num = 1
        line_start = 0
        
        for mo in self.TOKEN_REGEX.finditer(self.text):
            kind = mo.lastgroup
            value = mo.group(kind)
            
            if kind == 'WHITESPACE':
                continue
                
            if kind == 'COMMENT':
                line_num += 1
                line_start = mo.end()
                continue
                
            if kind == 'NEWLINE':
                yield Token(kind, value, line_num, mo.start() - line_start + 1)
                line_num += 1
                line_start = mo.end()
                continue
                
            if kind == 'MISMATCH':
                yield Token('ERROR', value, line_num, mo.start() - line_start + 1)
                continue
                
            yield Token(kind, value, line_num, mo.start() - line_start + 1)
