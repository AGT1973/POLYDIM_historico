import re
from typing import List, NamedTuple, Any

class Token(NamedTuple):
    type: str
    value: str
    line: int
    column: int

class LexerError(Exception):
    def __init__(self, message: str, line: int, column: int):
        super().__init__(f"LexerError at line {line}, col {column}: {message}")

class PolydimLexer:
    """
    Lexer for the POLYDIM programming language (.poly).
    POLYDIM is a categorical/topological language, so the tokens represent
    spaces, morphisms, and continuous flow operations rather than classical commands.
    """
    # Specification of tokens using regex
    # Keywords map to Topos and Geometric Algebra concepts
    KEYWORDS = {
        'Space', 'Morphism', 'Flow', 'Dim', 'Metric', 
        'apply', 'source', 'pipe', 'sink', 'MIX', 'PROJECT', 'FIXPOINT'
    }

    TOKEN_SPEC = [
        ('NUMBER',   r'\d+(\.\d*)?'),                  # Integer or decimal number
        ('ID',       r'[A-Za-z_][A-Za-z0-9_]*'),       # Identifiers
        ('ARROW',    r'->'),                           # Morphism direction
        ('COMPOSE',  r'∘|@'),                          # Morphism composition (circle or @)
        ('ASSIGN',   r'='),                            # Assignment
        ('COLON',    r':'),                            # Declaration separator
        ('COMMA',    r','),                            # Tuple separator
        ('LBRACE',   r'\{'),                           # Block start
        ('RBRACE',   r'\}'),                           # Block end
        ('LPAREN',   r'\('),                           # Param start
        ('RPAREN',   r'\)'),                           # Param end
        ('STRING',   r'"[^"]*"'),                      # String literals
        ('NEWLINE',  r'\n'),                           # Line endings
        ('SKIP',     r'[ \t]+'),                       # Spaces and tabs
        ('COMMENT',  r'//.*'),                         # Comments
        ('MISMATCH', r'.'),                            # Any other character
    ]

    def __init__(self):
        self.lexer_regex = '|'.join(f'(?P<{pair[0]}>{pair[1]})' for pair in self.TOKEN_SPEC)
        self.regex = re.compile(self.lexer_regex)

    def tokenize(self, code: str) -> List[Token]:
        tokens = []
        line_num = 1
        line_start = 0

        for mo in self.regex.finditer(code):
            kind = mo.lastgroup
            value = mo.group(kind)
            column = mo.start() - line_start

            if kind == 'NUMBER':
                value = float(value) if '.' in value else int(value)
            elif kind == 'ID' and value in self.KEYWORDS:
                kind = value
            elif kind == 'NEWLINE':
                line_start = mo.end()
                line_num += 1
                continue
            elif kind == 'SKIP' or kind == 'COMMENT':
                continue
            elif kind == 'MISMATCH':
                raise LexerError(f"Unexpected character {value!r}", line_num, column)
            
            tokens.append(Token(kind, value, line_num, column))
            
        return tokens

# Test básico de cordura del Lexer
if __name__ == '__main__':
    test_code = '''
    // Definition of a Topos manifold
    Space S1: Dim=1024, Metric="Clifford"
    Space S2: Dim=512, Metric="Clifford"

    Morphism M1: S1 -> S2 {
        apply PROJECT(basis="orthogonal")
    }

    Flow F1:
        source = S1
        pipe = M1 @ MIX(alpha=0.5)
        sink = S2
    '''
    lexer = PolydimLexer()
    for tok in lexer.tokenize(test_code):
        print(tok)
