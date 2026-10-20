from typing import List, Optional, Any
from .lexer import Token

class ASTNode:
    pass

class SpaceDef(ASTNode):
    def __init__(self, name: str, dim: int, metric: str):
        self.name = name
        self.dim = dim
        self.metric = metric

    def __repr__(self):
        return f"SpaceDef(name={self.name}, dim={self.dim}, metric={self.metric})"

class MorphismDef(ASTNode):
    def __init__(self, name: str, source: str, target: str, operations: List[str]):
        self.name = name
        self.source = source
        self.target = target
        self.operations = operations

    def __repr__(self):
        return f"MorphismDef(name={self.name}, {self.source}->{self.target}, ops={self.operations})"

class FlowDef(ASTNode):
    def __init__(self, name: str, source: str, pipe: List[str], sink: str):
        self.name = name
        self.source = source
        self.pipe = pipe
        self.sink = sink

    def __repr__(self):
        return f"FlowDef(name={self.name}, source={self.source}, pipe={self.pipe}, sink={self.sink})"

class ParserError(Exception):
    def __init__(self, message: str, token: Token):
        super().__init__(f"ParserError at line {token.line}, col {token.column}: {message}")

class PolydimParser:
    """
    Parser for POLYDIM. 
    Constructs a topological semantic graph (Simplicial Complex) instead of a standard execution tree.
    """
    def __init__(self, tokens: List[Token]):
        self.tokens = tokens
        self.pos = 0

    def peek(self) -> Optional[Token]:
        if self.pos < len(self.tokens):
            return self.tokens[self.pos]
        return None

    def consume(self, expected_type: str = None) -> Token:
        tok = self.peek()
        if not tok:
            raise Exception("Unexpected EOF")
        if expected_type and tok.type != expected_type:
            raise ParserError(f"Expected {expected_type}, got {tok.type}", tok)
        self.pos += 1
        return tok

    def parse(self) -> List[ASTNode]:
        nodes = []
        while self.peek() is not None:
            tok = self.peek()
            if tok.type == 'Space':
                nodes.append(self.parse_space())
            elif tok.type == 'Morphism':
                nodes.append(self.parse_morphism())
            elif tok.type == 'Flow':
                nodes.append(self.parse_flow())
            else:
                raise ParserError(f"Unknown top-level declaration starting with {tok.value}", tok)
        return nodes

    def parse_space(self) -> SpaceDef:
        self.consume('Space')
        name_tok = self.consume('ID')
        self.consume('COLON')
        self.consume('Dim')
        self.consume('ASSIGN')
        dim_tok = self.consume('NUMBER')
        self.consume('COMMA')
        self.consume('Metric')
        self.consume('ASSIGN')
        metric_tok = self.consume('STRING')
        
        # Quitamos las comillas del string
        metric_val = metric_tok.value.strip('"')
        return SpaceDef(name_tok.value, int(dim_tok.value), metric_val)

    def parse_morphism(self) -> MorphismDef:
        self.consume('Morphism')
        name_tok = self.consume('ID')
        self.consume('COLON')
        source_tok = self.consume('ID')
        self.consume('ARROW')
        target_tok = self.consume('ID')
        
        self.consume('LBRACE')
        operations = []
        while self.peek() and self.peek().type != 'RBRACE':
            self.consume('apply')
            op_tok = self.consume() # Puede ser PROJECT, MIX, FIXPOINT, etc.
            self.consume('LPAREN')
            # Leer los argumentos hasta el paréntesis derecho (simplificado)
            args = []
            while self.peek().type != 'RPAREN':
                args.append(self.consume().value)
            self.consume('RPAREN')
            operations.append(f"{op_tok.value}({''.join(args)})")
        
        self.consume('RBRACE')
        return MorphismDef(name_tok.value, source_tok.value, target_tok.value, operations)

    def parse_flow(self) -> FlowDef:
        self.consume('Flow')
        name_tok = self.consume('ID')
        self.consume('COLON')
        
        self.consume('source')
        self.consume('ASSIGN')
        source_tok = self.consume('ID')
        
        self.consume('pipe')
        self.consume('ASSIGN')
        pipe = []
        while self.peek() and self.peek().type not in ('sink', 'RBRACE'):
            tok = self.consume()
            pipe.append(tok.value)
            
        self.consume('sink')
        self.consume('ASSIGN')
        sink_tok = self.consume('ID')
        
        return FlowDef(name_tok.value, source_tok.value, pipe, sink_tok.value)

if __name__ == '__main__':
    from lexer import PolydimLexer
    test_code = '''
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
    lex = PolydimLexer()
    tokens = lex.tokenize(test_code)
    
    parser = PolydimParser(tokens)
    ast = parser.parse()
    for node in ast:
        print(node)
