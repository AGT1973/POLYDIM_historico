from compiler.lexer import PolydimLexer
from compiler.parser import PolydimParser
from compiler.topology_checker import TopologyChecker

source_code = """
Space S1: Dim=1024, Metric="Clifford"
Space S2: Dim=512, Metric="Clifford"

Morphism M1: S1 -> S2 {
    apply PROJECT(basis="orthogonal")
}

Flow F1:
    source = S1
    pipe = M1 @ MIX(alpha=0.5)
    sink = S2
"""

tokens = PolydimLexer().tokenize(source_code)
ast = PolydimParser(tokens).parse()

checker = TopologyChecker(ast)
for name, flow in checker.flows.items():
    print(f"Flow {name} pipe tokens: {flow.pipe}")

checker.analyze()
print("Success!")
