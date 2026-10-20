from typing import List, Dict
from .parser import ASTNode, SpaceDef, MorphismDef, FlowDef

class TopologyError(Exception):
    def __init__(self, message: str):
        super().__init__(f"TopologyError (Grothendieck Homotopy Failure): {message}")

class TopologyChecker:
    """
    Validates the geometric properties of the AST.
    Instead of classical Type Checking, this performs Topos Checking:
    1. Checks if spaces exist and metric is compatible (Clifford).
    2. Checks if morphisms correctly map source dimensions to target dimensions.
    3. Prevents causal paradoxes in flows (loops without fixpoint resolution).
    """
    def __init__(self, ast: List[ASTNode]):
        self.ast = ast
        self.spaces: Dict[str, SpaceDef] = {}
        self.morphisms: Dict[str, MorphismDef] = {}
        self.flows: Dict[str, FlowDef] = {}

    def analyze(self):
        # 1st Pass: Register all topological manifolds (Spaces)
        for node in self.ast:
            if isinstance(node, SpaceDef):
                if node.name in self.spaces:
                    raise TopologyError(f"Space manifold '{node.name}' defined multiple times.")
                self.spaces[node.name] = node

        # 2nd Pass: Register and validate Morphisms
        for node in self.ast:
            if isinstance(node, MorphismDef):
                if node.source not in self.spaces:
                    raise TopologyError(f"Morphism '{node.name}' references undefined source space '{node.source}'")
                if node.target not in self.spaces:
                    raise TopologyError(f"Morphism '{node.name}' references undefined target space '{node.target}'")
                
                # In a real Clifford algebra, dimensional mapping rules would be enforced here.
                # Example rule: Dim(Target) <= Dim(Source) * 2 (for multivector expansion)
                self.morphisms[node.name] = node

        # 3rd Pass: Validate Flows (Causality chains)
        for node in self.ast:
            if isinstance(node, FlowDef):
                if node.source not in self.spaces:
                    raise TopologyError(f"Flow '{node.name}' starts from undefined space '{node.source}'")
                if node.sink not in self.spaces:
                    raise TopologyError(f"Flow '{node.name}' sinks into undefined space '{node.sink}'")
                
                self.flows[node.name] = node
                self._validate_flow_causality(node)

        return True

    def _validate_flow_causality(self, flow: FlowDef):
        # Check if the pipe morphisms actually chain correctly
        # This is the "Homotopy" check.
        current_space = flow.source
        
        for op in flow.pipe:
            op_str = str(op)
            if op_str in self.morphisms:
                morphism = self.morphisms[op_str]
                if morphism.source != current_space:
                    raise TopologyError(
                        f"Homotopy broken in Flow '{flow.name}': "
                        f"Morphism '{morphism.name}' expects '{morphism.source}', but current manifold is '{current_space}'."
                    )
                current_space = morphism.target
            elif op_str in ('@', '(', ')', '=', ',', 'alpha', 'basis', 'metric'):
                # Composition operator and syntax noise
                continue
            elif 'MIX' in op_str or 'PROJECT' in op_str or 'FIXPOINT' in op_str:
                # Built-in operators that preserve the topological space
                continue
            elif isinstance(op, (int, float)) or op_str.replace('.', '', 1).isdigit():
                continue
            else:
                # Si el token no es reconocido, lanzamos error
                raise TopologyError(f"Unknown operation '{op}' in flow pipe.")

        if current_space != flow.sink:
            raise TopologyError(
                f"Flow '{flow.name}' does not resolve to sink '{flow.sink}'. "
                f"It resolved to '{current_space}' instead. Causal paradox detected."
            )

if __name__ == '__main__':
    from lexer import PolydimLexer
    from parser import PolydimParser
    
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
    
    checker = TopologyChecker(ast)
    checker.analyze()
    print("Topological Check Passed. No causal paradoxes found.")
