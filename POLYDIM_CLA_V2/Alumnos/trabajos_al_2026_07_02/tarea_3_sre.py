"""
POLYDIM - Enjambre de Validación - Test 3 / Escuadron 1
Rol: DevOps y SRE de IA

Objetivo:
    Framework de benchmarking que MONITOREA que las actualizaciones
    (gradientes / deltas de parametros) de un modelo que opera en el
    Algebra de Clifford Cl(3,0) queden restringidas a los componentes
    bivectoriales puros (grado 2), detectando fugas (leakage) hacia
    componentes escalares (grado 0), vectoriales (grado 1) o
    trivectoriales (grado 3).

Contexto matematico minimo:
    Base de Cl(3,0), 8 "blades":
        indice 0        -> 1            (grado 0, escalar)
        indices 1,2,3   -> e1, e2, e3    (grado 1, vector)
        indices 4,5,6   -> e12,e13,e23   (grado 2, bivector)
        indice 7        -> e123          (grado 3, trivector)

    Un multivector se representa como un vector numpy de 8 componentes.
"""

from __future__ import annotations

import time
import json
import numpy as np
from dataclasses import dataclass, field
from typing import Callable

# ---------------------------------------------------------------------------
# 1. Definicion de la estructura de grados del algebra Cl(3,0)
# ---------------------------------------------------------------------------

GRADE_INDICES = {
    0: np.array([0]),          # escalar
    1: np.array([1, 2, 3]),    # vector
    2: np.array([4, 5, 6]),    # bivector
    3: np.array([7]),          # trivector
}
BIVECTOR_MASK = np.zeros(8, dtype=bool)
BIVECTOR_MASK[GRADE_INDICES[2]] = True


def grade_norms(mv: np.ndarray) -> dict[int, float]:
    """Devuelve la norma L2 de cada grado presente en un multivector."""
    return {g: float(np.linalg.norm(mv[idx])) for g, idx in GRADE_INDICES.items()}


def project_to_bivector(mv: np.ndarray) -> np.ndarray:
    """Proyecta un multivector al subespacio bivectorial puro (grado 2)."""
    out = np.zeros_like(mv)
    out[BIVECTOR_MASK] = mv[BIVECTOR_MASK]
    return out


def geometric_product_cl3(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Producto geometrico simplificado en Cl(3,0), suficiente para demostrar
    la fuga entre grados (no es una implementacion completa de todas las
    reglas de anticonmutacion, pero preserva la propiedad clave que nos
    interesa: multiplicar dos bivectores puede generar escalar + bivector).

    Regla usada (simplificada, valida para pares de bivectores ortogonales
    tipo eij * ejk = +-ei k y eij * eij = -1):
    """
    out = np.zeros(8)
    # producto escalar-escalar y escalar-cualquier (conmuta, no genera fuga)
    out[0] += a[0] * b[0]
    out[1:] += a[0] * b[1:] + b[0] * a[1:]

    # producto bivector-bivector: puede generar escalar (autoproducto)
    # y bivector (producto cruzado entre planos distintos)
    biv_idx = GRADE_INDICES[2]
    for i in biv_idx:
        for j in biv_idx:
            coef = a[i] * b[j]
            if coef == 0:
                continue
            if i == j:
                # eij * eij = -1  -> fuga hacia grado 0
                out[0] += -coef
            else:
                # producto cruzado entre planos distintos -> otro bivector
                # (mapeo ciclico simplificado dentro de los 3 bivectores)
                k = biv_idx[(list(biv_idx).index(i) + list(biv_idx).index(j)) % 3]
                out[k] += coef
    return out


# ---------------------------------------------------------------------------
# 2. Definicion de politicas SLO (Service Level Objective) del monitor
# ---------------------------------------------------------------------------

@dataclass
class SLOPolicy:
    max_leakage_ratio: float = 1e-6   # fuga tolerada tras la mascara (debe ser ~0)
    max_latency_ms: float = 5.0       # latencia maxima por paso de monitoreo


@dataclass
class StepResult:
    step: int
    leakage_before_mask: float
    leakage_after_mask: float
    latency_ms: float
    passed: bool


# ---------------------------------------------------------------------------
# 3. Framework de Benchmarking / Monitoreo (rol SRE)
# ---------------------------------------------------------------------------

@dataclass
class BivectorRestrictionBenchmark:
    policy: SLOPolicy = field(default_factory=SLOPolicy)
    history: list[StepResult] = field(default_factory=list)

    def _leakage_ratio(self, mv: np.ndarray) -> float:
        total = np.linalg.norm(mv)
        if total == 0:
            return 0.0
        leaked = np.linalg.norm(mv[~BIVECTOR_MASK])
        return leaked / total

    def run_step(self, step: int, raw_update: np.ndarray,
                 recompose: Callable[[np.ndarray], np.ndarray] | None = None) -> StepResult:
        """
        Simula un paso de entrenamiento:
          1. Llega un update crudo (puede tener fuga en otros grados).
          2. Se mide la fuga ANTES de restringir.
          3. Se aplica la restriccion (proyeccion a bivector puro).
          4. Si se provee `recompose`, se reaplica una operacion del modelo
             (ej. producto geometrico con otro bivector) para verificar si
             la fuga REAPARECE luego de operar (la 'trampa' del ejercicio).
          5. Se mide la fuga DESPUES y se evalua contra el SLO.
        """
        t0 = time.perf_counter()

        leakage_before = self._leakage_ratio(raw_update)
        restricted = project_to_bivector(raw_update)

        if recompose is not None:
            restricted = recompose(restricted)

        leakage_after = self._leakage_ratio(restricted)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        passed = (
            leakage_after <= self.policy.max_leakage_ratio
            and latency_ms <= self.policy.max_latency_ms
        )

        result = StepResult(step, leakage_before, leakage_after, latency_ms, passed)
        self.history.append(result)
        return result

    def report(self) -> dict:
        total = len(self.history)
        failed = [r for r in self.history if not r.passed]
        return {
            "total_steps": total,
            "failed_steps": len(failed),
            "pass_rate": 1.0 - (len(failed) / total if total else 0.0),
            "avg_latency_ms": float(np.mean([r.latency_ms for r in self.history])) if total else 0.0,
            "max_leakage_after_mask": float(max((r.leakage_after_mask for r in self.history), default=0.0)),
            "failed_step_ids": [r.step for r in failed],
        }


# ---------------------------------------------------------------------------
# 4. Demostracion / caso de prueba
# ---------------------------------------------------------------------------

def _make_random_bivector(rng: np.random.Generator) -> np.ndarray:
    mv = np.zeros(8)
    mv[GRADE_INDICES[2]] = rng.normal(size=3)
    return mv


def _make_leaky_update(rng: np.random.Generator) -> np.ndarray:
    """Simula un update 'sucio': mezcla bivector con ruido en otros grados."""
    mv = rng.normal(scale=0.05, size=8)
    mv[GRADE_INDICES[2]] += rng.normal(scale=1.0, size=3)  # componente dominante
    return mv


def run_demo(n_steps: int = 50, seed: int = 7) -> dict:
    rng = np.random.default_rng(seed)
    bench = BivectorRestrictionBenchmark()

    # bivector "de referencia" contra el que se recompone, para exhibir
    # la trampa: el producto geometrico bivector*bivector reintroduce grado 0
    reference_bivector = _make_random_bivector(rng)

    def recompose(mv: np.ndarray) -> np.ndarray:
        return geometric_product_cl3(mv, reference_bivector)

    for step in range(n_steps):
        raw_update = _make_leaky_update(rng)
        bench.run_step(step, raw_update, recompose=recompose)

    return bench.report()


if __name__ == "__main__":
    resultado = run_demo()
    print(json.dumps(resultado, indent=2, ensure_ascii=False))
