@echo off
echo =======================================================================
echo  POLYDIM-CLA: Empirical Validation Suite Launcher
echo =======================================================================
echo.
echo This launcher executes the validation scripts and verifies mathematical axioms.
echo.

:: ---------------------------------------------------------
:: 1. CAYLEY-WOODBURY SOLVER
:: ---------------------------------------------------------
echo -----------------------------------------------------------------------
echo  [1/4] Running Cayley-Woodbury Equivalence Validation...
echo -----------------------------------------------------------------------
echo  * What this test checks: Checks if our O(N) Woodbury projection solver
echo    produces identical numerical results to the direct O(N3) Cayley rotor.
echo  * Expected output: Maximum Absolute Error (MAE) should be ~ 0.0 (near 1e-7).
echo.
python python_validation\jax_cayley_woodbury_validator.py
echo.

:: ---------------------------------------------------------
:: 2. MATHEMATICAL AXIOMS
:: ---------------------------------------------------------
echo -----------------------------------------------------------------------
echo  [2/4] Running Mathematical Axioms Validation...
echo -----------------------------------------------------------------------
echo  * What this test checks: Checks the Banach fixed point convergence, 
echo    Hodge gradient purification, and category commutative diagram error (eta).
echo  * Expected output: Banach convergence in < 5 steps, metrical error eta < 0.1.
echo.
python python_validation\polydim_cla_math_validator.py
echo.

:: ---------------------------------------------------------
:: 3. CONJECTURES & PRECISION LOSS
:: ---------------------------------------------------------
echo -----------------------------------------------------------------------
echo  [3/4] Running Conjectures & Precision Loss Validation...
echo -----------------------------------------------------------------------
echo  * What this test checks: Checks if Softmax commutes with projection (Conjecture 1)
echo    and measures Stiefel manifold decay under FP16/BF16 precision quantization.
echo  * Expected output: Softmax fails commutativity (~ 102% error). FP16/BF16
echo    orthogonality drift degrades by ~600x and ~5400x respectively.
echo.
python python_validation\polydim_cla_conjectures_validator.py
echo.

:: ---------------------------------------------------------
:: 4. RUST CORE PROTOTYPE
:: ---------------------------------------------------------
echo -----------------------------------------------------------------------
echo  [4/4] Checking compilable Rust Core Prototype...
echo -----------------------------------------------------------------------
where cargo >nul 2>nul
if %errorlevel% neq 0 (
    echo [NOTE] 'cargo' is not on your PATH. Skipping Rust prototype test execution.
    echo To build and run Rust prototype, please install Rust from https://rustup.rs
    echo.
) else (
    echo  * What this test checks: Runs 15 cross-language integration tests
    echo    (11 PBF0 interop tests + 4 ALIGN tests).
    echo  * Expected output: 15/15 tests passed, Cargo test prints "test result: ok".
    echo.
    cd rust_prototype
    echo Generating test fixtures...
    python test_pbf0_interop.py
    python test_align_cross_language.py
    echo Running Cargo test suite...
    cargo test
    cd ..
)

echo.
echo =======================================================================
echo  Validation Suite execution completed!
echo =======================================================================
pause
