# STUDENT TASK ASSIGNMENT — POLYDIM + MPAT5
# Version: V1.0 — 2026-06-15
# Orchestrator: AI (Claude Sonnet)
# Backlog: BACKLOG_STUDENTS_V13.json (1LytQX3M2U3QatPwLnZlHETRWwVb0VBaT)

---

## HOW TO START (every student)

1. Upload your skill to Claude.ai > Settings > Skills:
   - All students: mpat5-student-skill-V2_00_EN.md (1-WHsAt43wKvEUyVCtPOiLNjybloq3614)
   - POLYDIM tasks: polydim-core-skill-V8_0_EN.md (1QSwpEakVWHUfxZDjtCO39l_dW4Q-7N5-)

2. Open a new Claude session.

3. Read BACKLOG_STUDENTS_V13.json (1LytQX3M2U3QatPwLnZlHETRWwVb0VBaT) from Drive.

4. Take your assigned task (see table below). Lock it with your student ID.

5. Read the files listed in 'files_to_read' for your task.

6. Work. Save artifacts to POLYDIM/core/ or POLYDIM/docs/ (accessible Drive).
   MPAT5 tasks: save with MPAT5_DEST header for later move.

7. Close with 10-section relay. Update BACKLOG status to DONE.

---

## TASK ASSIGNMENTS (suggested by difficulty + interest)

```
Student  Task     Project  Title                                    Difficulty
────────────────────────────────────────────────────────────────────────────────
1        TASK_P07 POLYDIM  README V3 — V0.4 guide                  EASY
2        TASK_P05 POLYDIM  Research semantic drift (PEND_008)       MEDIUM
3        TASK_P01 POLYDIM  Update tests for V0.4 (ObjectND return)  MEDIUM
4        TASK_M05 MPAT5    Relay protocol for 18-AI coordination    MEDIUM
5        TASK_M02 MPAT5    Unikernels research for MPAT5            MEDIUM
6        TASK_P03 POLYDIM  polydim_flutter.dart implementation      MEDIUM
7        TASK_M01 MPAT5    MPAT5 initial BACKLOG.json               MEDIUM
8        TASK_M03 MPAT5    AESP contract specification              HARD
9        TASK_P04 POLYDIM  Distributed consistency spec (PEND_003)  HARD
10       TASK_M04 MPAT5    POLYDIM-MPAT5 integration design         HARD
```

TASK_P02 and TASK_P06 (HARD): reserved for when students finish their first task.

---

## WHAT EACH STUDENT PRODUCES

### TASK_P07 (Student 1) — EASY START
Read V0.3 README and V0.4 runtime. Write updated documentation.
No code. Just clear writing. Good for understanding the full system.

### TASK_P05 (Student 2) — RESEARCH
Web search: "embedding space drift detection", "cosine similarity monitoring".
Design algorithm for detecting when ALIGN matrix becomes stale.
Deliverable: research doc with Python implementation sketch.

### TASK_P01 (Student 3) — TESTING
V0.3 tests tested `receive()` returning dict.
V0.4 returns ObjectND. Locate failing tests, fix them, add 3 new invariant tests.
Run: python polydim_tests.py — must show 32+ tests passing.

### TASK_M05 (Student 4) — PROTOCOL DESIGN
Research: vector clocks, distributed locking, optimistic concurrency.
Apply to: BACKLOG.json with 18 concurrent writers.
Deliverable: improved protocol spec for relay + task locking at scale.

### TASK_M02 (Student 5) — INFRASTRUCTURE RESEARCH
Compare Firecracker, libkrun, Nanos for running Python agents.
Constraint: university server, limited budget, Argentina.
Deliverable: comparison table + recommendation + sample config.

### TASK_P03 (Student 6) — FLUTTER/DART PORT
Implement POLYDIM in Dart. Reference: Python V0.4.
Key: same deterministic symbols (same name → same vector on all platforms).
Test: binary roundtrip between Dart and Python.

### TASK_M01 (Student 7) — BACKLOG DESIGN
Read MIGRATION_LOGIC.md. Design the full BACKLOG.json for MPAT5 PHASE 2.
7 batches, proper task structure, dependencies, priorities.
This enables 18 AIs to work in parallel when PHASE 2 starts.

### TASK_M03 (Student 8) — CONTRACT DESIGN (HARD)
What can an AESP agent DO? What does it CONSUME? What does it PRODUCE?
Design the formal interface in the MPAT5 mesh.
Must be compatible with POLYDIM ObjectND types.

### TASK_P04 (Student 9) — DISTRIBUTED SYSTEMS (HARD)
Research CRDTs (Conflict-free Replicated Data Types).
Apply to: two AIs editing DIM_SQL and DIM_PYTHON of the same object simultaneously.
Key insight: POLYDIM dims are orthogonal (INV_003) — this may simplify consistency.

### TASK_M04 (Student 10) — INTEGRATION ARCHITECTURE (HARD)
How does POLYDIM fit inside MPAT5?
Map each MPAT5 node type to POLYDIM dimensions.
Propose new dims: DIM_AESP? DIM_UNIKERNEL? DIM_CONTRACT?
This is the bridge between both projects.

---

## SHARED RESOURCES

All students can read (no lock needed):
  polydim_runtime_v04.py    19hYkp5qFKE5G1bQTS-P3NfYJqCoaMcqA  ← main runtime
  SPEC_BINARY_FORMAT_V0.md  1pN5LVhC-fxTCMXbsmBcMvTA1TnAA73Jw  ← binary format
  MPAT5_MIGRATION_LOGIC_V1.md 1Wes1VJtVuhaqeTqhwhk4qgr0eflPDBR0 ← MPAT5 basis
  README_ALUMNOS_V2.md      13bGpgUOuQ6rLzJ3nE-47O0CVfO1stXL1  ← current README

All students save TO:
  POLYDIM/core/   (code)
  POLYDIM/docs/   (documentation + MPAT5 files with MPAT5_DEST header)
  POLYDIM/spec/   (specifications)
  POLYDIM/research/ (research)

---

## ORCHESTRATOR NOTES

Tasks are designed so no two students block each other.
TASK_P01 and TASK_P07 can run in parallel (different files).
TASK_M01 through TASK_M05 are all independent.
TASK_P02 and TASK_P06 unlock when students finish first tasks.

When 3+ students are done: reconvene to review results and plan PHASE 2.
MPAT5 PHASE 2 requires TASK_M01 (backlog) + TASK_M02 (unikernels) + TASK_M03 (contracts).

---
*STUDENT_ASSIGNMENTS_V1.md · 2026-06-15 · Orchestrator: AI*
