"""
qcivet demo 1: VQE drug-discovery pipeline.

a six-stage hybrid quantum-classical workflow that estimates the
ground-state energy of H2 (STO-3G basis) by VQE. the actual quantum
chemistry math is faked - we hard-code the textbook H2 reference
energy. the point of this script is to demonstrate integrity
verification end-to-end, not to do real chemistry.

stages:
  1. molecular_geometry       (classical)
  2. active_space_selection   (classical)
  3. hamiltonian_construction (classical)
  4. ansatz_synthesis         (classical)
  5. vqe_optimisation         (quantum - QPU calls in inner loop)
  6. result_interpretation    (classical)

we run four scenarios:
  * honest pipeline (baseline)
  * tampering: an attacker rewrites the active_space record after commit
  * quantum drift: the measured energy lies outside epsilon
  * global rewrite: a parallel chain that was never anchored
"""

import tempfile
import time
from pathlib import Path

from qcivet_realtime import (
    IntegrityVerifier,
    StageResult,
    Observable,
    IntegrityViolation,
    ExternalAnchor,
    run_pipeline,
)


# textbook ground-state energy of H2 in STO-3G, in Hartree.
# the energy observable should read this off the optimised state.
H2_GROUND_STATE_HARTREE = -1.137270174

# Energy tolerance for the VQE stage, in Hartree.
#
# NOTE ON SCALE. This is an energy tolerance, not a Pauli-expectation
# tolerance. The two live on different scales and must be calibrated
# separately. If the Hamiltonian is decomposed as H = sum_j c_j P_j,
# a per-Pauli deviation of eps propagates to an energy deviation of at
# most eps * sum_j |c_j|. The value below is illustrative for this
# six-stage walkthrough; a deployment calibrates it on the target
# backend from the honest-run floor, exactly as the paper does for the
# Pauli contract.
EPS_ENERGY = 0.04


# stages

def stage_geometry() -> StageResult:
    return StageResult(
        name="molecular_geometry",
        spec={
            "molecule": "H2",
            "atoms": [
                {"element": "H", "xyz": [0.0, 0.0, 0.0]},
                {"element": "H", "xyz": [0.0, 0.0, 0.7414]},
            ],
            "basis": "sto-3g",
            "charge": 0,
            "multiplicity": 1,
        },
        wall_time_ms=2.0,
    )


def stage_active_space(num_active_orbitals: int = 2) -> StageResult:
    return StageResult(
        name="active_space_selection",
        spec={
            "num_active_orbitals": num_active_orbitals,
            "num_active_electrons": 2,
            "frozen_core": False,
            "selector": "natural_orbitals",
        },
        wall_time_ms=8.0,
    )


def stage_hamiltonian() -> StageResult:
    return StageResult(
        name="hamiltonian_construction",
        spec={
            "encoding": "jordan_wigner",
            "num_qubits": 4,
            "num_pauli_terms": 15,
            "hamiltonian_norm": 1.857,
        },
        wall_time_ms=12.0,
    )


def stage_ansatz() -> StageResult:
    return StageResult(
        name="ansatz_synthesis",
        spec={
            "ansatz_family": "UCCSD",
            "num_parameters": 3,
            "circuit_depth": 14,
            "two_qubit_gate_count": 8,
        },
        wall_time_ms=6.0,
    )


def stage_vqe(measured_energy: float) -> StageResult:
    return StageResult(
        name="vqe_optimisation",
        spec={
            "backend": "ibm_brisbane",
            "shots_per_iter": 4096,
            "optimiser": "SPSA",
            "max_iterations": 60,
            "convergence_tol": 1e-3,
            "final_iterations": 47,
        },
        observables=[
            Observable(
                pauli="H",
                reference=H2_GROUND_STATE_HARTREE,
                epsilon=EPS_ENERGY,
                measured=measured_energy,
            ),
        ],
        wall_time_ms=51_400.0,
    )


def stage_interpretation(reported_energy: float) -> StageResult:
    return StageResult(
        name="result_interpretation",
        spec={
            "reported_ground_state_hartree": reported_energy,
            "reported_ground_state_eV": reported_energy * 27.2114,
            "downstream_use": "binding_energy_estimate",
            "fda_compliance_flag": True,
        },
        wall_time_ms=3.0,
    )


# scenarios

def scenario_0_clean(commit) -> None:
    commit(stage_geometry())
    commit(stage_active_space(num_active_orbitals=2))
    commit(stage_hamiltonian())
    commit(stage_ansatz())
    commit(stage_vqe(measured_energy=H2_GROUND_STATE_HARTREE + 0.012))
    commit(stage_interpretation(reported_energy=H2_GROUND_STATE_HARTREE + 0.012))


def scenario_1_tampering_active_space(commit) -> None:
    """attacker silently swaps active_space record after it has been
    committed (orbital count 2 -> 4). streaming check passes because
    the tamper happens AFTER the commit; verify_full_chain catches it.
    """
    commit(stage_geometry())
    rec = commit(stage_active_space(num_active_orbitals=2))
    rec.spec["num_active_orbitals"] = 4
    commit(stage_hamiltonian())
    commit(stage_ansatz())
    commit(stage_vqe(measured_energy=H2_GROUND_STATE_HARTREE + 0.012))
    commit(stage_interpretation(reported_energy=H2_GROUND_STATE_HARTREE + 0.012))


def scenario_2_quantum_drift(commit) -> None:
    """quantum-side attack: VQE converges to the wrong stationary
    point (or the optimiser is biased). measured energy lies outside
    epsilon - the engine halts at commit time.
    """
    commit(stage_geometry())
    commit(stage_active_space(num_active_orbitals=2))
    commit(stage_hamiltonian())
    commit(stage_ansatz())
    commit(stage_vqe(measured_energy=H2_GROUND_STATE_HARTREE + 0.10))
    commit(stage_interpretation(reported_energy=H2_GROUND_STATE_HARTREE + 0.10))


def scenario_3_anchor_rewrite(commit, *, anchor: ExternalAnchor,
                              verifier: IntegrityVerifier) -> None:
    """global rewrite: attacker reruns the entire pipeline offline
    with a different active_space spec. the new chain is locally
    consistent but never appeared in the timestamp log.
    verify_against_anchor catches it.
    """
    commit(stage_geometry())
    commit(stage_active_space(num_active_orbitals=2))
    commit(stage_hamiltonian())
    commit(stage_ansatz())
    commit(stage_vqe(measured_energy=H2_GROUND_STATE_HARTREE + 0.012))
    commit(stage_interpretation(reported_energy=H2_GROUND_STATE_HARTREE + 0.012))

    parallel = IntegrityVerifier()
    parallel.commit_stage(stage_geometry())
    parallel.commit_stage(stage_active_space(num_active_orbitals=4))
    parallel.commit_stage(stage_hamiltonian())
    parallel.commit_stage(stage_ansatz())
    parallel.commit_stage(stage_vqe(measured_energy=H2_GROUND_STATE_HARTREE + 0.020))
    parallel.commit_stage(stage_interpretation(reported_energy=H2_GROUND_STATE_HARTREE + 0.020))

    parallel.anchor = anchor
    parallel.verify_against_anchor()


# main

def section(title):
    print()
    print(f">>> {title}")
    print("-" * 58)


def main():
    tmp = Path(tempfile.mkdtemp())
    anchor_path = tmp / "anchor.log"

    section("DEMO 1.0  honest VQE pipeline")
    anchor = ExternalAnchor(anchor_path)
    v = IntegrityVerifier(anchor=anchor)
    ok, _, _ = run_pipeline(scenario_0_clean, v, label="vqe_clean")
    print(f"  result: ok={ok}")
    print(f"  audit trail length: {len(v.records)} records")
    print(f"  total verify latency: "
          f"{sum(r.verify_latency_ms for r in v.records):.3f} ms")
    print(f"  head hash: {v.head_hash()[:16]}...")

    section("DEMO 1.1  tampering with active_space spec")
    anchor = ExternalAnchor(anchor_path)
    v = IntegrityVerifier(anchor=anchor)
    ok, viol, _ = run_pipeline(
        scenario_1_tampering_active_space, v, label="vqe_tamper")
    print(f"  streaming result: ok={ok}")
    print("  running post-pipeline verify_full_chain()...")
    try:
        v.verify_full_chain()
        print("  RESULT: tamper undetected (this should never happen)")
    except IntegrityViolation as e:
        print(f"  RESULT: tamper DETECTED at stage {e.stage_index} "
              f"('{e.stage_name}'), kind={e.kind}")

    section("DEMO 1.2  quantum-side observable drift")
    anchor = ExternalAnchor(anchor_path)
    v = IntegrityVerifier(anchor=anchor)
    ok, viol, committed = run_pipeline(
        scenario_2_quantum_drift, v, label="vqe_drift")
    print(f"  result: ok={ok}")
    if viol is not None:
        print(f"  HALT at stage {viol.stage_index} "
              f"('{viol.stage_name}'), kind={viol.kind}")
        print(f"  stages committed before halt: "
              f"{[r.stage_name for r in committed]}")

    section("DEMO 1.3  globally consistent rewrite vs external anchor")
    anchor = ExternalAnchor(anchor_path)
    v = IntegrityVerifier(anchor=anchor)
    try:
        scenario_3_anchor_rewrite(
            commit=lambda r: v.commit_stage(r),
            anchor=anchor, verifier=v,
        )
        print("  RESULT: rewrite undetected (this should never happen)")
    except IntegrityViolation as e:
        print(f"  RESULT: rewrite DETECTED, kind={e.kind}")
        print(f"          {e}")

    section("DEMO 1 SUMMARY")
    print("  Scenario 0: honest pipeline  -> committed cleanly")
    print("  Scenario 1: spec tamper      -> caught by chain replay")
    print("  Scenario 2: quantum drift    -> caught at commit time")
    print("              (downstream stages never executed)")
    print("  Scenario 3: global rewrite   -> caught by anchor check")
    print()


if __name__ == "__main__":
    main()
