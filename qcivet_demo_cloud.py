"""
qcivet demo 3: cloud QPU auditing.

a meta-application: qcivet is used by the customer to audit a
cloud quantum service (IBM Quantum Platform, AWS Braket, Azure
Quantum, etc). the customer submits a hybrid workload, the
provider transpiles, schedules, and runs it on some backend,
then returns a result. the customer wants to verify after the
fact that:

  * the backend the provider claimed to use is actually the one
    the job ran on
  * the calibration data the provider published matches what was
    in effect at run time
  * the result has not been tampered with between QPU and customer

stages:
  1. customer_submission       (customer-side classical)
  2. cloud_transpilation       (provider-side classical)
  3. backend_assignment        (provider-side classical)
  4. calibration_verification  (cross-side: customer fetches snapshot,
                                provider attests)
  5. job_execution             (quantum - actual QPU run)
  6. result_delivery           (provider-to-customer classical)

four scenarios:
  * honest pipeline (baseline)
  * silent downgrade: provider claimed Heron, secretly ran on Eagle
  * calibration spoof: snapshot hash swapped after commit
  * global rewrite: chain rebuilt offline with a different backend

in production the customer would obtain the calibration snapshot
via the provider's API (e.g. backend.properties() in qiskit) and
the post-run observable estimates from a small set of "tracer"
circuits inserted alongside the customer's workload.
"""

import tempfile
from pathlib import Path

from qcivet_realtime import (
    IntegrityVerifier,
    StageResult,
    Observable,
    IntegrityViolation,
    ExternalAnchor,
    run_pipeline,
)


# Tolerance for the tracer-circuit observable.
#
# NOTE ON SCALE. The value below sits on the *simulated* device-noise
# scale: the calibration-derived models give an honest floor of about
# 0.028 for a Heron-class backend and about 0.056 for an Eagle-class
# one, so a threshold between the two separates them. Real hardware is
# noisier. On the archived ibm_fez run the honest floor is 0.074 and
# the calibrated tolerance is 0.175. A deployment must therefore
# calibrate on the machine that will run the workload, never on a
# simulator built from its published calibration data.
TRACER_REFERENCE = 0.0
EPS_TRACER_HERON = 0.05


# stages

def stage_submission() -> StageResult:
    return StageResult(
        name="customer_submission",
        spec={
            "customer_id": "ACME-CORP",
            "requested_backend_class": "heron_r2",
            "min_acceptable_backend": "heron_r2",
            "circuit_count": 256,
            "shots_per_circuit": 4096,
            "tracer_circuits_included": True,
            "submission_signature": "ed25519:abc...",
        },
        wall_time_ms=12.0,
    )


def stage_transpilation() -> StageResult:
    return StageResult(
        name="cloud_transpilation",
        spec={
            "transpiler_version": "qiskit-2.1.0",
            "optimisation_level": 1,
            "two_qubit_gate_count_total": 12_480,
            "transpiled_depth_p95": 84,
            "provider_signature": "ed25519:def...",
        },
        wall_time_ms=410.0,
    )


def stage_backend_assignment(backend_name: str) -> StageResult:
    return StageResult(
        name="backend_assignment",
        spec={
            "assigned_backend": backend_name,
            "backend_class": (
                "heron_r2" if backend_name in ("ibm_fez", "ibm_kingston")
                else "eagle_r3"
            ),
            "queue_position_at_assign": 7,
            "estimated_run_seconds": 38,
        },
        wall_time_ms=3.0,
    )


def stage_calibration_verification(
        snapshot_hash: str = "sha256:ca11ed..2e",
        snapshot_age_minutes: int = 17,
) -> StageResult:
    return StageResult(
        name="calibration_verification",
        spec={
            "snapshot_hash": snapshot_hash,
            "snapshot_age_minutes": snapshot_age_minutes,
            "single_qubit_error_median": 0.00021,
            "two_qubit_error_median": 0.00203,
            "readout_error_median": 0.0091,
            "snapshot_signed_by_provider": True,
        },
        wall_time_ms=18.0,
    )


def stage_execution(measured_tracer_dev: float,
                    backend_name: str) -> StageResult:
    return StageResult(
        name="job_execution",
        spec={
            "actual_backend": backend_name,
            "shots_total": 256 * 4096,
            "wall_time_seconds": 41.2,
            "qpu_compute_time_seconds": 36.7,
        },
        observables=[
            Observable(
                pauli="tracer_dev",
                reference=TRACER_REFERENCE,
                epsilon=EPS_TRACER_HERON,
                measured=measured_tracer_dev,
            ),
        ],
        wall_time_ms=41_200.0,
    )


def stage_delivery() -> StageResult:
    return StageResult(
        name="result_delivery",
        spec={
            "delivery_channel": "https",
            "result_payload_size_bytes": 4_700_000,
            "payload_signed_by_provider": True,
            "customer_received_timestamp": 1_731_847_200,
        },
        wall_time_ms=160.0,
    )


# scenarios

def scenario_0_clean(commit) -> None:
    commit(stage_submission())
    commit(stage_transpilation())
    commit(stage_backend_assignment(backend_name="ibm_fez"))
    commit(stage_calibration_verification())
    commit(stage_execution(
        measured_tracer_dev=0.025, backend_name="ibm_fez"))
    commit(stage_delivery())


def scenario_1_silent_downgrade(commit) -> None:
    """provider claimed a Heron-class backend but really ran on an
    Eagle-class one. the tracer-circuit observable then sits at the
    Eagle noise floor, above the Heron-calibrated tolerance, and the
    engine halts at job_execution.
    """
    commit(stage_submission())
    commit(stage_transpilation())
    commit(stage_backend_assignment(backend_name="ibm_fez"))
    commit(stage_calibration_verification())
    commit(stage_execution(
        measured_tracer_dev=0.075, backend_name="ibm_fez"))
    commit(stage_delivery())


def scenario_2_calibration_spoof(commit) -> None:
    """provider committed a fresh calibration snapshot, then
    silently swapped it for an older one (e.g. to hide a recent
    qubit failure). chain replay catches the spec mismatch.
    """
    commit(stage_submission())
    commit(stage_transpilation())
    commit(stage_backend_assignment(backend_name="ibm_fez"))
    rec = commit(stage_calibration_verification(
        snapshot_hash="sha256:fresh..01", snapshot_age_minutes=12))
    rec.spec["snapshot_hash"] = "sha256:stale..ff"
    rec.spec["snapshot_age_minutes"] = 980
    commit(stage_execution(
        measured_tracer_dev=0.025, backend_name="ibm_fez"))
    commit(stage_delivery())


def scenario_3_anchor_rewrite(commit, *, anchor: ExternalAnchor,
                              verifier: IntegrityVerifier) -> None:
    """audit log rewritten offline with a different backend assignment.
    locally consistent but never anchored. verify_against_anchor
    catches it.
    """
    commit(stage_submission())
    commit(stage_transpilation())
    commit(stage_backend_assignment(backend_name="ibm_fez"))
    commit(stage_calibration_verification())
    commit(stage_execution(
        measured_tracer_dev=0.025, backend_name="ibm_fez"))
    commit(stage_delivery())

    parallel = IntegrityVerifier()
    parallel.commit_stage(stage_submission())
    parallel.commit_stage(stage_transpilation())
    parallel.commit_stage(stage_backend_assignment(
        backend_name="ibm_brisbane"))
    parallel.commit_stage(stage_calibration_verification())
    parallel.commit_stage(stage_execution(
        measured_tracer_dev=0.030,
        backend_name="ibm_brisbane"))
    parallel.commit_stage(stage_delivery())

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

    section("DEMO 3.0  honest cloud-QPU auditing pipeline")
    anchor = ExternalAnchor(anchor_path)
    v = IntegrityVerifier(anchor=anchor)
    ok, _, _ = run_pipeline(scenario_0_clean, v, label="cloud_clean")
    print(f"  result: ok={ok}")
    print(f"  audit trail length: {len(v.records)} records")
    print(f"  total verify latency: "
          f"{sum(r.verify_latency_ms for r in v.records):.3f} ms")

    section("DEMO 3.1  silent backend downgrade (Heron claimed, "
            "Eagle delivered)")
    anchor = ExternalAnchor(anchor_path)
    v = IntegrityVerifier(anchor=anchor)
    ok, viol, committed = run_pipeline(
        scenario_1_silent_downgrade, v, label="cloud_downgrade")
    print(f"  result: ok={ok}")
    if viol is not None:
        print(f"  HALT at stage {viol.stage_index} "
              f"('{viol.stage_name}'), kind={viol.kind}")
        print(f"  stages committed before halt: "
              f"{[r.stage_name for r in committed]}")

    section("DEMO 3.2  calibration snapshot spoof")
    anchor = ExternalAnchor(anchor_path)
    v = IntegrityVerifier(anchor=anchor)
    ok, viol, _ = run_pipeline(
        scenario_2_calibration_spoof, v, label="cloud_spoof")
    print(f"  streaming result: ok={ok}")
    print("  running post-pipeline verify_full_chain()...")
    try:
        v.verify_full_chain()
        print("  RESULT: tamper undetected (this should never happen)")
    except IntegrityViolation as e:
        print(f"  RESULT: tamper DETECTED at stage {e.stage_index} "
              f"('{e.stage_name}'), kind={e.kind}")
        print(f"          tampered snapshot hash now: "
              f"{v.records[e.stage_index].spec['snapshot_hash']}")

    section("DEMO 3.3  globally consistent rewrite vs external anchor")
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

    section("DEMO 3 SUMMARY")
    print("  Scenario 0: honest pipeline    -> committed cleanly")
    print("  Scenario 1: silent downgrade   -> caught at commit time")
    print("              (Heron tolerance violated, downstream not run)")
    print("  Scenario 2: calibration spoof  -> caught by chain replay")
    print("  Scenario 3: global rewrite     -> caught by anchor check")
    print()


if __name__ == "__main__":
    main()
