"""
qcivet demo 2: quantum-assisted fraud detection.

a six-stage hybrid pipeline that uses a quantum kernel to
classify credit card transactions as fraud or legitimate. the
classifier is a kernel SVM whose kernel matrix entries come
from a small quantum circuit (a fidelity kernel).

stages:
  1. transaction_ingestion       (classical)
  2. feature_engineering         (classical)
  3. quantum_kernel_preparation  (classical)
  4. qpu_kernel_evaluation       (quantum - fidelity tests)
  5. classification              (classical SVM scoring)
  6. alert_decision              (classical act/don't-act)

four scenarios:
  * honest pipeline (baseline)
  * insider tampering: someone raises the alert threshold after commit
                        so real fraud no longer trips an alert
  * kernel poisoning: the QPU returns a kernel matrix that drifts
                       beyond epsilon - device drift or active attack
  * global rewrite: chain rebuilt offline with a different feature set
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


# Worst-case kernel-matrix entry deviation. The declared reference is
# the ideal noiseless kernel; the QPU should match it within epsilon.
#
# NOTE ON SCALE. A kernel-entry deviation is not a Pauli expectation
# deviation. The value below is illustrative for this walkthrough; a
# deployment calibrates it on the target backend from the honest-run
# floor.
KERNEL_REFERENCE = 0.0
EPS_KERNEL = 0.05

# alert thresholds. legitimate threshold raises an alert when SVM score
# > 0.65. the tampering scenario raises this to 0.95 so genuine fraud
# slips through.
LEGITIMATE_ALERT_THRESHOLD = 0.65
TAMPERED_ALERT_THRESHOLD = 0.95


# stages

def stage_ingestion() -> StageResult:
    return StageResult(
        name="transaction_ingestion",
        spec={
            "source": "card_network_feed",
            "batch_window_ms": 250,
            "num_transactions": 1024,
            "currency": "USD",
            "regulator_pii_flag": "tokenised",
        },
        wall_time_ms=4.0,
    )


def stage_features(num_features: int = 8) -> StageResult:
    return StageResult(
        name="feature_engineering",
        spec={
            "feature_set_id": "FS-2026-Q2",
            "num_features": num_features,
            "scaler": "robust_z",
            "feature_list_hash": "sha256:7b3c1e...d2",
        },
        wall_time_ms=22.0,
    )


def stage_kernel_prep() -> StageResult:
    return StageResult(
        name="quantum_kernel_preparation",
        spec={
            "kernel_type": "fidelity",
            "feature_map": "ZZFeatureMap",
            "num_qubits": 8,
            "circuit_depth": 11,
            "support_size": 256,
        },
        wall_time_ms=18.0,
    )


def stage_qpu_eval(measured_kernel_dev: float) -> StageResult:
    return StageResult(
        name="qpu_kernel_evaluation",
        spec={
            "backend": "ibm_brisbane",
            "shots_per_entry": 4096,
            "num_kernel_entries": 256 * 256,
            "circuit_layout": "linear",
        },
        observables=[
            Observable(
                pauli="K_dev",
                reference=KERNEL_REFERENCE,
                epsilon=EPS_KERNEL,
                measured=measured_kernel_dev,
            ),
        ],
        wall_time_ms=37_500.0,
    )


def stage_classification() -> StageResult:
    return StageResult(
        name="classification",
        spec={
            "model": "kernel_svm",
            "C": 1.0,
            "decision_score_distribution": {
                "p50": 0.21,
                "p95": 0.74,
                "p99": 0.92,
            },
        },
        wall_time_ms=14.0,
    )


def stage_alert(alert_threshold: float) -> StageResult:
    return StageResult(
        name="alert_decision",
        spec={
            "alert_threshold": alert_threshold,
            "block_action": "freeze_card",
            "regulator": "FinCEN",
            "audit_retention_days": 2555,
        },
        wall_time_ms=2.0,
    )


# scenarios

def scenario_0_clean(commit) -> None:
    commit(stage_ingestion())
    commit(stage_features(num_features=8))
    commit(stage_kernel_prep())
    commit(stage_qpu_eval(measured_kernel_dev=0.018))
    commit(stage_classification())
    commit(stage_alert(alert_threshold=LEGITIMATE_ALERT_THRESHOLD))


def scenario_1_threshold_tamper(commit) -> None:
    """insider rewrites the alert_decision spec after it has been
    committed, raising the threshold from 0.65 to 0.95 so real fraud
    no longer trips an alert. streaming check passes (tamper happens
    after commit). verify_full_chain catches it later.
    """
    commit(stage_ingestion())
    commit(stage_features(num_features=8))
    commit(stage_kernel_prep())
    commit(stage_qpu_eval(measured_kernel_dev=0.018))
    commit(stage_classification())
    rec = commit(stage_alert(alert_threshold=LEGITIMATE_ALERT_THRESHOLD))
    rec.spec["alert_threshold"] = TAMPERED_ALERT_THRESHOLD


def scenario_2_kernel_poisoning(commit) -> None:
    """QPU returns a kernel matrix whose worst-case entry deviates
    well beyond epsilon. could be device drift or active manipulation.
    the engine halts at commit time, downstream stages do not run.
    """
    commit(stage_ingestion())
    commit(stage_features(num_features=8))
    commit(stage_kernel_prep())
    commit(stage_qpu_eval(measured_kernel_dev=0.13))
    commit(stage_classification())
    commit(stage_alert(alert_threshold=LEGITIMATE_ALERT_THRESHOLD))


def scenario_3_anchor_rewrite(commit, *, anchor: ExternalAnchor,
                              verifier: IntegrityVerifier) -> None:
    """global rewrite: a parallel chain rebuilt offline with a
    different feature set (12 features instead of 8). locally
    consistent but never anchored, so verify_against_anchor catches it.
    """
    commit(stage_ingestion())
    commit(stage_features(num_features=8))
    commit(stage_kernel_prep())
    commit(stage_qpu_eval(measured_kernel_dev=0.018))
    commit(stage_classification())
    commit(stage_alert(alert_threshold=LEGITIMATE_ALERT_THRESHOLD))

    parallel = IntegrityVerifier()
    parallel.commit_stage(stage_ingestion())
    parallel.commit_stage(stage_features(num_features=12))
    parallel.commit_stage(stage_kernel_prep())
    parallel.commit_stage(stage_qpu_eval(measured_kernel_dev=0.022))
    parallel.commit_stage(stage_classification())
    parallel.commit_stage(stage_alert(alert_threshold=LEGITIMATE_ALERT_THRESHOLD))

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

    section("DEMO 2.0  honest fraud-detection pipeline")
    anchor = ExternalAnchor(anchor_path)
    v = IntegrityVerifier(anchor=anchor)
    ok, _, _ = run_pipeline(scenario_0_clean, v, label="fraud_clean")
    print(f"  result: ok={ok}")
    print(f"  audit trail length: {len(v.records)} records")
    print(f"  total verify latency: "
          f"{sum(r.verify_latency_ms for r in v.records):.3f} ms")
    print(f"  alert threshold committed: "
          f"{v.records[-1].spec['alert_threshold']}")

    section("DEMO 2.1  insider raises alert threshold after commit")
    anchor = ExternalAnchor(anchor_path)
    v = IntegrityVerifier(anchor=anchor)
    ok, viol, _ = run_pipeline(
        scenario_1_threshold_tamper, v, label="fraud_tamper")
    print(f"  streaming result: ok={ok}")
    print("  running post-pipeline verify_full_chain()...")
    try:
        v.verify_full_chain()
        print("  RESULT: tamper undetected (this should never happen)")
    except IntegrityViolation as e:
        print(f"  RESULT: tamper DETECTED at stage {e.stage_index} "
              f"('{e.stage_name}'), kind={e.kind}")
        print(f"          tampered threshold: "
              f"{v.records[e.stage_index].spec['alert_threshold']}")

    section("DEMO 2.2  QPU kernel poisoning")
    anchor = ExternalAnchor(anchor_path)
    v = IntegrityVerifier(anchor=anchor)
    ok, viol, committed = run_pipeline(
        scenario_2_kernel_poisoning, v, label="fraud_poison")
    print(f"  result: ok={ok}")
    if viol is not None:
        print(f"  HALT at stage {viol.stage_index} "
              f"('{viol.stage_name}'), kind={viol.kind}")
        print(f"  stages committed before halt: "
              f"{[r.stage_name for r in committed]}")

    section("DEMO 2.3  globally consistent rewrite vs external anchor")
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

    section("DEMO 2 SUMMARY")
    print("  Scenario 0: honest pipeline    -> committed cleanly")
    print("  Scenario 1: threshold tamper   -> caught by chain replay")
    print("  Scenario 2: kernel poisoning   -> caught at commit time")
    print("              (downstream stages never executed)")
    print("  Scenario 3: global rewrite     -> caught by anchor check")
    print()


if __name__ == "__main__":
    main()
