"""
Record-layer demonstration for the QCIVET framework.

builds a six-stage hybrid QPU pipeline, hashes every stage spec
into a chain (h_i = SHA256(h_{i-1} || canonical(spec_i))), then
runs four adversary scenarios:

  * tampering: edit a stored spec after the fact
  * injection: insert a fake stage record
  * skipping:  drop a real stage
  * full rewrite: recompute every hash to fake consistency

the verifier catches the first three by replay. the fourth
needs an external anchor (a previously-published hash) to
detect.

stdlib only - no qiskit, no external deps.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, asdict
from typing import Optional


# stages and records

@dataclass
class Stage:
    """one stage in the pipeline.

    name is just a label. spec is the contract we're protecting -
    in a real pipeline this is compiler version, backend id,
    calibration data, etc.
    """
    name: str
    spec: dict


@dataclass
class ChainRecord:
    """one entry in the audit chain after hashing a stage."""
    index: int
    stage: Stage
    prev_hash: str
    hash: str

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "stage_name": self.stage.name,
            "stage_spec": self.stage.spec,
            "prev_hash": self.prev_hash,
            "hash": self.hash,
        }


# hash + chain building

def H(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_bytes(spec: dict) -> bytes:
    return json.dumps(spec, sort_keys=True, separators=(",", ":")).encode()


def stage_hash(prev_hash: str, stage: Stage) -> str:
    return H(prev_hash.encode() + canonical_bytes(stage.spec))


# zero hash as the pre-pipeline anchor
GENESIS_HASH = "0" * 64


def build_chain(stages: list[Stage]) -> list[ChainRecord]:
    chain: list[ChainRecord] = []
    prev = GENESIS_HASH
    for i, st in enumerate(stages):
        h = stage_hash(prev, st)
        chain.append(ChainRecord(index=i, stage=st, prev_hash=prev, hash=h))
        prev = h
    return chain


# verifier

@dataclass
class VerificationResult:
    ok: bool
    reason: Optional[str] = None
    failed_index: Optional[int] = None


def verify_chain(
    records: list[ChainRecord],
    expected_genesis: str = GENESIS_HASH,
) -> VerificationResult:
    """rebuild every hash from scratch and check.

    catches:
      - spec tampering (a stored hash won't match the recomputed one)
      - missing or injected records (prev_hash linkage breaks)
      - wrong starting anchor (genesis mismatch)
    """
    if not records:
        return VerificationResult(False, "empty chain", None)

    if records[0].prev_hash != expected_genesis:
        return VerificationResult(
            False,
            f"genesis mismatch: expected {expected_genesis[:12]}..., "
            f"got {records[0].prev_hash[:12]}...",
            failed_index=0,
        )

    prev = expected_genesis
    for i, rec in enumerate(records):
        if rec.prev_hash != prev:
            return VerificationResult(
                False,
                f"link broken at index {i}: stored prev_hash "
                f"{rec.prev_hash[:12]}... != expected {prev[:12]}...",
                failed_index=i,
            )
        recomputed = stage_hash(prev, rec.stage)
        if recomputed != rec.hash:
            return VerificationResult(
                False,
                f"hash mismatch at index {i} "
                f"(stage '{rec.stage.name}'): stage spec has been "
                f"modified after the record was written",
                failed_index=i,
            )
        prev = rec.hash

    return VerificationResult(True, None, None)


# a six-stage hybrid QPU pipeline

def example_pipeline() -> list[Stage]:
    return [
        Stage(
            name="circuit_definition",
            spec={
                "algorithm": "VQE_H2",
                "n_qubits": 4,
                "ansatz": "hardware_efficient",
                "depth": 3,
                "version": "1.0.2",
            },
        ),
        Stage(
            name="transpilation",
            spec={
                "compiler": "qiskit-transpiler",
                "compiler_version": "2.4.1",
                "optimization_level": 3,
                "basis_gates": ["rz", "sx", "cx"],
            },
        ),
        Stage(
            name="backend_selection",
            spec={
                "provider": "ibm_quantum",
                "backend_id": "ibm_brisbane",
                "n_qubits": 127,
                "selected_at": "2026-04-29T10:15:00Z",
            },
        ),
        Stage(
            name="calibration",
            spec={
                "calibration_id": "cal_2026_04_29_07h",
                "T1_avg_us": 250.4,
                "T2_avg_us": 110.2,
                "single_qubit_error_avg": 0.00031,
                "two_qubit_error_avg": 0.0078,
            },
        ),
        Stage(
            name="execution",
            spec={
                "shots": 8192,
                "rep_delay_us": 250,
                "meas_level": 2,
                "job_id": "job_a1b2c3d4",
            },
        ),
        Stage(
            name="measurement_output",
            spec={
                "format": "counts_dict",
                "post_processing": "readout_error_mitigation",
                "n_distinct_outcomes": 14,
            },
        ),
    ]


# printing

def section(title: str) -> None:
    print()
    print(f">>> {title}")
    print("-" * 58)


def short(h: str, n: int = 12) -> str:
    return h[:n] + "..."


def print_chain(records: list[ChainRecord]) -> None:
    for rec in records:
        print(
            f"  [{rec.index}] {rec.stage.name:<22}  "
            f"prev={short(rec.prev_hash)}  hash={short(rec.hash)}"
        )


def print_verdict(label: str, res: VerificationResult) -> None:
    if res.ok:
        print(f"  Verifier on {label}: OK -- chain is consistent")
    else:
        idx = res.failed_index if res.failed_index is not None else "?"
        print(
            f"  Verifier on {label}: FAILED at index {idx}\n"
            f"    Reason: {res.reason}"
        )


# scenarios

def scenario_baseline() -> None:
    section("SCENARIO 0: honest pipeline (baseline)")
    stages = example_pipeline()
    chain = build_chain(stages)
    print_chain(chain)
    res = verify_chain(chain)
    print_verdict("honest chain", res)


def scenario_tampering() -> None:
    """attacker edits a stored spec after the chain was built.

    here: swap the calibration record so a stale, drifted calibration
    looks fresh. the attacker leaves the stored hash untouched - they
    don't know to recompute it (or they're lazy).
    """
    section("SCENARIO 1: A1 -- tampering with a stage spec")
    stages = example_pipeline()
    chain = build_chain(stages)

    print("  Adversary action: silently modifies the calibration "
          "record (index 3),")
    print("  changing single_qubit_error_avg from 0.00031 to "
          "0.00007 to hide drift.")

    tampered = [
        ChainRecord(
            index=r.index,
            stage=Stage(
                name=r.stage.name,
                spec={**r.stage.spec, "single_qubit_error_avg": 0.00007}
                if r.index == 3 else dict(r.stage.spec),
            ),
            prev_hash=r.prev_hash,
            hash=r.hash,
        )
        for r in chain
    ]
    res = verify_chain(tampered)
    print_verdict("tampered chain", res)


def scenario_injection() -> None:
    """attacker inserts a fake stage record into the chain.

    the fake record looks self-consistent (prev_hash + hash chain
    locally), but legitimate stages downstream still hold their
    original prev_hash, which now points to the wrong place.
    """
    section("SCENARIO 2: A2 -- injection of a fake stage")
    stages = example_pipeline()
    chain = build_chain(stages)

    print("  Adversary action: injects a fake 'post_calibration_patch'")
    print("  stage between calibration (3) and execution (4),")
    print("  trying to tweak parameters mid-flight.")

    fake_stage = Stage(
        name="post_calibration_patch",
        spec={
            "patch_id": "p_zz_attack",
            "patched_two_qubit_error_avg": 0.001,
        },
    )
    fake_prev = chain[3].hash
    fake_hash = stage_hash(fake_prev, fake_stage)
    fake_record = ChainRecord(
        index=4, stage=fake_stage, prev_hash=fake_prev, hash=fake_hash,
    )

    injected = chain[:4] + [fake_record] + [
        ChainRecord(
            index=r.index + 1,
            stage=r.stage,
            prev_hash=r.prev_hash,
            hash=r.hash,
        )
        for r in chain[4:]
    ]
    res = verify_chain(injected)
    print_verdict("injected chain", res)


def scenario_skipping() -> None:
    """attacker drops a stage. tries to make the run look like it
    used a stable pre-approved configuration when one stage was
    silently removed.
    """
    section("SCENARIO 3: A3 -- skipping a legitimate stage")
    stages = example_pipeline()
    chain = build_chain(stages)

    print("  Adversary action: silently drops the 'calibration' "
          "stage (index 3),")
    print("  trying to make the run look as if it used a stable, "
          "pre-approved calibration.")

    skipped = chain[:3] + [
        ChainRecord(
            index=r.index - 1,
            stage=r.stage,
            prev_hash=r.prev_hash,
            hash=r.hash,
        )
        for r in chain[4:]
    ]
    res = verify_chain(skipped)
    print_verdict("skipped chain", res)


def scenario_full_replay() -> None:
    """smarter attacker: tampers with one spec AND recomputes every
    downstream hash so the chain is locally self-consistent.

    the only way to catch this is comparing against an externally
    published hash (timestamping service, append-only log, etc.).
    """
    section("SCENARIO 4: full re-derivation after tampering")
    stages = example_pipeline()
    chain = build_chain(stages)

    published_final_hash = chain[-1].hash

    print("  Adversary action: tampers with calibration AND "
          "recomputes every downstream hash so the chain looks "
          "internally consistent.")

    new_stages = [Stage(s.name, dict(s.spec)) for s in stages]
    new_stages[3].spec["single_qubit_error_avg"] = 0.00007
    rebuilt = build_chain(new_stages)

    res_internal = verify_chain(rebuilt)
    print_verdict("rebuilt chain (internal check only)", res_internal)

    print()
    print(f"  Externally anchored final hash : {short(published_final_hash)}")
    print(f"  Rebuilt chain final hash       : {short(rebuilt[-1].hash)}")
    if rebuilt[-1].hash == published_final_hash:
        print("  External anchor: MATCH (this should never happen)")
    else:
        print("  External anchor: MISMATCH -- tampering detected by")
        print("                   external observer even though the")
        print("                   internal chain is self-consistent.")


# main

if __name__ == "__main__":
    scenario_baseline()
    scenario_tampering()
    scenario_injection()
    scenario_skipping()
    scenario_full_replay()

    print()
    print(">>> SUMMARY")
    print("-" * 58)
    print(
        "  - Tampering with a stored spec is caught by recomputing\n"
        "    the stage hash and comparing with the stored hash.\n"
        "  - Injection is caught when the next record's prev_hash\n"
        "    does not match the (now-different) hash sequence.\n"
        "  - Skipping is caught by the same prev_hash linkage.\n"
        "  - A globally consistent rewrite still fails against an\n"
        "    externally anchored final hash (timestamping service\n"
        "    or append-only log)."
    )
