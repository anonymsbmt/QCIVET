"""
qcivet runtime engine.

streams stage commits into a hash chain, with an optional observable
check on quantum stages and an optional external timestamp anchor.
the host pipeline calls commit_stage(...) once per stage and catches
IntegrityViolation if anything goes wrong.

stdlib only.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional


# hashing

GENESIS = "0" * 64  # genesis: zero hash


def H(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical(spec: dict) -> bytes:
    return json.dumps(spec, sort_keys=True, separators=(",", ":")).encode()


# dataclasses

@dataclass
class Observable:
    """one Pauli measurement, its reference, and tolerance.

    measured starts as None; filled in once the stage actually runs.
    """
    pauli: str
    reference: float
    epsilon: float
    measured: Optional[float] = None

    def deviation(self) -> float:
        if self.measured is None:
            raise ValueError("observable has no measured value yet")
        return abs(self.measured - self.reference)


@dataclass
class StageResult:
    """what a stage hands to the verifier when it finishes."""
    name: str
    spec: dict
    observables: list = field(default_factory=list)
    wall_time_ms: float = 0.0


@dataclass
class CommitRecord:
    """one entry in the audit log."""
    index: int
    stage_name: str
    spec: dict
    observables_snapshot: list
    prev_hash: str
    hash: str
    commit_time_unix: float
    verify_latency_ms: float


class IntegrityViolation(Exception):
    """raised when a stage fails a check.
    kind is one of: 'hash', 'observable', 'anchor'.
    """
    def __init__(self, message, stage_index, stage_name, kind):
        super().__init__(message)
        self.stage_index = stage_index
        self.stage_name = stage_name
        self.kind = kind


# external anchor

class ExternalAnchor:
    """stand-in for an RFC-3161 / Sigstore-Rekor / blockchain
    timestamp service. writes each submitted hash to a local
    append-only file. swap with a real service in production.
    """

    def __init__(self, log_path="qcivet_anchor.log"):
        self.log_path = Path(log_path)
        self.log_path.touch(exist_ok=True)

    def submit(self, payload_hash, label):
        line = json.dumps({
            "ts": time.time(),
            "label": label,
            "hash": payload_hash,
        })
        with self.log_path.open("a") as f:
            f.write(line + "\n")

    def all_anchored_hashes(self):
        out = []
        with self.log_path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                out.append(json.loads(line)["hash"])
        return out


# verifier

class IntegrityVerifier:
    """the verifier the host pipeline drives.

    typical use:

        v = IntegrityVerifier(anchor=ExternalAnchor("rekor.log"))
        try:
            for stage in pipeline:
                result = stage.run()
                v.commit_stage(StageResult(name=stage.name, spec=...))
        except IntegrityViolation as e:
            pipeline.abort()

    each commit is one hash plus a tail check; submillisecond on
    a normal laptop.
    """

    def __init__(self, anchor=None, expected_genesis=GENESIS):
        self.records = []
        self.anchor = anchor
        self.expected_genesis = expected_genesis
        self._head = expected_genesis

    def commit_stage(self, result):
        """append one stage to the chain. raises IntegrityViolation."""
        t0 = time.perf_counter()

        self._check_observables(result, len(self.records))

        prev = self._head
        new_hash = H(prev.encode() + canonical(result.spec))

        if self.records:
            if self.records[-1].hash != prev:
                raise IntegrityViolation(
                    f"chain head mismatch before committing "
                    f"'{result.name}': internal state desync",
                    stage_index=len(self.records),
                    stage_name=result.name,
                    kind="hash",
                )

        rec = CommitRecord(
            index=len(self.records),
            stage_name=result.name,
            spec=result.spec,
            observables_snapshot=[asdict(o) for o in result.observables],
            prev_hash=prev,
            hash=new_hash,
            commit_time_unix=time.time(),
            verify_latency_ms=(time.perf_counter() - t0) * 1000.0,
        )
        self.records.append(rec)
        self._head = new_hash

        if self.anchor is not None:
            try:
                self.anchor.submit(
                    new_hash, label=f"{rec.index}:{result.name}")
            except Exception as e:
                raise IntegrityViolation(
                    f"failed to anchor stage '{result.name}': {e}",
                    stage_index=rec.index,
                    stage_name=result.name,
                    kind="anchor",
                )

        return rec

    def verify_full_chain(self):
        """recompute every hash from scratch.

        called at end-of-pipeline (or on replay) to catch tampering
        that happened after a record was committed; the streaming
        check during commit_stage cannot see those.
        """
        prev = self.expected_genesis
        for i, rec in enumerate(self.records):
            if rec.prev_hash != prev:
                raise IntegrityViolation(
                    f"link broken at index {i}: stored prev "
                    f"{rec.prev_hash[:12]}... != expected "
                    f"{prev[:12]}...",
                    stage_index=i,
                    stage_name=rec.stage_name,
                    kind="hash",
                )
            recomputed = H(prev.encode() + canonical(rec.spec))
            if recomputed != rec.hash:
                raise IntegrityViolation(
                    f"hash mismatch at index {i} (stage "
                    f"'{rec.stage_name}'): spec was modified after "
                    f"the record was written",
                    stage_index=i,
                    stage_name=rec.stage_name,
                    kind="hash",
                )
            prev = rec.hash

    def verify_against_anchor(self):
        """compare local chain to the anchor log.

        catches global rewrites: a locally-consistent chain that
        was never anchored will not match.
        """
        if self.anchor is None:
            return
        anchored = self.anchor.all_anchored_hashes()
        local_hashes = [r.hash for r in self.records]
        n = len(local_hashes)
        if n == 0:
            return
        for start in range(len(anchored) - n + 1):
            if anchored[start:start + n] == local_hashes:
                return
        raise IntegrityViolation(
            "external anchor mismatch: the local chain cannot be "
            "located as a contiguous block in the anchor log "
            "(globally consistent rewrite suspected)",
            stage_index=len(self.records) - 1,
            stage_name=self.records[-1].stage_name,
            kind="anchor",
        )

    def head_hash(self):
        return self._head

    def export_audit_trail(self):
        return [asdict(r) for r in self.records]

    def _check_observables(self, result, stage_index):
        for obs in result.observables:
            if obs.measured is None:
                continue
            if obs.deviation() > obs.epsilon:
                raise IntegrityViolation(
                    f"observable {obs.pauli} on stage "
                    f"'{result.name}' violates contract: "
                    f"|measured - reference| = "
                    f"{obs.deviation():.4f} > eps = {obs.epsilon:.4f}",
                    stage_index=stage_index,
                    stage_name=result.name,
                    kind="observable",
                )


# convenience helper for the demos

def run_pipeline(pipeline_fn, verifier, label="pipeline", verbose=True):
    """drive a pipeline; gracefully handle IntegrityViolation.

    pipeline_fn(commit) should call commit(stage_result) for each
    finished stage. returns (ok, violation_or_None, committed_records).
    """
    if verbose:
        print(f"\n[{label}] starting pipeline")
    committed = []

    def commit(stage_result):
        rec = verifier.commit_stage(stage_result)
        committed.append(rec)
        if verbose:
            obs_summary = ""
            if stage_result.observables:
                worst = max(
                    (o.deviation() for o in stage_result.observables
                     if o.measured is not None),
                    default=0.0,
                )
                obs_summary = f"  worst-obs-dev={worst:.4f}"
            print(
                f"[{label}] stage {rec.index} '{rec.stage_name}' "
                f"committed in {rec.verify_latency_ms:.2f} ms"
                f"{obs_summary}"
            )
        return rec

    try:
        pipeline_fn(commit)
    except IntegrityViolation as e:
        if verbose:
            print(
                f"[{label}] HALT at stage {e.stage_index} "
                f"('{e.stage_name}', kind={e.kind}): {e}"
            )
        return False, e, committed

    if verbose:
        print(f"[{label}] pipeline completed cleanly")
    return True, None, committed
