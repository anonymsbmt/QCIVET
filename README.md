# QCIVET

Reference implementation and reproducibility artifacts for a runtime
integrity-verification framework for hybrid quantum-classical pipelines.

This repository is submitted for anonymous review. Author names,
affiliations and contact details have been removed.

---

## What the framework does

A hybrid pipeline runs on infrastructure its operator does not own:
cloud QPUs, third-party transpilers and middleware sit between the
specification a user writes and the result a downstream decision
consumes. QCIVET binds the two together with two independent checks.

**Record layer.** Every stage commits a canonical specification record
into a SHA-256 hash chain. Post-hoc tampering, fabricated stages and
omitted stages all break a link and are reported on replay. A globally
consistent rewrite, in which an attacker regenerates the whole chain
offline, passes the local check by construction; only an append-only
external anchor catches it.

**Semantic layer.** Every quantum stage additionally declares a family
of observables with reference values and a calibrated tolerance. A
substituted channel that agrees with the declared one on the monitored
observables passes; one that does not is halted at commit time, before
any downstream stage runs. The interesting case is a substitution
tuned against a weak contract: it can sit at the noise floor under a
single-Pauli check and be exposed by an informationally complete one.

---

## Repository contents

| File | Purpose |
|---|---|
| `qcivet_realtime.py` | Core engine: `IntegrityVerifier`, `StageResult`, `Observable`, `ExternalAnchor`, `CommitRecord`. Imported by all three application demos. |
| `quantum_oop_simulation.py` | Exact-analysis experiments: separation across candidate substitutions, synthetic depolarising-noise calibration, and a sweep over the over-rotation magnitude. |
| `quantum_oop_device_validation.py` | The same protocol under calibration-derived noise models for `FakeBrisbane` (Eagle r3) and `FakeFez` (Heron r2). |
| `quantum_oop_real_qpu.py` | Live run on a real IBM Heron r2 processor through the IBM Quantum cloud, using the `SamplerV2` primitive. Requires an IBM Quantum API token. |
| `hash_chain_demo.py` | Standalone record-layer prototype: honest baseline plus tampering, injection, skipping and full re-derivation. |
| `hash_chain_visualize.py` | Generates the multi-panel figure of the record-layer scenarios. |
| `qcivet_demo_vqe.py` | Application walkthrough: a six-stage VQE pipeline for ground-state energy estimation. |
| `qcivet_demo_fraud.py` | Application walkthrough: a six-stage quantum-kernel classification pipeline. |
| `qcivet_demo_cloud.py` | Application walkthrough: customer-side auditing of a cloud quantum service. |
| `qpu_results/` | Archived raw counts and per-cell summaries from the real-hardware run. |

---

## Reproducing the reported results

| Result | Produced by |
|---|---|
| Exact separation across the three candidates | `quantum_oop_simulation.py` |
| Calibration window (noise floor against detection target) | `quantum_oop_simulation.py`, function `plot_combined_calibration` |
| Separation under calibration-derived device noise | `quantum_oop_device_validation.py` |
| Separation on real hardware | `quantum_oop_real_qpu.py`, archived in `qpu_results/` |
| Record-layer scenarios | `hash_chain_demo.py`, figure by `hash_chain_visualize.py` |
| Anchor check against a globally consistent rewrite | scenario 3 of each `qcivet_demo_*.py` |
| Per-commit latency | stdout of any `qcivet_demo_*.py` |

---

## Installation

```bash
pip install -r requirements.txt
```

Tested on Python 3.10 and later with the versions pinned in
`requirements.txt`.

---

## Running

### Exact and simulated experiments

```bash
python quantum_oop_simulation.py
python quantum_oop_device_validation.py
```

The first runs without any quantum backend. The second builds noise
models from published IBM calibration data through
`NoiseModel.from_backend()`; these are simulations, not hardware runs.

### Application walkthroughs

```bash
python qcivet_demo_vqe.py
python qcivet_demo_fraud.py
python qcivet_demo_cloud.py
```

Each demo runs four scenarios on a six-stage pipeline: an honest
baseline, post-hoc specification tampering, a quantum-side deviation
beyond tolerance, and a globally consistent offline rewrite. The first
three are caught by the streaming check or by chain replay; the fourth
is caught only by the external anchor.

### Record layer in isolation

```bash
python hash_chain_demo.py
python hash_chain_visualize.py
```

### Real hardware

```bash
export QISKIT_IBM_TOKEN="your-token"
python quantum_oop_real_qpu.py
```

The script submits 54 circuits (3 candidates x 6 inputs x 3 Paulis) as
one batched job at 4096 shots each. On the archived run this took
137.8 s of wall time once the job left the queue. Queue waiting is
unpredictable and varies with backend load. Raw counts are written to
`qpu_results/` so the analysis can be repeated without resubmitting.

---

## Archived hardware run

| Field | Value |
|---|---|
| Backend | `ibm_fez` (IBM Heron r2, 156 qubits) |
| Job ID | `d7todq4t738s73ci59ug` |
| Date | 6 May 2026, 21:22 UTC |
| Plan | IBM Quantum Open Plan |
| Shots | 4096 per circuit |
| Total circuits | 54 |
| Queue waiting | approximately 15 s |
| Total wall time | 137.8 s |

### Worst-case observable deviation

| Candidate | worst {X, Y, Z} | worst {Z} |
|---|---|---|
| `B_good` | 0.0736 | 0.0614 |
| `B_bad` | 0.4853 | 0.3615 |
| `B_sneaky` | **1.4198** | 0.0790 |

The signature of the evading substitution survives on real hardware: a
large deviation under the informationally complete contract, and a
deviation at the honest noise floor under the single-Pauli one.

### Files

- `qpu_results/qpu_raw_ibm_fez_20260506_182451.json` - raw shot counts for every (candidate, input, Pauli) cell.
- `qpu_results/qpu_summary_ibm_fez_20260506_182451.json` - per-cell ideal, measured and deviation values, plus the headline worst-case numbers.

### Re-running the analysis from the archive

```bash
python quantum_oop_real_qpu.py --analyze-only \
    --raw qpu_results/qpu_raw_ibm_fez_20260506_182451.json
```

### Retrieving the original job

```python
from qiskit_ibm_runtime import QiskitRuntimeService
service = QiskitRuntimeService()
job = service.job("d7todq4t738s73ci59ug")
result = job.result()
```

---

## A note on gate order

The valid override `B_good` is built as `sdg`, `rx`, `s`. Qiskit applies
gates left to right in time, so the unitary is `S * Rx(theta) * S^dag`,
with the last gate leftmost in the matrix product. Since `S X S^dag = Y`,
this equals `Ry(theta)`, the declared channel `A`. Reading the product in
the opposite order would give `Ry(-theta)`, which is not what the circuit
does. The exact-analysis script reports a worst-case deviation of `0.000`
for `B_good`, and the archived hardware run gives `<X> = +0.9326` on the
`|0>` input, both consistent with `Ry(+theta)`.

---

## A note on tolerance scales

Different stages monitor different quantities, and their tolerances are
not interchangeable.

- A **Pauli-expectation** tolerance is dimensionless and bounded by the
  spectral radius of the observable. The archived hardware run gives an
  honest floor of 0.074 on `ibm_fez`, and the calibrated tolerance
  derived from it is 0.175.
- An **energy** tolerance is in Hartree. It relates to a per-Pauli
  tolerance through the Hamiltonian decomposition: for
  `H = sum_j c_j P_j`, a per-Pauli deviation of `eps` propagates to at
  most `eps * sum_j |c_j|` in energy.
- A **kernel-entry** or **tracer** deviation is again a different
  quantity with its own scale.

The tolerances hard-coded in the application walkthroughs are
illustrative and sit on the simulated-noise scale, where a Heron-class
model gives an honest floor near 0.028 and an Eagle-class model near
0.056. Real hardware is noisier: the archived run gives 0.074, a factor
of 2.6 above the Heron-class model. A deployment must therefore
calibrate on the backend that will run the workload. Calibrating
against a simulator understates the floor and halves the margin against
false alarms.

---

## Project structure

```
QCIVET/
+-- README.md
+-- LICENSE
+-- requirements.txt
+-- qcivet_realtime.py                 core engine
+-- quantum_oop_simulation.py          exact analysis and synthetic noise
+-- quantum_oop_device_validation.py   calibration-derived noise models
+-- quantum_oop_real_qpu.py            real IBM QPU run
+-- hash_chain_demo.py                 record-layer prototype
+-- hash_chain_visualize.py            record-layer figure generator
+-- qcivet_demo_vqe.py                 application walkthrough
+-- qcivet_demo_fraud.py               application walkthrough
+-- qcivet_demo_cloud.py               application walkthrough
+-- qpu_results/                       archived hardware run
    +-- qpu_raw_ibm_fez_20260506_182451.json
    +-- qpu_summary_ibm_fez_20260506_182451.json
```

---

## License

MIT. See `LICENSE`.
