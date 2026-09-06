"""
real IBM QPU run for the device validation experiments.

runs the same setup as the FakeBrisbane / FakeFez simulations
but on a real IBM quantum processor accessed
through the IBM Quantum cloud.

before you run this:
  1. create a free IBM Quantum account at https://quantum.ibm.com
  2. copy your API token from the account page
  3. set it as an environment variable:
        export IBM_QUANTUM_TOKEN="your-token-here"
     or paste it directly into the TOKEN variable below
  4. install the packages:
        pip install qiskit qiskit-ibm-runtime numpy

what the script does:
  - connects to IBM Quantum
  - picks the least-busy real device
  - submits one batch (3 candidates x 6 inputs x 3 paulis = 54 circuits)
  - waits for the job to finish
  - computes the worst-case observable deviations from experiment 6
  - saves the raw counts to JSON so you can re-analyze later

quota: a single full run is roughly 30-90 seconds of QPU compute time,
well within the 10-minute monthly Open Plan allowance. queue waiting
time is unpredictable - anywhere from minutes to hours.
"""

import json
import os
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler


# configuration

# paste your token
TOKEN = os.environ.get("IBM_QUANTUM_TOKEN", "PASTE-YOUR-TOKEN-HERE")

# shots per measurement
SHOTS = 4096

# trials per measurement
TRIALS = 1

# output directory
OUTDIR = Path("./qpu_results")
OUTDIR.mkdir(exist_ok=True)


# experiment definition (mirror of quantum_oop_simulation)

THETA = 2 * np.pi / 5

PAULI_X = np.array([[0, 1], [1, 0]], dtype=complex)
PAULI_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
PAULI_Z = np.array([[1, 0], [0, -1]], dtype=complex)


def make_class_A():
    qc = QuantumCircuit(1, name="A")
    qc.ry(THETA, 0)
    return qc


def make_class_B_good():
    # Qiskit applies gates left to right in time, so the unitary is
    # S * Rx(theta) * S^dag (the last gate is leftmost in the product).
    # Since S X S^dag = Y, this equals Ry(theta) = A exactly.
    # Check with: Operator(qc).data
    qc = QuantumCircuit(1, name="B_good")
    qc.sdg(0)
    qc.rx(THETA, 0)
    qc.s(0)
    return qc


def make_class_B_bad(delta=0.4):
    qc = QuantumCircuit(1, name=f"B_bad_d{delta}")
    qc.ry(THETA + delta, 0)
    return qc


def make_class_B_sneaky():
    qc = QuantumCircuit(1, name="B_sneaky")
    qc.ry(THETA, 0)
    qc.s(0)
    return qc


def make_input_circuits():
    inputs = {}
    inputs["s0"] = QuantumCircuit(1)
    qc = QuantumCircuit(1); qc.x(0); inputs["s1"] = qc
    qc = QuantumCircuit(1); qc.h(0); inputs["sp"] = qc
    qc = QuantumCircuit(1); qc.x(0); qc.h(0); inputs["sm"] = qc
    qc = QuantumCircuit(1); qc.ry(0.7, 0); qc.rz(1.3, 0)
    inputs["psi1"] = qc
    qc = QuantumCircuit(1); qc.ry(2.1, 0); qc.rz(0.4, 0)
    inputs["psi2"] = qc
    return inputs


def build_measurement_circuit(prep_circ, behavior_circ, pauli_label):
    """build a 1-qubit circuit that prepares the input, applies the
    candidate behavior, rotates to the requested Pauli basis, and
    measures."""
    qc = QuantumCircuit(1, 1)
    qc.compose(prep_circ, inplace=True)
    qc.compose(behavior_circ, inplace=True)
    if pauli_label == "X":
        qc.h(0)
    elif pauli_label == "Y":
        qc.sdg(0)
        qc.h(0)
    qc.measure(0, 0)
    return qc


def expectation_from_counts(counts):
    """convert a {bitstring: count} dict into <P> in [-1, +1]."""
    n0 = counts.get("0", 0) + counts.get("0 ", 0)
    n1 = counts.get("1", 0) + counts.get("1 ", 0)
    total = n0 + n1
    return (n0 - n1) / total if total > 0 else 0.0


def ideal_pauli_expectation_A(prep_circ, pauli_op):
    """noiseless reference value of <P> after applying class A on
    the prepared input."""
    from qiskit.quantum_info import Statevector, DensityMatrix
    qc = QuantumCircuit(1)
    qc.compose(prep_circ, inplace=True)
    qc.compose(make_class_A(), inplace=True)
    rho = DensityMatrix(Statevector.from_instruction(qc))
    return float(np.real(np.trace(rho.data @ pauli_op)))


# connect to IBM Quantum

def connect_and_pick_backend():
    """save the token, instantiate a service, return the least-busy
    real device that we can run on."""
    print("Connecting to IBM Quantum...")
    if TOKEN == "PASTE-YOUR-TOKEN-HERE":
        raise RuntimeError(
            "Please set IBM_QUANTUM_TOKEN env var or paste your "
            "token into the TOKEN variable at the top of this file."
        )

    service = QiskitRuntimeService(
        channel="ibm_quantum_platform",
        token=TOKEN,
    )

    # pick the least busy real device
    backend = service.least_busy(
        operational=True,
        simulator=False,
        min_num_qubits=1,
    )
    print(f"  Selected backend: {backend.name}")
    print(f"    num_qubits     = {backend.num_qubits}")
    print(f"    pending jobs   = {backend.status().pending_jobs}")
    return service, backend


# build the full circuit batch

def build_all_circuits():
    """return a list of (label, circuit) for every (candidate, input,
    pauli) combination."""
    candidates = {
        "B_good":   make_class_B_good(),
        "B_bad":    make_class_B_bad(0.4),
        "B_sneaky": make_class_B_sneaky(),
    }
    inputs = make_input_circuits()
    paulis = ["X", "Y", "Z"]

    items = []
    for cand_name, B in candidates.items():
        for in_label, prep in inputs.items():
            for p in paulis:
                qc = build_measurement_circuit(prep, B, p)
                label = f"{cand_name}__{in_label}__{p}"
                items.append((label, qc))
    return items


# run everything on the QPU

def run_on_qpu(backend, items, shots=SHOTS):
    """transpile every circuit for the chosen backend, then submit
    them as one batched Sampler job."""
    print(f"\nTranspiling {len(items)} circuits for {backend.name}...")
    labels, circs = zip(*items)
    # transpile to backend basis gates
    tcircs = transpile(list(circs), backend, optimization_level=1)
    print("  Transpilation done.")

    print(f"\nSubmitting one Sampler job with {len(items)} circuits "
          f"x {shots} shots each...")
    sampler = Sampler(mode=backend)
    job = sampler.run(list(tcircs), shots=shots)
    print(f"  Job ID: {job.job_id()}")
    print(f"  Initial status: {job.status()}")

    # poll until done
    print("\nWaiting for the job to complete. This may take a while.")
    print("Hit Ctrl-C if you need to abort; you can recover the job "
          "later via service.job(<job_id>).")
    t0 = time.time()
    last_status = None
    while True:
        status = job.status()
        if str(status) != last_status:
            elapsed = time.time() - t0
            print(f"  [{elapsed:7.1f}s] status: {status}")
            last_status = str(status)
        if str(status) in ("DONE", "ERROR", "CANCELLED"):
            break
        time.sleep(15)

    if str(job.status()) != "DONE":
        raise RuntimeError(f"Job ended in state {job.status()}")
    print(f"  Total wall time: {time.time() - t0:.1f}s")

    result = job.result()
    print("\nCollecting counts from the Sampler result...")
    out = {}
    for label, pub_result in zip(labels, result):
        # extract the counts (register name varies)
        data = pub_result.data
        reg_name = list(data.keys())[0] if hasattr(data, "keys") \
            else next(iter(data._fields))
        bitarr = getattr(data, reg_name) if not hasattr(data, "keys") \
            else data[reg_name]
        counts = bitarr.get_counts()
        out[label] = counts

    return out, job.job_id()


# analyze the counts

def analyze(counts_by_label):
    """aggregate worst-case observable deviation per candidate."""
    inputs = make_input_circuits()
    paulis = {"X": PAULI_X, "Y": PAULI_Y, "Z": PAULI_Z}
    candidates = ["B_good", "B_bad", "B_sneaky"]

    summary = {}
    full_table = []
    for cand in candidates:
        worst_full = 0.0
        worst_z_only = 0.0
        for in_label, prep in inputs.items():
            for p_label, p_op in paulis.items():
                key = f"{cand}__{in_label}__{p_label}"
                counts = counts_by_label[key]
                b_estimate = expectation_from_counts(counts)
                a_ideal = ideal_pauli_expectation_A(prep, p_op)
                dev = abs(b_estimate - a_ideal)
                full_table.append({
                    "candidate": cand,
                    "input": in_label,
                    "pauli": p_label,
                    "a_ideal": a_ideal,
                    "b_qpu_estimate": b_estimate,
                    "deviation": dev,
                    "counts": counts,
                })
                if dev > worst_full:
                    worst_full = dev
                if p_label == "Z" and dev > worst_z_only:
                    worst_z_only = dev
        summary[cand] = {
            "worst_full": worst_full,
            "worst_Z_only": worst_z_only,
        }
    return summary, full_table


def print_summary(backend_name, summary):
    print()
    print(f">>> RESULTS ON {backend_name}")
    print("-" * 58)
    print(f"  {'candidate':>10}  {'worst {X,Y,Z}':>14}  {'worst {Z}':>10}")
    for c, r in summary.items():
        print(f"  {c:>10}  {r['worst_full']:>14.4f}  "
              f"{r['worst_Z_only']:>10.4f}")
    print()
    print("  Expected pattern:")
    print("    B_good   : both columns SMALL (~ device noise floor)")
    print("    B_bad    : both columns ~0.4   (genuine violation)")
    print("    B_sneaky : full LARGE (~1.4), Z-only SMALL")


# main

def main():
    service, backend = connect_and_pick_backend()
    items = build_all_circuits()

    counts_by_label, job_id = run_on_qpu(backend, items)

    # save raw counts so you never have to re-submit
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    raw_path = OUTDIR / f"qpu_raw_{backend.name}_{timestamp}.json"
    with open(raw_path, "w") as f:
        json.dump({
            "backend": backend.name,
            "job_id": job_id,
            "shots": SHOTS,
            "counts_by_label": counts_by_label,
        }, f, indent=2)
    print(f"\nRaw counts saved to: {raw_path}")

    summary, full_table = analyze(counts_by_label)
    print_summary(backend.name, summary)

    analysis_path = OUTDIR / f"qpu_summary_{backend.name}_{timestamp}.json"
    with open(analysis_path, "w") as f:
        json.dump({
            "backend": backend.name,
            "summary": summary,
            "full_table": full_table,
        }, f, indent=2)
    print(f"Summary saved to: {analysis_path}")


if __name__ == "__main__":
    main()
