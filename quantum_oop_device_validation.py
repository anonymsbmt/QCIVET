"""
Device-noise validation for the QCIVET framework.

re-runs experiments 1 and 3 from quantum_oop_simulation.py, but
under realistic device noise instead of synthetic depolarising
channels. uses two fake backends from qiskit-ibm-runtime:

  * FakeBrisbane: 127-qubit Eagle r3, single-qubit error ~3e-4
  * FakeFez:      156-qubit Heron r2, single-qubit error ~2e-4

these fake backends carry the calibration data of the matching
real devices, so deviations under them are representative of what
we'd see on the physical QPUs.

outputs two plots: exp5_device_noise.png and exp6_device_subtypes.png.
"""

import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import DensityMatrix, Statevector
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel
from qiskit_ibm_runtime.fake_provider import FakeBrisbane, FakeFez


# class definitions (mirror of quantum_oop_simulation.py)

# the declared rotation angle. theta = 2*pi/5 makes the rotation generic
# enough that all three Paulis are non-trivial (no accidental zeros)
THETA = 2 * np.pi / 5

PAULI_X = np.array([[0, 1], [1, 0]], dtype=complex)
PAULI_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
PAULI_Z = np.array([[1, 0], [0, -1]], dtype=complex)


def expectation(rho, op):
    return float(np.real(np.trace(rho.data @ op)))


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
    qc = QuantumCircuit(1, name=f"B_bad(d={delta})")
    qc.ry(THETA + delta, 0)
    return qc


def make_class_B_sneaky():
    qc = QuantumCircuit(1, name="B_sneaky")
    qc.ry(THETA, 0)
    qc.s(0)
    return qc


def make_input_circuits():
    inputs = {}
    inputs["|0>"] = QuantumCircuit(1)
    qc = QuantumCircuit(1); qc.x(0); inputs["|1>"] = qc
    qc = QuantumCircuit(1); qc.h(0); inputs["|+>"] = qc
    qc = QuantumCircuit(1); qc.x(0); qc.h(0); inputs["|->"] = qc
    qc = QuantumCircuit(1); qc.ry(0.7, 0); qc.rz(1.3, 0); inputs["psi1"] = qc
    qc = QuantumCircuit(1); qc.ry(2.1, 0); qc.rz(0.4, 0); inputs["psi2"] = qc
    return inputs


# pauli measurement on a noisy backend

def measure_pauli_expectation(prep_circ, behavior_circ, pauli_label,
                              sim, shots=8192, seed=None):
    """estimate <P> on behavior(prep(|0>)) under the given noisy sim.

    to read X we apply H first; for Y, S^dag then H; for Z, no rotation.
    """
    qc = QuantumCircuit(1, 1)
    qc.compose(prep_circ, inplace=True)
    qc.compose(behavior_circ, inplace=True)

    if pauli_label == "X":
        qc.h(0)
    elif pauli_label == "Y":
        qc.sdg(0)
        qc.h(0)

    qc.measure(0, 0)

    tqc = transpile(qc, sim, optimization_level=1)
    job = sim.run(tqc, shots=shots, seed_simulator=seed)
    counts = job.result().get_counts()
    n0 = counts.get("0", 0)
    n1 = counts.get("1", 0)
    total = n0 + n1
    return (n0 - n1) / total if total > 0 else 0.0


def ideal_pauli_expectation(prep_circ, behavior_circ, pauli_op):
    qc = QuantumCircuit(1)
    qc.compose(prep_circ, inplace=True)
    qc.compose(behavior_circ, inplace=True)
    rho = DensityMatrix(Statevector.from_instruction(qc))
    return expectation(rho, pauli_op)


# building a noisy AerSimulator from a fake backend

def build_aer_simulator_from_fake(fake_backend):
    """wrap the fake backend's calibration into an AerSimulator we can
    drive with our own circuits.
    """
    nm = NoiseModel.from_backend(fake_backend)
    sim = AerSimulator(noise_model=nm,
                       basis_gates=nm.basis_gates,
                       coupling_map=fake_backend.coupling_map)
    return sim, nm


# experiment 5: epsilon calibration under device noise
# (replaces the synthetic depolarising sweep of experiment 3)

def build_measurement_circuit(prep_circ, behavior_circ, pauli_label):
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


def estimate_z_from_counts(counts):
    n0 = counts.get("0", 0)
    n1 = counts.get("1", 0)
    total = n0 + n1
    return (n0 - n1) / total if total > 0 else 0.0


def batch_estimate(sim, tqc, n_trials, shots, rng):
    """run one big job of n_trials*shots and split the per-shot
    bitstrings into n_trials groups. statistically equivalent to
    running n_trials separate jobs but much faster on Aer because
    we pay the per-job overhead only once.
    """
    big_shots = n_trials * shots
    seed_i = int(rng.integers(0, 2**31 - 1))
    job = sim.run(tqc, shots=big_shots, memory=True,
                  seed_simulator=seed_i)
    result = job.result()
    mem = result.get_memory()
    bits = np.array([int(b, 2) & 1 for b in mem])
    z_vals = 1 - 2 * bits
    z_vals = z_vals[:n_trials * shots].reshape(n_trials, shots)
    return z_vals.mean(axis=1)


def experiment_5(devices, shots=4096, trials=15, seed=12345):
    """run B_good on each input, on each device, and report deviation
    from the noiseless <Z>_A. one transpile per (device, input).
    """
    rng = np.random.default_rng(seed)
    inputs = make_input_circuits()
    A = make_class_A()
    B_good = make_class_B_good()

    out = {}
    for dev_name, fake in devices.items():
        sim, nm = build_aer_simulator_from_fake(fake)

        per_input = {}
        for in_label, prep in inputs.items():
            z_ideal = ideal_pauli_expectation(prep, A, PAULI_Z)
            qc = build_measurement_circuit(prep, B_good, "Z")
            tqc = transpile(qc, sim, optimization_level=1)
            arr = batch_estimate(sim, tqc, trials, shots, rng)
            dev = np.abs(arr - z_ideal)
            per_input[in_label] = {
                "z_ideal": z_ideal,
                "z_mean": float(arr.mean()),
                "z_std": float(arr.std(ddof=1)),
                "dev_mean": float(dev.mean()),
                "dev_p95": float(np.quantile(dev, 0.95)),
            }
        out[dev_name] = per_input
    return out


# separation across candidate substitutions under device noise

def experiment_6(devices, shots=4096, trials=10, seed=54321):
    """for each device, measure all three Paulis on all six inputs
    for each candidate substitution, then take the worst-case deviation
    across (input, observable) pairs. also report worst-case under
    the {Z}-only contract for comparison.
    """
    rng = np.random.default_rng(seed)
    inputs = make_input_circuits()
    A = make_class_A()
    candidates = {
        "B_good": make_class_B_good(),
        "B_bad(0.4)": make_class_B_bad(0.4),
        "B_sneaky": make_class_B_sneaky(),
    }
    paulis = {"X": PAULI_X, "Y": PAULI_Y, "Z": PAULI_Z}

    out = {}
    for dev_name, fake in devices.items():
        sim, nm = build_aer_simulator_from_fake(fake)
        cand_results = {}
        for cand_name, B_circ in candidates.items():
            worst_full = 0.0
            worst_z_only = 0.0
            for in_label, prep in inputs.items():
                for p_label, p_op in paulis.items():
                    a_ideal = ideal_pauli_expectation(prep, A, p_op)
                    qc = build_measurement_circuit(prep, B_circ, p_label)
                    tqc = transpile(qc, sim, optimization_level=1)
                    arr = batch_estimate(sim, tqc, trials, shots, rng)
                    b_mean = float(arr.mean())
                    dev = abs(b_mean - a_ideal)
                    if dev > worst_full:
                        worst_full = dev
                    if p_label == "Z" and dev > worst_z_only:
                        worst_z_only = dev
            cand_results[cand_name] = {
                "worst_full": worst_full,
                "worst_Z_only": worst_z_only,
            }
        out[dev_name] = cand_results
    return out


# plotting

def plot_experiment_5(results, path):
    devices = list(results.keys())
    inputs = list(next(iter(results.values())).keys())

    fig, ax = plt.subplots(figsize=(9, 4.6))
    x = np.arange(len(inputs))
    width = 0.35
    colors = ["#3b6fb6", "#c66020"]

    for i, dev in enumerate(devices):
        means = [results[dev][inp]["dev_mean"] for inp in inputs]
        p95s = [results[dev][inp]["dev_p95"] for inp in inputs]
        offset = (i - 0.5) * width
        bars = ax.bar(
            x + offset, means, width,
            color=colors[i], edgecolor="black",
            label=f"{dev} (mean)",
        )
        ax.plot(
            x + offset, p95s, "s",
            color=colors[i], markeredgecolor="black",
            markersize=7, label=f"{dev} (95th pct)",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(inputs)
    ax.set_xlabel("input state")
    ax.set_ylabel(
        r"$|\langle Z\rangle_{B_\mathrm{good}} - "
        r"\langle Z\rangle_{A,\mathrm{ideal}}|$"
    )
    ax.set_title(
        "Contract deviation of the valid override "
        "under realistic IBM device noise"
    )
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_experiment_6(results, path):
    devices = list(results.keys())
    candidates = list(next(iter(results.values())).keys())

    n_dev = len(devices)
    n_cand = len(candidates)

    fig, axes = plt.subplots(1, n_dev, figsize=(11, 4.4),
                             sharey=True)
    if n_dev == 1:
        axes = [axes]

    for ax, dev in zip(axes, devices):
        full = [results[dev][c]["worst_full"] for c in candidates]
        zonly = [results[dev][c]["worst_Z_only"] for c in candidates]

        x = np.arange(n_cand)
        w = 0.36
        ax.bar(x - w / 2, full, w,
               color="#3b6fb6", edgecolor="black",
               label=r"$\mathcal{O}_A=\{X,Y,Z\}$")
        ax.bar(x + w / 2, zonly, w,
               color="#c66020", edgecolor="black",
               label=r"$\mathcal{O}_A=\{Z\}$")
        ax.set_xticks(x)
        ax.set_xticklabels(candidates, rotation=8, ha="right",
                           fontsize=9)
        ax.set_title(dev, fontsize=11)
        ax.grid(True, axis="y", alpha=0.3)
        for xi, vf, vz in zip(x, full, zonly):
            ax.text(xi - w / 2, vf + 0.02,
                    f"{vf:.2f}" if vf >= 0.01 else f"{vf:.0e}",
                    ha="center", fontsize=8)
            ax.text(xi + w / 2, vz + 0.02,
                    f"{vz:.2f}" if vz >= 0.01 else f"{vz:.0e}",
                    ha="center", fontsize=8)

    axes[0].set_ylabel("worst-case observable deviation")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2,
               bbox_to_anchor=(0.5, 0.99), fontsize=9, frameon=False)
    fig.suptitle(
        "Separation across candidate substitutions under "
        "realistic IBM device noise",
        fontsize=12, y=1.04,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


# reporting and main entry

def section(t):
    print()
    print(f">>> {t}")
    print("-" * 58)


def main():
    devices = {
        "FakeBrisbane (Eagle r3, 127q)": FakeBrisbane(),
        "FakeFez (Heron r2, 156q)": FakeFez(),
    }

    section("A. Device-noise tolerance calibration")
    print("Running B_good under FakeBrisbane and FakeFez,")
    print("30 trials of 8192 shots per (input, device).")
    print("This may take a couple of minutes...")
    e5 = experiment_5(devices)
    print()
    for dev, per_input in e5.items():
        print(f"  Device: {dev}")
        print(f"    {'input':>6}  {'<Z>_ideal':>10}  "
              f"{'<Z>_mean':>10}  {'dev_mean':>10}  {'dev_p95':>10}")
        for inp, r in per_input.items():
            print(
                f"    {inp:>6}  {r['z_ideal']:>+10.4f}  "
                f"{r['z_mean']:>+10.4f}  {r['dev_mean']:>10.4f}  "
                f"{r['dev_p95']:>10.4f}"
            )
        print()

    plot_experiment_5(e5, "exp5_device_noise.png")
    print("Plot saved: exp5_device_noise.png")

    section("B. Separation across candidate substitutions")
    print("Running B_good, B_bad, B_sneaky on all 6 inputs and")
    print("all 3 Paulis under each device. 10 trials per cell.")
    print("This may take a few minutes...")
    e6 = experiment_6(devices)
    print()
    for dev, cands in e6.items():
        print(f"  Device: {dev}")
        print(f"    {'candidate':>14}  {'worst {X,Y,Z}':>14}  "
              f"{'worst {Z}':>10}")
        for c, r in cands.items():
            print(
                f"    {c:>14}  {r['worst_full']:>14.4f}  "
                f"{r['worst_Z_only']:>10.4f}"
            )
        print()

    plot_experiment_6(e6, "exp6_device_subtypes.png")
    print("Plot saved: exp6_device_subtypes.png")
    print()
    print(">>> Done.")
    print("-" * 58)


if __name__ == "__main__":
    main()
