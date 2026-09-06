"""
Exact-analysis experiments for the QCIVET framework.

four numerical experiments on a single qubit:
  1. substitution separation in the noiseless setting
  2. encapsulation as partial trace on the post-CNOT state
  3. epsilon calibration under synthetic depolarising noise
  4. delta-sweep for the over-rotated (bad) candidate

the declared channel A is R_y(2*pi/5). same setup as the device-noise
scripts, so the numbers here line up
cell-for-cell with the noisy ones there. three candidate
overrides B_good, B_sneaky, B_bad. all seeds are fixed.

needs: qiskit, qiskit-aer, numpy, matplotlib.
"""

import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import (
    DensityMatrix,
    Statevector,
    partial_trace,
)
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error


# pauli observables and quick utilities

# the three single-qubit Pauli matrices, indexed by their letter
PAULI = {
    "X": np.array([[0, 1], [1, 0]], dtype=complex),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "Z": np.array([[1, 0], [0, -1]], dtype=complex),
}


def expectation(rho: DensityMatrix, op_label: str) -> float:
    O = PAULI[op_label]
    return float(np.real(np.trace(rho.data @ O)))


def density_from_circuit(qc: QuantumCircuit, init_state=None) -> DensityMatrix:
    if init_state is None:
        sv = Statevector.from_instruction(qc)
    else:
        sv = init_state.evolve(qc)
    return DensityMatrix(sv)


# reference single-qubit input states we'll test on. same six
# states the device-noise scripts use.

def reference_inputs() -> dict:
    s0 = Statevector.from_label("0")
    s1 = Statevector.from_label("1")
    splus = Statevector.from_label("+")
    sminus = Statevector.from_label("-")
    # psi1, psi2: off-axis states, same as device_validation.py
    psi1_qc = QuantumCircuit(1)
    psi1_qc.ry(0.7, 0)
    psi1_qc.rz(1.3, 0)
    psi1 = Statevector.from_instruction(psi1_qc)
    psi2_qc = QuantumCircuit(1)
    psi2_qc.ry(2.1, 0)
    psi2_qc.rz(0.4, 0)
    psi2 = Statevector.from_instruction(psi2_qc)
    return {
        "|0>": s0, "|1>": s1,
        "|+>": splus, "|->": sminus,
        "psi1": psi1, "psi2": psi2,
    }


# the declared channel A and the three candidate substitutions.
# theta = 2*pi/5 - a generic angle that keeps all three Pauli
# expectations of A non-zero on every input. matches
# device_validation.py exactly.
THETA = 2 * np.pi / 5

# B_good   : same channel as A, written via S^dag R_x S. valid override.
# B_sneaky : R_y(theta) followed by S. preserves <Z>, flips <Y>. the
#            sneaky case that motivates a tomographically rich contract.
# B_bad    : misrotated by delta. obvious violation that grows with delta.

def make_class_A() -> QuantumCircuit:
    qc = QuantumCircuit(1, name="A")
    qc.ry(THETA, 0)
    return qc


def make_class_B_good() -> QuantumCircuit:
    # Qiskit applies gates left to right in time, so the unitary is
    # S * Rx(theta) * S^dag (the last gate is leftmost in the product).
    # Since S X S^dag = Y, this equals Ry(theta) = A exactly.
    # Check with: Operator(qc).data
    qc = QuantumCircuit(1, name="B_good")
    qc.sdg(0)
    qc.rx(THETA, 0)
    qc.s(0)
    return qc


def make_class_B_sneaky() -> QuantumCircuit:
    qc = QuantumCircuit(1, name="B_sneaky")
    qc.ry(THETA, 0)
    qc.s(0)
    return qc


def make_class_B_bad(delta: float) -> QuantumCircuit:
    qc = QuantumCircuit(1, name=f"B_bad({delta:+.2f})")
    qc.ry(THETA + delta, 0)
    return qc


# the contract preservation check

def worst_case_deviation(
    circ_super: QuantumCircuit,
    circ_sub: QuantumCircuit,
    observables=("X", "Y", "Z"),
    inputs: dict = None,
) -> dict:
    """sup over inputs and observables of |<O>_super - <O>_sub|.
    returns the per-pair table and the max."""
    if inputs is None:
        inputs = reference_inputs()

    table = {}
    sup = 0.0
    for in_name, in_state in inputs.items():
        rho_super = density_from_circuit(circ_super, init_state=in_state)
        rho_sub = density_from_circuit(circ_sub, init_state=in_state)
        for O in observables:
            d = abs(expectation(rho_super, O) - expectation(rho_sub, O))
            table[(in_name, O)] = d
            if d > sup:
                sup = d
    return {"table": table, "sup": sup}


# experiment 1: contract preservation under exact simulation

def experiment_1(epsilon: float = 1e-6) -> dict:
    """noiseless contract preservation across the three candidates."""
    qc_A = make_class_A()
    candidates = {
        "B_good (valid)":          make_class_B_good(),
        "B_sneaky (Z-preserving)": make_class_B_sneaky(),
        "B_bad (delta=0.4)":       make_class_B_bad(delta=0.4),
    }

    out = {}
    for name, circ in candidates.items():
        full = worst_case_deviation(qc_A, circ)
        only_Z = worst_case_deviation(qc_A, circ, observables=("Z",))
        out[name] = {
            "sup_over_XYZ_and_inputs": full["sup"],
            "sup_over_only_Z": only_Z["sup"],
            "passes_full_contract": full["sup"] <= epsilon,
            "passes_Z_only_contract": only_Z["sup"] <= epsilon,
            "table": full["table"],
        }
    return out


# experiment 2: encapsulation = partial trace

def experiment_2(alphas=None) -> list:
    """encapsulation = partial trace.

    take the post-CNOT state |Psi_AB> = alpha|00> + beta|11>, trace out
    qubit A, check that the reduced state on B is the diagonal matrix
    diag(|alpha|^2, |beta|^2). if it matches, the partial trace really
    does discard correlation info and only marginal probabilities
    survive - that is exactly what encapsulation means in OOP.
    """
    if alphas is None:
        alphas = [1.0, np.sqrt(0.9), np.sqrt(0.5), np.sqrt(0.1), 0.0]

    rows = []
    for alpha in alphas:
        beta = float(np.sqrt(1 - alpha**2))
        theta = 2 * np.arccos(alpha)
        qc = QuantumCircuit(2)
        qc.ry(theta, 0)
        qc.cx(0, 1)

        full_rho = density_from_circuit(qc)
        rho_B = partial_trace(full_rho, [0])
        analytical = np.diag([alpha**2, beta**2]).astype(complex)
        diff = float(np.linalg.norm(rho_B.data - analytical))

        rows.append({
            "alpha": float(alpha),
            "beta": beta,
            "rho_B_diag": (
                float(np.real(rho_B.data[0, 0])),
                float(np.real(rho_B.data[1, 1])),
            ),
            "frob_diff": diff,
        })
    return rows


# experiment 3: noise-aware epsilon calibration

def experiment_3(
    noise_levels=None,
    shots: int = 4096,
    trials: int = 20,
    seed: int = 12345,
) -> list:
    """epsilon calibration under synthetic depolarising noise.

    sweep p in [0, 0.1], for each p run B_good through a noisy AerSim,
    measure each Pauli, take the worst deviation across observables.
    repeat for `trials` independent runs and report mean/std/p95.
    """
    if noise_levels is None:
        noise_levels = [0.0, 0.001, 0.002, 0.005, 0.01,
                        0.02, 0.03, 0.05, 0.07, 0.10]

    rng = np.random.default_rng(seed)

    rho_A = density_from_circuit(make_class_A())
    ref = {O: expectation(rho_A, O) for O in ("X", "Y", "Z")}

    def estimate_pauli(state_prep_circ: QuantumCircuit,
                       basis: str, sim, shots, seed_val) -> float:
        qc = state_prep_circ.copy()
        if basis == "X":
            qc.h(0)
        elif basis == "Y":
            qc.sdg(0)
            qc.h(0)
        qc.measure_all()
        tqc = transpile(qc, sim)
        job = sim.run(tqc, shots=shots, seed_simulator=seed_val)
        counts = job.result().get_counts()
        n0 = counts.get("0", 0)
        n1 = counts.get("1", 0)
        total = n0 + n1
        return (n0 - n1) / total if total > 0 else 0.0

    results = []
    for p in noise_levels:
        nm = NoiseModel()
        if p > 0:
            err = depolarizing_error(p, 1)
            nm.add_all_qubit_quantum_error(err, ["rx", "ry", "rz", "u", "u3"])
        sim = AerSimulator(noise_model=nm if p > 0 else None)

        sup_devs = []
        for _ in range(trials):
            sup = 0.0
            for O in ("X", "Y", "Z"):
                est = estimate_pauli(
                    make_class_B_good(), O, sim, shots,
                    int(rng.integers(0, 2**31 - 1)),
                )
                d = abs(est - ref[O])
                if d > sup:
                    sup = d
            sup_devs.append(sup)

        arr = np.array(sup_devs)
        results.append({
            "p": p,
            "sup_dev_mean": float(arr.mean()),
            "sup_dev_std": float(arr.std(ddof=1)),
            "sup_dev_p95": float(np.quantile(arr, 0.95)),
        })
    return results


# experiment 4: delta-sweep for B_bad

def experiment_4(deltas=None) -> list:
    """delta-sweep for B_bad. how the worst-case deviation grows
    with the over-rotation angle delta."""
    if deltas is None:
        deltas = [0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8]

    qc_A = make_class_A()
    rows = []
    for d in deltas:
        circ_b = make_class_B_bad(d)
        info = worst_case_deviation(qc_A, circ_b)
        rows.append({"delta": d, "sup_dev": info["sup"]})
    return rows


# plotting

def plot_experiment_1(res: dict, path: str):
    names = list(res.keys())
    sup_full = [max(res[n]["sup_over_XYZ_and_inputs"], 1e-13) for n in names]
    sup_zonly = [max(res[n]["sup_over_only_Z"], 1e-13) for n in names]

    x = np.arange(len(names))
    w = 0.36
    fig, ax = plt.subplots(figsize=(8, 4.6))
    b1 = ax.bar(x - w/2, sup_full, w,
                label=r"contract $\mathcal{O}_A=\{X,Y,Z\}$",
                color="#3b6fb6", edgecolor="black")
    b2 = ax.bar(x + w/2, sup_zonly, w,
                label=r"weak contract $\mathcal{O}_A=\{Z\}$",
                color="#c66020", edgecolor="black")

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=9)
    ax.set_ylabel(r"worst-case deviation")
    ax.set_yscale("log")
    ax.legend(loc="lower right")
    for bars, vals in zip((b1, b2), (sup_full, sup_zonly)):
        for bar, v in zip(bars, vals):
            txt = "$<10^{-12}$" if v < 1e-12 else f"{v:.2g}"
            ax.text(bar.get_x() + bar.get_width()/2,
                    v * 2.0,
                    txt, ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_experiment_3(res: list, path: str):
    ps = np.array([r["p"] for r in res])
    means = np.array([r["sup_dev_mean"] for r in res])
    stds = np.array([r["sup_dev_std"] for r in res])
    p95s = np.array([r["sup_dev_p95"] for r in res])

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.errorbar(ps, means, yerr=stds, fmt="o-",
                color="#3b6fb6", ecolor="#3b6fb6", capsize=3,
                linewidth=1.6, markersize=6,
                label=r"mean $\pm 1\sigma$")
    ax.plot(ps, p95s, "s--", color="#c66020",
            linewidth=1.2, markersize=5,
            label=r"95th percentile (suggested $\varepsilon$ floor)")
    ax.set_xlabel("depolarizing noise probability $p$")
    ax.set_ylabel(r"$\sup_{O\in\{X,Y,Z\}}|\langle O\rangle_{B_{\rm good},\,p}"
                  r"-\langle O\rangle_A|$")
    ax.set_title("Noise-aware calibration of "
                 r"$\varepsilon$ (20 trials, 4096 shots)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_experiment_4(res: list, path: str):
    deltas = np.array([r["delta"] for r in res])
    devs = np.array([r["sup_dev"] for r in res])

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.plot(deltas, devs, "o-", color="#2a8c4a",
            linewidth=1.6, markersize=6)
    ax.set_xlabel(r"perturbation $\delta$ in $B_{\rm bad}=R_y(2\pi/5+\delta)$")
    ax.set_ylabel(r"$\sup_{\rho,\,O}|\langle O\rangle_{B_{\rm bad}}"
                  r"-\langle O\rangle_A|$")
    ax.set_title("Deviation grows with logical perturbation")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_combined_calibration(e3: list, e4: list, path: str):
    """noise floor and logic deviation on the same axes.
    epsilon must sit above the noise floor at the operating noise level
    but below the smallest logical deviation we want to detect.
    """
    ps = np.array([r["p"] for r in e3])
    p95s = np.array([r["sup_dev_p95"] for r in e3])
    deltas = np.array([r["delta"] for r in e4])
    devs = np.array([r["sup_dev"] for r in e4])

    fig, ax1 = plt.subplots(figsize=(7.6, 4.4))
    color1 = "#3b6fb6"
    ax1.plot(ps, p95s, "s--", color=color1, linewidth=1.4,
             markersize=5, label="honest noise floor (95th pct)")
    ax1.set_xlabel(r"depolarizing noise probability $p$",
                   color=color1)
    ax1.set_ylabel(r"worst-case observable deviation",
                   fontsize=10)
    ax1.tick_params(axis="x", labelcolor=color1)
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twiny()
    color2 = "#2a8c4a"
    ax2.plot(deltas, devs, "o-", color=color2, linewidth=1.4,
             markersize=5, label=r"deviation from over-rotation $\delta$")
    ax2.set_xlabel(r"logical perturbation $\delta$",
                   color=color2)
    ax2.tick_params(axis="x", labelcolor=color2)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


# main

def section(title: str):
    print()
    print(f">>> {title}")
    print("-" * 58)


if __name__ == "__main__":
    section("A. Contract preservation across candidate substitutions")
    e1 = experiment_1(epsilon=1e-6)
    print(f"  {'candidate':<26} {'sup XYZ':>12} {'sup Z-only':>12}  "
          f"{'full?':>7}  {'Z-only?':>8}")
    for name, info in e1.items():
        print(f"  {name:<26} {info['sup_over_XYZ_and_inputs']:>12.3e} "
              f"{info['sup_over_only_Z']:>12.3e}  "
              f"{str(info['passes_full_contract']):>7}  "
              f"{str(info['passes_Z_only_contract']):>8}")
    plot_experiment_1(e1, "exp1_subtypes.png")

    print("\n  Key observation: B_sneaky passes the {Z}-only contract but "
          "fails {X,Y,Z}.\n  This shows the importance of choosing a "
          "tomographically rich observable family.")

    section("B. Partial-trace sanity check (not reported in the paper)")
    e2 = experiment_2()
    print(f"  {'alpha':>8} {'|alpha|^2':>10} {'|beta|^2':>10} "
          f"{'rho_B[0,0]':>12} {'rho_B[1,1]':>12} {'frob.diff':>12}")
    for r in e2:
        print(f"  {r['alpha']:>8.4f} {r['alpha']**2:>10.4f} "
              f"{r['beta']**2:>10.4f} "
              f"{r['rho_B_diag'][0]:>12.6f} {r['rho_B_diag'][1]:>12.6f} "
              f"{r['frob_diff']:>12.3e}")
    print(f"\n  Max Frobenius deviation from analytical: "
          f"{max(r['frob_diff'] for r in e2):.3e}")

    section("C. Noise-aware tolerance calibration")
    e3 = experiment_3()
    print(f"  {'p':>8} {'sup-dev mean':>14} {'sup-dev std':>14} "
          f"{'sup-dev p95':>14}")
    for r in e3:
        print(f"  {r['p']:>8.4f} {r['sup_dev_mean']:>14.3e} "
              f"{r['sup_dev_std']:>14.3e} {r['sup_dev_p95']:>14.3e}")
    plot_experiment_3(e3, "exp3_noise_calibration.png")

    section("D. Over-rotation sweep")
    e4 = experiment_4()
    print(f"  {'delta':>8} {'sup-dev':>14}")
    for r in e4:
        print(f"  {r['delta']:>8.3f} {r['sup_dev']:>14.4f}")
    plot_experiment_4(e4, "exp4_delta_sweep.png")
    plot_combined_calibration(e3, e4, "exp_combined_calibration.png")

    section("Plots written:")
    print("  exp1_subtypes.png")
    print("  exp3_noise_calibration.png")
    print("  exp4_delta_sweep.png")
    print("  exp_combined_calibration.png")
