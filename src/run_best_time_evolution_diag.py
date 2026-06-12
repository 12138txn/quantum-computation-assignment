import csv
import importlib.util
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "4f_test_d1_fixed.py"
RESULT_ROOT = ROOT / "results"

PARAMS = {
    "sideband_ratio": 0.06,
    "eom_offset_kHz": 4.75,
    "blue_detuning_MHz": 60.0,
    "cooling_total_intensity_mW_cm2": 4.0,
}
TIME_N_FOCK = 6
STEADY_N_FOCK = 9
T_END_S = 5.0e-3
N_T = 240


def load_model():
    spec = importlib.util.spec_from_file_location("d1_model", MODEL_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_inputs(model):
    return model.replace(
        model.UserInputs(),
        sideband_ratio=PARAMS["sideband_ratio"],
        eom_offset_kHz=PARAMS["eom_offset_kHz"],
        blue_detuning_MHz=PARAMS["blue_detuning_MHz"],
        cooling_total_intensity_mW_cm2=PARAMS["cooling_total_intensity_mW_cm2"],
        include_opposite_eom_sideband=False,
        include_off_resonant_cross_coupling=False,
    )


def fit_tau_fixed_nss(t_s, n_arr, n_ss):
    t = np.asarray(t_s, dtype=float)
    n = np.asarray(n_arr, dtype=float)
    n0 = float(n[0])

    def model(t_eval, tau):
        return n_ss + (n0 - n_ss) * np.exp(-t_eval / tau)

    mask = t >= min(2.0e-5, 0.05 * float(t[-1]))
    try:
        popt, _ = curve_fit(
            model,
            t[mask],
            n[mask],
            p0=[5.0e-4],
            bounds=([1.0e-6], [0.1]),
            maxfev=20000,
        )
        return float(popt[0]), model(t, float(popt[0]))
    except Exception:
        return float("nan"), model(t, 5.0e-4)


def main():
    model = load_model()
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = RESULT_ROOT / f"best_time_evolution_diag_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=False)

    inp = make_inputs(model)
    dp = model.build_derived_params(inp)

    print(f"output_dir={out_dir}", flush=True)
    print(f"params={PARAMS}", flush=True)

    t0 = time.perf_counter()
    steady = model.steady_state_n_for_ratio(
        inp,
        ratio=PARAMS["sideband_ratio"],
        n_fock=STEADY_N_FOCK,
        solver_method="direct",
        show_progress=False,
    )
    steady_runtime_s = time.perf_counter() - t0
    print(
        f"steady n={STEADY_N_FOCK}: n_ss={steady['n_ss']:.8g}, "
        f"scatter={steady['total_scattering_rate_kHz']:.4g} kHz, runtime={steady_runtime_s:.1f}s",
        flush=True,
    )

    t0 = time.perf_counter()
    cool = model.cooling_module(
        inp,
        dp,
        n_fock=TIME_N_FOCK,
        t_end_s=T_END_S,
        n_t=N_T,
        show_progress=False,
        robust_integrator=False,
        solver_options={"method": "diag"},
    )
    time_runtime_s = time.perf_counter() - t0
    tau_fixed_s, n_fit = fit_tau_fixed_nss(cool["t_s"], cool["phonons"], steady["n_ss"])

    curve_csv = out_dir / f"best_time_curve_n{TIME_N_FOCK}_{stamp}.csv"
    with curve_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["t_s", "t_ms", "n", "n_smooth", "n_fit_fixed_nss", "pe"])
        for t_s, n, ns, nf, pe in zip(cool["t_s"], cool["phonons"], cool["phonons_smooth"], n_fit, cool["pe"]):
            writer.writerow([float(t_s), float(t_s * 1e3), float(n), float(ns), float(nf), float(pe)])

    scatter_hz = float(steady["total_scattering_rate_kHz"]) * 1e3
    n_initial = float(cool["phonons"][0])
    n_final = float(cool["n_final"])
    removed_final = max(n_initial - n_final, 1e-12)
    nsc_final = scatter_hz * T_END_S

    summary = {
        **PARAMS,
        "time_n_fock": TIME_N_FOCK,
        "steady_n_fock": STEADY_N_FOCK,
        "t_end_ms": T_END_S * 1e3,
        "n_t": N_T,
        "n_initial_truncated": n_initial,
        "n_final_time": n_final,
        "n_ss_steady_high": float(steady["n_ss"]),
        "tau_cool_envelope_ms": float(cool["tau_cool_s"] * 1e3),
        "tau_cool_fixed_nss_ms": float(tau_fixed_s * 1e3),
        "pe_mean_time": float(cool["pe_mean"]),
        "total_scattering_rate_kHz_steady": float(steady["total_scattering_rate_kHz"]),
        "Nsc_at_t_end": nsc_final,
        "chi_photons_per_quantum_at_t_end": nsc_final / removed_final,
        "eta_quantum_per_photon_at_t_end": removed_final / nsc_final,
        "steady_runtime_s": steady_runtime_s,
        "time_runtime_s": time_runtime_s,
        "curve_csv": str(curve_csv),
    }

    summary_csv = out_dir / f"best_time_summary_{stamp}.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)

    fig, ax = plt.subplots(figsize=(8.2, 5.4))
    ax.plot(cool["t_s"] * 1e3, cool["phonons"], lw=1.4, alpha=0.55, label="full master equation")
    ax.plot(cool["t_s"] * 1e3, cool["phonons_smooth"], lw=2.0, label="smoothed envelope")
    ax.plot(cool["t_s"] * 1e3, n_fit, "--", lw=1.8, label=f"fixed-nss fit tau={tau_fixed_s*1e3:.3f} ms")
    ax.axhline(float(steady["n_ss"]), color="tab:red", ls=":", lw=1.8, label=f"steady n_ss={steady['n_ss']:.4f}")
    ax.set_xlabel("time (ms)")
    ax.set_ylabel("<n>")
    ax.grid(alpha=0.35)
    ax.legend(loc="best")
    fig.tight_layout()
    n_plot = out_dir / f"best_time_n_curve_{stamp}.png"
    fig.savefig(n_plot, dpi=220)

    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    ax.plot(cool["t_s"] * 1e3, cool["pe"], lw=1.7)
    ax.set_xlabel("time (ms)")
    ax.set_ylabel("excited-state population")
    ax.grid(alpha=0.35)
    fig.tight_layout()
    pe_plot = out_dir / f"best_time_pe_curve_{stamp}.png"
    fig.savefig(pe_plot, dpi=220)

    print(f"summary={summary}", flush=True)
    print(f"CSV={curve_csv}", flush=True)
    print(f"SUMMARY_CSV={summary_csv}", flush=True)
    print(f"N_PLOT={n_plot}", flush=True)
    print(f"PE_PLOT={pe_plot}", flush=True)


if __name__ == "__main__":
    main()
