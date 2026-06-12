import argparse
import csv
import importlib.util
import math
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "4f_test_d1_fixed.py"
RESULT_ROOT = ROOT / "results"

FINITE_TIMES_S = [0.2e-3, 0.5e-3, 1.0e-3, 2.0e-3]
DEFAULT_TAU_SCENARIOS_MS = [0.2, 0.5, 1.0, 2.0]
DEFAULT_BLUE_VALUES_MHZ = [15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0, 60.0, 70.0, 80.0, 100.0]
DEFAULT_INTENSITY_VALUES = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0]


def load_model():
    spec = importlib.util.spec_from_file_location("d1_model", MODEL_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_float_list(text):
    values = []
    for part in str(text).split(","):
        part = part.strip()
        if part:
            values.append(float(part))
    if not values:
        raise ValueError("empty float list")
    return values


def safe_tag(value):
    text = f"{float(value):.6g}".replace("-", "m").replace(".", "p")
    return text


def row_key(row):
    return (
        round(float(row["sideband_ratio"]), 8),
        round(float(row["eom_offset_kHz"]), 8),
        round(float(row["blue_detuning_MHz"]), 8),
        round(float(row["cooling_total_intensity_mW_cm2"]), 8),
    )


def make_inputs(model, ratio, eom_offset_kHz, blue_detuning_MHz, intensity_mW_cm2):
    return model.replace(
        model.UserInputs(),
        sideband_ratio=float(ratio),
        eom_offset_kHz=float(eom_offset_kHz),
        blue_detuning_MHz=float(blue_detuning_MHz),
        cooling_total_intensity_mW_cm2=float(intensity_mW_cm2),
        include_opposite_eom_sideband=False,
        include_off_resonant_cross_coupling=False,
    )


def finite_metrics(n0, n_ss, tau_s, scattering_rate_hz, times_s=FINITE_TIMES_S):
    metrics = {}
    tau_s = max(float(tau_s), 1e-12)
    scattering_rate_hz = max(float(scattering_rate_hz), 0.0)
    for t_s in times_s:
        n_t = n_ss + (n0 - n_ss) * math.exp(-float(t_s) / tau_s)
        removed = n0 - n_t
        n_sc = scattering_rate_hz * float(t_s)
        if removed > 0 and n_sc > 0:
            chi = n_sc / removed
            eta = removed / n_sc
        else:
            chi = math.inf
            eta = 0.0
        suffix = f"{float(t_s) * 1e3:.1f}ms".replace(".", "p")
        metrics[f"n_at_{suffix}"] = n_t
        metrics[f"Nsc_at_{suffix}"] = n_sc
        metrics[f"chi_photons_per_quantum_{suffix}"] = chi
        metrics[f"eta_quantum_per_photon_{suffix}"] = eta
    return metrics


def steady_row(model, params, n_fock, solver_method):
    inp = make_inputs(
        model,
        params["sideband_ratio"],
        params["eom_offset_kHz"],
        params["blue_detuning_MHz"],
        params["cooling_total_intensity_mW_cm2"],
    )
    start = time.perf_counter()
    result = model.steady_state_n_for_ratio(
        inp,
        ratio=float(params["sideband_ratio"]),
        n_fock=int(n_fock),
        solver_method=solver_method,
        show_progress=False,
    )
    runtime_s = time.perf_counter() - start
    n0 = float(inp.initial_nbar)
    scattering_rate_hz = float(result["total_scattering_rate_kHz"]) * 1e3
    asymptotic_removed = max(n0 - float(result["n_ss"]), 1e-12)
    return {
        "scan_name": params["scan_name"],
        "sideband_ratio": float(params["sideband_ratio"]),
        "sideband_fraction_total": float(params["sideband_ratio"]) / (1.0 + float(params["sideband_ratio"])),
        "eom_offset_kHz": float(params["eom_offset_kHz"]),
        "blue_detuning_MHz": float(params["blue_detuning_MHz"]),
        "cooling_total_intensity_mW_cm2": float(params["cooling_total_intensity_mW_cm2"]),
        "n_fock": int(n_fock),
        "n_ss": float(result["n_ss"]),
        "pe_ss": float(result["pe_ss"]),
        "scattering_rate_kHz": float(result.get("scattering_rate_kHz", result["pe_scattering_rate_kHz"])),
        "total_scattering_rate_kHz": float(result["total_scattering_rate_kHz"]),
        "scattering_rate_hz": scattering_rate_hz,
        "scatter_hz_per_asymptotic_quantum": scattering_rate_hz / asymptotic_removed,
        "p_top": float(result["p_top"]),
        "p_tail_last2": float(result["p_tail_last2"]),
        "delta_phi_pol_pi": float(result["delta_phi_pol_pi"]),
        "solver_used": result["solver_used"],
        "runtime_s": runtime_s,
    }


def write_row(writer, file_handle, row):
    writer.writerow(row)
    file_handle.flush()


def choose_best_by(rows, key_fn, limit):
    selected = []
    seen = set()
    for row in sorted(rows, key=key_fn):
        key = row_key(row)
        if key in seen:
            continue
        selected.append(row)
        seen.add(key)
        if len(selected) >= limit:
            break
    return selected


def select_steady_candidates(rows, max_candidates):
    selected = []
    seen = set()

    def add_many(items):
        for item in items:
            key = row_key(item)
            if key not in seen:
                selected.append(item)
                seen.add(key)

    add_many(choose_best_by(rows, lambda r: r["n_ss"], 8))
    add_many(choose_best_by(rows, lambda r: r["scatter_hz_per_asymptotic_quantum"], 8))
    literature_band = [r for r in rows if 0.07 <= r["sideband_ratio"] <= 0.13]
    add_many(choose_best_by(literature_band, lambda r: r["scatter_hz_per_asymptotic_quantum"], 5))
    high_ratio_band = [r for r in rows if r["sideband_ratio"] >= 0.5]
    add_many(choose_best_by(high_ratio_band, lambda r: r["n_ss"], 5))
    default_like = [
        r for r in rows
        if abs(r["sideband_ratio"] - 0.1) < 1e-9
        or abs(r["sideband_ratio"] - 0.6) < 1e-9
        or abs(r["sideband_ratio"] - 0.08) < 1e-9
    ]
    add_many(choose_best_by(default_like, lambda r: (abs(r["eom_offset_kHz"] - 8.5), r["n_ss"]), 6))
    return selected[:max_candidates]


def select_high_cutoff_candidates(dynamic_rows, max_candidates):
    selected = []
    seen = set()

    def add_many(items):
        for item in items:
            key = row_key(item)
            if key not in seen:
                selected.append(item)
                seen.add(key)

    chi_key = "chi_photons_per_quantum_1p0ms"
    add_many(choose_best_by(dynamic_rows, lambda r: r.get(chi_key, math.inf), 8))
    add_many(choose_best_by(dynamic_rows, lambda r: r["n_ss_main"], 5))
    add_many(choose_best_by(dynamic_rows, lambda r: r["total_scattering_rate_kHz"], 5))
    literature = [r for r in dynamic_rows if 0.07 <= r["sideband_ratio"] <= 0.13]
    add_many(choose_best_by(literature, lambda r: r.get(chi_key, math.inf), 4))
    high_ratio = [r for r in dynamic_rows if r["sideband_ratio"] >= 0.5]
    add_many(choose_best_by(high_ratio, lambda r: r.get(chi_key, math.inf), 4))
    return selected[:max_candidates]


def build_scan_plan(smoke=False):
    if smoke:
        return [
            {
                "scan_name": "sideband_ratio",
                "sideband_ratio": ratio,
                "eom_offset_kHz": 8.5,
                "blue_detuning_MHz": 40.0,
                "cooling_total_intensity_mW_cm2": 4.0,
            }
            for ratio in [0.08, 0.1, 0.6]
        ]

    sideband_values = [round(float(x), 2) for x in np.arange(0.03, 0.301, 0.01)]
    sideband_values += [0.35, 0.40, 0.50, 0.60, 0.75, 0.90]
    eom_values = [float(x) for x in np.arange(-20.0, 30.01, 2.0)]

    plan = []
    for ratio in sideband_values:
        plan.append(
            {
                "scan_name": "sideband_ratio",
                "sideband_ratio": ratio,
                "eom_offset_kHz": 8.5,
                "blue_detuning_MHz": 40.0,
                "cooling_total_intensity_mW_cm2": 4.0,
            }
        )
    for ratio in [0.1, 0.6]:
        for offset in eom_values:
            plan.append(
                {
                    "scan_name": f"eom_offset_ratio_{safe_tag(ratio)}",
                    "sideband_ratio": ratio,
                    "eom_offset_kHz": offset,
                    "blue_detuning_MHz": 40.0,
                    "cooling_total_intensity_mW_cm2": 4.0,
                }
            )
    return plan


def add_followup_scans(rows, smoke=False):
    if smoke:
        return []

    followups = []
    best_by_ratio = {}
    for ratio in [0.1, 0.6]:
        candidates = [r for r in rows if abs(r["sideband_ratio"] - ratio) < 1e-9]
        if candidates:
            best_by_ratio[ratio] = min(candidates, key=lambda r: r["n_ss"])

    for ratio, baseline in best_by_ratio.items():
        for blue in DEFAULT_BLUE_VALUES_MHZ:
            followups.append(
                {
                    "scan_name": f"blue_detuning_ratio_{safe_tag(ratio)}",
                    "sideband_ratio": ratio,
                    "eom_offset_kHz": baseline["eom_offset_kHz"],
                    "blue_detuning_MHz": blue,
                    "cooling_total_intensity_mW_cm2": 4.0,
                }
            )

    blue_best_by_ratio = {}
    for ratio in [0.1, 0.6]:
        candidates = [
            r for r in rows
            if abs(r["sideband_ratio"] - ratio) < 1e-9 and r["scan_name"].startswith("blue_detuning")
        ]
        if candidates:
            blue_best_by_ratio[ratio] = min(candidates, key=lambda r: r["scatter_hz_per_asymptotic_quantum"])
        elif ratio in best_by_ratio:
            blue_best_by_ratio[ratio] = best_by_ratio[ratio]

    for ratio, baseline in blue_best_by_ratio.items():
        for intensity in DEFAULT_INTENSITY_VALUES:
            followups.append(
                {
                    "scan_name": f"intensity_ratio_{safe_tag(ratio)}",
                    "sideband_ratio": ratio,
                    "eom_offset_kHz": baseline["eom_offset_kHz"],
                    "blue_detuning_MHz": baseline["blue_detuning_MHz"],
                    "cooling_total_intensity_mW_cm2": intensity,
                }
            )
    return followups


def best_eom_baselines(rows):
    best_by_ratio = {}
    for ratio in [0.1, 0.6]:
        candidates = [r for r in rows if abs(r["sideband_ratio"] - ratio) < 1e-9]
        if candidates:
            best_by_ratio[ratio] = min(candidates, key=lambda r: r["n_ss"])
    return best_by_ratio


def build_blue_followups(rows, smoke=False):
    if smoke:
        return []
    followups = []
    for ratio, baseline in best_eom_baselines(rows).items():
        for blue in DEFAULT_BLUE_VALUES_MHZ:
            followups.append(
                {
                    "scan_name": f"blue_detuning_ratio_{safe_tag(ratio)}",
                    "sideband_ratio": ratio,
                    "eom_offset_kHz": baseline["eom_offset_kHz"],
                    "blue_detuning_MHz": blue,
                    "cooling_total_intensity_mW_cm2": 4.0,
                }
            )
    return followups


def build_intensity_followups(rows, smoke=False):
    if smoke:
        return []
    followups = []
    for ratio in [0.1, 0.6]:
        candidates = [
            r for r in rows
            if abs(r["sideband_ratio"] - ratio) < 1e-9 and r["scan_name"].startswith("blue_detuning")
        ]
        if candidates:
            baseline = min(candidates, key=lambda r: r["scatter_hz_per_asymptotic_quantum"])
        else:
            baseline = best_eom_baselines(rows).get(ratio)
        if baseline is None:
            continue
        for intensity in DEFAULT_INTENSITY_VALUES:
            followups.append(
                {
                    "scan_name": f"intensity_ratio_{safe_tag(ratio)}",
                    "sideband_ratio": ratio,
                    "eom_offset_kHz": baseline["eom_offset_kHz"],
                    "blue_detuning_MHz": baseline["blue_detuning_MHz"],
                    "cooling_total_intensity_mW_cm2": intensity,
                }
            )
    return followups


def plot_steady(rows, out_dir, stamp):
    def plot_scan(scan_name, x_key, y_key, filename, xlabel, ylabel):
        subset = sorted([r for r in rows if r["scan_name"] == scan_name], key=lambda r: r[x_key])
        if not subset:
            return
        fig, ax = plt.subplots(figsize=(8.0, 5.2))
        ax.plot([r[x_key] for r in subset], [r[y_key] for r in subset], "o-", lw=2)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.35)
        fig.tight_layout()
        fig.savefig(out_dir / filename, dpi=200)
        plt.close(fig)

    plot_scan("sideband_ratio", "sideband_ratio", "n_ss", f"steady_sideband_ratio_n_ss_{stamp}.png", "Is/Ic", "steady n_ss")
    plot_scan(
        "sideband_ratio",
        "sideband_ratio",
        "total_scattering_rate_kHz",
        f"steady_sideband_ratio_scattering_{stamp}.png",
        "Is/Ic",
        "scattering rate (kHz)",
    )
    for ratio in [0.1, 0.6]:
        scan_name = f"eom_offset_ratio_{safe_tag(ratio)}"
        plot_scan(scan_name, "eom_offset_kHz", "n_ss", f"steady_eom_ratio_{safe_tag(ratio)}_n_ss_{stamp}.png", "eom offset (kHz)", "steady n_ss")
        plot_scan(
            scan_name,
            "eom_offset_kHz",
            "total_scattering_rate_kHz",
            f"steady_eom_ratio_{safe_tag(ratio)}_scattering_{stamp}.png",
            "eom offset (kHz)",
            "scattering rate (kHz)",
        )
        blue_name = f"blue_detuning_ratio_{safe_tag(ratio)}"
        plot_scan(blue_name, "blue_detuning_MHz", "n_ss", f"steady_blue_ratio_{safe_tag(ratio)}_n_ss_{stamp}.png", "blue detuning (MHz)", "steady n_ss")
        intensity_name = f"intensity_ratio_{safe_tag(ratio)}"
        plot_scan(
            intensity_name,
            "cooling_total_intensity_mW_cm2",
            "n_ss",
            f"steady_intensity_ratio_{safe_tag(ratio)}_n_ss_{stamp}.png",
            "total intensity (mW/cm^2)",
            "steady n_ss",
        )

    fig, ax = plt.subplots(figsize=(7.2, 5.8))
    sc = ax.scatter(
        [r["total_scattering_rate_kHz"] for r in rows],
        [r["n_ss"] for r in rows],
        c=[r["sideband_ratio"] for r in rows],
        s=42,
        cmap="viridis",
        alpha=0.85,
    )
    ax.set_xlabel("scattering rate (kHz)")
    ax.set_ylabel("steady n_ss")
    ax.grid(alpha=0.3)
    fig.colorbar(sc, ax=ax, label="Is/Ic")
    fig.tight_layout()
    fig.savefig(out_dir / f"steady_tradeoff_scatter_n_{stamp}.png", dpi=200)
    plt.close(fig)


def plot_dynamic(rows, out_dir, stamp):
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(7.2, 5.8))
    sc = ax.scatter(
        [r["chi_photons_per_quantum_1p0ms"] for r in rows],
        [r["n_at_1p0ms"] for r in rows],
        c=[r["sideband_ratio"] for r in rows],
        s=54,
        cmap="plasma",
        alpha=0.9,
    )
    ax.set_xlabel("chi at 1 ms (photons / removed quantum)")
    ax.set_ylabel("n(t=1 ms)")
    ax.grid(alpha=0.3)
    fig.colorbar(sc, ax=ax, label="Is/Ic")
    fig.tight_layout()
    fig.savefig(out_dir / f"dynamic_efficiency_tradeoff_1ms_{stamp}.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 5.8))
    ax.scatter(
        [r["total_scattering_rate_kHz"] for r in rows],
        [r["tau_cool_ms"] for r in rows],
        c=[r["sideband_ratio"] for r in rows],
        s=54,
        cmap="viridis",
        alpha=0.9,
    )
    ax.set_xlabel("scattering rate (kHz)")
    ax.set_ylabel("tau_cool (ms)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / f"dynamic_tau_vs_scattering_{stamp}.png", dpi=200)
    plt.close(fig)


def write_tau_scenario_metrics(steady_rows, out_dir, stamp, tau_scenarios_ms):
    metric_csv = out_dir / f"finite_time_tau_scenarios_{stamp}.csv"
    fields = [
        "scan_name",
        "sideband_ratio",
        "sideband_fraction_total",
        "eom_offset_kHz",
        "blue_detuning_MHz",
        "cooling_total_intensity_mW_cm2",
        "n_fock",
        "n_ss",
        "total_scattering_rate_kHz",
        "p_tail_last2",
        "tau_model_ms",
    ]
    for t_s in FINITE_TIMES_S:
        suffix = f"{float(t_s) * 1e3:.1f}ms".replace(".", "p")
        fields += [
            f"n_at_{suffix}",
            f"Nsc_at_{suffix}",
            f"chi_photons_per_quantum_{suffix}",
            f"eta_quantum_per_photon_{suffix}",
        ]

    metric_rows = []
    with metric_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in steady_rows:
            for tau_ms in tau_scenarios_ms:
                out = {
                    "scan_name": row["scan_name"],
                    "sideband_ratio": row["sideband_ratio"],
                    "sideband_fraction_total": row["sideband_fraction_total"],
                    "eom_offset_kHz": row["eom_offset_kHz"],
                    "blue_detuning_MHz": row["blue_detuning_MHz"],
                    "cooling_total_intensity_mW_cm2": row["cooling_total_intensity_mW_cm2"],
                    "n_fock": row["n_fock"],
                    "n_ss": row["n_ss"],
                    "total_scattering_rate_kHz": row["total_scattering_rate_kHz"],
                    "p_tail_last2": row["p_tail_last2"],
                    "tau_model_ms": float(tau_ms),
                }
                out.update(
                    finite_metrics(
                        n0=1.0,
                        n_ss=float(row["n_ss"]),
                        tau_s=float(tau_ms) * 1e-3,
                        scattering_rate_hz=float(row["total_scattering_rate_kHz"]) * 1e3,
                    )
                )
                metric_rows.append(out)
                writer.writerow(out)

    for tau_ms in tau_scenarios_ms:
        subset = [r for r in metric_rows if abs(r["tau_model_ms"] - float(tau_ms)) < 1e-12]
        if not subset:
            continue
        fig, ax = plt.subplots(figsize=(7.2, 5.8))
        sc = ax.scatter(
            [r["chi_photons_per_quantum_1p0ms"] for r in subset],
            [r["n_at_1p0ms"] for r in subset],
            c=[r["sideband_ratio"] for r in subset],
            s=42,
            cmap="plasma",
            alpha=0.85,
        )
        ax.set_xlabel("chi at 1 ms (photons / removed quantum)")
        ax.set_ylabel("n(t=1 ms)")
        ax.grid(alpha=0.3)
        fig.colorbar(sc, ax=ax, label="Is/Ic")
        fig.tight_layout()
        fig.savefig(out_dir / f"finite_tau_{safe_tag(tau_ms)}ms_chi_vs_n_1ms_{stamp}.png", dpi=200)
        plt.close(fig)

    return metric_csv, metric_rows


def run_dynamic_for_candidate(model, candidate, n_fock, t_end_s, n_t, max_runtime_s, curve_dir):
    inp = make_inputs(
        model,
        candidate["sideband_ratio"],
        candidate["eom_offset_kHz"],
        candidate["blue_detuning_MHz"],
        candidate["cooling_total_intensity_mW_cm2"],
    )
    dp = model.build_derived_params(inp)
    start = time.perf_counter()
    cool = model.cooling_module(
        inp,
        dp,
        n_fock=int(n_fock),
        t_end_s=float(t_end_s),
        n_t=int(n_t),
        show_progress=True,
        robust_integrator=True,
        max_runtime_s=float(max_runtime_s),
    )
    runtime_s = time.perf_counter() - start

    tag = (
        f"ratio_{safe_tag(candidate['sideband_ratio'])}"
        f"_eom_{safe_tag(candidate['eom_offset_kHz'])}"
        f"_blue_{safe_tag(candidate['blue_detuning_MHz'])}"
        f"_I_{safe_tag(candidate['cooling_total_intensity_mW_cm2'])}"
    )
    curve_path = curve_dir / f"time_curve_{tag}.csv"
    with curve_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["t_s", "phonons", "phonons_smooth", "pe"])
        for values in zip(cool["t_s"], cool["phonons"], cool["phonons_smooth"], cool["pe"]):
            writer.writerow([float(x) for x in values])

    n0 = float(inp.initial_nbar)
    scattering_rate_hz = float(candidate["total_scattering_rate_kHz"]) * 1e3
    metrics = finite_metrics(n0, float(candidate["n_ss"]), float(cool["tau_cool_s"]), scattering_rate_hz)
    row = {
        "scan_name": candidate["scan_name"],
        "sideband_ratio": float(candidate["sideband_ratio"]),
        "sideband_fraction_total": float(candidate["sideband_fraction_total"]),
        "eom_offset_kHz": float(candidate["eom_offset_kHz"]),
        "blue_detuning_MHz": float(candidate["blue_detuning_MHz"]),
        "cooling_total_intensity_mW_cm2": float(candidate["cooling_total_intensity_mW_cm2"]),
        "n_fock_main": int(candidate["n_fock"]),
        "n_fock_dynamics": int(n_fock),
        "n_ss_main": float(candidate["n_ss"]),
        "n_final_mesolve": float(cool["n_final"]),
        "tau_cool_ms": float(cool["tau_cool_s"]) * 1e3,
        "pe_mean_mesolve": float(cool["pe_mean"]),
        "total_scattering_rate_kHz": float(candidate["total_scattering_rate_kHz"]),
        "p_tail_last2_main": float(candidate["p_tail_last2"]),
        "dynamic_runtime_s": runtime_s,
        "curve_csv": str(curve_path),
    }
    row.update(metrics)
    return row


def run_high_cutoff(model, candidate, n_fock, solver_method):
    params = {
        "scan_name": candidate["scan_name"],
        "sideband_ratio": candidate["sideband_ratio"],
        "eom_offset_kHz": candidate["eom_offset_kHz"],
        "blue_detuning_MHz": candidate["blue_detuning_MHz"],
        "cooling_total_intensity_mW_cm2": candidate["cooling_total_intensity_mW_cm2"],
    }
    row = steady_row(model, params, n_fock=n_fock, solver_method=solver_method)
    row["tau_cool_ms_from_dynamic"] = candidate.get("tau_cool_ms", math.nan)
    if not math.isnan(row["tau_cool_ms_from_dynamic"]):
        row.update(
            finite_metrics(
                n0=1.0,
                n_ss=float(row["n_ss"]),
                tau_s=float(row["tau_cool_ms_from_dynamic"]) * 1e-3,
                scattering_rate_hz=float(row["total_scattering_rate_kHz"]) * 1e3,
            )
        )
    return row


def main():
    parser = argparse.ArgumentParser(description="Layered cooling efficiency scan for the D1 model.")
    parser.add_argument("--smoke", action="store_true", help="run a small end-to-end test")
    parser.add_argument("--main-n-fock", type=int, default=6)
    parser.add_argument("--dynamic-n-fock", type=int, default=5)
    parser.add_argument("--high-n-fock", type=int, default=9)
    parser.add_argument("--solver", default="direct")
    parser.add_argument("--max-dynamic-candidates", type=int, default=24)
    parser.add_argument("--max-high-candidates", type=int, default=16)
    parser.add_argument("--dynamic-t-end-ms", type=float, default=2.0)
    parser.add_argument("--dynamic-n-t", type=int, default=80)
    parser.add_argument("--dynamic-max-runtime-s", type=float, default=300.0)
    parser.add_argument(
        "--tau-scenarios-ms",
        default=",".join(str(x) for x in DEFAULT_TAU_SCENARIOS_MS),
        help="comma-separated tau_cool scenarios for finite-time efficiency metrics",
    )
    parser.add_argument("--skip-dynamics", action="store_true")
    parser.add_argument("--skip-high-cutoff", action="store_true")
    parser.add_argument("--output-dir", default=str(RESULT_ROOT))
    args = parser.parse_args()

    if args.smoke:
        args.max_dynamic_candidates = min(args.max_dynamic_candidates, 2)
        args.max_high_candidates = min(args.max_high_candidates, 2)
        args.dynamic_n_t = min(args.dynamic_n_t, 20)
        args.dynamic_t_end_ms = min(args.dynamic_t_end_ms, 0.4)
        args.dynamic_max_runtime_s = min(args.dynamic_max_runtime_s, 90.0)

    tau_scenarios_ms = parse_float_list(args.tau_scenarios_ms)
    model = load_model()
    stamp = time.strftime("%Y%m%d_%H%M%S")
    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    out_dir = root / f"cooling_efficiency_scan_{stamp}"
    if args.smoke:
        out_dir = root / f"cooling_efficiency_scan_smoke_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=False)
    curve_dir = out_dir / "time_curves"
    curve_dir.mkdir(exist_ok=True)

    log_path = out_dir / f"run_log_{stamp}.txt"
    steady_csv = out_dir / f"steady_scan_main_{stamp}.csv"
    dynamic_csv = out_dir / f"dynamic_candidates_{stamp}.csv"
    high_csv = out_dir / f"high_cutoff_candidates_{stamp}.csv"

    def log(message):
        text = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}"
        print(text, flush=True)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(text + "\n")

    log(f"output_dir={out_dir}")
    log(f"main_n_fock={args.main_n_fock}, dynamic_n_fock={args.dynamic_n_fock}, high_n_fock={args.high_n_fock}")
    log("opposite_eom_sideband=False, off_resonant_cross_coupling=False")

    steady_fields = [
        "scan_name",
        "sideband_ratio",
        "sideband_fraction_total",
        "eom_offset_kHz",
        "blue_detuning_MHz",
        "cooling_total_intensity_mW_cm2",
        "n_fock",
        "n_ss",
        "pe_ss",
        "scattering_rate_kHz",
        "total_scattering_rate_kHz",
        "scattering_rate_hz",
        "scatter_hz_per_asymptotic_quantum",
        "p_top",
        "p_tail_last2",
        "delta_phi_pol_pi",
        "solver_used",
        "runtime_s",
    ]
    steady_rows = []
    initial_plan = build_scan_plan(smoke=args.smoke)
    with steady_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=steady_fields)
        writer.writeheader()
        log(f"steady initial points={len(initial_plan)}")
        for idx, params in enumerate(initial_plan, start=1):
            log(f"steady {idx}/{len(initial_plan)} start {params}")
            row = steady_row(model, params, n_fock=args.main_n_fock, solver_method=args.solver)
            steady_rows.append(row)
            write_row(writer, f, row)
            log(
                f"steady done n_ss={row['n_ss']:.6g}, scatter={row['total_scattering_rate_kHz']:.4g} kHz, "
                f"tail2={row['p_tail_last2']:.3e}, runtime={row['runtime_s']:.1f}s"
            )

        blue_followups = build_blue_followups(steady_rows, smoke=args.smoke)
        log(f"steady blue followup points={len(blue_followups)}")
        for idx, params in enumerate(blue_followups, start=1):
            log(f"steady blue {idx}/{len(blue_followups)} start {params}")
            row = steady_row(model, params, n_fock=args.main_n_fock, solver_method=args.solver)
            steady_rows.append(row)
            write_row(writer, f, row)
            log(
                f"steady blue done n_ss={row['n_ss']:.6g}, scatter={row['total_scattering_rate_kHz']:.4g} kHz, "
                f"tail2={row['p_tail_last2']:.3e}, runtime={row['runtime_s']:.1f}s"
            )

        intensity_followups = build_intensity_followups(steady_rows, smoke=args.smoke)
        log(f"steady intensity followup points={len(intensity_followups)}")
        for idx, params in enumerate(intensity_followups, start=1):
            log(f"steady intensity {idx}/{len(intensity_followups)} start {params}")
            row = steady_row(model, params, n_fock=args.main_n_fock, solver_method=args.solver)
            steady_rows.append(row)
            write_row(writer, f, row)
            log(
                f"steady intensity done n_ss={row['n_ss']:.6g}, scatter={row['total_scattering_rate_kHz']:.4g} kHz, "
                f"tail2={row['p_tail_last2']:.3e}, runtime={row['runtime_s']:.1f}s"
            )

    plot_steady(steady_rows, out_dir, stamp)
    metric_csv, tau_metric_rows = write_tau_scenario_metrics(steady_rows, out_dir, stamp, tau_scenarios_ms)
    log(f"finite-time tau scenario CSV={metric_csv}")

    dynamic_rows = []
    if not args.skip_dynamics:
        candidates = select_steady_candidates(steady_rows, max_candidates=args.max_dynamic_candidates)
        log(f"dynamic candidates={len(candidates)}")
        dynamic_fields = [
            "scan_name",
            "sideband_ratio",
            "sideband_fraction_total",
            "eom_offset_kHz",
            "blue_detuning_MHz",
            "cooling_total_intensity_mW_cm2",
            "n_fock_main",
            "n_fock_dynamics",
            "n_ss_main",
            "n_final_mesolve",
            "tau_cool_ms",
            "pe_mean_mesolve",
            "total_scattering_rate_kHz",
            "p_tail_last2_main",
            "dynamic_runtime_s",
            "curve_csv",
        ]
        for t_s in FINITE_TIMES_S:
            suffix = f"{float(t_s) * 1e3:.1f}ms".replace(".", "p")
            dynamic_fields += [
                f"n_at_{suffix}",
                f"Nsc_at_{suffix}",
                f"chi_photons_per_quantum_{suffix}",
                f"eta_quantum_per_photon_{suffix}",
            ]
        with dynamic_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=dynamic_fields)
            writer.writeheader()
            for idx, candidate in enumerate(candidates, start=1):
                log(
                    f"dynamic {idx}/{len(candidates)} start ratio={candidate['sideband_ratio']:.4g}, "
                    f"eom={candidate['eom_offset_kHz']:.4g} kHz, blue={candidate['blue_detuning_MHz']:.4g} MHz, "
                    f"I={candidate['cooling_total_intensity_mW_cm2']:.4g}"
                )
                row = run_dynamic_for_candidate(
                    model,
                    candidate,
                    n_fock=args.dynamic_n_fock,
                    t_end_s=args.dynamic_t_end_ms * 1e-3,
                    n_t=args.dynamic_n_t,
                    max_runtime_s=args.dynamic_max_runtime_s,
                    curve_dir=curve_dir,
                )
                dynamic_rows.append(row)
                write_row(writer, f, row)
                log(
                    f"dynamic done tau={row['tau_cool_ms']:.4g} ms, "
                    f"n1ms={row['n_at_1p0ms']:.5g}, chi1ms={row['chi_photons_per_quantum_1p0ms']:.5g}, "
                    f"runtime={row['dynamic_runtime_s']:.1f}s"
                )
        plot_dynamic(dynamic_rows, out_dir, stamp)

    if not args.skip_high_cutoff:
        source_rows = dynamic_rows if dynamic_rows else steady_rows
        candidates = (
            select_high_cutoff_candidates(source_rows, max_candidates=args.max_high_candidates)
            if dynamic_rows
            else select_steady_candidates(source_rows, max_candidates=args.max_high_candidates)
        )
        high_fields = steady_fields + ["tau_cool_ms_from_dynamic"]
        for t_s in FINITE_TIMES_S:
            suffix = f"{float(t_s) * 1e3:.1f}ms".replace(".", "p")
            high_fields += [
                f"n_at_{suffix}",
                f"Nsc_at_{suffix}",
                f"chi_photons_per_quantum_{suffix}",
                f"eta_quantum_per_photon_{suffix}",
            ]
        log(f"high cutoff candidates={len(candidates)}")
        with high_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=high_fields)
            writer.writeheader()
            for idx, candidate in enumerate(candidates, start=1):
                log(
                    f"high {idx}/{len(candidates)} start ratio={candidate['sideband_ratio']:.4g}, "
                    f"eom={candidate['eom_offset_kHz']:.4g} kHz, blue={candidate['blue_detuning_MHz']:.4g} MHz, "
                    f"I={candidate['cooling_total_intensity_mW_cm2']:.4g}"
                )
                row = run_high_cutoff(model, candidate, n_fock=args.high_n_fock, solver_method=args.solver)
                write_row(writer, f, row)
                log(
                    f"high done n_ss={row['n_ss']:.6g}, scatter={row['total_scattering_rate_kHz']:.4g} kHz, "
                    f"tail2={row['p_tail_last2']:.3e}, runtime={row['runtime_s']:.1f}s"
                )

    log(f"CSV steady={steady_csv}")
    log(f"CSV dynamic={dynamic_csv}")
    log(f"CSV high={high_csv}")
    log("done")
    print(f"OUTPUT_DIR={out_dir}")


if __name__ == "__main__":
    main()
