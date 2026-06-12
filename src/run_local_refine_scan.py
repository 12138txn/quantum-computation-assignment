import csv
import importlib.util
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "4f_test_d1_fixed.py"
RESULT_ROOT = ROOT / "results"

MAIN_N_FOCK = 6
HIGH_N_FOCK = 9
SOLVER = "direct"


def load_model():
    spec = importlib.util.spec_from_file_location("d1_model", MODEL_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def safe_tag(value):
    return f"{float(value):.6g}".replace("-", "m").replace(".", "p")


def make_params(scan_name, ratio, eom, blue, intensity):
    return {
        "scan_name": scan_name,
        "sideband_ratio": float(ratio),
        "eom_offset_kHz": float(eom),
        "blue_detuning_MHz": float(blue),
        "cooling_total_intensity_mW_cm2": float(intensity),
    }


def run_point(model, params, n_fock):
    inp = model.replace(
        model.UserInputs(),
        sideband_ratio=params["sideband_ratio"],
        eom_offset_kHz=params["eom_offset_kHz"],
        blue_detuning_MHz=params["blue_detuning_MHz"],
        cooling_total_intensity_mW_cm2=params["cooling_total_intensity_mW_cm2"],
        include_opposite_eom_sideband=False,
        include_off_resonant_cross_coupling=False,
    )
    start = time.perf_counter()
    result = model.steady_state_n_for_ratio(
        inp,
        ratio=params["sideband_ratio"],
        n_fock=n_fock,
        solver_method=SOLVER,
        show_progress=False,
    )
    runtime_s = time.perf_counter() - start
    scatter_hz = float(result["total_scattering_rate_kHz"]) * 1e3
    removed = max(float(inp.initial_nbar) - float(result["n_ss"]), 1e-12)
    return {
        **params,
        "sideband_fraction_total": params["sideband_ratio"] / (1.0 + params["sideband_ratio"]),
        "n_fock": int(n_fock),
        "n_ss": float(result["n_ss"]),
        "pe_ss": float(result["pe_ss"]),
        "total_scattering_rate_kHz": float(result["total_scattering_rate_kHz"]),
        "scatter_hz_per_asymptotic_quantum": scatter_hz / removed,
        "p_top": float(result["p_top"]),
        "p_tail_last2": float(result["p_tail_last2"]),
        "delta_phi_pol_pi": float(result["delta_phi_pol_pi"]),
        "solver_used": result["solver_used"],
        "runtime_s": runtime_s,
    }


def key(row):
    return (
        round(row["sideband_ratio"], 8),
        round(row["eom_offset_kHz"], 8),
        round(row["blue_detuning_MHz"], 8),
        round(row["cooling_total_intensity_mW_cm2"], 8),
    )


def choose_unique(rows, sort_key, limit):
    out = []
    seen = set()
    for row in sorted(rows, key=sort_key):
        k = key(row)
        if k in seen:
            continue
        out.append(row)
        seen.add(k)
        if len(out) >= limit:
            break
    return out


def plot_scan(rows, scan_name, x_key, out_dir, stamp):
    subset = sorted([r for r in rows if r["scan_name"] == scan_name], key=lambda r: r[x_key])
    if not subset:
        return
    fig, ax1 = plt.subplots(figsize=(8.0, 5.2))
    ax1.plot([r[x_key] for r in subset], [r["n_ss"] for r in subset], "o-", lw=2, color="tab:blue")
    ax1.set_xlabel(x_key)
    ax1.set_ylabel("n_ss", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.grid(alpha=0.35)
    ax2 = ax1.twinx()
    ax2.plot([r[x_key] for r in subset], [r["total_scattering_rate_kHz"] for r in subset], "s--", lw=1.8, color="tab:red")
    ax2.set_ylabel("scattering rate (kHz)", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")
    fig.tight_layout()
    fig.savefig(out_dir / f"{scan_name}_{stamp}.png", dpi=200)
    plt.close(fig)


def main():
    model = load_model()
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = RESULT_ROOT / f"local_refine_scan_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=False)
    main_csv = out_dir / f"local_refine_main_n{MAIN_N_FOCK}_{stamp}.csv"
    high_csv = out_dir / f"local_refine_high_n{HIGH_N_FOCK}_{stamp}.csv"
    log_path = out_dir / f"local_refine_{stamp}.log"

    fields = [
        "scan_name",
        "sideband_ratio",
        "sideband_fraction_total",
        "eom_offset_kHz",
        "blue_detuning_MHz",
        "cooling_total_intensity_mW_cm2",
        "n_fock",
        "n_ss",
        "pe_ss",
        "total_scattering_rate_kHz",
        "scatter_hz_per_asymptotic_quantum",
        "p_top",
        "p_tail_last2",
        "delta_phi_pol_pi",
        "solver_used",
        "runtime_s",
    ]

    def log(message):
        text = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}"
        print(text, flush=True)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(text + "\n")

    rows = []
    current = {"ratio": 0.1, "eom": 6.0, "blue": 60.0, "intensity": 4.0}
    scan_plan = []

    scan_plan.append((
        "ratio_refine_initial",
        "sideband_ratio",
        [0.06, 0.07, 0.08, 0.09, 0.10, 0.11, 0.12, 0.14, 0.16],
        lambda x: make_params("ratio_refine_initial", x, current["eom"], current["blue"], current["intensity"]),
    ))
    scan_plan.append((
        "eom_refine_initial",
        "eom_offset_kHz",
        list(np.arange(3.0, 9.01, 0.5)),
        lambda x: make_params("eom_refine_initial", current["ratio"], x, current["blue"], current["intensity"]),
    ))
    scan_plan.append((
        "blue_refine_after_eom",
        "blue_detuning_MHz",
        list(np.arange(52.0, 68.01, 2.0)),
        lambda x: make_params("blue_refine_after_eom", current["ratio"], current["eom"], x, current["intensity"]),
    ))
    scan_plan.append((
        "intensity_refine_after_blue",
        "cooling_total_intensity_mW_cm2",
        list(np.arange(2.5, 6.51, 0.5)),
        lambda x: make_params("intensity_refine_after_blue", current["ratio"], current["eom"], current["blue"], x),
    ))
    scan_plan.append((
        "eom_refine_final",
        "eom_offset_kHz",
        list(np.arange(max(0.0, current["eom"] - 2.0), current["eom"] + 2.01, 0.25)),
        lambda x: make_params("eom_refine_final", current["ratio"], x, current["blue"], current["intensity"]),
    ))

    with main_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        for scan_name, varied_key, values, maker in scan_plan:
            # The final eom scan is rebuilt after the earlier scans update current.
            if scan_name == "eom_refine_final":
                values = list(np.arange(max(0.0, current["eom"] - 2.0), current["eom"] + 2.01, 0.25))
                maker = lambda x, scan_name=scan_name: make_params(
                    scan_name,
                    current["ratio"],
                    x,
                    current["blue"],
                    current["intensity"],
                )

            log(f"scan {scan_name} start, varied={varied_key}, points={len(values)}, baseline={current}")
            scan_rows = []
            for idx, value in enumerate(values, start=1):
                params = maker(float(value))
                log(f"{scan_name} {idx}/{len(values)} start {params}")
                row = run_point(model, params, MAIN_N_FOCK)
                rows.append(row)
                scan_rows.append(row)
                writer.writerow(row)
                f.flush()
                log(
                    f"{scan_name} done n_ss={row['n_ss']:.7g}, scatter={row['total_scattering_rate_kHz']:.4g} kHz, "
                    f"tail2={row['p_tail_last2']:.3e}, runtime={row['runtime_s']:.1f}s"
                )

            best = min(scan_rows, key=lambda r: r["n_ss"])
            current = {
                "ratio": best["sideband_ratio"],
                "eom": best["eom_offset_kHz"],
                "blue": best["blue_detuning_MHz"],
                "intensity": best["cooling_total_intensity_mW_cm2"],
            }
            log(f"scan {scan_name} best_by_n={current}, n_ss={best['n_ss']:.7g}")

    for scan_name, varied_key, _, _ in scan_plan:
        plot_scan(rows, scan_name, varied_key, out_dir, stamp)

    candidates = []
    candidates.extend(choose_unique(rows, lambda r: r["n_ss"], 6))
    candidates.extend(choose_unique(rows, lambda r: r["scatter_hz_per_asymptotic_quantum"], 4))
    dedup = []
    seen = set()
    for row in candidates:
        k = key(row)
        if k not in seen:
            dedup.append(row)
            seen.add(k)
    candidates = dedup[:8]

    log(f"high cutoff candidates={len(candidates)}")
    high_rows = []
    with high_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for idx, row0 in enumerate(candidates, start=1):
            params = make_params(
                row0["scan_name"],
                row0["sideband_ratio"],
                row0["eom_offset_kHz"],
                row0["blue_detuning_MHz"],
                row0["cooling_total_intensity_mW_cm2"],
            )
            log(f"high {idx}/{len(candidates)} start {params}")
            row = run_point(model, params, HIGH_N_FOCK)
            high_rows.append(row)
            writer.writerow(row)
            f.flush()
            log(
                f"high done n_ss={row['n_ss']:.7g}, scatter={row['total_scattering_rate_kHz']:.4g} kHz, "
                f"tail2={row['p_tail_last2']:.3e}, runtime={row['runtime_s']:.1f}s"
            )

    best_high = min(high_rows, key=lambda r: r["n_ss"]) if high_rows else None
    best_eff_high = min(high_rows, key=lambda r: r["scatter_hz_per_asymptotic_quantum"]) if high_rows else None
    if best_high:
        log(f"BEST_N_HIGH={best_high}")
    if best_eff_high:
        log(f"BEST_EFF_HIGH={best_eff_high}")
    print(f"OUTPUT_DIR={out_dir}")


if __name__ == "__main__":
    main()
