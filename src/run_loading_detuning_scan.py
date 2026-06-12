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
REPORT_DIR = ROOT / "quantum_cooling_report_20260610_151021"
RESULT_ROOT = ROOT / "results"

PARAMS = {
    "sideband_ratio": 0.06,
    "eom_offset_kHz": 4.75,
    "cooling_total_intensity_mW_cm2": 4.0,
}

DETUNING_POINTS_MHZ = np.array([20, 25, 30, 35, 38, 40, 42, 45, 50, 55, 60, 65], dtype=float)
N_FOCK = 6
TAU_COOL_REF_S = 0.5082392430187784e-3
LOADING_T_END_S = 2.0


def load_model():
    spec = importlib.util.spec_from_file_location("d1_model", MODEL_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_input(model, blue_detuning_mhz):
    return model.replace(
        model.UserInputs(),
        sideband_ratio=PARAMS["sideband_ratio"],
        eom_offset_kHz=PARAMS["eom_offset_kHz"],
        blue_detuning_MHz=float(blue_detuning_mhz),
        cooling_total_intensity_mW_cm2=PARAMS["cooling_total_intensity_mW_cm2"],
        loading_t_end_s=LOADING_T_END_S,
        include_opposite_eom_sideband=False,
        include_off_resonant_cross_coupling=False,
    )


def write_rows(path, rows, fieldnames):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    model = load_model()
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = RESULT_ROOT / f"loading_detuning_scan_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=False)

    report_data = REPORT_DIR / "data"
    report_figures = REPORT_DIR / "figures"
    report_src = REPORT_DIR / "src"
    report_data.mkdir(parents=True, exist_ok=True)
    report_figures.mkdir(parents=True, exist_ok=True)
    report_src.mkdir(parents=True, exist_ok=True)

    rows = []
    curves = {}
    print(f"output_dir={out_dir}", flush=True)
    print(f"detuning_points_MHz={DETUNING_POINTS_MHZ.tolist()}", flush=True)

    for det_mhz in DETUNING_POINTS_MHZ:
        inp = make_input(model, det_mhz)
        dp = model.build_derived_params(inp)

        t0 = time.perf_counter()
        steady = model.steady_state_n_for_ratio(
            inp,
            ratio=PARAMS["sideband_ratio"],
            n_fock=N_FOCK,
            solver_method="direct",
            show_progress=False,
        )
        steady_runtime_s = time.perf_counter() - t0

        cool_result = {
            "n_final": float(steady["n_ss"]),
            "pe_mean": float(steady["pe_ss"]),
        }
        coll = model.collision_module(inp, dp, TAU_COOL_REF_S, cool_result=cool_result)
        t_load, probs = model.loading_module(inp, dp, coll["beta1_Hz"], coll["beta2_Hz"])
        p0, p1, p2 = probs

        row = {
            "blue_detuning_MHz": float(det_mhz),
            "sideband_ratio": PARAMS["sideband_ratio"],
            "eom_offset_kHz": PARAMS["eom_offset_kHz"],
            "cooling_total_intensity_mW_cm2": PARAMS["cooling_total_intensity_mW_cm2"],
            "n_fock": N_FOCK,
            "n_ss": float(steady["n_ss"]),
            "pe_ss": float(steady["pe_ss"]),
            "total_scattering_rate_kHz": float(steady["total_scattering_rate_kHz"]),
            "p_tail_last2": float(steady["p_tail_last2"]),
            "temperature_cold_uK": float(coll["temperature_cold_K"]) * 1e6,
            "temperature_hot_uK": float(coll["temperature_hot_K"]) * 1e6,
            "e_single_MHz": float(coll["e_single_Hz"]) / 1e6,
            "gamma_enc_Hz": float(coll["gamma_enc_Hz"]),
            "gamma_scatt_hot_Hz": float(coll["gamma_scatt_hot_Hz"]),
            "gamma_evap_Hz": float(coll["gamma_evap_Hz"]),
            "p_direct": float(coll["p_direct"]),
            "p_second_escape": float(coll["p_second_escape"]),
            "beta1_Hz": float(coll["beta1_Hz"]),
            "beta2_Hz": float(coll["beta2_Hz"]),
            "beta1_over_R": float(coll["beta1_Hz"]) / max(float(dp.mot_loading_rate_Hz), 1e-12),
            "beta2_over_R": float(coll["beta2_Hz"]) / max(float(dp.mot_loading_rate_Hz), 1e-12),
            "P0_final": float(p0[-1]),
            "P1_final": float(p1[-1]),
            "P2_final": float(p2[-1]),
            "P1_max": float(np.max(p1)),
            "t_at_P1_max_s": float(t_load[int(np.argmax(p1))]),
            "prob_sum_error": float(np.max(np.abs(np.sum(probs, axis=0) - 1.0))),
            "steady_runtime_s": float(steady_runtime_s),
        }
        rows.append(row)
        curves[float(det_mhz)] = (t_load, probs)
        print(
            "det={det:.1f} MHz, n_ss={n:.4g}, beta1/R={b1:.3g}, "
            "beta2/R={b2:.3g}, P1(2s)={p1f:.4f}".format(
                det=det_mhz,
                n=row["n_ss"],
                b1=row["beta1_over_R"],
                b2=row["beta2_over_R"],
                p1f=row["P1_final"],
            ),
            flush=True,
        )

    fieldnames = list(rows[0].keys())
    csv_name = f"loading_detuning_scan_{stamp}.csv"
    write_rows(out_dir / csv_name, rows, fieldnames)
    write_rows(report_data / "loading_detuning_scan.csv", rows, fieldnames)

    best = max(rows, key=lambda r: r["P1_final"])
    best_det = float(best["blue_detuning_MHz"])
    best_t, best_probs = curves[best_det]
    curve_rows = []
    for i, t_s in enumerate(best_t):
        curve_rows.append(
            {
                "blue_detuning_MHz": best_det,
                "t_s": float(t_s),
                "t_ms": float(t_s * 1e3),
                "P0": float(best_probs[0, i]),
                "P1": float(best_probs[1, i]),
                "P2": float(best_probs[2, i]),
            }
        )
    write_rows(out_dir / f"loading_time_curve_best_{stamp}.csv", curve_rows, list(curve_rows[0].keys()))
    write_rows(report_data / "loading_time_curve_best.csv", curve_rows, list(curve_rows[0].keys()))

    dets = np.array([r["blue_detuning_MHz"] for r in rows])
    p1_final = np.array([r["P1_final"] for r in rows])
    beta1_over_r = np.array([r["beta1_over_R"] for r in rows])
    beta2_over_r = np.array([r["beta2_over_R"] for r in rows])
    n_ss = np.array([r["n_ss"] for r in rows])

    fig, ax1 = plt.subplots(figsize=(8.2, 5.0))
    ax1.plot(dets, p1_final, "o-", lw=2.2, color="tab:blue", label="$P_1(2\\,s)$")
    ax1.axvline(2.0 * model.build_derived_params(make_input(model, best_det)).trap_depth_Hz / 1e6, color="0.35", ls=":", lw=1.6, label="$2U/h$")
    ax1.set_xlabel("blue detuning $\\Delta$ (MHz)")
    ax1.set_ylabel("single-atom probability")
    ax1.set_ylim(0.0, min(1.0, max(0.75, 1.08 * float(np.max(p1_final)))))
    ax1.grid(alpha=0.3)
    ax2 = ax1.twinx()
    ax2.semilogy(dets, beta1_over_r, "s--", color="tab:green", lw=1.7, label="$\\beta_1/R$")
    ax2.semilogy(dets, beta2_over_r, "^--", color="tab:red", lw=1.7, label="$\\beta_2/R$")
    ax2.set_ylabel("collision rate / loading rate")
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / f"loading_detuning_scan_{stamp}.png", dpi=220)
    fig.savefig(report_figures / "loading_detuning_scan.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    ax.plot(dets, n_ss, "o-", color="tab:purple", lw=2.0, label="$n_{ss}$")
    ax.set_xlabel("blue detuning $\\Delta$ (MHz)")
    ax.set_ylabel("steady-state phonon number")
    ax.grid(alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_dir / f"loading_scan_cooling_context_{stamp}.png", dpi=220)
    fig.savefig(report_figures / "loading_scan_cooling_context.png", dpi=220)
    plt.close(fig)

    selected = [20.0, 35.0, best_det, 45.0, 60.0]
    selected = []
    for candidate in [20.0, 35.0, best_det, 45.0, 60.0]:
        if candidate in curves and candidate not in selected:
            selected.append(candidate)
    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    for det in selected:
        t, probs = curves[det]
        ax.plot(t, probs[1], lw=2.0, label=f"{det:g} MHz")
    ax.set_xlabel("loading time (s)")
    ax.set_ylabel("$P_1(t)$")
    ax.set_ylim(0.0, min(1.0, max(0.75, 1.08 * max(np.max(curves[d][1][1]) for d in selected))))
    ax.grid(alpha=0.3)
    ax.legend(title="$\\Delta$")
    fig.tight_layout()
    fig.savefig(out_dir / f"loading_time_curves_{stamp}.png", dpi=220)
    fig.savefig(report_figures / "loading_time_curves.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    ax.plot(best_t, best_probs[0], lw=2.0, label="$P_0$")
    ax.plot(best_t, best_probs[1], lw=2.0, label="$P_1$")
    ax.plot(best_t, best_probs[2], lw=2.0, label="$P_2$")
    ax.set_xlabel("loading time (s)")
    ax.set_ylabel("probability")
    ax.set_ylim(0.0, 1.0)
    ax.grid(alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_dir / f"loading_best_state_probabilities_{stamp}.png", dpi=220)
    fig.savefig(report_figures / "loading_best_state_probabilities.png", dpi=220)
    plt.close(fig)

    (report_src / "run_loading_detuning_scan.py").write_text(Path(__file__).read_text(encoding="utf-8"), encoding="utf-8")

    print("best_loading_detuning_MHz={:.6g}".format(best_det), flush=True)
    print("best_P1_final={:.8f}".format(float(best["P1_final"])), flush=True)
    print(f"report_data={report_data}", flush=True)
    print(f"report_figures={report_figures}", flush=True)


if __name__ == "__main__":
    main()
