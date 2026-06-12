from dataclasses import dataclass, replace
import argparse
import csv
from functools import lru_cache
from pathlib import Path
import threading
import time

import numpy as np
from qutip import Qobj, basis, destroy, expect, ket2dm, liouvillian, mesolve, qeye, steadystate, tensor, thermal_dm
from scipy import sparse
from scipy.sparse.linalg import lsqr
from scipy.integrate import solve_ivp
from scipy.optimize import curve_fit, root_scalar
from scipy.special import gammainc, jv
from sympy import N, Rational
from sympy.physics.wigner import wigner_3j, wigner_6j

try:
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
except Exception:
    plt = None
else:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


# ==========================================
# 全局常数 (SI)
# ==========================================
H = 6.62607015e-34
HBAR = 1.054571817e-34
KB = 1.380649e-23
C = 299792458.0
M_RB87 = 1.44316060e-25

LAMBDA_D1 = 794.9788509e-9
GAMMA_D1_HZ = 5.75e6
GAMMA_D1_RAD = 2 * np.pi * GAMMA_D1_HZ
HFS_SPLITTING_HZ = 6.834682610e9
EHFS_EXCITED_HZ = 814.5e6
D1_SATURATION_INTENSITY_W_M2 = np.pi * H * C * GAMMA_D1_RAD / (3.0 * LAMBDA_D1**3)
W_M2_PER_MW_CM2 = 10.0

I_NUCLEAR = Rational(3, 2)
J_GROUND = Rational(1, 2)
J_EXCITED = Rational(1, 2)

# GROUND_STATES = [(1, -1), (1, 0), (1, 1), (2, -2), (2, -1), (2, 0), (2, 1), (2, 2)]
# EXCITED_STATES = [(2, -2), (2, -1), (2, 0), (2, 1), (2, 2)]
# INTERNAL_STATES = GROUND_STATES + EXCITED_STATES
# STATE_TO_INDEX = {st: idx for idx, st in enumerate(INTERNAL_STATES)}

GROUND_STATES = [('g', 1, -1), ('g', 1, 0), ('g', 1, 1), ('g', 2, -2), ('g', 2, -1), ('g', 2, 0), ('g', 2, 1), ('g', 2, 2)]
EXCITED_STATES = [('e', 1, -1), ('e', 1, 0), ('e', 1, 1), ('e', 2, -2), ('e', 2, -1), ('e', 2, 0), ('e', 2, 1), ('e', 2, 2)]
INTERNAL_STATES = GROUND_STATES + EXCITED_STATES
STATE_TO_INDEX = {st: idx for idx, st in enumerate(INTERNAL_STATES)}


@lru_cache(maxsize=None)
def _raw_cgc_weight(F: int, m: int, Fp: int, mp: int, q: int) -> float:
    if mp != m + q:
        return 0.0

    if ('g', F, m) not in GROUND_STATES or ('e', Fp, mp) not in EXCITED_STATES:
        return 0.0

    # Wigner-Eckart relative hyperfine dipole matrix element for D1.
    phase_exp_val = 2 * Fp + float(J_GROUND) + float(I_NUCLEAR) + 1.0 - mp
    phase = (-1.0) ** int(round(phase_exp_val))
    w6 = float(N(wigner_6j(Fp, F, 1, J_GROUND, J_EXCITED, I_NUCLEAR)))
    w3 = float(N(wigner_3j(Fp, 1, F, -mp, q, m)))

    return phase * np.sqrt((2 * F + 1) * (2 * Fp + 1)) * w6 * w3


@lru_cache(maxsize=None)
def _decay_norm_for_excited(Fp: int, mp: int) -> float:
    total = 0.0
    for F in (1, 2):
        for m in range(-F, F + 1):
            for q in (-1, 0, 1):
                total += abs(_raw_cgc_weight(F, m, Fp, mp, q)) ** 2

    if total <= 1e-15:
        raise ValueError(f"No valid D1 decay branches for F'={Fp}, m'={mp}")
    return float(np.sqrt(total))


@lru_cache(maxsize=None)
def cgc_weight(F: int, m: int, Fp: int, mp: int, q: int) -> float:
    raw = _raw_cgc_weight(F, m, Fp, mp, q)
    if raw == 0.0:
        return 0.0
    # Normalize explicitly so sum_{Fg,m,q} |c|^2 = 1 for each excited sublevel.
    return raw / _decay_norm_for_excited(Fp, mp)


def build_sigma_q_F(q: int, Fg: int):
    n_int = len(INTERNAL_STATES)
    op = 0
    for m in range(-Fg, Fg + 1):
        mp = m + q
        for Fp in (1, 2):
            if ('e', Fp, mp) not in EXCITED_STATES:
                continue
            coeff = cgc_weight(Fg, m, Fp, mp, q)
            if abs(coeff) < 1e-14:
                continue
            i_g = STATE_TO_INDEX[('g', Fg, m)]
            i_e = STATE_TO_INDEX[('e', Fp, mp)]

            ket_g = basis(n_int, i_g)
            ket_e = basis(n_int, i_e)
            op = op + coeff * ket_g * ket_e.dag()
    return op


@lru_cache(maxsize=None)
def build_sigma_q_F_to_Fp(q: int, Fg: int, Fp: int):
    n_int = len(INTERNAL_STATES)
    op = 0
    for m in range(-Fg, Fg + 1):
        mp = m + q
        if ('e', Fp, mp) not in EXCITED_STATES:
            continue
        coeff = cgc_weight(Fg, m, Fp, mp, q)
        if abs(coeff) < 1e-14:
            continue
        i_g = STATE_TO_INDEX[('g', Fg, m)]
        i_e = STATE_TO_INDEX[('e', Fp, mp)]
        ket_g = basis(n_int, i_g)
        ket_e = basis(n_int, i_e)
        op = op + coeff * ket_g * ket_e.dag()
    return op


def build_projectors():
    n_int = len(INTERNAL_STATES)
    p_f1 = 0
    p_f2 = 0
    p_e1 = 0
    p_e2 = 0
    p_e_total = 0

    # for (F, m) in GROUND_STATES:
    #     ket = basis(n_int, STATE_TO_INDEX[(F, m)])     #codex version
    for (lvl, F, m) in GROUND_STATES:
        ket = basis(n_int, STATE_TO_INDEX[(lvl, F, m)])
        proj = ket * ket.dag()
        if F == 1:
            p_f1 = p_f1 + proj
        else:
            p_f2 = p_f2 + proj

    for (lvl, F, m) in EXCITED_STATES:
        ket = basis(n_int, STATE_TO_INDEX[(lvl, F, m)])
        proj = ket * ket.dag()
        p_e_total = p_e_total + proj
        if F == 1:
            p_e1 = p_e1 + proj
        else:
            p_e2 = p_e2 + proj

    return p_f1, p_f2, p_e1, p_e2, p_e_total


def print_progress(prefix: str, current: int, total: int):
    ratio = current / max(total, 1)
    width = 24
    filled = int(width * ratio)
    bar = "█" * filled + "·" * (width - filled)
    print(f"\r{prefix} [{bar}] {current}/{total} ({ratio*100:5.1f}%)", end="", flush=True)
    if current >= total:
        print("")


def estimate_tau_cool(t_s: np.ndarray, n_arr: np.ndarray):
    t_span = max(float(t_s[-1] - t_s[0]), 1e-6)
    if len(t_s) < 10:
        tau_quick = max(float(t_s[-1]) * 0.3, 1e-5)
        return min(tau_quick, 0.9 * t_span), n_arr

    win = max(5, (len(n_arr) // 15) | 1)
    kernel = np.ones(win) / win
    if len(n_arr) <= win:
        tau_quick = max(float(t_s[-1]) * 0.3, 1e-5)
        return min(tau_quick, 0.9 * t_span), n_arr

    n_smooth_valid = np.convolve(n_arr, kernel, mode="valid")
    left = (win - 1) // 2
    right = win // 2
    t_valid = t_s[left: len(t_s) - right]

    tail = max(6, len(n_smooth_valid) // 6)
    n_inf = float(np.mean(n_smooth_valid[-tail:]))
    y = n_smooth_valid - n_inf
    mask = y > 1e-5

    if np.count_nonzero(mask) < 6:
        tau_quick = max(float(t_s[-1]) * 0.3, 1e-5)
        n_smooth_plot = np.interp(t_s, t_valid, n_smooth_valid)
        return min(tau_quick, 0.9 * t_span), n_smooth_plot

    t_fit = t_valid[mask]
    y_fit = y[mask]

    def exp_model(t, a, tau):
        return a * np.exp(-t / tau)

    try:
        popt, _ = curve_fit(
            exp_model,
            t_fit,
            y_fit,
            p0=[float(max(y_fit[0], 1e-4)), max(float(t_s[-1]) / 4, 2e-5)],
            bounds=([1e-8, 1e-6], [1e3, 1.0]),
            maxfev=12000,
        )
        tau = float(popt[1])
    except Exception:
        n_target = n_smooth_valid[0] * np.exp(-1.0)
        idx = int(np.argmin(np.abs(n_smooth_valid - n_target)))
        tau = float(t_valid[idx])

    tau = max(tau, 1e-5)
    tau = min(tau, 0.9 * t_span)
    n_smooth_plot = np.interp(t_s, t_valid, n_smooth_valid)
    return tau, n_smooth_plot


@dataclass
class UserInputs:
    """用户可改参数（都带单位）。"""

    trap_wavelength_nm: float = 850.0          # 光镊波长 (nm)
    trap_depth_mK: float = 1                   # 光镊阱深 (mK)
    trap_waist_um: float = 1.0                 # 光镊腰斑半径 (um)
    blue_detuning_MHz: float = 40.0            # 单光子蓝失谐 Δ (MHz)
    differential_light_shift_kHz: float = 0.0  # 差分光频移 δ_trap (kHz)
    eom_offset_kHz: float = 8.5                # EOM相对6.834GHz的人为偏置(kHz)
    cooling_total_intensity_mW_cm2: float = 4.0 # 参与 Λ 冷却的载波+边带总光强 (mW/cm^2)
    sideband_ratio: float = 0.08               # 边带/载波光强比 Is/Ic
    polarization_phase_pi: float = 0.6         # sigma+ 与 sigma- 驱动的相对相位，单位 pi
    retro_mirror_distance_m: float = 0.15      # 原子到 retro 反射镜的距离 (m)
    sideband_phase_sign: float = 1.0           # 有用边带相对载波的频率符号：+1 为蓝边带，-1 为红边带
    sideband_extra_phase_pi: float = 0.0       # 额外边带偏振相位修正，单位 pi
    include_opposite_eom_sideband: bool = False # 是否加入 EOM 对侧一阶边带的远失谐效应
    opposite_sideband_scattering: bool = True  # 对侧边带是否加入有效远失谐散射
    include_off_resonant_cross_coupling: bool = False # 是否加入载波/有用边带对错误基态的远失谐耦合
    cross_coupling_scattering: bool = True     # cross-coupling 是否加入有效远失谐散射
    lz_coupling_MHz: float = 3.0               # 碰撞 Condon 点 Landau-Zener 有效耦合 Omega/2pi
    mot_loading_rate_Hz: float = 4.0           # MOT->光镊单原子装载率 R (Hz)
    vacuum_lifetime_s: float = 10.0            # 真空背景气体碰撞寿命 (s)
    loading_t_end_s: float = 2.0               # 装载速率方程积分时间 (s)
    initial_nbar: float = 1                    # 初始热分布平均声子数 <n>
    min_temperature_uK: float = 2.0            # 碰撞温度估计的下限 (uK)
    sigma_energy_floor_J: float = 1e-30        # 能量展宽除零保护 (J)
    steady_reg_dephase_Hz: float = 20.0        # 稳态兜底正则化：基态去相干
    steady_reg_mix_Hz: float = 15.0            # 稳态兜底正则化：F=1/F=2 混合
    steady_reg_anchor_Hz: float = 8.0          # 稳态兜底正则化：弱锚定态
    steady_reg_motional_cool_Hz: float = 80.0  # 稳态兜底正则化：弱运动冷却
    show_envelope: bool = False                # 是否显示平滑包络


@dataclass
class DerivedParams:
    trap_wavelength_m: float
    trap_waist_m: float
    trap_depth_J: float
    trap_depth_Hz: float
    trap_depth_rad: float
    omega_t_rad: float
    trap_freq_Hz: float
    eta: float
    c3: float
    mot_loading_rate_Hz: float
    gamma_bg_Hz: float
    eom_microwave_Hz: float
    beta_eom: float
    sideband_ratio: float
    d1_saturation_intensity_mW_cm2: float
    carrier_intensity_mW_cm2: float
    sideband_intensity_mW_cm2: float


def solve_beta_from_ratio(target_ratio: float) -> float:
    def f(beta):
        return (jv(1, beta) / jv(0, beta)) ** 2 - target_ratio

    root = root_scalar(f, bracket=(0.01, 1.8), method="brentq")
    return float(root.root)


def build_derived_params(inp: UserInputs) -> DerivedParams:
    if inp.trap_wavelength_nm <= 0:
        raise ValueError("trap_wavelength_nm must be positive.")
    if inp.trap_depth_mK <= 0:
        raise ValueError("trap_depth_mK must be positive.")
    if inp.trap_waist_um <= 0:
        raise ValueError("trap_waist_um must be positive.")
    if inp.blue_detuning_MHz <= 0:
        raise ValueError("blue_detuning_MHz must be positive for blue-detuned grey molasses.")
    if inp.sideband_ratio <= 0:
        raise ValueError("sideband_ratio must be positive.")
    if inp.cooling_total_intensity_mW_cm2 <= 0:
        raise ValueError("cooling_total_intensity_mW_cm2 must be positive.")
    if inp.retro_mirror_distance_m < 0:
        raise ValueError("retro_mirror_distance_m must be non-negative.")
    if inp.sideband_phase_sign not in (-1.0, 1.0):
        raise ValueError("sideband_phase_sign must be +1.0 or -1.0.")
    if inp.lz_coupling_MHz <= 0:
        raise ValueError("lz_coupling_MHz must be positive.")
    if inp.loading_t_end_s <= 0:
        raise ValueError("loading_t_end_s must be positive.")
    if inp.min_temperature_uK <= 0:
        raise ValueError("min_temperature_uK must be positive.")
    if inp.sigma_energy_floor_J <= 0:
        raise ValueError("sigma_energy_floor_J must be positive.")

    trap_wavelength_m = inp.trap_wavelength_nm * 1e-9
    trap_waist_m = inp.trap_waist_um * 1e-6
    trap_depth_J = inp.trap_depth_mK * 1e-3 * KB
    trap_depth_Hz = trap_depth_J / H
    trap_depth_rad = 2 * np.pi * trap_depth_Hz

    omega_t_rad = np.sqrt(4.0 * trap_depth_J / (M_RB87 * trap_waist_m**2))
    trap_freq_Hz = omega_t_rad / (2 * np.pi)

    k_d1 = 2 * np.pi / LAMBDA_D1
    x0 = np.sqrt(HBAR / (2 * M_RB87 * omega_t_rad))
    eta = k_d1 * x0

    c3 = (3.0 / 4.0) * HBAR * GAMMA_D1_RAD / (k_d1**3)

    mot_loading_rate_Hz = float(inp.mot_loading_rate_Hz)
    gamma_bg_Hz = 1.0 / max(inp.vacuum_lifetime_s, 1e-6)

    eom_microwave_Hz = HFS_SPLITTING_HZ + inp.eom_offset_kHz * 1e3

    sideband_ratio = float(inp.sideband_ratio)
    beta_eom = solve_beta_from_ratio(sideband_ratio)
    carrier_intensity_mW_cm2 = inp.cooling_total_intensity_mW_cm2 / (1.0 + sideband_ratio)
    sideband_intensity_mW_cm2 = sideband_ratio * carrier_intensity_mW_cm2

    return DerivedParams(
        trap_wavelength_m=trap_wavelength_m,
        trap_waist_m=trap_waist_m,
        trap_depth_J=trap_depth_J,
        trap_depth_Hz=trap_depth_Hz,
        trap_depth_rad=trap_depth_rad,
        omega_t_rad=omega_t_rad,
        trap_freq_Hz=trap_freq_Hz,
        eta=eta,
        c3=c3,
        mot_loading_rate_Hz=mot_loading_rate_Hz,
        gamma_bg_Hz=gamma_bg_Hz,
        eom_microwave_Hz=eom_microwave_Hz,
        beta_eom=beta_eom,
        sideband_ratio=sideband_ratio,
        d1_saturation_intensity_mW_cm2=D1_SATURATION_INTENSITY_W_M2 / W_M2_PER_MW_CM2,
        carrier_intensity_mW_cm2=carrier_intensity_mW_cm2,
        sideband_intensity_mW_cm2=sideband_intensity_mW_cm2,
    )


def rabi_rad_from_intensity_mW_cm2(intensity_mW_cm2: float) -> float:
    intensity_w_m2 = max(float(intensity_mW_cm2), 0.0) * W_M2_PER_MW_CM2
    return float(GAMMA_D1_RAD * np.sqrt(intensity_w_m2 / (2.0 * D1_SATURATION_INTENSITY_W_M2)))


def cooling_rabi_frequencies(dp: DerivedParams) -> tuple[float, float]:
    omega_c = rabi_rad_from_intensity_mW_cm2(dp.carrier_intensity_mW_cm2)
    omega_r = rabi_rad_from_intensity_mW_cm2(dp.sideband_intensity_mW_cm2)
    return omega_c, omega_r


def cooling_polarization_phases(inp: UserInputs, dp: DerivedParams) -> tuple[float, float, float]:
    phi_c = inp.polarization_phase_pi * np.pi
    delta_phi = (
        inp.sideband_phase_sign
        * 4.0
        * np.pi
        * inp.retro_mirror_distance_m
        * dp.eom_microwave_Hz
        / C
        + inp.sideband_extra_phase_pi * np.pi
    )
    phi_r = phi_c + delta_phi
    return phi_c, phi_r, delta_phi


def off_resonant_effective_terms(
    inp: UserInputs,
    dp: DerivedParams,
    recoil,
    i_mot,
    omega_rad: float,
    phase_rad: float,
    optical_offset_Hz: float,
    ground_F_list: tuple[int, ...],
    include_scattering: bool,
    label: str,
):
    n_int = len(INTERNAL_STATES)
    zero = 0 * tensor(qeye(n_int), i_mot)
    h_eff = zero
    c_ops_eff = []

    for Fg in ground_F_list:
        ground_shift_Hz = HFS_SPLITTING_HZ if Fg == 1 else 0.0
        for Fp in (1, 2):
            excited_shift_Hz = EHFS_EXCITED_HZ if Fp == 1 else 0.0
            detuning_Hz = inp.blue_detuning_MHz * 1e6 + optical_offset_Hz - ground_shift_Hz + excited_shift_Hz
            detuning_rad = 2.0 * np.pi * detuning_Hz
            if abs(detuning_rad) < 2.0 * np.pi * 5.0e6:
                raise ValueError(
                    f"{label} is too close to resonance for off-resonant elimination: "
                    f"Fg={Fg}, Fp={Fp}, detuning={detuning_Hz/1e6:.3f} MHz."
                )

            sigma_p = build_sigma_q_F_to_Fp(+1, Fg, Fp)
            sigma_m = build_sigma_q_F_to_Fp(-1, Fg, Fp)
            exc = tensor(sigma_p.dag(), recoil) + np.exp(1j * phase_rad) * tensor(sigma_m.dag(), recoil.dag())
            h_eff = h_eff + (omega_rad**2 / (4.0 * detuning_rad)) * (exc.dag() * exc)

            if not include_scattering:
                continue

            amp = omega_rad / (2.0 * detuning_rad)
            for Fd in (1, 2):
                for m in range(-Fd, Fd + 1):
                    for q in (-1, 0, 1):
                        mp = m + q
                        if ('e', Fp, mp) not in EXCITED_STATES:
                            continue
                        coeff = cgc_weight(Fd, m, Fp, mp, q)
                        if abs(coeff) < 1e-14:
                            continue
                        ket_g = basis(n_int, STATE_TO_INDEX[('g', Fd, m)])
                        ket_e = basis(n_int, STATE_TO_INDEX[('e', Fp, mp)])
                        decay_int = coeff * ket_g * ket_e.dag()
                        c_ops_eff.append(np.sqrt(GAMMA_D1_RAD / 6.0) * amp * tensor(decay_int, recoil) * exc)
                        c_ops_eff.append(np.sqrt(GAMMA_D1_RAD / 6.0) * amp * tensor(decay_int, recoil.dag()) * exc)
                        c_ops_eff.append(np.sqrt(GAMMA_D1_RAD * (2.0 / 3.0)) * amp * tensor(decay_int, i_mot) * exc)

    return h_eff, c_ops_eff


def opposite_eom_sideband_terms(inp: UserInputs, dp: DerivedParams, n_fock: int, recoil, i_mot):
    n_int = len(INTERNAL_STATES)
    zero = 0 * tensor(qeye(n_int), i_mot)
    if not inp.include_opposite_eom_sideband:
        return zero, [], 0.0

    omega_opp = rabi_rad_from_intensity_mW_cm2(dp.sideband_intensity_mW_cm2)
    phi_c, _, delta_phi = cooling_polarization_phases(inp, dp)
    phi_opp = phi_c - delta_phi
    opposite_offset_Hz = -inp.sideband_phase_sign * dp.eom_microwave_Hz
    h_eff, c_ops_eff = off_resonant_effective_terms(
        inp,
        dp,
        recoil,
        i_mot,
        omega_rad=omega_opp,
        phase_rad=phi_opp,
        optical_offset_Hz=opposite_offset_Hz,
        ground_F_list=(1, 2),
        include_scattering=inp.opposite_sideband_scattering,
        label="opposite EOM sideband",
    )
    return h_eff, c_ops_eff, float(phi_opp)


def off_resonant_cross_coupling_terms(inp: UserInputs, dp: DerivedParams, n_fock: int, recoil, i_mot):
    n_int = len(INTERNAL_STATES)
    zero = 0 * tensor(qeye(n_int), i_mot)
    if not inp.include_off_resonant_cross_coupling:
        return zero, []

    omega_c, omega_r = cooling_rabi_frequencies(dp)
    phi_c, phi_r, _ = cooling_polarization_phases(inp, dp)

    h_carrier_f1, c_carrier_f1 = off_resonant_effective_terms(
        inp,
        dp,
        recoil,
        i_mot,
        omega_rad=omega_c,
        phase_rad=phi_c,
        optical_offset_Hz=0.0,
        ground_F_list=(1,),
        include_scattering=inp.cross_coupling_scattering,
        label="carrier cross-coupling to F=1",
    )
    h_sideband_f2, c_sideband_f2 = off_resonant_effective_terms(
        inp,
        dp,
        recoil,
        i_mot,
        omega_rad=omega_r,
        phase_rad=phi_r,
        optical_offset_Hz=inp.sideband_phase_sign * dp.eom_microwave_Hz,
        ground_F_list=(2,),
        include_scattering=inp.cross_coupling_scattering,
        label="useful sideband cross-coupling to F=2",
    )

    return h_carrier_f1 + h_sideband_f2, c_carrier_f1 + c_sideband_f2


def internal_dark_state_diagnostics(inp: UserInputs, output_dir=None, output_tag: str = "", show_progress: bool = True):
    dp = build_derived_params(inp)
    n_int = len(INTERNAL_STATES)
    delta_rad = 2 * np.pi * inp.blue_detuning_MHz * 1e6
    omega_c, omega_r = cooling_rabi_frequencies(dp)

    differential_shift_Hz = inp.differential_light_shift_kHz * 1e3
    eom_gap_Hz = dp.eom_microwave_Hz - HFS_SPLITTING_HZ
    delta_raman_rad = 2 * np.pi * (eom_gap_Hz - differential_shift_Hz)

    p_f1, p_f2, p_e1, p_e2, p_e = build_projectors()
    sigma_p_f1 = build_sigma_q_F(+1, 1)
    sigma_m_f1 = build_sigma_q_F(-1, 1)
    sigma_p_f2 = build_sigma_q_F(+1, 2)
    sigma_m_f2 = build_sigma_q_F(-1, 2)

    delta_e1_rad = delta_rad + 2 * np.pi * EHFS_EXCITED_HZ
    h_internal = delta_raman_rad * p_f1 - delta_rad * p_e2 - delta_e1_rad * p_e1

    phi_c, phi_r, delta_phi_pol = cooling_polarization_phases(inp, dp)
    exc_f2 = sigma_p_f2.dag() + np.exp(1j * phi_c) * sigma_m_f2.dag()
    exc_f1 = sigma_p_f1.dag() + np.exp(1j * phi_r) * sigma_m_f1.dag()
    h_drive = 0.5 * omega_c * (exc_f2 + exc_f2.dag()) + 0.5 * omega_r * (exc_f1 + exc_f1.dag())
    h_total = h_internal + h_drive

    c_ops = []
    for F in (1, 2):
        for m in range(-F, F + 1):
            for q in (-1, 0, 1):
                mp = m + q
                for Fp in (1, 2):
                    if ('e', Fp, mp) not in EXCITED_STATES:
                        continue
                    coeff = cgc_weight(F, m, Fp, mp, q)
                    if abs(coeff) < 1e-14:
                        continue
                    ket_g = basis(n_int, STATE_TO_INDEX[('g', F, m)])
                    ket_e = basis(n_int, STATE_TO_INDEX[('e', Fp, mp)])
                    c_ops.append(np.sqrt(GAMMA_D1_RAD) * coeff * ket_g * ket_e.dag())

    rho_ss = steadystate(h_total, c_ops, method="svd")
    pe_ss = float(expect(p_e, rho_ss))
    pf1_ss = float(expect(p_f1, rho_ss))
    pf2_ss = float(expect(p_f2, rho_ss))

    evals, evecs = h_total.eigenstates()
    rows = []
    for idx, (val, vec) in enumerate(zip(evals, evecs)):
        rho_vec = vec * vec.dag()
        pe_vec = float(expect(p_e, rho_vec))
        pf1_vec = float(expect(p_f1, rho_vec))
        pf2_vec = float(expect(p_f2, rho_vec))
        if pe_vec < 0.1:
            rows.append([idx, float(val / (2 * np.pi * 1e6)), pe_vec, pf1_vec, pf2_vec])

    arr = np.asarray(rows, dtype=float) if rows else np.empty((0, 5), dtype=float)
    out_dir = ensure_output_dir(output_dir) if output_dir is not None else None
    if out_dir is not None:
        tag = f"_{output_tag}" if output_tag else ""
        save_csv(
            out_dir / f"internal_dark_state_diagnostics{tag}.csv",
            "eigen_index,energy_MHz,pe,pf1,pf2",
            arr,
        )

    if show_progress:
        print("=" * 64)
        print("内部暗态/AC Stark 诊断")
        print(f"steady Pe                  = {pe_ss:.3e}")
        print(f"steady PF1, PF2             = {pf1_ss:.6f}, {pf2_ss:.6f}")
        print(f"delta_R/2pi                 = {delta_raman_rad/(2*np.pi*1e3):.3f} kHz")
        print(f"polarization delta_phi/pi   = {delta_phi_pol/np.pi:.6f}")
        print(f"Omega_c/2pi, Omega_s/2pi    = {omega_c/(2*np.pi*1e6):.3f}, {omega_r/(2*np.pi*1e6):.3f} MHz")
        for row in arr[:12]:
            dom_f = "F=1" if row[3] >= row[4] else "F=2"
            print(f"[{int(row[0]):02d}] {dom_f} E={row[1]:9.4f} MHz Pe={row[2]:.2e} PF1={row[3]:.3f} PF2={row[4]:.3f}")
        if out_dir is not None:
            print(f"诊断文件已保存到             = {out_dir}")
        print("=" * 64)

    return {
        "pe_ss": pe_ss,
        "pf1_ss": pf1_ss,
        "pf2_ss": pf2_ss,
        "eigen_table": arr,
        "omega_c": omega_c,
        "omega_r": omega_r,
        "delta_raman_rad": delta_raman_rad,
        "output_dir": out_dir,
    }


# ==========================================
# 模块一：双体碰撞动力学
# ==========================================
def _effective_temperature_from_cooling(inp: UserInputs, dp: DerivedParams, cool_result=None) -> float:
    if cool_result is not None and "n_final" in cool_result:
        n_eff = max(float(cool_result["n_final"]), 0.0)
    else:
        n_eff = max(float(inp.initial_nbar), 0.0)
    return max((n_eff + 0.5) * HBAR * dp.omega_t_rad / KB, inp.min_temperature_uK * 1e-6)


def _direct_escape_probability(u0_j: float, e_single_j: float, temp_k: float) -> float:
    if e_single_j >= u0_j:
        return 1.0
    if e_single_j <= 0.0:
        return 0.0

    kbt = KB * max(temp_k, 1e-9)
    x_u = u0_j / kbt
    x_l = max((u0_j - e_single_j) / kbt, 0.0)

    # P_direct = ∫_{U0-Esingle}^{U0} sqrt(E)exp(-E/kBT)dE / ∫_{0}^{U0} sqrt(E)exp(-E/kBT)dE
    # 用下不完全伽马函数 γ(3/2, x) 表示后，公共系数会约掉。
    g_u = gammainc(1.5, x_u)
    g_l = gammainc(1.5, x_l)
    denominator = max(g_u, 1e-15)
    numerator = np.clip(g_u - g_l, 0.0, 1.0)
    return float(np.clip(numerator / denominator, 0.0, 1.0))

def collision_module(inp: UserInputs, dp: DerivedParams, tau_cool_s: float, cool_result=None):
    from math import erf, erfc

    single_photon_detuning_Hz = inp.blue_detuning_MHz * 1e6
    differential_shift_Hz = inp.differential_light_shift_kHz * 1e3
    delta_eff_Hz = single_photon_detuning_Hz - differential_shift_Hz
    if delta_eff_Hz <= 0.0:
        raise ValueError(
            "Effective blue detuning must be positive: "
            f"blue_detuning_MHz={inp.blue_detuning_MHz:g}, "
            f"differential_light_shift_kHz={inp.differential_light_shift_kHz:g}."
        )
    e_single_Hz = delta_eff_Hz / 2.0
    e_single_j = H * e_single_Hz
    u0_j = dp.trap_depth_J
    e_total_j = 2.0 * e_single_j

    # --- 温度由冷却模块联动（无冷却结果时退化到 initial_nbar 对应温度） ---
    t_cold_k = _effective_temperature_from_cooling(inp, dp, cool_result)

    # --- LZ 激发概率（Condon 点） ---
    rc = (dp.c3 / (H * delta_eff_Hz)) ** (1.0 / 3.0)
    dVdR = 3.0 * dp.c3 / (rc**4)
    v_rel = np.sqrt(max(16.0 * KB * t_cold_k / (np.pi * M_RB87), 1e-12))
    alpha = abs(dVdR * v_rel / HBAR)

    omega_lac = 2 * np.pi * inp.lz_coupling_MHz * 1e6
    gamma_lz = (omega_lac**2) / (4.0 * max(alpha, 1.0))
    p_excite = 1.0 - np.exp(-2.0 * np.pi * gamma_lz)

    # --- 真实双体相遇碰撞率 Γ_enc = K2 * ∫n(r)^2 d^3r ---
    omega_bar = dp.omega_t_rad
    n2_integral = (M_RB87 * omega_bar**2 / (4.0 * np.pi * KB * t_cold_k)) ** 1.5
    sigma_lac = np.pi * rc**2
    k2 = sigma_lac * v_rel * p_excite
    gamma_collide = max(float(k2 * n2_integral), 0.0)

    # ================= 修改区域开始 =================
    
    # 1. 冷原子的稳态散射率（暗态不破缺，仅用于诊断和维持输出接口不报错）
    if cool_result is not None and "pe_mean" in cool_result:
        gamma_scatt_cold_hz = max(float(cool_result["pe_mean"]) * GAMMA_D1_HZ, 1e-6)
    else:
        gamma_scatt_cold_hz = 1e-3

    # 2. 估算暗态破缺后的热原子有效散射率 Gamma_scatt_hot
    # Optical scattering depends on the laser single-photon detuning.
    delta_rad = 2.0 * np.pi * single_photon_detuning_Hz
    
    omega_c_hot, omega_r_hot = cooling_rabi_frequencies(dp)

    # 两束 Λ 光场的总饱和参数；实际跃迁强度仍由 CG 系数在 Lindblad 模块中给出。
    s0 = 2.0 * (omega_c_hot**2 + omega_r_hot**2) / (GAMMA_D1_RAD**2)
    
    # 使用大失谐两能级模型计算剧烈运动热原子的有效自发辐射率
    gamma_scatt_hot_hz = (GAMMA_D1_HZ / 2.0) * s0 / (1.0 + s0 + 4.0 * (delta_rad / GAMMA_D1_RAD)**2)
    gamma_scatt_hot_hz = max(float(gamma_scatt_hot_hz), 0.0)

    # ================= 修改区域结束 =================

    # --- 质心热运动导致实验室系能量不对称：E1,2 = E/2 ± δE ---
    # sigma_e_j = np.sqrt(max(0.5 * e_total_j * KB * t_cold_k, 1e-30))  #codex version

    # 将下限放宽到 1e-60（适合能量平方的 SI 量级），或者直接把 max 放在外面：
    sigma_e_j = max(np.sqrt(max(0.5 * e_total_j * KB * t_cold_k, 0.0)), inp.sigma_energy_floor_J)

    delta_u_j = u0_j - e_single_j
    z = abs(delta_u_j) / (np.sqrt(2.0) * max(sigma_e_j, 1e-30))

    if delta_u_j > 0.0:
        p_single_escape_direct = erfc(z)
        p_double_escape_direct = 0.0
        regime = "蓝失谐单原子保留区 (E < 2U)"
    else:
        p_double_escape_direct = erf(z)
        p_single_escape_direct = erfc(z)
        regime = "蓝失谐双原子丢失区 (E > 2U)"

    p_single_escape_direct = float(np.clip(p_single_escape_direct, 0.0, 1.0))
    p_double_escape_direct = float(np.clip(p_double_escape_direct, 0.0, 1.0))

    p_stay = max(1.0 - p_single_escape_direct - p_double_escape_direct, 0.0)

    t_hot_k = t_cold_k + e_single_j / (3.0 * KB)

    # ================= 修改区域应用 =================
    # 滞留热原子的迟滞蒸发 (Kramers Evaporation) 竞争
    # 使用 gamma_scatt_hot_hz 替代原来的 gamma_scatt_hz
    gamma_evap = gamma_scatt_hot_hz * np.exp(-max(delta_u_j, 0.0) / max(KB * t_hot_k, 1e-20))
    # ===============================================

    cooling_rate = 1.0 / max(tau_cool_s, 1e-8)
    
    p_delayed_single = p_stay * (2.0 * gamma_evap) / (2.0 * gamma_evap + cooling_rate)

    beta1 = gamma_collide * (p_single_escape_direct + p_delayed_single)
    beta2 = gamma_collide * p_double_escape_direct

    p_direct = p_single_escape_direct
    p_second_escape = p_double_escape_direct

    return {
        "delta_eff_Hz": delta_eff_Hz,
        "e_single_Hz": e_single_Hz,
        "p_excite": p_excite,
        "k2_m3_per_s": k2,
        "gamma_enc_Hz": gamma_collide,
        "temperature_cold_K": t_cold_k,
        "temperature_hot_K": t_hot_k,
        "gamma_scatt_Hz": gamma_scatt_hot_hz,
        "gamma_scatt_cold_Hz": gamma_scatt_cold_hz,
        "gamma_scatt_hot_Hz": gamma_scatt_hot_hz,  
        "p_direct": p_direct,
        "p_second_escape": p_second_escape,
        "gamma_evap_Hz": float(gamma_evap) if np.isfinite(gamma_evap) else np.inf,
        "gamma_collide_Hz": gamma_collide,
        "beta1_Hz": beta1,
        "beta2_Hz": beta2,
        "regime": regime,
    }

# ==========================================
# 模块二：单原子冷却动力学（Lindblad）
# ==========================================
def cooling_module(
    inp: UserInputs,
    dp: DerivedParams, 
    n_fock: int = 5,
    # t_end_s: float = 1.6e-3,
    # n_t: int = 110,
    # show_progress: bool = False,   #原始参数
    t_end_s: float = 2.0e-3,
    n_t: int = 80,
    show_progress: bool = False,
    robust_integrator: bool = False,
    max_runtime_s: float = 90.0,
    debug_internal_diagnostics: bool = False,
    solver_options: dict | None = None,
):
    n_int = len(INTERNAL_STATES)

    delta_rad = 2 * np.pi * inp.blue_detuning_MHz * 1e6
    omega_c, omega_r = cooling_rabi_frequencies(dp)

    differential_shift_Hz = inp.differential_light_shift_kHz * 1e3
    eom_gap_Hz = dp.eom_microwave_Hz - HFS_SPLITTING_HZ
    delta_raman_rad = 2 * np.pi * (eom_gap_Hz - differential_shift_Hz)

    p_f1, p_f2, p_e1, p_e2, p_e = build_projectors()

    i_int = qeye(n_int)
    i_mot = qeye(n_fock)
    a = destroy(n_fock)
    n_op = a.dag() * a

    delta_e1_rad = delta_rad + 2 * np.pi * EHFS_EXCITED_HZ
    h_internal = delta_raman_rad * p_f1 - delta_rad * p_e2 - delta_e1_rad * p_e1
    h_mot = dp.omega_t_rad * n_op

    sigma_p_f1 = build_sigma_q_F(+1, 1)
    sigma_m_f1 = build_sigma_q_F(-1, 1)
    sigma_p_f2 = build_sigma_q_F(+1, 2)
    sigma_m_f2 = build_sigma_q_F(-1, 2)

    phi_c, phi_r, delta_phi_pol = cooling_polarization_phases(inp, dp)
    z_ld = a + a.dag()
    recoil = (1j * dp.eta * z_ld).expm()

    if debug_internal_diagnostics:
        internal_dark_state_diagnostics(inp, show_progress=True)
    
    # 不使用 Lamb-Dicke 一阶截断：直接使用精确反冲算符 R 和 R^
    exc_f2 = tensor(sigma_p_f2.dag(), recoil) + np.exp(1j * phi_c) * tensor(sigma_m_f2.dag(), recoil.dag())
    exc_f1 = tensor(sigma_p_f1.dag(), recoil) + np.exp(1j * phi_r) * tensor(sigma_m_f1.dag(), recoil.dag())
    h_af_f2 = 0.5 * omega_c * (exc_f2 + exc_f2.dag())
    h_af_f1 = 0.5 * omega_r * (exc_f1 + exc_f1.dag())
    

    h = tensor(h_internal, i_mot) + tensor(i_int, h_mot) + h_af_f2 + h_af_f1

    recoil_emit_fwd = recoil
    recoil_emit_bwd = recoil.dag()
    c_ops = []
    for F in (1, 2):
        for m in range(-F, F + 1):
            for q in (-1, 0, 1):
                mp = m + q
                for Fp in (1, 2):
                    if ('e', Fp, mp) not in EXCITED_STATES:
                        continue
                    coeff = cgc_weight(F, m, Fp, mp, q)
                    if abs(coeff) < 1e-14:
                        continue
                    ket_g = basis(n_int, STATE_TO_INDEX[('g', F, m)])
                    ket_e = basis(n_int, STATE_TO_INDEX[('e', Fp, mp)])
                    c_int = coeff * ket_g * ket_e.dag()
                    # 真实1D投影：1/3概率沿z轴产生正/负反冲，2/3概率横向发射（对z轴无反冲影响）
                    c_ops.append(np.sqrt(GAMMA_D1_RAD / 6.0) * tensor(c_int, recoil))
                    c_ops.append(np.sqrt(GAMMA_D1_RAD / 6.0) * tensor(c_int, recoil.dag()))
                    c_ops.append(np.sqrt(GAMMA_D1_RAD * (2.0/3.0)) * tensor(c_int, i_mot))

    """
    # 初态改为：冻结运动时的内态稳态 ⊗ 外部热声子分布
    exc_f2_int = sigma_p_f2.dag() + np.exp(1j * phi) * sigma_m_f2.dag()
    exc_f1_int = sigma_p_f1.dag() + np.exp(1j * phi) * sigma_m_f1.dag()
    h_af_f2_int = 0.5 * omega_c * (exc_f2_int + exc_f2_int.dag())
    h_af_f1_int = 0.5 * omega_r * (exc_f1_int + exc_f1_int.dag())
    h_int_total = h_internal + h_af_f2_int + h_af_f1_int

    c_ops_int = []
    for F in (1, 2):
        for m in range(-F, F + 1):
            for q in (-1, 0, 1):
                mp = m + q
                if (2, mp) not in EXCITED_STATES:
                    continue
                coeff = cgc_weight(F, m, 2, mp, q)
                if abs(coeff) < 1e-14:
                    continue
                ket_g = basis(n_int, STATE_TO_INDEX[(F, m)])
                ket_e = basis(n_int, STATE_TO_INDEX[(2, mp)])
                c_int = coeff * ket_g * ket_e.dag()
                c_ops_int.append(np.sqrt(GAMMA_D1_RAD) * c_int)

    # try:
    #     rho_int = steadystate(h_int_total, c_ops_int, method="power")
    # except Exception:
    #     psi0_int = basis(n_int, STATE_TO_INDEX[(2, 2)])
    #     rho_int = ket2dm(psi0_int)                #codex version

    # === 替换 cooling_module 中的 steadystate 初始化部分 ===
    try:
        # 优先使用 SVD，它是专门对付奇异矩阵（完美暗态）的数学利器
        rho_int = steadystate(h_int_total, c_ops_int, method="svd")
    except Exception:
        # SVD 兜底：加入极微弱的纯去相干，破坏绝对奇异性，但不影响真实的物理占据
        c_ops_reg = list(c_ops_int)
        for st in INTERNAL_STATES:
            ket = basis(n_int, STATE_TO_INDEX[st])
            c_ops_reg.append(np.sqrt(1e-5 * GAMMA_D1_RAD) * ket * ket.dag())
        rho_int = steadystate(h_int_total, c_ops_reg, method="direct")
    """
    
    psi0_int = basis(n_int, STATE_TO_INDEX[('g',2, 2)])
    rho_int = ket2dm(psi0_int)


    rho_mot = thermal_dm(n_fock, max(inp.initial_nbar, 0.0))
    rho0 = tensor(rho_int, rho_mot)
    
    

    tlist = np.linspace(0.0, t_end_s, n_t)
    pe_op = tensor(p_e, i_mot)
    n_tot_op = tensor(i_int, n_op)

    options = {
        "nsteps": 180000,
        "atol": 1e-8,
        "rtol": 1e-6,
        "method": "bdf",      #codex version
        "max_step": 1e-5,
        "store_final_state": True,
    }
    if solver_options is not None:
        options.update(solver_options)
    options["store_final_state"] = True
    method_name = str(options.get("method", "bdf")).lower()
    if method_name == "diag":
        allowed = {"method", "eigensolver_dtype", "store_final_state"}
        options = {key: value for key, value in options.items() if key in allowed}
    elif method_name == "krylov":
        allowed = {
            "method",
            "atol",
            "nsteps",
            "max_step",
            "min_step",
            "always_compute_step",
            "krylov_dim",
            "sub_system_tol",
            "algorithm",
            "store_final_state",
        }
        options = {key: value for key, value in options.items() if key in allowed}

    def solve_mesolve_with_retry(rho_init, t_eval):
        option_trials = [
            options,
            {
                **options,
                "nsteps": max(int(options["nsteps"]), 180000),
                "max_step": min(float(options["max_step"]), 5e-6),
            },
            {
                **options,
                "nsteps": max(int(options["nsteps"]), 220000),
                "max_step": min(float(options["max_step"]), 2e-6),
                "atol": 5e-8,
                "rtol": 5e-6,
            },
        ]

        last_error = None
        for trial_opt in option_trials:
            try:
                return mesolve(h, rho_init, t_eval, c_ops, e_ops=[n_tot_op, pe_op], options=trial_opt)
            except Exception as exc:
                last_error = exc
                if "Excess work done" not in str(exc):
                    raise
        raise last_error

    def solve_mesolve_once(rho_init, t_eval):
        return mesolve(h, rho_init, t_eval, c_ops, e_ops=[n_tot_op, pe_op], options=options)

    solver_runner = solve_mesolve_with_retry if robust_integrator else solve_mesolve_once

    if not show_progress:
        result = solver_runner(rho0, tlist)
        phonons = np.real(result.expect[0])
        pe = np.real(result.expect[1])
    else:
        segment_len = max(10, n_t // 12) if not robust_integrator else max(3, n_t // 28)
        segments = []
        start = 0
        while start < (n_t - 1):
            end = min(start + segment_len, n_t - 1)
            segments.append((start, end))
            start = end

        total_segments = len(segments)
        rho_curr = rho0
        t_collect = []
        n_collect = []
        pe_collect = []
        t_start_wall = time.perf_counter()

        print("开始计算冷却曲线...")
        for idx_seg, (i0, i1) in enumerate(segments, start=1):
            if robust_integrator and (time.perf_counter() - t_start_wall) > max(max_runtime_s, 1.0):
                print("\n[警告] 冷却计算达到时间预算，提前结束并返回当前已收敛区段结果。")
                break

            t_seg = tlist[i0 : i1 + 1]
            res_seg = solver_runner(rho_curr, t_seg)
            rho_curr = res_seg.final_state

            n_seg = np.real(res_seg.expect[0])
            pe_seg = np.real(res_seg.expect[1])

            if idx_seg == 1:
                t_collect.extend(t_seg.tolist())
                n_collect.extend(n_seg.tolist())
                pe_collect.extend(pe_seg.tolist())
            else:
                t_collect.extend(t_seg[1:].tolist())
                n_collect.extend(n_seg[1:].tolist())
                pe_collect.extend(pe_seg[1:].tolist())

            print_progress("冷却进度", idx_seg, total_segments)

        if robust_integrator and len(t_collect) == 0:
            t_collect = [float(tlist[0])]
            n_collect = [float(expect(n_tot_op, rho0))]
            pe_collect = [float(expect(pe_op, rho0))]

        tlist = np.array(t_collect)
        phonons = np.array(n_collect)
        pe = np.array(pe_collect)

    tau_cool_s, phonons_smooth = estimate_tau_cool(tlist, phonons)

    return {
        "t_s": tlist,
        "phonons": phonons,
        "phonons_smooth": phonons_smooth,
        "pe": pe,
        "tau_cool_s": tau_cool_s,
        "delta_raman_rad": delta_raman_rad,
        "delta_phi_pol_rad": delta_phi_pol,
        "omega_c": omega_c,
        "omega_r": omega_r,
        "n_final": float(phonons[-1]),
        "pe_mean": float(np.mean(pe[-max(5, len(pe)//6):])),
    }


def steady_state_n_for_ratio(
    inp: UserInputs,
    ratio: float,
    n_fock: int = 4,
    solver_method: str = "power",
    show_progress: bool = True,
    try_slow_methods: bool = False,
):
    """
    给定边带/载波光强比 Is/Ic，直接计算稳态平均声子数 <n>_ss（不做时间演化）。

    参数:
    - inp: 用户输入参数结构体
    - ratio: 指定的 Is/Ic
    - n_fock: 声子截断阶数（越大越准、越慢）
    - solver_method: qutip.steadystate 的求解方法（如 direct / power / iterative-gmres）
    - show_progress: 是否显示求解进度条
    - try_slow_methods: 是否在快方法失败后继续尝试慢方法（svd/eigen）
    """
    if ratio <= 0:
        raise ValueError("ratio 必须 > 0")

    dp = build_derived_params(replace(inp, sideband_ratio=float(ratio)))

    n_int = len(INTERNAL_STATES)
    delta_rad = 2 * np.pi * inp.blue_detuning_MHz * 1e6
    omega_c, omega_r = cooling_rabi_frequencies(dp)

    differential_shift_Hz = inp.differential_light_shift_kHz * 1e3
    eom_gap_Hz = dp.eom_microwave_Hz - HFS_SPLITTING_HZ
    delta_raman_rad = 2 * np.pi * (eom_gap_Hz - differential_shift_Hz)

    p_f1, _, p_e1, p_e2, p_e = build_projectors()
    i_int = qeye(n_int)
    i_mot = qeye(n_fock)
    a = destroy(n_fock)
    n_op = a.dag() * a

    delta_e1_rad = delta_rad + 2 * np.pi * EHFS_EXCITED_HZ
    h_internal = delta_raman_rad * p_f1 - delta_rad * p_e2 - delta_e1_rad * p_e1
    h_mot = dp.omega_t_rad * n_op

    sigma_p_f1 = build_sigma_q_F(+1, 1)
    sigma_m_f1 = build_sigma_q_F(-1, 1)
    sigma_p_f2 = build_sigma_q_F(+1, 2)
    sigma_m_f2 = build_sigma_q_F(-1, 2)

    phi_c, phi_r, delta_phi_pol = cooling_polarization_phases(inp, dp)
    recoil = (1j * dp.eta * (a + a.dag())).expm()

    exc_f2 = tensor(sigma_p_f2.dag(), recoil) + np.exp(1j * phi_c) * tensor(sigma_m_f2.dag(), recoil.dag())
    exc_f1 = tensor(sigma_p_f1.dag(), recoil) + np.exp(1j * phi_r) * tensor(sigma_m_f1.dag(), recoil.dag())
    h_af_f2 = 0.5 * omega_c * (exc_f2 + exc_f2.dag())
    h_af_f1 = 0.5 * omega_r * (exc_f1 + exc_f1.dag())

    h = tensor(h_internal, i_mot) + tensor(i_int, h_mot) + h_af_f2 + h_af_f1

    c_ops = []
    for F in (1, 2):
        for m in range(-F, F + 1):
            for q in (-1, 0, 1):
                mp = m + q
                for Fp in (1, 2):
                    if ('e', Fp, mp) not in EXCITED_STATES:
                        continue
                    coeff = cgc_weight(F, m, Fp, mp, q)
                    if abs(coeff) < 1e-14:
                        continue
                    ket_g = basis(n_int, STATE_TO_INDEX[('g', F, m)])
                    ket_e = basis(n_int, STATE_TO_INDEX[('e', Fp, mp)])
                    c_int = coeff * ket_g * ket_e.dag()
                    # Same 1D recoil projection model used in cooling_module:
                    # 1/6 emits along +z, 1/6 along -z, 2/3 transverse with no z recoil.
                    c_ops.append(np.sqrt(GAMMA_D1_RAD / 6.0) * tensor(c_int, recoil))
                    c_ops.append(np.sqrt(GAMMA_D1_RAD / 6.0) * tensor(c_int, recoil.dag()))
                    c_ops.append(np.sqrt(GAMMA_D1_RAD * (2.0 / 3.0)) * tensor(c_int, i_mot))

    h_opp, c_ops_opp, _ = opposite_eom_sideband_terms(inp, dp, n_fock, recoil, i_mot)
    h = h + h_opp
    c_ops.extend(c_ops_opp)
    h_cross, c_ops_cross = off_resonant_cross_coupling_terms(inp, dp, n_fock, recoil, i_mot)
    h = h + h_cross
    c_ops.extend(c_ops_cross)

    n_tot_op = tensor(i_int, n_op)
    pe_op = tensor(p_e, i_mot)

    def call_steadystate(h_op, c_list, method_name: str, weight_value=None):
        kwargs = {"method": method_name}
        if weight_value is not None:
            kwargs["weight"] = float(weight_value)
        if method_name == "direct":
            kwargs["use_rcm"] = True
            kwargs["use_wbm"] = True
        return steadystate(h_op, c_list, **kwargs)

    def run_ss_with_progress(h_op, c_list, method_name: str, prefix: str, weight_value=None):
        if not show_progress:
            return call_steadystate(h_op, c_list, method_name, weight_value=weight_value)

        result = {"rho": None, "err": None}
        done = threading.Event()

        def worker():
            try:
                result["rho"] = call_steadystate(h_op, c_list, method_name, weight_value=weight_value)
            except Exception as exc:
                result["err"] = exc
            finally:
                done.set()

        th = threading.Thread(target=worker, daemon=True)
        th.start()

        start = time.perf_counter()
        width = 24
        tick = 0
        while not done.wait(0.4):
            pos = tick % width
            bar = ["·"] * width
            bar[pos] = "█"
            elapsed = time.perf_counter() - start
            print(f"\r{prefix} [{''.join(bar)}] {elapsed:6.1f}s", end="", flush=True)
            tick += 1

        elapsed = time.perf_counter() - start
        print(f"\r{prefix} [完成{'█' * (width - 2)}] {elapsed:6.1f}s")

        if result["err"] is not None:
            raise result["err"]
        return result["rho"]

    physical_lsqr_methods = {"lsqr", "liouvillian-lsqr", "physical-lsqr", "lsqr-physical", "liouvillian-lsqr-physical"}
    regularized_lsqr_methods = {"fallback", "regularized-lsqr", "liouvillian-lsqr-regularized"}
    if solver_method in physical_lsqr_methods | regularized_lsqr_methods:
        method_candidates = []
    else:
        fast_methods = [solver_method, "power"]
        slow_methods = ["svd", "eigen"] if try_slow_methods else []
        method_candidates = fast_methods + slow_methods
    method_order = []
    for m in method_candidates:
        if m not in method_order:
            method_order.append(m)

    rho_ss = None
    used_method = None
    last_error = None

    weight_candidates = [None, 1.0, 10.0]

    def solve_liouvillian_lsqr(c_list, label: str, iter_lim: int = 12000):
        l_op = liouvillian(h, c_list)
        l_csr = l_op.data.as_scipy().tocsr()
        dim = h.shape[0]

        trace_vec = np.eye(dim, dtype=complex).reshape(dim * dim, order="F")
        tr_row = sparse.csr_matrix(trace_vec.reshape(1, -1))

        tr_weight = max(5.0, float(np.mean(np.abs(l_csr.data))) if l_csr.nnz > 0 else 5.0)
        a_aug = sparse.vstack([l_csr, tr_weight * tr_row], format="csr")
        b_aug = np.zeros(dim * dim + 1, dtype=complex)
        b_aug[-1] = tr_weight

        ls_out = lsqr(a_aug, b_aug, atol=1e-10, btol=1e-10, iter_lim=iter_lim)
        rho_mat = ls_out[0].reshape((dim, dim), order="F")
        rho_q = Qobj(rho_mat, dims=h.dims)
        rho_q = 0.5 * (rho_q + rho_q.dag())
        tr = rho_q.tr()
        if abs(tr) <= 1e-14:
            raise RuntimeError(f"{label} produced a near-zero-trace density matrix.")
        rho_q = rho_q / tr
        solver_label = f"{label}(itn={ls_out[2]},r1={ls_out[3]:.3g})"
        return rho_q, solver_label

    for idx, method in enumerate(method_order, start=1):
        method_ok = False
        for weight_value in weight_candidates:
            weight_text = "auto" if weight_value is None else f"{weight_value:g}"
            try:
                if show_progress:
                    print(f"稳态求解：尝试 {idx}/{len(method_order)}，方法={method}, weight={weight_text}")
                rho_ss = run_ss_with_progress(
                    h,
                    c_ops,
                    method,
                    prefix=f"steadystate({method},w={weight_text})",
                    weight_value=weight_value,
                )
                used_method = f"{method}(w={weight_text})"
                method_ok = True
                break
            except Exception as exc:
                last_error = exc
                if show_progress:
                    print(f"方法 {method} 在 weight={weight_text} 失败：{type(exc).__name__}: {str(exc)[:120]}")
        if method_ok:
            break

    if rho_ss is None and solver_method in physical_lsqr_methods:
        if show_progress:
            print("进入物理 Liouvillian LSQR 求解（不加入正则化通道）...")
        try:
            rho_ss, used_method = solve_liouvillian_lsqr(c_ops, "physical-liouvillian-lsqr")
        except Exception as exc:
            last_error = exc
            if show_progress:
                print(f"物理 LSQR 求解失败：{type(exc).__name__}: {str(exc)[:160]}")

    if rho_ss is None:
        # 兜底正则化：加入极弱去简并通道（内部态去相干+跨超精细混合+微弱运动冷却）
        # 仅在标准 steadystate 全失败时启用。
        reg_dephase = 2 * np.pi * inp.steady_reg_dephase_Hz
        reg_mix = 2 * np.pi * inp.steady_reg_mix_Hz
        reg_anchor = 2 * np.pi * inp.steady_reg_anchor_Hz
        reg_motional_cool = 2 * np.pi * inp.steady_reg_motional_cool_Hz
        c_ops_reg = list(c_ops)

        for (_, F, m) in GROUND_STATES:
            ket = basis(n_int, STATE_TO_INDEX[('g', F, m)])
            proj = ket * ket.dag()
            c_ops_reg.append(np.sqrt(reg_dephase) * tensor(proj, i_mot))

        for m in (-1, 0, 1):
            ket_f1 = basis(n_int, STATE_TO_INDEX[('g', 1, m)])
            ket_f2 = basis(n_int, STATE_TO_INDEX[('g', 2, m)])
            c_ops_reg.append(np.sqrt(reg_mix) * tensor(ket_f2 * ket_f1.dag(), i_mot))
            c_ops_reg.append(np.sqrt(reg_mix) * tensor(ket_f1 * ket_f2.dag(), i_mot))

        anchor = basis(n_int, STATE_TO_INDEX[('g', 2, 2)])
        for st in INTERNAL_STATES:
            idx_st = STATE_TO_INDEX[st]
            if idx_st == STATE_TO_INDEX[('g', 2, 2)]:
                continue
            ket_st = basis(n_int, idx_st)
            c_ops_reg.append(np.sqrt(reg_anchor) * tensor(anchor * ket_st.dag(), i_mot))

        c_ops_reg.append(np.sqrt(reg_motional_cool) * tensor(i_int, a))

        if show_progress:
            print("进入正则化兜底求解...")
        for idx, method in enumerate(method_order, start=1):
            method_ok = False
            for weight_value in weight_candidates:
                weight_text = "auto" if weight_value is None else f"{weight_value:g}"
                try:
                    if show_progress:
                        print(f"正则化稳态：尝试 {idx}/{len(method_order)}，方法={method}, weight={weight_text}")
                    rho_ss = run_ss_with_progress(
                        h,
                        c_ops_reg,
                        method,
                        prefix=f"steadystate_reg({method},w={weight_text})",
                        weight_value=weight_value,
                    )
                    used_method = f"{method}(w={weight_text})+regularized"
                    method_ok = True
                    break
                except Exception as exc:
                    last_error = exc
                    if show_progress:
                        print(f"正则化方法 {method} 在 weight={weight_text} 失败：{type(exc).__name__}: {str(exc)[:120]}")
            if method_ok:
                break

    if rho_ss is None:
        if show_progress:
            print("进入代数兜底求解（Liouvillian 最小二乘）...")

        l_op = liouvillian(h, c_ops_reg)
        l_csr = l_op.data.as_scipy().tocsr()
        dim = h.shape[0]

        trace_vec = np.eye(dim, dtype=complex).reshape(dim * dim, order="F")
        tr_row = sparse.csr_matrix(trace_vec.reshape(1, -1))

        tr_weight = max(5.0, float(np.mean(np.abs(l_csr.data))) if l_csr.nnz > 0 else 5.0)
        a_aug = sparse.vstack([l_csr, tr_weight * tr_row], format="csr")
        b_aug = np.zeros(dim * dim + 1, dtype=complex)
        b_aug[-1] = tr_weight

        result = {"x": None, "err": None}
        done = threading.Event()

        def ls_worker():
            try:
                ls_out = lsqr(a_aug, b_aug, atol=1e-10, btol=1e-10, iter_lim=8000)
                result["x"] = ls_out[0]
            except Exception as exc:
                result["err"] = exc
            finally:
                done.set()

        th = threading.Thread(target=ls_worker, daemon=True)
        th.start()

        if show_progress:
            width = 24
            tick = 0
            start = time.perf_counter()
            while not done.wait(0.4):
                pos = tick % width
                bar = ["·"] * width
                bar[pos] = "█"
                elapsed = time.perf_counter() - start
                print(f"\r代数兜底进度 [{''.join(bar)}] {elapsed:6.1f}s", end="", flush=True)
                tick += 1
            elapsed = time.perf_counter() - start
            print(f"\r代数兜底进度 [完成{'█' * (width - 2)}] {elapsed:6.1f}s")
        else:
            done.wait()

        if result["err"] is not None:
            last_error = result["err"]
        elif result["x"] is not None:
            rho_mat = result["x"].reshape((dim, dim), order="F")
            rho_q = Qobj(rho_mat, dims=h.dims)
            rho_q = 0.5 * (rho_q + rho_q.dag())
            tr = rho_q.tr()
            if abs(tr) > 1e-14:
                rho_q = rho_q / tr
                rho_ss = rho_q
                used_method = "regularized-liouvillian-lsqr-fallback"

    if rho_ss is None:
        raise RuntimeError(f"steadystate 求解失败（已尝试 {method_order} 及正则化兜底）: {last_error}")

    n_ss = float(expect(n_tot_op, rho_ss))
    pe_ss = float(expect(pe_op, rho_ss))
    opposite_scattering_rate_rad = 0.0
    for c_op in c_ops_opp:
        opposite_scattering_rate_rad += float(np.real(expect(c_op.dag() * c_op, rho_ss)))
    cross_scattering_rate_rad = 0.0
    for c_op in c_ops_cross:
        cross_scattering_rate_rad += float(np.real(expect(c_op.dag() * c_op, rho_ss)))
    pe_scattering_rate_rad = pe_ss * GAMMA_D1_RAD
    total_scattering_rate_rad = pe_scattering_rate_rad + opposite_scattering_rate_rad + cross_scattering_rate_rad
    rho_mot_ss = rho_ss.ptrace(1)
    motional_populations = np.real(np.diag(rho_mot_ss.full()))
    motional_populations = np.clip(motional_populations, 0.0, None)
    pop_sum = float(np.sum(motional_populations))
    if pop_sum > 0.0:
        motional_populations = motional_populations / pop_sum
    p_top = float(motional_populations[-1]) if len(motional_populations) else 0.0
    p_tail_last2 = float(np.sum(motional_populations[-2:])) if len(motional_populations) >= 2 else p_top

    return {
        "ratio": float(ratio),
        "n_ss": n_ss,
        "pe_ss": pe_ss,
        "pe_scattering_rate_kHz": float(pe_scattering_rate_rad / 1e3),
        "opposite_scattering_rate_kHz": float(opposite_scattering_rate_rad / 1e3),
        "cross_scattering_rate_kHz": float(cross_scattering_rate_rad / 1e3),
        "total_scattering_rate_kHz": float(total_scattering_rate_rad / 1e3),
        "p_top": p_top,
        "p_tail_last2": p_tail_last2,
        "motional_populations": motional_populations,
        "solver_used": used_method,
        "delta_raman_kHz": float(delta_raman_rad / (2 * np.pi * 1e3)),
        "delta_phi_pol_pi": float(delta_phi_pol / np.pi),
        "omega_c_MHz": float(omega_c / (2 * np.pi * 1e6)),
        "omega_r_MHz": float(omega_r / (2 * np.pi * 1e6)),
    }


# ==========================================
# 模块三：宏观装载统计（速率方程）
# ==========================================
def loading_rate_equations(_t, y, r_load, beta1, beta2, gamma_bg):
    p0, p1, p2 = y
    dp0 = -r_load * p0 + beta2 * p2 + gamma_bg * p1
    dp1 = r_load * p0 - r_load * p1 - gamma_bg * p1 + (beta1 + 2.0 * gamma_bg) * p2
    dp2 = r_load * p1 - (beta1 + beta2 + 2.0 * gamma_bg) * p2
    return [dp0, dp1, dp2]


def loading_module(inp: UserInputs, dp: DerivedParams, beta1: float, beta2: float):
    y0 = [1.0, 0.0, 0.0]
    t_end = inp.loading_t_end_s
    t_span = (0.0, t_end)
    t_eval = np.linspace(t_span[0], t_span[1], 800)
    sol = solve_ivp(
        loading_rate_equations,
        t_span,
        y0,
        args=(dp.mot_loading_rate_Hz, beta1, beta2, dp.gamma_bg_Hz),
        t_eval=t_eval,
    )
    return sol.t, sol.y


def ensure_output_dir(output_dir) -> Path:
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def make_run_output_dir(root_dir, run_label: str) -> tuple[Path, str]:
    root = ensure_output_dir(root_dir)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    safe_label = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in run_label)
    run_dir = root / f"{safe_label}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir, timestamp


def save_csv(path: Path, header: str, data: np.ndarray):
    np.savetxt(path, data, delimiter=",", header=header, comments="", fmt="%.12g")


def parse_scan_values(text: str | None, default_values: np.ndarray) -> np.ndarray:
    if text is None or text.strip() == "":
        return np.asarray(default_values, dtype=float)

    if ":" in text:
        parts = [p.strip() for p in text.split(":") if p.strip()]
        if len(parts) != 3:
            raise ValueError("Range scan values must use start:stop:count.")
        start, stop, count = [float(p) for p in parts]
        return np.linspace(start, stop, int(count))
    parts = [p.strip() for p in text.split(",") if p.strip()]
    return np.asarray([float(p) for p in parts], dtype=float)


def run_pipeline(
    inp: UserInputs,
    show_plots: bool = False,
    n_fock: int = 5,
    t_end_s: float = 2.0e-3,
    n_t: int = 80,
    output_dir=None,
    output_tag: str = "",
):
    dp = build_derived_params(inp)
    try:
        cool = cooling_module(
            inp,
            dp,
            n_fock=n_fock,
            t_end_s=t_end_s,
            n_t=n_t,
            show_progress=True,
            robust_integrator=False,
        )
    except Exception as exc:
        if "Excess work done" not in str(exc):
            raise
        print("\n[提示] 默认积分器在该参数点失败，自动切换到稳健积分模式重试...")
        cool = cooling_module(
            inp,
            dp,
            n_fock=n_fock,
            t_end_s=t_end_s,
            n_t=n_t,
            show_progress=True,
            robust_integrator=True,
            max_runtime_s=90.0,
        )
    # cool = cooling_module(inp, dp, n_fock=5, t_end_s=8.0e-3, n_t=150)
    coll = collision_module(inp, dp, cool["tau_cool_s"], cool_result=cool)
    t_load, probs = loading_module(inp, dp, coll["beta1_Hz"], coll["beta2_Hz"])
    p0, p1, p2 = probs

    out_dir = ensure_output_dir(output_dir) if output_dir is not None else None
    if out_dir is not None:
        tag = f"_{output_tag}" if output_tag else ""
        save_csv(
            out_dir / f"cooling_time_curve{tag}.csv",
            "t_s,nbar,nbar_smooth,pe",
            np.column_stack([cool["t_s"], cool["phonons"], cool["phonons_smooth"], cool["pe"]]),
        )
        save_csv(
            out_dir / f"loading_time_curve{tag}.csv",
            "t_s,P0_empty,P1_single,P2_double",
            np.column_stack([t_load, p0, p1, p2]),
        )
    if plt is not None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        if inp.show_envelope:
            ax1.plot(cool["t_s"] * 1e3, cool["phonons"], lw=1.2, alpha=0.4, color="tab:blue", label="原始 <n>")
            ax1.plot(cool["t_s"] * 1e3, cool["phonons_smooth"], lw=2.0, color="tab:blue", label="平滑包络")
        else:
            ax1.plot(cool["t_s"] * 1e3, cool["phonons"], lw=2.0, color="tab:blue", label="原始 <n>")
        ax1.axvline(cool["tau_cool_s"] * 1e3, color="tab:red", ls="--", label=r"$\tau_{cool}$")
        ax1.set_xlabel("时间 (ms)")
        ax1.set_ylabel(r"$\langle n \rangle$")
        ax1.set_title(r"模块2：$\Lambda$-AGM 冷却")
        ax1.grid(alpha=0.35)
        ax1.legend()

        ax2.plot(t_load, p0, "--", lw=2, label="P0（空阱）")
        ax2.plot(t_load, p1, "-", lw=2, label="P1（单原子）")
        ax2.plot(t_load, p2, ":", lw=2, label="P2（双原子）")
        ax2.set_xlabel("时间 (s)")
        ax2.set_ylabel("概率")
        ax2.set_title("模块3：装载统计")
        ax2.grid(alpha=0.35)
        ax2.legend()

        plt.tight_layout()
        if out_dir is not None:
            fig.savefig(out_dir / f"overview_cooling_loading{tag}.png", dpi=200)
        if show_plots:
            plt.show()
        else:
            plt.close(fig)
    elif show_plots:
        print("[提示] matplotlib 不可用，跳过图形显示。")

    print("=" * 64)
    print("输入参数（关键参数）")
    print(f"trap_wavelength_nm          = {inp.trap_wavelength_nm:.2f}")
    print(f"trap_depth_mK               = {inp.trap_depth_mK:.3f}")
    print(f"trap_waist_um               = {inp.trap_waist_um:.3f}")
    print(f"blue_detuning_MHz           = {inp.blue_detuning_MHz:.3f}")
    print(f"differential_shift_kHz      = {inp.differential_light_shift_kHz:.3f}")
    print(f"eom_offset_kHz              = {inp.eom_offset_kHz:.3f}")
    print(f"cooling_total_intensity     = {inp.cooling_total_intensity_mW_cm2:.3f} mW/cm^2")
    print(f"sideband_ratio Is/Ic        = {inp.sideband_ratio:.3f}")
    print(f"polarization_phase/pi       = {inp.polarization_phase_pi:.3f}")
    print(f"lz_coupling_MHz             = {inp.lz_coupling_MHz:.3f}")
    print(f"mot_loading_rate_Hz (R)     = {inp.mot_loading_rate_Hz:.3f}")
    print(f"vacuum_lifetime_s           = {inp.vacuum_lifetime_s:.1f}")
    print(f"loading_t_end_s             = {inp.loading_t_end_s:.3f}")
    print(f"initial_nbar                = {inp.initial_nbar:.2f}")
    print(f"show_envelope               = {inp.show_envelope}")
    print("-")
    print("自动计算参数")
    print(f"trap_freq                   = {dp.trap_freq_Hz/1e3:.2f} kHz")
    print(f"eta (Lamb-Dicke)            = {dp.eta:.3f}")
    print(f"D1 I_sat                    = {dp.d1_saturation_intensity_mW_cm2:.3f} mW/cm^2")
    print(f"Ic, Is                      = {dp.carrier_intensity_mW_cm2:.3f}, {dp.sideband_intensity_mW_cm2:.3f} mW/cm^2")
    print(f"EOM microwave               = {dp.eom_microwave_Hz/1e9:.9f} GHz")
    print(f"EOM beta (for Is/Ic={dp.sideband_ratio:.3f}) = {dp.beta_eom:.4f} rad")
    _, _, delta_phi_pol = cooling_polarization_phases(inp, dp)
    print(f"polarization delta_phi/pi   = {delta_phi_pol/np.pi:.6f}")
    print(f"Omega_c/2pi, Omega_s/2pi    = {cool['omega_c']/(2*np.pi*1e6):.3f}, {cool['omega_r']/(2*np.pi*1e6):.3f} MHz")
    print(f"MOT loading rate R          = {dp.mot_loading_rate_Hz:.3f} Hz")

    differential_shift_Hz = inp.differential_light_shift_kHz * 1e3
    effective_delta_2u_hz = 2.0 * dp.trap_depth_Hz
    lab_delta_2u_hz = effective_delta_2u_hz + differential_shift_Hz
    print(f"E=2U对应有效蓝失谐阈值      = {effective_delta_2u_hz/1e6:.3f} MHz")
    print(f"换算到输入单光子失谐阈值      = {lab_delta_2u_hz/1e6:.3f} MHz")
    print(f"(U/h={dp.trap_depth_Hz/1e6:.3f} MHz, 差分光频移={differential_shift_Hz/1e6:.3f} MHz)")
    print("判据：当 Δ_eff = Δ - δ_trap > 2U/h 时，更容易进入双原子同时逃逸区")
    print("-")
    print("模块输出")
    print(f"delta_R/2pi                 = {cool['delta_raman_rad']/(2*np.pi):.2f} Hz")
    print(f"delta_R/2pi                 = {cool['delta_raman_rad']/(2*np.pi*1e3):.3f} kHz")
    print(f"tau_cool                    = {cool['tau_cool_s']*1e3:.3f} ms")
    print(f"碰撞工作区                  = {coll['regime']}")
    print(f"E_single/h                  = {coll['e_single_Hz']/1e6:.3f} MHz")
    print(f"T_cold                      = {coll['temperature_cold_K']*1e6:.2f} uK")
    print(f"T_hot                       = {coll['temperature_hot_K']*1e6:.2f} uK")
    print(f"K2                          = {coll['k2_m3_per_s']:.3e} m^3/s") 
    print(f"Gamma_enc                   = {coll['gamma_enc_Hz']:.3f} Hz")
    print(f"Gamma_scatt_hot             = {coll['gamma_scatt_hot_Hz']:.3f} Hz")
    print(f"Gamma_scatt_cold            = {coll['gamma_scatt_cold_Hz']:.3f} Hz")
    print(f"P_direct                    = {coll['p_direct']:.3f}")
    print(f"P_second_escape             = {coll['p_second_escape']:.3f}")
    print(f"Gamma_evap                  = {coll['gamma_evap_Hz']:.3f} Hz")
    print(f"beta1 (2->1)                = {coll['beta1_Hz']:.3f} Hz")
    print(f"beta2 (2->0)                = {coll['beta2_Hz']:.3f} Hz")
    print(f"稳态单原子装载率 P1         = {p1[-1]*100:.2f} %")
    print(f"稳态空阱率 P0               = {p0[-1]*100:.2f} %")

    p2_vs_p0_ratio = p2[-1] / max(p0[-1], 1e-12)
    print(f"稳态 P2/P0                  = {p2_vs_p0_ratio:.3f}")
    print("诊断判据：若 beta2 < R，则可能出现 P2 > P0（双原子周转慢于装载）")
    print(f"当前 beta2/R                = {coll['beta2_Hz']/max(dp.mot_loading_rate_Hz, 1e-12):.3f}")
    if p2[-1] > p0[-1]:
        print("警告：当前参数下 P2 > P0，这通常是装载过快或双体损失过慢导致。")
    if out_dir is not None:
        print(f"结果文件已保存到             = {out_dir}")
    print("=" * 64)

    return {
        "derived": dp,
        "cooling": cool,
        "collision": coll,
        "loading_t": t_load,
        "loading_probs": probs,
        "output_dir": out_dir,
    }


def scan_ratio_effect(inp: UserInputs, ratio_list=None, show_plots: bool = True):
    if ratio_list is None:
        ratio_list = np.linspace(0.03, 0.22, 6)

    tau_ms_list = []
    n_final_list = []
    pe_mean_list = []
    p1_ss_list = []

    total = len(ratio_list)
    print("开始扫描光强比（逐点主方程求解）...")

    for idx_scan, r in enumerate(ratio_list, start=1):
        inp_scan = replace(inp, sideband_ratio=float(r))
        dp = build_derived_params(inp_scan)

        cool = cooling_module(inp_scan, dp, n_fock=4, t_end_s=0.9e-3, n_t=75, show_progress=False)
        coll = collision_module(inp_scan, dp, cool["tau_cool_s"], cool_result=cool)
        _, probs = loading_module(inp_scan, dp, coll["beta1_Hz"], coll["beta2_Hz"])

        tau_ms_list.append(cool["tau_cool_s"] * 1e3)
        n_final_list.append(cool["n_final"])
        pe_mean_list.append(cool["pe_mean"])
        p1_ss_list.append(probs[1, -1] * 100.0)
        print_progress("扫描进度", idx_scan, total)

    if plt is not None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        ax1.plot(ratio_list, tau_ms_list, "o-", lw=2, label=r"$\tau_{cool}$")
        ax1.plot(ratio_list, n_final_list, "s--", lw=2, label=r"稳态 $\langle n\rangle$")
        ax1.plot(ratio_list, pe_mean_list, "d:", lw=2, label=r"暗态漏光 $\langle P_e\rangle$")
        ax1.set_xlabel("边带/载波光强比 $I_s/I_c$")
        ax1.set_ylabel("冷却指标")
        ax1.set_title("光强比对冷却效果的影响")
        ax1.grid(alpha=0.35)
        ax1.legend()

        ax2.plot(ratio_list, p1_ss_list, "o-", lw=2, color="tab:orange", label="稳态 P1")
        ax2.set_xlabel("边带/载波光强比 $I_s/I_c$")
        ax2.set_ylabel("稳态单原子装载率 P1 (%)")
        ax2.set_title("光强比对装载率的影响")
        ax2.grid(alpha=0.35)
        ax2.legend()

        plt.tight_layout()
        if show_plots:
            plt.show()
        else:
            plt.close(fig)

    best_idx = int(np.argmax(p1_ss_list))
    print("-" * 64)
    print("光强比扫描结果")
    print("说明：扫描图由 mF 分辨主方程逐点计算（为提速，使用较小Fock截断）")
    print(f"最佳 I_s/I_c               = {ratio_list[best_idx]:.3f}")
    print(f"对应稳态装载率 P1          = {p1_ss_list[best_idx]:.2f} %")
    print(f"对应冷却时间常数           = {tau_ms_list[best_idx]:.3f} ms")
    print("-" * 64)


def scan_steady_cooling_1d(
    inp: UserInputs,
    param_name: str,
    values=None,
    n_fock: int = 4,
    solver_method: str = "power",
    output_dir=None,
    output_tag: str = "",
    show_plots: bool = False,
):
    if param_name not in {"sideband_ratio", "blue_detuning_MHz"}:
        raise ValueError("param_name must be 'sideband_ratio' or 'blue_detuning_MHz'.")

    if values is None:
        if param_name == "sideband_ratio":
            values = np.linspace(0.03, 0.22, 6)
        else:
            values = np.linspace(10.0, 80.0, 8)
    values = np.asarray(values, dtype=float)

    rows = []
    total = len(values)
    print(f"开始稳态冷却单维扫参: {param_name}")
    for idx, value in enumerate(values, start=1):
        inp_scan = replace(inp, **{param_name: float(value)})
        ratio = inp_scan.sideband_ratio
        result = steady_state_n_for_ratio(
            inp_scan,
            ratio=ratio,
            n_fock=n_fock,
            solver_method=solver_method,
            show_progress=False,
        )
        sideband_fraction = ratio / (1.0 + ratio)
        scattering_rate_kHz = result["pe_ss"] * GAMMA_D1_RAD / 1e3
        rows.append({
            "param_value": float(value),
            "blue_detuning_MHz": float(inp_scan.blue_detuning_MHz),
            "sideband_ratio_Is_over_Ic": float(ratio),
            "sideband_fraction_total": float(sideband_fraction),
            "n_fock": int(n_fock),
            "n_ss": float(result["n_ss"]),
            "pe_ss": float(result["pe_ss"]),
            "scattering_rate_kHz": float(scattering_rate_kHz),
            "opposite_scattering_rate_kHz": float(result["opposite_scattering_rate_kHz"]),
            "cross_scattering_rate_kHz": float(result["cross_scattering_rate_kHz"]),
            "total_scattering_rate_kHz": float(result["total_scattering_rate_kHz"]),
            "p_top": float(result["p_top"]),
            "p_tail_last2": float(result["p_tail_last2"]),
            "delta_raman_kHz": float(result["delta_raman_kHz"]),
            "delta_phi_pol_pi": float(result["delta_phi_pol_pi"]),
            "omega_c_MHz": float(result["omega_c_MHz"]),
            "omega_s_MHz": float(result["omega_r_MHz"]),
            "solver_used": str(result["solver_used"]),
        })
        print_progress("稳态扫参进度", idx, total)

    arr = np.asarray(
        [
            [
                row["param_value"],
                row["blue_detuning_MHz"],
                row["sideband_ratio_Is_over_Ic"],
                row["sideband_fraction_total"],
                row["n_fock"],
                row["n_ss"],
                row["pe_ss"],
                row["scattering_rate_kHz"],
                row["opposite_scattering_rate_kHz"],
                row["cross_scattering_rate_kHz"],
                row["total_scattering_rate_kHz"],
                row["p_top"],
                row["p_tail_last2"],
                row["delta_raman_kHz"],
                row["delta_phi_pol_pi"],
                row["omega_c_MHz"],
                row["omega_s_MHz"],
            ]
            for row in rows
        ],
        dtype=float,
    )
    best_idx = int(np.argmin(arr[:, 5]))

    out_dir = ensure_output_dir(output_dir) if output_dir is not None else None
    if out_dir is not None:
        tag = f"_{output_tag}" if output_tag else ""
        stem = f"steady_scan_{param_name}{tag}"
        csv_path = out_dir / f"{stem}.csv"
        fieldnames = [
            "param_value",
            "blue_detuning_MHz",
            "sideband_ratio_Is_over_Ic",
            "sideband_fraction_total",
            "n_fock",
            "n_ss",
            "pe_ss",
            "scattering_rate_kHz",
            "opposite_scattering_rate_kHz",
            "cross_scattering_rate_kHz",
            "total_scattering_rate_kHz",
            "p_top",
            "p_tail_last2",
            "delta_raman_kHz",
            "delta_phi_pol_pi",
            "omega_c_MHz",
            "omega_s_MHz",
            "solver_used",
        ]
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    if plt is not None:
        fig, ax1 = plt.subplots(figsize=(7, 5))
        ax1.plot(arr[:, 0], arr[:, 5], "o-", lw=2, color="tab:blue", label=r"$\langle n\rangle_{ss}$")
        ax1.set_xlabel(param_name)
        ax1.set_ylabel(r"$\langle n\rangle_{ss}$")
        ax1.grid(alpha=0.35)
        ax2 = ax1.twinx()
        ax2.plot(arr[:, 0], arr[:, 6], "s--", lw=1.8, color="tab:red", label=r"$P_e$")
        ax2.set_ylabel(r"$P_e$")
        fig.suptitle(f"稳态冷却扫参: {param_name}")
        lines = ax1.get_lines() + ax2.get_lines()
        labels = [line.get_label() for line in lines]
        ax1.legend(lines, labels, loc="best")
        plt.tight_layout()
        if out_dir is not None:
            fig.savefig(out_dir / f"{stem}.png", dpi=200)
        if show_plots:
            plt.show()
        else:
            plt.close(fig)
    elif show_plots:
        print("[提示] matplotlib 不可用，跳过图形显示。")

    print("-" * 64)
    print(f"稳态扫参结果: {param_name}")
    print(f"最佳 {param_name}             = {arr[best_idx, 0]:.6g}")
    print(f"最小 n_ss                    = {arr[best_idx, 5]:.6g}")
    print(f"对应 pe_ss                   = {arr[best_idx, 6]:.3e}")
    if out_dir is not None:
        print(f"结果文件已保存到             = {out_dir}")
    print("-" * 64)

    return {
        "param_name": param_name,
        "values": values,
        "table": arr,
        "best_idx": best_idx,
        "output_dir": out_dir,
    }


def scan_two_photon_detuning_effect(
    inp: UserInputs,
    detuning_kHz_list=None,
    n_fock: int = 8,
    t_end_s: float = 1.2e-3,
    n_t: int = 80,
    show_plots: bool = True,
):
    """
    扫描两光子失谐 δ_2ph（kHz），评估冷却速度与末态声子数。

    说明：
    - 这里定义 δ_2ph/2π (kHz) = eom_offset_kHz - differential_light_shift_kHz
    - 扫描时保持其它参数不变，仅改变 eom_offset_kHz 来实现指定 δ_2ph
    """
    if detuning_kHz_list is None:
        detuning_kHz_list = np.linspace(-120.0, 120.0, 9)

    detuning_kHz_list = np.asarray(detuning_kHz_list, dtype=float)
    total = len(detuning_kHz_list)

    tau_ms_list = []
    n0_list = []
    n_final_list = []
    gain_list = []
    delta_actual_kHz_list = []

    print("开始扫描两光子失谐（逐点主方程求解）...")

    for idx, d2ph_kHz in enumerate(detuning_kHz_list, start=1):
        inp_scan = replace(inp, eom_offset_kHz=inp.differential_light_shift_kHz + float(d2ph_kHz))
        dp_scan = build_derived_params(inp_scan)
        cool = cooling_module(inp_scan, dp_scan, n_fock=n_fock, t_end_s=t_end_s, n_t=n_t, show_progress=False)

        n0 = float(cool["phonons"][0])
        nf = float(cool["n_final"])
        gain = n0 - nf
        delta_actual_kHz = float(cool["delta_raman_rad"] / (2 * np.pi * 1e3))

        tau_ms_list.append(cool["tau_cool_s"] * 1e3)
        n0_list.append(n0)
        n_final_list.append(nf)
        gain_list.append(gain)
        delta_actual_kHz_list.append(delta_actual_kHz)
        print_progress("两光子失谐扫描进度", idx, total)

    tau_ms_arr = np.asarray(tau_ms_list)
    gain_arr = np.asarray(gain_list)
    n_final_arr = np.asarray(n_final_list)
    delta_actual_arr = np.asarray(delta_actual_kHz_list)

    cooling_mask = gain_arr > 0.0
    if np.any(cooling_mask):
        idx_local = int(np.argmin(tau_ms_arr[cooling_mask]))
        best_idx = int(np.where(cooling_mask)[0][idx_local])
        best_reason = "在确实降温(gain>0)的点中，tau_cool 最小"
    else:
        best_idx = int(np.argmax(gain_arr))
        best_reason = "扫描区间内无净降温，选取加热最弱点(gain 最大)"

    if show_plots and plt is not None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        ax1.plot(detuning_kHz_list, tau_ms_arr, "o-", lw=2, label=r"$\tau_{cool}$ (ms)")
        ax1.plot(detuning_kHz_list, n_final_arr, "s--", lw=2, label=r"末态 $\langle n\rangle$")
        ax1.set_xlabel(r"设定两光子失谐 $\delta_{2ph}/2\pi$ (kHz)")
        ax1.set_ylabel("冷却指标")
        ax1.set_title("两光子失谐对冷却速度的影响")
        ax1.grid(alpha=0.35)
        ax1.legend()

        ax2.plot(detuning_kHz_list, gain_arr, "d-", lw=2, color="tab:green", label=r"$\Delta n=n(0)-n(t_{end})$")
        ax2.axhline(0.0, color="tab:red", ls="--", lw=1.5)
        ax2.set_xlabel(r"设定两光子失谐 $\delta_{2ph}/2\pi$ (kHz)")
        ax2.set_ylabel("净冷却量")
        ax2.set_title("两光子失谐对净冷却量的影响")
        ax2.grid(alpha=0.35)
        ax2.legend()

        plt.tight_layout()
        plt.show()
    elif show_plots:
        print("[提示] matplotlib 不可用，跳过图形显示。")

    print("-" * 64)
    print("两光子失谐扫描结果")
    print(f"最佳设定失谐             = {detuning_kHz_list[best_idx]:.2f} kHz")
    print(f"对应实际 delta_R/2pi      = {delta_actual_arr[best_idx]:.3f} kHz")
    print(f"对应 tau_cool             = {tau_ms_arr[best_idx]:.3f} ms")
    print(f"对应 n(0)->n(end)         = {n0_list[best_idx]:.3f} -> {n_final_arr[best_idx]:.3f}")
    print(f"对应净冷却量 Delta n      = {gain_arr[best_idx]:.3f}")
    print(f"判据说明                  = {best_reason}")
    print("-" * 64)

    return {
        "detuning_kHz": detuning_kHz_list,
        "delta_actual_kHz": delta_actual_arr,
        "tau_ms": tau_ms_arr,
        "n0": np.asarray(n0_list),
        "n_final": n_final_arr,
        "gain": gain_arr,
        "best_idx": best_idx,
        "best_setting_kHz": float(detuning_kHz_list[best_idx]),
    }


def quick_self_test():
    """Run a short end-to-end numerical check that finishes quickly."""
    inp = UserInputs()
    dp = build_derived_params(inp)

    max_norm_error = 0.0
    for Fp in (1, 2):
        for mp in range(-Fp, Fp + 1):
            branch_sum = 0.0
            for F in (1, 2):
                for m in range(-F, F + 1):
                    for q in (-1, 0, 1):
                        branch_sum += abs(cgc_weight(F, m, Fp, mp, q)) ** 2
            max_norm_error = max(max_norm_error, abs(branch_sum - 1.0))

    cool = cooling_module(inp, dp, n_fock=2, t_end_s=1.0e-5, n_t=5, show_progress=False)
    coll = collision_module(inp, dp, cool["tau_cool_s"], cool_result=cool)
    _, probs = loading_module(inp, dp, coll["beta1_Hz"], coll["beta2_Hz"])
    prob_sum_error = abs(float(np.sum(probs[:, -1])) - 1.0)

    print("=" * 64)
    print("快速自检完成")
    print(f"CG 分支归一化最大误差       = {max_norm_error:.3e}")
    print(f"trap_freq                   = {dp.trap_freq_Hz/1e3:.2f} kHz")
    print(f"eta                         = {dp.eta:.3f}")
    print(f"D1 I_sat                    = {dp.d1_saturation_intensity_mW_cm2:.3f} mW/cm^2")
    print(f"Ic, Is                      = {dp.carrier_intensity_mW_cm2:.3f}, {dp.sideband_intensity_mW_cm2:.3f} mW/cm^2")
    print(f"Omega_c/2pi, Omega_s/2pi    = {cool['omega_c']/(2*np.pi*1e6):.3f}, {cool['omega_r']/(2*np.pi*1e6):.3f} MHz")
    print(f"quick n_final               = {cool['n_final']:.6f}")
    print(f"quick tau_cool              = {cool['tau_cool_s']*1e6:.3f} us")
    print(f"quick pe_mean               = {cool['pe_mean']:.3e}")
    print(f"Gamma_scatt_hot             = {coll['gamma_scatt_hot_Hz']:.3f} Hz")
    print(f"beta1                       = {coll['beta1_Hz']:.3f} Hz")
    print(f"beta2                       = {coll['beta2_Hz']:.3f} Hz")
    print(f"loading prob sum error      = {prob_sum_error:.3e}")
    print("=" * 64)

    if max_norm_error > 1e-10:
        raise RuntimeError("CG normalization self-test failed.")
    if prob_sum_error > 1e-7:
        raise RuntimeError("Loading probability conservation self-test failed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rb87 D1 lambda-enhanced grey molasses simulation")
    parser.add_argument(
        "--mode",
        choices=("quick", "medium", "full", "paper"),
        default="quick",
        help="run preset: quick self-test, medium trend run, full baseline, or slower paper-quality run",
    )
    parser.add_argument("--full", action="store_true", help="alias for --mode full")
    parser.add_argument("--scan-ratio", action="store_true", help="run the sideband/carrier ratio scan after --full")
    parser.add_argument(
        "--scan-cooling",
        choices=("sideband_ratio", "blue_detuning_MHz"),
        help="run a 1D steady-state cooling scan for the selected parameter",
    )
    parser.add_argument(
        "--scan-values",
        help="scan values as comma list, e.g. 0.03,0.08,0.15, or range start:stop:count",
    )
    parser.add_argument("--scan-n-fock", type=int, default=4, help="Fock cutoff for steady-state scans")
    parser.add_argument(
        "--scan-solver",
        default="power",
        choices=(
            "power",
            "direct",
            "svd",
            "eigen",
            "lsqr",
            "physical-lsqr",
            "regularized-lsqr",
        ),
        help="steady-state solver for --scan-cooling; lsqr means physical Liouvillian LSQR without regularization",
    )
    parser.add_argument("--diagnose-internal", action="store_true", help="run internal dark-state and AC Stark diagnostics")
    parser.add_argument(
        "--include-opposite-sideband",
        action="store_true",
        help="include the off-resonant opposite first-order EOM sideband in steady-state scans",
    )
    parser.add_argument(
        "--include-cross-coupling",
        action="store_true",
        help="include off-resonant carrier/sideband cross-coupling to the wrong ground hyperfine state",
    )
    default_results_dir = Path(__file__).resolve().parent / "results"
    parser.add_argument("--output-dir", default=str(default_results_dir), help="root directory for timestamped result folders")
    parser.add_argument("--show-plots", action="store_true", help="display matplotlib windows")
    args = parser.parse_args()

    user_inputs = UserInputs()
    if args.include_opposite_sideband:
        user_inputs = replace(user_inputs, include_opposite_eom_sideband=True)
    if args.include_cross_coupling:
        user_inputs = replace(user_inputs, include_off_resonant_cross_coupling=True)
    mode = "full" if args.full else args.mode
    presets = {
        "medium": {"n_fock": 4, "t_end_s": 5.0e-4, "n_t": 50},
        "full": {"n_fock": 5, "t_end_s": 2.0e-3, "n_t": 80},
        "paper": {"n_fock": 8, "t_end_s": 2.0e-3, "n_t": 120},
    }

    output_root = ensure_output_dir(args.output_dir)

    if args.diagnose_internal:
        out_dir, tag = make_run_output_dir(output_root, "diagnose_internal")
        print(f"输出目录: {out_dir}")
        internal_dark_state_diagnostics(user_inputs, output_dir=out_dir, output_tag=tag, show_progress=True)

    if mode == "quick" and args.scan_cooling is None and not args.diagnose_internal:
        quick_self_test()
    else:
        if mode != "quick":
            print(f"运行模式: {mode} -> {presets[mode]}")
            out_dir, tag = make_run_output_dir(output_root, f"time_{mode}")
            print(f"输出目录: {out_dir}")
            run_pipeline(user_inputs, show_plots=args.show_plots, output_dir=out_dir, output_tag=tag, **presets[mode])
        if args.scan_ratio:
            print("[提示] --scan-ratio 是旧的时间演化扫描，不保存 CSV；稳态冷却扫参请用 --scan-cooling sideband_ratio。")
            scan_ratio_effect(user_inputs, show_plots=args.show_plots)
        if args.scan_cooling is not None:
            if args.scan_cooling == "sideband_ratio":
                defaults = np.linspace(0.03, 0.22, 6)
            else:
                defaults = np.linspace(10.0, 80.0, 8)
            values = parse_scan_values(args.scan_values, defaults)
            out_dir, tag = make_run_output_dir(output_root, f"steady_scan_{args.scan_cooling}")
            print(f"输出目录: {out_dir}")
            scan_steady_cooling_1d(
                user_inputs,
                args.scan_cooling,
                values=values,
                n_fock=args.scan_n_fock,
                solver_method=args.scan_solver,
                output_dir=out_dir,
                output_tag=tag,
                show_plots=args.show_plots,
            )
