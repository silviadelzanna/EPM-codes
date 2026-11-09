"""
raman_calibration.py
=====================
Calibrates the three sub-spectra (RBM, DGCC, 2D) of a laser measurement
against their respective calibration lamp spectra using a LINEAR fit,
and merges them into one combined output file.

Folder structure:
    <SPECTRA_FOLDER>/<step>/        -> sample files for that step (pris / annealed / dw)
    <SPECTRA_FOLDER>/calib_lamps/   -> lamp calibration files (shared by all steps)

File identification:
    - CALIBRATION file : contains a lamp tag (ne / xe / ar / kr)
    - SAMPLE file      : contains NO lamp tag
    - region           : rbm / dgcc / 2d
    - laser            : e.g. 488nm

Example:
    Lucius_dw_rbm_xe_50x_gr600_slit300_15A_488nm_spectro80to470_15cycles_120s.csv
        -> calibration file for RBM region, Xe lamp
    Lucius_dw_rbm_50x_gr600_slit300_15A_488nm_spectro80to470_15cycles_120s.csv
        -> sample file for RBM region

All spectra are two-column files: Raman shift (cm-1), intensity.

Output (per step):
    - combined calibrated spectrum          -> <folder>/<step>/calibration/
    - peak detection plot  (1x3, wide)      -> <folder>/<step>/calibration/
    - calibration fit plot (2x3, wide)      -> <folder>/<step>/calibration/
    - before/after plot    (2x2, square)    -> <folder>/<step>/calibration/

Usage:
    python raman_calibration.py
    python raman_calibration.py --folder ./data
"""

import os
import re
import glob
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.optimize import curve_fit

# ═══════════════════════════════════════════════════════════════════
#  USER SETTINGS
# ═══════════════════════════════════════════════════════════════════

SPECTRA_FOLDER   = r"C:\Users\andre\ucloud\Documents\Samples\Lucius\LabRAM\coalesence_test"
FILE_EXTENSIONS  = (".csv", ".txt", ".dat")

DELIMITER        = "\t"
SKIP_HEADER      = 0
X_COLUMN         = 0            # Raman shift column
Y_COLUMN         = 1            # intensity column

REGIONS          = ['rbm']#,'g','2d']
LAMPS            = ["ne", "xe", "ar", "kr"]
LASER_LIST       = [487.986, 514.532, 568.188]    # supported lasers: 487.986, 514.532, 568.188, 633
STEP_LIST        = ['saph']#,'annealed','dw']

# Peak detection / matching
PEAK_HEIGHT_PERCENTILE = 10
PEAK_PROMINENCE = 20
PEAK_MIN_DISTANCE      = 3
MATCH_TOLERANCE_NM     = 0.5
FIT_ORDER              = 1      # LINEAR fit
FIT_WINDOW             = 0.1    # ± range around each peak for the Lorentzian fit (nm)

OUTPUT_SUFFIX     = "_COMBINED_CALIBRATED.csv"
CALIBRATION_DIR   = "calibration"       # output subfolder inside each step folder
CALIB_LAMPS_DIR   = "calib_lamps_saph"       # subfolder of SPECTRA_FOLDER with lamp files
SAVE_FIT_PLOTS    = False               # save plots as PNG
SHOW_FIT_PLOTS    = True                # show plots on screen
SEP_REGIONS       = True

OUTLIER_SIGMA   = 2     # rejection threshold: residual > OUTLIER_SIGMA * std
MAX_REJECT_ITER = 5     # safety limit on refit iterations


# ═══════════════════════════════════════════════════════════════════
#  Calibration lamp lines (MKS/Newport 6030-6033), nm
# ═══════════════════════════════════════════════════════════════════

FIT_WINDOW_CM      = 5  # was FIT_WINDOW (nm); e.g. ~0.3 nm ≈ 7–12 cm-1 depending on laser
MATCH_TOLERANCE_CM = 5   # was MATCH_TOLERANCE_NM; pick based on your dispersion

# LAMP_LINES = {
#     "kr": [366.53, 367.96, 377.34, 428.30, 431.96, 436.26, 437.61, 440.0,
#            442.52, 445.39, 446.37, 450.24, 549.09, 550.07, 556.22, 557.03,
#            558.04, 583.29, 587.09, 587.99, 599.39, 601.22, 605.61, 642.1,
#            645.63, 669.92, 681.31, 690.47, 722.41, 728.98, 742.55, 748.61,
#            758.74, 760.15, 768.52, 769.45, 774.68, 785.48, 791.34, 805.95,
#            810.44, 811.29, 819.01, 826.32, 828.11, 829.81, 850.89, 877.67,
#            892.87],
#     "ne": [336.99, 341.79, 344.77, 345.42, 346.66, 347.26, 349.81, 350.12,
#            351.52, 352.05, 359.35, 360.02, 363.37, 470.44, 492.32, 503.78,
#            508.04, 511.65, 514.5, 520.39, 533.08, 534.11, 534.33, 540.06,
#            564.96, 565.67, 574.83, 576.44, 580.45, 582.02, 585.25, 588.19,
#            590.25, 594.48, 597.55, 598.79, 603.0, 607.43, 609.62, 612.85,
#            614.31, 616.36, 618.22, 621.73, 626.65, 630.48, 631.81, 633.44,
#            638.3, 640.23, 650.65, 653.29, 659.9, 665.21, 667.83, 671.7,
#            692.95, 702.41, 703.24, 705.3, 705.91, 717.39, 724.52, 743.89,
#            747.24, 748.89, 753.58, 754.41, 794.32, 808.25, 813.64, 830.03,
#            837.76, 841.84, 849.54],
#     "ar": [355.43, 384.9, 404.44, 415.86, 416.4, 418.19, 419.1, 419.8,
#            420.07, 425.12, 425.94, 426.63, 427.22, 427.4, 430.01, 432.0,
#            433.36, 434.52, 641.63, 667.73, 675.28, 687.13, 693.77, 696.54,
#            703.03, 706.72, 714.7, 727.29, 737.21, 738.4, 750.39, 751.46,
#            763.51, 772.38, 794.82, 800.62, 801.48, 810.37, 811.53, 826.45,
#            840.82, 842.46],
#     "xe": [450.1, 452.47, 458.28, 462.43, 467.12, 469.7, 473.42, 480.7,
#            482.97, 484.33, 491.65, 502.83, 618.24, 646.97, 666.89, 672.8,
#            682.73, 688.22, 711.96, 728.53, 758.47, 764.2, 823.16, 828.01,
#            834.68, 840.92],
# }

NIST_LAMP_FILE = os.path.join(r'C:\Users\andre\ucloud\Documents\Coding\lamp_calibration', "NIST_lamp_eddited.csv")


def load_lamp_lines(path):
    """Load lamp lines from a 3-column headerless CSV:
    lamp name (ne, ar, ...), wavelength (nm), relative intensity.
    Returns {lamp: {"wl": array(nm), "int": array(rel. intensity)}}."""
    df = pd.read_csv(path, header=None, names=["lamp", "wl_nm", "rel_int"])

    # clean up
    df = df.dropna()
    df["lamp"]    = df["lamp"].astype(str).str.strip().str.lower()
    df["wl_nm"]   = pd.to_numeric(df["wl_nm"],   errors="coerce")
    df["rel_int"] = pd.to_numeric(df["rel_int"], errors="coerce")
    df = df.dropna()

    lamp_lines = {}
    for lamp, grp in df.groupby("lamp"):
        grp = grp.sort_values("wl_nm")
        lamp_lines[lamp] = {
            "wl":  grp["wl_nm"].to_numpy(),
            "int": grp["rel_int"].to_numpy(),
        }
        print(f"  📄 Loaded {len(grp)} {lamp.upper()} lines from "
              f"{os.path.basename(path)}")
    return lamp_lines


LAMP_LINES = load_lamp_lines(NIST_LAMP_FILE)

# ═══════════════════════════════════════════════════════════════════
#  Conversion helpers
# ═══════════════════════════════════════════════════════════════════

def raman_shift_to_wavelength(laser_nm, shift_cm):
    return 1e7 / (1e7 / laser_nm - np.asarray(shift_cm, dtype=float))


def wavelength_to_raman_shift(laser_nm, wl_nm):
    return 1e7 / laser_nm - 1e7 / np.asarray(wl_nm, dtype=float)


def lamp_lines_in_shift(lamp, laser_nm):
    """Known lamp emission lines converted to Raman shift (cm-1),
    together with their relative intensities.
    Returns (shift_cm, rel_int), both sorted by shift."""
    wl  = LAMP_LINES[lamp]["wl"]
    ri  = LAMP_LINES[lamp]["int"]
    cm  = wavelength_to_raman_shift(laser_nm, wl)
    srt = np.argsort(cm)
    return cm[srt], ri[srt]
# ═══════════════════════════════════════════════════════════════════
#  File discovery / parsing   (unchanged)
# ═══════════════════════════════════════════════════════════════════

def parse_filename(filename):
    """
    Extract region, lamp, step and laser wavelength from a filename.
    A file WITH a lamp tag is a calibration file.
    A file WITHOUT a lamp tag is a sample file.
    """
    name = os.path.basename(filename).lower()
    tokens = re.split(r"[_\-.]", name)

    region = next((r for r in REGIONS if r in tokens), None)
    lamp   = next((l for l in LAMPS   if l in tokens), None)
    step   = next((s for s in STEP_LIST if s in tokens), None)
    is_cal = lamp is not None            # <-- lamp tag = calibration file

    laser = None
    m = re.search(r"(\d+(?:\.\d+)?)nm", name)
    if m:
        laser = float(m.group(1))

    return {"path": filename, "region": region, "lamp": lamp,
            "laser": laser, "step": step, "is_cal": is_cal}


def _list_spectra(folder):
    files = []
    for ext in FILE_EXTENSIONS:
        files.extend(glob.glob(os.path.join(folder, f"*{ext}")))
    return files


def find_files(folder, laser_nm, step):
    step_folder = os.path.join(folder, step)
    lamp_folder = os.path.join(folder, CALIB_LAMPS_DIR)

    if not os.path.isdir(step_folder):
        print(f"  ⚠️  Step folder not found: {step_folder}")
    if not os.path.isdir(lamp_folder):
        print(f"  ⚠️  Lamp folder not found: {lamp_folder}")

    sample_parsed = [parse_filename(f) for f in _list_spectra(step_folder)]
    sample_parsed = [p for p in sample_parsed
                     if p["region"] is not None
                     and not p["is_cal"]
                     and (p["laser"] is None or abs(p["laser"] - laser_nm) < 0.5)
                     and OUTPUT_SUFFIX.lower() not in p["path"].lower()]

    cal_parsed = [parse_filename(f) for f in _list_spectra(lamp_folder)]
    cal_parsed = [p for p in cal_parsed
                  if p["region"] is not None
                  and p["is_cal"]
                  and (p["laser"] is None or abs(p["laser"] - laser_nm) < 0.5)]

    pairs = {}
    for region in REGIONS:
        cal    = next((p for p in cal_parsed    if p["region"] == region), None)
        sample = next((p for p in sample_parsed if p["region"] == region), None)
        pairs[region] = {"sample": sample, "cal": cal}
    return pairs

# ═══════════════════════════════════════════════════════════════════
#  Loading / peak detection / matching / fitting  — all in cm-1 space
# ═══════════════════════════════════════════════════════════════════

def load_spectrum(path):
    try:
        df = pd.read_csv(path, delimiter=DELIMITER, skiprows=SKIP_HEADER, header=None)
        x = df.iloc[:, X_COLUMN].values.astype(float)
        y = df.iloc[:, Y_COLUMN].values.astype(float)
    except Exception:
        raw = np.loadtxt(path, delimiter=DELIMITER, skiprows=SKIP_HEADER)
        x, y = raw[:, X_COLUMN], raw[:, Y_COLUMN]
    return x, y


def lorentzian(x, x0, gamma, A, y0):
    """Lorentzian with offset: A * gamma² / ((x-x0)² + gamma²) + y0"""
    return A * gamma**2 / ((x - x0)**2 + gamma**2) + y0


def detect_peaks(x, y, lamp, laser_nm, ax=None):
    """Detect peaks in the Raman-shift (cm-1) spectrum, refine each position
    with a Lorentzian fit in a ±FIT_WINDOW_CM range, and (optionally) plot."""
    min_height = np.percentile(y, PEAK_HEIGHT_PERCENTILE)
    prominence = PEAK_PROMINENCE
    idx, _ = find_peaks(y, height=min_height, prominence=prominence,
                        distance=PEAK_MIN_DISTANCE)

    positions, intensities, fits = [], [], []

    for i in idx:
        x0_init = x[i]

        # select data ±FIT_WINDOW_CM around initial peak position
        mask = (x >= x0_init - FIT_WINDOW_CM) & (x <= x0_init + FIT_WINDOW_CM)
        xw, yw = x[mask], y[mask]

        if len(xw) < 5:
            positions.append(x0_init)
            intensities.append(y[i])
            continue

        y0_guess    = yw.min()
        A_guess     = y[i] - y0_guess
        gamma_guess = FIT_WINDOW_CM / 4
        p0 = [x0_init, gamma_guess, A_guess, y0_guess]

        try:
            popt, _ = curve_fit(
                lorentzian, xw, yw, p0=p0,
                bounds=([x0_init - FIT_WINDOW_CM/5, 1e-6, 0,         -np.inf],
                        [x0_init + FIT_WINDOW_CM/5, FIT_WINDOW_CM*2, np.inf, np.inf]),
                maxfev=5000)
            x0_fit, gamma_fit, A_fit, y0_fit = popt

            positions.append(x0_fit)
            intensities.append(A_fit + y0_fit)
            fits.append((xw, lorentzian(xw, *popt)))
        except (RuntimeError, ValueError):
            yr = yw - yw.min()
            positions.append(np.average(xw, weights=yr) if yr.sum() > 0 else x0_init)
            intensities.append(y[i])

    positions = np.array(positions)
    intensities = np.array(intensities)

    # --- plot into the provided axis ---
    if ax is not None:
        # known lamp lines, converted to Raman shift for this laser
        known_cm, known_int = lamp_lines_in_shift(lamp, laser_nm)
        in_range = (known_cm >= x.min()) & (known_cm <= x.max())
        known_cm, known_int = known_cm[in_range], known_int[in_range]
        ax.vlines(known_cm, ymin=y.min(), ymax=y.max(), color='orange',
                  linestyle='--', lw=0.8, label='Known lamp lines (cm$^{-1}$)')
        ax.vlines(known_cm, ymin=y.min(), ymax=y.max(), color='orange',
                  linestyle='--', lw=0.8, label='Known lamp lines (cm$^{-1}$)')
        ax.plot(x, y, color='steelblue', lw=0.8, label='Spectrum')
        ax.plot(x[idx], y[idx], "x", color='gray', label='find_peaks (initial)')
        for xw, yfit in fits:
            ax.plot(xw, yfit, 'r-', lw=1, alpha=0.8)
        ax.vlines(positions, ymin=y.min(), ymax=intensities,
                  color='green', linestyle=':', lw=0.8)
        ax.plot([], [], 'r-', label='Lorentzian fits')
        ax.set_xlabel(r"Raman shift (cm$^{-1}$)")
        ax.set_ylabel("Intensity")
        ax.legend(fontsize=7)
        # plt.show()

    return positions, intensities


def match_peaks(detected_cm, detected_int, lamp, laser_nm,
                tolerance=MATCH_TOLERANCE_CM):
    """Match known lamp lines to detected peaks (cm-1).
    Lamp lines are processed in order of decreasing NIST relative
    intensity: the strongest lines are assigned first to the nearest
    free detected peak, then progressively weaker lines."""
    known_cm, known_int = lamp_lines_in_shift(lamp, laser_nm)

    # strongest lamp lines first
    order = np.argsort(known_int)[::-1]
    used_detected, matches = set(), []

    for j in order:
        d = np.abs(detected_cm - known_cm[j])
        for i in np.argsort(d):
            if d[i] > tolerance:
                break                       # nearest free peak too far
            if i in used_detected:
                continue                    # peak already taken by a stronger line
            matches.append((detected_cm[i], known_cm[j]))
            used_detected.add(i)
            break
    return matches


def describe_fit(cal_func):
    c = cal_func.coeffs
    if len(c) == 1:
        return f"true = {c[0]:.4f} (constant)"
    else:
        return f"true = {c[0]:.6f}·meas {c[1]:+.4f}"


def fit_linear(matches):
    """Polynomial fit: measured Raman shift -> true Raman shift (both cm-1).
    Iteratively removes points with residual > OUTLIER_SIGMA * std and refits.
    With <= 3 points, falls back to a constant offset (true = meas + c)."""
    min_points = FIT_ORDER + 1
    if len(matches) < min_points:
        raise ValueError(f"Need >= {min_points} matched lines for an order-{FIT_ORDER} "
                         f"fit, got {len(matches)}")

    x = np.array([m[0] for m in matches])
    y = np.array([m[1] for m in matches])

    rejected_x, rejected_y = [], []

    for iteration in range(MAX_REJECT_ITER + 1):
        if len(x) <= 2:
            c = np.mean(y - x)
            cal_func = np.poly1d([1.0, c])
        else:
            coeffs, cov = np.polyfit(x, y, 1, cov='unscaled')
            cal_func = np.poly1d(coeffs)

        residuals = y - cal_func(x)
        std = residuals.std()

        if std > 0:
            outlier_mask = np.abs(residuals) > OUTLIER_SIGMA * std
        else:
            outlier_mask = np.zeros_like(residuals, dtype=bool)

        if not outlier_mask.any() or iteration == MAX_REJECT_ITER:
            break

        if (~outlier_mask).sum() < min_points:
            print(f"  ⚠️  Outlier rejection would leave < {min_points} points — "
                  f"stopping rejection")
            break

        for xi, yi, ri in zip(x[outlier_mask], y[outlier_mask], residuals[outlier_mask]):
            print(f"  ✂️  Rejected line: meas={xi:.2f} cm-1, true={yi:.2f} cm-1, "
                  f"residual={ri:+.3f} cm-1 (> {OUTLIER_SIGMA}×σ = {OUTLIER_SIGMA*std:.3f} cm-1)")
            rejected_x.append(xi)
            rejected_y.append(yi)

        x, y = x[~outlier_mask], y[~outlier_mask]

    rms = np.sqrt(np.mean(residuals**2))
    kept_matches = list(zip(x, y))
    rejected = list(zip(rejected_x, rejected_y))

    return cal_func, rms, len(x), kept_matches, rejected

# ═══════════════════════════════════════════════════════════════════
#  Plotting
# ═══════════════════════════════════════════════════════════════════

def plot_linear_fit(region, lamp, matches, cal_func, rms, axes, rejected=None):
    """Plot matched lamp lines, fit, and residuals (all in cm-1)."""
    ax1, ax2 = axes
    x = np.array([m[0] for m in matches])   # measured shift (kept)
    y = np.array([m[1] for m in matches])   # true shift
    residuals = y - cal_func(x)

    xf = np.linspace(x.min(), x.max(), 200)
    ax1.plot(xf, cal_func(xf), "r-", lw=1.2, label=f"fit: {describe_fit(cal_func)}")
    ax1.plot(x, y, "ko", ms=5, label=f"matched {lamp.upper()} lines (n={len(x)})")

    if rejected:
        rx = np.array([r[0] for r in rejected])
        ry = np.array([r[1] for r in rejected])
        ax1.plot(rx, ry, "x", color="orange", ms=8, mew=2,
                 label=f"rejected (n={len(rx)})")

    ax1.set_ylabel(r"True Raman shift (cm$^{-1}$)")
    ax1.set_title(f"{region.upper()} — {lamp.upper()} lamp\nRMS = {rms:.3f} cm$^{{-1}}$",
                  fontsize=10)
    ax1.legend(fontsize=7)
    ax1.grid(alpha=0.3)

    ax2.axhline(0, color="r", lw=1)
    ax2.plot(x, residuals, "ko", ms=5)
    if rejected:
        ax2.plot(rx, ry - cal_func(rx), "x", color="orange", ms=8, mew=2)

    std = residuals.std()
    if std > 0:
        ax2.axhspan(-OUTLIER_SIGMA * std, OUTLIER_SIGMA * std,
                    alpha=0.1, color="green", label=f"±{OUTLIER_SIGMA}σ")
        ax2.legend(fontsize=7)

    ax2.set_xlabel(r"Measured Raman shift (cm$^{-1}$)")
    ax2.set_ylabel(r"Residual (cm$^{-1}$)")
    ax2.grid(alpha=0.3)


def plot_before_after(calibrated, laser_nm, step, out_dir):
    """Square 2x2 figure: before vs after calibration per region + combined."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    fig.suptitle(f"Before vs After calibration — {laser_nm} nm — {step}",
                 fontsize=13)

    axes_flat = axes.ravel()

    for i, region in enumerate(REGIONS):
        ax = axes_flat[i]
        df = next((d for d in calibrated if d["region"].iloc[0] == region), None)
        if df is None:
            ax.set_title(f"{region.upper()} — no data")
            ax.set_axis_off()
            continue

        ax.plot(df["raman_shift_raw"], df["intensity"],
                color="gray", lw=0.8, alpha=0.7, label="before")
        ax.plot(df["raman_shift_cm-1"], df["intensity"],
                color="crimson", lw=0.8, label="after")

        shift = np.mean(df["raman_shift_cm-1"] - df["raman_shift_raw"])
        ax.set_title(f"{region.upper()}  (mean Δ = {shift:+.2f} cm$^{{-1}}$)",
                     fontsize=10)
        ax.set_xlabel(r"Raman shift (cm$^{-1}$)")
        ax.set_ylabel("Intensity")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    ax = axes_flat[3]
    combined = pd.concat(calibrated, ignore_index=True)

    before = combined.sort_values("raman_shift_raw")
    after  = combined.sort_values("raman_shift_cm-1")
    ax.plot(before["raman_shift_raw"], before["intensity"],
            color="gray", lw=0.8, alpha=0.7, label="before")
    ax.plot(after["raman_shift_cm-1"], after["intensity"],
            color="crimson", lw=0.8, label="after")

    ax.set_title("Combined", fontsize=10)
    ax.set_xlabel(r"Raman shift (cm$^{-1}$)")
    ax.set_ylabel("Intensity")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    fig.tight_layout(rect=[0, 0, 1, 0.95])

    if SAVE_FIT_PLOTS:
        fig.savefig(os.path.join(out_dir,
                    f"before_after_{laser_nm}nm_{step}.png"), dpi=200)

    return fig

# ═══════════════════════════════════════════════════════════════════
#  Per-region calibration  — everything done in Raman-shift space
# ═══════════════════════════════════════════════════════════════════

def calibrate_region(region, pair, laser_nm, out_dir,
                     ax_peaks=None, axes_fit=None):
    sample, cal = pair["sample"], pair["cal"]
    if sample is None:
        print(f"  ⚠️  [{region.upper()}] no sample spectrum found — skipping")
        return None
    if cal is None:
        print(f"  ⚠️  [{region.upper()}] no calibration file found — skipping")
        return None

    lamp = cal["lamp"]
    print(f"\n  [{region.upper()}] sample: {os.path.basename(sample['path'])}")
    print(f"  [{region.upper()}] cal:    {os.path.basename(cal['path'])} (lamp: {lamp.upper()})")

    # --- load calibration spectrum: already in Raman shift (cm-1) ---
    rs_cal, y_cal = load_spectrum(cal["path"])
    srt = np.argsort(rs_cal)
    rs_cal, y_cal = rs_cal[srt], y_cal[srt]

    # --- detect peaks in cm-1, compare to lamp lines converted to cm-1 ---
    peak_cm, peak_int = detect_peaks(rs_cal, y_cal, lamp, laser_nm, ax=ax_peaks)
    print(f"  [{region.upper()}] detected {len(peak_cm)} peaks")

    matches = match_peaks(peak_cm, peak_int, lamp, laser_nm)
    print(f"  [{region.upper()}] matched {len(matches)} peaks to {lamp.upper()} lines")

    # --- fit measured shift -> true shift, with outlier rejection ---
    cal_func, rms, n, kept_matches, rejected = fit_linear(matches)
    print(f"  [{region.upper()}] fit (order {FIT_ORDER}): {describe_fit(cal_func)}  "
          f"(RMS = {rms:.3f} cm-1, {n} lines, {len(rejected)} rejected)")

    if axes_fit is not None:
        plot_linear_fit(region, lamp, kept_matches, cal_func, rms,
                        axes_fit, rejected=rejected)

    # --- apply correction directly to the sample's Raman-shift axis ---
    rs_smp, y_smp = load_spectrum(sample["path"])
    rs_corr = cal_func(rs_smp)

    return pd.DataFrame({
        "raman_shift_cm-1": rs_corr,
        "raman_shift_raw":  rs_smp,
        "intensity": y_smp,
        "region": region,
    })

# ═══════════════════════════════════════════════════════════════════
#  Main   (unchanged except titles)
# ═══════════════════════════════════════════════════════════════════

def main(folder, laser_nm, step):
    print("=" * 70)
    print(f"  RAMAN CALIBRATION — laser {laser_nm} nm — step: {step} — "
          f"folder: {os.path.abspath(folder)}")
    print("=" * 70)

    out_dir = os.path.join(folder, CALIBRATION_DIR)
    os.makedirs(out_dir, exist_ok=True)

    pairs = find_files(folder, laser_nm, step)

    fig_peaks, axes_peaks = plt.subplots(1, 3, figsize=(18, 5))
    fig_peaks.suptitle(f"Detected lamp peaks (cm$^{{-1}}$) — {laser_nm} nm — {step}",
                       fontsize=13)

    fig_fit, axes_fit = plt.subplots(
        2, 3, figsize=(18, 7),
        gridspec_kw={"height_ratios": [3, 1]})
    fig_fit.suptitle(f"Order-{FIT_ORDER} calibration fits (cm$^{{-1}}$) — "
                     f"{laser_nm} nm — {step}", fontsize=13)
    for col in range(3):
        axes_fit[0, col].sharex(axes_fit[1, col])

    calibrated = []
    for i, region in enumerate(REGIONS):
        result = calibrate_region(
            region, pairs[region], laser_nm, out_dir,
            ax_peaks=axes_peaks[i],
            axes_fit=(axes_fit[0, i], axes_fit[1, i]))
            
        if result is not None:
            calibrated.append(result)

            if SEP_REGIONS:
                result = result.drop(columns=["raman_shift_raw"])
                result = result.sort_values("raman_shift_cm-1").reset_index(drop=True)
                out_path = os.path.join(out_dir, str(round(laser_nm)) + 'nm_' + step + '_' + region + OUTPUT_SUFFIX)
                result.to_csv(out_path, index=False, header=False)
        else:
            axes_peaks[i].set_title(f"{region.upper()} — no data")
            axes_fit[0, i].set_title(f"{region.upper()} — no data")

    fig_peaks.tight_layout(rect=[0, 0, 1, 0.94])
    fig_fit.tight_layout(rect=[0, 0, 1, 0.93])

    if SAVE_FIT_PLOTS:
        fig_peaks.savefig(os.path.join(out_dir,
                          f"peaks_{laser_nm}nm_{step}.png"), dpi=200)
        fig_fit.savefig(os.path.join(out_dir,
                        f"fits_order{FIT_ORDER}_{laser_nm}nm_{step}.png"), dpi=200)

    fig_cmp = None
    if calibrated:
        fig_cmp = plot_before_after(calibrated, laser_nm, step, out_dir)

    if SHOW_FIT_PLOTS:
        plt.show()
    else:
        plt.close(fig_peaks)
        plt.close(fig_fit)
        if fig_cmp is not None:
            plt.close(fig_cmp)

    if not calibrated:
        print(f"\n❌ No spectra were calibrated for step '{step}'. "
              f"Check filenames/settings.")
        return




    combined = pd.concat(calibrated, ignore_index=True)
    combined = combined.drop(columns=["raman_shift_raw"])
    combined = combined.sort_values("raman_shift_cm-1").reset_index(drop=True)

    first_sample = next(p["sample"]["path"] for p in pairs.values()
                        if p["sample"] is not None)
    base = os.path.splitext(os.path.basename(first_sample))[0]
    for r in REGIONS:
        base = re.sub(rf"(^|_){r}(_|$)", r"\1all\2", base, flags=re.IGNORECASE)
    out_path = os.path.join(out_dir, base + OUTPUT_SUFFIX)

    # combined.to_csv(out_path, index=False, header=False)
    # print(f"\n💾 Saved combined calibrated spectrum ({len(combined)} points): {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Calibrate RBM/DGCC/2D Raman spectra and merge.")
    ap.add_argument("--folder", default=SPECTRA_FOLDER,
                    help="folder containing the step subfolders and calib_lamps/")
    args = ap.parse_args()

    for laser in LASER_LIST:
        for step in STEP_LIST:
            main(args.folder, laser, step)