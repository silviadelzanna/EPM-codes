"""
raman_calibration.py
=====================
Calibrates the three sub-spectra (RBM, DGCC, 2D) of a laser measurement
against their respective calibration lamp spectra using a LINEAR fit,
and merges them into one combined output file.

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

Output:
    - combined calibrated spectrum  -> <folder>/calibration/
    - linear fit plot per region    -> <folder>/calibration/

Usage:
    python raman_calibration.py
    python raman_calibration.py --folder ./data --laser 488
"""

import os
import re
import glob
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# ═══════════════════════════════════════════════════════════════════
#  USER SETTINGS
# ═══════════════════════════════════════════════════════════════════

SPECTRA_FOLDER   = r"C:\Users\andre\ucloud\Documents\Samples\Lucius\LabRAM\v3"
LASER_NM         =  487.986#487.986, 514.532, 568.188, 633, 
FILE_EXTENSIONS  = (".csv", ".txt", ".dat")

DELIMITER        = "\t"
SKIP_HEADER      = 0
X_COLUMN         = 0            # Raman shift column
Y_COLUMN         = 1            # intensity column

REGIONS          = ['rbm', 'dgcc', '2d']
LAMPS            = ["ne", "xe", "ar", "kr"]
LASER_LIST       = [ 487.986,514.532, 568.188]  # supported lasers 487.986,514.532, 568.188,

# Peak detection / matching
PEAK_HEIGHT_PERCENTILE = 10
PEAK_PROMINENCE_FACTOR = 0.01
PEAK_MIN_DISTANCE      = 3
MATCH_TOLERANCE_NM     = 0.5
FIT_ORDER              = 0      # LINEAR fit

OUTPUT_SUFFIX     = "_COMBINED_CALIBRATED.csv"
CALIBRATION_DIR   = "calibration"       # subfolder for all output
SAVE_FIT_PLOTS    = False                # save fit plots as PNG
SHOW_FIT_PLOTS    = False                # show fit plots on screen

OUTLIER_SIGMA   = 2   # rejection threshold: residual > OUTLIER_SIGMA * std
MAX_REJECT_ITER = 5     # safety limit on refit iterations


# ═══════════════════════════════════════════════════════════════════
#  Calibration lamp lines (MKS/Newport 6030-6033), nm
# ═══════════════════════════════════════════════════════════════════

LAMP_LINES = {
    "kr": [366.53, 367.96, 377.34, 428.30, 431.96, 436.26, 437.61, 440.0,
           442.52, 445.39, 446.37, 450.24, 549.09, 550.07, 556.22, 557.03,
           558.04, 583.29, 587.09, 587.99, 599.39, 601.22, 605.61, 642.1,
           645.63, 669.92, 681.31, 690.47, 722.41, 728.98, 742.55, 748.61,
           758.74, 760.15, 768.52, 769.45, 774.68, 785.48, 791.34, 805.95,
           810.44, 811.29, 819.01, 826.32, 828.11, 829.81, 850.89, 877.67,
           892.87],
    "ne": [336.99, 341.79, 344.77, 345.42, 346.66, 347.26, 349.81, 350.12,
           351.52, 352.05, 359.35, 360.02, 363.37, 470.44, 492.32, 503.78,
           508.04, 511.65, 514.5, 520.39, 533.08, 534.11, 534.33, 540.06,
           564.96, 565.67, 574.83, 576.44, 580.45, 582.02, 585.25, 588.19,
           590.25, 594.48, 597.55, 598.79, 603.0, 607.43, 609.62, 612.85,
           614.31, 616.36, 618.22, 621.73, 626.65, 630.48, 631.81, 633.44,
           638.3, 640.23, 650.65, 653.29, 659.9, 665.21, 667.83, 671.7,
           692.95, 702.41, 703.24, 705.3, 705.91, 717.39, 724.52, 743.89,
           747.24, 748.89, 753.58, 754.41, 794.32, 808.25, 813.64, 830.03,
           837.76, 841.84, 849.54],
    "ar": [355.43, 384.9, 404.44, 415.86, 416.4, 418.19, 419.1, 419.8,
           420.07, 425.12, 425.94, 426.63, 427.22, 427.4, 430.01, 432.0,
           433.36, 434.52, 641.63, 667.73, 675.28, 687.13, 693.77, 696.54,
           703.03, 706.72, 714.7, 727.29, 737.21, 738.4, 750.39, 751.46,
           763.51, 772.38, 794.82, 800.62, 801.48, 810.37, 811.53, 826.45,
           840.82, 842.46],
    "xe": [450.1, 452.47, 458.28, 462.43, 467.12, 469.7, 473.42, 480.7,
           482.97, 484.33, 491.65, 502.83, 618.24, 646.97, 666.89, 672.8,
           682.73, 688.22, 711.96, 728.53, 758.47, 764.2, 823.16, 828.01,
           834.68, 840.92],
}

# ═══════════════════════════════════════════════════════════════════
#  Conversion helpers
# ═══════════════════════════════════════════════════════════════════

def raman_shift_to_wavelength(laser_nm, shift_cm):
    return 1e7 / (1e7 / laser_nm - np.asarray(shift_cm, dtype=float))

def wavelength_to_raman_shift(laser_nm, wl_nm):
    return 1e7 / laser_nm - 1e7 / np.asarray(wl_nm, dtype=float)

# ═══════════════════════════════════════════════════════════════════
#  File discovery / parsing
# ═══════════════════════════════════════════════════════════════════

def parse_filename(filename):
    """
    Extract region, lamp and laser wavelength from a filename.
    A file WITH a lamp tag is a calibration file.
    A file WITHOUT a lamp tag is a sample file.
    """
    name = os.path.basename(filename).lower()
    tokens = re.split(r"[_\-.]", name)

    region = next((r for r in REGIONS if r in tokens), None)
    lamp   = next((l for l in LAMPS   if l in tokens), None)
    is_cal = lamp is not None            # <-- lamp tag = calibration file

    laser = None
    m = re.search(r"(\d+(?:\.\d+)?)nm", name)
    if m:
        laser = float(m.group(1))

    return {"path": filename, "region": region, "lamp": lamp,
            "laser": laser, "is_cal": is_cal}


def find_files(folder, laser_nm):
    """Find and classify all spectra for the given laser wavelength."""
    files = []
    for ext in FILE_EXTENSIONS:
        files.extend(glob.glob(os.path.join(folder, f"*{ext}")))

    parsed = [parse_filename(f) for f in files]

    for f in parsed:
        print(f)
    parsed = [p for p in parsed
              if p["region"] is not None
              and (p["laser"] is None or abs(p["laser"] - laser_nm) < 0.5)
              and OUTPUT_SUFFIX.lower() not in p["path"].lower()]

    pairs = {}
    for region in REGIONS:
        cal    = next((p for p in parsed if p["region"] == region and p["is_cal"]), None)
        sample = next((p for p in parsed if p["region"] == region and not p["is_cal"]), None)
        pairs[region] = {"sample": sample, "cal": cal}
    return pairs

# ═══════════════════════════════════════════════════════════════════
#  Loading / peak detection / matching / fitting
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


from scipy.optimize import curve_fit

FIT_WINDOW = 0.1   # ± range around each peak used for the Lorentzian fit
                    # (same units as x-axis: cm⁻¹ ... or nm if you fit in wavelength space!)

def lorentzian(x, x0, gamma, A, y0):
    """Lorentzian with offset: A * gamma² / ((x-x0)² + gamma²) + y0"""
    return A * gamma**2 / ((x - x0)**2 + gamma**2) + y0


def detect_peaks(x, y, lamp):
    """Detect peaks, then refine each position with a Lorentzian fit
    in a ±FIT_WINDOW range around the initial peak position."""
    min_height = np.percentile(y, PEAK_HEIGHT_PERCENTILE)
    prominence = PEAK_PROMINENCE_FACTOR * (y.max() - y.min())
    idx, _ = find_peaks(y, height=min_height, prominence=prominence,
                        distance=PEAK_MIN_DISTANCE)

    positions = []
    intensities = []
    fits = []   # store fits for plotting

    for i in idx:
        x0_init = x[i]

        # select data ±FIT_WINDOW around initial peak position
        mask = (x >= x0_init - FIT_WINDOW) & (x <= x0_init + FIT_WINDOW)
        xw, yw = x[mask], y[mask]

        if len(xw) < 5:
            # too few points to fit — keep raw position
            positions.append(x0_init)
            intensities.append(y[i])
            continue

        # initial guesses
        y0_guess    = yw.min()
        A_guess     = y[i] - y0_guess
        gamma_guess = FIT_WINDOW / 4
        p0 = [x0_init, gamma_guess, A_guess, y0_guess]

        try:
            popt, _ = curve_fit(
                lorentzian, xw, yw, p0=p0,
                bounds=([x0_init - FIT_WINDOW/5, 1e-6, 0,      -np.inf],
                        [x0_init + FIT_WINDOW/5, FIT_WINDOW*2, np.inf,  np.inf]),
                maxfev=5000)
            x0_fit, gamma_fit, A_fit, y0_fit = popt

            positions.append(x0_fit)
            intensities.append(A_fit + y0_fit)          # peak height from fit
            fits.append((xw, lorentzian(xw, *popt)))
        except (RuntimeError, ValueError):
            # fit failed — fall back to centroid refinement
            yr = yw - yw.min()
            positions.append(np.average(xw, weights=yr) if yr.sum() > 0 else x0_init)
            intensities.append(y[i])

    positions = np.array(positions)
    intensities = np.array(intensities)

    known = [p for p in np.array(LAMP_LINES[lamp]) if x.min() <= p <= x.max()]

    # --- plot: raw peaks + Lorentzian fits + refined positions ---
    plt.figure(figsize=(10, 5))
    plt.vlines(known, ymin=y.min(), ymax=y.max(), color='orange', linestyle='--', lw=0.8, label='Known lamp lines')

    plt.plot(x, y, color='steelblue', lw=0.8, label='Spectrum')
    plt.plot(x[idx], y[idx], "x", color='gray', label='find_peaks (initial)')
    for xw, yfit in fits:
        plt.plot(xw, yfit, 'r-', lw=1, alpha=0.8)
    plt.vlines(positions, ymin=y.min(), ymax=intensities,
               color='green', linestyle=':', lw=0.8)
    plt.plot([], [], 'r-', label='Lorentzian fits')
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Intensity")
    plt.title("Detected Peaks (Lorentzian-refined)")
    plt.legend(fontsize=8)

    if SHOW_FIT_PLOTS:
        plt.show()
    else:
        plt.close()

    return positions, intensities


def match_peaks(detected_wl, detected_int, lamp, tolerance=MATCH_TOLERANCE_NM):
    """Intensity-priority nearest-neighbor matching to known lamp lines."""
    known = np.array(LAMP_LINES[lamp])
    order = np.argsort(detected_int)[::-1]        # strongest first
    used_known, matches = set(), []

    for i in order:
        d = np.abs(known - detected_wl[i])
        for j in np.argsort(d):
            if d[j] > tolerance:
                break
            if j in used_known:
                continue
            matches.append((detected_wl[i], known[j]))
            used_known.add(j)
            break
    print(matches)
    return matches

def fit_equation_str(cal_func):
    """Human-readable fit equation for any polynomial order."""
    c = cal_func.coeffs
    if cal_func.order == 0:
        return f"true = meas {c[0] and c[-1]:+.4f}" if False else f"true = meas + offset? no"
    return ""


def describe_fit(cal_func):
    """Return a readable equation string for 0th or 1st order fits."""
    c = cal_func.coeffs
    if len(c) == 1:                       # 0th order: true = const
        return f"true = {c[0]:.4f} (constant)"
    else:                                 # 1st order: true = a*meas + b
        return f"true = {c[0]:.6f}·meas {c[1]:+.4f}"


def fit_linear(matches):
    """Polynomial fit (order FIT_ORDER): measured wavelength -> true wavelength.
    Iteratively removes points with residual > OUTLIER_SIGMA * std and refits.

    Note: for FIT_ORDER = 0 the fit 'true = const' is rarely what you want —
    usually you want a constant OFFSET (true = meas + c). That is handled
    below by fitting the residual y - x instead."""
    min_points = FIT_ORDER + 1
    if len(matches) < min_points:
        raise ValueError(f"Need >= {min_points} matched lines for an order-{FIT_ORDER} "
                         f"fit, got {len(matches)}")

    x = np.array([m[0] for m in matches])
    y = np.array([m[1] for m in matches])

    rejected_x, rejected_y = [], []

    for iteration in range(MAX_REJECT_ITER + 1):
        if len(matches) <= 3:
            # constant offset: true = meas + c   (fit c = mean of y - x)
            c = np.mean(y - x)
            error = np.std(y - x)
            cal_func = np.poly1d([1.0, c])      # slope fixed to 1
        else:
            coeffs, cov = np.polyfit(x, y, 1, cov='unscaled')
            error = np.sqrt(np.diag(cov))
            cal_func = np.poly1d(coeffs)

        residuals = y - cal_func(x)
        std = residuals.std()

        # find outliers (std can be 0 if all points identical / only 1 point)
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

        # report and remove outliers
        for xi, yi, ri in zip(x[outlier_mask], y[outlier_mask], residuals[outlier_mask]):
            print(f"  ✂️  Rejected line: meas={xi:.3f} nm, true={yi:.3f} nm, "
                  f"residual={ri:+.4f} nm (> {OUTLIER_SIGMA}×σ = {OUTLIER_SIGMA*std:.4f} nm)")
            rejected_x.append(xi)
            rejected_y.append(yi)

        x, y = x[~outlier_mask], y[~outlier_mask]

    rms = np.sqrt(np.mean(residuals**2))
    kept_matches = list(zip(x, y))
    rejected = list(zip(rejected_x, rejected_y))

    return cal_func, rms, len(x), kept_matches, rejected


def plot_linear_fit(region, lamp, matches, cal_func, rms, out_dir, rejected=None):
    """Plot matched lamp lines, the fit, residuals, and rejected outliers."""
    x = np.array([m[0] for m in matches])   # measured (kept points)
    y = np.array([m[1] for m in matches])   # true
    residuals = y - cal_func(x)

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(8, 7), sharex=True,
        gridspec_kw={"height_ratios": [3, 1]})

    # top: fit
    xf = np.linspace(x.min(), x.max(), 200)
    ax1.plot(xf, cal_func(xf), "r-", lw=1.2, label=f"fit: {describe_fit(cal_func)}")
    ax1.plot(x, y, "ko", ms=5, label=f"matched {lamp.upper()} lines (n={len(x)})")

    if rejected:
        rx = np.array([r[0] for r in rejected])
        ry = np.array([r[1] for r in rejected])
        ax1.plot(rx, ry, "x", color="orange", ms=8, mew=2,
                 label=f"rejected outliers (n={len(rx)})")

    ax1.set_ylabel("True wavelength (nm)")
    ax1.set_title(f"[{region.upper()}] Order-{FIT_ORDER} calibration fit — "
                  f"{lamp.upper()} lamp (RMS = {rms:.4f} nm)")
    ax1.legend()
    ax1.grid(alpha=0.3)

    # bottom: residuals
    ax2.axhline(0, color="r", lw=1)
    ax2.plot(x, residuals, "ko", ms=5)

    if rejected:
        r_res = ry - cal_func(rx)
        ax2.plot(rx, r_res, "x", color="orange", ms=8, mew=2)

    std = residuals.std()
    if std > 0:
        ax2.axhspan(-OUTLIER_SIGMA * std, OUTLIER_SIGMA * std,
                    alpha=0.1, color="green", label=f"±{OUTLIER_SIGMA}σ")
        ax2.legend(fontsize=8)

    ax2.set_xlabel("Measured wavelength (nm)")
    ax2.set_ylabel("Residual (nm)")
    ax2.grid(alpha=0.3)

    fig.tight_layout()

    if SAVE_FIT_PLOTS:
        png_path = os.path.join(out_dir, f"fit_order{FIT_ORDER}_{region}_{lamp}.png")
        fig.savefig(png_path, dpi=200)
        print(f"  [{region.upper()}] fit plot saved: {png_path}")

    if SHOW_FIT_PLOTS:
        plt.show()
    else:
        plt.close(fig)

# ═══════════════════════════════════════════════════════════════════
#  Per-region calibration
# ═══════════════════════════════════════════════════════════════════

def calibrate_region(region, pair, laser_nm, out_dir):
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

    # --- load calibration spectrum & convert to wavelength ---
    rs_cal, y_cal = load_spectrum(cal["path"])
    wl_cal = raman_shift_to_wavelength(laser_nm, rs_cal)
    srt = np.argsort(wl_cal)
    wl_cal, y_cal = wl_cal[srt], y_cal[srt]

    # --- detect and match peaks ---
    peak_wl, peak_int = detect_peaks(wl_cal, y_cal,lamp)
    print(f"  [{region.upper()}] detected {len(peak_wl)} peaks")

    matches = match_peaks(peak_wl, peak_int, lamp)
    print(f"  [{region.upper()}] matched {len(matches)} peaks to {lamp.upper()} lines")

    # --- linear fit with outlier rejection ---
    # --- fit with outlier rejection ---
    cal_func, rms, n, kept_matches, rejected = fit_linear(matches)
    print(f"  [{region.upper()}] fit (order {FIT_ORDER}): {describe_fit(cal_func)}  "
          f"(RMS = {rms:.4f} nm, {n} lines, {len(rejected)} rejected)")

    plot_linear_fit(region, lamp, kept_matches, cal_func, rms, out_dir, rejected=rejected)

    # --- apply to sample spectrum ---
    rs_smp, y_smp = load_spectrum(sample["path"])
    wl_meas = raman_shift_to_wavelength(laser_nm, rs_smp)
    wl_corr = cal_func(wl_meas)
    print(cal_func.coeffs)
    print(cal_func)
    rs_corr = wavelength_to_raman_shift(laser_nm, wl_corr)

    plt.plot(rs_smp, rs_corr - rs_smp, label="raw sample", color="gray", alpha=0.5)

    if SHOW_FIT_PLOTS:
        plt.show()
    else:
        plt.close()

    return pd.DataFrame({
        "raman_shift_cm-1": rs_corr,
        "intensity": y_smp,
        "region": region,
    })

# ═══════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════

def main(folder, laser_nm):
    print("=" * 70)
    print(f"  RAMAN CALIBRATION — laser {laser_nm} nm — folder: {os.path.abspath(folder)}")
    print("=" * 70)

    # create output folder <folder>/calibration
    out_dir = os.path.join(folder, CALIBRATION_DIR)
    os.makedirs(out_dir, exist_ok=True)

    pairs = find_files(folder, laser_nm)

    calibrated = []
    for region in REGIONS:
        result = calibrate_region(region, pairs[region], laser_nm, out_dir)
        if result is not None:
            calibrated.append(result)

    if not calibrated:
        print("\n❌ No spectra were calibrated. Check filenames/settings.")
        return

    # merge and sort by Raman shift
    combined = pd.concat(calibrated, ignore_index=True)
    combined = combined.sort_values("raman_shift_cm-1").reset_index(drop=True)

    # build output name from first sample file, region token -> "all"
    first_sample = next(p["sample"]["path"] for p in pairs.values()
                        if p["sample"] is not None)
    base = os.path.splitext(os.path.basename(first_sample))[0]
    for r in REGIONS:
        base = re.sub(rf"(^|_){r}(_|$)", r"\1all\2", base, flags=re.IGNORECASE)
    out_path = os.path.join(out_dir, base + OUTPUT_SUFFIX)

    if True:#input('Save combined calibrated spectrum? (y/n): ').strip().lower() == 'y':
        combined.to_csv(out_path, index=False, header = False)
        print(f"\n💾 Saved combined calibrated spectrum ({len(combined)} points): {out_path}")
    else:
        print("❌ Aborted saving combined spectrum.")
        return


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Calibrate RBM/DGCC/2D Raman spectra and merge.")
    ap.add_argument("--folder", default=SPECTRA_FOLDER, help="folder containing the spectra")
    ap.add_argument("--laser",  default=LASER_NM, type=float, help="laser wavelength (nm)")
    args = ap.parse_args()
    for laser in LASER_LIST:
        main(args.folder, laser)