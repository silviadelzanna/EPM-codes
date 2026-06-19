import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.special import voigt_profile
from scipy.sparse.linalg import spsolve
from scipy import sparse
from scipy.linalg import cholesky

# --- Your existing functions ---
def gauss(x, A, mu, sigma, offset):
    return A * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2)) + offset

def gauss_fitter(x, y):
    p0 = [np.max(y), x[np.argmax(y)], 1, np.min(y)]
    popt, cov = curve_fit(gauss, x, y, p0=p0)
    return popt


def als(y, lam=1e6, p=0.1, itermax=10):
    r"""
    Implements an Asymmetric Least Squares Smoothing
    baseline correction algorithm (P. Eilers, H. Boelens 2005)
 
    Baseline Correction with Asymmetric Least Squares Smoothing
    based on https://web.archive.org/web/20200914144852/https://github.com/vicngtor/BaySpecPlots
 
    Baseline Correction with Asymmetric Least Squares Smoothing
    Paul H. C. Eilers and Hans F.M. Boelens
    October 21, 2005
 
    Description from the original documentation:
 
    Most baseline problems in instrumental methods are characterized by a smooth
    baseline and a superimposed signal that carries the analytical information: a series
    of peaks that are either all positive or all negative. We combine a smoother
    with asymmetric weighting of deviations from the (smooth) trend get an effective
    baseline estimator. It is easy to use, fast and keeps the analytical peak signal intact.
    No prior information about peak shapes or baseline (polynomial) is needed
    by the method. The performance is illustrated by simulation and applications to
    real data.
 
 
    Inputs:
        y:
            input data (i.e. chromatogram of spectrum)
        lam:
            parameter that can be adjusted by user. The larger lambda is,
            the smoother the resulting background, z
        p:
            wheighting deviations. 0.5 = symmetric, <0.5: negative
            deviations are stronger suppressed
        itermax:
            number of iterations to perform
    Output:
        the fitted background vector
 
    """
    L = len(y)
    D = sparse.eye(L, format='csc')
    D = D[1:] - D[:-1]  # numpy.diff( ,2) does not work with sparse matrix. This is a workaround.
    D = D[1:] - D[:-1]
    D = D.T
    w = np.ones(L)
    for i in range(itermax):
        W = sparse.diags(w, 0, shape=(L, L))
        Z = W + lam * D.dot(D.T)
        z = spsolve(Z, w * y)
        w = p * (y > z) + (1 - p) * (y < z)
    return z


# --- Main script ---
data_path = r"C:\Users\andre\ucloud\Documents"
sample = "Otho"
setup = "LabRAM"
step = "S41"
laser = '633'
folder = os.path.join(data_path, 'Samples', sample, setup)

cutoff_list = [90, 120, 150, 100]

files = []
for (dirpath, dirnames, filenames) in os.walk(folder):
    files.extend(filenames)
    break

for file in files:
    if step in file and laser in file:
        print(f"Processing file: {file}")

        if laser == '488':
            cutoff = cutoff_list[0]
        elif laser == '515':
            cutoff = cutoff_list[1]
        elif laser == '568':
            cutoff = cutoff_list[2]
        elif laser == '633':
            cutoff = cutoff_list[3]


        df_file = pd.read_csv(os.path.join(folder, file), sep='\t', header=None)

        df_file[1] = (df_file[1] - np.min(df_file[1])) / (np.max(df_file[1]) - np.min(df_file[1]))

        gauss_params = gauss_fitter(df_file[0][df_file[0] < 10], df_file[1][df_file[0] < 10])
        df_file[0] -= gauss_params[1]


        laser_line_params = pd.read_csv(os.path.join(folder.replace(setup,'other'), 'laserline_params.csv'), sep=',')
        if step in laser_line_params['Step'].values and int(laser) in laser_line_params['Laser'].values and setup in laser_line_params['Setup'].values:
            print("Laser line parameters already exist for this step, laser, and setup.")
        else:
            print(step, laser, setup)
            print(laser in laser_line_params['Laser'].values, step in laser_line_params['Step'].values, setup in laser_line_params['Setup'].values)
            laser_line_params = pd.concat([laser_line_params, pd.DataFrame([[step, laser, setup, gauss_params[0], gauss_params[1], gauss_params[2], gauss_params[3]]], columns=laser_line_params.columns)], ignore_index=True)
            laser_line_params.to_csv(os.path.join(folder.replace(setup,'other'), 'laserline_params.csv'), index=False)

        # plt.plot(df_file[0], df_file[1], label=file)
        # plt.show()

        df_file[1] = (df_file[1] - np.min(df_file[1])) / (np.max(df_file[1]) - np.min(df_file[1]))
        df_file = df_file[df_file[0] >cutoff]

        region_RBM = (cutoff,1000)
        region_photo = (cutoff,np.max(df_file[0]))

        asls_params_RBM = [0.9e7, 0.008, 10] # 1e7, 0.01, 10
        asls_params_photo = [9e7, 0.0001, 10] # 6e7, 0.0005, 10



        # corection_regions = [region1, region2]
        # aslsparams = [asls_params1, asls_params2]

        # background_spectra = []

        background = pd.DataFrame(np.zeros_like(df_file[1]), index=df_file.index)
        background_spectrum_photo = pd.DataFrame(als(df_file[1][(df_file[0] >= region_photo[0]) & (df_file[0] < region_photo[1])].values, lam=asls_params_photo[0], p=asls_params_photo[1], itermax=asls_params_photo[2]), index=df_file.index[(df_file[0] >= region_photo[0]) & (df_file[0] < region_photo[1])])
        background[(df_file[0] >= region_photo[0]) & (df_file[0] < region_photo[1])] += background_spectrum_photo

        # df_file[1] -= background[0]

        plt.plot(df_file[0], df_file[1], label='Original', color='red', linestyle='-')

        df_file[1] -= background[0]

        background_spectrum_RBM = pd.DataFrame(als(df_file[1][(df_file[0] >= region_RBM[0]) & (df_file[0] < region_RBM[1])].values, lam=asls_params_RBM[0], p=asls_params_RBM[1], itermax=asls_params_RBM[2]), index=df_file.index[(df_file[0] >= region_RBM[0]) & (df_file[0] < region_RBM[1])])
        
        df_file[1] += background[0]
        background[(df_file[0] >= region_RBM[0]) & (df_file[0] < region_RBM[1])] += background_spectrum_RBM

        plt.plot(df_file[0], background[0], label='background', color='red', linestyle='--')
        plt.plot(df_file[0], df_file[1] - background[0], label='Baseline Corrected', color='green', linestyle='-')
        plt.legend()
        plt.show()


        if input("Save baseline corrected data? (y/n): ").lower() == 'y':
            df_file[1] -= background[0]
            df_file = df_file[df_file[0] < 2800]
            os.makedirs(os.path.join(folder, 'Baseline_Corrected'), exist_ok=True)
            df_file.to_csv(os.path.join(folder, 'Baseline_Corrected', file.replace('.txt', '_baseline_corrected.txt')), sep='\t', header=False, index=False)






