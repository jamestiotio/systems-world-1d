# 10.007 1D Project PCR Fluorescence Model
# Created by James Raphael Tiovalen (2020)

# Import libraries
import numpy as np
from scipy.optimize import curve_fit
from sklearn import metrics

# Define exponential function
def exponential(C, alpha, beta):
    return alpha * np.exp(beta * C)

# Define logistic function
def logistic(C, k, d):
    return 450 / (1 + np.exp(- k * (C - d)))

# Define cube root function
def cube_root(C, alpha, beta):
    return (((alpha * C) - beta) ** (1 / 3)) + 219.1416667

if __name__ == '__main__':
    cycles = np.array([8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36]).astype(np.float64)
    fluorescences = np.array([8.2, 19.1, 37.6, 70.4, 145.9, 219.1416667, 296.7, 367.8, 378.2, 403.3, 424.5, 436.6, 445.1, 446.8, 448.0]).astype(np.float64)

    # Question A2
    old_popt, old_pcov = curve_fit(exponential, cycles[0:5], fluorescences[0:5], method="dogbox", bounds=(0, np.inf))
    old_r_squared = round(metrics.r2_score(exponential(cycles[0:5], *old_popt), fluorescences[0:5]), 3)
    print(f"Equation: F = {old_popt[0]} * e^({old_popt[1]} * C)")
    print(f"R^2 = {old_r_squared}")

    # Question A6
    popt, pcov = curve_fit(logistic, cycles, fluorescences, method="trf", bounds=(0, np.inf))
    r_squared = round(metrics.r2_score(logistic(cycles, *popt), fluorescences), 3)
    print(f"Equation: F = 450 / (1 + e^(- {popt[0]} * (C - {popt[1]})))")
    print(f"R^2 = {r_squared}")
    
    # Question A8
    new_popt, new_pcov = curve_fit(cube_root, cycles, fluorescences, method="lm")
    new_r_squared = round(metrics.r2_score(cube_root(cycles, *new_popt), fluorescences), 3)
    print(f"Equation: F = ((({new_popt[0]} * C) - {new_popt[1]}) ** (1 / 3)) + 219.1416667")
    print(f"R^2 = {new_r_squared}")