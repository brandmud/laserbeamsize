"""Functions for fitting the Bessel function to the laser beam size data."""
import pickle
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import scipy.special as sp

def bessel_fit_function(r, I_0, r_scale, r_shift):
    """Function to fit the Bessel function to the laser beam size data."""
    return I_0*(2*sp.j1(np.pi*(r+r_shift)*r_scale)/(np.pi*(r+r_shift)*r_scale))**2


def gaussian_fit_function(r, I_0, sigma, mu):
    """Function to fit a Gaussian to the laser beam size data."""
    return I_0*np.exp(-(r-mu)**2/(2*sigma**2))


def fit_bessel(r, I):
    I_0 = I.max()
    fit_params, _ = curve_fit(bessel_fit_function, r, I, p0=[I_0, 1, 0])
    return fit_params
    

def main():
    """Main function."""
    # load data
    with open(r'D:\brandmueller\PhotonicsBasedPhotoAcoustics\programming\pbpa\measurement\data\test_bessel_fit.pkl', 'rb') as f:
        data = pickle.load(f)
    fit_params = fit_bessel(data['r'], data['z'])
    print(fit_params)
    plt.figure()
    plt.plot(data['r'], data['z'], 'o', label='data')
    plt.plot(data['r'], bessel_fit_function(data['r'], *fit_params), label='fit')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()
