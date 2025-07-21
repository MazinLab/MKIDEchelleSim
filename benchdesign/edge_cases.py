import numpy as np
import scipy.interpolate as interp
import scipy
import scipy.signal as sig
import scipy.ndimage as ndi
import matplotlib.pyplot as plt
import time
import astropy.units as u
import copy
from astropy.constants import h, c
from lmfit import Parameters, minimize

from momospecsim.spectrograph import GratingSetup, SpectrographSetup
from momospecsim.detector import MKIDDetector, wave_to_phase
from momospecsim.utils.general import nearest_idx, gauss

"""
The purpose of these methods is to determine the set of grating and spectrograph design parameters that result in
maximal order-packing onto the wavelength range given a specified R at 800nm (with the assumption that R will be
double at 400nm). These design parameters may or may not be able to be custom designed. Variable R and phase_offset
are turned off to assume the same pixel response.
"""


def desired_grating(R0, sep, minw=400*u.nm, maxw=800*u.nm, stype='default', plot=True):
    minw_phase = wave_to_phase(waves=minw.to(u.nm).value, minwave=minw, maxwave=maxw)
    maxw_phase = wave_to_phase(maxw.to(u.nm).value, minwave=minw, maxwave=maxw)
    phase_array = np.linspace(-1, 0, 10000)

    wave_FWHM = np.abs(maxw.to(u.nm).value / R0)
    FWHMi_phase = wave_to_phase(waves=(maxw.to(u.nm).value-wave_FWHM/2), minwave=minw, maxwave=maxw)
    FWHMf_phase = wave_to_phase(waves=(maxw.to(u.nm).value+wave_FWHM/2), minwave=minw, maxwave=maxw)
    phase_FWHM = FWHMf_phase - FWHMi_phase

    idxs = [nearest_idx(phase_array, maxw_phase)]
    peak = phase_array[idxs[0]]
    while peak > minw_phase:
        peak -= phase_FWHM*sep
        idx = nearest_idx(phase_array, peak)
        idxs.append(idx)

    full_phase = np.linspace(-1, 0, 10000)
    pixlast_idx = idxs[:-1]
    pixfirst_idx = idxs[1:]

    # now to find an echelle grating that has mlam = m'lam' for unknown m/m' as optimization problem
    params = Parameters()
    params.add('delta', min=0, max=np.pi/2, expr=f'asin(m0*{maxw.to(u.nm).value}/(2*d))')
    params.add('d', groove_init)
    params.add('m0', m0_init, min=1)
    params.add('lam1', maxw.to(u.nm).value-wave_FWHM*sep, min=minw.to(u.nm).value, max=maxw.to(u.nm).value)
    params.add('m1', min=2, expr=f'm0+1')

    def fit_func(params, ):
        return

    # fit the oversimplified model for last pixel to the grating equation results
    opt_params = minimize(fcn=fit_func, params=params, args=())
    #grating = Grating()
    #spectro = Spectrograph()

    if plot:
        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        axes = ax.ravel()
        plt.suptitle(f'R_0 = {R0}, Peak Separation = {sep}')

        # 0 is the full spectrum blaze
        axes[0].grid()

        # 1 is the hist of pix 0
        axes[1].grid()
        summed1 = np.full_like(phase_array, 0)
        for n, i in enumerate(pixfirst_idx):
            g = gauss(full_phase, full_phase[i], phase_FWHM/(2*np.sqrt(2*np.log(2))), 1)
            summed1 += g
            axes[1].plot(phase_array, g, label=f'm{n}')
            axes[1].set_xlim([minw_phase, maxw_phase])
            axes[1].set_title("First Pixel Model (m0 peak at last pixel's m1 peak)")
            axes[1].set_ylabel('Relative Transmission')
            axes[1].set_xlabel(r"Phase ($\times \pi /2$)")
            axes[1].legend()
        axes[1].plot(phase_array, summed1, 'k', alpha=0.5)
        plt.tight_layout()

        # 2 is the hist of pix 2048
        axes[2].grid()
        summed2 = np.full_like(phase_array, 0)
        for n, i in enumerate(pixlast_idx):
            g = gauss(phase_array, phase_array[i], phase_FWHM/(2*np.sqrt(2*np.log(2))), 1)
            summed2 += g
            axes[2].plot(phase_array, g, label=f'm{n}')
            axes[2].set_xlim([minw_phase, maxw_phase])
            axes[2].set_title('Last Pixel Model (800nm peak at m0)')
            axes[2].set_ylabel('Relative Transmission')
            axes[2].set_xlabel(r"Phase ($\times \pi /2$)")
            axes[2].legend()
        axes[2].plot(phase_array, summed2, 'k', alpha=0.5)
        plt.tight_layout()
        plt.show()

    #return spectro

if __name__ == '__main__':
    desired_grating(7.5, 2.1)
    #desired_grating(7.5, 1.33)
    #desired_grating(15, 2.1)
    #desired_grating(15, 1.33)
