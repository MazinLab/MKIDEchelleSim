import numpy as np
import scipy.interpolate as interp
import scipy
import scipy.signal as sig
import scipy.ndimage as ndi
import matplotlib.pyplot as plt
import time

import astropy.units as u
from momospecsim.spectra import PhoenixModel, AtmosphericTransmission, FilterTransmission, TelescopeTransmission
from momospecsim.spectrograph import GratingSetup, SpectrographSetup
from momospecsim.detector import MKIDDetector
from momospecsim.engine import Engine
from momospecsim.spectra import clip_spectrum


tic = time.time()

def minsep(m, R0=15, R0_l=800):
    u.photlam = u.photon / u.s / u.cm ** 2 / u.AA
    exptime = 1 * u.s
    # c_beta = .1  # cos(beta)/cos(beta_center) at the end of m0
    npix = 2048
    R0 = R0
    R0_l = R0_l * u.nm
    l0 = 800 * u.nm
    m0 = m

    l0_center = l0 / (1 + 1 / (2 * m0))
    # l0_fsr = l0_center/m0
    R0_min = .8 * R0
    n_sigma_mkid = 3
    osamp = 10
    minwave = 400 * u.nm
    maxwave = 800 * u.nm
    pixels_per_res_elem = 2.5
    pixel_size = 20 * u.micron  # 20 um in nm
    focal_length = 300 * u.mm

    m_max = m
    mw = l0_center * m0 / m_max * (1 - 1 / m_max / 2)
    while mw > minwave:
        m_max += 1
        mw = l0_center*m0/m_max * (1- 1/m_max/2)
    # m_max = m+4

    # The groove length and angles need to be ~self-consistent
    # From the grating equation with Littrow
    # m0*l0_center=2*groove_length*np.sin(incident_angle)
    # We want to map l0 to the detector so
    # A=dbeta/dlambda = m0/groove_length/np.cos(incident_angle)
    # DELTA beta = 2*np.arctan(pixel_size*npix/2/focal_length)
    # DELTA lambda = (l0_center/m0)
    # two equations two unknowns:

    angular_dispersion = 2 * np.arctan(pixel_size * npix / 2 / focal_length) / (l0_center / m0)
    incident_angle = np.arctan((l0_center / 2 * angular_dispersion).value) * u.rad
    groove_length = m0 * l0_center / 2 / np.sin(incident_angle)
    blaze_angle = incident_angle  # +10*u.deg  # a good bit off blaze

    detector = MKIDDetector(npix, pixel_size, R0, R0_l)
    grating = GratingSetup(incident_angle, blaze_angle, incident_angle, groove_length)
    spectrograph = SpectrographSetup(m0, m_max, l0, pixels_per_res_elem, focal_length, grating, detector)

    for s in spectrograph.info_str():
        print(s)

    pw = spectrograph.pixel_wavelengths()
    mw = detector.mkid_resolution_width(pw, detector.pixel_indices)
    x = (-np.diff(pw, axis=0)/mw[:-1]).min(1)
    print(x[::-1])
    return spectrograph

# Grating throughput impact, is a function of wavelength and grating angles, needs to be handled for each order
# blaze_efficiencies = spectrograph.blaze(pw)

# order_mask = spectrograph.order_mask(inbound.waveset, fsr_edge=False)
# for o,m, b in zip(spectrograph.orders, order_mask, blaze_efficiencies):
#     plt.plot(inbound.waveset[m], b[m], label=f'Order {o}')
# plt.legend()
# plt.show()


"""
R15@800 aim for m10-19
R10@800 aiming for m6-11 ok
R7.5@800 aim for m4-7

"""
