"""Purpose of file is to determine required R for optimal efficiency of stock gratings and to determine
order efficiency for MKIDs of fiducial R"""

import numpy as np
import scipy.interpolate as interp
import scipy
import scipy.signal as sig
import scipy.ndimage as ndi
import matplotlib.pyplot as plt
import time
import astropy.units as u

from momospecsim.spectrograph import GratingSetup, SpectrographSetup, NEWPORT_GRATINGS
from momospecsim.detector import MKIDDetector

u.photlam = u.photon / u.s / u.cm ** 2 / u.AA

m = 4
R0 = 15
R0_l = 800
# c_beta = .1  # cos(beta)/cos(beta_center) at the end of m0
npix = 2048
R0 = R0
R0_l = R0_l * u.nm
l0 = 800 * u.nm
m0 = m
minwave = 400 * u.nm
maxwave = 800 * u.nm
pixels_per_res_elem = 2.5
pixel_size = 20 * u.micron  # 20 um in nm
focal_length = 300 * u.mm
MKID_FWHM_MARGIN = 1.33

# The groove length and angles need to be ~self-consistent
# From the grating equation with Littrow
# m0*l0_center=2*groove_length*np.sin(incident_angle=reflected_angle)
# We want to map l0 to the detector so
# A=dbeta/dlambda = m0/groove_length/np.cos(incident_angle)
# DELTA beta = 2*np.arctan(pixel_size*npix/2/focal_length)
# DELTA lambda = (l0_center/m0)
# two equations two unknowns:

# angular_dispersion = 2 * np.arctan(pixel_size * npix / 2 / focal_length) / (l0_center / m0)

if __name__ == '__main__':
    # Investigate grating
    detector = MKIDDetector(npix, pixel_size, R0, R0_l)
    grating = NEWPORT_GRATINGS['451E']  # or maybe 452E
    m0, m_max = 7, 14
    grating = GratingSetup(0, (34.38 * u.deg).to(u.rad).value, 0, 1e6*u.nm/147.84)  #10-19, 6-11, 4-7
    #m0, m_max = 4, 7
    #grating = GratingSetup(0, (22.24 * u.deg).to(u.rad).value, 0, 1e6*u.nm/170.87)
    
    #grating = GratingSetup(0, (15.25 * u.deg).to(u.rad).value, 0, 1e6*u.nm/184.95)
    
    spectrograph = SpectrographSetup((m0, m_max), l0, pixels_per_res_elem, focal_length, grating, detector)
    
    w = spectrograph.pixel_wavelengths()
    
    
    # littrow config, incident=reflected, chosen such that the dispersion of the order
    # containing 800 nm roughly fills the detector (or at least the fraction which is equivalent
    # to the fraction of that orders FSR that is below 800nm
    # we then also need to verify that the FSR at 800 is greater than say 1.33 MKID_FWHM
    
    # target_angular_dispersion is as high as we can go in m0 (spectrograph.angular_dispersion().mean(0)[0])
    # such that the wavelength in the pixel getting l_max in the next order is at least
    # 1.33*detector.mkid_resolution_width(l_max, 0)  (don't care about pixel as we aren't randomizing R0)
    # so spectrograph.
    
    x = []
    
    for i in np.linspace(20, grating.delta.value, num=10):
        spectrograph.m0 = 3
        spectrograph.m_max = 30
        spectrograph.set_beta_center(i)
        edges = spectrograph.edge_wave(fsr=True)
        use_orders = (edges[:, 0] <= maxwave) & (edges[:, 1] >= minwave)
        if use_orders.sum() < 2:
            continue
    
        spectrograph.m0, spectrograph.m_max = spectrograph.orders[use_orders][[0, -1]]
        spectrograph.plot_echellogram(center_orders=True)
        edges = spectrograph.edge_wave(fsr=True)
        detector_edges = spectrograph.edge_wave(fsr=False)
    
        gaps = np.array([np.select([detector_edges > edges, detector_edges <= edges],
                                   [detector_edges - edges, np.zeros_like(edges)])[:, 0],
                         np.select([detector_edges < edges, detector_edges > edges],
                                   [edges - detector_edges, np.zeros_like(edges)])[:, 1]]).T * u.nm
    
        print(f"a={i:.1f} deg m={spectrograph.m0}-{spectrograph.m_max}")
        # if (gaps > 0).any():
        #     print('Gaps found')
        print(f"Fractional coverage: {1 - gaps.sum() / (maxwave - minwave):.2f}")
    
        needed_mkid_r = detector_edges[:-1]/np.abs(np.diff(detector_edges,axis=0))*MKID_FWHM_MARGIN
        print(f"MKID R needed at long and short {needed_mkid_r[0,1]:.1f} - {needed_mkid_r[-1,0]:.1f}")
        print(f"System R: {spectrograph.average_res[0]:.0f}")
    
    
    grating = NEWPORT_GRATINGS['149E']  # or maybe 452E
    grating.empiric_blaze_factor = 0.82
    m0, m_max = 9, 19
    spectrograph = SpectrographSetup((m0, m_max), l0, pixels_per_res_elem, focal_length, grating, detector)
    spectrograph.plot_echellogram(center_orders=True, blaze=True)
    edges = spectrograph.edge_wave(fsr=True)
    detector_edges = spectrograph.edge_wave(fsr=False)
    
    gaps = np.array([np.select([detector_edges > edges, detector_edges <= edges],
                               [detector_edges - edges, np.zeros_like(edges)])[:, 0],
                     np.select([detector_edges < edges, detector_edges > edges],
                               [edges - detector_edges, np.zeros_like(edges)])[:, 1]]).T * u.nm
    
    needed_mkid_r = detector_edges[:-1] / np.abs(np.diff(detector_edges, axis=0)) * MKID_FWHM_MARGIN
    r800 = (needed_mkid_r[0, 1] * detector_edges[0, 1]) / maxwave
    print(f"  a={spectrograph.beta_central_pixel:.1f} deg m={spectrograph.m0}-{spectrograph.m_max}")
    print(f"     % coverage: {1 - gaps.sum() / (maxwave - minwave):.2f}")
    print(f"     R800 {r800:.1f}")
    print(f"     System R: {spectrograph.average_res[0]:.0f}")
    o = f'{np.log10(spectrograph.mean_blaze_eff_est(n=200).mean()):.1f}'
    print(f"     Blaze %: {spectrograph.mean_blaze_eff_est(n=200).mean():.2f} O(%)={o}")
