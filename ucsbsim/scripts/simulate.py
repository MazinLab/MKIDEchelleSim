import os
import sys
import warnings
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import scipy
import scipy.interpolate as interp
import astropy.units as u
import time
from datetime import datetime as dt
import logging
import argparse
from specutils import Spectrum1D
from synphot import SpectralElement

from mkidpipeline.photontable import Photontable
from ucsbsim.mkidspec.spectra import get_spectrum, apply_bandpass, AtmosphericTransmission, FilterTransmission, \
    TelescopeTransmission, FineGrid, clip_spectrum
from ucsbsim.mkidspec.spectrograph import GratingSetup, SpectrographSetup
from ucsbsim.mkidspec.detector import MKIDDetector, wave_to_phase
import ucsbsim.mkidspec.engine as engine
from ucsbsim.mkidspec.plotting import quick_plot
from ucsbsim.mkidspec.simsettings import SpecSimSettings

# TODO this will need work as the pipeline will probably default to MEC HDF headers
from mkidpipeline.steps.buildhdf import buildfromarray

"""
Simulation of an MKID spectrograph observation run. The steps are:
-A source spectrum is loaded.
-Atmosphere, telescope, and/or filter bandpasses may be applied depending on whether source originates from sky or lab.
-It is multiplied by the blaze efficiency of the grating, which determines how much of the flux that got through the
 filter(s) will be incident on each pixel of the detector.
-It is broadened according to the optical LSF, the limit to the resolution of the optics used.
-It is convolved with the MKID resolution width as a function of wavelength to simulate MKID wavelength discrimination.
 This puts the spectrum into pixel space (flux) as each pixel has a different dlambda, whereas before it was in
  wavelength space (flux density).
-The photons are randomly assigned phase and timestamp according to Poisson statistics and MKID-specific properties
 such as dead time and minimum trigger energy.
-The photon table is saved to an h5 file.
"""


class LoadFromFile(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        with values as f:
            # parse arguments in the file and store them in the target namespace
            parser.parse_args(f.read().split(), namespace)


if __name__ == '__main__':
    tic = time.perf_counter()  # recording start time for script

    # ==================================================================================================================
    # PARSE COMMAND LINE ARGUMENTS
    # ==================================================================================================================
    parser = argparse.ArgumentParser(description='MKID Spectrograph Simulation')

    # optional simulation args:
    parser.add_argument('--type_spectra', default='flat', type=str,
                        help='The type of spectra: can be "blackbody", "phoenix", "flat", "emission", '
                             '"sky_emission", or "from_file".')
    parser.add_argument('--outdir', default='outdir', type=str, help='Directory for output files.')
    parser.add_argument('-rs', '--random_seed', default=10, type=int,
                        help='Random seed for reproducibility.')
    parser.add_argument('--R0s_file', default='R0s.csv', type=str,
                        help="Filename of the R0s file, will be created if it doesn't exist.")
    parser.add_argument('--phaseoffset_file', default='phase_offsets.csv', type=str,
                        help="Filename of the phase offset file, will be created if it doesn't exist.")
    parser.add_argument('--resid_file', default='resids.csv', type=str,
                        help="Filename of the resonator IDs, will be created if it doesn't exist.")
    parser.add_argument('-sf', '--spec_file', default=None,
                        help='Directory/filename of spectrum, REQUIRED if spectra is "emission" or "from_file".')
    parser.add_argument('-dist', type=float, default=5,  # Sirius A, brightest star in the night sky
                        help='Distance to target star in parsecs, REQUIRED if spectra is "phoenix".')
    parser.add_argument('-rad', default=1, type=float,
                        help='Radius of target star in R_sun, REQUIRED if spectra is "phoenix".')
    parser.add_argument('-T', default=4000, type=float,
                        help='Temperature of target in K, REQUIRED if spectra is "blackbody" or "phoenix".')
    parser.add_argument('-et', '--exptime', default=250, type=float,
                        help='The total exposure time of the observation [sec].')
    parser.add_argument('--telearea', default=np.pi * 4 ** 2, type=float, help='The telescope area [cm2].')
    parser.add_argument('--fov', default=1, type=float, help='Field of view [arcsec2].')
    parser.add_argument('--simpconvol', action='store_true', default=False,
                        help='If passed, indicates that a faster, simplified convolution should be conducted.')
    parser.add_argument('--on_sky', action='store_true', default=False,
                        help='If passed, the observation is conducted "on-sky" instead of in the laboratory and'
                             'indicates the spectrum will be atmospherically/telescopically attenuated and have'
                             'night sky emission lines added in.')
    parser.add_argument('--reflect', default=0.9, type=float,
                        help='Factor to attenuate spectrum due to telescope reflectivity, between 0 and 1, '
                             'REQUIRED if "on_sky" is True.')
    parser.add_argument('--minw', default=330, type=float, help='The min operating wavelength [nm].')
    parser.add_argument('--maxw', default=850, type=float, help='The max operating wavelength [nm].')

    # optional spectrograph args:
    parser.add_argument('--npix', default=2048, type=int, help='The linear # of pixels in the '
                                                               'MKID detector. TODO upgrade for multiple rows.')
    parser.add_argument('--pixsize', default=20, type=float,
                        help='The width of the MKID pixel in the dispersion direction [um].')
    parser.add_argument('-R0', default=15, type=float, help='The R at the defined wavelength l0.')
    parser.add_argument('-l0', default='same',
                        help="The wavelength for which R0 is defined [nm]. Can be 'same' to be equal to 'maxw' arg.")
    parser.add_argument('--osamp', default=10, type=int,
                        help='# of samples to use for the smallest dlambda during convolution.')
    parser.add_argument('--nsig', default=3, type=float,
                        help='# of sigma to use for Gaussian during convolution.')
    parser.add_argument('--alpha', default=28.3, type=float, help='Angle of incidence [deg].')
    parser.add_argument('--beta', default='littrow',
                        help="Diffraction angle at the central pixel [deg]. Pass 'littrow' to be equal to 'alpha'.")
    parser.add_argument('--delta', default=63, type=float, help='Blaze angle [deg].')
    parser.add_argument('-d', '--groove_length', default=((1 / 316) * u.mm).to(u.nm).value, type=float,
                        help='The groove length of the grating [nm].')
    parser.add_argument('--m0', default=4, type=int, help='The initial order.')
    parser.add_argument('--m_max', default=7, type=int, help='The final order.')
    parser.add_argument('-ppre', '--pixels_per_res_elem', default=2.5, type=float,
                        help='Pixels per spectrograph resolution element.')
    parser.add_argument('--focallength', default=300, type=float,
                        help='The focal length of the detector [mm].')
    parser.add_argument('--plot', action='store_true', default=False, help='If passed, shows final plots.')

    # get optional args by importing from arguments file:
    parser.add_argument('--args_file', default=None, type=open, action=LoadFromFile,
                        help='.txt file with arguments written exactly as they would be in the command line.')

    # get arguments & simulation settings
    args = parser.parse_args()
    sim = SpecSimSettings(
        minwave_nm=args.minw,
        maxwave_nm=args.maxw,
        npix=args.npix,
        pixelsize_um=args.pixsize,
        designR0=args.R0,
        l0_nm=args.l0,
        alpha_deg=args.alpha,
        delta_deg=args.delta,
        beta_deg=args.beta,
        groove_length_nm=args.groove_length,
        m0=args.m0,
        m_max=args.m_max,
        pixels_per_res_elem=args.pixels_per_res_elem,
        focallength_mm=args.focallength,
        R0s_file=args.R0s_file,
        phaseoffset_file=args.phaseoffset_file,
        resid_file=args.resid_file,
        type_spectra=args.type_spectra,
        spec_file=args.spec_file,
        exptime_s=args.exptime,
        telearea_cm2=args.telearea,
        distance_ps=args.dist,
        radius_Rsun=args.rad,
        temp_K=args.T,
        on_sky=args.on_sky,
        simpconvol=args.simpconvol
    )

    # ==================================================================================================================
    # CHECK FOR OR CREATE DIRECTORIES
    # ==================================================================================================================
    now = dt.now()
    try:
        os.makedirs(name=args.outdir, exist_ok=True)
    except FileNotFoundError:
        pass
    try:
        os.makedirs(name=os.path.dirname(sim.R0s_file), exist_ok=True)
    except FileNotFoundError:
        pass
    try:
        os.makedirs(name=os.path.dirname(sim.phaseoffset_file), exist_ok=True)
    except FileNotFoundError:
        pass
    try:
        os.makedirs(name=os.path.dirname(sim.resid_file), exist_ok=True)
    except FileNotFoundError:
        pass

    # ==================================================================================================================
    # START LOGGING TO FILE
    # ==================================================================================================================
    logger = logging.getLogger('simulate')
    logging.basicConfig(level=logging.INFO)
    logger.info(msg=f"The simulation of a(n) {sim.type_spectra} spectrum observation is recorded."
                     f"\nThe date and time are: {now.strftime('%Y-%m-%d %H:%M:%S')}.")

    # ==================================================================================================================
    # INSTANTIATE SPECTROGRAPH & DETECTOR
    # ==================================================================================================================
    np.random.seed(args.random_seed)
    logger.info(f'The random seed has been set to {args.random_seed}.')
    
    try:  # check for the spectral resolution file, create if not exist
        R0s = np.loadtxt(fname=sim.R0s_file, delimiter=',')
        logger.info(f'The pixel Rs @ {sim.l0} nm were imported from {sim.R0s_file}.')
    except IOError:
        R0s = np.random.uniform(low=.85, high=1.15, size=sim.npix) * sim.designR0
        np.savetxt(fname=sim.R0s_file, X=R0s, delimiter=',')
        logger.info(msg=f'The pixel Rs @ {sim.l0} were randomly generated from R0 and saved to {sim.R0s_file}.')

    try:  # check for the phase offset file, create if not exist
        phase_offsets = np.loadtxt(fname=sim.phaseoffset_file, delimiter=',')
        logger.info(msg=f'The pixel center phase offsets were imported from {sim.phaseoffset_file}.')
    except IOError:
        phase_offsets = np.random.uniform(low=.8, high=1.2, size=sim.npix)
        np.savetxt(fname=sim.phaseoffset_file, X=phase_offsets, delimiter=',')
        logger.info(msg=f'The pixel phase offsets were generated randomly and saved to {sim.phaseoffset_file}.')

    try:  # check for the resonator IDs, create if not exist
        resid_map = np.loadtxt(fname=sim.resid_file, delimiter=',')
        logger.info(msg=f'The resonator IDs were imported from {sim.resid_file}.')
    except IOError:
        resid_map = np.arange(sim.npix, dtype=int) * 10 + 100
        np.savetxt(fname=sim.resid_file, X=resid_map, delimiter=',')
        logger.info(msg=f'The resonator IDs were generated from {resid_map.min()} to {resid_map.max()}.')

    detector = MKIDDetector(n_pix=sim.npix, pixel_size=sim.pixelsize, design_R0=sim.designR0, l0=sim.l0, R0s=R0s,
                            phase_offsets=phase_offsets, resid_map=resid_map)
    spectro = SpectrographSetup(order_range=sim.order_range, final_wave=sim.l0,
                                pixels_per_res_elem=sim.pixels_per_res_elem,
                                focal_length=sim.focallength, grating=sim.grating, detector=detector)
    eng = engine.Engine(spectrograph=spectro)

    # shorten commonly used properties:
    nord = spectro.nord
    lambda_pixel = spectro.pixel_wavelengths().to(u.nm)
    lambda_left = spectro.pixel_wavelengths(edge='left')

    # ==================================================================================================================
    # SIMULATION STARTS
    # ==================================================================================================================

    # obtaining spectra:
    spectrum = get_spectrum(
        spectrum_type=sim.type_spectra,
        distance=sim.distance,
        radius=sim.radius,
        teff=sim.temp,
        spec_file=sim.spec_file,
        minwave=sim.minwave,
        maxwave=sim.maxwave,
        on_sky=sim.on_sky,
        fov=args.fov
    )  # though all args are passed, type_spectra determines which will be used

    # populate bandpasses based on on-sky or lab observation and always have finer grid spacing and min/max filter:
    bandpasses = [FineGrid(min=sim.minwave, max=sim.maxwave), FilterTransmission()]
    if sim.on_sky:
        bandpasses.append(AtmosphericTransmission())
        bandpasses.append(TelescopeTransmission(reflectivity=args.reflect))

    # apply bandpasses:
    bandpass_spectrum = apply_bandpass(spectra=spectrum, bandpass=bandpasses)

    # clip spectrum in order to blaze within limits
    clipped_spectrum = clip_spectrum(x=bandpass_spectrum, clip_range=(sim.minwave, sim.maxwave))

    # blaze spectrum and directly integrate into pixel space to verify:
    blazed_spectrum, masked_waves, masked_blaze = eng.blaze(wave=clipped_spectrum.waveset, spectra=clipped_spectrum(clipped_spectrum.waveset))
    blazed_int_spec = np.array([
        eng.lambda_to_pixel_space(
            array_wave=clipped_spectrum.waveset,
            array=blazed_spectrum[i],
            leftedge=lambda_left[i]
        ) for i in range(nord)
    ])

    # optically-broaden spectrum:
    broadened_spectrum = eng.optically_broaden(wave=clipped_spectrum.waveset, flux=blazed_spectrum)
    broad_int_spec = np.array([
        eng.lambda_to_pixel_space(
            array_wave=clipped_spectrum.waveset,
            array=broadened_spectrum[i],
            leftedge=lambda_left[i]
        ) for i in range(nord)
    ])

    # conducting the convolution with MKID resolution widths:
    convol_wave, convol_result, mkid_kernel = eng.convolve_mkid_response(wave=clipped_spectrum.waveset,
                                                                         spectral_fluxden=broadened_spectrum,
                                                                         oversampling=args.osamp,
                                                                         n_sigma_mkid=args.nsig, simp=sim.simpconvol)

    # putting convolved spectrum through MKID observation sequence:
    photons, observed, reduce_factor = detector.observe(convol_wave=convol_wave, convol_result=convol_result,
                                                        minwave=sim.minwave, maxwave=sim.maxwave,
                                                        exptime=sim.exptime, area=sim.telearea)

    # saving final photon list to h5 file, store linear phase conversion in header:
    h5_file = f'{args.outdir}/{sim.type_spectra}.h5'
    buildfromarray(array=photons[:observed], user_h5file=h5_file)
    pt = Photontable(file_name=h5_file, mode='write')
    pt.update_header(key='sim_settings', value=sim)
    pt.update_header(key='phase_expression', value='0.6 * (freq_allwave - freq_minw) / (freq_maxw - freq_minw) - 0.8')
    pt.disablewrite()  # allows other scripts to open the table

    # at this point we are simulating the pipeline and have gone past the "wavecal" part. Next is >to spectrum.
    logger.info(msg=f'Saved photon table to {h5_file}.')
    logger.info(msg=f'Total simulation time: {((time.perf_counter() - tic) / 60):.2f} min.')
    # ==================================================================================================================
    # SIMULATION ENDS
    # ==================================================================================================================

    # ==================================================================================================================
    # DEBUGGING PLOTS:
    # ==================================================================================================================
    if args.plot:
        warnings.filterwarnings(action="ignore")  # ignore tight_layout warnings

        """Calculations for plotting"""
        # separate photons by resid (pixel) and realign (no offset):
        idx = [np.where(photons[:observed].resID == resid_map[j]) for j in range(sim.npix)]
        photons_realign = [(photons[:observed].wavelength[idx[j]] / phase_offsets[j]).tolist() for j in range(sim.npix)]

        masked_broad = [eng.optically_broaden(wave=masked_waves[i], flux=masked_blaze[i], axis=0) for i in range(nord)]

        # integrating the convolution to go to pixel-order array size:
        convol_int = np.sum(convol_result, axis=0)

        # use FSR to bin and order sort:
        fsr = spectro.fsr(order=spectro.orders).to(u.nm)
        hist_bins = np.empty((nord + 1, sim.npix))  # choosing rough histogram bins by using FSR of each pixel/wave
        hist_bins[0, :] = (lambda_pixel[-1, :] - fsr[-1] / 2).value
        hist_bins[1:, :] = [(lambda_pixel[i, :] + fsr[i] / 2).value for i in range(nord)[::-1]]
        hist_bins = wave_to_phase(waves=hist_bins, minwave=sim.minwave, maxwave=sim.maxwave)

        photons_binned = np.empty((nord, sim.npix))
        for j in range(sim.npix):
            photons_binned[:, j], _ = np.histogram(a=photons_realign[j], bins=hist_bins[:, j], density=False)

        # normalize to level of convolution since that's where it came from and calculate noise:
        photons_binned = (
                photons_binned * u.ph * reduce_factor[None, :] / (sim.exptime.to(u.s) * sim.telearea.to(u.cm ** 2))).to(
            u.ph / u.cm ** 2 / u.s).value

        """Plotting only below"""
        # phase/pixel space plot to verify that phases are within proper values and orders are more-or-less visible
        bin_edges = np.linspace(-1, -0.1, 100)
        centers = bin_edges[:-1] + np.diff(bin_edges) / 2
        hist_array = np.zeros([sim.npix, len(bin_edges) - 1])
        for j in detector.pixel_indices:
            if photons_realign[j]:
                counts, edges = np.histogram(a=photons_realign[j], bins=bin_edges)
                hist_array[j, :] = np.array([float(x) for x in counts])
        plt.imshow(hist_array[:, ::-1].T, extent=[1, sim.npix, -1, -0.1], aspect='auto', norm=LogNorm())
        cbar = plt.colorbar()
        cbar.ax.set_ylabel('Photon Count')
        plt.title("Phase & Pixel Binning of Observed Photons")
        plt.xlabel("Pixel Index")
        plt.ylabel(r"Phase ($\times \pi /2$)")
        plt.tight_layout()
        plt.show()
        
        fig, axes = plt.subplots(2, 1, figsize=(11, 8.5))
        axes = axes.ravel()
        plt.suptitle(f"Intermediate plots for MKIDSpec {sim.type_spectra} spectrum simulation", fontweight='bold')

        # plotting bandpassed and blazed/broadened spectrum:
        quick_plot(ax=axes[0], x=[bandpass_spectrum.waveset.to(u.nm)],
                   y=[bandpass_spectrum(bandpass_spectrum.waveset) / np.max(
                       bandpass_spectrum(bandpass_spectrum.waveset))],
                   labels=['Incident, R~50,000'], color='r', first=True, xlim=[300, 900])
        broad_max = np.max([np.max(masked_broad[i].value) for i in range(nord)])
        quick_plot(ax=axes[0], x=masked_waves,
                   y=[masked_broad[i] / broad_max for i in range(nord)], color='g',
                   labels=['Blazed+Broadened, R~3500'] + ['_nolegend_' for o in spectro.orders[:-1]],
                   title="Comparison of Input (Instrument-incident) and Blazed/Opt.-Broadened Spectra (not relative)",
                   ylabel=r"Normalized Flux Density")

        # plotting comparison between flux-integrated spectrum and integrated/convolved spectrum, must be same,
        # also plotting final counts FSR-binned
        quick_plot(ax=axes[1], x=lambda_pixel, y=photons_binned[::-1], color='k', linewidth=1, linestyle='--',
                   labels=['Obs. Count to Flux'] + ['_nolegend_' for o in spectro.orders[:-1]], first=True)
        quick_plot(ax=axes[1], x=lambda_pixel, y=convol_int, color='red', linewidth=1.5, alpha=0.4,
                   labels=["Post-Convol"] + ['_nolegend_' for o in range(nord - 1)])
        quick_plot(ax=axes[1], x=lambda_pixel, y=blazed_int_spec, color='b', ylabel=r"Flux (phot $cm^{-2} s^{-1})$",
                   labels=["Input-integrated"] + ['_nolegend_' for o in range(nord - 1)], alpha=0.4, linewidth=1.5,
                   xlabel='Wavelength (nm)', title=r"Comparison of Input (integrated to pixel space), "
                                                   r"Post-Convolution (integrated), and Observed Photon Count (FSR-binned)")
        fig.tight_layout()
        plt.show()
    pass