# global imports
import os
import numpy as np
import scipy
import astropy.units as u
import time
from datetime import datetime as dt
import logging
import argparse
import copy
import pickle

from specutils import Spectrum1D
from synphot import SpectralElement

from mkidpipeline.photontable import Photontable
from mkidpipeline.steps.buildhdf import buildfromarray  # TODO: probably defaults to MEC headers

# local imports
from momospecsim.spectra import Target, Throughput
from momospecsim.optics import Telescope, Fiber, Grating, Spectrograph
from momospecsim.detector import MKIDDetector, wave_to_phase
import momospecsim.engine as engine
from momospecsim.simsettings import SpecSimSettings
from momospecsim.utils.general import LoadFromFile

"""
Simulation of an MKID spectrometer observation.
The steps are:
    -The chosen source spectrum is loaded.
    -Atmosphere, telescope, and/or filter bandpasses may be applied.
    -It is multiplied by the blaze efficiency of the grating.
    -It is broadened according to the optical line spread function.
    -It is convolved with the MKID resolution width.
        This puts the spectrum in flux as each pixel has a different dlambda, whereas before it was in flux density.
    -The photons are randomly assigned phase and timestamp according to Poisson statistics and MKID-specific properties
     such as dead time and minimum trigger energy.
    -The photon table is saved to an h5 file.
"""

if __name__ == '__main__':
    tic = time.perf_counter()  # recording start time for script

    # ==================================================================================================================
    # PARSE COMMAND LINE ARGUMENTS
    # ==================================================================================================================
    parser = argparse.ArgumentParser(description='MKID Spectrometer Simulation')

    # general simulation args:
    parser.add_argument('--outdir', default='outdir', type=str, help='Directory for output files.')
    parser.add_argument('--plot', action='store_true', default=False, help='If passed, shows plots.')
    parser.add_argument('--simpconvol', action='store_true', default=False,
                        help='If passed, indicates that a faster, simplified MKID convolution should be conducted.')
    parser.add_argument('--waveconvol', action='store_true', default=False,
                        help='If passed, indicates that the MKID convolution is w.r.t wavelength instead of energy.')

    # observation object args:
    parser.add_argument('--spectype', default='flat', type=str,
                        help='The type of spectrum can be: "blackbody", "phoenix", "flat", "emission", '
                             '"sky_emission", or "from_file". "sky_emission" overrides "on_sky", will be True.')
    parser.add_argument('--on_sky', action='store_true', default=False,
                        help='If passed, the observation is conducted "on-sky" instead of in the laboratory and'
                             'indicates the spectrum will be atmospherically/telescopically attenuated, take an '
                             'additional throughput hit from the long fiber coupling, tip/tilt, and seeing, and have'
                             'night sky emission lines added in.')
    parser.add_argument('-et', '--exptime', default=250, type=float, help='The exposure time [sec].')
    parser.add_argument('-sf', '--spec_file', default=None,
                        help='Directory/filename of spectrum, REQUIRED if spectrum is "emission" or "from_file".')
    parser.add_argument('-dist', type=float, default=5,  # Sirius A, brightest star in the night sky
                        help='Distance to target star [parsecs], used if spectrum is "blackbody"/"phoenix".')
    parser.add_argument('-rad', default=1, type=float,
                        help='Radius of target star [# of R_sun], used if spectrum is "blackbody"/"phoenix".')
    parser.add_argument('-T', default=4000, type=float,
                        help='Temperature of target in K, used if spectrum is "blackbody"/"phoenix".')
    parser.add_argument('--objsize', default=5, type=float, help='Angular size of object [mas].')
    parser.add_argument('--seeing', default=1, type=float, help='Seeing disk diameter [arcsec].')

    # not required telescope/fiber args:
    parser.add_argument('--telename', default=None, type=str,
                        help='Telescope or system filename for throughput calculation, used if "on_sky" is True.')
    parser.add_argument('-aperture', default=None, type=float, help='Telescope aperture [mm].')
    parser.add_argument('--telefocal', default=None, type=float, help='Telescope focal length [mm].')
    parser.add_argument('-tfn','--telefibername', default=None, type=str,
                        help='On-sky fiber filename for throughput calculation.')
    parser.add_argument('-tfl', '--telefiberlength', default=None, type=float, 
                        help='On-sky fiber length [cm].')
    parser.add_argument('-tfNA', '--telefiberNA', default=None, type=float, 
                        help='On-sky fiber num. aperture.')
    parser.add_argument('-tfc', '--telefibercore', default=None, type=float, 
                        help='On-sky fiber core size [um].')
    parser.add_argument('-tfa', '--telefiberangle', default=None, type=float, 
                        help='On-sky fiber angle with respect to incoming light from telescope [arcsec].')
    # required fiber args:
    parser.add_argument('--objfiber_dist', default=5, type=float, 
                        help='Distance between object (or telefiber) and fiber array [mm].')
    parser.add_argument('-fan', '--fiberarrayname',
                        default='../momospecsim/simfiles/thruput/FG025LJA 0_10 NA.txt', type=str,
                        help='Fiber array filename for throughput calculation.')
    parser.add_argument('-fal', '--fiberarraylength', default=25, type=float, 
                        help='Fiber array length [cm].')
    parser.add_argument('-faNA', '--fiberarrayNA', default=0.1, type=float, 
                        help='Fiber array num. aperture.')
    parser.add_argument('-fac','--fiberarraycore', default=25, type=float, 
                        help='Fiber array core size [um].')
    
    # spectrograph/detector args:
    parser.add_argument('--minw', default=330, type=float, help='The min operating wavelength [nm].')
    parser.add_argument('--maxw', default=850, type=float, help='The max operating wavelength [nm].')
    parser.add_argument('--npix', default=2048, type=int, help='The linear # of pixels in the array.')
    parser.add_argument('--pix_size', default=20, type=float,
                        help='The width of the MKID pixel in the dispersion direction [um].')
    parser.add_argument('-R0', default=15, type=float, help='The R at the defined wavelength l0.')
    parser.add_argument('-l0', default=800,
                        help="The wavelength for which R0 is defined [nm]. Can be 'same' to be equal to 'maxw' arg.")
    parser.add_argument('--osamp', default=10, type=int,
                        help='# of samples to use for the smallest dlambda [or dEnergy] during convolution.')
    parser.add_argument('--nsig', default=3, type=float,
                        help='# of sigmas to use in Gaussian kernel during convolution.')
    parser.add_argument('--alpha', default=28.3, type=float, help='Angle of incidence [deg].')
    parser.add_argument('--beta', default='littrow',
                        help="Diffraction angle at the central pixel [deg]. Pass 'littrow' to be equal to 'alpha'.")
    parser.add_argument('--delta', default=63, type=float, help='Blaze angle [deg].')
    parser.add_argument('-d', '--groove_length', default=3164.56, type=float,
                        help='The groove length of the grating [nm].')  # 316 lines per mm
    parser.add_argument('--m0', default=4, type=int, help='The initial order.')
    parser.add_argument('--m_max', default=7, type=int, help='The final order.')
    parser.add_argument('-ppre', '--pixels_per_res_elem', default=2.5, type=float,
                        help='Number of pixels per spectrograph resolution element.')
    parser.add_argument('--focal_length', default=300, type=float, help='Focal length of detector [mm].')
    parser.add_argument('-rs', '--randomseed', default=10, type=int,
                        help='Random seed for detector reproducibility.')
    parser.add_argument('--resid_file', default='outdir/resids.csv', type=str,
                        help="Filename of the resonator IDs, will be created if it doesn't exist.")

    # get args by importing from arguments file instead:
    parser.add_argument('--args_file', default=None, type=open, action=LoadFromFile,
                        help='.txt file with arguments written exactly as they would be in the command line.'
                             'Pass only this argument if being used. See "simulate_args.txt" for example.')

    # get arguments:
    args = parser.parse_args()

    E_convol = False if args.waveconvol else True  # changes simulation to convolve with either energy or wavelength

    # ==================================================================================================================
    # CHECK FOR OR CREATE DIRECTORIES
    # ==================================================================================================================
    now = dt.now()
    for d in [args.outdir, os.path.dirname(args.resid_file)]:
        try:
            os.makedirs(name=d, exist_ok=True)
        except FileNotFoundError:
            pass

    # ==================================================================================================================
    # START LOGGING TO FILE
    # ==================================================================================================================
    logger = logging.getLogger('simulate')
    logging.basicConfig(level=logging.INFO)
    logger.info(msg=f"An MKID spectrometer observation of a(n) {args.spectype} spectrum is being simulated."
                    f"\nThe date and time are: {now.strftime('%Y-%m-%d %H:%M:%S')}.")

    # ==================================================================================================================
    # INSTANTIATE ALL CLASSES
    # ==================================================================================================================
    target = Target(spectype=args.spectype, 
                    dist=args.dist, 
                    rad=args.rad, 
                    temp=args.T, 
                    spec_file=args.spec_file,
                    minwave=args.minw,
                    maxwave=args.maxw,
                    objsize=args.objsize,
                    seeing=args.seeing,
                    on_sky=args.on_sky)
    telescope = Telescope(aperture=args.aperture, focal_length=args.telefocal, filename=args.telename)
    telefiber = Fiber(filename=args.telefibername, 
                      num_aperture=args.telefiberNA, 
                      length=args.telefiberlength,
                      core_size=args.telefibercore,
                      incident_angle=args.telefiberangle)
    fiberarray = Fiber(filename=args.fiberarrayname,
                       num_aperture=args.fiberarrayNA,
                       length=args.fiberarraylength,
                       core_size=args.fiberarraycore)
    detector = MKIDDetector(npix=args.npix,
                            pix_size=args.pix_size,
                            design_R0=args.R0,
                            l0=args.l0,
                            randomseed=args.randomseed,
                            resid_file=args.resid_file)
    grating = Grating(alpha=args.alpha, delta=args.delta, beta_center=args.beta, groove_length=args.groove_length)
    spectro = Spectrograph(m0=args.m0,
                           m_max=args.m_max,
                           final_wave=args.l0,
                           pixels_per_res_elem=args.pixels_per_res_elem,
                           focal_length=args.focal_length,
                           grating=grating,
                           detector=detector)
    eng = engine.Engine(spectrograph=spectro)
    
    # shorten commonly used properties:
    #nord = spectro.nord  # number of orders
    #lambda_pixel = spectro.pixel_wavelengths().to(u.nm)  # expected wavelength at pixel center

    # ==================================================================================================================
    # SIMULATION STARTS
    # ==================================================================================================================
    # initialize spectrum:
    if args.on_sky:
        target.size = target.init_size(aperture=telescope.aperture, fiber_angle=telefiber.accept_angle)
    else:
        target.size = target.init_size(fiber_angle=fiberarray.accept_angle)
    target.spectrum = target.init_spectrum()  # applies fov (and sky emission lines)
    target.waveset = target.spectrum.waveset
    if args.plot:
        target.plot(title='Initial Spectrum')

    # multiply by atmosphere and telescope throughput, still flux density:
    if args.on_sky:
        target.spectrum *= (Throughput('../momospecsim/simfiles/atmosphere/transmission.dat') * telescope.thruput)
    
    # attenuate spectrum and size via telescope fiber (if any):
    telefiber.attenuate(target=target, aperture=telescope.aperture, fnum=telescope.fnum)
    
    # have light diverge (either from telefiber or object):
    target.diverge(fiber=fiberarray, distance=args.objfiber_dist)
    
    # attenuate spectrum via fiber array:
    fiberarray.attenuate(target=target)  # TODO figure out if this works fiber to fiber and obj to fiber
    if args.plot:
        target.plot(title='Telescope/Fiber/Atmo-Attenuated Spectrum')

    # attenuate through spectrograph collimating lens, focusing lens, 2 fridge filters, and detector fill-factor:
    target.spectrum *= (Throughput('../momospecsim/simfiles/thruput/AC508-150-A.csv') *
                        Throughput('../momospecsim/simfiles/thruput/ACT508-300-A.csv') *
                        Throughput('../momospecsim/simfiles/thruput/Asahi_YSC0750.csv') *
                        Throughput('../momospecsim/simfiles/thruput/Asahi_YSC0750.csv')
                        * 0.87)  # detector fill-factor and absorption (need to do QE measurement
    
    # blaze, attenuate, and broaden via grating:
    target.clip_spectrum()  # clipping to useful range, arbitrarily discarding near-zero regions
    target.spectrum = target.spectrum(target.waveset) * spectro.blaze(target.waveset)
    if args.plot:
        spectro.blaze_plot(title='Blazed Spectrum', waves=target.waveset, spectrum=target.spectrum)
    
    blazed_spectrum = copy.deepcopy(target.spectrum)

    # optically-broaden spectrum (convolution with line spread function):
    target.optically_broaden(nondim_lsf_width=spectro.nondimensional_lsf_width)

    # convolve with MKID resolution widths:
    convol_wave, convol_result, mkid_kernel = eng.convolve_mkid_response(wave=target.waveset,
                                                                         spectral_fluxden=target.spectrum,
                                                                         oversampling=args.osamp,
                                                                         n_sigma_mkid=args.nsig,
                                                                         simp=args.simpconvol,
                                                                         energy=E_convol)

    # conduct MKID observation sequence:
    photons, observed, reduce_factor = detector.observe(convol_wave=convol_wave, convol_result=convol_result,
                                                        minwave=args.minw * u.nm, maxwave=args.maxw * u.nm, energy=E_convol,
                                                        randomseed=args.randomseed, exptime=args.exptime * u.s)

    sim = SpecSimSettings(outdir=args.outdir,
                          simpconvol=args.simpconvol,
                          waveconvol=args.waveconvol,
                          target=target,
                          telescope=telescope,
                          telefiber=telefiber,
                          fiberarray=fiberarray,
                          spectrograph=spectro,
                          detector=detector)

    # saving final photon list to h5 file, store linear phase conversion in header:
    h5_file = f'{args.outdir}/{args.spectype}.h5'
    
    target.photonlist = photons[:observed]
    if args.plot:
        target.plot_heatmap(detector)

    # buildfromarray(array=target.photonlist, user_h5file=h5_file)
    # pt = Photontable(file_name=h5_file, mode='write')
    # pt.update_header(key='sim_settings', value=sim)
    # pt.disablewrite()  # allows other scripts to open the table
    # 
    # logger.info(msg=f'Saved spectrum photon table to {h5_file}.')

    if args.plot:
        target.plot_comparison(spectro, detector, eng, convol_result, blazed_spectrum, reduce_factor, args.exptime)
    
    logger.info(msg=f'Simulation complete. Total time: {((time.perf_counter() - tic) / 60):.2f} min. Exiting.')

