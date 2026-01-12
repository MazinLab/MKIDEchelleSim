# global imports
import numpy as np
import astropy.units as u
import time
import warnings
from datetime import datetime as dt
import argparse
import logging
import os
import tqdm
from numpy.polynomial.legendre import Legendre

from mkidpipeline.photontable import Photontable

# local imports
from momospecsim.steps.fitmsf import fitmsf, bin_width_from_MKID_R
from momospecsim.steps.ordersort import ordersort
from momospecsim.steps.wavecal import wavecal
from momospecsim.steps.extract import extract
from momospecsim.simsettings import SpecSimSettings
from momospecsim.msf import MKIDSpreadFunction
import momospecsim.utils.general as gen
from momospecsim.detector import wave_to_phase, phase_to_wave, sorted_table, Pixel


if __name__ == "__main__":
    tic = time.perf_counter()  # recording start time for script

    # ==================================================================================================================
    # PARSE COMMAND LINE ARGUMENTS
    # ==================================================================================================================
    parser = argparse.ArgumentParser(description='MKID Spectrometer Data Reduction')

    # optional script args:
    parser.add_argument('--outdir', default='outdir', type=str, help='Directory for the output files.')
    parser.add_argument('--plot', action='store_true', default=False, help='If passed, show plots.')
    parser.add_argument('--debug', action='store_true', default=False, help='If passed, show debug plots.')

    # optional MSF args:
    parser.add_argument('--msf', default='outdir/flat.h5',
                        help='Directory/name of the flat/blackbody spectrum photon table .h5 file OR'
                             'Directory/name of the complete MKID Spread Function .pkl file.')
    parser.add_argument('--bin_range', default=(-1.5, 0), type=tuple,
                        help='Start/stop range for phase histogram.')
    parser.add_argument('--missing_order_pix', nargs='*',
                        default=[0, 349, 3, 350, 1299, 1, 0, 1299, 13, 1300, 2047, 2, 1300, 2047, 24],
                        help='Array of [startpix, endpix, missing-orders as single digit indexed from 1, and repeat],'
                             'e.g.: 0 999 13 1000 1999 25 2000 2047 4'
                             'will become [0, 999, 13,  1000, 1999, 25,  2000, 2047, 4]'
                             'where       sta sto  ord   sta  sto  ord   sta   sto  ord')
    # TODO: eradicate missing order pix for neighboring pixel knowledge

    # optional wavecal args:
    parser.add_argument('--wavecal', default='outdir/emission.h5',
                        help='Directory/name of the emission lamp spectrum photon table .h5 file OR'
                             'Directory/name of the order-sorted emission lamp spectrum .fits file OR'
                             'Directory/name of the complete wavelength calibration solution .npz file.'
                             'Pass any other argument, such as "False", to disable this step.')
    parser.add_argument('--elem', default=None, type=str,
                        help="Emission lamp element(s) in use, i.e., 'hgar' for Mercury-Argon. Wavecal will not"
                             "be conducted if this argument is None.")
    parser.add_argument('--degree', default=4, type=int, help="Polynomial degree to use in wavecal.")
    parser.add_argument('--iters', default=5, type=int,
                        help="Number of iterations to loop through for identifying and discarding lines.")
    parser.add_argument('--manual_fit', action='store_true', default=False,
                        help="If passed, indicates user should click plot to align observation and linelist.")
    parser.add_argument('--residual_max', default=85e3, type=float,
                        help="Maximum residual allowed between fit wavelength and atlas in m/s. (float)")
    parser.add_argument('--width', default=3, type=int, help="Width in pixels for scipy.find_peaks.")
    parser.add_argument('--shift_window', default=0.05, type=float,
                        help="Fraction of columns to use in the alignment of individual orders, 0 to disable.")
    parser.add_argument('--dim', default='1D', type=str,
                        help="Return a '1D' (pixel direction) or '2D' (pixel+order directions) fitting solution.")

    # optional observation args:
    parser.add_argument('--extract', default='outdir/phoenix.h5',
                        help='Directory/name of the on-sky observation spectrum photon table .h5 file OR'
                             'Directory/name of the order-sorted observation spectrum .fits file.'
                             'Pass any other argument, such as "False", to disable this step.')
    
    # get optional args by importing from arguments file:
    parser.add_argument('--args_file', default=None, type=open, action=gen.LoadFromFile,
                        help='.txt file with arguments written exactly as they would be in the command line.'
                             'Pass only this argument if being used. See "mkidspec_args.txt" for example.')

    args = parser.parse_args()

    plot = True if args.plot or args.debug else args.plot
    
    # ==================================================================================================================
    # START LOGGING
    # ==================================================================================================================
    logger = logging.getLogger('reduce')
    logging.basicConfig(level=logging.INFO)
    logger.info(msg=f"An MKID spectrometer observation is being reduced."
                    f"\nThe date and time are: {dt.now().strftime('%Y-%m-%d %H:%M:%S')}.")

    # ==================================================================================================================
    # PARSE STEPS TO RUN
    # ==================================================================================================================
    steps = []  # list to append steps in use

    # MSF
    if args.msf.lower().endswith('.h5'):  # the MSF has yet to be fit
        msf_table = Photontable(file_name=args.msf)
        sim = msf_table.query_header('sim_settings')

        # extract resid map from file:
        resid_map = np.loadtxt(fname=sim.resid_file, delimiter=',')
        photons_pixel = sorted_table(table=msf_table, resid_map=resid_map)  # get list of photons in each pixel
        logger.info('Unwrapping photon phases.')
        for l in photons_pixel:
            l[l > 0] -= 2

        # retrieve the detector, spectrograph, and engine:
        eng = sim.engine
        spectro = eng.spectrograph
        detector = spectro.detector

        # shortening some longer variable names:
        nord = spectro.nord
        pixels = detector.pixel_indices
        pix_waves = spectro.pixel_wavelengths().to(u.nm)[::-1]  # flip order axis to be in ascending phase/lambda
        pix_E = gen.wave_to_energy(pix_waves).value  # convert to energy

        # pre-bin each pixel with the same bin edges and get centers for plotting:
        bin_width = bin_width_from_MKID_R(detector.design_R0)
        bin_edges = np.arange(args.bin_range[0], args.bin_range[1], bin_width)
        bin_centers = bin_edges[:-1] + np.diff(bin_edges) / 2

        # define phase grid for plotting/integrating more accurately
        fine_phase_grid = np.linspace(args.bin_range[0], args.bin_range[1], 1000)

        leg_e = Legendre(coef=(0, 0, 0), domain=[-1, 0])  # setup the energy Legendre object

        warnings.filterwarnings('ignore', category=RuntimeWarning)  # suppresses warning that occurs each fit

        pixel_dict = {}
        logger.info('Fitting pixel by pixel.')
        for p in tqdm.tqdm(range(int(len(pixels)/2)-1, len(pixels)-1)):  # do the non-linear least squares fit for each pixel
            pixel_dict.update({f"{p}": Pixel(p, nord, spectro.orders, resid_map[p], photons_pixel[p], bin_edges, bin_centers, pix_E[:, p], fine_phase_grid)})
            pixel_dict[f"{p}"].fit(leg_e)
            
            if pixel_dict[f"{p}"].all_orders:
                pixel_dict[f"{p}"].extract_model(leg_e)
                
            # use adjacent pixels to fit pixels with missing orders, starting from the middle
            elif pixel_dict[f"{p - 1}"].all_orders:
                # get the ratio of the order amplitudes wrt largest
                max_amp = np.max(pixel_dict[f"{p - 1}"].fit_amp)
                ratio = pixel_dict[f"{p - 1}"].fit_amp / max_amp
                pixel_dict[f"{p}"].fit(leg_e, ratio)

            pixel_dict[f"{p}"].extract_model(leg_e)
            pixel_dict[f"{p}"].get_order_edges()
            pixel_dict[f"{p}"].order_sort()
            pixel_dict[f"{p}"].plot(leg_e, args.debug)

        for p in tqdm.tqdm(range(len(pixels)-1, int(len(pixels)/2)-1, -1)):
            if not pixel_dict[f"{p}"].all_orders and pixel_dict[f"{p + 1}"].all_orders:
                # get the ratio of the order amplitudes wrt largest
                max_amp = np.max(pixel_dict[f"{p + 1}"].fit_amp)
                ratio = pixel_dict[f"{p + 1}"].fit_amp / max_amp
                pixel_dict[f"{p}"].fit(leg_e, ratio)
                pixel_dict[f"{p}"].extract_model(leg_e)
                pixel_dict[f"{p}"].get_order_edges()
                pixel_dict[f"{p}"].order_sort()
                pixel_dict[f"{p}"].plot(leg_e, args.debug)

        msf_obj = pixeldict_to_msf(pixel_dict)

    elif args.msf.lower().endswith('.pkl'):  # the MSF file already exists
        msf_obj = MKIDSpreadFunction(filename=args.msf)
        sim = msf_obj.sim_settings
    else:
        raise ValueError('Unknown MSF file type. The reduction cannot continue without a valid file.')

    # wavecal
    if args.wavecal.lower().endswith('.h5'):  # the table is not order-sorted and wavecal has yet to be done
        wavecal_table = Photontable(file_name=args.wavecal)
        steps.append('wt_sort')
        steps.append('wavecal')
        # TODO add clauses that check whether msf/wavecal/extract sim objects are equal
    elif args.wavecal.lower().endswith('.fits'):  # the wavecal has yet to be done
        wavecal_fits = args.wavecal
        steps.append('wavecal')
        # TODO add clause that checks array size matches with msf sim object
    elif args.wavecal.lower().endswith('.npz'):  # the wavecal file already exists
        wavecal_file = args.wavecal
    else:
        logger.info('No valid wavecal file was passed. Skipping wavelength calibration.')

    # extract
    if args.extract.lower().endswith('.h5'):  # the table is not order-sorted or extracted
        obs_table = Photontable(file_name=args.extract)
        steps.append('et_sort')
        steps.append('extract')
    elif args.extract.lower().endswith('.fits'):  # the observation is awaiting extraction
        obs_fits = args.extract
        steps.append('extract')
    else:
        logger.info('No valid observation file was passed. Skipping extraction.')

    logger.info(f'The {steps} step(s) will be conducted.')

    # ==================================================================================================================
    # START DATA REDUCTION STEPS
    # ==================================================================================================================
    if 'msf' in steps:
        # first separate the estimates for which pixels may be missing which orders:
        missing_order_pix = np.reshape(list(map(int, args.missing_order_pix)), (-1, 3))
        missing_order_pix = [[(missing_order_pix[i, 0], missing_order_pix[i, 1]),
              [int(o)-1 for o in str(missing_order_pix[i, 2])]] for i in range(missing_order_pix.shape[0])]

        # obtain the MKID Spread Function
        msf_obj = fitmsf(msf_table=msf_table,
                         sim=sim,
                         resid_map=sim.resid_file,
                         outdir=args.outdir,
                         bin_range=args.bin_range,
                         missing_order_pix=missing_order_pix,
                         plot=plot,
                         debug=args.debug)

    if 'wt_sort' in steps:
        # bin the wavecal table
        wavecal_fits = ordersort(table=wavecal_table,
                                 filename='emission',
                                 msf=msf_obj,
                                 resid_map=sim.resid_file,
                                 outdir=args.outdir,
                                 plot=plot)
    try:
        if 'wavecal' in steps:
            # get orders
            orders = range(sim.order_range[0], sim.order_range[1]+1)[::-1]
            # obtain the wavecal
            wavecal_file = wavecal(wavecal_fits=wavecal_fits,
                                   orders=orders,
                                   elem=args.elem.lower(),
                                   minw=sim.minwave,
                                   maxw=sim.maxwave,
                                   residual_max=args.residual_max,
                                   degree=args.degree,
                                   iters=args.iters,
                                   dim=args.dim,
                                   shift_window=args.shift_window,
                                   manual_fit=args.manual_fit,
                                   width=args.width,
                                   outdir=args.outdir,
                                   plot=plot)
    except IOError:
        logger.info('Skipping wavecal.')
    if 'et_sort' in steps:
        # bin the observation table
        obs_fits = ordersort(table=obs_table,
                             filename='observation',
                             msf=msf_obj,
                             resid_map=sim.resid_file,
                             outdir=args.outdir,
                             plot=plot)
    try:
        if 'extract' in steps:
            # retrieve the extracted observation spectrum
            extract(obs_fits=obs_fits,
                    wavecal_file=wavecal_file,
                    plot=plot)
    except IOError:
        logger.info('Skipping extraction.')

logger.info(f'Data reduction complete. Total time: {((time.perf_counter() - tic) / 60):.2f} min. Exiting.')
