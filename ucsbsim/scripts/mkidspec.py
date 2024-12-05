# global imports
import numpy as np
import astropy.units as u
import time
from datetime import datetime as dt
import argparse
import logging
import os
from mkidpipeline.photontable import Photontable

# local imports
from ucsbsim.mkidspec.steps.fitmsf import fitmsf
from ucsbsim.mkidspec.steps.ordersort import ordersort
from ucsbsim.mkidspec.steps.wavecal import wavecal
from ucsbsim.mkidspec.steps.extract import extract
from ucsbsim.mkidspec.simsettings import SpecSimSettings
from ucsbsim.mkidspec.msf import MKIDSpreadFunction
from ucsbsim.mkidspec.utils.general import LoadFromFile


if __name__ == "__main__":
    tic = time.perf_counter()  # recording start time for script

    # ==================================================================================================================
    # PARSE COMMAND LINE ARGUMENTS
    # ==================================================================================================================
    parser = argparse.ArgumentParser(description='MKID Spectrometer Data Reduction')

    # optional script args:
    parser.add_argument('--outdir', default='outdir', type=str, help='Directory for the output files.')
    parser.add_argument('--plot', action='store_true', default=False, help='If passed, show all plots.')

    # optional MSF args:
    parser.add_argument('--msf', default='outdir/flat.h5',
                        help='Directory/name of the flat/blackbody spectrum photon table .h5 file OR'
                             'Directory/name of the complete MKID Spread Function .pkl file.'
                             'Pass any other argument, such as "False", to disable this step.')
    parser.add_argument('--bin_range', default=(-1.5, 0), type=tuple,
                        help='Start/stop range for phase histogram.')
    parser.add_argument('--missing_order_pix', nargs='*',
                        default=[0, 349, 3, 350, 1299, 1, 0, 1299, 13, 1300, 2047, 2, 1300, 2047, 24],
                        help='Array of [startpix, endpix, missing-orders as single digit indexed from 1, and repeat],'
                             'e.g.: 0 999 13 1000 1999 25 2000 2047 4'
                             'will become [0, 999, 13,  1000, 1999, 25,  2000, 2047, 4]'
                             'where       sta sto  ord   sta  sto  ord   sta   sto  ord')

    # optional wavecal args:
    parser.add_argument('--wavecal', default='outdir/emission.h5',
                        help='Directory/name of the emission lamp spectrum photon table .h5 file OR'
                             'Directory/name of the order-sorted emission lamp spectrum .fits file OR'
                             'Directory/name of the complete wavelength calibration solution .npz file.'
                             'Pass any other argument, such as "False", to disable this step.')
    parser.add_argument('--elem', default='hgar', type=str,
                        help="Emission lamp element(s) in use, i.e., 'hgar' for Mercury-Argon.")
    parser.add_argument('--orders', nargs='*', default=[7, 6, 5, 4], type=list,
                        help="Orders to be used, space-delimited.")  # TODO have order numbers mean something
    parser.add_argument('--degree', default=4, type=int, help="Polynomial degree to use in wavecal.")
    parser.add_argument('--iters', default=5, type=int,
                        help="Number of iterations to loop through for identifying and discarding lines.")
    parser.add_argument('--manual_fit', action='store_true', default=False,
                        help="If passed, indicates user should click plot to align observation and linelist.")
    parser.add_argument('--residual_max', default=85e3, type=float,
                        help="Maximum residual allowed between fit wavelength and atlas in m/s. (float)")
    parser.add_argument('--width', default=3, type=int, help="Width in pixels for matching peaks.")
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
    parser.add_argument('--args_file', default=None, type=open, action=LoadFromFile,
                        help='.txt file with arguments written exactly as they would be in the command line.'
                             'Pass only this argument if being used. See "mkidspec_args.txt" for example.')

    args = parser.parse_args()
    
    # ==================================================================================================================
    # START LOGGING
    # ==================================================================================================================
    logger = logging.getLogger('mkidspec')
    logging.basicConfig(level=logging.INFO)

    # ==================================================================================================================
    # PARSE STEPS TO RUN
    # ==================================================================================================================
    steps = []  # list to append steps in use

    # MSF
    if args.msf.lower().endswith('.h5'):  # the MSF has yet to be fit
        msf_table = Photontable(file_name=args.msf)
        steps.append('msf')
        sim = msf_table.query_header('sim_settings')
    elif args.msf.lower().endswith('.pkl'):  # the MSF file already exists
        msf_obj = MKIDSpreadFunction(filename=args.msf)
        sim = msf_obj.sim_settings

    # wavecal
    if args.wavecal.lower().endswith('.h5'):  # the table is not order-sorted and wavecal has yet to be done
        wavecal_table = Photontable(file_name=args.wavecal)
        steps.append('wt_sort')
        steps.append('wavecal')
    elif args.wavecal.lower().endswith('.fits'):  # the wavecal has yet to be done
        wavecal_fits = args.wavecal
        steps.append('wavecal')
    elif args.wavecal.lower().endswith('.npz'):  # the wavecal file already exists
        wavecal_file = args.wavecal

    # extract
    if args.wavecal:
        if args.extract.lower().endswith('.h5'):  # the table is not order-sorted or extracted
            obs_table = Photontable(file_name=args.extract)
            steps.append('ot_sort')
            steps.append('extract')
        elif args.extract.lower().endswith('.fits'):  # the observation is awaiting extraction
            obs_fits = args.extract
            steps.append('extract')

    logger.info(f'The {steps} step(s) will be conducted.')

    # ==================================================================================================================
    # START DATA REDUCTION STEPS
    # ==================================================================================================================
    if 'msf' in steps:
        # first separate the estimates for which pixels may be missing which orders:
        missing_order_pix = np.reshape(list(map(int, args.missing_order_pix)), (-1, 3))
        missing_order_pix = \ 
            [[(missing_order_pix[i, 0], missing_order_pix[i, 1]),
              [int(o)-1 for o in str(missing_order_pix[i, 2])]] for i in range(missing_order_pix.shape[0])]
        msf_obj = fitmsf(msf_table=msf_table,
                         sim=sim,
                         resid_map=sim.resid_file,
                         outdir=args.outdir,
                         bin_range=args.bin_range,
                         missing_order_pix=missing_order_pix,
                         plot=args.plot)
    if 'wt_sort' in steps:
        wavecal_fits = ordersort(table=wavecal_table,
                                 filename='emission',
                                 msf=msf_obj,
                                 resid_map=sim.resid_file,
                                 outdir=args.outdir,
                                 plot=args.plot)
    if 'wavecal' in steps:
        wavecal_file = wavecal(wavecal_fits=wavecal_fits,
                               orders=args.orders,
                               elem=args.elem,
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
                               plot=args.plot)
    if 'ot_sort' in steps:
        obs_fits = ordersort(table=obs_table,
                             filename='observation',
                             msf=msf_obj,
                             resid_map=sim.resid_file,
                             outdir=args.outdir,
                             plot=args.plot)
    if 'extract' in steps:
        extract(obs_fits=obs_fits,
                wavecal_file=wavecal_file,
                plot=args.plot)

logger.info(f'Data reduction complete. Total time: {((time.perf_counter() - tic) / 60):.2f} min. Exiting.')
