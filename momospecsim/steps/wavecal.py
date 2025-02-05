import time

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
import sys
import os
import logging
import time
from datetime import datetime as dt
import argparse
from astropy import units as u
import scipy.interpolate as interp

from ucsbsim.mkidspec.spectrograph import SpectrographSetup
from ucsbsim.mkidspec.spectra import EmissionModel

sys.path.insert(1, '/home/kimc/pycharm/PyReduce/pyreduce')
from wavelength_calibration import WavelengthCalibration, WavelengthCalibrationInitialize, LineList

"""
Emission line wavelength calibration. The steps are:
-Load the emission line photon table
-
"""  # TODO

logger = logging.getLogger('wavecal')


def wavecal(
        wavecal_fits,
        orders,
        elem,
        minw,
        maxw,
        residual_max,
        degree,
        iters,
        dim,
        shift_window,
        manual_fit,
        width,
        outdir,
        plot
):

    if dim == '1D':
        deg = degree
    elif dim == '2D':
        deg = (degree, degree)

    # open observed spectrum and get wave/flux
    spectrum = fits.open(wavecal_fits)
    obs_flux = np.array([np.array(spectrum[1].data[n]) for n, i in enumerate(orders)])
    obs_wave = np.array([np.array(spectrum[4].data[n]) for n, i in enumerate(orders)])

    # open entire line list:
    file = pd.read_csv(f'/home/kimc/pycharm/KIDSpecSim/ucsbsim/mkidspec/linelists/{elem}.csv', delimiter=',')
    line_flux = []
    valid_idx = []
    for n, i in enumerate(file['intens']):
        try:
            line_flux.append(float(i[2:-1]))
            valid_idx.append(n)
        except ValueError:
            pass
    line_flux = np.array(line_flux)
    line_wave = np.array([float(i[2:-1]) * 10 for i in file['obs_wl_air(nm)'][valid_idx]])  # nm to Angstrom

    if not os.path.exists(f'/home/kimc/pycharm/PyReduce/pyreduce/wavecal/atlas/{elem}_list.txt'):
        # this is the theoretical reference line spectrum:
        theo_lines = EmissionModel(f'mkidspec/linelists/{elem}.csv',
                                   minwave=minw, maxwave=maxw, target_R=1000000)
        col1 = fits.Column(name='wave', format='E', array=theo_lines.waveset.value)
        col2 = fits.Column(name='spec', format='E', array=theo_lines(theo_lines.waveset).value)
        cols = fits.ColDefs([col1, col2])
        hdu = fits.BinTableHDU.from_columns(cols)
        hdu_list = fits.HDUList([fits.PrimaryHDU(), hdu])
        hdu_list.writeto(f'/home/kimc/pycharm/PyReduce/pyreduce/wavecal/atlas/{elem}.fits', overwrite=True)

        f = open(f'/home/kimc/pycharm/PyReduce/pyreduce/wavecal/atlas/{elem}_list.txt', 'w+')
        for i in range(len(line_wave)):
            f.write(f'{line_wave[i]} {elem.capitalize()}I\n')
        f.close()

    # on an order by order basis, cut away lines that are too close together or too dim
    pixel_interp = [interp.interp1d(
        obs_wave[n],
        range(len(obs_wave[0])),
        bounds_error=False,
        fill_value=0
    ) for n, i in enumerate(orders)]
    posc = np.array([pixel_interp[n](line_wave) for n, i in enumerate(orders)])  # Pixel Position (before fit)

    # remove 10% of the least intense lines
    sort_idx = np.argsort(line_flux)
    percent = int(0.1 * len(sort_idx))
    sort_idx = sort_idx[percent:]
    cut_wave = line_wave[sort_idx]
    cut_flux = line_flux[sort_idx]

    # re-sort back to increasing wavelength
    sort_idx = np.argsort(cut_wave)
    cut_wave = cut_wave[sort_idx]
    cut_flux = cut_flux[sort_idx]

    # keep only the most intense lines within a certain range:
    min_dl = np.min(np.diff(obs_wave[0]))  # the smallest extent of a pixel [Ang]
    start_range = 0
    new_line_idx = []
    for n, l in enumerate(cut_wave[:-1]):
        diff = cut_wave[n + 1] - cut_wave[start_range]
        if diff > min_dl:
            stop_range = n + 1
            copy_flux = np.copy(cut_flux)
            copy_flux[:start_range] = 0
            copy_flux[stop_range:] = 0
            new_line_idx.append(np.argmax(copy_flux))
            start_range = stop_range

    new_line_wave = cut_wave[new_line_idx]
    new_line_flux = cut_flux[new_line_idx]

    # create LineList object
    posc = np.array([pixel_interp[n](new_line_wave) for n, i in enumerate(orders)])  # Pixel Position (before fit)
    linelist = LineList()
    for n, i in enumerate(orders):
        for m, (w, p, h) in enumerate(zip(new_line_wave, posc[n], new_line_flux)):
            if posc[n][m]:
                linelist.add_line(w, n, p, 2, h, True)

    # load wavecal module
    module = WavelengthCalibration(
        threshold=residual_max,  # Residual threshold in m/s above which to remove lines (diff/wave*c)
        degree=deg,  # polynomial degree of the wavelength fit in (pixel, order) direction
        iterations=iters,  # Number of iterations in the remove residuals, auto id, loop
        dimensionality=dim,  # Whether to use 1d or 2d fit
        shift_window=shift_window,  # Fraction of columns to use in the alignment of individual orders. 0 to disable
        manual=manual_fit,  # Whether to manually align the reference instead of using cross correlation
        lfc_peak_width=width,  # Laser Frequency Peak width (for scipy.signal.find_peaks)
        element=elem,
        medium="vac",  # use vac if the linelist provided is in 'air' already, air will convert
        plot=plot,
        plot_title=f'{degree}D Poly./{residual_max:.0e} Max Resid./Element: {elem.capitalize()}'
    )
    # combines the orders and returns 2D solution
    wave_result, coef, residuals, fit_lines = module.execute(obs_flux, linelist)

    rms = [np.sqrt(np.mean(residuals[fit_lines['order'] == n] ** 2)) for n, i in enumerate(orders)]
    print(f'RMS (km/s): {[round(i / 1000, 2) for i in rms]}, Avg: {np.sqrt(np.mean(residuals ** 2)) / 1000:.2f}')
    print(f"{np.sum(fit_lines['flag'])} used out of {len(fit_lines)}")

    np.savez(f'{outdir}/wavecal.npz', wave_result=wave_result, coef=coef, rms=rms, linelist=linelist, orders=orders)
    logger.info(f'\nSaved wavecal solution and linelist to {outdir}/wavecal.npz.')

    if plot:
        new_file = np.load(f'{outdir}/wavecal.npz')

        for n, i in enumerate(orders):
            plt.grid()
            plt.ylim(bottom=0)
            use_idx = np.logical_and(fit_lines['wll'] > obs_wave[n][0], fit_lines['wll'] < obs_wave[n][-1])
            lines_use = fit_lines[use_idx]
            max_flux = np.max(lines_use['height'])
            for m, w in enumerate(lines_use):
                if not w['flag']:
                    plt.axvline(w['wll'], color='black', ymin=0, ymax=w['height'] / max_flux)
                else:
                    plt.axvline(w['wll'], color='red', alpha=0.5, ymin=0, ymax=w['height'] / max_flux)
            plt.plot(obs_wave[n], obs_flux[n] / np.max(obs_flux[n]), 'gray', label='Guess')
            plt.plot(new_file['wave_result'][n], obs_flux[n] / np.max(obs_flux[n]), 'blue', label=f'Fit Result')
            plt.title(f"Wavecal Order {i}. Used Lines in Red, Unused in Black\n"
                      f'Order {degree} Poly./Element: {elem.capitalize()}')
            plt.ylabel('Photon Count')
            plt.xlabel(r'Wavelength ($\AA$)')
            plt.legend()
            plt.show()

    return f'{outdir}/wavecal.npz'


if __name__ == '__main__':
    tic = time.perf_counter()  # recording start time for script

    # ==================================================================================================================
    # CONSTANTS
    # ==================================================================================================================

    # ==================================================================================================================
    # PARSE ARGUMENTS
    # ==================================================================================================================
    arg_desc = '''
               Wavelength calibration from an emission line source spectrum.
               --------------------------------------------------------------
               This program loads the emission line photon table and allows user to
               recover a 2D wavecal solution.
               '''
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter, description=arg_desc)

    # required wavecal args:
    parser.add_argument('outdir',
                        metavar='OUTPUT_DIRECTORY',
                        help='Directory for the output files (str).')
    parser.add_argument('wavecal_file',
                        metavar='FITS_FILE',
                        help='Directory/name of the FITS file to be used for wavecal.')

    # optional wavecal args:
    parser.add_argument('--elem', default='hgar', type=str,
                        help="Emission lamp element in use, i.e., 'hgar' for Mercury-Argon.")
    parser.add_argument('--orders', default=[7, 6, 5, 4], type=list,
                        help="Orders to be used. Useful if you only want to fit 1 order at a time.")
    parser.add_argument('--degree', default=4, type=int, help="Polynomial degree to use in wavecal.")
    parser.add_argument('--iters', default=5, type=int,
                        help="Number of iterations to loop through for identifying and discarding lines.")
    parser.add_argument('--manual_fit', action='store_true', default=False, type=bool,
                        help="If passed, indicates user should click plot to align observation and linelist.")
    parser.add_argument('--residual_max', default=85e3, type=float,
                        help="Maximum residual allowed between fit wavelength and atlas in m/s. (float)")
    parser.add_argument('--width', default=3, type=int, help="Width in pixels when searching for matching peaks.")
    parser.add_argument('--shift_window', default=0.05, type=float,
                        help="Fraction of columns to use in the alignment of individual orders, 0 to disable.")
    parser.add_argument('--dim', default='1D', type=str,
                        help="Return a '1D' (pixel direction) or '2D' (pixel+order directions) fitting solution.")
    parser.add_argument('--minw', default=400, type=float, help='The minimum wavelength of the spectrograph in nm.')
    parser.add_argument('--maxw', default=800, type=float, help='The maximum wavelength of the spectrograph in nm.')
    parser.add_argument('--plot', action='store_true', default=False, type=bool, help='If passed, plots will be shown.')

    # extract arguments
    args = parser.parse_args()

    # ==================================================================================================================
    # CHECK AND/OR CREATE DIRECTORIES
    # ==================================================================================================================
    os.makedirs(f'{args.output_dir}/logging', exist_ok=True)

    # ==================================================================================================================
    # START LOGGING TO FILE
    # ==================================================================================================================
    now = dt.now()
    logger = logging.getLogger('wavecal')
    logging.basicConfig(level=logging.DEBUG)
    logger.info("The process of wavelength calibration from an emission line spectrum is recorded."
                 f"\nThe date and time are: {now.strftime('%Y-%m-%d %H:%M:%S')}.")

    # ==================================================================================================================
    # WAVELENGTH CALIBRATION STARTS
    # ==================================================================================================================
    # linelist = LineList()
    # for n, i in enumerate(args.orders):
    #     # takes each order and conducts MCMC line matching
    #     module_init = WavelengthCalibrationInitialize(
    #         degree=args.degree,  # polynomial degree of the wavelength fit
    #         plot=True,
    #         plot_title=f'Deg. {args.degree} Poly./{10} Walkers/{50000} Steps/{args.residual_max} Max Resid./Element: {args.elem}',
    #         wave_delta=10,  # wavelength uncertainty on the initial guess in Angstrom
    #         nwalkers=100,  # number of walkers in the MCMC
    #         steps=50000,  # number of steps in the MCMC
    #         resid_delta=args.residual_max,
    #         # residual uncertainty allowed when matching observation with known lines (diff/wave*c)
    #         cutoff=10,  # minimum value in the spectrum to be considered a spectral line,
    #         # if the value is above (or equal 1) it defines the percentile of the spectrum
    #         smoothing=5,  # gaussian smoothing on wavecal spectrum before the MCMC in pixel scale, disable with 0
    #         element=args.elem,
    #         medium="vac"
    #     )
    #     ll = module_init.execute(np.array([obs_flux[n]]), [(obs_wave[n][0], obs_wave[n][-1])])
    #     ll = ll.data
    #     for li in ll:
    #         linelist.add_line(li['wlc'], n, li['posc'], li['width'], li['height'], li['flag'])

    wavecal(
        wavecal_fits=args.wavecal_file,
        orders=args.orders,
        elem=args.elem,
        minwave=args.minw,
        maxwave=args.maxw,
        residual_max=args.residual_max,
        degree=args.degree,
        iters=args.iters,
        dim=args.dim,
        shift_window=args.shift_window,
        manual_fit=args.manual_fit,
        width=args.width,
        outdir=args.outdir,
        plot=args.plot
    )

    logger.info(f'\nTotal script runtime: {((time.perf_counter() - tic) / 60):.2f} min.')
    # ==================================================================================================================
    # MSF EXTRACTION ENDS
    # ==================================================================================================================
