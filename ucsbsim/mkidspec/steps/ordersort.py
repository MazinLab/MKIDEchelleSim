import numpy as np
import warnings
from copy import deepcopy
import matplotlib.pyplot as plt
import time
from datetime import datetime as dt
import logging
import astropy.units as u
import argparse
from astropy.io import fits
from astropy.table import Table
import os

from ucsbsim.mkidspec.spectrograph import GratingSetup, SpectrographSetup
from ucsbsim.mkidspec.detector import MKIDDetector, wave_to_phase, sorted_table
import ucsbsim.mkidspec.engine as engine
from ucsbsim.mkidspec.plotting import quick_plot
from synphot.models import BlackBodyNorm1D, ConstFlux1D
from synphot import SourceSpectrum
from ucsbsim.mkidspec.msf import MKIDSpreadFunction
from mkidpipeline.photontable import Photontable
import ucsbsim.mkidspec.utils.general as gen

"""
Application of the virtual pixel boundaries and errors on an spectrum using the MSF products. The steps are:
-Open the MSF products: order bin edges and covariance matrices.
-Open the observation/emission photon table and bin for virtual pixels/orders.
-Calculate the errors on each point by multiplying the covariance matrices through the spectrum.
-Save counts, errors, and estimate of wave range to FITS.
-Show final spectrum as plot.
"""

logger = logging.getLogger('ordersort')


def ordersort(
        table: Photontable,
        filename: str,
        msf,
        outdir: str,
        plot: bool,
        resid_map,
):
    if isinstance(resid_map, str):
        resid_map = np.loadtxt(fname=resid_map, delimiter=',')
    photons_pixel = sorted_table(table=table, resid_map=resid_map)

    if isinstance(msf, str):
        msf = MKIDSpreadFunction(filename=msf)
    sim = msf.sim_settings
    phis = msf.waves
    sigs = msf.sigmas
    logger.info(f'Obtained MKID Spread Function from {msf}.')

    # INSTANTIATE SPECTROGRAPH & DETECTOR:
    spectro = sim.spectrograph
    detector = spectro.detector
    pixels = detector.pixel_indices

    # shorten some variables
    nord = spectro.nord
    lambda_pixel = spectro.pixel_wavelengths().to(u.nm)[::-1]

    spec = np.zeros([nord, sim.npix])
    for p in pixels:  # binning photons by MSF bins edges
        spec[:, p], _ = np.histogram(photons_pixel[p], bins=msf.order_edges[:, p])

    spec_unfixed = deepcopy(spec)
    
    #pixels = [1433]
    
    bleed_p = np.zeros([nord, sim.npix])
    bleed_n = np.zeros([nord, sim.npix])
    
    logger.info('Order sorting...')
    for p in pixels:
        spec[:, p] = np.dot(spec[:, p], np.linalg.inv(msf.cov_matrix[:, :, p]))
        spec[:, p] = np.around(spec[:, p])
        spec[spec < 0] = 0
        bleed_p[:, p] = [np.sum(spec[:, p] * msf.cov_matrix[:, o, p]) - spec[o, p] * msf.cov_matrix[o, o, p] for o in range(nord)]
        bleed_n[:, p] = [np.sum(spec[o, p] * msf.cov_matrix[o, :, p]) - spec[o, p] * msf.cov_matrix[o, o, p] for o in range(nord)]

    warnings.filterwarnings(action="ignore")  # ignore FITS warnings
    # saving extracted and unblazed spectrum to file
    fits_file = f'{outdir}/{filename}.fits'
    hdu_list = fits.HDUList([fits.PrimaryHDU(),
                             fits.BinTableHDU(Table(spec), name='Corrected Spectrum'),
                             fits.BinTableHDU(Table(bleed_p), name='+ Bleed Errors'),
                             fits.BinTableHDU(Table(bleed_n), name='- Bleed Errors'),
                             fits.BinTableHDU(Table(lambda_pixel.to(u.Angstrom)), name='Guess Wave'),
                             fits.BinTableHDU(Table(spec_unfixed), name='Uncorrected Spectrum')])

    hdu_list.writeto(fits_file, output_verify='ignore', overwrite=True)
    logger.info(f'The extracted spectrum with its errors has been saved to {fits_file}.')

    if plot:
        spectrum = fits.open(fits_file)

        # plot the spectrum unblazed with the error band:
        fig, ax = plt.subplots(int(np.ceil(nord/2)), 2, figsize=(15, int(5*nord)), sharex=True)
        axes = ax.ravel()
        for i in range(nord):
            axes[i].grid()
            axes[i].plot(pixels, spectrum[1].data[i])
            axes[i].set_title(f'Order {spectro.orders[::-1][i]}')
        axes[-1].set_xlabel("Pixel Index")
        axes[-2].set_xlabel("Pixel Index")
        axes[0].set_ylabel('Photon Count')
        axes[2].set_ylabel('Photon Count')
        plt.suptitle(f'Sorted spectrum')
        plt.tight_layout()
        plt.show()
    return fits_file


if __name__ == '__main__':
    tic = time.time()  # recording start time for script

    # ==================================================================================================================
    # PARSE ARGUMENTS
    # ==================================================================================================================
    arg_desc = '''
    Extract a spectrum using the MKID Spread Function.
    --------------------------------------------------------------
    This program loads the observation photon table and uses the MSF bins and covariance matrix to
    extract the spectrum.
    '''
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter, description=arg_desc)

    # required MSF args:
    parser.add_argument('outdir', metavar='OUTPUT_DIRECTORY',
                        help='Directory for the output files (str).')
    parser.add_argument('msf_file',
                        metavar='MKID_SPREAD_FUNCTION_FILE',
                        help='Directory/name of the MKID Spread Function file (str).')
    parser.add_argument('obstable',
                        metavar='OBSERVATION_PHOTON_TABLE',
                        help='Directory/name of the observation spectrum photon table (str).')
    parser.add_argument('--plot', action='store_true', default=False, type=bool, help='If passed, plots will be shown.')

    # get arguments
    args = parser.parse_args()

    # ==================================================================================================================
    # CHECK AND/OR CREATE DIRECTORIES
    # ==================================================================================================================
    os.makedirs(f'{args.output_dir}/logging', exist_ok=True)

    # ==================================================================================================================
    # START LOGGING TO FILE
    # ==================================================================================================================
    now = dt.now()
    logger = logging.getLogger('ordersort')
    logging.basicConfig(level=logging.DEBUG)
    logger.info(f"The extraction of an observed spectrum is recorded."
                 f"\nThe date and time are: {now.strftime('%Y-%m-%d %H:%M:%S')}.")

    obs_table = Photontable(file_name=args.obstable)
    sim = obs_table.query_header('sim_settings')
    spectra = sim.spectra_type

    # ==================================================================================================================
    # OBSERVATION SPECTRUM EXTRACTION STARTS
    # ==================================================================================================================
    ordersort(
        table=obs_table,
        filename=spectra,
        msf_file=args.msf_file,
        outdir=args.outdir,
        plot=args.plot
    )

    '''
    # obtain the pixel-space blaze function:
    wave = np.linspace(sim.minwave.value - 100, sim.maxwave.value + 100, 10000) * u.nm
    # retrieving the blazed calibration spectrum shape assuming it is known and converting to pixel-space wavelengths:
    if sim.type_spectra == 'blackbody':
        spectra = SourceSpectrum(BlackBodyNorm1D, temperature=sim.temp)  # flux for star of 1 R_sun at distance of 1 kpc
    else:
        spectra = SourceSpectrum(ConstFlux1D, amplitude=1)  # only blackbody supported now
    blazed_spectrum, _, _ = eng.blaze(wave, spectra)
    # TODO can we assume any knowledge about blaze shape? if not, how to divide out eventually?
    lambda_left = spectro.pixel_wavelengths(edge='left').to(u.nm).value
    blaze_shape = [eng.lambda_to_pixel_space(wave, blazed_spectrum[i], lambda_left[i]) for i in range(nord)][::-1]
    blaze_shape /= np.max(blaze_shape)  # normalize max to 1
    blaze_shape[blaze_shape == 0] = 1  # prevent divide by 0 or very small num. issue
    '''

    logger.info(f'Total script runtime: {((time.time() - tic) / 60):.2f} min.')
    # ==================================================================================================================
    # OBSERVATION SPECTRUM EXTRACTION ENDS
    # ==================================================================================================================

