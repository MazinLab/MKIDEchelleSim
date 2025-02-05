import numpy as np
import matplotlib.pyplot as plt
import time
import logging
import astropy.units as u
import argparse
from astropy.io import fits
from astropy.table import Table
import os

"""
Bringing it all together. This is the stage where we load and bring together:
- The observation spectrum separated into virtual pixels and the errors on each
- The wavecal solution from the emission spectrum
to subtract order-bleeding and recover a final spectrum.
This script may also:
- fit an empiric blaze function to be divided out
"""

logger = logging.getLogger('extract')


def extract(
        obs_fits,
        wavecal_file,
        plot
):
    wavecal_file = np.load(f'{wavecal_file}')
    wavecal = wavecal_file['wave_result']
    orders = wavecal_file['orders']

    spectrum = fits.open(f'{obs_fits}', mode='update')
    obs_flux_corr = np.array([np.array(spectrum[1].data[n]) for n, i in enumerate(orders)])
    bleed_p = np.array([np.array(spectrum[2].data[n]) for n, i in enumerate(orders)])
    bleed_n = np.array([np.array(spectrum[3].data[n]) for n, i in enumerate(orders)])
    guess_wave = np.array([np.array(spectrum[4].data[n]) for n, i in enumerate(orders)])
    obs_flux = np.array([np.array(spectrum[5].data[n]) for n, i in enumerate(orders)])
    hdu_list = fits.HDUList([spectrum[0],
                             spectrum[1],
                             spectrum[2],
                             spectrum[3],
                             spectrum[5],
                             fits.BinTableHDU(Table(wavecal), name='Wavecal')])
    hdu_list.writeto(obs_fits, output_verify='ignore', overwrite=True)
    logger.info(f'The FITS file has been updated with the wavecal: {obs_fits}.')

    err_p_corr = np.around(np.sqrt(np.array(bleed_p) + obs_flux_corr))
    err_n_corr = np.around(np.sqrt(np.array(bleed_n) + obs_flux_corr))

    err_p = np.around(np.sqrt(np.array(bleed_p) + obs_flux))
    err_n = np.around(np.sqrt(np.array(bleed_n) + obs_flux))

    if plot:
        # comparing the wavecal solution to the simulation wavelengths
        plt.grid()
        for n, i in enumerate(orders[:-1]):
            plt.plot(guess_wave[n] / 10, obs_flux[n], 'r', linewidth=0.5)
            plt.plot(wavecal[n] / 10, obs_flux[n], 'b', linewidth=0.5)
        plt.plot(guess_wave[-1] / 10, obs_flux[-1], 'r', linewidth=0.5, label='Initial Guess')
        plt.plot(wavecal[-1] / 10, obs_flux[-1], 'b', linewidth=0.5, label='Wavecal')
        plt.title('Comparison between guess and calibration')
        plt.xlabel('Wavelength (nm)')
        plt.ylabel('Photon Count')
        plt.legend()
        plt.tight_layout()
        plt.show()

        # plot with errors
        fig, ax = plt.subplots(1, 2, sharey=True, figsize=(10, 5))
        axes = ax.ravel()

        axes[0].grid()
        for n, i in enumerate(orders):
            axes[0].plot(wavecal[n] / 10, obs_flux[n], 'k', linewidth=0.5)
        axes[0].set_title('Uncorrected Spectrum')
        axes[0].set_xlabel('Wavelength (nm)')
        axes[0].set_ylabel('Photon Count')

        axes[1].grid()
        for n, i in enumerate(orders):
            specerr_pcorr = obs_flux_corr[n] + err_p_corr[n]
            specerr_pcorr[specerr_pcorr < 0] = 0
            specerr_ncorr = obs_flux_corr[n] - err_n_corr[n]
            specerr_ncorr[specerr_ncorr < 0] = 0
            axes[1].fill_between(wavecal[n] / 10, specerr_ncorr, specerr_pcorr, edgecolor='r',
                                 facecolor='r', linewidth=0.5)
            axes[1].plot(wavecal[n] / 10, obs_flux_corr[n], 'k', linewidth=0.5)
        axes[1].set_title('Order-Bleed Subtraction-Corrected (Error in Red)')
        axes[1].set_xlabel('Wavelength (nm)')
        plt.tight_layout()
        plt.show()
        pass


if __name__ == '__main__':
    arg_desc = '''
    Extract a spectrum using the uncalibrated spectrum and wavecal.
    --------------------------------------------------------------
    This program loads the observation spectrum on the phase grid and applies the wavecal solution to
    extract the spectrum. It will compare the simulation input to the final product if applicable.
    '''
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter, description=arg_desc)

    # required extraction args:
    parser.add_argument('obs_fits',
                        metavar='FITS_FILE',
                        help='Directory/name of the FITS file containing the observation (str).')
    parser.add_argument('wavecal_file',
                        metavar='WAVECAL_FILE',
                        help='Directory/name of the wavelength calibration .npz file (str).')
    parser.add_argument('--plot', action='store_true', default=False, type=bool, help='If passed, plots will be shown.')

    # get arguments
    args = parser.parse_args()

    extract(obs_fits=args.obs_fits, wavecal_file=args.wavecal_file, plot=args.plot)
