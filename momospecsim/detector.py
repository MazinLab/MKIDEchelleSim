from copy import deepcopy
import numpy as np
import astropy.units as u
import logging

import scipy.signal
from scipy.constants import c
from astropy.constants import h, c
from itertools import combinations
from sklearn.cluster import k_means
from numpy.polynomial.legendre import Legendre
import matplotlib.pyplot as plt
from mpl_point_clicker import clicker
from mpl_toolkits.axes_grid1 import make_axes_locatable
from lmfit import Parameters, minimize
import tqdm

from mkidpipeline.photontable import Photontable

from momospecsim.utils.general import sig_to_R, wave_to_energy, energy_to_wave, gauss, gauss_intersect, nearest_idx
from filterphot import mask_deadtime
from momospecsim.engine import draw_photons
from momospecsim.steps.fitmsf import init_params, e0_from_params, cov_from_params, phis_from_grating_eq, fit_func

logger = logging.getLogger('detector')


def sorted_table(table: Photontable, resid_map):
    """
    :param table: Photontable object
    :param resid_map: the resonator ID list
    :return: a list of photon wavelengths sorted by resonator ID
    """
    phases = table.query(column='wavelength')
    resID = table.query(column='resID')
    logger.info('Sorting photon table by pixel...')
    table = []
    for j in tqdm.tqdm(resid_map):
        table.append(phases[np.where(resID == j)])
    return table


def wave_to_phase(waves, minwave, maxwave):
    """
    range is -pi to pi
    smaller wavelengths wrap beginning at -pi and larger wavelengths wrap beginning at pi
    line is from (freq_minw, -0.8) to (freq_maxw, -0.2)
    if -1.1, negative: -1.1+2*max_phase, positive: if 1.1, 1.1+2*min_phase, repeating until between -1 to 1
    linear equation: y = (y2-y1)/(x2-x1)*(x-x1) + y1 = 0.6/(freq_maxw-freq_minw)*(x-freq_minw) - 0.8
    also wraps phase
    :param waves: wavelengths in nm
    :param minwave: minimum wavelength
    :param maxwave: maximum wavelength
    :return: phase values corresponding to wavelength
    """
    if isinstance(waves, u.Quantity):
        waves = waves.to(u.nm).value
    shape = np.shape(waves)
    waves = np.array(waves).flatten()

    freq_minw = (c * u.m / u.s / minwave).decompose()  # this will be mapped to -0.8
    freq_maxw = (c * u.m / u.s / maxwave).decompose()  # this will be mapped to -0.2
    freqs = (c * u.m / u.s / (waves * u.nm)).decompose()  # converted wavelength to frequency (Hz)
    phases = np.nan_to_num((0.6 / (freq_maxw - freq_minw) * (freqs - freq_minw)).decompose().value - 0.8,
                           posinf=0, neginf=-1)
    phases = np.reshape(phases, shape)
    return phases


def phase_to_wave(phases, minwave, maxwave):
    """
    linear equation: x = (y-y1)*(x2-x1)/(y2-y1) + x1 = (y + 0.8)*(freq_maxw-freq_minw)/0.6 + freq_minw
    
    :param phases: phase values
    :param minwave: minimum wavelength
    :param maxwave: maximum wavelength
    :return: given a phase, assuming the same linear equation is used, return the wavelength
    """
    freq_minw = (c * u.m / u.s / minwave).decompose()
    freq_maxw = (c * u.m / u.s / maxwave).decompose()
    freqs = ((phases+0.8)*(freq_maxw-freq_minw)/0.6 + freq_minw).decompose()
    return c * u.m / u.s / freqs


class MKIDDetector:
    def __init__(self,
                 npix: int,
                 pix_size: u.Quantity,
                 design_R0: float,
                 l0: u.Quantity,
                 randomseed: int,
                 resid_file: str,
                 fixedR: bool = False):
        """
        Simulation of an MKID detector array

        :param int npix: number of pixels in linear array
        :param u.Quantity pix_size: physical size of each pixel as u.Quantity
        :param float R0: spectral resolution of the longest wavelength in spectrometer range
        :param u.Quantity l0: longest wavelength in spectrometer range in nm
        :param int randomseed: numpy random seed for generating or saving files
        :param str resid_file: resonator ID filename
        :param bool fixedR: whether to fix exactly (True) or vary randomly (False) the R
        """
        self.npix = npix
        self.pix_size = pix_size * u.um
        self.length = self.npix * pix_size
        self.l0 = l0 * u.nm
        self.design_R0 = design_R0
        self.pixel_indices = np.arange(self.npix, dtype=int)
        self.randomseed = randomseed
        self.fixedR = fixedR
        logger.info(f'The random seed is set to {randomseed}.')

        if fixedR:
            self.R0s = np.full(shape=self.npix, fill_value=self.design_R0)
            logger.info(msg=f'The pixel Rs @ {l0} were fixed exactly at {design_R0} for every pixel.')
        else:
            np.random.seed(randomseed)
            self.R0s = np.random.uniform(low=.85, high=1.15, size=npix) * design_R0
            logger.info(msg=f'The pixel Rs @ {l0} were randomly generated about {design_R0}.')

        np.random.seed(randomseed)
        self.phase_offsets = np.random.uniform(low=.8, high=1.2, size=npix)
        logger.info(msg=f'The pixel phase offsets were randomly generated.')

        self.resid_file = resid_file
        try:  # check for the resonator IDs, create if not exist
            self.resid_map = np.loadtxt(fname=resid_file, delimiter=',')
            logger.info(msg=f'The resonator IDs were imported from {resid_file}.')
        except IOError:
            np.random.seed(randomseed)
            self.resid_map = np.arange(npix, dtype=int) * 10 + 100
            np.savetxt(fname=resid_file, X=self.resid_map, delimiter=',')
            logger.info(msg=f'The resonator IDs were generated from {self.resid_map.min()} to {self.resid_map.max()}.')
        logger.info('MKID Detector initialized.')

    def R0(self, pixel: int):
        """
        :param pixel: the pixel index or indices
        :return: spectral resolution for given pixel
        """
        if pixel not in self.pixel_indices:
            raise ValueError(f"Pixel {pixel + 1} not in instantiated detector, max of {self.npix}.")
        return self.R0s[pixel.astype(int)]

    def mkid_constant(self, pixel):
        """
        :param pixel: the pixel index or indices
        :return: MKID constant for given pixel, R0 * l0
        """
        return self.R0(pixel) * self.l0


    def mkid_resolution_width(self, wave, pixel, energy=False):
        """
        :param wave: wavelength/energy(s) as u.Quantity
        :param pixel: the pixel index or indices
        :param energy: True to pass and return energy
        :return: FWHM of the MKID at given wavelength/energy and pixel
        """
        if energy:
            wave = wave.to(u.nm, equivalencies=u.spectral())
        else:
            rc = self.mkid_constant(pixel)

            try:
                if wave.shape != rc.shape:
                    if wave.ndim == rc.ndim:
                        raise ValueError('Arrays of the same dimensions much have matching shapes')
                    if wave.shape[-1] != rc.shape[-1]:
                        raise ValueError('Arrays of differing dimension must match along the final dimension')
                    rc = rc[None, :]
            except AttributeError:  # allow non-array args
                pass

        if energy:
            return (wave**2*wave.to(u.eV, equivalencies=u.spectral())**2 / (self.R0(pixel)*self.l0*h*c)).to(u.eV)
        else:
            return wave ** 2 / rc


    def observe(self, convol_wave, convol_result, phase: bool = True, minwave=None, maxwave=None, energy=False, 
                randomseed=None, **draw_kwargs):
        """
        :param convol_wave: wavelength array that matches convol_result
        :param convol_result: convolution array
        :param bool phase: True if resulting recarray to be in phase values not wavelength
        :param minwave: pass value of spectrograph minwave for phase=True
        :param maxwave: pass value of spectrograph maxwave for phase=True
        :param energy: True to conduct observation in energies
        :param draw_kwargs: additional keyword args to pass to draw_photons (exptime, area, etc.)
        :param randomseed: random seed for reproducibility
        :return: recarray of observed photons, total number observed
        """
        from mkidcore.binfile.mkidbin import PhotonNumpyType

        # random draw for wavelengths and energies based on convolution
        arrival_times, arrival_wavelengths, reduce_factor = draw_photons(convol_wave, convol_result, energy=energy,
                                                                         randomseed=randomseed, **draw_kwargs)

        pixel_count = np.array([x.size for x in arrival_times])
        total_photons = pixel_count.sum()

        merge_time_window_s = 1e-6 * u.s
        MIN_TRIGGER_ENERGY = 1 / (1.5 * u.um)
        SATURATION_WAVELENGTH_NM = 350 * u.nm
        DEADTIME = 10 * u.us

        logger.info("Beginning MKID detector observation sequence with:"
                     f"\n\tMinimum trigger energy: {MIN_TRIGGER_ENERGY:.3e}"
                     f"\n\tPhoton merge time: {merge_time_window_s:.0e}"
                     f"\n\tSaturation wavelength: {SATURATION_WAVELENGTH_NM}"
                     f"\n\tDeadtime: {DEADTIME}")
        logger.warning(f'Simulated dataset may take up to {total_photons * 16 / 1024 ** 3:.2} GB of RAM.')

        # create empty arrays for observation
        photons = np.recarray(total_photons, dtype=PhotonNumpyType)
        photons[:] = 0
        photons.weight[:] = 1.0
        observed = 0
        total_merged = 0
        total_missed = []

        # begin deadtime/merging/min. wave processes:
        for pixel, n in enumerate(tqdm.tqdm(pixel_count)):
            if not n:
                continue

            # get photon energies and arrival times for pixel
            a_times = arrival_times[pixel]
            arrival_order = a_times.argsort()
            a_times = a_times[arrival_order]
            energies = 1 / arrival_wavelengths[pixel].to(u.um, equivalencies=u.spectral())[arrival_order]

            # merge photon energies within 1us
            to_merge = (np.diff(a_times) < merge_time_window_s).nonzero()[0]
            if to_merge.size:
                cluster_starts = to_merge[np.concatenate(([0], (np.diff(to_merge) > 1).nonzero()[0] + 1))]
                cluser_last = to_merge[(np.diff(to_merge) > 1).nonzero()[0]] + 1
                cluser_last = np.append(cluser_last, to_merge[-1] + 1)  # inclusive
                for start, stop in zip(cluster_starts, cluser_last):
                    merge = slice(start + 1, stop + 1)
                    energies[start] += energies[merge].sum()
                    energies[merge] = np.nan
                    total_merged += energies[merge].size

            # Filter those with too low of energy that won't trigger detection
            will_trigger = energies > MIN_TRIGGER_ENERGY
            if not will_trigger.any():
                continue
            a_times = a_times[will_trigger].value * 1e6  # turn into us
            dead = DEADTIME.value
            # drop photons that arrive within the deadtime
            #detected = mask_deadtime(a_times, DEADTIME)
            detected = np.ones(len(a_times), bool)
            i = 0
            while i < a_times.size:
                # checks how many elements after arrival time are within deadtime
                temp = a_times[i+1:] - (a_times[i] + dead)
                n_dead = (temp < 0).sum()
                try:
                    # assigns all elements within deadtime non-detection flag
                    detected[i+1:i+1+n_dead] = False
                except ValueError:
                    continue
                i += n_dead + 1

            # determine all photons missed
            missed = will_trigger.sum() - detected.sum()
            total_missed.append(missed)

            # limits wavelengths to saturation wavelength of MKID
            measured_wavelengths = 1000 / energies[will_trigger][detected]
            measured_wavelengths.clip(SATURATION_WAVELENGTH_NM, out=measured_wavelengths)

            # add photons to the pot
            a_times = a_times[detected]
            sl = slice(observed, observed + a_times.size)
            photons.wavelength[sl] = measured_wavelengths
            #photons.time[sl] = a_times * 1e6  # in microseconds
            photons.time[sl] = a_times
            photons.resID[sl] = self.resid_map[pixel]
            observed += a_times.size

        if phase:  # converts wavelengths to MKID response phase
            photons.wavelength = wave_to_phase(photons.wavelength, minwave, maxwave)
            logger.info('Converting to phase...')
            for j in tqdm.tqdm(self.pixel_indices):  # sorting photons by resID (i.e. pixel) and multiplying phase center offsets
                photons.wavelength[np.where(photons.resID == self.resid_map[j])] *= self.phase_offsets[j]

            # wraps photon phases so they remain between -pi and pi
            shape = np.shape(photons.wavelength)
            phases = np.array(photons.wavelength).flatten()
            while True:
                phases[phases < -1] += 2
                phases[phases > 1] -= 2
                if (phases < -1).sum() == 0 and (phases < -1).sum() == 0:
                    break
            photons.wavelength = np.reshape(phases, shape)

        logger.info(f'Completed detector observation sequence.\n'
                     f'Merged: {total_merged}\n'
                     f'Deadtime miss: {np.sum(total_missed)}\n'
                     f'Observed: {observed}')
        return photons, observed, reduce_factor


class Pixel:
    def __init__(self, n, nord, orders, resid, photonlist, bin_edges, bin_centers, model_energies, fine_grid):
        self.n = n
        self.nord = nord
        self.orders = orders
        self.resid = resid
        self.photonlist = photonlist
        self.model_energies = model_energies
        self.bin_edges = bin_edges
        self.bin_centers = bin_centers
        self.binned_counts = np.histogram(self.photonlist, bins=bin_edges)[0]
        self.fine_phase_grid = fine_grid
        
        self.leg_s = None
        self.init_params = None
        self.opt_params = None
        self.redchi2 = None
        self.fit_phi = None
        self.fit_sig = None
        self.fit_amp = None
        self.gausses = None
        self.gausses_i = None
        self.covariance = None
        self.p_err = None
        self.m_err = None
        self.order_edges = np.zeros(nord + 1)
        self.order_edges[0] = -2
        self.ord_counts = None
        self.true_counts = None
        self.plot_int = False
        
        self.all_orders = False
    
    def cluster(self):
        save_phi, save_sig, save_amp, residual = [], [], [], []
        for n_use in range(2, self.nord + 1):
            
            # use peaks for initial cluster guess
            peaks, props = scipy.signal.find_peaks(self.binned_counts, height=5, distance=6)
            if len(peaks) < n_use:
                need = n_use - len(peaks)
                init = np.append(np.argsort(props['peak_heights'])[::-1],[0] * need)
            else:
                init = np.argsort(props['peak_heights']).astype(int)[::-1][:n_use]
            init = self.bin_centers[peaks[init]].reshape(-1, 1)
            
            # find cluster centers given number of clusters to find
            center, labels, _ = k_means(self.photonlist.reshape(-1, 1), n_use, init=init)
            ascend_order = np.argsort(center.flatten())
            init_phi = center.flatten()[ascend_order]  # sort as clusters are not always in ascending order

            # find cluster standard devations
            clusters = [self.photonlist[np.argwhere(labels == i).flatten()] for i in range(n_use)]
            init_sig = np.array([np.std(clusters[i]) for i in range(n_use)])[ascend_order]
            
            init_amp = np.array([self.binned_counts[nearest_idx(self.bin_centers, init_phi[i])] for i in range(n_use)])
            init_amp[init_amp < 10] = 100
            init_sig[init_sig == 0] = np.average(init_sig)
            gausses = np.sum([gauss(self.bin_centers, init_phi[i], init_sig[i], init_amp[i]) for i in range(n_use)], axis=0)
            gauss_1 = deepcopy(gausses)
            gauss_1[gauss_1 < 1] = 1
            residual.append(np.sum((np.divide(self.binned_counts - gausses, np.sqrt(gauss_1)))**2))
            if self.n == 1467:
                pass
            save_phi.append(init_phi)
            save_sig.append(init_sig)
            save_amp.append(init_amp)
            
            init_amp_p = np.array([self.binned_counts[nearest_idx(self.bin_centers, init.flatten()[i])] for i in range(n_use)])
            init_amp_p[init_amp_p < 10] = 100
            gausses_p = np.sum([gauss(self.bin_centers, init.flatten()[i], init_sig[i], init_amp[i]) for i in range(n_use)], axis=0)
            gauss_1p = deepcopy(gausses)
            gauss_1p[gauss_1p < 1] = 1
            residual.append(np.sum((np.divide(self.binned_counts - gausses_p, np.sqrt(gauss_1p)))**2))
            save_phi.append(init.flatten())
            save_sig.append(init_sig)
            save_amp.append(init_amp_p)

        min_idx = np.argmin(residual)
        fit_later = True if len(save_phi[min_idx]) < self.nord else False
        
        return save_phi[min_idx], save_sig[min_idx], save_amp[min_idx], fit_later
    
    def fit(self, leg_e, ratio=None):
        cluster_phi, cluster_sig, cluster_amp, fit_later = self.cluster()
        
        if ratio is not None:  # passing amplitude ratio of previous/following pixel triggers seq.
            max_amp = cluster_amp.max()
            ratio_1 = cluster_amp / max_amp  # amp ratio among current pixel
            n_missing = self.nord - len(cluster_phi)  # number of orders missing
            locs = list(combinations(range(self.nord), n_missing))  # all possible order combos
            corrs = []
            for loc in locs:
                ratio_2 = list(deepcopy(ratio_1))
                for l in loc:
                    ratio_2.insert(l, 0)
                corrs.append(np.dot(ratio_2, ratio))  # cross correlate for the best case
            missing_orders = locs[np.argmax(corrs)]
            p_poly = np.polynomial.polynomial.Polynomial.fit(np.delete(range(self.nord), missing_orders), cluster_phi, 1)
            cluster_phi = p_poly(range(self.nord))
            cluster_amp = [self.binned_counts[nearest_idx(self.bin_centers, phi)] for phi in cluster_phi]
            fit_later = False
        
        if fit_later:
            return  # stop further fitting until all other pixels are finished

        self.all_orders = True
        
        self.leg_s = Legendre(coef=(0, 0, 0), domain=[self.model_energies[0] / self.model_energies[-1] + 0.5, 0.5])  # setup the special sigma Legendre

        cluster_sig = np.array([np.average(cluster_sig)] * self.nord) if ratio is not None else cluster_sig

        self.init_params = init_params(phi_guess=cluster_phi, e_guess=self.model_energies, s_guess=cluster_sig, a_guess=cluster_amp)
        self.opt_params = minimize(fcn=fit_func,  # do nl least squares fitting, return optimized parameter set
                              params=self.init_params,
                              args=(self.bin_centers,  # x_phases
                                    self.binned_counts,  # y_counts
                                    self.orders,  # orders
                                    leg_e,  # energy legendre poly object
                                    self.leg_s))  # sigma legendre poly object

        if not self.opt_params.success or self.opt_params.redchi > 10:  # if unsuccessful, try fitting again with constraints
            c_params = init_params(phi_guess=cluster_phi, e_guess=self.model_energies, s_guess=cluster_sig,
                                   a_guess=cluster_amp, w_constr=True)
            c_opt_params = minimize(fcn=fit_func,
                                    params=c_params,  # params
                                    args=(self.bin_centers,  # x_phases
                                          self.binned_counts,  # y_counts
                                          self.orders,  # orders
                                          leg_e,  # energy legendre poly object
                                          self.leg_s))  # sigma legendre poly object

            if c_opt_params.redchi < self.opt_params.redchi:  # choose the best set of parameters based on redchi2
                self.opt_params = c_opt_params
                self.init_params = c_params

        plot_int = False
        if not self.opt_params.success:  # log which pixels failed to fit
            logger.warning(f'\nPixel {self.n} failed to converge/fit.')
            self.plot_int = True  # overrides plot argument to show any failed fits
        self.redchi2 = self.opt_params.redchi  # save redchi2 to global

        return
        
    def extract_model(self, leg_e, degree: int = 2):
        phi_0 = self.opt_params.params['phi_0'].value
        e_coef = np.array([self.opt_params.params[f'e{c}'].value for c in range(1, degree + 1)])  # no e0
        s_coef = np.array([self.opt_params.params[f's{c}'].value for c in range(degree + 1)])
        self.fit_amp = np.array([self.opt_params.params[f'O{i}_amp'].value for i in range(self.nord)])
        self.fit_amp[self.fit_amp < 1] = 1  # prevents error when finding gaussian intersections
        
        e_coef_convert = Legendre([0, e_coef[0], e_coef[1]], domain=[-1, 0]).convert().coef
        fit_e0_convert = e0_from_params(e1=e_coef_convert[1], e2=e_coef_convert[2], phi_0=phi_0)  # get 0th E coef from other params
        fit_e0 = Legendre([fit_e0_convert, e_coef_convert[1], e_coef_convert[2]]).convert(domain=[-1,0]).coef[0]
        
        setattr(leg_e, 'coef', [fit_e0, e_coef[0], e_coef[1]])  # regenerate the energy legendre poly
        setattr(self.leg_s, 'coef', s_coef)  # regenerate the sigma legendre poly
        self.fit_phi = phis_from_grating_eq(orders=self.orders, phi_0=phi_0, leg=leg_e,
                                        coefs=[fit_e0_convert, e_coef_convert[1], e_coef_convert[2]])  # get other gaussian means
        self.fit_sig = self.leg_s(leg_e(self.fit_phi))  # get all gaussian sigmas

        # store models to array:
        self.gausses_i = fit_func(params=self.opt_params.params, x_phases=self.fine_phase_grid, orders=self.orders,
                                      leg_e=leg_e, leg_s=self.leg_s)  # the individual gaussian models
        self.gausses = np.sum(self.gausses_i, axis=1)  # all gaussians collapsed into one model

    def get_order_edges(self):
        for i in range(self.nord - 1):
            try:
                self.order_edges[i + 1] = gauss_intersect(self.fit_phi[[i, i + 1]],
                                                            self.fit_sig[[i, i + 1]],
                                                            self.fit_amp[[i, i + 1]])  # find the virtual pixel boundaries
            except ValueError:
                if i == 0:  # if the 1st order, makes the 1-to-2 border into 3 sigmas away from order 2
                    self.order_edges[i + 1] = self.fit_phi[i + 1] - self.fit_sig[i + 1] * 3
                elif i == self.nord - 1:  # if the last order, makes the 2ndtolast-to-last border 3 sigs from 2ndtolast
                    self.order_edges[i + 1] = self.fit_phi[i - 1] + self.fit_sig[i - 1] * 3
                else:  # if the intersection cant be found, manually click the location of the order indicated
                    click_edge = None
                    fig = plt.figure()
                    ax = fig.add_subplot(111)
                    ax.grid()
                    ax.bar(self.bin_centers, self.binned_counts, width=self.bin_centers[1] - self.bin_centers[0], linewidth=0,
                           color='k', label='Data')
                    for n, g in enumerate(self.gausses_i.T):
                        ax.plot(self.fine_phase_grid, g, label=f'{self.orders[::-1][n]}')
                    ax.set_title(f'CLICK THE BOUNDARY BETWEEN ORDER {self.orders[::-1][i]-1} AND '
                                 f'{self.orders[::-1][i]}\n then exit the plot')
                    ax.set_xlabel(r'Phase $\times 2\pi$')
                    ax.set_ylabel('Photon Count')
                    klicker = clicker(ax, ["event"])
                    plt.tight_layout()
                    plt.show()
                    self.order_edges[i + 1, p] = klicker.get_positions()['event'][0, 0]
        
    def order_sort(self):
        try:
            # re-histogram the photon table using the virtual pixel edges:
            self.ord_counts, _ = np.histogram(self.photonlist, bins=self.order_edges)

            # find order-bleeding covariance:
            self.covariance = cov_from_params(params=self.opt_params.params, model=self.gausses_i, nord=self.nord,
                                                  order_edges=self.order_edges, x_phases=self.fine_phase_grid)

            cov_inv = np.linalg.inv(self.covariance)  # take the inverse
            self.true_counts = np.dot(self.ord_counts, cov_inv)  # matrix math to retrieve 'true' counts

            # obtain the count error on the MSF-specific spectrum:
            self.p_err = [
                np.sum(self.true_counts * self.covariance[:, m]) - self.true_counts[m] * self.covariance[m, m] for m in
                range(self.nord)]
            self.m_err = [
                np.sum(self.true_counts * self.covariance[m]) - self.true_counts[m] * self.covariance[m, m] for m in
                range(self.nord)]
            if np.abs(np.sum(self.true_counts) - np.sum(self.ord_counts)) > 1:
                logger.warning(f'Pixel {self.n} total calculated and actual counts are '
                               f'{np.abs(np.sum(self.true_counts) - np.sum(self.ord_counts)):.0f} photons apart.')

        except ValueError:  # if the solution cannot be found the pixel is rendered inert
            self.ord_counts = np.nan
            self.true_counts = np.nan
            self.covariance = np.nan
            self.p_err = np.nan
            self.m_err = np.nan
            self.plot_int = False
            logger.warning(f'Pixel {p} has been discarded.')

    def plot(self, leg_e, debug):
        # plot the individual pixels flagged for plotting:
        if self.plot_int or debug:
            fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14, 8))
            axes = axes.ravel()
            ax1 = axes[0]
            ax2 = axes[1]

            plt.suptitle(f'Pixel {self.n}: {"SUCCESS" if self.opt_params.success else "FAILURE"}')

            size1 = '30%'
            size2 = '100%'

            divider1 = make_axes_locatable(ax1)
            divider2 = make_axes_locatable(ax2)

            res1 = divider1.append_axes("top", size=size1, pad=0)
            res2 = divider1.append_axes("bottom", size=size1, pad=0)
            ax2_2 = divider2.append_axes("bottom", size=size2, pad=0)

            ax1.figure.add_axes(res1)
            ax1.figure.add_axes(res2)
            ax2.figure.add_axes(ax2_2)

            # get the initial guess:
            pre_gauss = fit_func(self.init_params, self.fine_phase_grid, orders=self.orders, leg_e=leg_e, leg_s=self.leg_s,
                                 to_sum=True)

            # get the initial residuals and weighted reduced chi^2:
            pre_residual = fit_func(self.init_params, self.bin_centers, y_counts=self.binned_counts, orders=self.orders,
                                    leg_e=leg_e, leg_s=self.leg_s)
            N_dof = len(pre_residual) - self.opt_params.nvarys
            pre_red_chi2 = np.sum(pre_residual ** 2) / N_dof

            # get the post-fitting residuals:
            opt_residual = fit_func(self.opt_params.params, self.bin_centers, y_counts=self.binned_counts, orders=self.orders,
                                    leg_e=leg_e, leg_s=self.leg_s)

            # first half of figure with data and models:
            ax1.grid()
            ax1.bar(self.bin_centers, self.binned_counts, width=self.bin_centers[1] - self.bin_centers[0], linewidth=0, color='k',
                    label='Data')  # plotting the histogram data
            ax1.plot(self.fine_phase_grid, pre_gauss, color='gray', label='Init. Guess')  # the initial guess model
            for i, y in zip(self.orders[::-1], self.gausses_i.T):
                ax1.plot(self.fine_phase_grid, y, label=f'Order {i}')  # the individual order post-fitting models
            ax1.set_ylabel('Photon Count')
            #for b in self.order_edges[:-1]:
            #    ax1.axvline(b, linestyle='--', color='black')  # the virtual pixel boundaries
            #ax1.axvline(self.order_edges[-1], linestyle='--', color='black', label='Order Edges')
            ax1.set_xlim([-1.2, 0])
            ax1.legend()

            res1.grid()
            for x, y in zip(self.bin_centers[:-1], pre_residual[:-1]):
                res1.plot(x, y, '.r')  # the initial weighted residuals
            res1.plot(self.bin_centers[-1], pre_residual[-1], '.r', label=r'Pre Red. $\chi^2=$'f'{pre_red_chi2:.1f}')
            res1.set_ylabel('Weighted Resid.')
            res1.set_xlim([-1.2, 0])

            res2.grid()
            for x, y in zip(self.bin_centers[:-1], opt_residual[:-1]):
                res2.plot(x, y, '.', color='purple')  # the post-fitting weighted residuals
            res2.plot(self.bin_centers[-1], opt_residual[-1], label=r'Post Red. $\chi^2=$'f'{self.redchi2:.1f}')
            res2.set_ylabel('Weighted Resid.')
            res2.set_xlabel(r'Phase $\times 2\pi$')
            res2.set_xlim([-1.2, 0])
            #for b in self.order_edges:
            #    res2.axvline(b, linestyle='--', color='black')  # adding in the virtual pixel boundaries

            # second figure with fitting result polynomials:
            if not np.isnan(self.fit_phi[0]) and not np.isnan(self.fit_phi[-1]):  # changing plot range in case orders missing
                new_x = np.linspace(self.fit_phi[0] - 0.01, self.fit_phi[-1] + 0.01, 1000)
            elif not np.isnan(self.fit_phi[0]):
                new_x = np.linspace(self.fit_phi[0] - 0.01, self.fit_phi[-2] + 0.01, 1000)
            elif not np.isnan(self.fit_phi[-1]):
                new_x = np.linspace(self.fit_phi[1] - 0.01, self.fit_phi[-1] + 0.01, 1000)

            def e_poly_linear(x):  # define the linear equation if the legendre poly had no quad term
                b = leg_e(self.fit_phi[0]) - self.fit_phi[0] * (leg_e(self.fit_phi[-1]) - leg_e(self.fit_phi[0])) / (
                        self.fit_phi[-1] - self.fit_phi[0])
                return (leg_e(self.fit_phi[-1]) - leg_e(self.fit_phi[0])) / (self.fit_phi[-1] - self.fit_phi[0]) * x + b

            masked_reg = energy_to_wave(leg_e(new_x) * self.model_energies[-1] * u.eV)  # calculate the regular legendre
            masked_lin = energy_to_wave(e_poly_linear(new_x) * self.model_energies[-1] * u.eV)  # calc the linear legendre
            deviation = masked_reg - masked_lin  # take the difference

            e_coef = np.array([self.opt_params.params[f'e{c}'].value for c in range(1, 3)])  # no e0
            s_coef = np.array([self.opt_params.params[f's{c}'].value for c in range(3)])

            ax2.grid()
            ax2.plot(new_x, deviation, color='k')  # plotting difference
            for m, i in enumerate(self.fit_phi):
                ax2.plot(i, energy_to_wave(leg_e(i) * self.model_energies[-1] * u.eV) - energy_to_wave(
                    e_poly_linear(i) * self.model_energies[-1] * u.eV), '.',
                         markersize=10, label=f'Order {self.orders[::-1][m]}')  # plot locations of orders
            ax2.set_ylabel('Fitting Solution Dev. from Linear (nm)')
            ax2.legend()

            ax2_2.grid()
            ax2_2.set_ylabel('R')
            ax2_2.set_xlabel(r'Energy (eV)')
            ax2_2.invert_xaxis()
            s_eval = self.leg_s(leg_e(new_x))  # retrieve sigmas from solution
            R = sig_to_R(s_eval, leg_e(new_x))  # convert to spectral res.
            ax2_2.plot(leg_e(new_x), R, color='k')  # plot the R
            for m, i in enumerate(self.fit_phi):
                ax2_2.plot(leg_e(i), sig_to_R(self.fit_sig[m], leg_e(i)), '.', markersize=10,
                           label=f'Order {self.orders[::-1][m]}')  # plot the individual orders
            #ax2.set_title(
           #     r'$E(\varphi)=$'f'{e_coef[1]:.2e}P_2+{e_coef[0]:.2f}P_1+{fit_e0:.2f}P_0\n'
            #    r'$\sigma(E)=$'f'{s_coef[2]:.2e}P_2+{s_coef[1]:.2e}P_1+{s_coef[0]:.2e}P_0'
            #)  # print the 2 solution functions

            ax1.set_xticks([])  # removes axis labels
            ax2.set_xticks([])
            res1.set_xticks([])

            plt.show()
