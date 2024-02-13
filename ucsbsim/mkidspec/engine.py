import numpy as np
import scipy.interpolate as interp
from scipy.signal import oaconvolve, find_peaks, gaussian
import scipy.ndimage as ndi
from scipy.constants import h, c
import astropy.units as u
import matplotlib.pyplot as plt
import logging

from ucsbsim import sortarray
from ucsbsim.mkidspec.plotting import quick_plot
from mkidpipeline.photontable import Photontable
from lmfit import Parameters, minimize

u.photlam = u.photon / u.s / u.cm ** 2 / u.AA  # photon flux per wavelength
SIG2WID = 2 * np.sqrt(np.log(2))


def sorted_table(table: Photontable, resid_map):
    # SORT PHOTONS BY RESID:
    phases = table.query(column='wavelength')
    resID = table.query(column='resID')
    idx = [np.where(resID == j) for j in resid_map]
    return [phases[j].tolist() for j in idx]  # list of photons in each pixel


def nearest_index(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


def wave_to_eV(waves):
    return ((h*u.J*u.s) * (c * u.m / u.s) / waves).to(u.eV)


def eV_to_wave(eVs):
    return ((h*u.J*u.s) * (c * u.m / u.s) / eVs).to(u.nm)


def _determine_apodization(x, pixel_samples_frac, pixel_max_npoints):
    """
    :param x: wavelength array
    :param pixel_samples_frac: number of samples for every pixel
    :return: apodization as a fraction of each sample, with partial fractions at either edge of pixel

    Samples is a bit of a misnomer in that this is the number of normalized mkid kernel width elements that would
    cover the pixel. So 20.48 elements would be 21 sample points with one at the center and 10 to either side.
    The 10th would be sampling the pixel just a bit shy of its edge (.24 * .5/(20.48/2), where the edge of the
    pixel is at .5. The next sample is out of the wave domain of the pixel, though its bin might still include
    some flux from within the pixel: (.5-(np.ceil(pixel_samples_frac)-pixel_samples_frac)).clip(0)/2

    Note: mkidsigma = dl_pixel*dl_mkid_max/sampling/pixel_samples so MKID R error induced by sampling is
    mkid_sigma_error =
            -dl_pixel*dl_mkid_max/sampling/pixel_samples_frac**2 * (pixel_samples_frac-pixel_samples_frac).std()
    plt.imshow(mkid_sigma_error/dl_mkid_pixel,aspect=2048/5, origin='lower', interpolation='nearest')
    plt.colorbar()
    plt.show()
    but actually this many not be an issue at all and the sampling isn't the points its where those points HIT the
    pixel wavelengths relative to each other and that is handled by the interpolation!

    A (a*dp/dp)%1 >= .5
    v0   v1     v2                 v3    v4
    |  e  |  e  |    e     ||       |  e  | ....
    0     1     2   2.5  pixedge_a  3     4
    flux/dp = v0 + v1 + v2 + v3*(a-floor(a)-.5)
    coord=floor(nsamp/2)+1

    B (a*dp/dp)%1 <=.5
    v0   v1     v2                 v3    v4
    |  e  |  e  |     ||       e   |  e  | ....
    0     1     2  pixedge_a  2.5  3     4
    flux/dp = v0 + v1 + v2*(a-floor(a)+.5)
    coord=floor(nsamp/2)
    """
    # TODO this many not be quite right, cf. interp_wave
    # each pixel split into npoints
    in_pixel = np.abs(x[:, None, None]) <= (pixel_samples_frac / 2)
    # only the points that are within the boundaries of each pixel

    edge = pixel_samples_frac / 2
    last = edge.astype(int)  # farthest left and right points in any given pixel
    caseb = last == np.round(edge).astype(int)
    # keeps only the cases where cutting off decimals is equal to rounding

    apod = edge - edge.astype(int) - .5  # compute case A
    apod[caseb] += 1  # correct values to caseb

    apod_ndx = last + (~caseb)  # Case A needs to go one further
    # ~ is bit-wise inversion -> False becomes True and vice versa. Adding True is +1, False +0
    apod_ndx = np.array([pixel_max_npoints // 2 - apod_ndx, pixel_max_npoints // 2 + apod_ndx])
    apod_ndx.clip(0, pixel_max_npoints - 1, out=apod_ndx)

    interp_apod = np.zeros((pixel_max_npoints,) + pixel_samples_frac.shape)
    ord_ndx = np.arange(interp_apod.shape[1], dtype=int)[None, :, None]
    pix_ndx = np.arange(interp_apod.shape[2], dtype=int)[None, None, :]
    interp_apod[in_pixel] = 1
    interp_apod[apod_ndx[0], ord_ndx, pix_ndx] = apod
    interp_apod[apod_ndx[1], ord_ndx, pix_ndx] = apod

    logging.info("\nDetermined apodization, excess transmission at pixel edges were trimmed.")
    return interp_apod


def draw_photons(convol_wave,
                 convol_result,
                 area: u.Quantity = np.pi * (4 * u.cm) ** 2,
                 exptime: u.Quantity = 1 * u.s
                 ):
    """
    :param convol_wave: wavelength array that matches result
    :param convol_result: convolution array
    :param area: surface area of the telescope
    :param exptime: exposure time of the observation
    :return: the arrival times and randomly chosen wavelengths from CDF
    """
    # Now, compute the CDF from dNdE and set up an interpolating function
    logging.info(f"\nBeginning photon draw, with exposure time: {exptime} and telescope area: {area:.2f}.")
    cdf_shape = int(np.prod(convol_result.shape[:2])), convol_result.shape[-1]  # reshaped to 5 * photons, 2048 pix
    result_p = convol_result.reshape(cdf_shape)
    wave_p = convol_wave.reshape(cdf_shape)
    sort_idx = np.argsort(wave_p, axis=0)
    result_pix = sortarray.sort(result_p, sort_idx)  # sorting by wavelength for proper CDF shape
    wave_pix = sortarray.sort(wave_p, sort_idx)

    cdf = np.cumsum(result_pix, axis=0)
    rest_of_way_to_photons = area * exptime
    cdf *= rest_of_way_to_photons
    cdf = cdf.decompose()  # todo this is a slopy copy, decomposes units into photons
    total_photons = cdf[-1, :]

    # putting Poisson draw after limiting because MKID saturation rate
    if total_photons.value.max() / exptime.value > 1000:  # max 1000 counts per pixel per second
        total_photons_ltd = (total_photons.value / total_photons.value.max() * 1000 * exptime.value).astype(int)
        N = np.random.poisson(total_photons_ltd)
        # Now assume that you want N photons as a Poisson random number for each pixel
        logging.info(f'\tLimited to 1000 photons per pixel per second.')
    else:
        N = np.random.poisson(total_photons.value.astype(int))
        total_photons_ltd = total_photons.value
        logging.info(f'\tMax photons per pixel per second: {N.max() / exptime.value}.')

    reduce_factor = total_photons.value / total_photons_ltd
    cdf /= total_photons

    logging.info("\tBeginning random draw for photon wavelengths (from CDF) and arrival times (from uniform random).")
    # Decide on wavelengths and times
    l_photons, t_photons = [], []
    for i, (x, n) in enumerate(zip(cdf.T, N)):
        cdf_interp = interp.interp1d(x, wave_pix[:, i].to('nm').value, fill_value=0, bounds_error=False, copy=False)
        l_photons.append(cdf_interp(np.random.uniform(0, 1, size=n)) * u.nm)
        t_photons.append(np.random.uniform(0, 1, size=n) * exptime)
    logging.info("Completed photon draw, obtained random arrival times and wavelengths for individual photons.")
    return t_photons, l_photons, reduce_factor


def n_bins(n_data: int, method: str = 'rice'):
    # TODO include more algorithms based on number of data points
    """
    :param int n_data: number of data points
    :param str method: # of bin method to use, only Rice available currently
    :return: best number of bins based on various algorithms
    """
    if method == 'rice':
        return int(4 * n_data ** (1 / 3))  # multiplied by 2 because sparsest pixel likely only has 2 peaks (need 4)
    else:
        raise ValueError(f'Method {method} not supported.')

def gauss(x, mu, sig, A):
    """
    :param x: wavelength
    :param mu: mean
    :param sig: sigma
    :param A: amplitude
    :return: value of the Gaussian
    """
    return (A * np.exp(- (x - mu) ** 2. / (2. * sig ** 2.))).T


def lmfit_gaussians(params, x, y, n_gauss):
    """
    :param params: lmfit Parameter class object containing parameters
    :param x: wavelength bin centers
    :param y: counts for each bin
    :param n_gauss: number of gaussians to fit in pixel
    :return: residual between sum of all Gaussian functions on the order axis and actual data
    """
    mu, sig, A = param_to_array(params, n_gauss, post_opt=False)
    return np.sum(gauss(x, mu[:, None], sig[:, None], A[:, None]), axis=1) - y


def _quad_formula(a, b, c):
    """
    :return: positive quadratic formula result for ax^2 + bx + c = 0
    """
    return (-b + np.sqrt(b ** 2 - 4 * a * c)) / (2 * a)


def gauss_intersect(mu, sig, A):
    """
    :param mu: means of Gaussians in order
    :param sig: sigmas of Gaussians in order
    :param A: amplitudes of Gaussians in order
    :return: analytic calculation of the intersection point between 1D Gaussian functions
    """
    n = len(mu)
    if n != len(sig) or n != len(A):
        raise ValueError("mu, sig, and A must all be the same size.")
    a = [1 / sig[i] ** 2 - 1 / sig[i + 1] ** 2 for i in range(n - 1)]
    b = [2 * mu[i + 1] / sig[i + 1] ** 2 - 2 * mu[i] / sig[i] ** 2 for i in range(n - 1)]
    c = [(mu[i] / sig[i]) ** 2 - (mu[i + 1] / sig[i + 1]) ** 2 - 2 * np.log(A[i] / A[i + 1]) for i in range(n - 1)]
    return _quad_formula(np.array(a), np.array(b), np.array(c))


def param_to_array(params, n_gauss, post_opt=True):
    """
    :param params: Parameter() object
    :param n_gauss: number of Gaussian functions to separate
    :param post_opt: True means extract optimized params, don't add them
    :return: adding new parameters or separating them as arrays
    """
    param = params.params if post_opt else params
    try:
        mu = np.array([param[f'mu_{i}'].value for i in range(n_gauss)])
        sig = np.array([param[f'sig_{i}'].value for i in range(n_gauss)])
        A = np.array([param[f'A_{i}'].value for i in range(n_gauss)])
    except KeyError:
        try:
            mu = np.array([param[f'mu_{i}'].value for i in range(n_gauss - 1)])
            sig = np.array([param[f'sig_{i}'].value for i in range(n_gauss - 1)])
            A = np.array([param[f'A_{i}'].value for i in range(n_gauss - 1)])
        except KeyError:
            try:
                mu = np.array([param[f'mu_{i}'].value for i in range(n_gauss - 2)])
                sig = np.array([param[f'sig_{i}'].value for i in range(n_gauss - 2)])
                A = np.array([param[f'A_{i}'].value for i in range(n_gauss - 2)])
            except KeyError:
                mu = np.array([param[f'mu_{i}'].value for i in range(n_gauss - 3)])
                sig = np.array([param[f'sig_{i}'].value for i in range(n_gauss - 3)])
                A = np.array([param[f'A_{i}'].value for i in range(n_gauss - 3)])
    return mu, sig, A


def _get_params(n_gauss, guess, bounds=None):
    """
    :param n_gauss: number of Gaussians being fit
    :param guess: as list, guess values of mu, sig, and A
    :return: Parameter class object containing parameters with bounds and guesses
    """
    params = Parameters()
    # s = 1 if n_gauss == self.spectrograph.nord-1 else 0
    mus = guess[0]
    sigs = guess[1]
    As = guess[2]

    if bounds is not None:  # TODO add bounds to args
        bounds = bounds
    else:
        bounds = [[[] for i in range(n_gauss)], [[] for i in range(n_gauss)], [[] for i in range(n_gauss)]]

    for i in range(n_gauss):
        params.add(f'mu_{i}', value=mus[i], min=mus[i] - sigs[i], max=mus[i] + sigs[i])
        params.add(f'sig_{i}', value=sigs[i], min=sigs[i] / 2, max=1.5 * sigs[i])
        params.add(f'A_{i}', value=As[i], min=0)
    return params


class Engine:
    def __init__(self, spectrograph):
        self.spectrograph = spectrograph

    def blaze(self, wave, spectra):
        """
        :param wave: wavelengths
        :param spectra: flux
        :return: blazed and masked spectra
        """
        blaze_efficiencies = self.spectrograph.blaze(wave)
        order_mask = self.spectrograph.order_mask(wave.to(u.nm), fsr_edge=False)
        blazed_spectrum = blaze_efficiencies * spectra(wave.to(u.nm))
        masked_blaze = [blazed_spectrum[i, order_mask[i]] for i in range(len(self.spectrograph.orders))]
        masked_waves = [wave[order_mask[i]].to(u.nm) for i in range(len(self.spectrograph.orders))]
        logging.info('Multiplied spectrum with blaze efficiencies.')
        return blazed_spectrum, masked_waves, masked_blaze

    def optically_broaden(self, wave, flux: u.Quantity, axis: int = 1):
        """
        :param wave: wavelength array
        :param flux: array of flux, may be a multidimensional array as long as the last dimension matches wave
        :param axis: the axis in which to optically-broaden
        :return: Gaussian filtered spectrum from spectrograph LSF width

        The optical PSF comes from effects prior to, from, and after the grating. The PSF will be both chromatic
        and non-Gaussian with chromaticity stemming from both the optics and from aberrations as a result of slit
        images taking different paths through the optics. The former would slowly vary over the full wavelength
        domain while the latter would vary over a single order (as well as over the wavelengths in the order). It is
        reasonable to assume that an achromatic Gaussian may be used to represent the intensity profile of slit image
        produced by the camera for a well-designed optical spectrograph. Since this is an effect on the image it can
        be freely done on a per-order basis without worry about interplay between the orders, this facilitates
        low-cost support for a first order approximation of chromaticity by varying the Gaussian width with each order.

        It is technically a sinc of the grating convolved with the optical spot.
        Kernel width is function of order, data is a function of order.

        NB a further approximation can be made by moving a space space with constant sampling in dl/l=c and arguing
        that the width of the LSF is directly proportional to lambda. Doing this does change the effective resolution
        though so care should be taken that there are sufficient samples per pixel, with this approximation the kernel
        a single kernel of fixed width in dl/lambda.

        For now though just treat it as constant and define at the middle of the wavelength range.
        """
        sample_width = wave.mean() * self.spectrograph.nondimensional_lsf_width / np.diff(wave).mean()
        return ndi.gaussian_filter1d(flux, sample_width, axis=axis) * flux.unit

    def build_mkid_kernel(self, n_sigma: float, sampling):
        """
        :param n_sigma: number of sigma to include in MKID kernel
        :param sampling: smallest sample size
        :return: a kernel corresponding to an MKID resolution element of width covering n_sigma on either side.
        mkid_kernel_npoints specifies how many points are used, though the kernel will round up to the next odd number.
        """
        max_mkid_kernel_width = 2 * n_sigma * self.spectrograph.dl_mkid_max / SIG2WID  # FWHM to standev
        mkid_kernel_npoints = np.ceil((max_mkid_kernel_width / sampling).si.value).astype(int)  # points in Gaussian
        if not mkid_kernel_npoints % 2:  # ensuring it is odd so 1 point is at kernel center
            mkid_kernel_npoints += 1
        return gaussian(mkid_kernel_npoints, (self.spectrograph.dl_mkid_max / sampling).si.value / SIG2WID)
        # takes standev not FWHM, width/sampling is the dimensionless standev

    def mkid_kernel_waves(self, n_points, n_sigma=3, oversampling=10):
        """
        :param n_points: number of points in kernel
        :param n_sigma: number of sigma to either side of mean
        :param oversampling: factor by which to oversample smallest wavelength extent
        :return: wavelength array corresponding to kernel for each pixel
        """
        pixel_rescale = self.spectrograph.pixel_rescale(oversampling).to(u.nm)
        dl_mkid_max = self.spectrograph.dl_mkid_max.to(u.nm)
        sampling = self.spectrograph.sampling(oversampling).to(u.nm)
        npix = self.spectrograph.detector.n_pixels
        nord = self.spectrograph.nord
        return np.array([[np.linspace(-n_sigma * (pixel_rescale[i, j] * dl_mkid_max / sampling).value / SIG2WID,
                                      n_sigma * (pixel_rescale[i, j] * dl_mkid_max / sampling).value / SIG2WID,
                                      n_points) for j in range(npix)] for i in range(nord)])

    def convolve_mkid_response(self,
                               wave,
                               spectral_fluxden,
                               oversampling: float = 10,
                               n_sigma_mkid: float = 3,
                               simp: bool = False
                               ):
        """
        :param wave: wavelength array
        :param spectral_fluxden: flux density array
        :param oversampling: factor by which to sample smallest wavelength extent
        :param n_sigma_mkid: number of sigma to use in kernel
        :param simp: True for simplified convolution, False for full
        :return: convolution products of the spectrum with the MKID response
        """
        lambda_pixel = self.spectrograph.pixel_wavelengths()
        dl_pixel = self.spectrograph.dl_pixel
        pixel_rescale = self.spectrograph.pixel_rescale(oversampling)

        if simp:
            data = np.zeros(dl_pixel.shape)
        else:
            pixel_max_npoints = self.spectrograph.pixel_max_npoints(oversampling)
            pixel_samples_frac = self.spectrograph.pixel_samples_frac(oversampling)
            x = np.linspace(-pixel_max_npoints // 2, pixel_max_npoints // 2, num=pixel_max_npoints)
            interp_wave = (x[:, None, None] / pixel_samples_frac) * dl_pixel + lambda_pixel
            # wavelength range for samplings in each order/pixel [nm]

            interp_apod = _determine_apodization(x, pixel_samples_frac, pixel_max_npoints)  # [dimensionless]
            # apodization determines the amount of partial flux to use from a bin that is partially outside given pixel
            data = np.zeros(interp_apod.shape)

        # Compute the convolution data
        for i, bs in enumerate(spectral_fluxden):  # interpolation to fill in gaps of spectrum
            specinterp = interp.interp1d(wave.to('nm').value, bs, fill_value=0, bounds_error=False)
            if simp:
                data[i, :] = specinterp(lambda_pixel[i, :].to('nm'))
            else:
                data[:, i, :] = specinterp(interp_wave[:, i, :].to('nm').value)
                # [n samplings, 5 orders, 2048 pixels] [photlam]

        if not simp:
            # Apodize to handle fractional bin flux apportionment
            data *= interp_apod  # flux which is not part of a given pixel is removed

        # Do the convolution
        mkid_kernel = self.build_mkid_kernel(n_sigma_mkid, self.spectrograph.sampling(oversampling))  # returns:
        # a normalized-to-one-at-peak Gaussian is divided into tiny sections as wide as the sampling (dl_pix_min/10)

        if simp:
            result = data * spectral_fluxden.unit * mkid_kernel[:, None, None]
            logging.info('Multiplied function with MKID response.')
        else:
            sl = slice(pixel_max_npoints // 2, -(pixel_max_npoints // 2))
            result = oaconvolve(data, mkid_kernel[:, None, None], mode='full', axes=0)[sl] * u.photlam
            # takes the 67 sampling axis and turns it into ~95000 points each for each order/pixel [~95000, 5, 2048]
            # oaconvolve can't deal with units so you have to give it them
            logging.info('Fully-convolved spectrum with MKID response.')

        # TODO I'm not sure this is right, might be simply sampling
        result *= pixel_rescale[None, ...]
        # To fluxden in each rescaled pixel sample multiplied by the sample width (same as [None, :, :])
        # units are [nm * # / Ang / cm^2 / s] dlambda*photlam = FLUX (# / cm^2 / s) not fluxden
        # Compute the wavelengths for the output, converting back to the original sampling, cleverness is done
        # wavelengths span a certain range for each order/pixel [~xxxxx, 5, 2048] [nm] they are sample center
        result_wave = (np.arange(result.shape[0]) - result.shape[0] // 2)[:, None, None] * pixel_rescale[None, ...] \
                      + lambda_pixel

        return result_wave, result

    def lambda_to_pixel_space(self, array_wave, array, leftedge):
        """
        Conducts a direct integration of the fluxden on the wavelength scale to convert to pixel space.
        :param array_wave: wavelength
        :param array: fluxden as photlam
        :param leftedge: pixel left edge wavelengths in AA
        :return: direct integration to put flux density of spectrum into pixel space
        """
        if isinstance(array, u.Quantity):
            array = np.nan_to_num(array.to(u.ph / u.nm / u.cm ** 2 / u.s).value)
        else:
            array = np.nan_to_num(array)
        if isinstance(array_wave, u.Quantity):
            array_wave = np.nan_to_num(array_wave.to(u.nm).value)
        else:
            array_wave = np.nan_to_num(array_wave)
        if isinstance(leftedge, u.Quantity):
            leftedge = np.nan_to_num(leftedge.to(u.nm).value)
        else:
            leftedge = np.nan_to_num(leftedge)
        flux_int = interp.InterpolatedUnivariateSpline(array_wave, array, k=1, ext=1)
        integrated = [flux_int.integral(leftedge[j], leftedge[j + 1])
                      for j in self.spectrograph.detector.pixel_indices[:-1]]
        last = [flux_int.integral(leftedge[-1], leftedge[-1] + (leftedge[-1] - leftedge[-2]))]
        return integrated + last

    def fit_gaussians(
            self,
            pixel: int,
            photons,
            n_ord: int,
            bin_edges,
            plot=False
    ):
        """
        :param pixel: pixel
        :param photons: list of photon wavelengths
        :param n_ord: number of orders
        :param bin_edges: sequence of bin_edges for pixel histogram
        :param plot: True if viewing the fit for the pixel
        :return: optimized and guess parameters for mu, sigma, and amplitude
        """

        # for any photons that got wrapped around to 1 at -1, connect it back so we can histogram
        for n, p in enumerate(photons):
            while photons[n] > 0:
                photons[n] -= 2

        counts, edges = np.histogram(photons, bins=bin_edges)

        # turn integer counts into floats
        counts = np.array([float(x) for x in counts])

        # change the edges into center points
        centers = edges[:-1] + np.diff(edges) / 2

        # find peaks and retrieve info
        peaks, peak_dict = find_peaks(counts, height=3, width=1, distance=int(len(bin_edges) / 20))
        peak_heights = peak_dict['peak_heights']
        peak_widths = peak_dict['widths']
        n_gauss = min([n_ord, len(peaks)])

        # get up to n_ord peaks, less if no flux in some order
        top_peak = peaks[np.argsort(peak_heights)[::-1][:n_gauss]]
        top_peak = np.array([int(i) for i in top_peak])

        # get the FWHM so they can be turned into sigma guesses
        top_width = peak_widths[np.argsort(peak_heights)[::-1][:n_gauss]]

        # reorder the found peaks/widths in phase
        ordered_peak = top_peak[np.argsort(centers[top_peak])]
        ordered_width = top_width[np.argsort(centers[top_peak])]

        # retrieve guesses for lmfit
        mu_guess = centers[ordered_peak]
        A_guess = counts[ordered_peak]
        sig_guess = (np.abs(bin_edges[1] - bin_edges[0]) * ordered_width) / SIG2WID

        # turn guesses into parameter objects and do the lmfit minimization
        guess = [mu_guess, sig_guess, A_guess]
        params = _get_params(n_gauss, guess=guess)
        opt_params = minimize(lmfit_gaussians, params, args=(centers, counts, n_gauss), method='least_squares')
        mu, sig, A = param_to_array(opt_params, n_gauss, post_opt=True)

        # plot the best-fit 
        if plot:
            fig, ax = plt.subplots(1, 1)
            quick_plot(ax, [centers], [counts], labels=['Data'], first=True, color='k')
            quick_plot(ax, [mu_guess], [A_guess], labels=['Guess'], marker='.')
            ax.fill_between(centers, 0, counts, color='k')
            x = [np.linspace(bin_edges[0], bin_edges[-1], 1000) for i in range(n_gauss)]
            quick_plot(ax, x, gauss(x, mu[:, None], sig[:, None], A[:, None]).T,
                       labels=[f'Order {i}' for i in self.spectrograph.orders[::-1]],
                       title=f'Least-Squares Fit for Pixel {pixel}', xlabel='Phase', ylabel='Photon Count')
            plt.legend()
            plt.show()

        return (mu, sig, A), counts, opt_params.redchi
