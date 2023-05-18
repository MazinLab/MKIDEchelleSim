import numpy as np
import scipy.interpolate as interp
import scipy
import scipy.signal as sig
import scipy.ndimage as ndi
import astropy.units as u
import matplotlib.pyplot as plt
import logging

import sortarray
from lmfit import Parameters, minimize, fit_report
from mkidpipeline import photontable as pt

u.photlam = u.photon / u.s / u.cm ** 2 / u.AA  # photon flux per wavelength


def quick_plot(ax, x, y, labels=None, title=None, xlabel=None, ylabel=None, ylim=None, twin=None, first=False, **kwargs):
    """
    Condenses matplotlib plotting into one line.
    :param ax: axes for plotting
    :param x: x axis data
    :param y: y axis data
    :param labels: labels for each set of data
    :param title: title of subplot
    :param xlabel: x axis label
    :param ylabel: y axis label
    :param ylim: y axis upper limit (convenient for twin)
    :param twin: x axis twin color, if any
    :param first: True if first line for plot, will make grid
    :param kwargs: color, linestyle, markerstyle, etc.
    :return: the desired plot (not shown, use plt.show() after)
    """
    if first:
        ax.grid()
    if labels is None:
        labels = ['_nolegend_' in range(len(x))]
    for n, (w, f) in enumerate(zip(x, y)):
        ax.plot(w, f, label=labels[n], **kwargs)
    if title is not None:
        ax.set_title(title)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if ylim is not None:
        ax.set_ylim(top=ylim)
    ax.legend()
    if twin is not None:
        ax.tick_params(axis="y", labelcolor=twin)
        ax.yaxis.label.set_color(twin)
        ax.spines['right'].set_color(twin)
        ax.legend(loc='lower right', labelcolor='linecolor')


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
                 limit_to: int = 1000,
                 area: u.Quantity = np.pi * (4 * u.cm) ** 2,
                 exptime: u.Quantity = 1 * u.s
                 ):
    """
    :param convol_wave: wavelength array that matches result
    :param convol_result: convolution array
    :param area: surface area of the telescope
    :param exptime: exposure time of the observation
    :param limit_to: photons-per-pixel limit to prevent time-consuming simulation
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
    cdf = cdf.decompose()  # todo this is a slopy copy, decomposes units into MKS
    total_photons = cdf[-1, :]

    # putting Poisson draw after limiting because int64 error
    if total_photons.value.max() > limit_to:
        total_photons_ltd = (total_photons.value / total_photons.value.max() * limit_to).astype(int)
        # crappy limit to 1000 photons per pixel for testing
        N = np.random.poisson(total_photons_ltd)
        # Now assume that you want N photons as a Poisson random number for each pixel
        logging.info(f'Limited to a maximum of {limit_to} photons per pixel.')
    else:
        N = np.random.poisson(total_photons.value.astype(int))
        logging.info(f'Max # of photons per pixel: {total_photons.value.max():.2f}.')

    cdf /= total_photons

    logging.info("Beginning random draw for photon wavelengths (from CDF) and arrival times (from uniform random).")
    # Decide on wavelengths and times
    l_photons, t_photons = [], []
    for i, (x, n) in enumerate(zip(cdf.T, N)):
        cdf_interp = interp.interp1d(x, wave_pix[:, i].to('nm').value, fill_value=0, bounds_error=False, copy=False)
        l_photons.append(cdf_interp(np.random.uniform(0, 1, size=n)) * u.nm)
        t_photons.append(np.random.uniform(0, 1, size=n) * exptime)
    logging.info("Completed photon draw, obtained random arrival times and wavelengths for individual photons.")
    return t_photons, l_photons


def _n_bins(n_data):
    # TODO include more algorithms based on number of data points
    """
    :param n_data: number of data points
    :return: best number of bins based on various algorithms
    """
    #if n_data > x:
    #    return bins
    #if 500 <= n_data < y:
    #    return int(np.log2(n_data)+1)
    #if y <= n_data < z:
    #    return bins


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
    return np.sum(gauss(x[:, None], mu, sig, A), axis=0) - y


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
    mu = np.array([param[f'mu_{i}'].value for i in range(n_gauss)])
    sig = np.array([param[f'sig_{i}'].value for i in range(n_gauss)])
    A = np.array([param[f'A_{i}'].value for i in range(n_gauss)])
    return mu, sig, A


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
        blazed_spectrum = blaze_efficiencies * spectra(wave)
        masked_blaze = [blazed_spectrum[i, order_mask[i]] for i in range(len(self.spectrograph.orders))]
        masked_waves = [wave[order_mask[i]].to(u.nm) for i in range(len(self.spectrograph.orders))]
        logging.info('Multiplied spectrum with blaze efficiencies.')
        return blazed_spectrum, masked_waves, masked_blaze

    def pixelspace_blaze(self, wave, spectra):
        """
        :param wave: wavelengths
        :param spectra: flux
        :return: convolved-blazed spectrum
        """
        blazed_spectrum, masked_waves, masked_blaze = self.blaze(wave, spectra)
        pix_leftedge = self.spectrograph.pixel_center_wavelengths(edge='left').to(u.nm).value
        logging.info('Converted blazed spectrum to pixel space.')
        return [self.lambda_to_pixel_space(masked_waves[i], masked_blaze[i], pix_leftedge[i])
                for i in range(self.spectrograph.nord)]

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
        max_mkid_kernel_width = 2 * n_sigma * self.spectrograph.dl_mkid_max() / 2.355  # FWHM to standev
        mkid_kernel_npoints = np.ceil((max_mkid_kernel_width / sampling).si.value).astype(int)  # points in Gaussian
        if not mkid_kernel_npoints % 2:  # ensuring it is odd so 1 point is at kernel center
            mkid_kernel_npoints += 1
        return sig.gaussian(mkid_kernel_npoints, (self.spectrograph.dl_mkid_max() / sampling).si.value / 2.355)
        # takes standev not FWHM, width/sampling is the dimensionless standev

    def mkid_kernel_waves(self, n_points, n_sigma=3, oversampling=10):
        """
        :param n_points: number of points in kernel
        :param n_sigma: number of sigma to either side of mean
        :param oversampling: factor by which to oversample smallest wavelength extent
        :return: wavelength array corresponding to kernel for each pixel
        """
        pixel_rescale = self.spectrograph.pixel_rescale(oversampling).to(u.nm)
        dl_mkid_max = self.spectrograph.dl_mkid_max().to(u.nm)
        sampling = self.spectrograph.sampling(oversampling).to(u.nm)
        npix = self.spectrograph.detector.n_pixels
        nord = self.spectrograph.nord
        return np.array([[np.linspace(-n_sigma * (pixel_rescale[i, j] * dl_mkid_max / sampling).value / 2.355,
                                      n_sigma * (pixel_rescale[i, j] * dl_mkid_max / sampling).value / 2.355,
                                      n_points) for j in range(npix)] for i in range(nord)])

    def determine_mkid_convolution_sampling(self, oversampling: float = 10):
        """
        TODO obsolete, remove once all debugged.
        :param oversampling: factor by which to oversample the smallest change in wavelength in any pixel
        :return: tuple of various detector properties

        Each MKID has its own spectral resolution, which we like to approximate as a Gaussian with width depending
        on wavelength. The spectrograph is sending only certain wavelengths in each order to each pixel.
        This incident spectrum can then be convolved with a Gaussian of varying width and the resulting distribution
        sampled to find "seen" wavelengths. If the variation in R_MKID over a single pixel is small then this can be
        approximated as a convolution with a per-order Gaussian and the results summed.

        kernel width is function of pixel, order
        data is a function of pixel, order

        With say an R_MKID_800nm=15 we would have about a 55nm FWHM and targeting a spectral resolution of 3400
        w/2.5 pix dpix ~ 0.05 nm.
        Well sampling (say 10 samples per pixel) would imply then a 3sigma kernel of about 33000 elements there would
        be so the kernel array would be [norder, npixel, 3300*osamp] or about 1.25GB for a float and 10x osamp.
        This would need to be convolved with the data which would be [norder, npixel, osamp] (negligible)

        However here we can almost certainly play with the sampling to do this more efficiently. By adopting a
        nonuniform sampling (e.g. by allowing dl to be different for each pixel) we are effectively scaling the width
        of the applied Gaussian, which could then be a single oversampled Gaussian out to some nsigma in normalized
        coordinates.

        In the spectrograph the sampling at a given pixel flows from the spectrograph design
        dl_pixel = Dbeta_subtended*lambda(pixel)/2/tan(beta(pixel))=Dbeta_subtended*sigma_grating*cos(beta(pixel))/m
        which by design will be ~ FSR/npix for the last order at the center beta will vary about 6 degrees over the
        order and which will translate to a <±10% effect for likely grating angles so call the pixel sampling that of
        the order average which can be found to be ~ l_center_limiting_order/Npix/order about 0.0355 nm to 0.078 nm
        (for a 400-800 m=5-9 spectrograph).
        The dl_MKID will be l^2/R0/l0. So we require (2*n_sigma_mkid*l_max^2/R0_min/l0) / min(dl_pixel) * osamp
        points across the kernel in the worst case.
        For pixels we want at least osamp samples across the pixel's so that would mean the sampling will need to be
        at least min(dl_pixel)/osamp and at most 10*max(dl_pixel)/min(dl_pixel) samples across a pixel

        For the extrema:
        2*sigma*dl_MKID_800 = 400  # this is the smallest gaussian kernel domain
        2*sigma*dl_MKID_400 = 67  # R0_max not min, this is the smallest gaussian kernel domain
        dl_pix_800 = 0.0781   # the maximum change in wavelength across a pixel
        dl_pix_400 = 0.0373   # l0_center/npix*(1-c_beta*m0/m_max)/m_max  #the minimum change in wavelength across a
        pixel, note the lower angular width at higher order
        """
        spectrograph = self.spectrograph
        detector = self.spectrograph.detector
        max_beta_m0 = spectrograph.grating.beta(spectrograph.l0, spectrograph.m0)  # max reflection angle [rad]
        min_beta_mmax = spectrograph.grating.beta(spectrograph.minimum_wave, spectrograph.m_max)
        # min reflection angle [rad]

        # The maximum and minimum change in wavelength across a pixel
        #  NB You can get a crude approximation with the following where c_beta is the fractional change in beta across the order
        #     e.g. for 6 degrees with a central angle of 45 about .1 in the minimum order
        #     dl_pix_min_wave = spectrograph.central_wave(m0)/npix*(1-c_beta*m0/m_max)/m_max
        #     dl_pix_max_wave = spectrograph.central_wave(m0)/npix*(1+c_beta)/m0
        dl_pix_max_wave = spectrograph.pixel_scale / spectrograph.grating.angular_dispersion(spectrograph.m0,
                                                                                             max_beta_m0)
        # angles per pix/(dbeta/dlambda) [rad/(rad/nm)] = [nm]
        dl_pix_min_wave = spectrograph.pixel_scale / spectrograph.grating.angular_dispersion(spectrograph.m_max,
                                                                                             min_beta_mmax)  # ^

        # Need to compute the worst case kernel width which will be towards the long wavelength end in the lowest order
        # wave_ord0=np.linspace(-.5,.5, num=osamp*detector.n_pixels)*spectrograph.fsr(m0)+spectrograph.central_wave(m0)
        # dl_mkid_max = (wave_ord0**2/detector.mkid_constant(spectrograph.wavelength_to_pixel(wave_ord0))).max()
        dl_mkid_max = (spectrograph.l0 ** 2 / detector.mkid_constant(detector.pixel_indices)).max()
        # largest MKID resolution width is ~ l0/R0 * 1.15 ~ 63 nm, where l0 is 800 nm and R0 is 15.
        sampling = dl_pix_min_wave / oversampling
        # sampling rate is based on min change in wavelength across pixel divided by osamp rate [~ 0.004nm]

        # pixel wavelength centers (nord, npixel)
        # linear array for lowest order, then broadcast andscale via grating equation i.e. m/m'

        # lambda_pixel = (np.linspace(-.5, .5, num=detector.n_pixels) * spectrograph.fsr(m0) +
        #                 spectrograph.central_wave(m0)) * (m0/spectrograph.orders)[:,None]
        # lambda_pixel_pad1 = ((np.linspace(-.5,.5, num=detector.n_pixels+1)-1/detector.n_pixels/2) *
        #                       spectrograph.fsr(m0) + spectrograph.central_wave(m0)) * (m0/spectrograph.orders)[:,None]
        # dl_pixel = np.diff(lambda_pixel_pad1, axis=1)  #dl spectrograph subtended by pixel

        lambda_pixel = spectrograph.pixel_center_wavelengths()  # wavelengths given halfway through each pixel [nm]
        dl_pixel = spectrograph.pixel_scale / spectrograph.angular_dispersion()  # change in wave for each pixel [nm]
        dl_mkid_pixel = detector.mkid_resolution_width(lambda_pixel, detector.pixel_indices)
        # MKID resolution width for each wavelength

        # Each pixel width is some fraction of the kernel's sigma, to handle the stretching properly we need to make
        # sure the correct number of samples are used for the pixel flux. dl_mkid_max/sampling points is mkid sigma
        # so a pixel of dl_pixel width with a mkid sigma of dl_mkid_pixel has dl_mkid_max/sampling points in
        # dl_mkid_pixel so `dl_pixel` nanometers corresponds to dl_pixel/dl_mkid_pixel * dl_mkid_max/sampling points
        # so the sampling of that pixel is dl_mkid_pixel*sampling/dl_mkid_max
        # pixel_samples = dl_pixel/dl_mkid_pixel * dl_mkid_max/sampling
        pixel_rescale = dl_mkid_pixel * sampling / dl_mkid_max
        # rescaled sampling size in wavelength [nm] for each pixel/order

        pixel_samples_frac = (dl_pixel / pixel_rescale).si.value
        # new number of samples for each pixel (min ~18, max ~66)
        pixel_max_npoints = np.ceil(pixel_samples_frac.max()).astype(int)
        # max number of samples for any pixel (will use for all)
        if not pixel_max_npoints % 2:  # ensure there is a point at the pixel center
            pixel_max_npoints += 1

        print(f"\tDetermined MKID convolution sampling rate, from {int(pixel_samples_frac.min())} up to "
              f"{pixel_max_npoints.max()} points per pixel.")
        return (pixel_samples_frac, pixel_max_npoints, pixel_rescale, dl_pixel, lambda_pixel,
                dl_mkid_max, sampling)

    def convolve_mkid_response(self,
                               wave,
                               spectral_fluxden,
                               oversampling: float = 10,
                               n_sigma_mkid: float = 3,
                               full=True
                               ):
        """
        :param wave: wavelength array
        :param spectral_fluxden: flux density array
        :param oversampling: factor by which to sample smallest wavelength extent
        :param n_sigma_mkid: number of sigma to use in kernel
        :param full: True for full convolution, False for simplified
        :return: convolution products of the spectrum with the MKID response
        """
        lambda_pixel = self.spectrograph.pixel_center_wavelengths()
        dl_pixel = self.spectrograph.dl_pixel()
        pixel_rescale = self.spectrograph.pixel_rescale(oversampling)

        if full:
            pixel_max_npoints = self.spectrograph.pixel_max_npoints(oversampling)
            pixel_samples_frac = self.spectrograph.pixel_samples_frac(oversampling)
            x = np.linspace(-pixel_max_npoints // 2, pixel_max_npoints // 2, num=pixel_max_npoints)
            interp_wave = (x[:, None, None] / pixel_samples_frac) * dl_pixel + lambda_pixel
            # wavelength range for samplings in each order/pixel [nm]

            interp_apod = _determine_apodization(x, pixel_samples_frac, pixel_max_npoints)  # [dimensionless]
            # apodization determines the amount of partial flux to use from a bin that is partially outside given pixel
            data = np.zeros(interp_apod.shape)
        else:
            data = np.zeros(dl_pixel.shape)

        # Compute the convolution data
        for i, bs in enumerate(spectral_fluxden):  # interpolation to fill in gaps of spectrum
            specinterp = interp.interp1d(wave.to('nm').value, bs, fill_value=0, bounds_error=False)
            if full:
                data[:, i, :] = specinterp(interp_wave[:, i, :].to('nm').value)
                # [n samplings, 5 orders, 2048 pixels] [photlam]
            else:
                data[i, :] = specinterp(lambda_pixel[i, :].to('nm'))

        if full:
            # Apodize to handle fractional bin flux apportionment
            data *= interp_apod  # flux which is not part of a given pixel is removed

        # Do the convolution
        mkid_kernel = self.build_mkid_kernel(n_sigma_mkid, self.spectrograph.sampling(oversampling))  # returns:
        # a normalized-to-one-at-peak Gaussian is divided into tiny sections as wide as the sampling (dl_pix_min/10)

        if full:
            sl = slice(pixel_max_npoints // 2, -(pixel_max_npoints // 2))
            result = sig.oaconvolve(data, mkid_kernel[:, None, None], mode='full', axes=0)[sl] * u.photlam
            # takes the 67 sampling axis and turns it into ~95000 points each for each order/pixel [~95000, 5, 2048]
            # oaconvolve can't deal with units so you have to give it them
            logging.info('Fully-convolved spectrum with MKID response.')
        else:
            result = data * spectral_fluxden.unit * mkid_kernel[:, None, None]
            logging.info('Multiplied function with MKID response.')

        # TODO I'm not sure this is right, might be simply sampling
        result *= pixel_rescale[None, ...]
        # To fluxden in each rescaled pixel sample multiplied by the sample width (same as [None, :, :])
        # units are [nm * # / Ang / cm^2 / s] dlambda*photlam = FLUX (# / cm^2 / s) not fluxden
        # Compute the wavelengths for the output, converting back to the original sampling, cleverness is done
        # wavelengths span a certain range for each order/pixel [~xxxxx, 5, 2048] [nm] they are sample center
        result_wave = (np.arange(result.shape[0]) - result.shape[0] // 2)[:, None, None] * pixel_rescale[None, ...] \
                      + lambda_pixel

        return result_wave, result

    def multiply_mkid_response(self, wave, spectral_fluxden, oversampling: float = 10, n_sigma_mkid: float = 3):
        """
        TODO obsolete, remove once all debugged.
        :param wave: wavelength array
        :param spectral_fluxden: flux density array
        :param oversampling: factor by which to oversample the smallest extent pixel
        :param n_sigma_mkid: number of sigma to use in kernel
        :return: instead of convolving the MKID response just multiply it by the average flux density in the pixel
        """
        spectral_fluxden = spectral_fluxden.to(u.ph / u.cm ** 2 / u.s / u.nm)
        flux = np.zeros(self.spectrograph.dl_pixel().shape)
        for i, bs in enumerate(spectral_fluxden):
            specinterp = interp.interp1d(wave.to('nm'), bs, fill_value=0, bounds_error=False, copy=False)
            flux[i, :] = specinterp(self.spectrograph.pixel_center_wavelengths()[i, :].to('nm'))

        mkid_kernel = self.build_mkid_kernel(n_sigma_mkid, self.spectrograph.sampling(oversampling))  # returns:
        # a normalized-to-one-at-peak Gaussian is divided into tiny sections as wide as the sampling (dl_pix_min/10)

        result = flux * spectral_fluxden.unit.decompose() * mkid_kernel[:, None, None]

        # Compute the wavelengths for the output, converting back to the original sampling, cleverness is done
        result_wave = (np.arange(result.shape[0]) - result.shape[0] // 2)[:, None, None] * \
                      self.spectrograph.pixel_rescale[None, ...] + self.spectrograph.pixel_center_wavelengths()

        result *= self.spectrograph.pixel_rescale[None, ...]
        print("Conducted simplified 'convolution' with MKID response, "
              "multiplied spectrum with average pixel flux density.")
        return result_wave, result, mkid_kernel

    def lambda_to_pixel_space(self, array_wave, array, leftedge):
        """
        Conducts a direct integration of the fluxden on the wavelength scale to convert to pixel space.
        :param array_wave: wavelength
        :param array: fluxden
        :param leftedge: pixel left edge wavelengths
        :return: direct integration to put flux density of spectrum into pixel space
        """
        flux_int = interp.InterpolatedUnivariateSpline(array_wave.to(u.nm).value, array.to(u.photlam).value, k=1, ext=1)
        integrated = [flux_int.integral(leftedge[j], leftedge[j + 1])
                      for j in range(self.spectrograph.detector.n_pixels - 1)]
        last = [flux_int.integral(leftedge[-1], leftedge[-1] + (leftedge[-1] - leftedge[-2]))]
        return integrated + last
        # if pixel == self.spectrograph.detector.n_pixels - 1:
        #    pixel -= 1
        # if pix_leftedge[order,pixel] > minwave.value:
        #    idx_i = np.where(np.abs((array_wave.to(u.nm).value - pix_leftedge[order,pixel]) < 1e-3))[0][0]
        # else:
        ###if pix_leftedge[order, pixel+1] > minwave.value:
        #    idx_f = np.where(np.abs((array_wave.to(u.nm).value - pix_leftedge[order, pixel + 1]) < 1e-3))[0][0]
        #    return np.sum(array[idx_i:idx_f].to(u.photlam)) * (array_wave[2] - array_wave[1])
        # else:
        #   return 0*u.photlam*u.nm

    def open_table(self, resid_map, table):
        """
        :return: wavelengths for each photon divided into resIDs (pixels)
        """
        file = pt.Photontable(table)
        waves = file.query(column='wavelength')
        resID = file.query(column='resID')
        idx = [np.where(resID == resid_map[j]) for j in range(self.spectrograph.detector.n_pixels)]
        photons_pixel = [waves[idx[j]].tolist() for j in range(self.spectrograph.detector.n_pixels)]
        return photons_pixel

    def _get_params(self, pixel, max_phot, n_gauss):
        params = Parameters()
        s = 1 if n_gauss == 4 else 0
        mus = self.spectrograph.pixel_center_wavelengths()[s:, pixel].to(u.nm).value
        sigs = self.spectrograph.sigma_mkid_pixel()[s:, pixel].to(u.nm).value
        As = [max_phot for i in range(n_gauss)]
        for i in range(n_gauss):
            params.add(f'mu_{i}', value=mus[i], min=mus[i] - sigs[i], max=mus[i] + sigs[i], vary=True)
            params.add(f'sig_{i}', value=sigs[i], min=sigs[i] - sigs[i] / 2, max=sigs[i] + sigs[i] / 2, vary=True)
            params.add(f'A_{i}', value=As[i], min=0, max=2 * max_phot, vary=True)
        return params

    def fit_gaussians(self, pixel: int, photons, n_gauss: int, plot=False):
        # TODO update binning to be more robust
        #bins = _n_bins(len(photons))
        counts, edges = np.histogram(photons, bins=int(5*len(photons)**(1/3)))  # binning photons for shape fitting
        counts = np.array([float(x) for x in counts])
        centers = edges[:-1] + np.diff(edges) / 2
        params = self._get_params(pixel, np.max(photons), n_gauss)
        opt_params = minimize(lmfit_gaussians, params, args=(centers, counts, n_gauss), method='least_squares')

        if plot:
            fig, ax = plt.subplots(1, 1)
            ax.grid()
            quick_plot(ax, centers, counts, labels=['Data'])
            quick_plot(ax, [np.linspace(
                centers[0], centers[-1], 10000)] * 5, gauss(np.linspace(centers[0], centers[-1], 10000), mu, sig, A),
                       labels=[f'Order {i}' for i in self.spectrograph.orders], title=f'Fit for Pixel {pixel}',
                       xlabel='Wavelength (nm)', ylabel='Photon Count')
            ax.legend()
            plt.show()

        return opt_params, params
