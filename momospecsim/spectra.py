import numpy as np
import logging
import pandas as pd
import sys
from astropy import units as u
from astropy.constants import R_sun
from specutils import Spectrum1D
from synphot import SpectralElement, SourceSpectrum, units, blackbody
from synphot.models import Box1D, BlackBody1D, ConstFlux1D, Empirical1D
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import scipy.ndimage as ndi
import warnings

from momospecsim.utils.general import gauss
from momospecsim.detector import wave_to_phase
from mkidpipeline.photontable import Photontable
from mkidpipeline.steps.buildhdf import buildfromarray

u.photlam = u.photon / u.s / u.cm ** 2 / u.AA  # new unit name, photon flux per wavelength

logger = logging.getLogger('spectra')


def Throughput(filename):
    """
    :param filename: wavelength in nm, .txt: space-delimited, .txt/.csv: header is wavelength/transmission, .dat: none
    :return: throughput as dimensionless SpectralElement object
    """
    if filename.endswith('.csv') or filename.endswith('.txt'):
        delim = ',' if filename.endswith('.csv') else ' '
        file = pd.read_csv(filename, delimiter=delim)
        w = np.array(file['wavelength'])[::-1] * u.nm
        thru = np.array(file['transmission'])[::-1] * u.dimensionless_unscaled
        thru[thru < 0] = 0
        thru = Spectrum1D(spectral_axis=w, flux=thru)

    elif filename.endswith('.dat'):
        file = np.genfromtxt(filename)
        thru = Spectrum1D(spectral_axis=file[:, 0] * u.nm, flux=file[:, 1] * u.dimensionless_unscaled)
    
    return SpectralElement.from_spectrum1d(thru)


def AtmosphericTransmission():
    """
    :return: atmospheric transmission as SpectralElement object
    """
    x = np.genfromtxt('../momospecsim/simfiles/atmosphere/transmission.dat')
    spec = Spectrum1D(spectral_axis=x[:, 0] * u.nm, flux=x[:, 1] * u.dimensionless_unscaled)
    return SpectralElement.from_spectrum1d(spec)


def FridgeTransmission():
    """
    :return: transmission through two Asahi supercold fridge filters as SpectralElement object
    """
    file = pd.read_csv('simfiles/thruput/Asahi_YSC0750.csv', delimiter=',')
    thru = np.array(file['transmission'])[::-1] * u.dimensionless_unscaled
    thru[thru < 0] = 0
    w = np.array(file['wavelength'])[::-1] * u.nm
    return SpectralElement.from_spectrum1d(Spectrum1D(spectral_axis=w, flux=np.dot(thru, thru)))


def FineGrid(min, max, npoints=100000):
    """
    :param min: minimum wavelength as u.Quantity
    :param max: maximum wavelength as u.Quantity
    :param npoints: number of points in grid
    :return: returns input spectrum with this grid spacing, in case it is lower resolution 
    """
    w = np.linspace(min.to(u.nm).value - 100, max.to(u.nm).value + 100, npoints) * u.nm
    t = np.ones(100000) * u.dimensionless_unscaled
    return SpectralElement.from_spectrum1d(Spectrum1D(spectral_axis=w, flux=t))


def apply_bandpass(spectra, bandpass):
    """
    :param spectra: spectra to apply bandpasses to, as list or object
    :param bandpass: the filter(s) to be applied, as list or object
    :return: original spectrum multiplied with bandpasses
    """
    if not isinstance(spectra, list):
        spectra = [spectra]
        not_list = True
    if not isinstance(bandpass, list):
        bandpass = [bandpass]
    for i, s in enumerate(spectra):
        for b in bandpass:
            s *= b
        spectra[i] = s
    logger.info(f'Multipled spectrum with given bandpass.')
    if not_list:
        return spectra[0]
    else:
        return spectra


def SkyEmission(fov):
    """
    :param fov: field of view in arcsec^2
    :return: night sky emission in photons/sec/m^2/um
    """
    file = np.genfromtxt('../momospecsim/simfiles/sky_emission/radiance.dat')
    w = file[:, 0] * u.nm
    f = file[:, 1] * u.ph / u.s / u.m ** 2 / u.um
    spec = Spectrum1D(spectral_axis=w, flux=f * fov)
    return SourceSpectrum.from_spectrum1d(spec)


def PhoenixModel(distance: float, radius: float, teff: float, feh=0, logg=4.8):
    """
    :param distance: distance to star
    :param radius: radius of star
    :param float teff: effective temperature of star
    :param feh: metallicity
    :param logg: log of surface gravity
    :return: Phoenix model of star with given properties as SourceSpectrum object
    """
    from expecto import get_spectrum
    sp = SourceSpectrum.from_spectrum1d(get_spectrum(T_eff=teff, log_g=logg, Z=feh, cache=True))
    e_sp = sp.integrate(flux_unit=units.FLAM, integration_type='analytical')
    default_distance = radius * np.sqrt(sigma_sb * teff ** 4 * u.K ** 4 / e_sp).decompose()
    sp /= ((distance / default_distance) ** 2).decompose()

    return sp


def BlackbodyModel(distance: float, radius: float, teff: float):
    """
    :param distance: distance to star as u.Quantity
    :param radius: radius of star as u.Quantity
    :param float teff: effective temperature of model star
    :return: blackbody model of star as SourceSpectrum object
    """
    sp = SourceSpectrum(BlackBody1D, temperature=teff)  # returns photlam per steradian
    return sp


def FlatModel(minwave, maxwave, flux_level=1e6):
    """
    :return: model which returns the same flux density at all wavelengths
    """
    sp = SourceSpectrum(ConstFlux1D, amplitude=flux_level * units.FLAM)
    waves = np.arange(minwave.to(u.nm).value, maxwave.to(u.nm).value, 0.01) * u.nm

    # for the typical lab environment
    watt = 3 * u.W  # typical lamp wattage
    dist = 40 * u.cm  # approx distance from lamp to camera
    flux_w = watt / dist ** 2
    e_sp = sp.integrate(wavelengths=waves, flux_unit=units.FLAM, integration_type='analytical')
    ratio = 1 if e_sp == 0 else (flux_w / e_sp).decompose()

    sp = SourceSpectrum(ConstFlux1D, amplitude=flux_level * u.photlam)
    return SourceSpectrum.from_spectrum1d(Spectrum1D(flux=sp(waves) * ratio, spectral_axis=waves))


def EmissionModel(filename, minwave, maxwave, target_R=50000):
    """
    :param filename: file name of the emission line list, with wavelength in nm, FROM NIST
    :param minwave: the min wave of the desired model, in nm or as u.Quantity
    :param maxwave: the max wave of the desired model, in nm or as u.Quantity
    :param target_R: spectral resolution to diffraction limit line spectrum
    :return: full emission spectrum, intensity converted to photlam
    """
    file = pd.read_csv(filename, delimiter=',')
    flux = np.array(file['intens'])
    wave = np.array(file['obs_wl_air(nm)'])
    try:  # see if file comes with wavelength uncertainties on lines
        uncert = np.array(file['unc_obs_wl'])
    except KeyError:
        uncert = np.full(wave.shape, 0.0010)

    if isinstance(flux[0], str):  # parse the flux strings, some are empty, dont use them
        include = np.full(len(flux), False)
        for n, i in enumerate(flux):
            try:
                flux[n] = float(i[2:-1])
                wave[n] = float(wave[n][2:-1])
                if isinstance(uncert[n], str):
                    uncert[n] = float(uncert[n][2:-1])
                include[n] = True
            except ValueError:
                include[n] = False
        flux = flux[include]
        wave = wave[include]
        uncert = uncert[include]
    target_dl = (wave[0] + wave[-1]) / 2 / target_R  # determine dlambda for given R
    sigma_factor = target_dl / min(uncert)  # 3 sigma approx to 1st Airy ring

    if isinstance(minwave, u.Quantity):
        minwave = minwave.to(u.nm).value
    if isinstance(maxwave, u.Quantity):
        maxwave = maxwave.to(u.nm).value

    # create gaussians with some width for each line and sum
    wave_grid = np.arange(minwave, maxwave, target_dl)
    line_gauss = gauss(wave_grid[None, :].astype(float), wave[:, None].astype(float),
                       uncert[:, None].astype(float) * sigma_factor / 3, flux[:, None].astype(float))
    spectrum = np.sum(line_gauss, axis=1)
    sp = SourceSpectrum.from_spectrum1d(Spectrum1D(flux=spectrum * u.photlam, spectral_axis=wave_grid * u.nm))

    # for the typical lab environment
    watt = 3 * u.W  # typical emission lamp wattage
    dist = 40 * u.cm  # approx distance from lamp to camera
    flux_w = watt / dist ** 2
    e_sp = sp.integrate(flux_unit=units.FLAM, integration_type='analytical')
    ratio = (flux_w / e_sp).decompose()  # attentuation factor
    return sp * ratio


def SpecFromFile(filename: str):
    """
    :param filename: Directory/filename of spectrum, wavelengths must be in nm
    :return: spectrum from file
    """
    file = np.genfromtxt(filename)
    return SourceSpectrum(Empirical1D, points=file[:, 0] * u.nm, lookup_table=file[:, 1] * u.photlam)


class Target:
    def __init__(
            self,
            spectype: str = None,
            dist: float = None,
            rad: float = None, 
            temp: float = None,
            spec_file: str = None,
            minwave: float = None,
            maxwave: float = None,
            objsize: float = None, 
            seeing: float = None,
            on_sky: bool = False
    ):
        """
        :param spectype: type of spectrum to be simulated
        :param dist: distance to target in parsecs (only used if target is on_sky)
        :param rad: radius of the target in units of R_sun (only used when spectype is blackbody/phoenix)
        :param temp: temperature of spectrum in K (only used when spectype is blackbody/phoenix)
        :param spec_file: directory/filename of spectrum (required if spectype is emission/from_file)
        :param minwave: minimum operating wavelength in nm
        :param maxwave: maximum operating wavelength in nm
        :param size: angular diameter of target in mas (only used when spectype is blackbody/phoenix)
        :param seeing: seeing disk diameter in arcsec (only used if target is on_sky)
        :param on_sky: pass True if observation is on sky
        """
        self.spectype = spectype
        self.dist = dist
        self.rad = rad
        self.temp = temp
        self.spec_file = spec_file
        self.minwave = minwave * u.nm
        self.maxwave = maxwave * u.nm
        self.objsize = objsize
        self.seeing = seeing
        self.on_sky = True if spectype is 'sky_emission' else on_sky
        self._spectrum = None  # will be updated as spectrum goes through changes
        self._size = None  # will be updated as apparent size goes through changes
        self._waveset = None
        self._photonlist = None
        self.photons_realign = None
        logger.info('Target initialized.')

    @property
    def spectrum(self):
        return self._spectrum
    
    @spectrum.setter
    def spectrum(self, spectrum):
        self._spectrum = spectrum
        
    @property
    def size(self):
        return self._size
    
    @size.setter
    def size(self, size):
        self._size = size
        
    @property
    def waveset(self):
        return self._waveset
    
    @waveset.setter
    def waveset(self, waveset):
        self._waveset = waveset

    @property
    def photonlist(self):
        return self._photonlist

    @photonlist.setter
    def photonlist(self, photonlist):
        self._photonlist = photonlist

    @property
    def fov(self):
        """
        :return: field of view of target in arcsec
        """
        return np.pi * (self.size / 2) ** 2

    def init_spectrum(self):
        """
        :return: initial spectrum of chosen type
        """
        if self.spectype == 'blackbody':
            logger.info(f'Obtained blackbody model spectrum.')
            spec = BlackbodyModel(distance=self.dist, radius=self.rad, teff=self.temp)
        elif self.spectype == 'phoenix':
            logger.info(f'Obtained Phoenix model spectrum.')
            spec = PhoenixModel(distance=self.dist, radius=self.rad, teff=self.temp)
        elif self.spectype == 'flat':
            logger.info(f'Obtained flat-field model spectrum.')
            spec = FlatModel(self.minwave, self.maxwave)
        elif self.spectype == 'emission':
            logger.info(f'Obtained {self.spec_file} emission spectrum.')
            spec = EmissionModel(self.spec_file, self.minwave, self.maxwave)
        elif self.spectype == 'sky_emission':
            logger.info(f'Obtained sky emission spectrum.')
            spec = FlatModel(self.minwave, self.maxwave, flux_level=0)
        elif self.spectype == 'from_file':
            logger.info('Obtained spectrum from file.')
            spec = SpecFromFile(filename=self.spec_file)
        else:
            raise ValueError("Only 'blackbody', 'phoenix', 'flat', 'emission', 'sky_emission', "
                             "or 'from_file' are supported for spectype.")
        init_spectrum = spec * self.fov * FineGrid(self.minwave, self.maxwave)  # increases sampling rate
        if self.on_sky:
            return init_spectrum + SkyEmission(fov=self.fov)
        return init_spectrum

    def clip_spectrum(self, clip_range: tuple=None):
        """
        :param tuple clip_range: wavelength range to retain
        :return: SourceSpectrum with all entries outside of clip_range discarded
        """
        if self.spectrum is None:
            raise ValueError("spectrum is None, nothing to clip!")
        clip_range = [self.minwave, self.maxwave] if clip_range is None else clip_range
        mask = (self.spectrum.waveset >= clip_range[0]) & (self.spectrum.waveset <= clip_range[-1])
        self.spectrum = SourceSpectrum.from_spectrum1d(Spectrum1D(
            spectral_axis=self.spectrum.waveset[mask],
            flux=self.spectrum(self.spectrum.waveset[mask])))
        logger.info(f"Clipped spectrum to{clip_range}.")
        self.waveset = self.spectrum.waveset

    def init_size(self, aperture=None, fiber_angle=None):
        """
        :param aperture: telescope aperture diameter in Astropy units
        :param fiber_angle: acceptance angle of fiber in rad
        :return: initial apparent viewing diameter of target in arcsec
        """
        if self.on_sky:  # takes largest of Airy disk, seeing disk, and object size
            airy = ((1.029 * (self.minwave + self.maxwave) / 2 / aperture).decompose() * u.rad).to(u.arcsec).value
            return np.max([self.size / 1000, self.seeing, airy])
        return self.dist / np.tan(fiber_angle) * 2  # returns size given fiber acceptance angle

    def diverge(self, fiber, distance):
        """
        :param fiber: Fiber object
        :param distance: distance between target and new fiber
        :return: None (target spectrum throughput reduced, such as direct fiber to fiber coupling)
        """
        if self.spectrum is None:
            raise ValueError("spectrum is None, nothing to diverge!")
        new_size = self.size + 2 * distance / np.tan(fiber.accept_angle)
        ratio = self.size / new_size
        self.spectrum *= ratio

    def optically_broaden(self, nondim_lsf_width, axis: int = 1):
        """
        :param nondim_lsf_width: Spectrograph nondimensional LSF width
        :param axis: axis in which to optically-broaden
        :return: None (spectrum is optically-broadened by spectrograph)
        
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

        Treat it as constant and define at the middle of the wavelength range.
        """
        if self.spectrum is None:
            raise ValueError("spectrum is None, nothing to broaden!")
        sample_width = self.waveset.mean() * nondim_lsf_width / np.diff(self.waveset).mean()
        self.spectrum = ndi.gaussian_filter1d(self.spectrum, sample_width / (2 * np.sqrt(2 * np.log(2))), axis=axis) * self.spectrum.unit

    def plot(self, title=''):
        if self.spectrum is None:
            raise ValueError("spectrum is None, nothing to plot!")
        plt.grid()
        plt.plot(self.waveset.to(u.nm), self.spectrum(self.waveset))
        plt.title(title)
        plt.xlabel('Wavelength (nm)')
        plt.ylabel(r'Photon Flux Density (ph $\AA^{-1} cm^{-2} s^{-1}$)')
        plt.tight_layout()
        plt.show()

    def plot_heatmap(self, detector):
        npix = detector.npix
        idx = [np.where(self.photonlist.resID == detector.resid_map[j]) for j in range(npix)]
        self.photons_realign = [(self.photonlist.wavelength[idx[j]] / detector.phase_offsets[j]).tolist() for j in range(npix)]

        bin_edges = np.linspace(-1, -0.1, 100)
        centers = bin_edges[:-1] + np.diff(bin_edges) / 2
        hist_array = np.zeros([npix, len(bin_edges) - 1])
        for j in detector.pixel_indices:
            if self.photons_realign[j]:
                counts, edges = np.histogram(a=self.photons_realign[j], bins=bin_edges)
                hist_array[j, :] = np.array([float(x) for x in counts])
        plt.imshow(hist_array[:, ::-1].T, extent=[1, npix, -1, -0.1], aspect='auto', norm=LogNorm())
        cbar = plt.colorbar()
        cbar.ax.set_ylabel('Photon Count')
        plt.title(f"Binned Pixel Heat Map w/o Offset")
        plt.xlabel("Pixel Index")
        plt.ylabel(r"Phase ($\times \pi /2$)")
        plt.tight_layout()
        plt.show()
        
    def plot_comparison(self, spectrograph, detector, engine, convol_result, blazed_spectrum, reduce_factor, exptime):
        logger.info(msg='Plotting for debugging...')
        warnings.filterwarnings(action="ignore")  # ignore tight_layout warnings

        lambda_pixel = spectrograph.pixel_wavelengths().to(u.nm)
        nord = spectrograph.nord
        npix = detector.npix
        
        # sum the convolution to go to pixel-order array size:
        convol_sum = np.sum(convol_result, axis=0)

        # use FSR to bin and order sort:
        fsr = spectrograph.fsr(order=spectrograph.orders).to(u.nm)
        hist_bins = np.empty((nord + 1, npix))  # choosing rough histogram bins by using FSR of each pixel/wave
        hist_bins[0, :] = (lambda_pixel[-1, :] - fsr[-1] / 2).value
        hist_bins[1:, :] = [(lambda_pixel[i, :] + fsr[i] / 2).value for i in range(nord)[::-1]]
        hist_bins = wave_to_phase(waves=hist_bins, minwave=self.minwave, maxwave=self.maxwave)

        photons_binned = np.empty((nord, npix))
        for j in range(npix):
            photons_binned[:, j], _ = np.histogram(a=self.photons_realign[j], bins=hist_bins[:, j], density=False)

        # normalize to level of convolution since that's where it came from and calculate noise:
        photons_binned = (
                photons_binned * u.ph * reduce_factor[None, :] / (exptime * u.s)).to(u.ph / u.s).value

        lambda_left = spectrograph.pixel_wavelengths(edge='left')
        blazed_int_spec = np.array([engine.lambda_to_pixel_space(array_wave=self.waveset,
                                                              array=blazed_spectrum[i],
                                                              leftedge=lambda_left[i]) for i in range(nord)])

        # plotting comparison between flux-integrated spectrum, integrated/convolved spectrum, & final counts FSR-binned
        plt.grid()
        for n in range(nord - 1):
            plt.plot(lambda_pixel[n], photons_binned[::-1][n], color='k', linewidth=1, linestyle='--')
            plt.plot(lambda_pixel[n], convol_sum[n], color='red', linewidth=1.5, alpha=0.5)
            plt.plot(lambda_pixel[n], blazed_int_spec[n], color='b')

        plt.ylabel(r"Flux (phot $cm^{-2} s^{-1})$")
        plt.xlabel('Wavelength (nm)')
        plt.title('Comparison of Pre/Post-Convolution and Photon Table Spectrum')
        plt.plot(lambda_pixel[-1], photons_binned[::-1][-1], color='k', linewidth=1, linestyle='--',
                 label='Photon Table Binned')
        plt.plot(lambda_pixel[-1], convol_sum[-1], color='r', linewidth=1.5, alpha=0.5, label='Post-Convolution')
        plt.plot(lambda_pixel[-1], blazed_int_spec[-1], color='b', label='Pre-Convolution')
        plt.tight_layout()
        plt.legend()
        plt.show()
        pass

