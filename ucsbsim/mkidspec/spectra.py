import numpy as np
import logging
import pandas as pd
import sys
from astropy import units as u
from astropy.constants import sigma_sb, h, c, R_earth
from specutils import Spectrum1D
from synphot import SpectralElement, SourceSpectrum, units
from synphot.models import Box1D, BlackBodyNorm1D, ConstFlux1D, Empirical1D

from ucsbsim.mkidspec.utils.general import gauss

u.photlam = u.photon / u.s / u.cm ** 2 / u.AA  # new unit name, photon flux per wavelength

logger = logging.getLogger('spectra')


def AtmosphericTransmission():
    """
    :return: atmospheric transmission as SpectralElement object
    """
    x = np.genfromtxt('../mkidspec/simfiles/transmission.dat')
    spec = Spectrum1D(spectral_axis=x[:, 0] * u.nm, flux=x[:, 1] * u.dimensionless_unscaled)
    return SpectralElement.from_spectrum1d(spec)


def TelescopeTransmission(reflectivity: float = .9):
    """
    :param reflectivity: attenuation due to reflectivity of the telescope as fraction, (1 means no reflection)
    :return: transmission due to telescope reflectivity, smaller at higher wavelengths, as SpectralElement object
    """
    w = np.linspace(300, 1500, 10000) * u.nm
    t = np.linspace(1, .95, 10000) * reflectivity * u.dimensionless_unscaled
    spec = Spectrum1D(spectral_axis=w, flux=t)
    return SpectralElement.from_spectrum1d(spec)


def FilterTransmission():
    """
    :return: transmission through fridge filter as SpectralElement object
    """
    file = pd.read_csv('../mkidspec/simfiles/fridge_filter.csv', delimiter=',')
    flux = np.array(file['transmission'])[::-1]*u.dimensionless_unscaled
    flux[flux < 0] = 0
    spec = Spectrum1D(spectral_axis=np.array(file['wavelength'])[::-1]*u.nm, flux=flux)
    return SpectralElement.from_spectrum1d(spec)


def FineGrid(min, max, npoints=100000):
    """
    :param min: minimum wavelength as u.Quantity
    :param max: maximum wavelength as u.Quantity
    :param npoints: number of points in grid
    :return: returns input spectrum with this grid spacing, in case it is lower resolution 
    """
    w = np.linspace(min.to(u.nm).value-100, max.to(u.nm).value+100, npoints) * u.nm
    t = np.ones(100000) * u.dimensionless_unscaled
    ones = Spectrum1D(spectral_axis=w, flux=t)
    return SpectralElement.from_spectrum1d(ones)


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
    :param fov: field of view at center wavelength in arcsec2
    :return: night sky emission
    """
    file = np.genfromtxt('../mkidspec/simfiles/sky_emission/radiance.dat')
    spec = Spectrum1D(spectral_axis=file[:, 0]*u.nm, flux=file[:, 1]*(fov/2)**2*u.ph/u.s/u.m**2/u.um)
    return SourceSpectrum.from_spectrum1d(spec)


def PhoenixModel(distance: float, 
                 radius: float, 
                 teff: float, 
                 feh=0, 
                 logg=4.8, 
                 desired_magnitude=None, 
                 on_sky=False, 
                 fov=None):
    """
    :param distance: distance to star
    :param radius: radius of star
    :param float teff: effective temperature of star
    :param feh: metallicity
    :param logg: log of surface gravity
    :param desired_magnitude: magnitude with which to normalize model spectrum, optional
    :param bool on_sky: True if on sky observation with sky emission lines
    :param fov: field of view at center wavelength in arcsec2
    :return: Phoenix model of star with given properties as SourceSpectrum object
    """
    from expecto import get_spectrum
    sp = SourceSpectrum.from_spectrum1d(get_spectrum(T_eff=teff, log_g=logg, Z=feh, cache=True))
    if desired_magnitude is not None:
        sp.normalize(desired_magnitude, band=SpectralElement.ObsBandpass('johnson,v'))
    e_sp = sp.integrate(flux_unit=units.FLAM, integration_type='analytical')
    default_distance = radius*np.sqrt(sigma_sb*teff**4*u.K**4/e_sp).decompose()
    sp /= ((distance/default_distance)**2).decompose()
    if on_sky:
        return sp + SkyEmission(fov)
    return sp


def BlackbodyModel(distance: float, radius: float, teff: float, on_sky: bool, fov=None):
    """
    :param distance: distance to star
    :param radius: radius of star
    :param float teff: effective temperature of model star
    :param bool on_sky: True if simulation ob sky observation with sky emission lines
    :param fov: field of view in arcsec^2
    :return: blackbody model of star as SourceSpectrum object
    """
    sp = SourceSpectrum(BlackBodyNorm1D, temperature=teff)
    e_sp = sp.integrate(flux_unit=units.FLAM, integration_type='analytical')
    default_distance = radius*np.sqrt(sigma_sb*teff**4*u.K**4/e_sp).decompose()
    sp /= ((distance / default_distance)**2).decompose()

    if on_sky:
        return sp + SkyEmission(fov)
    return sp


def FlatModel():
    """
    :return: model which returns the same flux density at all wavelengths
    """
    sp = SourceSpectrum(ConstFlux1D, amplitude=1e6*units.FLAM)

    # for the typical lab environment
    watt = 3*u.W  # typical lamp wattage
    dist = 40*u.cm  # approx distance from lamp to camera
    flux_w = watt/dist**2
    e_sp = sp.integrate(wavelengths=np.arange(3000, 9000, 0.1), flux_unit=units.FLAM, integration_type='analytical')
    ratio = (flux_w/e_sp).decompose()
    sp = SourceSpectrum(ConstFlux1D, amplitude=1e6 * u.photlam)
    return sp * ratio


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
    watt = 3*u.W  # typical emission lamp wattage
    dist = 40*u.cm  # approx distance from lamp to camera
    flux_w = watt/dist**2
    e_sp = sp.integrate(flux_unit=units.FLAM, integration_type='analytical')
    ratio = (flux_w/e_sp).decompose()  # attentuation factor
    return sp*ratio


def SpecFromFile(filename: str, wave_units: u.Quantity):
    """
    :param filename: Directory/filename of spectrum
    :param wave_units: units of wavelength, as u.Quantity
    :return: spectrum from file
    """
    file = np.genfromtxt(filename)
    return SourceSpectrum(Empirical1D, points=file[:, 0]*wave_units, lookup_table=file[:, 1]*u.photlam)


def get_spec(spectrum_type: str, distance=None, radius=None, teff=None, spec_file=None, minwave=None,
                 maxwave=None, on_sky=False, fov=None, wave_units=u.nm):
    """
    :param str spectrum_type: 'blackbody', 'phoenix', 'flat', 'emission', 'sky_emission', or 'from_file' only
    :param distance: distance to star
    :param radius: radius of star
    :param teff: effective temperature for blackbody or phoenix model spectrum
    :param spec_file: file name for the emission line list or spectrum from file
    :param minwave: minimum wavelength
    :param maxwave: maximum wavelength
    :param bool on_sky: True if observation is on sky and has atmospheric attenuation/sky emission/etc,
                        will always be ignored for flat and emission spectra
    :param fov: field of view in arcsec^2
    :param wave_units: wavelength units for spectrum from_file
    :return: SourceSpectrum object of chosen spectrum
    """
    if spectrum_type == 'blackbody':
        logger.info(f'Obtained blackbody model spectrum.')
        return BlackbodyModel(distance=distance, radius=radius, teff=teff, on_sky=on_sky, fov=fov)
    elif spectrum_type == 'phoenix':
        logger.info(f'Obtained Phoenix model spectrum.')
        return PhoenixModel(distance=distance, radius=radius, teff=teff, on_sky=on_sky, fov=fov)
    elif spectrum_type == 'flat':
        logger.info(f'Obtained flat-field model spectrum.')
        return FlatModel()
    elif spectrum_type == 'emission':
        logger.info(f'Obtained {spec_file} emission spectrum.')
        return EmissionModel(spec_file, minwave, maxwave)
    elif spectrum_type == 'sky_emission':
        logger.info(f'Obtained sky emission spectrum.')
        return SkyEmission(fov)
    elif spectrum_type == 'from_file':
        logger.info('Obtained spectrum from file.')
        return SpecFromFile(filename=spec_file, wave_units=wave_units)
    else:
        raise ValueError("Only 'blackbody', 'phoenix', 'flat', 'emission', 'sky_emission', "
                         "or 'from_file' are supported for spectrum_type.")


def clip_spectrum(x, clip_range):
    """
    :param x: SourceSpectrum object containing desired spectrum
    :param tuple clip_range: wavelength range to retain
    :return: SourceSpectrum with all entries outside of clip_range discarded
    """
    mask = (x.waveset >= clip_range[0]) & (x.waveset <= clip_range[-1])
    logger.info(f"Clipped spectrum to{clip_range}.")
    return SourceSpectrum.from_spectrum1d(Spectrum1D(spectral_axis=x.waveset[mask], flux=x(x.waveset[mask])))
