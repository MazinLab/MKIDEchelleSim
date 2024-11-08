import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
from astropy.constants import R_sun

from ucsbsim.mkidspec.spectrograph import GratingSetup, SpectrographSetup
from ucsbsim.mkidspec.detector import MKIDDetector
import ucsbsim.mkidspec.engine as engine

class SpecSimSettings:
    def __init__(
            self,
            minwave_nm: float,
            maxwave_nm: float,
            npix: int,
            pixelsize_um: float,
            designR0: float,
            l0_nm: float,
            alpha_deg: float,
            delta_deg: float,
            beta_deg: float,
            groove_length_nm: float,
            m0: int,
            m_max: int,
            pixels_per_res_elem: float,
            focallength_mm: float,
            R0s_file: str = None,
            phaseoffset_file: str = None,
            resid_file: str = None,
            type_spectra: str = None,
            spec_file: str = None,
            exptime_s: float = None,
            telearea_cm2: float = None,
            distance_ps: float = None,
            radius_Rsun: float = None,
            temp_K: float = None,
            on_sky: bool = None,
            simpconvol: bool = None
    ):
        """
        :param float minwave_nm: The minimum wavelength of the spectrometer in nm.
        :param float maxwave_nm: The maximum wavelength of the spectrometer in nm.
        :param int npix: The number of pixels in the MKID detector.
        :param float pixelsize_um: The length of the MKID pixel in the dispersion direction in um.
        :param float designR0: The expected resolution at l0.
        :param float l0_nm: The longest wavelength of any order in use in nm.
        :param float alpha_deg: The incidence angle of light on the grating in degrees.
        :param float delta_deg: The grating blaze angle in degrees.
        :param float beta_deg: The reflectance angle at the central pixel in degrees.
        :param float groove_length_nm: The distance between slits of the grating in nm.
        :param int m0: The initial order, at the longer wavelength end.
        :param int m_max: The final order, at the shorter wavelength end.
        :param float pixels_per_res_elem: Number of pixels per spectral resolution element for the spectrograph.
        :param float focallength_mm: The focal length of the detector in mm.
        :param str R0s_file: Directory/filename of the R0s file.
        :param str phaseoffset_file: Directory/filename of the pixel phase center offsets file.
        :param str resid_file: Directory/filename of the resonator IDs file.
        :param str type_spectra: The type of spectrum to be simulated.
        :param str spec_file: Directory/filename of the spectrum file.
        :param float exptime_s: The exposure time of the observation in seconds.
        :param float telearea_cm2: The telescope area of the observation in cm2.
        :param distance_ps: The distance to target in parsecs.
        :param radius_Rsun: The radius of the target in units of R_sun.
        :param float temp_K: The temperature of the spectrum in K.
        :param on_sky: True if the observation is simulated on sky (atmosphere, sky emission, etc.).
        :param bool simpconvol: True if conducting a simplified convolution with MKIDs.
        """
        self.minwave = float(minwave_nm) * u.nm if not isinstance(minwave_nm, u.Quantity) else minwave_nm
        self.maxwave = float(maxwave_nm) * u.nm if not isinstance(maxwave_nm, u.Quantity) else maxwave_nm
        self.npix = int(npix)
        self.pixelsize = float(pixelsize_um) * u.micron if not isinstance(pixelsize_um, u.Quantity) else pixelsize_um
        self.designR0 = float(designR0)
        if l0_nm == 'same':
            self.l0 = self.maxwave
        else:
            self.l0 = float(l0_nm)*u.nm if not isinstance(l0_nm, u.Quantity) else l0_nm
        self.alpha = np.deg2rad(float(alpha_deg))
        self.delta = np.deg2rad(float(delta_deg))
        if beta_deg == 'littrow':
            self.beta = self.alpha
        else:
            self.beta = np.deg2rad(float(beta_deg))
        self.groove_length = float(groove_length_nm)*u.nm if not isinstance(groove_length_nm,
                                                                            u.Quantity) else groove_length_nm
        self.order_range = (int(m0), int(m_max))
        self.pixels_per_res_elem = float(pixels_per_res_elem)
        self.focallength = float(focallength_mm)*u.mm if not isinstance(focallength_mm, u.Quantity) else focallength_mm
        self.R0s_file = R0s_file
        self.phaseoffset_file = phaseoffset_file
        self.resid_file = resid_file
        self.type_spectra = type_spectra
        self.spec_file = spec_file
        if exptime_s is not None:
            self.exptime = float(exptime_s)*u.s if not isinstance(exptime_s, u.Quantity) else exptime_s
        if telearea_cm2 is not None:
            self.telearea = float(telearea_cm2)*u.cm**2 if not isinstance(telearea_cm2, u.Quantity) else telearea_cm2
        if distance_ps is not None:
            self.distance = float(distance_ps) * u.parsec if not isinstance(distance_ps, u.Quantity) else distance_ps
        if radius_Rsun is not None:
            self.radius = float(radius_Rsun) * R_sun
        if temp_K is not None:
            self.temp = float(temp_K)
        self.on_sky = on_sky
        self.simpconvol = simpconvol

    def __eq__(self, other):
        if self.__class__ != other.__class__:
            return False
        else:
            return self.__dict__ == other.__dict__
    
    @property
    def detector(self):
        return MKIDDetector(n_pix=self.npix, pixel_size=self.pixelsize, design_R0=self.designR0, l0=self.l0)

    @property
    def grating(self):
        return GratingSetup(alpha=self.alpha, delta=self.delta, beta_center=self.beta, groove_length=self.groove_length)

    @property
    def spectrograph(self):
        return SpectrographSetup(order_range=self.order_range, final_wave=self.l0,
                                pixels_per_res_elem=self.pixels_per_res_elem,
                                focal_length=self.focallength, grating=self.grating, detector=self.detector)
    
    @property
    def engine(self):
        return engine.Engine(spectrograph=self.spectrograph)