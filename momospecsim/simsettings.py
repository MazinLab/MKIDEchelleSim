# global imports
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
from astropy.constants import R_sun
import logging

# local imports
from momospecsim.spectra import Target
from momospecsim.optics import Fiber, Telescope, Grating, Spectrograph
from momospecsim.detector import MKIDDetector
from momospecsim.engine import Engine

logger = logging.getLogger('specsimsettings')


class SpecSimSettings:
    def __init__(
            self,
            outdir: str = '',
            simpconvol: bool = False,
            waveconvol: bool = False,
            target: Target = None,
            telescope: Telescope = None,
            telefiber: Fiber = None,
            fiberarray: Fiber = None,
            spectrograph: Spectrograph = None,
            detector: MKIDDetector = None
    ):
        """
        :param outdir: output directory
        :param simpconvol: if passed, conducts simplified convolution
        :param waveconvol: if passed, conducts convolution w.r.t. wavelength not energy
        :param target: Target object
        :param telescope: Telescope object
        :param telefiber: telescope Fiber object
        :param fiberarray: Fiber array object
        :param spectrograph: Spectrograph object
        :param detector: MKIDDetector object
        """

        self.outdir = outdir
        self.simpconvol = simpconvol
        self.waveconvol = waveconvol
        if target is not None:
            self.spectype = target.spectype
            self.dist = target.dist
            self.rad = target.rad
            self.temp = target.temp
            self.spec_file = target.spec_file
            self.minwave = target.minwave.value if target.minwave is not None else target.minwave
            self.maxwave = target.maxwave.value if target.maxwave is not None else target.maxwave
            self.objsize = target.objsize
            self.seeing = target.seeing
            self.on_sky = target.on_sky
        if telescope is not None:
            self.aperture = telescope.aperture
            self.telefocal = telescope.focal_length if telescope.focal_length is None else telescope.focal_length.value
            self.telename = telescope.filename
        if telefiber is not None:
            self.telefibername = telefiber.filename
            self.telefiberNA = telefiber.num_aperture
            self.telefiberlength = telefiber.length
            self.telefibercore = telefiber.core_size
            self.telefiberangle = telefiber.incident_angle
        if fiberarray is not None:
            self.fiberarrayname = fiberarray.filename
            self.fiberarrayNA = fiberarray.num_aperture
            self.fiberarraylength = fiberarray.length
            self.fiberarraycore = fiberarray.core_size
        if spectrograph is not None:
            self.m0 = spectrograph.m0
            self.m_max = spectrograph.m_max
            self.l0 = spectrograph.l0.value
            self.pixels_per_res_elem = spectrograph.nominal_pixels_per_res_elem
            self.focal_length = spectrograph.focal_length.value
            self.alpha = np.rad2deg(spectrograph.grating.alpha.value)
            self.delta = np.rad2deg(spectrograph.grating.delta.value)
            self.beta_center = np.rad2deg(spectrograph.grating.beta_center.value)
            self.groove_length = spectrograph.grating.d.value
            self.npix = spectrograph.detector.npix
            self.pix_size = spectrograph.detector.pix_size.value
            self.R0 = spectrograph.detector.design_R0
            self.randomseed = spectrograph.detector.randomseed
            self.resid_file = spectrograph.detector.resid_file
        logger.info('SpecSimSettings initialized.')

    def __eq__(self, other):
        if self.__class__ != other.__class__:
            return False
        else:
            return self.__dict__ == other.__dict__

    @property
    def target(self):
        return Target(self.spectype, self.dist, self.rad, self.temp, self.spec_file, self.minwave, self.maxwave,
                      self.objsize, self.seeing, self.on_sky)

    @property
    def telescope(self):
        return Telescope(self.aperture, self.telefocal, self.telename)
    
    @property
    def telefiber(self):
        return Fiber(self.telefibername, self.telefiberNA, self.telefiberlength, self.telefibercore,
                     self.telefiberangle)
    
    @property
    def fiberarray(self):
        return Fiber(self.fiberarrayname, self.fiberarrayNA, self.fiberarraylength, self.fiberarraycore)

    @property
    def grating(self):
        return Grating(self.alpha, self.delta, self.beta_center, self.groove_length)
    
    @property
    def detector(self):
        return MKIDDetector(self.npix, self.pix_size, self.R0, self.l0, self.randomseed, self.resid_file)

    @property
    def spectrograph(self):
        return Spectrograph(self.m0, self.m_max, self.l0, self.pixels_per_res_elem, self.focal_length, self.grating,
                            self.detector)

    @property
    def engine(self):
        return Engine(self.spectrograph)
