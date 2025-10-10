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
import momospecsim.engine as engine

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
        self.target = target
        self.telescope = telescope
        self.telefiber = telefiber
        self.fiberarray = fiberarray
        self.spectrograph = spectrograph
        self.detector = detector
        logger.info('SpecSimSettings initialized.')

    def __eq__(self, other):
        if self.__class__ != other.__class__:
            return False
        else:
            return self.__dict__ == other.__dict__

    @property
    def engine(self):
        """
        :return: Engine class based on simulation settings.
        """
        return engine.Engine(spectrograph=self.spectrograph)
