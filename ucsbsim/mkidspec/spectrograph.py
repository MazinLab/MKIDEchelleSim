import numpy as np
import astropy.units as u
from astropy.constants import h, c
import logging

from ucsbsim.mkidspec.detector import MKIDDetector

logger = logging.getLogger('spectrograph')


class GratingSetup:
    def __init__(self,
                 alpha: float,
                 delta: float,
                 beta_center: float,
                 groove_length: u.Quantity):
        """
        Simulation of an echelle/echellete grating for spectrometer.

        :param float alpha: incident angle in radians
        :param float delta: blaze angle in radians
        :param float beta_center: reflectance angle in radians
        :param u.Quantity groove_length: d, length/groove in units of wavelength (same as schroeder sigma)
        """
        self.alpha = alpha*u.rad
        self.delta = delta*u.rad
        self.beta_center = beta_center*u.rad
        self.d = groove_length
        self.empiric_blaze_factor = 1.0


    def __str__(self) -> str:
        """
        :return: grating parameters as formatted string
        """
        return (f"alpha={np.rad2deg(self.alpha):.2f}\n"
                f"delta={np.rad2deg(self.delta):.2f}\n"
                f"beta={np.rad2deg(self.beta_center):.2f}\n"
                f"l={self.d.to('mm'):.2f}/l ({1 / self.d.to('mm'):.2f})")


    def blaze(self, beta, m):
        """
        Blaze efficiency function follows Casini & Nelson 2014 J Opt Soc Am A eq 25 with notation modified to
        match Schroder.
        :param beta: reflectance angle in radians
        :param m: order
        :return: grating efficiency out of 1
        """
        k = np.cos(beta) * np.cos(self.alpha - self.delta) / (np.cos(self.alpha) * np.cos(beta - self.delta))
        k[k > 1] = 1
        q4 = np.cos(self.delta) - np.sin(self.delta) / np.tan((self.alpha + beta) / 2)
        rho = np.cos(self.delta) if self.alpha < self.delta else np.cos(self.alpha) / np.cos(self.alpha - self.delta)
        # 2 different rho depending on whether alpha or delta is larger
        
        logger.info(f"Calculated relative throughput (grating efficiency).")
        
        ret = k * np.sinc((m * rho * q4).value) ** 2
        return self.empiric_blaze_factor * ret


    def beta(self, wave: u.Quantity, m, energy: bool=False):
        """
        :param u.Quantity wave: wavelength(s)
        :param m: order
        :param bool energy: True to pass energy
        :return: reflectance angle in radians
        """
        if energy:
            wave = wave.to(u.nm, equivalencies=u.spectral())
        return np.arcsin(m * wave / self.d - np.sin(self.alpha))


    def wave(self, beta, m, energy: bool=False):
        """
        :param beta: reflectance angle in radians
        :param m: order
        :param bool energy: True to return energy
        :return: the wavelength of beta in that order
        """
        if energy:
            return (self.d * (np.sin(beta) + np.sin(self.alpha)) / m).to(u.eV, equivalencies=u.spectral())
        else:
            return self.d * (np.sin(beta) + np.sin(self.alpha)) / m


    def resolution(self, entrance_beam_size: u.Quantity, order):
        """
        :param u.Quantity entrance_beam_size: size of the incoming beam
        :param order: order
        :return: the limited resolution of the grating configuration
        """
        return order * entrance_beam_size / (self.d * np.cos(self.alpha))


    def resolution_eff(self, entrance_beam_size: u.Quantity, order, wave: u.Quantity, phi: float, tele_d: u.Quantity):
        """
        :param u.Quantity entrance_beam_size: size of the incoming beam
        :param order: order
        :param wave: wavelength(s) as u.Quantity
        :param phi: angular slit width (small angle approx: width/tele_f_len) in radians
        :param u.Quantity tele_d: telescope diameter
        :return: the effective resolution of the grating configuration
        """
        return self.resolution(entrance_beam_size, order) * wave / (phi * tele_d)


    def angular_dispersion(self, m, beta, energy: bool=False):
        """
        :param m: order
        :param beta: reflectance angle
        :param bool energy: return dbeta/dE
        :return: angular dispersion [rad/wave], Schroder A dbeta/dlambda (or dbeta/dE)
        """
        if energy:
            wave = self.wave(beta, m)
            return ((m / (self.d * np.cos(beta)) * u.rad)*
                    (h*c/wave.to(u.eV, equivalencies=u.spectral())**2)).decompose().to(u.rad/u.eV)
        else:
            return m / (self.d * np.cos(beta)) * u.rad


class SpectrographSetup:
    def __init__(self,
                 order_range: tuple,
                 final_wave: u.Quantity,
                 pixels_per_res_elem: float,
                 focal_length: u.Quantity,
                 grating: GratingSetup,
                 detector: MKIDDetector):
        """
        :param tuple order_range: order range of the spectrograph
        :param u.Quantity final_wave: longest wavelength at the edge of detector
        :param float pixels_per_res_elem: number of pixels per resolution element of spectrometer
        :param u.Quantity focal_length: the focal length of the detector
        :param GratingSetup grating: configured grating
        :param MKIDDetector detector: configured detector
        :return simulated spectrograph
        """
        if order_range[0] > order_range[1]:
            order_range = order_range[::-1]
        self.m0 = order_range[0]
        self.m_max = order_range[1]
        self.l0 = final_wave
        self.grating = grating
        self.detector = detector
        self.focal_length = focal_length
        self.pixel_scale = np.arctan(self.detector.pixel_size / self.focal_length)
        self.beta_central_pixel = self.grating.beta_center
        self.nord = int(self.m_max - self.m0 + 1)
        self.nominal_pixels_per_res_elem = pixels_per_res_elem
        self.nondimensional_lsf_width = 1 / self.design_res
        logger.info(f'The spectrograph has been setup with the following properties:'
                    f'\n\tl0: {self.l0}'
                    f'\n\tR0: {self.detector.design_R0}'
                    f'\n\tOrders: {self.orders}'
                    f'\n\tFocal length: {self.focal_length}'
                    f'\n\tIncidence angle: {np.rad2deg(self.grating.alpha):.3f}'
                    f'\n\tReflectance angle: {np.rad2deg(self.beta_central_pixel):.2f}'
                    f'\n\tGroove length: {self.grating.d:.2f}'
                    f'\n\t# of pixels: {self.detector.n_pixels}'
                    f'\n\tPixel size: {self.detector.pixel_size}'
                    f'\n\tPixels per res. element: {self.nominal_pixels_per_res_elem}')


    def set_beta_center(self, beta, littrow: bool = False):
        """
        :param beta: reflectance angle in degrees or u.Quantity
        :param bool littrow: True if alpha=beta
        :return: changes the reflectance angle of the central pixel (may change alpha if littrow)
        """
        if not isinstance(beta, u.Quantity):
            beta *= u.deg
        self.beta_central_pixel = beta
        if littrow:
            self.grating.alpha = beta


    @property
    def orders(self):
        """
        :return: the orders used in the spectrometer based on inputs
        """
        try:
            assert self._orders[0] == (self.m0, self.m_max)
        except (AttributeError, AssertionError):
            self._orders = (self.m0, self.m_max), np.arange(self.m0, self.m_max + 1, dtype=int)
        return self._orders[1]


    def minimum_wave(self, energy: bool=False):
        """
        :param energy: True if returning energy
        :return: the minimum wavelength of the lowest order based on FSR
        """
        if energy:
            return (self.central_wave(self.m_max) - self.fsr(self.m_max) / 2).to(u.eV, equivalencies=u.spectral())
        else:
            return self.central_wave(self.m_max) - self.fsr(self.m_max) / 2


    def info_str(self):
        """
        :return: formatted information about the spectrograph as a string
        """
        gstr = str(self.grating)
        betactr = np.rad2deg(self.grating.beta(self.central_wave(self.m0), self.m0))
        gstr += f'\nb={betactr:.2f}'
        ret = [f'    {x}' for x in gstr.split('\n')]
        ret.insert(0, 'Grating:')
        beta_ext = np.rad2deg(self.beta_for_pixel(self.detector.pixel_indices[[0, -1]] + .5))
        ret.append("    beta extent: {:.2f} - {:.2f}".format(*beta_ext))
        ret.append('Orders:')
        for o in self.orders[::-1]:
            w_c = self.central_wave(o)
            w_i = w_c - self.fsr(o) / 2
            w_f = w_c + self.fsr(o) / 2
            p_i = self.wavelength_to_pixel(w_i, o)
            p_f = self.wavelength_to_pixel(w_f, o)
            ret.append(f"    m{o:2} @ {w_c:.0f}: {w_i:.0f} - {w_f:.0f}, {p_i:.0f} - {p_f:.0f}")
        return ret


    def pixel_for_beta(self, beta):
        """
        :param beta: reflectance angle in radians
        :return: pixel at beta
        """
        delta_angle = np.tan(beta - self.beta_central_pixel)
        return self.focal_length * delta_angle / self.detector.pixel_size + self.detector.n_pixels / 2


    def beta_for_pixel(self, pixel):
        """
        :param pixel: pixel index
        :return: reflectance angle (radians) at pixel
        """
        center_offset = self.detector.pixel_size * (pixel - self.detector.n_pixels / 2)
        return self.beta_central_pixel + np.arctan(center_offset / self.focal_length)

    @property
    def max_beta_m0(self):
        """
        :return: largest reflectance angle (which is at the initial order)
        """
        return self.grating.beta(self.l0, self.m0)


    @property
    def min_beta_mmax(self):
        """
        :return: smallest reflectance angle (which is at the final order)
        """
        return self.grating.beta(self.minimum_wave(), self.m_max)


    def blaze(self, wave):
        """
        :param wave: wavelength
        :return: blaze throughput out of 1
        """
        return self.grating.blaze(self.grating.beta(wave, self.orders[:, None]), self.orders[:, None])


    def mean_blaze_eff_est(self, n=10):
        """
        :return: for debugging
        """
        edges = self.edge_wave(fsr=True)
        detector_edges = self.edge_wave(fsr=False)

        ow = np.array([np.select([detector_edges > edges, detector_edges <= edges],
                                 [detector_edges, edges])[:, 0],
                       np.select([detector_edges < edges, detector_edges > edges],
                                 [detector_edges, edges])[:, 1]]).T * u.nm
        v = self.blaze(np.array(list(map(lambda x: np.linspace(*x, num=n), ow))) * u.nm).mean(1)
        return v.value


    def order_mask(self, wave, fsr_edge: bool = False, energy: bool=False):
        """
        :param wave: wavelength(s) as u.Quantity
        :param bool fsr_edge: True to mask at the FSR, goes to detector edge if not
        :param bool energy: True to pass energies
        :return: a boolean array [norders, wave.size] where true means wavelengths are in that order
        """
        if energy:
            wave = wave.to(u.nm, equivalencies=u.spectral())

        if fsr_edge:
            o = self.orders[:, None]
            c_wave = self.pixel_to_wavelength(self.detector.n_pixels / 2, o)
            fsr = c_wave / o
            return np.abs(wave - c_wave) < fsr / 2
        else:
            x = self.wavelength_to_pixel(wave, self.orders[:, None])
            return (x >= 0) & (x < self.detector.n_pixels)


    def edge_wave(self, fsr=True):
        """
        :param fsr: True to return the FSR edges
        :return: return the wavelengths at detector edge
        """
        pix = self.detector.pixel_indices[[0, self.detector.n_pixels // 2, -1]] + .5
        fiducial_waves = self.pixel_to_wavelength(pix, self.orders[:, None])
        if not fsr:
            return fiducial_waves[:, [0, -1]]

        central_fsr = fiducial_waves[:, 1] / self.orders
        fsr_edges = (u.Quantity([-central_fsr / 2, central_fsr / 2]) + fiducial_waves[:, 1]).T

        return fsr_edges


    def central_wave(self, order, energy: bool=False):
        """
        :param order: order number
        :param bool energy: True to return energy
        :return: wavelength at the center of the order
        """
        l0_center = self.l0 / (1 + 1 / (2 * self.m0))
        
        if energy:
            return (l0_center * self.m0 / order).to(u.eV, equivalencies=u.spectral())
        else:
            return l0_center * self.m0 / order


    def fsr(self, order):
        """
        :param order: order number
        :return: the width of the free spectral range of that order
        """
        return self.central_wave(order) / order


    def wavelength_to_pixel(self, wave, m, energy: bool=False):
        """
        :param wave: wavelength(s) as u.Quantity
        :param m: order
        :param bool energy: True to pass energy
        :return: fractional pixel location of given wavelength
        """
        if energy:
            wave = wave.to(u.nm, equivalencies=u.spectral())
        return self.pixel_for_beta(self.grating.beta(wave, m))


    def pixel_to_wavelength(self, pixel, m, energy: bool=False):
        """
        :param pixel: pixel
        :param m: order
        :param bool energy: True to return energy
        :return: wavelength for given pixel as u.Quantity
        """
        if energy:
            return self.grating.wave(self.beta_for_pixel(pixel), m).to(u.eV, equivalencies=u.spectral())
        else:
            return self.grating.wave(self.beta_for_pixel(pixel), m)


    def pixel_wavelengths(self, edge: str=None, energy: bool=False):
        """
        :param str edge: left or right indicates the edges of the pixel instead of exactly at center
        :param bool energy: True to return energy
        :return: array of pixel center (or left/right edge) wavelengths for every order

        Note that wavelengths will be computed outside each order's FSR.
        """
        if edge == 'left':
            wave = self.pixel_to_wavelength(self.detector.pixel_indices, self.orders[:, None])
        elif edge == 'right':
            wave = self.pixel_to_wavelength(self.detector.pixel_indices + 1, self.orders[:, None])
        else:
            wave = self.pixel_to_wavelength(self.detector.pixel_indices + .5, self.orders[:, None])

        if energy:
            return wave.to(u.eV, equivalencies=u.spectral())
        else:
            return wave


    def dl_pix_max_wave(self, energy: bool=False):
        """
        :param bool energy: True to return dE
        :return: maximum change in wavelength/energy in any pixel
        """
        return self.pixel_scale / self.grating.angular_dispersion(self.m0, self.max_beta_m0, energy=energy)


    def dl_pix_min_wave(self, energy: bool=False):
        """
        :param bool energy: True to return dE
        :return: minimum change in wavelength in any pixel
        """
        return self.pixel_scale / self.grating.angular_dispersion(self.m_max, self.min_beta_mmax, energy=energy)


    def dl_mkid_max(self, energy: bool=False):
        """
        :param bool energy: True to return dE
        :return: largest MKID resolution width
        """
        return (self.l0*self.l0.to(u.eV, equivalencies=u.spectral())**2/
                (self.detector.R0(self.detector.pixel_indices)*h*c)).max()


    def sampling(self, oversampling, energy: bool=False):
        """
        :param oversampling: factor by which to oversample smallest wavelength extent
        :param bool energy: True to return energy
        :return: size of sampling as u.Quantity
        """
        return self.dl_pix_max_wave(energy=energy) / oversampling


    def dl_pixel(self, energy: bool=False):
        """
        :param bool energy: True to return energy
        :return: change in wavelength (energy) for every pixel
        """
        return self.pixel_scale / self.angular_dispersion(energy=energy)


    def dl_mkid_pixel(self, energy: bool=False):
        """
        :param bool energy: True to return energy
        :return: MKID resolution width for every pixel
        """
        return self.detector.mkid_resolution_width(
            self.pixel_wavelengths(energy=energy), self.detector.pixel_indices, energy=energy)


    def pixel_rescale(self, oversampling, energy: bool=False):
        """
        :param oversampling: factor by which to oversample smallest wavelength extent
        :param bool energy: True to return energy
        :return: The sample size in wavelength units for every pixel. Every MKID resolution width divided by the
                 total # of samples that are in the largest width, which is the largest width divided by the
                 smallest sample size.
        """
        return (self.dl_mkid_pixel(energy=energy) * self.sampling(oversampling, energy=energy) / 
                self.dl_mkid_max(energy=energy)).decompose().to(u.eV)


    def pixel_samples_frac(self, oversampling, energy: bool=False):
        """
        :param oversampling: factor by which to oversample smallest wavelength extent
        :param bool energy: True to return energy
        :return: number of samples for every pixel, retrieved by dividing change in wavelength for a pixel
                 by the sample size for that pixel.
        """
        return (self.dl_pixel(energy=energy) / self.pixel_rescale(oversampling, energy=energy)).si.value


    def pixel_max_npoints(self, oversampling, energy: bool=False):
        """
        :param oversampling: factor by which to oversample smallest wavelength extent
        :param bool energy: True to return energy
        :return: the maximum number of points in any given pixel, as an integer value, ensuring there is atleast one
                 sample at pixel center
        """
        pixel_max_npoints = np.ceil(self.pixel_samples_frac(oversampling, energy=energy).max()).astype(int)
        if not pixel_max_npoints % 2:  # ensure there is a point at the pixel center
            pixel_max_npoints += 1
        return pixel_max_npoints


    def sigma_mkid_pixel(self, energy: bool=False):
        """
        :param bool energy: True to return energy
        :return: the standard deviation for each pixel resolution
        """
        return self.dl_mkid_pixel(energy=energy) / 2.355


    def angular_dispersion(self, energy: bool=False):
        """
        :param bool energy: True to return energy
        :return: The angular dispersion at the center of each pixel for each order (nord, npixel)
        """
        beta = self.beta_for_pixel(self.detector.pixel_indices + .5)
        return self.grating.angular_dispersion(self.orders[:, None], beta, energy=energy)


    @property
    def design_res(self):
        """
        :return: design resolution for spectrometer, assume that m0 FSR fills detector with some sampling
        """
        dlambda = self.fsr(self.m0) / self.detector.n_pixels * self.nominal_pixels_per_res_elem
        return self.l0 / dlambda


    @property
    def average_res(self):
        """
        :return: actual average resolution for spectrometer
        """
        w = self.edge_wave(fsr=False)
        return (w.mean(1) / (np.diff(w, axis=1).T / self.detector.n_pixels * self.nominal_pixels_per_res_elem)).ravel()


    def plot_echellogram(self, center_orders: bool=True, title: str='', blaze: bool=False):
        """
        :param bool center_orders: True to center the orders
        :param str title: title of plot
        :param bool blaze: True to plot blaze
        :return: debugging plots
        """
        import matplotlib.pyplot as plt
        w = self.pixel_wavelengths()
        b = self.blaze(w)

        fig, axes = plt.subplots(2 + int(blaze), 1, figsize=(6, 10 + 4 * int(blaze)))
        if title:
            plt.suptitle(title)
        plt.sca(axes[0])
        plt.title(f'a={self.beta_central_pixel:.1f} m={self.m0}-{self.m_max}')
        fsr_edges = self.edge_wave(fsr=True)
        for ii, i in enumerate(self.orders):
            waves = w[ii, [0, self.detector.n_pixels // 2, -1]]
            plt.plot(self.wavelength_to_pixel(waves, i), [i] * 3, '*', color=f'C{ii}')
            plt.plot(self.wavelength_to_pixel(fsr_edges[ii], i), [i] * 2, '.', color=f'C{ii}')
        plt.xlabel('Pixel')
        plt.ylabel('Order')
        plt.sca(axes[1])
        for ii, i in enumerate(self.orders):
            waves = w[ii, [0, self.detector.n_pixels // 2, -1]]
            oset = waves[1] if center_orders else 0
            plt.plot(waves - oset, [i] * 3, '*', color=f'C{ii}')
            plt.plot(fsr_edges[ii] - oset, [i] * 2, '.', color=f'C{ii}',
                     label=f'$\lambda=${waves[1]:.0f}')
        plt.legend()
        plt.xlabel('Center relative wavelength (nm)')
        plt.ylabel('Order')
        if blaze:
            plt.sca(axes[2])
            plt.plot(w.T, b.T)
            plt.xlabel('Wavelength (nm)')
            plt.ylabel('Blaze')
        plt.tight_layout()
        plt.show()


# misc. debugging variables below
GRATING_CATALOG = np.loadtxt('/home/kimc/pycharm/KIDSpecSim/benchdesign/newport_masters.txt', delimiter=',',
                             dtype=[('name', 'U10'), ('l', 'f4'), ('blaze', 'f4'),
                                    ('width', 'f4'), ('height', 'f4'), ('stock', 'U10')])
GRATING_CATALOG['l'] = 1e6/GRATING_CATALOG['l']

NEWPORT_GRATINGS = {x['name']: GratingSetup(
    0,
    (x['blaze']*u.deg).to(u.rad).value,
    0,
    x['l'] * u.nm) for x in GRATING_CATALOG}
