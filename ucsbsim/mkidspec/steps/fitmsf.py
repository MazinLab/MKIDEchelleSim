# global imports
import copy
from copy import deepcopy
import numpy as np
import warnings
from numpy.polynomial.legendre import Legendre
import scipy.interpolate as interp
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from mpl_point_clicker import clicker
from mpl_toolkits.axes_grid1 import make_axes_locatable
import tqdm
from sklearn.cluster import k_means
import astropy.units as u
import time
import argparse
import logging
import os
from lmfit import Parameters, minimize
from mkidpipeline.photontable import Photontable

# local imports
import ucsbsim.mkidspec.engine as engine
from ucsbsim.mkidspec.msf import MKIDSpreadFunction
from ucsbsim.mkidspec.detector import wave_to_phase, phase_to_wave, sorted_table
import ucsbsim.mkidspec.utils.general as gen

"""
Obtain the MKID Spread Function (MSF) from a calibration (flat-field or known-temperature blackbody) spectrum.
The steps are:
-Load the calibration photon table.
-Fit model function to each pixel.
-Use Gaussian function intersections to obtain virtual pixel bin edges for each order.
-Calculate fractional bleed between orders and converts them into an n_ord x n_ord "covariance" matrix for each
 pixel. This matrix how much of each order's flux was potentially grouped into another order.
 This will later be used to determine the error on the follow-on extracted spectra.
-Saves newly obtained bin edges and covariance matrices to files.
"""

logger = logging.getLogger('fitmsf')


def bin_width_from_MKID_R(R):
    """
    :param R: the MKID resolution
    :return: the necessary bin-width for an average of 10 points across 6 sigma
    """
    dlam_FWHM = 0.5/R  # find the width in phase space
    dlam_sig = dlam_FWHM/2.355  # convert to standev
    return 6*dlam_sig / 10  # want an average of 10 points across a 6 sigma span of the Gaussian


def init_params(phi_guess, e_guess, s_guess, a_guess, e_domain=[-1, 0], w_constr: bool = False):
    """
    :param phi_guess: n_ord guess values for phase centers
    :param e_guess: n_ord guess values for energies of given phases
    :param s_guess: n_ord guess values for sigmas of given energies
    :param a_guess: n_ord guess values for amplitudes of given phases
    :param e_domain: domain for the legendre polynomial
    :param w_constr: True to constrain parameters
    :return: an lmfit Parameter object with the populated parameters and guess values
    """
    parameters = Parameters()

    # use Legendre polyfitting on the theoretical/peak-finding data to get guess coefs
    e_coefs = Legendre.fit(x=phi_guess, y=e_guess / e_guess[-1], deg=2, domain=e_domain).coef
    s_coefs = Legendre.fit(x=e_guess / e_guess[-1], y=s_guess, deg=2, domain=[e_guess[0] / e_guess[-1], 1]).coef
    # the domain of the sigmas is scaled such that the order_0 energy is = to 1

    if w_constr:
        # add energy coefs to params object:
        parameters.add(name=f'e1', value=e_coefs[1], max=0)  # must be negative
        parameters.add(name=f'e2', value=e_coefs[2], min=-0.1, max=0.1)

        # add the sigma coefs to params object:
        parameters.add(name=f's0', value=s_coefs[0], min=0, max=0.1)  # must be positive
        parameters.add(name=f's1', value=s_coefs[1])
        parameters.add(name=f's2', value=s_coefs[2], min=-1e-2, max=1e-2)

        # add phi_0s to params object:
        parameters.add(name=f'phi_0', value=phi_guess[-1], min=phi_guess[-1] - 0.2, max=phi_guess[-1] + 0.2)

        # add amplitudes to params object:
        for n, a in enumerate(a_guess):
            parameters.add(name=f'O{n}_amp', value=a, min=0, max=np.max(a_guess)*10)
    else:
        # add energy coefs to params object:
        parameters.add(name=f'e1', value=e_coefs[1])
        parameters.add(name=f'e2', value=e_coefs[2])
    
        # add the sigma coefs to params object:
        parameters.add(name=f's0', value=s_coefs[0])
        parameters.add(name=f's1', value=s_coefs[1])
        parameters.add(name=f's2', value=s_coefs[2])
    
        # add phi_0s to params object:
        parameters.add(name=f'phi_0', value=phi_guess[-1])
    
        # add amplitudes to params object:
        for n, a in enumerate(a_guess):
            parameters.add(name=f'O{n}_amp', value=a)

    return parameters


def fit_func(params: Parameters, x_phases, y_counts=None, orders=None, leg_e=None, leg_s=None, plot: bool = False,
             to_sum: bool = False):
    """
    :param Parameters params: Parameters object containing all params with initial guesses
    :param x_phases: the grid of phases to be sampled along
    :param y_counts: the histogram data details photon count in each bin, must be same shape as x_phases
    :param orders: a list of the orders incident on the detector, in ascending order
    :param leg_e: the Legendre poly object with the proper phase domain
    :param leg_s: the Legendre poly object with the proper energy domain
    :param bool plot: whether to show the plots of the fit
    :param bool to_sum: whether to sum the fitting function or leave the individual Gaussians
    :return: residuals, fitting function separated, or fitting function summed
    """

    # turn dictionary of param values into separate variables
    e1, e2, s0, s1, s2, phi_0, *amps = tuple(params.valuesdict().values())

    # obtain the 0th order energy coef
    e0 = e0_from_params(e1, e2, phi_0)

    # pass coef parameters to polys:
    setattr(leg_e, 'coef', (e0, e1, e2))
    setattr(leg_s, 'coef', (s0, s1, s2))

    try:
        # calculate the other phi_m based on phi_0:
        phis = phis_from_grating_eq(orders, phi_0, leg=leg_e, coefs=[e0, e1, e2])

        # get sigmas at each phase center:
        sigs = leg_s(leg_e(phis))

        # put the phi, sigma, and fit_amps together for Gaussian model:
        gauss_i = np.nan_to_num(gen.gauss(x_phases, phis[:, None], sigs[:, None], np.array(amps)[:, None]))
        gauss_i[gauss_i < 0] = 0
        model = np.sum(gauss_i, axis=1).flatten()

        if y_counts is not None:
            if np.iscomplex(phis).any() or not np.isfinite(phis).any():  # return super high residual for such cases
                residual = np.full(y_counts.shape, np.max(y_counts)/np.sqrt(np.max(y_counts)))

            else:
                # get the residuals and weighted reduced chi^2:
                model_1 = deepcopy(model)
                model_1[model_1 < 1] = 1
                residual = np.divide(y_counts-model, np.sqrt(model_1))

    except IndexError:
        if y_counts is not None:
            residual = np.full(y_counts.shape, np.max(y_counts)/np.sqrt(np.max(y_counts)))
        else:
            if to_sum:
                model = np.full(x_phases.shape, 0)

    if plot:  # for debugging only
        logger.warning(msg='The fitter will plot every single fitting iteration. You have been warned.')
        N_dof = len(residual) - len(params)
        red_chi2 = np.sum(residual ** 2) / N_dof
        fig = plt.figure(1)
        ax = fig.add_axes((.1, .3, .8, .6))
        ax.grid()
        ax.plot(x_phases, y_counts, label='Data')
        for n, i in enumerate(gauss_i.T):
            ax.plot(x_phases, i, label=f'O{7 - n}')
        ax.legend()
        ax.set_ylabel('Count')
        ax.set_xlim([-1.5, 0])
        ax.set_xticklabels([])
        ax.legend()
        res = fig.add_axes((.1, .1, .8, .2))
        res.grid()
        res.plot(x_phases, np.nan_to_num(residual), '.', color='purple')
        res.set_ylabel('Residual')
        res.set_xlabel('Phase')
        res.set_xlim([-1.5, 0])
        res.text(-0.3, 10, f'Red. Chi^2={red_chi2:.1f}')
        plt.suptitle("lmfit Fitting Iteration")
        plt.show()

    if y_counts is None:  # returns model functions if only x data is passed
        if to_sum:  # returns all gaussians summed together
            return model
        else:  # returns individual order gaussians separated
            return gauss_i
    else:  # returns residuals if x and y data are passed
        return np.nan_to_num(residual)


def extract_params(params: Parameters, nord: int, degree: int = 2):
    """
    :param Parameters params: Parameters object
    :param int nord: number of orders
    :param int degree: the polynomial degree
    :return: the extracted parameters
    """
    phi_0 = params['phi_0'].value
    e_coefs = np.array([params[f'e{c}'].value for c in range(1, degree + 1)])  # no e0
    s_coefs = np.array([params[f's{c}'].value for c in range(degree + 1)])
    amps = np.array([params[f'O{i}_amp'].value for i in range(nord)])

    return phi_0, e_coefs, s_coefs, amps


def e0_from_params(e1: float, e2: float, phi_0: float):
    """
    :param float e1: the 1st order coef
    :param float e2: the 2nd order coef
    :param float phi_0: the initial order phase center
    :return: the Legendre poly solved for the 0th order coef given dimensionless energy
    """
    return 1 - e2 * (1 / 2 * (3 * ((phi_0 + 0.5) * 2) ** 2 - 1)) - e1 * 2 * (phi_0 + 0.5)


def phis_from_grating_eq(orders, phi_0: float, leg: Legendre, coefs=None):
    """
    :param orders: orders of the spectrograph in ascending numbers
    :param float phi_0: the phase center of the initial order
    :param Legendre leg: the energy legendre poly object
    :param coefs: the coefs of the energy poly in ascending degree
    :return: the phase centers of the other orders constrained by the grating equation
    """
    grating_eq = orders[1:][::-1] / orders[0] * leg(phi_0)

    phis = []
    for i in grating_eq:
        roots = np.roots([6 * coefs[2], 6 * coefs[2] + 2 * coefs[1], coefs[2] + coefs[0] + coefs[1] - i])

        if len(roots) == 1:  # when there is only 1 root, use it
            phis.append(roots[0])
        elif ~np.isfinite(roots).all():  # if both roots are invalid, raise error
            raise ValueError("All calculated roots are not valid, check equation.")
        else:  # if there are 2 valid roots, use the one in the proper range
            phis.append(roots[(-1.5 < roots) & (roots < phi_0)][0])

    return np.append(np.array(phis), phi_0)


def cov_from_params(params: Parameters, model, nord, order_edges, x_phases):
    """
    :param Parameters params: Parameters object
    :param model: the model function split up into orders
    :param nord: the number of orders
    :param order_edges: the indices of the virtual pixel edges
    :param x_phases: phase grid to be used
    :return: the covariance between orders as nord x nord array
    
    this function works by suppressing each order successively and then calculate the percentage 'hit' in counts
    that the other orders take as a result
    
    # v giving order, > receiving order [g_idx, r_idx, pixel]
    #      9   8   7   6   5
    # 9 [  1   #   #   #   #  ]  < how much of order 9 is in every other other as a fraction of order 9
    # 8 [  #   1   #   #   #  ]
    # 7 [  #   #   1   #   #  ]
    # 6 [  #   #   #   1   #  ]
    # 5 [  #   #   #   #   1  ]
    #      ^ how much of other orders is in order 9, every column must be a fraction of every orders
    """
    model_int = [interp.InterpolatedUnivariateSpline(x=x_phases, y=model[:, m], k=1, ext='zeros') for m in range(nord)]
    model_sum = np.array([m.integral(order_edges[0], order_edges[-1]) for m in model_int])
    cov = np.zeros([nord, nord])
    for o in range(nord):
        cov[o, :] = np.array([np.nan_to_num(model_int[o].integral(order_edges[m], order_edges[m + 1]) / model_sum[o],
                                            posinf=0, neginf=0) for m in range(nord)])
        cov[o, model_sum == 0] = 0
        if np.array_equal(cov[o, :], np.zeros([nord])):
            cov[o, o] = 1
    cov[cov < 0] = 0
    return cov


def fitmsf(msf_table: Photontable,
           sim,
           outdir: str,
           resid_map,
           bin_range: tuple = (-1.5, 0),
           missing_order_pix=None,
           plot: bool = False,
           debug: bool = False):
    """
    :param Photontable msf_table: photon table object for the MSF-specific spectrum
    :param sim: simulation settings object
    :param str outdir: save directory
    :param resid_map: resonator id list
    :param tuple bin_range: range of phases for binning table
    :param missing_order_pix: list of pixel ranges and orders that may be missing between them
    :param bool plot: True to plot final plots and all failed plots
    :param bool debug: True to plot all plots regardless
    :return: MSF object
    """
    
    # extract resid map from file if needed:
    resid_map = np.loadtxt(fname=resid_map, delimiter=',') if isinstance(resid_map, str) else resid_map

    photons_pixel = sorted_table(table=msf_table, resid_map=resid_map)  # get list of photons in each pixel
    photons_pixel = [np.array(l)[np.logical_and(np.array(l) > -1.5, np.array(l) < 0)] for l in photons_pixel]

    # retrieve the detector, spectrograph, and engine:
    eng = sim.engine
    spectro = eng.spectrograph
    detector = spectro.detector

    # shortening some longer variable names:
    nord = spectro.nord
    pixels = detector.pixel_indices
    pix_waves = spectro.pixel_wavelengths().to(u.nm)[::-1]  # flip order axis to be in ascending phase/lambda
    pix_E = gen.wave_to_energy(pix_waves).value  # convert to energy

    # pre-bin each pixel with the same bin edges and get centers for plotting:
    bin_width = bin_width_from_MKID_R(detector.design_R0)
    bin_edges = np.arange(bin_range[0], bin_range[1], bin_width)
    bin_centers = bin_edges[:-1] + np.diff(bin_edges) / 2
    bin_counts = np.zeros([len(bin_centers), sim.npix])
    for p in pixels:
        bin_counts[:, p], _ = np.histogram(photons_pixel[p], bins=bin_edges)
    
    # define phase grid for plotting/integrating more accurately
    fine_phase_grid = np.linspace(bin_range[0], bin_range[1], 1000)
 
    # create arrays to place loop values:
    red_chi2 = np.full(sim.npix, fill_value=1e6)  # reduced chi square
    all_fit_phi = np.empty([nord, sim.npix])  # fitting result gaussian means
    all_fit_sig = np.empty([nord, sim.npix])  # fitting result gaussian sigma
    gausses = np.zeros([1000, sim.npix])  # the entire n_ord Gaussian model summed
    gausses_i = np.zeros([1000, nord, sim.npix])  # model separated by orders
    covariance = np.zeros([nord, nord, sim.npix])  # order overlap covariance matrix
    p_err = np.zeros([nord, sim.npix])  # positive error on counts
    m_err = np.zeros([nord, sim.npix])  # negative error on counts
    order_edges = np.zeros([nord + 1, sim.npix])  # virtual pixel bin edges
    order_edges[0, :] = -2  # make the leftmost bin sufficient for our purposes
    ord_counts = np.zeros([nord, sim.npix])  # the counts for the MSF specific spectrum by pixel and order
    true_count = np.zeros([nord, sim.npix])  # storing the MSF spectrum 'true' counts, will not be passed on later
    
    leg_e = Legendre(coef=(0, 0, 0), domain=[-1,0])  # setup the energy Legendre object

    warnings.filterwarnings('ignore', category=RuntimeWarning)  # suppresses warning that occurs each fit

    mo = []
    for m in missing_order_pix:  # determine maximum # of missing orders
        mo.append(len(m[1]))
    mo = max(mo)

    for p in tqdm.tqdm(pixels):  # do the non-linear least squares fit for each pixel
        leg_s = Legendre(coef=(0, 0, 0), domain=[pix_E[0, p] / pix_E[-1, p], 1])  # setup the special sigma Legendre

        n_redchis, n_params = [], []  # temporary storage for holding possible cases
        for n_use in range(nord-mo, nord+1):  # looping through possible n_ords based on argument
            # find cluster centers given a number of clusters to find
            center, labels, _ = k_means(photons_pixel[p].reshape(-1, 1), n_use, random_state=0)
            phi_init = np.sort(center.flatten()) # sort as clusters are not always in ascending order

            # find cluster standard devations, average to reduce outliers (sigmas cannot deviate too much)
            clusters = [photons_pixel[p][np.argwhere(labels == i).flatten()] for i in range(n_use)]
            sig_init = np.average([np.std(clusters[i]) for i in range(n_use)])

            if n_use != nord:  # in cases where the current # of orders is less than the total expected # of orders
                for m in missing_order_pix:  # parsing the argument stating which orders may be missing where
                    mo_range, mos = m[0], m[1]  # pixel range, orders that may be missing
                    if mo_range[0] <= p <= mo_range[1] and n_use == nord-len(mos):  # if we are in the right category
                        # use polyfit to insert the missing cluster centers into the list found earlier
                        p_poly = np.polynomial.polynomial.Polynomial.fit(np.delete(range(nord), mos), phi_init, 1)
                        phi_init = p_poly(range(nord))

            # find the amplitudes (counts) of the data at those cluster centers:
            amp_init = bin_counts[[gen.nearest_idx(bin_centers, phi) for phi in phi_init], p]
                
            # populate Parameters object with initial guesses:
            param = init_params(phi_guess=phi_init, e_guess=pix_E[:, p], s_guess=[sig_init]*5, a_guess=amp_init)
            n_params.append(param)

            # get the initial guess fitting metric:
            pre_residual = fit_func(param, bin_centers, y_counts=bin_counts[:, p], orders=spectro.orders, leg_e=leg_e,
                                    leg_s=leg_s)
            N_dof = len(pre_residual) - len(param)
            n_redchis.append(np.sum(pre_residual**2) / N_dof)  # calculate the reduced chi2

        params = n_params[np.argmin(n_redchis)]  # chose the set of parameters with the lowest initial redchi2
        opt_params = minimize(fcn=fit_func,  # do nl least squares fitting, return optimized parameter set
                              params=params,
                              args=(bin_centers,  # x_phases
                                    bin_counts[:, p],  # y_counts
                                    spectro.orders,  # orders
                                    leg_e,  # energy legendre poly object
                                    leg_s))  # sigma legendre poly object

        if not opt_params.success:  # if unsuccessful, try fitting again with constraints
            c_params = init_params(phi_guess=phi_init, e_guess=pix_E[:, p], s_guess=sig_init, a_guess=amp_init,
                                   w_constr=True)
            c_opt_params = minimize(fcn=fit_func,
                                    params=c_params,  # params
                                    args=(bin_centers,  # x_phases
                                          bin_counts[:, p],  # y_counts
                                          spectro.orders,  # orders
                                          leg_e,  # energy legendre poly object
                                          leg_s))  # sigma legendre poly object

            if c_opt_params.redchi < opt_params.redchi:  # choose the best set of parameters based on redchi2
                opt_params = c_opt_params
                params = c_params

        plot_int = False
        if not opt_params.success:  # log which pixels failed to fit
            logger.warning(f'\nPixel {p} failed to converge/fit.')
            plot_int = True  # overrides plot argument to show any failed fits
        red_chi2[p] = opt_params.redchi  # save redchi2 to global
        
        # extract the fitting parameters:
        fit_phi0, fit_e_coef, fit_s_coef, fit_amps = extract_params(params=opt_params.params, nord=nord)
        fit_amps[fit_amps < 1] = 1  # prevents error when finding gaussian intersections
        fit_e0 = e0_from_params(e1=fit_e_coef[0], e2=fit_e_coef[1], phi_0=fit_phi0)  # get 0th E coef from other params
        setattr(leg_e, 'coef', [fit_e0, fit_e_coef[0], fit_e_coef[1]])  # regenerate the energy legendre poly
        setattr(leg_s, 'coef', fit_s_coef)  # regenerate the sigma legendre poly
        fit_phis = phis_from_grating_eq(orders=spectro.orders, phi_0=fit_phi0, leg=leg_e,
                                        coefs=[fit_e0, fit_e_coef[0], fit_e_coef[1]])  # get other gaussian means
        fit_sigs = leg_s(leg_e(fit_phis))  # get all gaussian sigmas
        all_fit_phi[:, p], all_fit_sig[:, p] = fit_phis, fit_sigs  # save results to global to be saved in MSF

        # store models to array:
        gausses_i[:, :, p] = fit_func(params=opt_params.params, x_phases=fine_phase_grid, orders=spectro.orders,
                                      leg_e=leg_e, leg_s=leg_s)  # the individual gaussian models
        gausses[:, p] = np.sum(gausses_i[:, :, p], axis=1)  # all gaussians collapsed into one model

        for i in range(nord - 1):
            try:
                order_edges[i + 1, p] = gen.gauss_intersect(fit_phis[[i, i + 1]],
                                                            fit_sigs[[i, i + 1]],
                                                            fit_amps[[i, i + 1]])  # find the virtual pixel boundaries
            except ValueError:
                if i == 0:  # if the 1st order, makes the 1-to-2 border into 3 sigmas away from order 2
                    order_edges[i + 1, p] = fit_phis[i+1] - fit_sigs[i+1]*3
                elif i == nord - 1:  # if the last order, makes the 2ndtolast-to-last border 3 sigs from 2ndtolast
                    order_edges[i + 1, p] = fit_phis[i - 1] + fit_sigs[i - 1] * 3
                else:  # if the intersection cant be found, manually click the location of the order indicated
                    click_edge = None
                    fig = plt.figure()
                    ax = fig.add_subplot(111)
                    ax.grid()
                    ax.bar(bin_centers, bin_counts[:, p], width=bin_centers[1] - bin_centers[0], linewidth=0, color='k',
                            label='Data')
                    ax.plot(fine_phase_grid, gausses[:, p], label=f'Gaussian Fit')
                    ax.set_title(f'CLICK THE BOUNDARY BETWEEN ORDER {spectro.orders[::-1][i]-1} AND '
                                 f'{spectro.orders[::-1][i]}\n then exit the plot')
                    ax.set_xlabel(r'Phase $\times 2\pi$')
                    ax.set_ylabel('Photon Count')
                    klicker = clicker(ax, ["event"])
                    plt.show()
                    order_edges[i + 1, p] = klicker.get_positions()['event'][0, 0]
                    continue

        try:
            # re-histogram the photon table using the virtual pixel edges:
            ord_counts[:, p], _ = np.histogram(photons_pixel[p], bins=order_edges[:, p])
            
            # find order-bleeding covariance:
            covariance[:, :, p] = cov_from_params(params=opt_params.params, model=gausses_i[:, :, p], nord=nord,
                                                  order_edges=order_edges[:, p], x_phases=fine_phase_grid)

            cov_inv = np.linalg.inv(covariance[:, :, p])  # take the inverse
            true_count[:, p] = np.dot(ord_counts[:, p], cov_inv)  # matrix math to retrieve 'true' counts
            
            # obtain the count error on the MSF-specific spectrum:
            p_err[:, p] = [
                np.sum(true_count[:, p] * covariance[:, m, p]) - true_count[m, p] * covariance[m, m, p] for m in range(nord)]
            m_err[:, p] = [
                np.sum(true_count[:, p] * covariance[m, :, p]) - true_count[m, p] * covariance[m, m, p] for m in range(nord)]
            if np.abs(np.sum(true_count[:, p]) - np.sum(ord_counts[:, p])) > 1:
               logger.warning(f'Pixel {p} total calculated and actual counts are '
                              f'{np.abs(np.sum(true_count[:, p]) - np.sum(ord_counts[:, p])):.0f} photons apart.')

        except ValueError:  # if the solution cannot be found the pixel is rendered inert
            ord_counts[:, p] = np.nan
            true_count[:, p] = np.nan
            covariance[:, :, p] = np.nan
            p_err[:, p] = np.nan
            m_err[:, p] = np.nan
            plot_int = False
            logger.warning(f'Pixel {p} has been discarded.')

        # plot the individual pixels flagged for plotting:
        if plot_int or debug:            
            fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14, 8))
            axes = axes.ravel()
            ax1 = axes[0]
            ax2 = axes[1]

            plt.suptitle(f'Pixel {p}: {"SUCCESS" if opt_params.success else "FAILURE"}')

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
            pre_gauss = fit_func(params, fine_phase_grid, orders=spectro.orders, leg_e=leg_e, leg_s=leg_s, to_sum=True)

            # get the initial residuals and weighted reduced chi^2:
            pre_residual = fit_func(params, bin_centers, y_counts=bin_counts[:, p], orders=spectro.orders, leg_e=leg_e,
                                    leg_s=leg_s)
            N_dof = len(pre_residual) - opt_params.nvarys
            pre_red_chi2 = np.sum(pre_residual ** 2) / N_dof

            # get the post-fitting residuals:
            opt_residual = fit_func(opt_params.params, bin_centers, y_counts=bin_counts[:, p], orders=spectro.orders,
                                    leg_e=leg_e, leg_s=leg_s)

            # first half of figure with data and models:
            ax1.grid()
            ax1.bar(bin_centers, bin_counts[:, p], width=bin_centers[1]-bin_centers[0], linewidth=0, color='k',
                    label='Data')  # plotting the histogram data
            ax1.plot(fine_phase_grid, pre_gauss, color='gray', label='Init. Guess')  # the initial guess model
            for y in gausses_i[:, :, p].T:
                ax1.plot(fine_phase_grid, y, label=f'Order {i}')  # the individual order post-fitting models
            ax1.set_ylabel('Photon Count')
            for b in order_edges[:-1, p]:
                ax1.axvline(b, linestyle='--', color='black')  # the virtual pixel boundaries
            ax1.axvline(order_edges[-1, p], linestyle='--', color='black', label='Order Edges')
            ax1.set_xlim([-1.2, 0])
            ax1.legend()

            res1.grid()
            for x, y in zip(bin_centers[:-1], pre_residual[:-1]):
                res1.plot(x, y, '.r')  # the initial weighted residuals
            res1.plot(bin_centers[-1], pre_residual[-1], '.r', label=r'Pre Red. $\chi^2=$'f'{pre_red_chi2:.1f}')
            res1.set_ylabel('Weighted Resid.')
            res1.set_xlim([-1.2, 0])

            res2.grid()
            for x, y in zip(bin_centers[:-1], opt_residual[:-1]):
                res2.plot(x, y, '.', color='purple')  # the post-fitting weighted residuals
            res2.plot(bin_centers[-1], opt_residual[-1], label=r'Post Red. $\chi^2=$'f'{red_chi2[p]:.1f}')
            res2.set_ylabel('Weighted Resid.')
            res2.set_xlabel(r'Phase $\times 2\pi$')
            res2.set_xlim([-1.2, 0])
            for b in order_edges[:, p]:
                res2.axvline(b, linestyle='--', color='black')  # adding in the virtual pixel boundaries

            # second figure with fitting result polynomials:
            if not np.isnan(fit_phis[0]) and not np.isnan(fit_phis[-1]):  # changing plot range in case orders missing
                new_x = np.linspace(fit_phis[0] - 0.01, fit_phis[-1] + 0.01, 1000)
            elif not np.isnan(fit_phis[0]):
                new_x = np.linspace(fit_phis[0] - 0.01, fit_phis[-2] + 0.01, 1000)
            elif not np.isnan(fit_phis[-1]):
                new_x = np.linspace(fit_phis[1] - 0.01, fit_phis[-1] + 0.01, 1000)

            def e_poly_linear(x):  # define the linear equation if the legendre poly had no quad term
                b = leg_e(fit_phis[0]) - fit_phis[0] * (leg_e(fit_phis[-1]) - leg_e(fit_phis[0])) / (
                        fit_phis[-1] - fit_phis[0])
                return (leg_e(fit_phis[-1]) - leg_e(fit_phis[0])) / (fit_phis[-1] - fit_phis[0]) * x + b

            masked_reg = gen.energy_to_wave(leg_e(new_x) * pix_E[-1, p] * u.eV)  # calculate the regular legendre
            masked_lin = gen.energy_to_wave(e_poly_linear(new_x) * pix_E[-1, p] * u.eV)  # calc the linear legendre
            deviation = masked_reg - masked_lin  # take the difference
            
            ax2.grid()
            ax2.plot(new_x, deviation, color='k')  # plotting difference
            for m, i in enumerate(fit_phis):
                ax2.plot(i, gen.energy_to_wave(leg_e(i) * pix_E[-1, p] * u.eV) - gen.energy_to_wave(
                    e_poly_linear(i) * pix_E[-1, p] * u.eV), '.',
                         markersize=10, label=f'Order {spectro.orders[::-1][m]}')  # plot locations of orders
            ax2.set_ylabel('Fitting Sol. Deviation from Linear (nm)')
            ax2.legend()

            ax2_2.grid()
            ax2_2.set_ylabel('R')
            ax2_2.set_xlabel(r'Energy (eV)')
            ax2_2.invert_xaxis()
            s_eval = leg_s(leg_e(new_x))  # retrieve sigmas from solution
            R = gen.sig_to_R(s_eval, leg_e(new_x))  # convert to spectral res.
            ax2_2.plot(leg_e(new_x), R, color='k')  # plot the R
            for m, i in enumerate(fit_phis):
                ax2_2.plot(leg_e(i), gen.sig_to_R(fit_sigs[m], leg_e(i)), '.', markersize=10,
                           label=f'Order {spectro.orders[::-1][m]}')  # plot the individual orders
            ax2.set_title(
                r'$E(\varphi)=$'f'{fit_e_coef[1]:.2e}P_2+{fit_e_coef[0]:.2f}P_1+{fit_e0:.2f}P_0\n'
                r'$\sigma(E)=$'f'{fit_s_coef[2]:.2e}P_2+{fit_s_coef[1]:.2e}P_1+{fit_s_coef[0]:.2e}P_0'
            )  # print the 2 solution functions

            ax1.set_xticks([])  # removes axis labels
            ax2.set_xticks([])
            res1.set_xticks([])

            plt.show()

    if plot:  # additional plots of all fits
        gausses[gausses < 0.01] = 0.01  # fills in the white spaces for the graph, doesn't mean anything
        # plot all solutions for all pixels in heat map:
        plt.imshow(gausses[::-1], extent=[1, sim.npix, bin_centers[0], bin_centers[-1]], aspect='auto', norm=LogNorm())
        cbar = plt.colorbar()
        cbar.ax.set_ylabel('Photon Count')
        plt.title("Fit MSF Model")
        plt.xlabel("Pixel Index")
        plt.ylabel(r"Phase ($\times \pi /2$)")
        plt.tight_layout()
        plt.show()

        # plot the weighted red. chi2 of all the pixels:
        plt.grid()
        plt.plot(detector.pixel_indices[red_chi2 < 1e6], red_chi2[red_chi2 < 1e6], '.', markersize=3)
        plt.semilogy()
        plt.title(r'Weighted Reduced $\chi^2$ for All Pixels')
        plt.ylabel(r'Weighted Red. $\chi^2$')
        plt.xlabel('Pixel Index')
        plt.show()

        # plot the corrected MSF spectrum with error bars:
        fig, ax = plt.subplots(int(np.ceil(nord/2)), 2, figsize=(15, int(5*nord)), sharex=True)
        axes = ax.ravel()
        for i in range(nord):
            axes[i].grid()
            axes[i].errorbar(pixels, true_count[i], yerr=np.array([m_err[i], p_err[i]]), fmt='-k', ecolor='r')
            axes[i].set_title(f'Order {spectro.orders[::-1][i]}')
            axes[i].set_ylim(bottom=0)
        axes[-1].set_xlabel("Pixel Index")
        axes[-2].set_xlabel("Pixel Index")
        axes[0].set_ylabel('Photon Count')
        axes[2].set_ylabel('Photon Count')
        plt.suptitle('Bleed-Corrected MSF Spectrum w/ Errors in Read')
        plt.tight_layout()
        plt.show()

    # assign bin edges, covariance matrices, virtual pix centers, and simulation settings to MSF class and save:
    msf = MKIDSpreadFunction(order_edges=order_edges, cov_matrix=covariance, waves=all_fit_phi, sigmas=all_fit_sig,
                             bins=bin_edges, sim_settings=sim)
    msf_file = f'{outdir}/msf.pkl'
    msf.save(msf_file)
    logger.info(f'Finished fitting all pixels. Saved MSF to {msf_file}.')
    return msf


if __name__ == '__main__':
    tic = time.perf_counter()  # recording start time for script

    # ==================================================================================================================
    # PARSE ARGUMENTS
    # ==================================================================================================================
    arg_desc = '''
               Extract the MSF Spread Function from the calibration spectrum.
               --------------------------------------------------------------
               This program loads the calibration photon table and conducts non-linear least squares fits to determine
               bin edges and the covariance matrix.
               '''
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter, description=arg_desc)

    # required MSF args:
    parser.add_argument('outdir', help='Directory for the output files (str).')
    parser.add_argument('msf_table', help='Directory/name of the MSF calibration photon table.')

    # optional MSF args:
    parser.add_argument('--bin_range', default=(-1, 0), type=tuple,
                        help='Start and stop of range for histogram.')
    parser.add_argument('--missing_order_pix', nargs='*',
                        default=[0, 349, 3, 350, 1299, 1, 0, 1299, 13, 1300, 2047, 2, 1300, 2047, 24],
                        help='Array of [startpix, endpix, missing-orders as single digit indexed from 1, and repeat],'
                             'e.g.: 0 999 13 1000 1999 25 2000 2047 4'
                             'will become [0, 999, 13,  1000, 1999, 25,  2000, 2047, 4]'
                             'where       sta sto  ord   sta  sto  ord   sta   sto  ord')
    parser.add_argument('--plot', action='store_true', default=False, type=bool,
                        help='If passed, plots will be shown.')

    # set arguments as variables
    args = parser.parse_args()

    # ==================================================================================================================
    # CHECK AND/OR CREATE DIRECTORIES
    # ==================================================================================================================
    os.makedirs(name=f'{args.outdir}', exist_ok=True)

    # ==================================================================================================================
    # START LOGGING TO FILE
    # ==================================================================================================================
    now = dt.now()
    logger = logging.getLogger('fitmsf')
    logging.basicConfig(level=logging.INFO)
    logger.info(msg="The process of modeling the MKID Spread Function (MSF) is recorded."
                     f"\nThe date and time are: {now.strftime('%Y-%m-%d %H:%M:%S')}.")
    # ==================================================================================================================
    # MSF EXTRACTION STARTS
    # ==================================================================================================================
    msf_table = Photontable(file_name=args.msf_table)
    missing_order_pix = np.reshape(list(map(int, args.missing_order_pix)), (-1, 3))
    missing_order_pix = [[(missing_order_pix[i, 0], missing_order_pix[i, 1]),
          [int(o) - 1 for o in str(missing_order_pix[i, 2])]] for i in range(missing_order_pix.shape[0])]

    fitmsf(
        msf_table=msf_table,
        sim=msf_table.query_header('sim_settings'),
        outdir=args.outdir,
        bin_range=args.bin_range,
        missing_order_pix=missing_order_pix,
        plot=args.plot
    )

    logger.info(f'\nTotal script runtime: {((time.perf_counter() - tic) / 60):.2f} min.')
    # ==================================================================================================================
    # MSF EXTRACTION ENDS
    # ==================================================================================================================


