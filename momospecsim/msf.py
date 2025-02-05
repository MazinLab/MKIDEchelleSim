import pickle
import numpy as np
import matplotlib.pyplot as plt


class MKIDSpreadFunction:
    def __init__(self, order_edges=None, cov_matrix=None, waves=None, sigmas=None, bins=None,filename: str = '',
                 sim_settings=None):
        """
        :param order_edges: n_ord+1 by n_pix array of wavelength or phase bins
        :param cov_matrix: n_ord by n_ord by n_pix array of covariance fractions belonging to another order
        :param waves: n_ord by n_pix array of retrieved wavelengths or phases
        :param sigmas: n_ord by n_pix array of retrieved order standevs in phase
        :param bins: smaller bin edges for phase histogram
        :param str filename: where the MKIDSpreadFunction exists or where to save a new file
        :param sim_settings: simulation settings for the spectrograph
        """
        self.filename = filename
        if filename:
            self._load()
        else:
            self.order_edges = order_edges
            self.cov_matrix = cov_matrix
            self.waves = waves
            self.sigmas = sigmas
            self.bins = bins
            self.sim_settings = sim_settings

    def save(self, filename: str):
        """
        :param filename: file name for saving
        :return: pickles MSF to file, cannot use .npz due to sim_settings object
        """
        with open(filename, 'wb') as f:
            pickle.dump(self.order_edges, f)
            pickle.dump(self.cov_matrix, f)
            pickle.dump(self.waves, f)
            pickle.dump(self.sigmas, f)
            pickle.dump(self.bins, f)
            pickle.dump(self.sim_settings, f)

    def _load(self):
        with open(self.filename, 'rb') as f:
            self.order_edges = pickle.load(f)
            self.cov_matrix = pickle.load(f)
            self.waves = pickle.load(f)
            self.sigmas = pickle.load(f)
            self.bins = pickle.load(f)
            self.sim_settings = pickle.load(f)
