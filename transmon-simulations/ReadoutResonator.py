import numpy as np
from numpy import random, linspace, real, imag, abs, angle, exp
from qutip import ket2dm, tensor, basis
from matplotlib.pyplot import *


class ReadoutResonator:

    def __init__(self, f_res, Ql, Qc, line_att=0.0345, phi=0.5, noise_sigma=1e-4, delay=0., alpha = 0):
        self._f_res = f_res
        self._Ql = Ql
        self._Qc = Qc
        self._line_att = line_att
        self._phi = phi
        self._noise_sigma = noise_sigma
        self._delay = delay
        self._alpha = alpha

    def set_qubit_parameters(self, g1, g2, f1, f2, alpha_1, alpha_2):
        self._g1 = g1
        self._g2 = g2
        self._f1 = f1
        self._f2 = f2
        self._alpha_1 = alpha_1
        self._alpha_2 = alpha_2

    def set_noise_sigma(self, sigma):
        self._noise_sigma = sigma

    def S_param(self, f, fr_shift):
        try:
            size = len(f)
        except TypeError:
            size = 1

        noise = 1 * random.normal(size=size, scale=self._noise_sigma / 2) + \
                1j * random.normal(size=size, scale=self._noise_sigma / 2)

        env_term = self._line_att * np.exp(-2 * np.pi * 1j * self._delay * f+1j*self._alpha)
        own_term = exp(1j * self._phi) * (self._Ql / self._Qc) / (
                1 + 2 * 1j * self._Ql * ((f / (self._f_res - fr_shift)) - 1))
        result = env_term * (1 - own_term) + noise
        return result if size > 1 else result[0]

    def get_dipsersive_shifts(self):

        chi_1_01 = -self._g1 ** 2 / (self._f1 - self._f_res)
        chi_1_12 = -2 * self._g1 ** 2 / (self._f1 - self._alpha_1 - self._f_res)
        chi_1_23 = -3 * self._g1 ** 2 / (self._f1 - 2 * self._alpha_1 - self._f_res)
        chi_2_01 = -self._g2 ** 2 / (self._f2 - self._f_res)
        chi_2_12 = -2 * self._g2 ** 2 / (self._f2 - self._alpha_2 - self._f_res)
        chi_2_23 = -3 * self._g2 ** 2 / (self._f2 - 2 * self._alpha_2 - self._f_res)

        shift_00 = 0
        shift_10 = 2 * chi_1_01 - chi_1_12
        shift_20 = chi_1_01 + chi_1_12 - chi_1_23
        shift_01 = 2 * chi_2_01 - chi_2_12
        shift_02 = chi_2_01 + chi_2_12 - chi_2_23
        shift_11 = shift_01 + shift_10
        shift_21 = shift_20 + shift_01
        shift_12 = shift_10 + shift_02
        shift_22 = shift_20 + shift_02

        return [[shift_00, shift_01, shift_02],
                [shift_10, shift_11, shift_12],
                [shift_20, shift_21, shift_22]]

    def measurement_operator(self, f, shifts):
        N = len(shifts)
        return sum([sum(
            [ket2dm(tensor(basis(N, i), basis(N, j))) * self.S_param(f, shift) for j, shift in
             enumerate(row)]) for
             i, row in enumerate(shifts)])

    def plot_all(self, fs):
        if fs is None:
            fs = linspace(self._f_res - 10e-3, self._f_res + 10e-3, 201)
        fig, axes = subplots(nrows=2, ncols=2, figsize=(15, 5))
        axes = axes.ravel()
        shifts = self.get_dipsersive_shifts()

        for idx, digest in enumerate([abs, angle, real, imag]):
            for i, row in enumerate(shifts):
                for j, shift in enumerate(row):
                    axes[idx].plot(fs, digest(self.S_param(fs, shift)), label="%d%d" % (i, j))
        legend()

    def plot(self, shift=0, fs=None, digest=real):
        if fs is None:
            fs = linspace(self._f_res - 10e-3, self._f_res + 10e-3, 201)
        plot(fs, digest(self.S_param(fs, shift)))
