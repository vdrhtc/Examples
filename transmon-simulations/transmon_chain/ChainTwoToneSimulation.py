from multiprocessing.pool import Pool

from transmon_chain.TransmonChain import TransmonChain
from tqdm import tqdm

import matplotlib
from matplotlib import ticker, colorbar as clb
matplotlib.use('Qt5Agg')
from matplotlib.pyplot import *
from numpy import *
from qutip import *

import os

os.environ["OMP_NUM_THREADS"]="1"

class ChainTwoToneSimulation:

    def __init__(self, chain: TransmonChain, sweet_spots, periods, chain_hilbert_space_truncation):
        self._chain = chain
        self._sweet_spots = sweet_spots
        self._periods = periods
        self._truncation = chain_hilbert_space_truncation
        self._phis = []


    def set_grid(self, currs, freqs):

        self._currs = currs
        self._freqs = freqs


        for i in range(0, self._chain.get_length()):
            self._phis.append((self._currs-self._sweet_spots[i])/self._periods[i])
        self._phis = array(self._phis).T

    def set_amps(self, amp1, amp2):
        self.amp1 = amp1
        self.amp2 = amp2

    def generate_caches(self):
        self._chain.clear_caches()

        print("Generating caches", end="")

        self._chain.set_phi(self._phis[0])
        for freq in self._freqs:
            self._chain.set_omega(2*pi*freq)
            self._chain.build_RF_subtrahend()
        print(".", end="")

        self._chain.set_omega(2 * pi * self._freqs[0])
        for phi in self._phis:
            self._chain.set_phi(phi)
            self._chain.build_H_full()
        print(".", end="")

        self._chain.build_c_ops()
        self._chain.build_RWA_driving()

        print(".OK")
        self._chain.build_low_energy_kets(self._truncation)



    def _column_calc(self, j):
        column = []
        self._chain.set_phi(self._phis[j])

        c_ops = self._chain.build_c_ops().copy()
        for i in range(len(c_ops)):
            c_ops[i] = self._chain.truncate_to_low_population_subspace(c_ops[i])

        for freq in self._freqs:
            self._chain.set_omega(2*pi*freq)
            H = self._chain.build_H_RWA() + self._chain.build_RWA_driving()
            H = self._chain.truncate_to_low_population_subspace(H)
            column.append(steadystate(H, c_ops))

        return column

    def run(self, n_cpus):

        with Pool(n_cpus) as p:
            self.spec = list(
                tqdm(p.imap(self._column_calc, range(len(self._currs))),
                              total=len(self._currs),
                              smoothing=0))
