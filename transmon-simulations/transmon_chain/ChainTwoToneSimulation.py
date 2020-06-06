from multiprocessing.pool import Pool

from transmon_chain.TransmonChain import TransmonChain
from tqdm import tqdm_notebook

import matplotlib
from matplotlib import ticker, colorbar as clb
# matplotlib.use('Qt5Agg')
from matplotlib.pyplot import *
from numpy import *
from qutip import *

from itertools import product 


class ChainTwoToneSimulation:

    def __init__(self, chain: TransmonChain, sweet_spots, periods, chain_hilbert_space_truncation, 
                 max_simultaneously_above_first_excited, steady_kwargs = None):
        self._chain = chain
        self._sweet_spots = sweet_spots
        self._periods = periods
        self._truncation = chain_hilbert_space_truncation
        self._max_simultaneously_above_first_excited = max_simultaneously_above_first_excited
        self._steady_kwargs = steady_kwargs if steady_kwargs is not None else {}
        self._phis = []
        self._Omegas = []
        self._freqs = []
        self._solver_error_coords = []
    
    
    def setter(self):
        pass

    def set_grid(self, freqs, currs=None, Omegas = None):

        self._currs = currs
        self._freqs = freqs
        self._Omegas = Omegas
        
        if self._currs is not None:
            for i in range(0, self._chain.get_length()):
                self._phis.append((self._currs-self._sweet_spots[i])/self._periods[i])
            self._params = array(self._phis).T
            self.setter = self._chain.set_phi
        else:
            self._params = self._Omegas
            self.setter = self._chain.set_Omega
            
            
    def generate_caches(self):
        self._chain.clear_caches()

        print("Generating caches", end="")

        self.setter(self._params[0])
        for freq in self._freqs:
            self._chain.set_omega(2*pi*freq)
            self._chain.build_RF_subtrahend()
        print(".", end="")

        self._chain.set_omega(2 * pi * self._freqs[0])
        for param in self._params:
            self.setter(param)
            self._chain.build_H_full()
            self._chain.build_RWA_driving()

        print(".", end="")

        self._chain.build_c_ops()
        self._chain.build_RWA_driving()

        print(".OK")
        self._chain.build_low_energy_kets(self._truncation, self._max_simultaneously_above_first_excited)

        self._c_ops = self._chain.build_c_ops().copy()
        for i in range(len(self._c_ops)):
            self._c_ops[i] = self._chain.truncate_to_low_population_subspace(self._c_ops[i])
            
        
    def _iteration(self, param_ids):
        param_idx, freq_idx = param_ids
        self.setter(self._params[param_idx])
        error_coords = []
     
        freq = self._freqs[freq_idx]
        self._chain.set_omega(2*pi*freq)
        H = self._chain.build_H_RWA() + self._chain.build_RWA_driving()
        H = self._chain.truncate_to_low_population_subspace(H)
        try:
            ans = steadystate(H, self._c_ops, **self._steady_kwargs)
        except Exception as e:
            print(f"Solver error at x: {param_idx}, y: {freq_idx}, {freq:.2f}: {e}")
            error_coords.append((param_idx, freq_idx))
            kwargs = self._steady_kwargs.copy()
            kwargs["use_rcm"]=True
            ans = steadystate(H, self._c_ops, **kwargs)
        
        return ans, error_coords
    
   
    def run(self, n_cpus):
        param_combinations = list(product(range(len(self._params)), 
                                          range(len(self._freqs))))
            
        with Pool(n_cpus) as p:
            results = list(
                tqdm_notebook(p.imap(self._iteration, param_combinations),
                              total=len(param_combinations),
                              smoothing=0))
            self._solver_error_coords = sum([result[1] for result in results])
            self.spec = reshape(array([result[0] for result in results], dtype=object), (len(self._params), len(self._freqs)))
