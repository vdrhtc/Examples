import glob

from scipy import *
from qutip import *
from multiprocessing import Pool
from tqdm import tqdm_notebook
import os

class ATSplittingSimulation:

    def __init__(self, dts, s1, s2, period1, period2, ):
        self.dts = dts
        self.s1 = s1
        self.s2 = s2
        self.period1 = period1
        self.period2 = period2
        self.options = Options(rhs_reuse=True, nsteps=1e5)

    def set_amplitudes(self, amp1, amp2):
        self.amp1 = amp1
        self.amp2 = amp2

    def set_grid(self, freq1s, freq2s):
        self.freq1s, self.freq2s = freq1s, freq2s

    def set_fluxes_from_current(self, current):
        self.phi1 = (current - self.s1) / self.period1 + 1 / 2  # in terms of pi
        self.phi2 = (current - self.s2) / self.period2

    def generate_caches(self):
        for freq1 in self.freq1s:
            self.freq1 = freq1
            self.freq2 = self.freq2s[0]
            self.H()
        for freq2 in self.freq2s:
            self.freq1 = self.freq1s[0]
            self.freq2 = freq2
            self.H()

    def H(self):
        Hdr1 = self.dts.two_qubit_operator(qubit1_operator=self.dts._tr1.Hdr_cont_RF_RWA(self.amp1))
        Hdr2 = self.dts._tr2.Hdr_cont(self.amp2)
        Hdr2 = [self.dts.two_qubit_operator(qubit2_operator=Hdr2[0]), Hdr2[1]]
        return [self.dts.H_RF_RWA(self.phi1, self.phi2, rotating_frame_freq=self.freq1) + Hdr1,
                Hdr2]

    def H_RF_RWA(self):
        return self.dts.H_RF_RWA(self.phi1, self.phi2, self.freq1) + \
               self.dts.Hdr_cont_RF_RWA((self.amp1, self.amp2))

    def steady(self):
        delta_f2 = abs(self.freq2 - self.freq1)
        if delta_f2 == 0:
            return steadystate(self.H_RF_RWA(), self.dts.c_ops(self.phi1, self.phi2))
        else:
            time_of_propagation = 1 / delta_f2
            # print("Propagation time:", time_of_propagation, "ns")

            U = propagator(self.H(),
                           time_of_propagation,
                           c_op_list=self.dts.c_ops(self.phi1, self.phi2),
                           args={'wd2': delta_f2 * 2 * pi},
                           options=self.options, unitary_mode='single',
                           parallel=False, progress_bar=None)  # , num_cpus=1)

            return propagator_steadystate(U)

    def _column_calc(self, freq2):
        phase_column = []
        self.freq2 = freq2

        for freq1 in self.freq1s:  # range (0, self.Lf, self.Lf//self.res_f):
            self.freq1 = freq1
            phase_column.append(self.steady())

        return phase_column

    def run(self, n_cpus):
        with Pool(n_cpus) as p:
            self.spec = list(tqdm_notebook(p.imap(self._column_calc, self.freq2s),
                                           total=len(self.freq2s),
                                           smoothing=0))

            for f in glob.glob("*.pyx"):
                os.remove(f)
