from numpy import linspace, pi, ones_like
from qutip import *
from ReadoutResonator import *
from tqdm import tnrange, tqdm_notebook
import logging

logging.getLogger().setLevel(logging.WARNING)
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

import matplotlib.pyplot as plt
from multiprocessing import Pool
import glob, os
import pickle


class TwoToneSimulation:

    def __init__(self, dts, s1, s2, period1, period2):

        self.dts = dts
        self.s1 = s1
        self.s2 = s2
        self.period1 = period1
        self.period2 = period2

        self.options = Options(rhs_reuse=True, nsteps=1e4)

        self.spec = None

    def set_grid(self, curr1, curr2, freq1, freq2, freq_steps, curr_steps):

        self.currs = linspace(curr1, curr2, curr_steps)  # was (2, 6, 301)
        self.freqs = linspace(freq1, freq2, freq_steps)  # was (5.1, 5.5, 401)

        self.phi1s = (self.currs - self.s1) / self.period1 + 1 / 2  # in terms of pi
        self.phi2s = (self.currs - self.s2) / self.period2

    def set_amps(self, amp1, amp2):
        try:
            if len(amp1) > 1:
                self.amp1s = amp1
            else:
                self.amp1s = amp1[0]*ones_like(self.currs)
        except:
            self.amp1 = amp1*ones_like(self.currs)
        
        try:
            if len(amp2) > 1:
                self.amp2s = amp2
            else:
                self.amp2s = amp2[0]*ones_like(self.currs)
        except:
            self.amp2s = amp2*ones_like(self.currs)

    def generate_caches(self):
        self.dts.clear_caches()

        print("Generating caches", end="")
        for freq in self.freqs:
            self.amp1 = self.amp1s[0]
            self.amp2 = self.amp2s[0]
            self.phi1 = self.phi1s[0]
            self.phi2 = self.phi2s[0]
            self.freq = freq
            self.H()
        print(".", end="")

        for j in range(len(self.currs)):
            self.amp1 = self.amp1s[j]
            self.amp2 = self.amp2s[j]
            self.phi1 = self.phi1s[j]
            self.phi2 = self.phi2s[j]
            self.freq = self.freqs[0]
            self.H()

        print(".", end="\nOK")

    def generate_H(self):
        return [self.dts.H(self.phi1, self.phi2)] + \
               self.dts.Hdr_cont((self.amp1, self.amp2))

    def generate_H_RF_RWA(self):
        return self.dts.H_RF_RWA(self.phi1, self.phi2, self.freq) + \
               self.dts.Hdr_cont_RF_RWA((self.amp1, self.amp2))

    def H(self):
        return

    def steady_full(self):

        time_of_propagation = 1 / self.freq

        U = propagator(self.H(), time_of_propagation,
                       c_op_list=self.dts.c_ops(self.phi1, self.phi2),
                       args={'wd1': self.freq * 2 * pi, 'wd2': self.freq * 2 * pi},
                       options=self.options, unitary_mode='single',
                       parallel=False)  # , num_cpus=1)

        return propagator_steadystate(U)

    def steady_RWA(self):
        try:
            return steadystate(self.H(), c_op_list=self.dts.c_ops(self.phi1, self.phi2))
        except:
            return steadystate(self.H(), c_op_list=self.dts.c_ops(self.phi1, self.phi2), use_rcm=True)

    def steady(self):
        return

    def _column_calc(self, j):

        column = []
        self.phi1 = self.phi1s[j]
        self.phi2 = self.phi2s[j]
        self.amp1 = self.amp1s[j]
        self.amp2 = self.amp2s[j]

        for freq in self.freqs:  # range (0, self.Lf, self.Lf//self.res_f):
            self.freq = freq
            column.append(self.steady())

        return column

    def run(self, n_cpus):

        with Pool(n_cpus) as p:
            self.spec = list(
                tqdm_notebook(p.imap(self._column_calc, range(len(self.currs))),
                              total=len(self.currs),
                              smoothing=0))

        for f in glob.glob("*.pyx"):
            os.remove(f)
        # pickling the results
        # now = datetime.datetime.now()
        # dat_str = now.strftime("%d_%b_%H:%M")
        # pickle_data = {'data':self.spec, 'dts':self.dts,'amps':[self.amp1, self.amp2],'durs':[self.dur1, self.dur2], 'class':self}

        # pickle_out = open(("./tin/"+dat_str+".pickle"),"wb")
        # pickle.dump(pickle_data, pickle_out)
        # pickle_out.close()
        pickle_data = {'data': self.spec, 'dts': self.dts, 'amps': [self.amp1, self.amp2],
                       'class': self}
        pickle_out = open("double_tone.pickle", "wb")
#         pickle.dump(pickle_data, pickle_out)
        pickle_out.close()
        return self.spec

    def plot(self, n_state):
        # convert proper points of multidimensional spectrum into 2-D array
        self.g = []
        for i in range(0, self.res_ph):
            gg = []
            for j in range(0, self.res_f):  # range (0, self.Lf, self.Lf//self.res_f):
                gg.append(self.spec[i][j][n_state][0][n_state].real)  ## [i][j] correct
            self.g.append(gg)

        try:
            ans = array(self.g).T
            xax = []
            for i in self.ph_list:
                xax.append((self.currs)[i])
            yax = []
            for j in self.f_list:  # range(0, self.Lf, self.Lf//self.res_f):
                yax.append((self.freqs)[j])
            x_axis = array(xax)
            y_axis = array(yax)
            x_step = x_axis[1] - x_axis[0]
            x = np.concatenate((x_axis - x_step / 2, [x_axis[-1] + x_step / 2]))
            y_step = y_axis[1] - y_axis[0]
            y = np.concatenate((y_axis - y_step / 2, [y_axis[-1] + y_step / 2]))

            plt.pcolormesh(x, y, ans.real)
            plt.colorbar(use_gridspec=True).set_label('real part', rotation=270)
            plt.title('Two-tone spectroscopy')
            plt.ylabel('Frequency [GHz]')
            plt.xlabel('Current [e-4 A]');
            # plt.grid()

        except:

            raise Exception("You must 'run' first")

        # return self.g
