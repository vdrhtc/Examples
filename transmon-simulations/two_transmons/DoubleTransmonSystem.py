from scipy import *
from qutip import *
from matplotlib.pyplot import *
from tqdm import *


class DoubleTransmonSystem:

    def __init__(self, tr1, tr2, g):
        self._tr1 = tr1
        self._tr2 = tr2
        self._tr_list = [tr1, tr2]
        self._g = g

        self._Huu_cache = {}

    def clear_caches(self):
        self._Huu_cache = {}
        self._tr1.clear_caches()
        self._tr2.clear_caches()

    def two_qubit_operator(self, qubit1_operator=None, qubit2_operator=None):

        mask = [identity(self._tr1.get_Ns()), identity(self._tr2.get_Ns())]

        if qubit1_operator is not None:
            mask[0] = qubit1_operator
        if qubit2_operator is not None:
            mask[1] = qubit2_operator

        return tensor(*mask)

    def H(self, phi1, phi2):
        H = self.two_qubit_operator(qubit1_operator=self._tr1.H_diag_trunc(phi1)) + \
            self.two_qubit_operator(qubit2_operator=self._tr2.H_diag_trunc(phi2)) + \
            self.Hint(phi1, phi2)
        return H - H[0, 0]

    def H_RF_RWA(self, phi1, phi2, rotating_frame_freq):
        H = self.two_qubit_operator(qubit1_operator=self._tr1.H_diag_trunc(phi1)) + \
            self.two_qubit_operator(qubit2_operator=self._tr2.H_diag_trunc(phi2)) + \
            self.Hint_RF_RWA(phi1, phi2)

        return H - H[0, 0] - self._Huu(rotating_frame_freq)

    def _Huu(self, freq):
        try:
            return self._Huu_cache[freq]
        except:
            self._Huu_cache[freq] = freq * 2 * pi * self.two_qubit_operator(
                qubit1_operator=ket2dm(self._tr1.e_state())) + \
                                    2 * freq * 2 * pi * self.two_qubit_operator(
                qubit1_operator=ket2dm(self._tr1.f_state())) + \
                                    freq * 2 * pi * self.two_qubit_operator(
                qubit2_operator=ket2dm(self._tr2.e_state())) + \
                                    2 * freq * 2 * pi * self.two_qubit_operator(
                qubit2_operator=ket2dm(self._tr2.f_state()))
            return self._Huu_cache[freq]

    def H_td(self, waveform1, waveform2):
        return [self.Hc()] + self.Hj_td(waveform1, waveform2) + [self.Hint()]

    def H_td_diag_approx(self, waveform1, waveform2):
        Hj_td1 = self._tr1.H_td_diag_trunc_approx(waveform1)
        Hj_td1[0] = self.two_qubit_operator(qubit1_operator=Hj_td1[0])
        Hj_td2 = self._tr2.H_td_diag_trunc_approx(waveform2)
        Hj_td2[0] = self.two_qubit_operator(qubit2_operator=Hj_td2[0])
        return [Hj_td1, Hj_td2, self.Hint(waveform1[0], waveform2[0])]

    def H_diag_approx(self, phi1, phi2):
        return self.two_qubit_operator(qubit1_operator=self._tr1.H_diag_trunc_approx(phi1)) + \
               self.two_qubit_operator(qubit2_operator=self._tr2.H_diag_trunc_approx(phi2)) + \
               self.Hint(phi1, phi2)

    def Hc(self):
        return self.two_qubit_operator(qubit1_operator=self._tr1.Hc()) + \
               self.two_qubit_operator(qubit2_operator=self._tr2.Hc())

    def Hj(self, phi1, phi2):
        return self.two_qubit_operator(qubit1_operator=self._tr1.Hj(phi1)) + \
               self.two_qubit_operator(qubit2_operator=self._tr2.Hj(phi2))

    def Hj_td(self, waveform1, waveform2):
        Hj_td1 = self._tr1.Hj_td(waveform1)
        Hj_td1[0] = self.two_qubit_operator(qubit1_operator=Hj_td1[0])
        Hj_td2 = self._tr2.Hj_td(waveform2)
        Hj_td2[0] = self.two_qubit_operator(qubit2_operator=Hj_td2[0])
        return [Hj_td1, Hj_td2]

    def Hint(self, phi1, phi2):
        return self.two_qubit_operator(self._tr1.n(phi1), self._tr2.n(phi2)) * self._g

    def Hint_RF_RWA(self, phi1, phi2, td = False):
        op1 = self._tr1.n(phi1).data.copy()
        op2 = self._tr2.n(phi2).data.copy()
        # zeroing triangles
        for i in range(op1.shape[0]):
            for j in range(op1.shape[1]):
                if i < j:
                    op1[i, j] = 0
                if i > j:
                    op2[i, j] = 0
        half = self.two_qubit_operator(Qobj(op1), Qobj(op2))
        if td:
            return [self._g * (half + half.dag()), "cos((wd1-wd2)*t)"]
        else:
            return self._g * (half + half.dag())

    def Hdr(self, amplitudes, durations, starts, phases):
        Hdr1 = self._tr1.Hdr_cont(amplitudes[0], durations[0], starts[0], phases[0])
        Hdr1 = [self.two_qubit_operator(qubit1_operator=Hdr1[0]), Hdr1[1]]

        Hdr2 = self._tr2.Hdr(amplitudes[1], durations[1], starts[1], phases[1])
        Hdr2 = [self.two_qubit_operator(qubit2_operator=Hdr2[0]), Hdr2[1]]

        return [Hdr1] + [Hdr2]

    def Hdr_cont(self, amplitudes):
        Hdr1 = self._tr1.Hdr_cont(amplitudes[0])
        Hdr1 = [self.two_qubit_operator(qubit1_operator=Hdr1[0]), Hdr1[1]]

        Hdr2 = self._tr2.Hdr_cont(amplitudes[1])
        Hdr2 = [self.two_qubit_operator(qubit2_operator=Hdr2[0]), Hdr2[1]]

        return [Hdr1] + [Hdr2]

    def Hdr_cont_RF_RWA(self, amplitudes):
        return self.two_qubit_operator(qubit1_operator=self._tr1.Hdr_cont_RF_RWA(amplitudes[0])) + \
               self.two_qubit_operator(qubit2_operator=self._tr2.Hdr_cont_RF_RWA(amplitudes[1]))

    def _remove_global_phase(self, state):
        state_full = state.full()
        return state * sign(state_full[argmax(abs(state_full))])[0]

    def gg_state(self, phi1, phi2, energy=False):
        evals, evecs = self.H(phi1, phi2).eigenstates()
        evec = self._remove_global_phase(evecs[0])

        return evec if not energy else (evec, evals[0])

    def e_state(self, phi1, phi2, qubit_idx, energy=False):
        evals, evecs = self.H(phi1, phi2).eigenstates()

        if qubit_idx == 1:
            model_state = self.two_qubit_operator(self._tr1.e_state(), self._tr2.g_state())
        elif qubit_idx == 2:
            model_state = self.two_qubit_operator(self._tr1.g_state(), self._tr2.e_state())

        for idx, evec in enumerate(evecs):
            if abs((evec.dag() * model_state).full()) > 0.9:
                evec = self._remove_global_phase(evec)
                return evec if not energy else (evec, evals[idx])

    def ee_state(self, phi1, phi2, energy=False):
        evals, evecs = self.H(phi1, phi2).eigenstates()
        model_state = self.two_qubit_operator(self._tr1.e_state(), self._tr2.e_state())
        for idx, evec in enumerate(evecs):
            if abs((evec.dag() * model_state).full()) > 0.9:
                evec = self._remove_global_phase(evec)
                return evec if not energy else (evec, evals[idx])

    def c_ops(self, phi1, phi2):
        c_ops = []
        c_ops1 = self._tr1.c_ops(phi1)
        for c_op in c_ops1:
            c_ops.append(self.two_qubit_operator(qubit1_operator=c_op))
        c_ops2 = self._tr2.c_ops(phi2)
        for c_op in c_ops2:
            c_ops.append(self.two_qubit_operator(qubit2_operator=c_op))
        return c_ops

    def plot_spectrum(self, phi1s, phi2s, currents=None):
        assert len(phi1s) == len(phi2s)

        fluxes = list(zip(phi1s, phi2s))

        fixed_flux_spectra = []
        for phi1, phi2 in tqdm_notebook(fluxes, desc='Energy levels calculation'):
            H = self.H(phi1, phi2)
            evals = H.eigenenergies()
            fixed_flux_spectra.append(evals)

        fixed_flux_spectra = array(fixed_flux_spectra)
        eigenlevels = fixed_flux_spectra.T
        transitions_from_g = eigenlevels - eigenlevels[0]
        
        self._single_photon_transitions = transitions_from_g[1:3].T / 2 / pi
        self._two_photon_transitions = transitions_from_g[3:6].T / 2 / pi / 2

        if currents is not None:
            plot(currents, transitions_from_g[1:3].T / 2 / pi)
            plot(currents, transitions_from_g[3:6].T / 2 / pi / 2)
        else:
            plot(phi1s, transitions_from_g[1:3].T / 2 / pi)
            plot(phi1s, transitions_from_g[3:6].T / 2 / pi / 2)

        grid()

    def plot_per_qubit_xyz_dynamics(self, phi1, phi2, Ts, states):
        states_rf = []
        for t, state in zip(Ts, states):
            U = (1j * t * self.H_diag_approx(phi1, phi2)).expm()
            states_rf.append(U * state * U.dag())

        X1, Y1, Z1 = [], [], []
        X2, Y2, Z2 = [], [], []
        for state in states_rf:
            X1.append(expect(self.two_qubit_operator(qubit1_operator=self._tr1.sx()), state))
            Y1.append(expect(self.two_qubit_operator(qubit1_operator=self._tr1.sy()), state))
            Z1.append(expect(self.two_qubit_operator(qubit1_operator=self._tr1.sz()), state))

            X2.append(expect(self.two_qubit_operator(qubit2_operator=self._tr2.sx()), state))
            Y2.append(expect(self.two_qubit_operator(qubit2_operator=self._tr2.sy()), state))
            Z2.append(expect(self.two_qubit_operator(qubit2_operator=self._tr2.sz()), state))

        fig, axes = subplots(1, 2, figsize=(15, 3))
        axes[0].plot(Ts, X1, label=r"$\langle x\rangle$")
        axes[0].plot(Ts, Y1, label=r"$\langle y\rangle$")
        axes[0].plot(Ts, (array(Z1) + 1) / 2, label=r"$\langle z\rangle$")
        axes[1].plot(Ts, X2)
        axes[1].plot(Ts, Y2)
        axes[1].plot(Ts, (array(Z2) + 1) / 2)
        for ax in axes:
            ax.grid()
        axes[0].legend()
