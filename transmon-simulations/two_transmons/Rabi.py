from scipy import *
from qutip import *
from matplotlib.pyplot import *

class Rabi:
    
    def __init__(self, dts, Ts, qubit_to_drive, readout_resonator):
        self._dts = dts
        self._Ts = Ts
        
        self._drive_amplitude = 0.01*2*pi
        self._duration = Ts[-1]-Ts[0]
        self._qubit_to_drive = qubit_to_drive
        self._r = readout_resonator
        
 
        
        self._rho0 = self._dts.gg_state(1/2, 1/2)
        self._rho0 = self._rho0*self._rho0.dag()
        
    def build_waveforms(self):
        waveform1 = ones_like(self._Ts)*1/2
        waveform2 = ones_like(self._Ts)*1/2
        return waveform1, waveform2
  
        
    def run(self):
        options = Options(nsteps=20000, store_states=True)
        
        self._c_ops =  self._dts.c_ops(1/2, 1/2)
        
        self._H = self._dts.H_td_diag_approx(*self.build_waveforms())
        
        E0 = self._dts.gg_state(1/2, 1/2, True)[1]
        E10 = self._dts.e_state(1/2, 1/2, 1, True)[1]
        E01 = self._dts.e_state(1/2, 1/2, 2, True)[1]
        self._freqs = ((E10 - E0)/2/pi, (E01 - E0)/2/pi)
        
        amplitudes = [0,0]
        amplitudes[self._qubit_to_drive-1] = 0.01*2*pi
        self._H += self._dts.Hdr(amplitudes, 
                           (self._duration, self._duration), 
                           (0, 0),
                           (0, 0))

        self._result = mesolve(self._H, 
                               self._rho0, 
                               self._Ts, 
                               c_ops = self._c_ops, 
                               progress_bar=True, 
                               options=options, 
                               args = {'wd%d'%self._dts.get_single_transmons()[0].get_index(): self._freqs[0]*2*pi,
                                       'wd%d'%self._dts.get_single_transmons()[1].get_index(): self._freqs[1]*2*pi},
                               e_ops = [tensor(ket2dm(basis(3, 1)), identity(3)), 
                                        tensor(identity(3), ket2dm(basis(3, 1)))])
        return self._result
    
    def visualize_joint_readout(self, f, chi1, chi2):
        data = array([expect(state, self._r.measurement_operator(f, chi1, chi2)) for state in self._result.states])
        fig, axes = subplots(2, 1)
        axes[0].plot(self._Ts, data.real)
        axes[1].plot(self._Ts, data.imag, "C1")
