from scipy import *
from qutip import *
from matplotlib.pyplot import *
from itertools import product

from ZPulse import ZPulse

class Tomography:
    
    def __init__(self, dts, Ts, params, readout_resonator):
        self._dts = dts
        self._Ts = Ts
        self._params = params
        self._r = readout_resonator
        
        self._drive_amplitude = 0.01*2*pi/2
        self._pi_duration = 50*2
        
        self._q1_rotations = [(0, 0),
                              (self._pi_duration/2, 0),
                              (self._pi_duration/2, pi/2),
                              (self._pi_duration/2, pi),
                              (self._pi_duration/2, -pi/2),
                              (self._pi_duration, 0), 
                              (self._pi_duration, pi/2)]
            
        self._q2_rotations = [(0, 0), 
                              (self._pi_duration/2, 0),
                              (self._pi_duration/2, pi/2), 
                              (self._pi_duration/2, pi),
                              (self._pi_duration/2, -pi/2),
                              (self._pi_duration, 0),
                              (self._pi_duration, pi/2)]
        self._2q_rotations = list(product(self._q1_rotations, self._q2_rotations))
        
        self._rho0 = dts.ee_state(1/2,1/2)
#         self._rho0 = 1/2*(dts.gg_state(1/2,1/2)+dts.e_state(1/2, 1/2,1)+
#                           dts.e_state(1/2, 1/2,2)+dts.ee_state(1/2, 1/2))
#         self._rho0 = dts.e_state(1/2, 1/2, 1)
#         self._rho0 = 1/sqrt(2)*(dts.e_state(1/2, 1/2, 1)*1j+dts.e_state(1/2, 1/2, 2))
        self._rho0 = self._rho0*self._rho0.dag()
        
    def build_waveforms(self):
        waveform1 = ZPulse(self._Ts, self._params).waveform()
        waveform2 = ones_like(self._Ts)*1/2
        return waveform1, waveform2
        
    def run(self):
        self._options = Options(nsteps=20000, store_states=True)
        
        self._c_ops = self._dts.c_ops(1/2, 1/2)
        
        self._results = []
        
        E0 = self._dts.gg_state(1/2, 1/2, True)[1]
        E10 = self._dts.e_state(1/2, 1/2, 1, True)[1]
        E01 = self._dts.e_state(1/2, 1/2, 2, True)[1]
        self._freqs = ((E10 - E0)/2/pi, (E01 - E0)/2/pi)
        
        self._results = parallel_map(self._tomo_step, self._2q_rotations)
            
        return self._results
    
    def _tomo_step(self, rotations):
        dur1, dur2 = rotations[0][0], rotations[1][0]
        phi1, phi2 = rotations[0][1], rotations[1][1]

        self._H = self._dts.H_td_diag_approx(*self.build_waveforms())
        self._H += self._dts.Hdr([self._drive_amplitude]*2, 
                                 (dur1, dur2), 
                                 (self._params["duration"]+20, self._params["duration"]+20+self._pi_duration+10),
                                 (phi1, phi2),
                                 self._freqs)

        return mesolve(self._H, self._rho0, 
                       self._Ts, c_ops = self._c_ops, 
                       progress_bar=True, 
                       options=self._options)
                       
    def _joint_expect(self, state, f, chi1, chi2):
        return expect(state, self._r.measurement_operator(f, chi1, chi2))
    
    def visualize_joint_readout(self, f, chi1, chi2):
        
        fig, axes = subplots(1, 2, figsize=(15,5))
        for result in self._results:
            data = array(serial_map(self._joint_expect, result.states, task_args = (f, chi1, chi2)))
            axes[0].plot(self._Ts, data.real)
            axes[1].plot(self._Ts, data.imag)
