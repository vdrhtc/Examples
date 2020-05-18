from scipy import *
from qutip import *
from matplotlib.pyplot import *

from two_transmons.ZPulse import *

class VacuumRabiSimulation:
    
    def __init__(self, dts, Ts, params, readout_resonator):
        self._dts = dts
        self._Ts = Ts
        self._params = params
        self._r = readout_resonator
        
        self._rho0 = dts.e_state(1/2, 1/2, 1)
#         self._rho0 = 1/2*(dts.gg_state(1/2, 1/2)+dts.e_state(1/2, 1/2, 1)+dts.e_state(1/2, 1/2, 2)+dts.ee_state(1/2, 1/2))
        self._rho0 = self._rho0*self._rho0.dag()
#         self._rho0 = tensor(dts._tr1.e_state(params["phi_base_level"]), dts._tr2.g_state(1/2))
   
    def build_waveforms(self):
        waveform1 = ZPulse(self._Ts, self._params).waveform()
        waveform2 = ones_like(self._Ts)*1/2
        return waveform1, waveform2
        
    def run(self):
        options = Options(nsteps=20000, store_states=True)
        
        self._c_ops = []  #dts.c_ops(1/2, 1/2)
        
        self._result = mesolve(self._dts.H_td_diag_approx(*self.build_waveforms()), self._rho0, 
                               self._Ts, c_ops = self._c_ops, progress_bar=True, options=options)
        return self._result
    
    def visualize_projections(self):
        state1 = self._rho0
        state2 = self._dts.e_state(1/2, 1/2, 2)
        state2 = state2*state2.dag()
        
        projections1 = []
        projections2 = []
        
        for state in self._result.states:
            if len(self._c_ops) == 0:
                state = state*state.dag()
            projections1.append(expect(state, state1))
            projections2.append(expect(state, state2))
        
        plt.plot(self._Ts, np.abs(projections1))
        plt.plot(self._Ts, np.abs(projections2))
        
    def visualize_joint_readout(self, f):
        data = array([expect(state, self._r.measurement_operator(f, self._r.get_dipsersive_shifts())) \
                      for state in self._result.states])
        fig, axes = subplots(1, 2, figsize=(15,5))
        axes[0].plot(self._Ts, data.real)
        axes[1].plot(self._Ts, data.imag, "C1")