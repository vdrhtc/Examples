from scipy import *
from qutip import *
from matplotlib.pyplot import *

class TransmonControl:
    
    def __init__(self, tr):
        
        self._tr = tr
        
        self._detuning = 0
        
        self._exc_freq = tr.ge_freq_approx(1/2) + self._detuning
        
        self._H = [tr.H_diag_trunc_approx(1/2), 
                   tr.Hdr(0.01*2*pi, 25, 0), 
                   tr.Hdr(0.01*2*pi, 25, 30, phase=0)]
        
        self._rho0 = ket2dm(tr.g_state())
        
        self._Ts = linspace(0, 80, 100)
        
        self._c_ops = tr.c_ops(1/2)
     
        
        
        
    def run(self):
        options = Options(nsteps=20000, store_states=True)
        
        self._result = mesolve(self._H, self._rho0, 
                               self._Ts,args = {'wd%d'%self._tr.get_index():self._exc_freq*2*pi},c_ops = self._c_ops,
                               progress_bar=True, options = options, e_ops = [])
        return self._result
    
    def visualize_dynamics(self):
        Us = [(-1j*t*self._tr.H_diag_trunc_approx(1/2)).expm() for t in self._Ts]
        
        plot(self._Ts, [expect(U.dag()*state*U, ket2dm(self._tr.e_state())) \
                        for U, state in zip(Us, self._result.states)])
        plot(self._Ts, [expect(U.dag()*state*U, 1/2*ket2dm(self._tr.e_state()+self._tr.g_state())) \
                        for U, state in zip(Us, self._result.states)])
        plot(self._Ts, [expect(U.dag()*state*U, 1/2*ket2dm(self._tr.e_state()+1j*self._tr.g_state())) \
                        for U, state in zip(Us, self._result.states)])
        
        grid()
