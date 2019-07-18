import numpy as np
from numpy import *
from qutip import *

class Transmon:
    
    def __init__(self, Ec, Ej, d, gamma_rel, gamma_phi, Nc):
        self._Ec = Ec
        self._Ej = Ej
        self._d = d
        self._Nc = Nc
        self._Ns = Nc*2+1
        self._gamma_rel = gamma_rel
        self._gamma_phi = gamma_phi
        
        self._N_trunc = 3
    
    def _truncate(self, operator):
        return Qobj(operator[:self._N_trunc, :self._N_trunc])
    
    def Hc(self):
        return 4 * (self._Ec) * charge(self._Nc) ** 2
    
    def Hj(self, phi):
        return - self._Ej / 2 * tunneling(self._Ns, 1) * self._phi_coeff(phi)
    
    def Hj_td(self, phi_waveform):
        return [- self._Ej / 2 * tunneling(self._Ns, 1), self._phi_coeff(phi_waveform)]
    
    def H_diag_trunc(self, phi):
        H_charge_basis = self.Hc()+self.Hj(phi)
        evals, evecs = H_charge_basis.eigenstates()
        return self._truncate(H_charge_basis.transform(evecs))
    
    def H_diag_trunc_approx(self, phi):
        H_charge_basis = self.Hc()+self.Hj(0)
        evals, evecs = H_charge_basis.eigenstates()
        H_dta = self._truncate(H_charge_basis.transform(evecs))*sqrt(self._phi_coeff(phi))
        return H_dta - H_dta[0,0]
    
    def H_td_diag_trunc_approx(self, waveform):
        # approximating f_q = f_q^max * sqrt(cos sqrt(1+ d^2tan^2))
        return [self.H_diag_trunc(0), sqrt(self._phi_coeff(waveform))]
    
    def g_state(self, phi):
#         evals, evecs = self.H(phi).eigenstates()
#         return evecs[0]
        return basis(self._N_trunc, 0)
    
    def e_state(self, phi):
#         evals, evecs = self.H(phi).eigenstates()
#         return evecs[1]
        return basis(self._N_trunc, 1)

    def eigenlevels_approx(self, phi):
        evals = self.H_diag_trunc_approx(phi).eigenenergies()
        return evals
    
    def ge_freq_approx(self, phi):
        evals = self.H_diag_trunc_approx(phi).eigenenergies()
        return (evals[1]-evals[0])/2/pi
    
    def n(self, phi):
        H_charge_basis = self.Hc()+self.Hj(phi)
        evals, evecs = H_charge_basis.eigenstates()
        return self._truncate(charge(self._Nc).transform(evecs))
    
    def lowering(self, phi):
#         evals, evecs = self.H(phi).eigenstates()
#         return sum([self.n().matrix_element(evecs[j], evecs[j+1]) /
#                     self.n().matrix_element(evecs[0], evecs[1]) *
#                     evecs[j]*evecs[j+1].dag() for j in range(0, self._Ns-1)])
        evecs = [basis(self._N_trunc, i) for i in range(self._N_trunc)]
        return sum([self.n(phi).matrix_element(evecs[j], evecs[j+1]) /
                    self.n(phi).matrix_element(evecs[0], evecs[1]) *
                    evecs[j]*evecs[j+1].dag() for j in range(0, self._N_trunc-1)]) 
    
    def rotating_dephasing(self, phi):
        evecs = [basis(self._N_trunc, i) for i in range(self._N_trunc)]
        return sum([2*j*evecs[j]*evecs[j].dag() for j in range(0, self._N_trunc)]) 
    
    def c_ops(self, phi):
        return [sqrt(self._gamma_rel)*self.lowering(phi), sqrt(self._gamma_phi/2)*self.rotating_dephasing(phi)]
    
    def _phi_coeff(self, phi):
        return abs(np.cos(phi*pi)) * (1 + (self._d * np.tan(phi*pi)) ** 2) ** 0.5
    
    def get_Ns(self):
        return self._N_trunc
    
    def Hdr(self, amplitude, duration, start, phase = 0, freq = None):
        if freq is None:
            freq = self.ge_freq_approx(1/2)
        return [self.n(1/2)/self.n(1/2).matrix_element(self.g_state(1/2), self.e_state(1/2)), 
                "%f*cos(2*pi*%.16f*t+%f)*(1+np.sign(t-%f))*(1+np.sign(-t+%f))/4"%\
                (amplitude, freq, phase, start, start+duration)]

    def sz(self):
        return ket2dm(basis(3, 0))-ket2dm(basis(3,1))
        
    def sx(self):
        return basis(3, 0)*basis(3,1).dag()+basis(3,1)*basis(3,0).dag()
    
    def sy(self):
        return -1j*basis(3, 0)*basis(3,1).dag()+1j*basis(3,1)*basis(3,0).dag()
