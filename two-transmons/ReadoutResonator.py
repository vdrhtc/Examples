from scipy import *
from qutip import *
from matplotlib.pyplot import *

class ReadoutResonator:
    
    def __init__(self, f_res, Ql, Qc, line_att = 0.0345, phi=0.5, noise_sigma = 1e-4, delay = 0.):
        self._f_res = f_res
        self._Ql = Ql
        self._Qc = Qc
        self._line_att = line_att
        self._phi = phi
        self._noise_sigma = noise_sigma
        self._delay = delay
        
    def set_noise_sigma(self, sigma):
        self._noise_sigma = sigma
    
    def S_param(self, f, fr_shift):
        try:
            size = len(f)
        except TypeError:
            size = 1
            
        noise = 1 * random.normal(size = size, scale = self._noise_sigma/2) + \
                1j*random.normal(size = size, scale = self._noise_sigma/2)
        
        env_term = self._line_att*np.exp(-2*np.pi*1j*self._delay)
        own_term = exp(1j*self._phi)*(self._Ql/self._Qc)/(1 + 2*1j*self._Ql*((f/(self._f_res-fr_shift))-1))
        result = env_term*(1 - own_term) + noise
        return result if size>1 else result[0]
    
    def measurement_operator(self, f, chi1, chi2, N = 3):
        S00 = self.S_param(f, 0)
        S10 = self.S_param(f, chi1)
        S01 = self.S_param(f, chi2)
        S11 = self.S_param(f, chi1+chi2)
        return ket2dm(tensor(basis(N,0), basis(N,0)))*S00 +\
               ket2dm(tensor(basis(N,1), basis(N,0)))*S10 +\
               ket2dm(tensor(basis(N,0), basis(N,1)))*S01 +\
               ket2dm(tensor(basis(N,1), basis(N,1)))*S11
    
    def plot(self, shift = 0, fs = None, digest = real):
        if fs is None:
            fs = linspace(self._f_res-10e-3, self._f_res+10e-3, 201)
        plot(fs, digest(self.S_param(fs, shift)))

