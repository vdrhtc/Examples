from scipy import *
from qutip import *
from matplotlib.pyplot import *

class ZPulse:
    
    def __init__(self, Ts, params):
        self._params = params
        self._Ts = Ts
        
    def _step_rising(self):
        params = self._params
        return (tanh((self._Ts - 2 * params["tanh_sigma"] - params["start"]) / params["tanh_sigma"])+1)/2
    
    def _step_falling(self):
        params = self._params
        return (tanh((-self._Ts - 2 * params["tanh_sigma"] + params["start"] 
                      + params["duration"]) / params["tanh_sigma"])+1)/2
    
    def _normalized_pulse(self):
        raw = self._step_rising()*self._step_falling()
        normalized = raw/max(raw)
        return normalized
    
    def waveform(self):
        offset = self._params["phi_offset"]
        base = self._params["phi_base_level"]
        return self._normalized_pulse()*offset+base
    
    def plot(self):
        plot(self._Ts, self.waveform())
