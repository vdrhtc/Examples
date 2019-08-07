from qutip import *
from tqdm import tqdm_notebook
from scipy.constants import h, hbar, k as kb
from itertools import product
from scipy import optimize
from importlib import reload
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

class Dynamics1:
    
    
    def __init__(self, dts, s1, s2, period1, period2, amp1, amp2, dur1, dur2):
        
        self.dts = dts
        self.s1 = s1
        self.s2 = s2 
        self.period1 = period1
        self.period2 = period2
        
        self.options = Options(rhs_reuse = True, nsteps=1e4)
        
        
        self.X = linspace(0,2,2) 
        self.Y = linspace(0,2,2)    
       
        self.spec = None
        self.amp1 = amp1
        self.amp2 = amp2
        self.dur1 = dur1
        self.dur2 = dur2
       

    def _steady(self):
   
        time_of_propagation = 1/self.freq

        U = propagator(self.Hp, time_of_propagation, c_op_list = self.dts.c_ops(self.phi1, self.phi2),
                       args = {'wd1':self.freq*2*pi, 'wd2':self.freq*2*pi}, options=self.options, unitary_mode='single', 
                       parallel=False, progress_bar= None)#, num_cpus=1)
    
        return propagator_steadystate(U)
  
        
    def _phase_calc(self, j):
        
        phase_column = []  
        self.phi1 = self.fl_vec1[j]
        self.phi2 = self.fl_vec2[j]
        
        self.Hp = [self.dts.H(self.phi1, self.phi2)] + self.dts.Hdr([self.amp1, self.amp2],
                                                         [self.dur1, self.dur2], [0, 0], [self.phi1, self.phi2]) 
        
        for i in self.f_list: #range (0, self.Lf, self.Lf//self.res_f): 
            self.freq = self.Y[i]
            phase_column.append(self._steady())

        return phase_column

    def run(self, curr1, curr2, freq1, freq2, res_f, res_ph):
        self.X = linspace(curr1, curr2, res_ph)  #was (2, 6, 301)
        self.Y = linspace(freq1, freq2, res_f) #was (5.1, 5.5, 401)
        
        self.fl_vec1 = (self.X - self.s1)/self.period1 + 1/2 #in terms of pi
        self.fl_vec2 = (self.X - self.s2)/self.period2
        
        self.Lph = len(self.fl_vec1)
        self.Lf = len(self.Y)
        self.res_f = res_f
        self.res_ph = res_ph
        self.ph_list = np.array(np.linspace(0., self.Lph-1, res_ph), dtype = int)
        self.f_list = np.array(np.linspace(0., self.Lf-1, res_f), dtype = int)
        
        #self.ph_list = list(range(0, self.Lph, (self.Lph)//(self.res_ph-1)))
        #self.f_list = list(range (0, self.Lf, (self.Lf)//(self.res_f-1)))
        self.spec = parfor(self._phase_calc, self.ph_list, num_cpus = 8) 
        return self.spec
    
    def run_pb(self, res_f, res_ph, n_cpus):
        self.X = linspace(2, 6, res_ph) 
        self.Y = linspace(5.1, 5.5, res_f)
        
        self.fl_vec1 = (self.X - self.s1)/self.period1 + 1/2 #in terms of pi
        self.fl_vec2 = (self.X - self.s2)/self.period2
        
        self.Lph = len(self.fl_vec1)
        self.Lf = len(self.Y)
        
        self.res_f = res_f
        self.res_ph = res_ph
        self.ph_list = np.array(np.linspace(0., self.Lph-1, res_ph), dtype = int)
        self.f_list = np.array(np.linspace(0., self.Lf-1, res_f), dtype = int)
        self.spec = []
        with Pool(n_cpus) as p:
        #self.ph_list = list(range(0, self.Lph, (self.Lph)//(self.res_ph-1)))
        #self.f_list = list(range (0, self.Lf, (self.Lf)//(self.res_f-1)))
            self.spec = list(tqdm_notebook(p.imap(self._phase_calc, self.ph_list), total = len(self.ph_list), smoothing=0))
            
        for f in glob.glob("*.pyx"):
            os.remove(f)
        #pickling the results
        pickle_data = {'data':self.spec, 'dts':self.dts,'amps':[self.amp1, self.amp2],'durs':[self.dur1, self.dur2], 'class':self}
        pickle_out = open("double_tone.pickle","wb")
        pickle.dump(pickle_data, pickle_out)
        pickle_out.close()
        #pickle_in = open("double_tone.pickle","rb")
        #example_dict2 = pickle.load(pickle_in)
        #pickle_in.close()
        return self.spec
    
    def plot(self, n_state):
        #convert proper points of multidimensional spectrum into 2-D array
        self.g = []
        for i in range(0, self.res_ph):
            gg = []
            for j in range(0, self.res_f):   #range (0, self.Lf, self.Lf//self.res_f):
                gg.append(self.spec[i][j][n_state][0][n_state].real) ## [i][j] correct
            self.g.append(gg)
        
        
        
        try: 
            ans = array(self.g).T
            xax = []
            for i in self.ph_list:
                xax.append((self.X)[i])
            yax = []
            for j in self.f_list:#range(0, self.Lf, self.Lf//self.res_f):
                yax.append((self.Y)[j]) 
            x_axis = array(xax)
            y_axis = array(yax)
            x_step = x_axis[1] - x_axis[0]
            x = np.concatenate((x_axis - x_step/2, [x_axis[-1] + x_step/2]))
            y_step = y_axis[1] - y_axis[0]
            y = np.concatenate((y_axis - y_step/2, [y_axis[-1] + y_step/2]))

            plt.pcolormesh(x, y, ans.real)
            plt.colorbar(use_gridspec = True).set_label('real part', rotation=270)
            plt.title('Double-tone spectroscopy')
            plt.ylabel('Frequency [GHz]')
            plt.xlabel('Current [e-4 A]');
            plt.grid()
            
        except: 
            
            raise Exception("You must 'run' first")

        #return self.g


