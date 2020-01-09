from qutip import *
from tqdm import tqdm_notebook
from scipy.constants import h, hbar, k as kb
from itertools import product
from scipy import optimize
from importlib import reload

from tqdm import tnrange, tqdm_notebook
import logging
logging.getLogger().setLevel(logging.WARNING)
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import matplotlib.pyplot as plt
from multiprocessing import Pool
import glob, os
import pickle
from numpy import *
import numpy as np

import Transmon
reload(Transmon)
from Transmon import *

import TransmonControl
reload(TransmonControl)
from TransmonControl import *

import DoubleTransmonSystem
reload(DoubleTransmonSystem)
from DoubleTransmonSystem import *

class Sidebands:
    
    
    def __init__(self, dts):

        self.dts = dts
        
        self.options = Options(rhs_reuse = True, nsteps=1e4)
                 
    def _steady1 (self, phi1, phi2, freq, Hp):
        
        time_of_propagation = 1/freq
        U = propagator(Hp, time_of_propagation, c_op_list = self.dts.c_ops(phi1,phi2), args = {'wd1':freq*2*pi, 'wd2':freq*2*pi},
               options=self.options, unitary_mode='single', parallel=False, progress_bar= None)
    
        return propagator_steadystate(U) ## formatted [num_of_colomn][0][num of element]
    
    def _amp_calc(self, amp):
        
        amp_column = []  
        phi1 = self.fl_vec1[self.phase]
        phi2 = self.fl_vec2[self.phase]
        dur1 = 2e4  #duration of pulse and right now - of the time scale
        dur2 = 2e4 
        Hp = [self.dts.H(phi1,phi2)] + self.dts.Hdr([amp, amp], [dur1, dur2], [0,0], [phi1,phi2]) # as we have cut out freq from Transmon
  
        for freq in self.freq_vec:     # here and after - it all works well!
            amp_column.append(self._steady1(phi1, phi2, freq, Hp = Hp))

        print(amp)
        return amp_column
    
    def run(self, start_freq, stop_freq, res_freq, start_amp, stop_amp,res_amp, phase, n_cpus): #phase is a number from 0 to 300
        self.freq1 = start_freq
        self.freq2 = stop_freq
        self.amp1  = start_amp
        self.amp2  = stop_amp
        self.phase = phase
        self.n_cpus = n_cpus
        self.res_amp  = res_amp
        self.res_freq = res_freq
        self.freq_vec = np.linspace(self.freq1, self.freq2, self.res_freq)
        self.amp_vec  = np.linspace(self.amp1,  self.amp2,  self.res_amp)
        #####################################
        X = linspace(2, 6, 301) ##currents
        Y = linspace(5.1, 5.5, 401) ##frequencies, in GHz

        SweetSpot1 = 4.14 #4.14
        SweetSpot2 = 4.12

        T1 = 11.0 #higher "\/" #11.1 #periods of the spectrum curves
        T2 = 6.4 #lower "/\    #6.4 
        self.fl_vec1 = (X - SweetSpot1)/T1 + 1/2 #in terms of pi
        self.fl_vec2 = (X - SweetSpot2)/T2
        #######################################################
        with Pool(self.n_cpus) as p:
            self.spec = list(tqdm_notebook(p.imap(self._amp_calc, self.amp_vec), total = len(self.amp_vec), smoothing=0))
            
        for f in glob.glob("*.pyx"):
            os.remove(f)
        
        return self.spec
    
    def plot(self, n, log = False):
        try:
            
            ann = zeros((self.res_freq, self.res_amp)).T
            
            for i in range (0, self.res_amp):
                for j in range (0, self.res_freq):                                    
                                ann[i][j] = self.spec[i][j][n][0][n].real
            if log == True:
                ann = log10(ann)
            an = array(ann).T
            xax = self.amp_vec
            yax = self.freq_vec
            x_axis = array(xax)
            y_axis = array(yax)
            x_step = x_axis[1]-x_axis[0]
            x = np.concatenate((x_axis-x_step/2, [x_axis[-1]+x_step/2]))
            y_step = y_axis[1]-y_axis[0]
            y = np.concatenate((y_axis-y_step/2, [y_axis[-1]+y_step/2]))

            plt.figure(figsize = (10,4), dpi = 282) #size in inches, dpi is mine
            plt.pcolormesh(x_axis, y_axis, an)
            plt.colorbar().set_label('real part', rotation=270)
            plt.title('Amplitude sweep in the intersection spot')
            plt.ylabel('Frequency [GHz]')
            plt.xlabel('Amplitude');
            plt.grid()

        except: 
            
            raise Exception("You must 'run' first")

        #return self.g


