from itertools import chain

from transmon_chain.TransmonChain import TransmonChain
from tqdm import tqdm

import matplotlib
from matplotlib import ticker, colorbar as clb
matplotlib.use('Qt5Agg')
from matplotlib.pyplot import *
from numpy import *


chain = TransmonChain(length = 5,
                      transmon_truncation=3,
                      transmon_diagonalization_dimension=15,
                      )

chain.set_Ec(0.25*2*pi)
chain.set_Ej(20*2*pi)
chain.set_asymmetry(0.7)
chain.set_J(.05*2*pi)
chain.set_phi(0)
chain.set_gamma_phi(1e-4)
chain.set_gamma_rel(1e-4)

chain.build_transmons()
chain.build_low_energy_kets(4)

H_chain = chain.build_H_full()
H_chain = chain.truncate_to_low_population_subspace(H_chain)

evals, evecs = H_chain.eigenstates()

plot((evals-evals[0])/2/pi, ".")
plot(ones_like(evals)-0.05*len(evals), (evals-evals[0])/2/pi, ".")
xlabel("State N")
ylabel("Energy [GHz]")
show()

print()


exit()

levels = []

phis = linspace(0, .5, 50)
for phi in tqdm(phis):
    phi_arr = [0.25, .25 , .25, .25, .25]
    phi_arr[0] = phi
    chain.set_phi(phi_arr)
    H_chain = chain.build_H_full()
    evals, evecs = H_chain.eigenstates()
    levels.append((evals-evals[0])/2/pi)

figure()
plot(array(levels)[:, 1:6])
plot(array(levels)[:, 6:21]/2, "--")
# plot(array(levels)[:, 6:16]/2, "--")

show()
