
import oldphi
import numpy as np
import pyphi
import makegridslibrary as me
import itertools
import pickle
import time

pyphi.config.MEASURE = pyphi.constants.EMD
pyphi.config.PARTITION_MECHANISMS = False

start = time.time()

N = 9
d = 3

C_BGR_N9 = me.make_cmgrid_withbackground(N, d)
R_BGR_N9 = me.make_random_withbackground(N, d)

thresh = 2


# short_version = 0 # all temps and states
short_version = 1  # only 1 temp and bar input, all states
# short_version = 2 # only 3 tempereatus anmd few states

# do_plot = True
do_plot = False

if short_version == 1:
    temperatures = np.array([0,])
elif short_version == 2:
    temperatures = np.arange(0, 1.1, .5)
else:
    temperatures = np.arange(0, 1.1, .1)

if short_version == 1 or short_version == 0:
    bar_inputs = [(0, 0, 0, 1, 1)]
    shuffled_inputs = [(0, 0, 1, 0, 1)]
else:
    bar_inputs = [(0, 0, 0, 1, 1), (0, 0, 1, 1, 0), (0, 1, 1, 0, 0),
                  (1, 0, 0, 0, 1)]
    shuffled_inputs = [(0, 0, 1, 0, 1), (1, 0, 0, 1, 0), (0, 1, 0, 0, 1),
                       (1, 0, 1, 0, 0)]
print(bar_inputs)
print(shuffled_inputs)

results_bar = dict()
results_shuffled = dict()

big_results_bar = dict()
big_results_shuffled = dict()

network_labels = ['Grid', 'Random']

T_C_BGR_N9 = dict()
T_R_BGR_N9 = dict()
for temp in temperatures:
    print('Temperature %1.2f' % temp)
    T_C_BGR_N9[temp] = me.make_tpm_withbackground(N, C_BGR_N9, thresh, temp)
    T_R_BGR_N9[temp] = me.make_tpm_withbackground(N, R_BGR_N9, thresh, temp)


bar_state = [0,1,1,1,0,0,0,0,1,1]
# bar_state = [1,1,0,1,0,0,0,0,1,1]

# myphi = oldphi
myphi = pyphi

network = myphi.Network(T_C_BGR_N9[0], connectivity_matrix=C_BGR_N9)
thegrid = myphi.Subsystem(network, tuple([0]*18), range(N))
m1 = thegrid.effect_repertoire((0,1), (0,1,2))

# mip_bar = myphi.compute.big_mip(thegrid)

