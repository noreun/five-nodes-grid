import makegridslibrary as me
import scipy.io as scio
import numpy as np
import pyphi
import pickle

pyphi.config.PARALLEL_CONCEPT_EVALUATION = True

with open('/data/nsdm/pyphi/C_2d_9_matrix.mat', 'rb') as f:
    CR_mat = scio.loadmat(f)
C_BGR_N9_2D = CR_mat['CG']

T_G_BGR_N9_2D = me.make_tpm_gibbs(9, C_BGR_N9_2D, 3, [], 4)

network = pyphi.Network(T_G_BGR_N9_2D, connectivity_matrix=C_BGR_N9_2D)
the_grid = pyphi.Subsystem(network, (0,0,0,0,0,0,0,0,0), range(9))

from pyphi.compute.concept import constellation

# mip_the_grid = pyphi.compute.big_mip(the_grid)

