

import numpy as np
import scipy.io as scio
import compositional_mi as cmi
import pickle
import time
import multiprocessing as mp

start = time.time()


# where to save the results
mi_output_dir = '/data/nsdm/pyphi/dynamics/mi'

# load dynamics from the output of matlab script: states_visited_grid.m
dynamics_input = '/data/nsdm/pyphi/dynamics'

keep_cores = 2

N = 5
# N = 16

# mi_function = cmi.measure_dte

# mi_functions = [cmi.measure_d2n, cmi.measure_d1]
# mi_functions = [cmi.measure_d2n]
# mi_functions = [cmi.measure_d1]
# mi_functions = [cmi.measure_d1_v2]
# mi_functions = [cmi.measure_dte_v2]
# mi_functions = [cmi.measure_d1_effect, cmi.measure_d1_cause, cmi.measure_d1_mice]
mi_functions = [cmi.measure_d1_mice]

taus = [0, 1, 5]
# taus = [1, 2, 5]
# taus = [0]
# taus = [1]

# mi_output_file = 'grid_rand_bar_shu_taus_0_1_2_5.pickle'
# mi_output_file = 'grid_rand_bar_shu_taus_1_2_5.pickle'
# mi_output_file = 'grid_rand_bar_shu.pickle'

# mi_output_file = 'grid_rand_bar_shu_T_0.5_1.pickle'
# states_files = {'bar_grid': 'all_states_5_nodes_bar_cm_grid_fpp_T_0.5_1.0_11.mat',
#                 'shu_grid': 'all_states_5_nodes_shu_cm_grid_fpp_T_0.5_1.0_11.mat',
#                 'bar_rand': 'all_states_5_nodes_bar_cm_rand_fpp_T_0.5_1.0_11.mat',
#                 'shu_rand': 'all_states_5_nodes_shu_cm_rand_fpp_T_0.5_1.0_11.mat'}

# mi_output_file = 'grid_rand_bar_shu_new.pickle'
# states_files = {'bar_grid': 'all_states_5_nodes_bar_cm_grid_fpp.mat',
#                 'shu_grid': 'all_states_5_nodes_shu_cm_grid_fpp.mat',
#                 'bar_rand': 'all_states_5_nodes_bar_cm_rand_fpp.mat',
#                 'shu_rand': 'all_states_5_nodes_shu_cm_rand_fpp.mat'}

# mi_output_file = 'grid_rand_non_test_T_0.0_2.8_new2.pickle'
# states_files = {'non_grid': 'all_states_5_nodes_non_cm_grid_fpp_T_0.0_2.8_12_new.mat',
#                 'non_rand': 'all_states_5_nodes_non_cm_rand_fpp_T_0.0_2.8_12_new.mat'}

mi_output_file = 'grid_non_N_9_2d_taus_0_1_5.pickle'
states_files = {'non_grid': 'all_states_2d_9_nodes_voi_cm9noinput_T_0.0_4.0_17.mat'}



# mi_output_file = 'grid_non_N_16_taus_0_1.pickle'
# states_files = {'non_grid': 'all_states_16_nodes_cm16selfwrap_T_0.0_1.4_15.mat'}

# mi_output_file = 'grid_non_N_5_taus_0_1.pickle'
# states_files = {'non_grid': 'all_states_5_nodes_cm16selfwrap_T_0.0_1.4_15.mat'}

# bar_grid_states = '%s_grid_grid/all_states_5_nodes_cmGG50_T_0.0_2.8_12.mat'
# shu_grid_states = '%s_grid_grid/all_states_5_nodes_cmSG50_T_0.0_2.8_12.mat'
# bar_rand_states = '%s_grid_grid/all_states_5_nodes_cmGR50_T_0.0_2.8_12.mat'
# shu_rand_states = '%s_grid_grid/all_states_5_nodes_cmSR50_T_0.0_2.8_12.mat'

states = {state:    np.transpose(scio.loadmat(
                        '%s/%s' % (dynamics_input, states_files[state])
                    )['all_states'], (0, 2, 3, 1))
          for state in states_files}


for mi_function in mi_functions:

    # organize output in a dictionary
    mis = {tau: {state: [] for state in states} for tau in taus}
    # for mit in mi_pool:
    #     mis[mit[0][0]][mit[0][1]] = mit[1:2]


    def run_mi_function(params):

        _start = time.time()

        (tau, itemp, irun) = params
        these_states = states[condition][iN, itemp, irun, :]
        if tau:
            _mi = mi_function(these_states, N, tau)
        else:
            _mi = mi_function(these_states, N)

        print(condition,
              ' , tau : ', tau,
              ' , temp : ', itemp+1, ' out of ', ntemp,
              ' , run ', irun+1, ' out of ', nrun,
              ' , took : ', int(time.time() - _start), ' seconds.')

        return itemp, irun, _mi

    for tau in taus:
        for condition in states:

            nstates = states[condition].shape

            iN = 0  # only the 5 nodes system
            nrun = nstates[2]
            ntemp = nstates[1]

            # nrun = 8
            # ntemp = 2

            mi_pool = mp.Pool(mp.cpu_count()-keep_cores).map(run_mi_function,
                                    [(tau, itemp, irun)
                                        for itemp in range(ntemp)
                                        for irun in range(nrun)
                                     ])
            d = np.zeros((ntemp, nrun))
            all = {itemp: {irun: [] for irun in range(nrun)} for itemp in range(ntemp)}
            for mi in mi_pool:
                d[mi[0]][mi[1]] = mi[2][0]
                all[mi[0]][mi[1]] = mi[2]

            d_mean = np.zeros(ntemp)
            d_std = np.zeros(ntemp)
            for (itemp, drun) in enumerate(d):
                d_mean[itemp] = np.mean(drun)
                d_std[itemp] = np.std(drun)

            mis[tau][condition] = (d_mean, d_std, all)

            with open('%s/%s_%s.tmp' % (mi_output_dir, mi_function.__name__,
                                        mi_output_file), 'wb') as f:
                pickle.dump(mis, f)

    # save results
    print('saving results...')
    with open('%s/%s_%s' % (mi_output_dir, mi_function.__name__, mi_output_file),
              'wb') as f:
        pickle.dump(mis, f)
    print('done')

end = time.time()
print('Total running time : ', end='')
print(end - start, end='')
print(' seconds.')
