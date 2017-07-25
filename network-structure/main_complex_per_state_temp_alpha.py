
import numpy as np
import pyphi
import sabrinasfunctions as sf
import itertools
import pickle
import time

start = time.time()

# do_plot = True
do_plot = False

do_one_state = 0
# do_one_state = 1
# do_one_state = 2

results_filename = '/data2/dynacomplex/fivenodes_grid_ENTROPY_WEDGE_10_temp_1_alphas.pickle'
# results_filename = '/data2/dynacomplex/fivenodes_grid_ENTROPY_WEDGE_0.2_temp_10_alphas.pickle'
# results_filename = '/data2/dynacomplex/fivenodes_grid_ENTROPY_WEDGE_0.2_temp_2_alphas.pickle'
# results_filename = '/data2/dynacomplex/fivenodes_grid_ENTROPY_WEDGE_2_temps_3_alphas.pickle'
# results_filename = '/data2/dynacomplex/fivenodes_grid_main_complex_ENTROPY_WEDGE_DIFFSUM_1_temp_11_alphas.pickle'
# results_filename = '/data2/dynacomplex/fivenodes_grid_main_complex_ENTROPY_WEDGE_DIFFSUM_3_temp_11_alphas.pickle'

# results_filename = '/data2/dynacomplex/ASYMMETRIC_STATES_sixnodes_grid_nowrap_main_complex_ENTROPY_WEDGE_DIFFSUM_1_temp_5_alphas.pickle'
# results_filename = '/data2/dynacomplex/sixnodes_grid_nowrap_main_complex_ENTROPY_WEDGE_DIFFSUM_1_temp_11_alphas.pickle'

# pyphi.config.MEASURE = pyphi.constants.EMD
#pyphi.config.PARTITION_MECHANISMS = False

pyphi.config.MEASURE = pyphi.constants.ENTROPY
pyphi.config.PARTITION_MECHANISMS = True
pyphi.config.BIG_DISTANCE = pyphi.constants.DIFFSUM

# pyphi.config.MEASURE = pyphi.constants.ENTROPY
# pyphi.config.PARTITION_MECHANISMS = False

#N = 6
N = 5
d = 3
t = 2
C_BGR_N5 = sf.make_cmgrid(N, d)
C_BGR_N6 = sf.make_cmgrid(N, d, wrap=0)

# temperatures = np.array([4.])
# temperatures = np.array([1, 1.25, 2.25])
#temperatures = np.array([0, .2, 2.5])
#temperatures = np.array([.2])
# temperatures = np.array([0, .2])
# temperatures = np.array([0,])
# temperatures = np.arange(0, 1, .2)
# temperatures = np.arange(0, 1, .1)
temperatures = np.arange(0, 1, .1)

# cuts = [(0, 4), (4, 0), (2, 3), (3, 2)]
# strengths = [0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1]
cuts = [(2, 3), (3, 2)]
# strengths = [0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1]
# strengths = [.1, .3, .5, .7, .9]
# strengths = [0, .3, .6, .9]
# strengths = [0, 1]
strengths = [1]
TPM_G_N5 = dict()
# TPM_G_N6 = dict()
for temp in temperatures:

    print('Temperature %1.2f' % temp)
    TPM_G_N5[temp] = dict()
    # TPM_G_N6[temp] = dict()
    for strength in strengths:

        print('Strengh %1.2f' % strength)
        these_cuts = [(c[0], c[1], strength) for c in cuts]
        TPM_G_N5[temp][strength] = sf.make_tpm(N, C_BGR_N5, t, temp, these_cuts)
        # TPM_G_N6[temp][strength] = sf.make_tpm(N, C_BGR_N6, t, temp, these_cuts)


results_grid = dict()
results_cut = dict()
for (itemp, temp) in enumerate(temperatures):

    print('Temperature %1.2f (%d of %d)' % (temp, itemp+1, len(temperatures)))
    results_grid[temp] = dict()
    for strength in strengths:

        print('\n\nStrength %1.2f\n\n' % strength)
        cm = C_BGR_N5
        tpm = TPM_G_N5[temp][strength]
        # cm = C_BGR_N6
        # tpm = TPM_G_N6[temp][strength]

        results_grid[temp][strength] = dict()

        network = pyphi.Network(tpm, connectivity_matrix=cm)

        # this_grid = dict()
        # n_concepts = nchoosek(5,1) + nchoosek(5,2) + nchoosek(5,3) + nchoosek(5,4) + nchoosek(5,5) == 31
        # n_concepts = 32
        # this_grid_big = (np.zeros((2 ** N)), np.zeros((n_concepts, 2 ** N)))

        all_states = list(itertools.product((0, 1), repeat=N))

        if do_one_state:
            if do_one_state > 1:
                print('WARNING doing only NON SYMETRIC')
                # a = all_states
                # all_states = a[0:8] + a[9:16] + a[17:24] + a[25:26] + a[27:28] + a[29:32] + a[33::2]

                all_states_o = list(itertools.product((0, 1), repeat=N))
                all_states = []
                all_states_i = []
                for i in range(2 ** N):
                    n = ("{0:0" + str(N) + "b}").format(i)
                    rn = n[::-1]
                    rs = int(rn, 2)
                    if rs in all_states_i:
                        continue
                    all_states.append(all_states_o[i])
                    all_states_i.append(i)
                    print('%d : %s' % (i, ("{0:0"+str(N)+"b}").format(i)))

            else:
                print('WARNING doing only ONE state')
                all_states = [all_states[0]]

        for (istate, state) in enumerate(all_states):

            print('')
            print('State : %s (%d of %d)' % (state, istate, len(all_states)))

            try:
                start_one = time.time()

                the_grid_main = pyphi.compute.main_complex(network, state)

                print("{} ({} concepts)".format(the_grid_main.phi, len(the_grid_main.unpartitioned_constellation)))
                print(the_grid_main.subsystem)
                print(the_grid_main.cut)
                results_grid[temp][strength][state] = (cm, tpm, the_grid_main)
                the_grid_main.subsystem.clear_caches()

                print("--- %s seconds ---" % (time.time() - start_one))
            except pyphi.exceptions.StateUnreachableError:
                print('INVALID')
                results_grid[temp][strength][state] = ()
                # this_grid[state] = []
                # this_grid_big[0][istate] = -1
                # this_grid_big[1][:, istate] = -1
                continue

                #             print ('\t\{} ({} out of {}) : '.format(state, istate+1, 2**N), end='')

                #             print ('{:1.5f} ({:1.5f} * {:1.5f} ({} concepts))'.format(the_grid_main.phi * sum(concepts_bar),
                #                                                                        the_grid_main.phi, sum(concepts_bar),
                #                                                                         len(concepts_bar) ))

            with open(results_filename + '.tmp', 'wb') as f:
                pickle.dump((results_grid, results_cut), f)

        print('')

with open(results_filename, 'wb') as f:
    pickle.dump((results_grid,results_cut,strengths), f)

end = time.time()
print('Executiong time : ', end = '')
print(end - start, end='')
print(' seconds.')



