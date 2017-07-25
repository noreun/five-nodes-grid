
import numpy as np
import pyphi
import makegridslibrary as me
import itertools
import pickle
import time
import scipy.io as scio

# pyphi.config.MEASURE = pyphi.constants.EMD
#pyphi.config.PARTITION_MECHANISMS = False

pyphi.config.MEASURE = pyphi.constants.ENTROPY
#pyphi.config.MEASURE = pyphi.constants.L2

pyphi.config.PARTITION_MECHANISMS = True
pyphi.config.BIG_DISTANCE = pyphi.constants.DIFFSUM

#pyphi.config.PARALLEL_CONCEPT_EVALUATION = True
#pyphi.config.PARALLEL_CUT_EVALUATION = False

# pyphi.config.MEASURE = pyphi.constants.ENTROPY
# pyphi.config.PARTITION_MECHANISMS = False


pyphi.config.COMPOSITIONAL_CONCEPT = True
# pyphi.config.COMPOSITIONAL_CONCEPT = False

pyphi.config.LOGGING_CONFIG['file']['level'] = 'INFO'

start = time.time()

N = 5
d = 3

# C_BGR_N5 = me.make_cmgrid_withbackground(N, d)
# R_BGR_N5 = me.make_random_withbackground(N, d)

# load previously saved to avoid variance in results
with open('/data/nsdm/pyphi/C_matrix.mat', 'rb') as f:
    CR_mat = scio.loadmat(f)
C_BGR_N5 = CR_mat['CG']
R_BGR_N5 = CR_mat['CR']

thresh = 2

# results_filename = '/data/nsdm/pyphi/fivenodes_barVSshuffled_gridVSrandom.pickle'

#results_filename = '/data/nsdm/pyphi/fivenodes_barVSshuffled_gridVSrandom_new_code.pickle'
#results_filename = '/data/nsdm/pyphi/fivenodes_barVSshuffled_gridVSrandom_ENTROPY_WEDGE.pickle'
#results_filename = '/data/nsdm/pyphi/fivenodes_barVSshuffled_gridVSrandom_ENTROPY_WEDGE_DIFFSUM_D_3.pickle'
#results_filename = '/data/nsdm/pyphi/fivenodes_barVSshuffled_gridVSrandom_ENTROPY_WEDGE_DIFFSUM_D_2_MAIN.pickle'
#results_filename = '/data/nsdm/pyphi/fivenodes_barVSshuffled_gridVSrandom_ENTROPY_WEDGE_DIFFSUM_T_0_3_D_2_MAIN.pickle'
#results_filename = '/data/nsdm/pyphi/fivenodes_barVSshuffled_gridVSrandom_ENTROPY_WEDGE_DIFFSUM_T_0_3_D_2_MAIN_SAVED.pickle'
#results_filename = '/data/nsdm/pyphi/fivenodes_barVSshuffled_gridVSrandom_ENTROPY_WEDGE_DIFFSUM_T_0_1_D_2_MAIN_SAVED.pickle'
# results_filename = '/data/nsdm/pyphi/fivenodes_barVSshuffled_gridVSrandom_ENTROPY_WEDGE_DIFFSUM_T_0_1_D_2_MAIN_SAVED_RANDOM.pickle'
# results_filename = '/data/nsdm/pyphi/fivenodes_barVSshuffled_gridVSrandom_ENTROPY_WEDGE_DIFFSUM_T_.5_1_D_2_MAIN_SAVED.pickle'

# results_filename = '/data/nsdm/pyphi/fivenodes_noinput_gridVSrandom_ENTROPY_WEDGE_DIFFSUM_T_0_3_D_2_MAIN_SAVED_new.pickle'
# results_filename = '/data/nsdm/pyphi/fivenodes_noinput_gridVSrandom_ENTROPY_WEDGE_DIFFSUM_T_0.5_1.0_D_2_MAIN_SAVED.pickle'
# results_filename = '/data/nsdm/pyphi/fivenodes_noinput_gridVSrandom_ENTROPY_WEDGE_DIFFSUM_T_0.5_1.0_D_2_L2_MAIN_SAVED.pickle'

# results_filename = '/data/nsdm/pyphi/fivenodes_noinput_gridVSrandom_ENTROPY_WEDGE_DIFFSUM_T_.75_D_2_MAIN_SAVED_COMPOSITE_1_state_old.pickle'
# results_filename = '/data/nsdm/pyphi/fivenodes_noinput_gridVSrandom_ENTROPY_WEDGE_DIFFSUM_T_.75_D_2_MAIN_SAVED_COMPOSITE_1_state.pickle'
results_filename = '/data/nsdm/pyphi/fivenodes_noinput_gridVSrandom_ENTROPY_WEDGE_DIFFSUM_T_.75_1.5_D_2_MAIN_SAVED_COMPOSITE_new.pickle'

# results_filename = '/data/nsdm/pyphi/fivenodes_barVSshuffled_gridVSrandom_1temp_1input_new_old_cut.pickle'
# results_filename = '/data/nsdm/pyphi/fivenodes_barVSshuffled_gridVSrandom_new_measure.pickle'
# results_filename = '/data/nsdm/pyphi/fivenodes_barVSshuffled_gridVSrandom_5temps_new_measure_old_cut.pickle'
# results_filename = '/data/nsdm/pyphi/fivenodes_barVSshuffled_gridVSrandom_1temp_1input_new_measure.pickle'

short_version = 0 # all temps and states
# short_version = 1  # only 1 temp and bar input, all states
# short_version = 2 # only 1 tempereatus and 1 state

# do_plot = True
do_plot = False

# force_full_system = True
force_full_system = False

if short_version == 1:
    temperatures = np.array([0,])
elif short_version == 2:
    temperatures = np.array([.75,])
else:
    # temperatures = np.arange(0, 3, .25)
    # temperatures = np.arange(0, 1.75, .4)
    # temperatures = np.arange(0, 1.5, .1)
    # temperatures = np.arange(.5, 1.01, .05)
    # temperatures = np.arange(.75, 1.51, .25)
    temperatures = np.array([.75,])

bar_inputs = [(0, 0, 0, 0, 0)]
shuffled_inputs = []
# bar_inputs = [(0, 0, 0, 1, 1)]
# shuffled_inputs = [(0, 0, 1, 0, 1)]

print(bar_inputs)
print(shuffled_inputs)

results_bar = dict()
results_shuffled = dict()

big_results_bar = dict()
big_results_shuffled = dict()

network_labels = ['Grid', 'Random']

#network_labels = ['Grid']
#network_labels = ['Random']

T_C_BGR_N5 = dict()
T_R_BGR_N5 = dict()
for temp in temperatures:
    print('Temperature %1.2f' % temp)
    # T_C_BGR_N5[temp] = me.make_tpm_withbackground(N, C_BGR_N5, thresh, temp)
    # T_R_BGR_N5[temp] = me.make_tpm_withbackground(N, R_BGR_N5, thresh, temp)
    T_C_BGR_N5[temp] = me.make_tpm_withbackground_gibbs(N, C_BGR_N5, temp, [], thresh)
    T_R_BGR_N5[temp] = me.make_tpm_withbackground_gibbs(N, R_BGR_N5, temp, [], thresh)

for network_label in network_labels:

    print('\n\nMatching for %s \n\n' % network_label)

    if network_label == 'Grid':
        tpms = T_C_BGR_N5
        cm = C_BGR_N5
    elif network_label == 'Random':
        tpms = T_R_BGR_N5
        cm = R_BGR_N5
    else:
        raise ValueException('unknown network %s' % network_label)

    results_bar[network_label] = dict()
    results_shuffled[network_label] = dict()

    big_results_bar[network_label] = dict()
    big_results_shuffled[network_label] = dict()

    for (itemp, temp) in enumerate(temperatures):

        print('Temperature %1.2f (%d of %d)' % (temp, itemp, len(temperatures)))

        print('\tFinding matching for the structured')

        network = pyphi.Network(tpms[temp], connectivity_matrix=cm)

        big_results_bar[network_label][temp] = dict()
        big_results_shuffled[network_label][temp] = dict()

        this_bar = dict()
        for (ibar, bar_input) in enumerate(bar_inputs):

            # n_concepts = nchoosek(5,1) + nchoosek(5,2) + nchoosek(5,3) + nchoosek(5,4) + nchoosek(5,5) == 31
            n_concepts = 32
            this_bar_big = [np.zeros((2 ** N)), np.zeros((n_concepts, 2 ** N)), {}]

            all_states = list(itertools.product((0, 1), repeat=N))

            if short_version == 2:
                # all_states = [all_states[x] for x in
                #               np.random.randint(1, 2 ** N, 2)]
                all_states = [all_states[0],]

            print('\t\tBar {} out of {} ({} states)): '.format(ibar + 1,
                                                              len(bar_inputs),
                                                              len(all_states)),
                  end='')

            for (istate, this_state) in enumerate(all_states):

                startc = time.time()

                print('{} '.format(istate + 1), end='')
                bar_state = this_state + bar_input
                try:
                    if force_full_system:
                        thegrid = pyphi.Subsystem(network, bar_state, range(N))
                        mip_bar = pyphi.compute.big_mip(thegrid)
                    else:
                        mip_bar = pyphi.compute.main_complex(network, bar_state)

                    concepts_bar = [x.phi for x in
                                    mip_bar.unpartitioned_constellation]

                    # save simple output: big phi and small phis
                    this_bar[bar_state] = (mip_bar.phi, concepts_bar)

                    # save big output: small phi per state and mechanism
                    this_bar_big[0][istate] = mip_bar.phi
                    for x in mip_bar.unpartitioned_constellation:

                        mechanism_str = list('0' * N)
                        mechanism = x.mechanism
                        for y in mechanism:
                            mechanism_str[y] = '1'
                        mechanism_str.reverse()
                        iconcept = int("".join(mechanism_str), 2)
                        # print(mechanism, end='')
                        # print(' : ', iconcept)
                        this_bar_big[1][iconcept, istate] = x.phi

                    this_bar_big[2][istate] = mip_bar

                except pyphi.exceptions.StateUnreachableError:
                    this_bar[bar_state] = (-1, [])
                    this_bar_big[0][istate] = -1
                    this_bar_big[1][:, istate] = -1
                    this_bar_big[2][istate] = []
                    continue

                    #             print ('\t\{} ({} out of {}) : '.format(bar_state, istate+1, 2**N), end='')

                    #             print ('{:1.5f} ({:1.5f} * {:1.5f} ({} concepts))'.format(mip_bar.phi * sum(concepts_bar),
                    #                                                                        mip_bar.phi, sum(concepts_bar),
                    #                                                                         len(concepts_bar) ))

                endc = time.time()
                print(endc - startc, end='')
                print(', ', end='')

            print('')

            big_results_bar[network_label][temp][bar_input] = this_bar_big

        results_bar[network_label][temp] = this_bar

        print('\tFinding matching for the shuffled')
        this_shuffled = dict()
        for (ishuffled, shuffled_input) in enumerate(shuffled_inputs):

            all_states = list(itertools.product((0, 1), repeat=N))
            this_shuffled_big = [np.zeros((2 ** N)), np.zeros((n_concepts, 2 ** N)), {}]

            if short_version == 2:
                # all_states = [all_states[x] for x in
                #               np.random.randint(1, 2 ** N, 2)]
                all_states = [all_states[0],]

            print(
                '\t\tShufled {} out of {} ({} states)): '.format(ishuffled + 1,
                                                                 len(
                                                                     shuffled_inputs),
                                                                 len(
                                                                     all_states)),
                end='')

            for (istate, this_state) in enumerate(all_states):

                print('{}, '.format(istate + 1), end='')
                shuffled_state = this_state + shuffled_input

                try:
                    if force_full_system:
                        thegrid = pyphi.Subsystem(network, shuffled_state, range(N))
                        mip_shuffled = pyphi.compute.big_mip(thegrid)
                    else:
                        mip_shuffled = pyphi.compute.main_complex(network, shuffled_state)

                    concepts_shuffled = [x.phi for x in
                                         mip_shuffled.unpartitioned_constellation]
                    this_shuffled[shuffled_state] = (
                    mip_shuffled.phi, concepts_shuffled)

                    # save big output: small phi per state and mechanism
                    this_shuffled_big[0][istate] = mip_shuffled.phi
                    for x in mip_shuffled.unpartitioned_constellation:

                        mechanism_str = list('0' * N)
                        mechanism = x.mechanism
                        for y in mechanism:
                            mechanism_str[y] = '1'
                        mechanism_str.reverse()
                        iconcept = int("".join(mechanism_str), 2)
                        # print(mechanism, end='')
                        # print(' : ', iconcept)
                        this_shuffled_big[1][iconcept, istate] = x.phi

                    this_shuffled_big[2][istate] = mip_shuffled


                except pyphi.exceptions.StateUnreachableError:
                    this_shuffled[shuffled_state] = (-1, [])
                    this_shuffled_big[0][istate] = -1
                    this_shuffled_big[1][:, istate] = -1
                    this_shuffled_big[2][istate] = []
                    continue

                    #             print ('\t\t{} ({} out of {}) : '.format(shuffled_state, istate+1, 2**N), end='')

                    #             print ('{:1.5f} ({:1.5f} * {:1.5f} ({} concepts))'.format(mip_shuffled.phi * sum(concepts_shuffled),
                    #                                                                        mip_shuffled.phi, sum(concepts_shuffled),
                    #                                                                         len(concepts_shuffled) ))

            print('')
            big_results_shuffled[network_label][temp][shuffled_input] = this_shuffled_big

        results_shuffled[network_label][temp] = this_shuffled

        pickle.dump((results_bar, results_shuffled),
                    open(results_filename + '.tmp', 'wb'))

pickle.dump((results_bar, results_shuffled, big_results_bar, big_results_shuffled, C_BGR_N5, R_BGR_N5, temperatures), open(results_filename, 'wb'))

end = time.time()
print('Executiong time : ', end = '')
print(end - start, end='')
print(' seconds.')



