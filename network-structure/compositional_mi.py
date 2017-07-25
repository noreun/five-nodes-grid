

"""
=====
compositional mutual information
=====

Degeneracy-like compositional mutual information measures.

These functions try to implement approximations of \Phi by measuring
how much the system states depends on the states of each scale in the sytem.
Otherwise saying, how much each scale "causes" (in the statistical sense) the
system to be in a particular state.

References:

    * Tononi, G., Sporns, O., & Edelman, G. M. (1999). Measures of degeneracy
    and redundancy in biological networks.
    Proceedings of the National Academy of Sciences, 96(6), 3257–3262.

    * Oizumi, M., Tsuchiya, N., & Amari, S. (2016). Unified framework for
    information integration based on information geometry.
    Proceedings of the National Academy of Sciences, 113(51), 14817–14822.


"""

import itertools
import numpy as np
from sklearn.metrics import mutual_info_score
from operator import or_
from functools import reduce


# Calculates the mutual information between the whole system X and itself.
# Then extrapolates the expected mutual information of a non-integrated system
# as being composed by the sum of smaller subsets of size K and its complement:
#
#    MI_k + MI_{\simk}
#
# where
#
#    MI_k = <MI(X_j^k; X)>_j
#
#    and
#
#    MI_{\simk} = <MI(X-X_j^k; x)>_j
#
# Assuming that the system is integrated, integrated information (the whole is
# more than the sum of subsets of size k) should be:
#
#    D2N = \sum_{k=0}^(N/2)  M_k + M_{\simk} - MI(X,X)
#
# Expected values:
#
#   states : ndarray of size T, where each position is a state in time t, and
#               each bit on or off correspond to a binary unit on or of
#
#   N      : number of unities in the system (maximum state number should
#               be 2**N)
#
#   tau    : lag between the time series used to calculate the MI of the whole
#               system and the subsets of size k. This way is is possible to
#               consider how subsets of size k in the paste X_k(t - \tau)
#               constrain the whole system in the future X(t) (default tau = 0)
#
# Returns:
#
#   D2N, M_k, M_\simk}, MI(X,X)
#
def measure_d2n(states, N, tau=0):

    # N2 = int(np.ceil(N/2))
    N2 = int(N/2)

    # MI(X; O)
    full_m = mutual_info_score(states[tau:], states[:-tau or None]) / np.log(2)

    dn = np.zeros(N2)
    mn = np.zeros(N2)
    mcn = np.zeros(N2)
    for n in range(N2):

        # take the mean of MI between all sets of size n+1 and the system
        possible_elements = list(itertools.combinations(range(N), n + 1))
        for nn in possible_elements:

            # MI(X_j^k; O)
            mask = reduce(or_, [0b1 << element for element in nn])
            statesn = [s & mask for s in states]
            m = mutual_info_score(states[tau:], statesn[:-tau or None]) / np.log(2)

            # MI(X-X_j^k; O)
            nn_complement = list(set(range(N)) - set(nn))
            mask = reduce(or_, [0b1 << element for element in nn_complement])
            statesnc = [s & mask for s in states]
            mc = mutual_info_score(states[tau:], statesnc[:-tau or None]) / np.log(2)

            # D = MI(X_j^k; O) + MI(X-X_j^k; O) - MI(X; O)
            dn[n] += m + mc - full_m

            # keep track for understanding the behaviour of the measure
            mn[n] += m
            mcn[n] += mc

        dn[n] /= len(possible_elements)
        mn[n] /= len(possible_elements)
        mcn[n] /= len(possible_elements)

    # sum the contribution os each subset
    return np.sum(dn), mn, mcn, full_m


# Calculates the tranfer entropy between one unit and the whole system
#
#    MI_1 = <MI(X_j^1; X)>_j
#
#  Then extrapolates the expected tranfer entropy of a larger sub-set of
# size K in the case of independent unities as being multiples of the
#  mutual information of a single unit. Assuming that the system is
# integrated, integrated information (the whole is more than the sum of
# subsets of size k) should be:
#
#    D1 = \sum_{k=0}^N  k * MI_1 - <MI(X_j^k; X)>_j
#
# Expected values:
#
#   states : ndarray of size T, where each position is a state in time t, and
#               each bit on or off correspond to a binary unit on or of
#
#   N      : number of unities in the system (maximum state number should
#               be 2**N)
#
#   tau    : lag between the time series used to calculate the MI of the whole
#               system and the subsets of size k. This way is is possible to
#               consider how subsets of size k in the paste X_k(t - \tau)
#               constrain the whole system in the future X(t) (default tau = 0)
#
# Returns:
#
#   D1N, <MI(X_j^k; X)>_j
#
def measure_d1_effect(states, N, tau=0, max_sample=10):

    from numpy.random import choice

    mn = np.zeros(N)
    dn = np.zeros(N)
    for n in range(0, N):

        # take the mean of MI between all sets of size n+1 and the system
        possible_elements = list(itertools.combinations(range(N), n + 1))
        ncomb = len(possible_elements)

        # sample maximum of max_sample mecanims
        if ncomb > max_sample:
            possible_elements = [possible_elements[i]
                                 for i in choice(ncomb, max_sample, False)]

        all_mn = dict()
        for elements in possible_elements:
            mask = reduce(or_, [0b1 << element for element in elements])
            statesn = [s & mask for s in states]
            all_mn[elements] = mutual_info_score(states[tau:], statesn[:-tau or None]) / np.log(2)
            mn[n] += all_mn[elements]
        mn[n] /= ncomb

        # subtracted the expected for independent system
        dn[n] = (n + 1) * mn[0] - mn[n]

    # sum the contribution os each subset
    return np.sum(dn), mn, all_mn


def measure_d1_cause(states, N, tau=0, max_sample=10):

    from numpy.random import choice

    mn = np.zeros(N)
    dn = np.zeros(N)
    for n in range(0, N):

        # take the mean of MI between all sets of size n+1 and the system
        possible_elements = list(itertools.combinations(range(N), n + 1))
        ncomb = len(possible_elements)

        # sample maximum of max_sample mecanims
        if ncomb > max_sample:
            possible_elements = [possible_elements[i]
                                 for i in choice(ncomb, max_sample, False)]

        all_mn = dict()
        for elements in possible_elements:
            mask = reduce(or_, [0b1 << element for element in elements])
            statesn = [s & mask for s in states]
            all_mn[elements] = mutual_info_score(states[:-tau or None], statesn[tau:]) / np.log(2)
            mn[n] += all_mn[elements]
        mn[n] /= ncomb

        # subtracted the expected for independent system
        dn[n] = (n + 1) * mn[0] - mn[n]

    # sum the contribution os each subset
    return np.sum(dn), mn, all_mn

def measure_d1_mice(states, N, tau=0, max_sample=10):

    from numpy.random import choice

    mn = np.zeros(N)
    dn = np.zeros(N)
    for n in range(0, N):

        # take the mean of MI between all sets of size n+1 and the system
        possible_elements = list(itertools.combinations(range(N), n + 1))
        ncomb = len(possible_elements)

        # sample maximum of max_sample mecanims
        if ncomb > max_sample:
            possible_elements = [possible_elements[i]
                                 for i in choice(ncomb, max_sample, False)]

        all_mn = dict()
        for elements in possible_elements:
            mask = reduce(or_, [0b1 << element for element in elements])
            statesn = [s & mask for s in states]
            m_effect = mutual_info_score(states[tau:], statesn[:-tau or None]) / np.log(2)
            m_cause = mutual_info_score(states[:-tau or None], statesn[tau:]) / np.log(2)
            all_mn[elements] = np.min([m_effect, m_cause])
            mn[n] += all_mn[elements]
        mn[n] /= ncomb

        # subtracted the expected for independent system
        dn[n] = (n + 1) * mn[0] - mn[n]

    # sum the contribution os each subset
    return np.sum(dn), mn, all_mn

def measure_dte_v2(states, N, tau=0, max_sample=10):

    # first first order
    mn = np.zeros(N)
    for element in range(N):
        mask = 0b1 << element
        states1 = [s & mask for s in states]
        statesn1 = [s & ~mask for s in states]
        mn[0] += mutual_info_score(statesn1[tau:], states1[:-tau or None]) / np.log(2)
    mn[0] /= N

    dn = np.zeros(N)
    for n in range(1, N):

        # take the mean of MI between all sets of size n+1 and the system
        possible_elements = list(itertools.combinations(range(N), n + 1))
        ncomb = len(possible_elements)

        # sample maximum of max_sample mecanims
        if ncomb > max_sample:
            possible_elements = \
                [possible_elements[i] for i in np.random.randint(0, ncomb, max_sample)]

        all_mn = dict()
        for elements in possible_elements:
            mask = reduce(or_, [0b1 << element for element in elements])
            statesn = [s & mask for s in states]
            statesnn = [s & ~mask for s in states]
            all_mn[elements] = mutual_info_score(statesnn[tau:], statesn[:-tau or None]) / np.log(2)
            mn[n] += all_mn[elements]

        mn[n] /= ncomb

        # subtracted the expected for independent system
        dn[n] = (n + 1) * mn[0] - mn[n]

    # sum the contribution os each subset
    return np.sum(dn), mn, all_mn

