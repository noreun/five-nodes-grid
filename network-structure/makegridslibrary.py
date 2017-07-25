
# coding: utf-8

# In[1]:
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from operator import itemgetter
import scipy 
import scipy.misc as sc
import itertools
import random
import pyphi

#make random networks of size N with connection d with self-connections
def make_random(N,d): 
    
    A=np.identity(N)
    zaehl=(d-1)*np.ones(N)
   
    i=0
    count=0
    while i<N and count<20:
        distr=np.zeros(N)
        for x in zip(range(N), itertools.count()):
            distr[x[1]]=x[1]
        distr3=distr    
        distr3[distr3==i]=-1
        distr3[zaehl==0]=-1
        distr2=distr3[distr3>-1]
        ndistr2un=len(distr2)
        if ndistr2un<d-1 and count<20:
            count=count+1
        else:
            v=np.array(random.sample(range(ndistr2un),d-1))
            zaehl2=zaehl
            zaehl2[distr2[v].astype(int)]=zaehl[distr2[v].astype(int)]-1
            zaehl=zaehl2
            A[i,distr2[v].astype(int)]=1
            i=i+1
        if count>=20:
            print("reseting...")
            count=0
            i=0
            A=np.identity(N)
            zaehl=(d-1)*np.ones(N)

    return A

make_random(10,3)

#---------------------
def make_random_regoutput(N,d): 
    
    A=np.identity(N)
    zaehl=(d-1)*np.ones(N)
   
    i=0
    
    for i in range(N):
        z = np.arange(N)
        z=np.delete(z, i)
        
        zl=list(z)
        zz=random.sample(zl,d-1)
        
        A[i,zz]=1
        
        
    return A
#----------------------

def make_random_reginput(N,d): 
    
    A=np.identity(N)
    zaehl=(d-1)*np.ones(N)
   
    i=0
    
    for i in range(N):
        z = np.arange(N)
        z=np.delete(z, i)
        
        zl=list(z)
        zz=random.sample(zl,d-1)
        
        A[zz,i]=1
        
        
    return A
        

#----------------------
def make_cmgrid(N,d):
    #connect to both of your neighbors
    cm=np.identity(N);
    
    k=int((d-1)/2)
    
    for i in range(N):
        if i+k>N-1:
            l=i+k-N
            cm[i,i:N]=1
            cm[i,0:l+1]=1
        else:
            cm[i,i:i+k+1]=1
        
        if i-k<0:
            l=i-k
            cm[i,0:i]=1
            cm[i,N+l:N]=1
        else:
            cm[i,i-k:i]=1
        
    return cm

#----------------------

def make_cmgrid_k(N,k):
    #connect to k neighbors
    cm=np.identity(N);

    for i in range(N):

        # to the right
        for ik in range(1,k+1):

            # next
            l = i + ik

            # if border, wrap
            if l >= N:
                l -= N

            cm[i,l] = 1

            # preivous
            l = i - ik

            # if border, wrap
            if l < 0:
                l += N

            cm[i,l] = 1

    return cm

#----------------------

#create the tpm
def make_tpm(N, cm,thresh):
    M = 2**N
    tpm = np.zeros([M, N])
    for i in range(M):
        state = pyphi.convert.loli_index2state(i, N)
        for node in range(N):
            tpm[i, node] = np.dot(cm.transpose()[node], state) >= thresh
            
    return tpm

#----------------------

def combinations(iterable, r):
    # combinations('ABCD', 2) --> AB AC AD BC BD CD
    # combinations(range(4), 3) --> 012 013 023 123
    pool = tuple(iterable)
    n = len(pool)
    if r > n:
        return
    indices = list(range(r))
    yield tuple(pool[i] for i in indices)
    while True:
        for i in reversed(range(r)):
            if indices[i] != i + n - r:
                break
        else:
            return
        indices[i] += 1
        for j in range(i+1, r):
            indices[j] = indices[j-1] + 1
        yield tuple(pool[i] for i in indices)


#----
def all_states(n):
    """Return all binary states for a system.
    Args:
        n (int): The number of elements in the system.
    Yields:
        tuple[int]: The next state of an ``n``-element system, in LOLI order.
    """
    if n == 0:
        return

    for state in itertools.product((0, 1), repeat=n):
        yield state[::-1]  # Convert to LOLI-ordering


#---------
#output=all reachable states given a connectivity matrix and a threshold
def all_reachable_states(cm,thresh):
    N=len(cm)
    S=2**N
    Op=np.zeros(shape=(N,2**N))
    for i, state in enumerate(pyphi.utils.all_states(N)):
        statearray=np.asarray(state)
        Op[:,i]=np.dot(cm.transpose(), statearray) >= thresh

    b = np.ascontiguousarray(Op.transpose()).view(np.dtype((np.void, Op.transpose().dtype.itemsize * Op.transpose().shape[1])))
    _, idx = np.unique(b, return_index=True)
    Op=Op[:,idx]

    #how many states are realizable
    real=len(Op.transpose())
    print('only', real, 'out of ',S, 'states are possible')

    return Op

#----------


#output=all reachable states given a connectivity matrix and a threshold
def all_reachable_states_random(cm):
    N=len(cm)
    S=2**N
    Op=np.zeros(shape=(N,2**N))
    scm=sum(cm)
    thresh=ceil(scm/2)
    
    for i, state in enumerate(pyphi.utils.all_states(N)):
        statearray=np.asarray(state)
        Op[:,i]=np.dot(cm.transpose(), statearray) >= thresh

    b = np.ascontiguousarray(Op.transpose()).view(np.dtype((np.void, Op.transpose().dtype.itemsize * Op.transpose().shape[1])))
    _, idx = np.unique(b, return_index=True)
    Op=Op[:,idx]

    #how many states are realizable
    real=len(Op.transpose())
    print('only', real, 'out of ',S, 'states are possible')

    return Op


########--------
#create two 1d grids connected by a bridge
def make_cmtwogrid(N,d):
    #connect to both of your neighbors
    l=int(N/2)
    a=make_cmgrid(l,d)
    b=np.zeros(shape=(N,N))
    
    for i in range(l):
        b[i][0:l]=a[i][0:l]
        b[i+l][l:N]=a[i][0:l]
    
    #craete the bridge
    r1=int(np.floor(random.uniform(0,l)))
    r2=int(np.floor(random.uniform(l,N)))
    b[r1][r2]=1
    find=np.nonzero(b[r2])
    
    b[find[0][1]][r2]=0
    
    if r1>0:
        b[r1][r1-1]=0
        b[find[0][1]][r1-1]=1
    else:
        b[r1][r1+1]=0
        b[find[0][1]][r1+1]=1
    
    return b


## Netowrks with background conditions:

def make_cmgrid_withbackground(N,d):
    C=np.zeros((2*N,2*N),int)
    C[0:N,0:N]=make_cmgrid(N,d)
    #add the ``structured'' background conditions
    for i in range(N):
        C[N+i,i]=1
        #C[N+i,(i+1)%N]=1

    return C


def make_random_withbackground(N,d):
    R=np.zeros((2*N,2*N),int)
    R[0:N,0:N]=make_random(N,d)
    #add the ``structured'' background conditions
    for i in range(N):
        R[N+i,i]=1
        #R[N+i,(i+1)%N]=1

    return R


def make_tpm_withbackground(N, cm, threshh, noise=0, cuts=[]):  # pick thresh==0 so it calculate the thresh
    # assumes that the background nodes are at the end, so cm dim is 2*N,2*N
    # state by node
    if threshh > 0:
        thresh = threshh
    else:
        thresh = int((sum(cm) + 1) / 2)
    N2 = 2 * N
    M = 2 ** N2
    tpm = np.zeros([M, N2])
    for i in range(M):

        state = pyphi.convert.loli_index2state(i, N2)

        for node in range(N):
            act = np.dot(cm.transpose()[node], state)
            if act > thresh:
                tpm[i, node] = (1-noise/2)
            else:
                tpm[i, node] = noise/2
            # elif act < thresh:
            #     tpm[i, node] = noise/2
            # else:
            #     tpm[i, node] = 0.5

        # increase noise up to chance (alpha = 1) in the specified nodes
        for cut in cuts:
            node = cut[0]
            alpha = cut[1]
            tpm[i, node] = tpm[i, node] * (1 - alpha) + .5 * alpha

        for node2 in range(N):
            tpm[i, node2 + N] = 0.5

    return tpm


def make_tpm_gibbs(N, cm, temp=0, cuts=[], thresh=0):  # pick thresh==0 so it calculate the thresh
    M = 2 ** N
    tpm = np.zeros([M, N])

    if thresh < 1:
        thresh=int(N/2)

    for (source, target, strength) in cuts:
        cm[source, target] = strength

    for i in range(M):

        state = pyphi.convert.loli_index2state(i, N)

        for node in range(N):
            energy = np.dot(cm.transpose()[node], state) - (thresh+1)/2
            tpm[i, node] = 1 / (1 + np.exp(-(1/temp)*energy))

    return tpm

def make_tpm_withbackground_gibbs(N, cm, temp=0, cuts=[], thresh=0):  # pick thresh==0 so it calculate the thresh
    N2 = 2 * N
    M = 2 ** N2
    tpm = np.zeros([M, N2])

    if thresh < 1:
        thresh=int(N/2)

    for (source, target, strength) in cuts:
        cm[source, target] = strength

    for i in range(M):

        state = pyphi.convert.loli_index2state(i, N2)

        for node in range(N):
            energy = np.dot(cm.transpose()[node], state) - (thresh+1)/2
            tpm[i, node] = 1 / (1 + np.exp(-(1/temp)*energy))

        for node2 in range(N):
            tpm[i, node2 + N] = 0.5

    return tpm


# run only until converges
def running_behavior_conv(cm, N, iterations, state, threshh, do_plot=True,
                     do_print=True, return_final_state=False):  # pick threshh ==0 to calculate it

    lcm = len(cm)
    scm = sum(cm)

    if threshh > 0:
        thresh = threshh
    else:
        thresh = (scm + 1) / 2

    fixbackground = state[N:lcm]
    states = [state]
    for i in range(iterations - 1):
        statenew = np.dot(cm.transpose(), states[-1]) >= thresh
        statenew[N:lcm] = fixbackground

        states = np.append(states, [statenew], axis=0)

        if sum(abs(statenew - states[-2])) == 0:
            break

    statenew1 = states.transpose()
    statenew2 = states[::-1]

    if do_plot:
        fig = plt.figure()

        ax = fig.add_subplot(1, 2, 1)
        ax.set_aspect('equal')
        plt.imshow(statenew2, interpolation='nearest', cmap=plt.cm.PuBu)
        plt.colorbar()

        ax = fig.add_subplot(1, 2, 2)
        ax.set_aspect('equal')
        plt.imshow(statenew2[:, 0:N] - statenew2[:, N:(2 * N)],
                   interpolation='nearest', cmap=plt.cm.bwr)
        plt.colorbar()

        plt.show()

    distance = np.sum(np.abs(statenew[0:N] - fixbackground))

    if do_print:
        print('Distance: %d' % distance)

    if return_final_state:
        return states[-1]
    else:
        return distance

# cm whole conn matrix including backgroun, N is the nr of nodes in the ``main complex'', state as initial cond
def running_behavior(cm, N, iterations, state, threshh, do_plot=True,
                     do_print=True):  # pick threshh ==0 to calculate it

    lcm = len(cm)
    scm = sum(cm)

    if threshh > 0:
        thresh = threshh
    else:
        thresh = (scm + 1) / 2

    fixbackground = state[N:lcm]
    statenew = np.zeros((lcm, iterations), int)
    statenew[:, 0] = state
    for i in range(iterations - 1):
        statenew[:, i + 1] = np.dot(cm.transpose(), statenew[:, i]) >= thresh
        statenew[N:lcm, i + 1] = fixbackground


        # print('this is the new state\n', statenew[:,i+1])

    statenew1 = statenew.transpose()
    statenew2 = statenew1[::-1]

    if do_plot:
        fig = plt.figure()

        ax = fig.add_subplot(1, 2, 1)
        ax.set_aspect('equal')
        plt.imshow(statenew2, interpolation='nearest', cmap=plt.cm.PuBu)
        plt.colorbar()

        ax = fig.add_subplot(1, 2, 2)
        ax.set_aspect('equal')
        plt.imshow(statenew2[:, 0:N] - statenew2[:, N:(2 * N)],
                   interpolation='nearest', cmap=plt.cm.bwr)
        plt.colorbar()

        plt.show()

    distance = np.sum(np.abs(statenew[0:N, i + 1] - fixbackground))

    if do_print:
        print('Distance: %d' % distance)

    return distance

def difference_of_matching(cm, N, iterations, state,
                           threshh):  # threshh==0 calculates the thresh as maj
    lcm = len(cm)
    scm = sum(cm)
    if threshh > 0:
        thresh = threshh
    else:
        thresh = (scm + 1) / 2

    fixbackground = state[N:lcm]
    statenew = np.zeros((lcm, iterations), int)
    statenew[:, 0] = state
    dist = np.zeros(iterations, int)
    for i in range(iterations - 1):
        statenew[:, i + 1] = np.dot(cm.transpose(), statenew[:, i]) >= thresh
        statenew[N:lcm, i + 1] = fixbackground
        distance = statenew[0:N, i + 1] - fixbackground
        dist[i + 1] = sum(abs(distance))
        # print('this is the new state\n', statenew[:,i+1])

    finaldist = dist[iterations - 1]

    return finaldist


def pick_random_state(N, nrfiring, nrfiringbackground):
    vv = random.sample(range(N), nrfiring)
    bgrvv = np.asarray(random.sample(range(N), nrfiringbackground))
    bgrvv = bgrvv + N
    bgrvv = bgrvv
    vvar = np.zeros(2 * N, int)
    vvar[vv] = 1
    vvar[bgrvv] = 1
    background = vvar[N:]

    return vvar, background
