#*****************************************************************************
#********************************  MODULES  **********************************
#*****************************************************************************

import matplotlib.pyplot as plt
import numpy as np
import itertools

from scipy.spatial.distance import cdist
#*****************************************************************************
#********************************  CLASSES  **********************************
#*****************************************************************************

#-----------------------------------------------------------------------------
class bichronic:
  #----------------------------------------------------
  def __init__(self,mcF,mcB,tpmF,tpmB,flow,directionality):
    self.directionality = directionality
    self.flow = flow
    self.forward = bichronic_direction(mcF,tpmF)
    self.backward = bichronic_direction(mcB,tpmB)
    
class bichronic_direction:
  def __init__(self,mc,tpm):
    self.main_complex = mc
    self.tpm = tpm
    self.Phi = np.array([m.phi for m in mc])

#-----------------------------------------------------------------------------
class node:
  #----------------------------------------------------
  def __init__(self,mechanism,initialstate = 0, varargin = None):
    def myand(input):
      return int(np.prod(input)==1)
    def myor(input):
      return int(sum(input)>0)
    def xor(input):
      val = sum(input)
      return int((val!=len(input))*(val!=0))
    def maj(input):
      return int(sum(input)>(len(input)/2))
    myFuncs = {'AND':myand,'OR':myor,'XOR':xor,'MAJ':maj}
    self.mechanism = mechanism
    self._update = myFuncs[mechanism]
    self.state = initialstate
    self.parents = []
  #----------------------------------------------------
  def update(self,inputs):
    return self._update(inputs)      

#-----------------------------------------------------------------------------
class net:
  #----------------------------------------------------
  def __init__(self,cm,mechanismList,coords = []):
    self.N = len(mechanismList)
    self.nodes = [node(mechanism) for mechanism in mechanismList]
    self.parents = {str(n):[] for n in range(self.N)}
    self.children = {str(n):[] for n in range(self.N)}
    for i in range(len(cm)):
      for j in range(len(cm)):
        if cm[i][j]==1:
          self.parents[str(j)].append(i)
          self.children[str(i)].append(j)
    self.currentState = np.array([node.state for node in self.nodes])
    self.setCoords(coords)
  #----------------------------------------------------
  def update(self):
    newState = [] 
    for n in range(self.N):
      inputs = [self.currentState[i] for i in self.parents[str(n)]]
      newState.append(self.nodes[n].update(inputs))
    for n in range(self.N):
      self.nodes[n].state = newState[n]
    self.oldState = self.currentState
    self.currentState = np.array(newState)
  #----------------------------------------------------
  def tpm(self):
    states = gen_states(self.N)
    tpm = np.zeros(np.shape(states))
    for s in range(2**self.N):
      self.currentState = states[s]
      self.update()
      tpm[s] = self.currentState
    return tpm
  #----------------------------------------------------
  def setCoords(self,coords=[]):
    trans = lambda x:np.transpose(np.matlib.repmat(x,2,1))
    if len(coords) == 0:
      coords = np.random.rand(self.N,2)*np.sqrt(self.N)
      vel = np.zeros(np.shape(coords))
      force = np.zeros([self.N,2])
      for t in range(500):
        distances = cdist(coords,coords)
        for n in range(self.N):
          repulsion = sum((coords[n]-coords)/trans(distances[n]+1))
          attraction = sum([coords[i] for i in self.parents[str(n)]] -coords[n])
          force[n] = repulsion + attraction
        vel = .9*vel + .1*force
        coords = coords + .2*vel
        #plt.close()
        #plt.figure(figsize = (10,10))
        #x = [coords[i][0] for i in range(self.N)]
        #y = [coords[i][1] for i in range(self.N)]
        #plt.plot(x,y,'o')
        #if t%50==0:
        #  for n in range(self.N):
        #    for m in self.parents[str(n)]:
        #      plt.plot([coords[n][0],coords[m][0]],[coords[n][1],coords[m][1]],color='black',alpha=.1)
        #  plt.pause(1)
        #plt.pause(.01)
        v2 = np.mean(np.mean(vel**2))
        #print(str(t)+','+str(v2))
        if v2<1:
          break
      x = np.array([coord[0] for coord in coords])
      y = np.array([coord[1] for coord in coords])
      x = x - min(x)
      x = x/max(x)
      y = y - min(y)
      y = y/max(y)
      self.coords = np.array([np.array([x[n],y[n]]) for n in range(self.N)])
  #----------------------------------------------------   
  def plot_vertices(self):
    coords = self.coords
    for n in range(self.N):
      if self.currentState[n]==0:plt.plot(coords[n][0],coords[n][1],'o',color = [1,0,0])
      else: plt.plot(coords[n][0],coords[n][1],'o',color = [0,0,1])
  #----------------------------------------------------   
  def plot_dynamics(self):
    plt.figure(figsize = (10,10))
    coords = self.coords
    for n in range(self.N):
      for m in self.parents[str(n)]:
        plt.plot([coords[n][0],coords[m][0]],[coords[n][1],coords[m][1]],color='black',alpha=.1)
    try:
      while (self.currentState != self.oldState).any():
        self.plot_vertices()
        plt.xlim([-.1,1.1])
        plt.ylim([-.1,1.1])
        self.update()
        plt.pause(.01)
    except KeyboardInterrupt:
      print('I love cheeese.')
  #----------------------------------------------------   
  def seed(self):
    self.oldState = self.currentState
    self.currentState = np.random.randint(0,2,self.N)
#*****************************************************************************
#*******************************  FUNCTIONS  *********************************
#*****************************************************************************

#-----------------------------------------------------------------------------
def flat(l):
  '''
  Function the flattens an array 
    
  Keyword arguments:
    l -- input list
  '''
  return [item for sublist in l for item in sublist]
#-----------------------------------------------------------------------------
def argsort(seq):
    '''
    Index list of the sorted elements of a list. Taken from: 
    http://stackoverflow.com/questions/3071415/efficient-method-to-calculate
        -the-rank-vector-of-a-list-in-python
    
    Keyword arguments:
      seq -- input sequence
    '''
    return sorted(range(len(seq)), key=seq.__getitem__)
#-----------------------------------------------------------------------------
def logistic(x):
  '''
  Vectorized logistic function: 
    1/(1+e^-x)
    
  Keyword arguments:
    x -- input scalar or vector
  '''
  
  try:
    return 1/(1+np.exp(-x))          #evaluates if x is a scalar
  except TypeError:
    return list(map(logistic,x))    #evaluates if x in an array
#-----------------------------------------------------------------------------
def energy(state,coupling,bias):
  '''
  Energy and density of a single state in the Ising Model
  
  Keyword arguments:
    state     --  (1,N) array of ones and zeros
    coupling  --  (N,N) array of coupling coefficients
    bias      --  positive scalar biasing the 0 state
  '''
  #convert state into 1s and -1s
  state = 2*state - 1
  #find the energy density coming from the coupling
  density = -(np.dot(coupling,state)-bias)
  #Find the total energy of the state
  return density
#-----------------------------------------------------------------------------
def energies(states,coupling,bias):
  '''
  Energy and density of each state in the Ising Model
  
  Keyword arguments:
    states    --  (1,N) array of ones and zeros
    coupling  --  (N,N) array of coupling coefficients
    bias      --  positive scalar biasing the 0 state
  '''
  Es = []
  for state in states:
    Es.append(energy(state,coupling,bias))
  return np.array(Es)
#-----------------------------------------------------------------------------
def pruneLR(states):
  '''
  Prune the states that are LR symmetric from a list of states. That is, if
  two states in a list are equivalent under a flip across the center of the
  state, then one of the states is removed from the list. Returns the list of
  valid states, as well as the list of valid indexes.
    cm = mf.gen_CM_grid(dimension, number of nodes, radius of neighborhood, periodic tag, self connection tag)

    dimension: dimension of grid (1 for you)
    number of nodes:
    duh
    radius of neighborhood:
    how many nearest neighbors a node connects to (Sabrina’s d)
    periodic tag:
    True (periodic boundary conditions) or False (edge boundary conditions)
    self connection tag:
    True(self connections exist) or False (no self connections)

    FOR EXAMPLE, if I want to create a 1-d grid with 5 nodes connecting only to the left and right neighbors, with periodic boundary conditions and self connections:
    cm = gen_CM_grid(1, 5, 1, True, True)


create the state-to-node TPM:

    tpm = mf.gen_TPM_s2n( weight matrix, temperature, bias)

    weight matrix:
    The coupling between connected nodes. If the network is bidirectional, then this is .5*cm.
    (the reason here is that every entry in the cm is counted so each pair of connected nodes will couple with a strength of 2)
    temperature:
    The level of stochasticity. T = 0 is a deterministic system. Note that in a deterministic system with no self connections, the
    TPM will not be all 1s and 0s, because if the inputs to a node are tied (equal 1s and 0s) then the node will choose its next state by
    flipping a coin.
    Andrew(last I saw) and I use a T = 4 level of stochasticity.
    bias: An external bias(magnetic field) that breaks the symmetry between 0 and 1. It is in the direction of the 0 state, so a large bias
    will push the system towards the Zen State.



  
  Keyword arguments:
    states     --  (M,N) row array of states
  '''
  N = len(states)
  copies = []
  #loop through all the states in the list
  for n in range(N-1):
    #flip the state
    flippedState = np.fliplr([states[n]])[0]
    #loop through all the states after given state to see if theyre equivalent
    for m in range(n+1,N):
      if np.prod(states[m]==flippedState):
        #if they are equivalent, then save the index of the equivalent state
        copies.append(m)
  #remove all the copies to create a list of indices of the unique states
  valid = list(set(range(N))-set(copies))
  return states[valid],valid
#-----------------------------------------------------------------------------
def pruneProbs(probs,states):
  '''
  Looks at a list of probabilities of pruned states, and fixes the
  probabilities.
  
  Keyword arguments:
    probs     --  (1,N) array of ones and zeros
    states    --  (N,N) array of coupling coefficients
  '''
  N = len(states)
  for n in range(N):
    if not(np.prod(states[n] == np.fliplr([states[n]])[0])):
      probs[n] = 2*probs[n]
  return probs/np.sum(probs)
#-----------------------------------------------------------------------------
def state_probs(states,coupling,bias):
  '''
  The probability of each state in an Ising Model
  
  Keyword arguments:
    states    --  states to calculate the probability of
    coupling  --  the coupling of the Ising model
    bias
  '''
  bf = lambda s:np.exp(-sum((2*s-1)*energy(s,coupling,bias)))
  probs = np.array(list(map(bf,states)))
  return probs/sum(probs)
#-----------------------------------------------------------------------------
def state_entropy(states):
  return 0

#*****************************************************************************
#************************* OBJECT GENERATORS *********************************
#*****************************************************************************

#-----------------------------------------------------------------------------
def gen_1Dgrid(n,c=1, periodic=False):
  '''
  Generate connectivity matrix for a 1-D grid, with or without periodic
  boundary conditions.
  
  Keyword arguments:
    n         -- positive integer number of nodes
    c         -- number of neighbors for each node
    periodic  -- boolean tag for boundary conditions (default False)
  '''
  #creates periodic connectivity matrix by shifting the identity matrix
  cm = np.zeros([n,n])
  for m in range(1,c+1):
    cm += np.roll(np.eye(n),m,1)+np.roll(np.eye(n),-m,1)
  #checks boundary conditions and removes
  if not(periodic) or n==2:
    for r in range(n):
      if r+c<n:
        cm[r][(r+c+1):] = 0
      if r-c>=0:
        cm[r][:(r-c)] = 0
  cm[cm>1]=1
  return cm
#-----------------------------------------------------------------------------
def gen_CM_grid(d, n, c, periodic=False, self_conn=False):
  '''
  Generate connectivity matrix for a d-D grid, with or without periodic
  boundary conditions and self connections. If it is periodic then the 
  connectivity matrix describes a torus.
  
  Keyword arguments:
    d         -- positive integer dimension of grid
    n         -- positive integer number of nodes
    c         -- number of neighbors to go out to
    periodic  -- boolean tag for boundary conditions (default False)
    self_conn -- boolean tag for self connections (default False)
  '''
  #Takes the symmetrized Kronecker product since
  # cm(L^d) = cm(L^(d-1)) x Id + Id x cm(L^(d-1))
  # for L^n the n-dimensional lattice (or torus)
  if d == 1:
    cm = gen_1Dgrid(n,c,periodic)
  else:
    cm = np.kron(gen_CM_grid(d-1,n,c,periodic),np.eye(n)) +     \
          np.kron(np.eye(n**(d-1)),gen_1Dgrid(n,c,periodic))
  vint = np.vectorize(int)
  if self_conn:
    cm = cm + np.identity(n**d)
  return vint(cm)
#-----------------------------------------------------------------------------
def gen_TPM_s2n(coupling, temperature=1, bias=0):
  '''
  Generate the transition probability matrix for an ising model with a given 
  coupling, temperature, and magnetic field. Second output is the energy of
  all the states that the system can be in.
  
  Keyword arguments:
    coupling    -- (N,N) array of coupling coefficients
    temperature -- positive scalar that increases stochasticity
    bias        -- positive scalar biasing the 0 state (default 0)
  '''
  # TODO preallocate lists to be filled in
  tpm = []

  # Create an array of all the possible states for hte system
  states = gen_states(len(coupling))

  #calculate the energy of each state and determine the probability that
  #each node transitions into being a 1.
  for state in states:
    energy_density = energy(state,coupling,bias)
    if temperature > 0:
      tpm.append(logistic(-2.0*energy_density/temperature))
    else:
      tpm.append((np.sign(energy_density)+1)/2)
  return np.array(tpm)
#-----------------------------------------------------------------------------
def gen_states(n):
  '''
  Generate an array of all the states of n bits
  
  Keyword arguments:
    n    -- positive integer number of bits
  '''
  states = np.fliplr(np.array(
            list(map(list,itertools.product([0,1],repeat = n)))
            ))
  return states
  
#*****************************************************************************
#************************* FANCY PLOTTING ************************************
#*****************************************************************************

#-----------------------------------------------------------------------------
def errPlot(xdata,ydata,yerror,sigmas,shade_color):
  '''
  Error plot for ydata that has errors associated with it
  
  Keyword arguments:
    xdata      -- independent variable data
    ydata      -- dependent variable data (mean)
    yerror     -- errors in dependent variable data (standard deviation)
    sigmas     -- number of standard deviations to go out to
    shade_data -- color used for the shading
  '''
  plt.plot(xdata,ydata,'-o',color = 'black')
  myalpha = .5/(np.log(sigmas+1)+np.pi**2/12)
  #print(myalpha)
  for n in range(1,sigmas+1):
    plt.fill_between(xdata,ydata-n*yerror,ydata+n*yerror,
                      color = shade_color,alpha = myalpha/n,lw = 0)
#-----------------------------------------------------------------------------
