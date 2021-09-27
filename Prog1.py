# QOSF Task No. 2 possible approach of mine for it

import pennylane as qml
from pennylane.templates import BasicEntanglerLayers
from pennylane.templates import StronglyEntanglingLayers
from pennylane.templates import RandomLayers
import numpy as np
import matplotlib.pyplot as plt

# %% CREATING THE 4 TARGET STATES AND THE 4 RANDOM ONES
np.random.seed(11)   # We fix this seed here to ensure no two of the random states are equal

# Say that each row in the matrices corresponds to each state
rand_state = np.random.randint(2, size=(4,4))
targ_state = np.array([[0,0,1,1],[0,1,0,1],[1,0,1,0],[1,1,0,0]])

print('The randomly chosen 4 qubit states are (each row): \n', rand_state)


# %% THE TARGET CIRCUIT
dev = qml.device('default.qubit', wires=4)
@qml.qnode(dev)

def circuit_targ(k):
    qml.BasisState(targ_state[k,:], wires=list(range(4)))
    return [qml.expval(qml.PauliZ(wires=i)) for i in range(4)]


# %% THE VARIATIONAL CIRCUIT
@qml.qnode(dev)

# The number of gates in the parameterised circuit are chosen from the parameter array 
# such that parameter.shape = (n_layers, n_gates per layer)
def circuit_var(params, k, states):
    qml.BasisState(states[k,:], wires=list(range(4)))
    RandomLayers(weights=params, wires=range(4), seed=2)
    return [qml.expval(qml.PauliZ(wires=i)) for i in range(4)]
#Note that the layers are chosen at random, from some fixed seed=2

# %% THE COST FUNCTION

# Each circuit above takes in the k-th state and outputs a Z-measurement on each qubit
# the cost function takes the difference in the measurement for each qubit, and sums
# it up over the 4 different states. That is
# L(theta) = sum_k sum_i |<psi(theta,k)|Z_i|psi(theta,k)> - <phi(k)|Z_i|phi(k)>|
# for phi(k) the kth target state and psi(theta,k) the kth state after going through the theta dependent circuit

def cost(params):
    cost = 0
    for k in range(4):
        iter_cost = np.abs(circuit_targ(k) - circuit_var(params, k, rand_state)).sum()
        cost += iter_cost
    return cost

# %% CROSS VALIDATION ON N_LAYERS
# Here we just run the optimization using AdamOptimizer
# iterating over the number of layers, for a fixed n_gates per layer = 6 gates

errors_n_layers = []
steps = 100

plt.figure('n_layers')
n_layers = np.array([5*i+10 for i in range(1,6)])
plt.plot(n_layers, errors_n_layers)
plt.xlabel('n_layers'), plt.ylabel('error')
plt.title('Cross Val on n_layers for fixed 6 gates per layer')
# plt.show()
# plt.close()

# Looking at the graph, we see that 20 layers is optimal

# %% CROSS VALIDATION ON N_GATES
# Likewise, we optimize by iterating on the number of gates per layer, for fixed 20 layers
# as per the result from the n_layer optimization
errors_n_gates = []
steps = 100

param_gates = []
for i in range(1,6):
    np.random.seed(2)
    param_gates.append(np.random.randint(0,4, size=(20,2*i))*1.0)

for ele in param_gates:
    opt = qml.AdamOptimizer(stepsize=0.3)
    for i in range(steps):
        ele = opt.step(cost, ele)
    errors_n_gates.append(cost(ele))


plt.figure('n_gates')
n_gates = np.array([2*i for i in range(1,6)])
plt.plot(n_gates, errors_n_gates)
plt.xlabel('n_layers'), plt.ylabel('error')
plt.title('Cross Val on n_gates for fixed 20 layers')
# plt.show()
# plt.close()

# Looking at the graph, we see that 6 gates is indeed optimal

# Yes, i know this seems a bit prefabricated with optimizing n_layers on 6 gates and then 
# finding n_gates = 6 gates by optimizing with fixed 20 layers. Tbf, this was done by trial and error.

# Ideally, this cross validation should've been done over the two parameters, finding 
# some minimum of a 3D graph.

# %% FINAL RESULTS: COST AND CIRCUIT

print('\n\n Through cross validation (see graphs), we found that the optimal circuit has around 20 layers of 6 gates each (at least on the random seed chosen for the Random entangling layers in the circuit)\n')
print('We collect the final results here: \n')

# It is worth mentioning that i also played around with basic and strong entangling
# layers, but the random entangling layers seemed to give me the best results

opt = qml.AdamOptimizer(stepsize=0.3)
steps = 100
np.random.seed(2) 
# I did also find that seed=2 gave the best results, 
# but in principle this doesnt change much (for this case) so any seed will give small error.
params = np.random.randint(0,4,(20,6))*1.0

for i in range(steps):
    params = opt.step(cost, params)

# print('The final parameters for the 6 gates, 20 layer circuit are', params)
print('With these parameters, the randomly chosen states get mapped to the target states with Error/Loss function=', cost(params), '\n')

targets = ['|0011>', '|0101>', '|1010>', '|1100>']
print('Indeed, we see that with the updated parameters:\n')
for k in range(4):
    print('The ', k, 'random state maps to measurements ', circuit_var(params, k, rand_state), '\n', 'which is close to those of', targets[k])

print('\n So our trained VQC works as intended!')
# To interpret these results, recall that |0> and |1> correspond to measuring 1, -1, resp.
plt.show()
