import markovmixing as mkm
import networkx as nx
import numpy as np

####################### 50-CYCLE EXAMPLE #########################
# create a graph
G = nx.cycle_graph(50)
    
# create a MarkovChain that is lazy simple random walk on the graph
mc = mkm.nx_graph_lazy_srw(G)

# plot the total variation mixing
mc.add_random_delta_distributions(1)
mc.plot_tv_mixing(y_tol=0.01, threshold=0.01)  


####################### BIASED LINE EXAMPLE #########################
# create the transition matrix
P = mkm.line_lazy_transition_matrix(1000, p=0.51)

# create the MarkovChain with the given transition matrix
mc = mkm.MarkovChain(P)
    
# add some initial distributions
for i in [0,500,999]:
	mc.add_distributions(mkm.delta_distribution(1000,x=i))
        
# plot the total variation mixing
mc.plot_tv_mixing(y_tol=0.01, threshold=1e-5)


####################### NBRW EXAMPLE #########################
import matplotlib.pyplot as plt

# load a 6-regular graph with 50.000 nodes from file
G_6_regular = nx.read_sparse6('6_regular.s6')

# get the adjacency matrix
A = nx.to_scipy_sparse_matrix(G_6_regular)

# transition matrix for NBRW on the graph from the adjacency matrix 
P = mkm.graph_nbrw_transition_matrix(A)

# Markov chain with the transition marix
mc = mkm.MarkovChain(P)

# the stationary distribution is uniform
mc.set_stationary(mkm.uniform_distribution(mc.get_n()))

# add a random starting position to the Markov chain
mc.add_random_delta_distributions(1)

# determine the mixing
mc.compute_tv_mixing()

# plot the mixing
(x,tv) = mc.distribution_tv_mixing(0)
plt.plot(x, tv, marker='o', linestyle='dashed')
plt.title("Mixing of NBRW on a 6-regular graph")
plt.xlabel("t")
plt.ylabel("Toal variation distance to stationarity")
plt.show()	


####################### THE PACKAGE IN DETAIL #########################
# construct a Markov chain from a transition matrix
P = mkm.circle_transition_matrix(5)
mc = mkm.MarkovChain(P)

# add 3 distributions
mc.add_distributions(np.array([0,0,0,0.5,0.5]))
mc.add_random_delta_distributions(2)

# set the stationary distribution
mc.set_stationary(np.ones(5)/5)

mc.iterate_all_distributions_to_stationarity()

print mc.get_iteration_times(1)
print mc.get_iterations(1)

(t,tv) = mc.distribution_tv_mixing(1)
print t
print tv

mc.print_info()