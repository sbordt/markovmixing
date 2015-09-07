import markovmixing as mkm
import networkx as nx
import matplotlib.pyplot as plt

# load a random 6-regular graph with 50.000 nodes from file
G_6_regular = nx.read_sparse6('6_regular.s6')

# get the adjacency matrix
A = nx.to_scipy_sparse_matrix(G_6_regular)

# transition matrix for SRW on the graph from the adjacency matrix 
P = mkm.graph_srw_transition_matrix(A)

# Markov chain with the transition marix
mc = mkm.MarkovChain(P)

# stationary distribution of SRW on a graph is deg(x)/2*|E|
mc.set_stationary(mkm.graph_srw_stationary_distribution(A))

# add a random starting position to the Markov chain
mc.add_distributions(mkm.random_delta_distributions(mc.get_n(),1))

# determine the mixing in total variation 
mc.compute_tv_mixing()

# plot the mixing
(x,tv) = mc.distribution_tv_mixing(0)
plt.plot(x, tv)
plt.xlabel("t")
plt.ylabel("Distance to stationary distribution in total variation")
plt.show()	




# load a random 3-regular graph with 50.000 nodes from file
G_3_regular = nx.read_sparse6('3_regular.s6')

# analyze the mixing in total variation of the NBRW on the graph 
# (it is a Markov chain with 150.000 states)
mkm.nx_graph_analyze_nbrw(G_3_regular)






