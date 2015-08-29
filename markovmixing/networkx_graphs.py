"""This file contains various methods to setup Markov chains
as random walks on graphs from the networkx package.
"""

import markovmixing as mkm

def nx_graph_srw(G):
	"""Returns the srw on the graph G 
	"""
	import networkx as nx

	A = nx.to_scipy_sparse_matrix(G)
	P = mkm.adjacency_to_srw_sparse_transition_matrix(A)

	return mkm.MarkovChain(P)

def nx_graph_lazy_srw(G):
	"""Returns the srw on the graph G 
	"""
	import networkx as nx

	A = nx.to_scipy_sparse_matrix(G)
	P = mkm.adjacency_to_lazy_srw_sparse_transition_matrix(A)

	return mkm.MarkovChain(P)

def nx_graph_nbrw(G):
	import networkx as nx

	raise Exception('not implemented')


def nx_graph_analyze_srw(G):
	import networkx as nx
	import matplotlib.pyplot as plt

	mc = mkm.nx_graph_lazy_srw(G)
	mc.add_distributions(mkm.random_dirac_delta_distributions(nx.number_of_nodes(G),1))
	mc.iterate_all_distributions_to_stationarity()
	
	(x,tv) = mc.get_distribution_tv_mixing(0)
	plt.plot(x, tv)
	plt.xlabel("t")
	plt.ylabel("Distance to stationary distribution in total variation")
	plt.show()	

	mc.print_info()
