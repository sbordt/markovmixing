"""This module contains methods to setup Markov chains
as random walks on graphs from the networkx package.
"""

import markovmixing as mkm

def nx_graph_srw(G):
	"""Returns the Markov chain for the SRW on the graph G. 
	"""
	import networkx as nx

	A = nx.to_scipy_sparse_matrix(G)
	P = mkm.graph_srw_transition_matrix(A)
	mc = mkm.MarkovChain(P)
	mc.set_stationary(mkm.graph_srw_stationary_distribution(A))

	return mc

def nx_graph_lazy_srw(G):
	"""Returns the Markov chain for the lazy SRW on the graph G. 
	"""
	import networkx as nx

	A = nx.to_scipy_sparse_matrix(G)
	P = mkm.lazy(mkm.graph_srw_transition_matrix(A))
	mc = mkm.MarkovChain(P)
	mc.set_stationary(mkm.graph_srw_stationary_distribution(A))

	return mc

def nx_graph_nbrw(G):
	"""Returns the Markov chain for the NBRW on the graph G. 
	"""
	import networkx as nx

	A = nx.to_scipy_sparse_matrix(G)
	P = mkm.graph_nbrw_transition_matrix(A)
	mc = mkm.MarkovChain(P)
	mc.set_stationary(mkm.uniform_distribution(mc.get_n()))

	return mc

