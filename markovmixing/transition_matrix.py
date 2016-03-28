""" Methods for transition matrices.

Transition matrices are represented by scipy sparse matrices (csr).
"""
import numpy
import scipy.sparse as ssp
import markovmixing as mkm

def is_transition_matrix(P):
	""" Check whether p is a transtion matrix.

	P: A numpy ndarray
	"""
	# check whether we have a square matrix
	if P.ndim == 1:
		if P.shape != 1:
			return False
	elif P.ndim == 2:
		if P.shape[0] != P.shape[1]:
			return false
	else: 
		return False

	# check whether rows sum to 1
	# (be generous with the tolerance, this is not a check for the numerics) 
	return (numpy.abs(P.sum(axis=1)-1) < 1e-5).all()

def lazy(P):
	"""
	For a given transion matrix P, return the lazy version (P+I)/2.

	P: A scipy sparse matrix
	"""
	return (P.tocsr()+ssp.eye(P.shape[0],format='csr'))/2.

##################################################################
# Construct transition matrices from graph adjacency matrices
##################################################################

def graph_srw_transition_matrix(A):
	"""
	For a graph given by an adjacency matrix A, construct the 
	transition matrix of the srw on the graph.

	Transition and adjacency matrix have the same dimesion, and the
	indexing of states corresponds.

	A: An adjacency matrix, symmetric
	"""
	(I,J,V) = ssp.find(A)
	n = A.shape[0]

	P = ssp.lil_matrix((n,n))
	nnz = I.shape[0]

	row_start = 0
	while row_start < nnz:
		row = I[row_start]

		# find the end of the row
		row_end = row_start
		while row_end < nnz and I[row_end] == row:
			row_end = row_end+1

		# srw probability
		p = 1. / (row_end-row_start)

		# fill P
		for row_entry in range(row_start, row_end):
			P[row, J[row_entry]] = p

		# continue with the next row
		row_start = row_end

	return P.tocsr()

def graph_nbrw_transition_matrix(A):
	"""
	For a graph given by an adjacency matrix A, construct the 
	transition matrix of the nbrw on the graph.

	The transition matrix does generally have different dimensions than
	the adjacency matrix.

	A: An adjacency matrix, symmetric
	"""
	A = ssp.dok_matrix(A)

	# construct a bijection between edges in the graph
	# and vertex numbers in the state space of the markov chain
	edge_to_number = {}
	number_to_edge = []
	
	for edge in A.iterkeys():
		edge_to_number[edge] = len(number_to_edge)
		number_to_edge.append(edge)

	# create the transition matrix row by row (that is edge by edge)
 	n = len(number_to_edge)
	P = ssp.lil_matrix((n,n))
	A = A.tolil()

	for irow,edge_x_y in enumerate(number_to_edge):
		# find all (y,z)-edges in A
		z_nodes = []

		for z in A.getrowview(edge_x_y[1]).nonzero()[1]:
			if z != edge_x_y[0]:
				z_nodes.append(z)

		# select z uniformly 
		if len(z_nodes) == 0:
			raise Exception('NBRW is not well-defined on a graph with vertices of degree 1')
		p = 1./len(z_nodes)

		# fill the right places in the matrix
		for z in z_nodes:
			P[irow,edge_to_number[(edge_x_y[1],z)]] = p

	return P.tocsr()

def tree_nbrw_transition_matrix(A, root):
	"""
	For a rooted tree given by an adjacency matrix A and a root, construct the 
	transition matrix of the nbrw on the tree, starting at the root.

	Transition and adjacency matrix have the same dimension, and the
	indexing of states corresponds.

	The nbrw will stay at leaves with probability 1.

	A: An adjacency matrix, symmetric
	root: Index of root node in the adjacency matrix.
	"""
	A = ssp.lil_matrix(A)
	n = A.shape[0]
	P = ssp.lil_matrix((n,n))

	def fill_row(node, parent):
		neighbors = A.getrowview(node).nonzero()[1]

		if len(neighbors) == 0:
			raise Exception('Invalid adjacency matrix or the root has no child.')

		# a leaf - break recursion
		if parent != -1 and len(neighbors) == 1:
			P[node,node] = 1
			return

		# transition probability
		p = 1./len(neighbors)
		
		if parent != -1:
			p = 1./(len(neighbors)-1)		

		for child in neighbors:
			if child != parent:
				P[node,child] = p

				fill_row(child, node)

	# fill rows of P starting from the root using bfs, i.e.
	# like the nbrw traverses the graph
	fill_row(root, -1)

	return P.tocsr()

##################################################################
# Utility methods to create transition matrices
##################################################################

def line_lazy_transition_matrix(n, p = 0.5):
	"""
	Returns the transition matrix of the lazy biased random walk
	on the n-path.

	p : probability to go to the higher number not regarding 
	lazyness (defaults to 0.5) 
	"""
	P = ssp.lil_matrix((n,n))
	P[0,0] = 0.5
	P[0,1] = 0.5
	P[n-1,n-1] = 0.5
	P[n-1,n-2] = 0.5

	for i in range(1,n-1):
		P[i,i-1] = (1-p)/2
		P[i,i] = 0.5
		P[i,i+1] = p/2

	return P.tocsr()

def circle_transition_matrix(n, p = 0.5, lazy = True):
	"""
	Returns the transition matrix of the (possibly lazy) biased random walk
	on the n-cycle.

	p : probability to go to the higher number not regarding 
	lazyness (defaults to 0.5) 
	lazy: should the walk be lazy? Defaults to True
	"""
	P = ssp.lil_matrix((n,n))
	
	P[0,n-1] = (1-p)
	P[0,1] = p

	P[n-1,n-2] = (1-p)
	P[n-1,0] = p

	for i in range(1,n-1):
		P[i,i-1] = (1-p)
		P[i,i+1] = p

	if lazy:
		P = mkm.lazy(P)

	return P.tocsr()

def hypercube_transition_matrix(n, lazy = True):
	"""
	Returns the transition matrix of the (possibly lazy) random walk 
	on the n-dimensional hypercube.

	n: dimension. The chain has 2^n states
	lazy: should the walk be lazy? Defaults to True
	""" 
	k = pow(2,n)
	P = ssp.lil_matrix((k,k))
	p = 1./n

 	# use our infinite wisdom on bitwise operators to swap bit j in the number i
	for i in range(0,k):
		for j in range(0,n):
			P[i, i ^ ( (i ^ (~ (i & (1 << j))) ) & (1 << j) )] = p

	if lazy:
		P = mkm.lazy(P)

	return P.tocsr()