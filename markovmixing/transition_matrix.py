import numpy
import scipy.sparse as ssp

def is_transition_matrix(p):
	""" Check whether p is a transtion matrix

	p: a numpy ndarray
	"""
	# check whether we have a square matrix
	if p.ndim == 1:
		if p.shape != 1:
			return False
	elif p.ndim == 2:
		if p.shape[0] != p.shape[1]:
			return false
	else: 
		return False

	# check whether rows sum to 1
	# (be generous with the tolerance, this is not a check for the numerics) 
	return (numpy.abs(p.sum(axis=1)-1) < 1e-5).all()

def lazy(p):
	"""
	For a given transion matrix p, return the lazy version (p+I)/2

	p: a scipy sparse matrix
	"""
	return (p.tocsr()+ssp.eye(p.shape[0],format='csr'))/2.

##################################################################
# Construct transition matrices from graph adjacency matrices
##################################################################

def graph_srw_transition_matrix(A):
	"""
	For a graph given by an adjacency matrix A, construct the 
	transition matrix of the srw on the graph.

	A: an adjacency matrix, symmetric
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

	A: an adjacency matrix, symmetric
	"""
	A = A.todok()

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


##################################################################
# Utility methods to create transition matrices
##################################################################

def line_lazy_transition_matrix(n, p = 0.5):
	"""
	Returns the transition matrix for the lazy biased random walk
	on the n-path.

	p : probability to go to the higher number not regarding 
	lazyness (defaults to 0.5) 
	"""
	P = ssp.lil_matrix((n,n))
	P[0,0] = 0.5
	P[0,1] = 0.5
	P[n-1,n-1] = 0.5
	P[n-1, n-2] = 0.5

	for i in range(1,n-1):
		P[i,i-1] = (1-p)/2
		P[i,i] = 0.5
		P[i,i+1] = p/2

	return P.tocsr()