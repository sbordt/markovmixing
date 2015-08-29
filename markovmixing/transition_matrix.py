import numpy
import scipy.sparse as ssp

""" Check wheather p is a transtion matrix

p: a numpy ndarray
"""
def is_transition_matrix(p):
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
	return (numpy.abs(p.sum(axis=1)-1) < 1e-12).all()


##################################################################
# Convert adjancency to (sparse, lazy) srw transition matrices
##################################################################
def adjacency_to_srw_transition_matrix(A):
	P = A.copy()
	n = A.shape[0]

	for irow in range(0,n):
		P[irow,:] = P[irow,:] / P[irow,:].sum()

	return P

def adjacency_to_lazy_srw_transition_matrix(A):
	n = A.shape[0]

	return (adjacency_to_transition_matrix(A) + numpy.identity(n)) / 2.

def adjacency_to_srw_sparse_transition_matrix(A):
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

def adjacency_to_lazy_srw_sparse_transition_matrix(A):
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

		# lazy srw probability
		p = 0.5 / (row_end-row_start)

		# fill P
		for row_entry in range(row_start, row_end):
			P[row, J[row_entry]] = p

		P[row,row] = 0.5

		# continue with the next row
		row_start = row_end

	return P.tocsr()

##################################################################
# Utility methods to create transition matrices
##################################################################

# lazy random walk on the line
def line_lazy_transition_matrix(n, p = 0.5):
	P = ssp.lil_matrix((n,n))
	P[0,0] = 0.5
	P[0,1] = 0.5
	P[n-1,n-1] = 0.5
	P[n-1, n-2] = 0.5

	for i in range(1,n-1):
		P[i,i-1] = (1-p)/2
		P[i,i] = 0.5
		P[i,i+1] = p/2

	return P