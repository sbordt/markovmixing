""" This file implements simultaneously iterating several 
distributions wrt. a Markov chain using multiple CPU cores.
The core method is iterate_distributions.
"""
import numpy
import scipy.sparse as ssp
import scipy.io as sio

from multiprocessing import Pool
import tempfile 
import shutil	

def load_matrix(tmpdir):
	mat = sio.loadmat(tmpdir+"/A.mat")
	return mat['A'].tocsr()

def save_matrix(tmpdir,A):
	sio.savemat(tmpdir+"/A.mat", {'A': A})
	return

def load_x(tmpdir,i):
	mat = sio.loadmat(tmpdir+"/x_"+`i`+".mat")
	return mat['x'][0]

def save_x(tmpdir,x,i):
	# handle both vector and matrix inputs
	if x.ndim == 1:
		sio.savemat(tmpdir+"/x_"+`i`+".mat", {'x': x})	
	else:
		sio.savemat(tmpdir+"/x_"+`i`+".mat", {'x': x[:,i]})
	return 

# util to handle vector and matrix inputs
def get_nvec(x):
	if x.ndim == 2:
		return x.shape[1]
	return 1

def iterate_distributions(p,d,k):
	""" Iterate some probability distributions with respect to a 
	transition matrix. Uses matrix_vector_iteration to perform the iteration.

	p: A transition matrix
	d: An ndarray where every row is taken as a distribution
	k: The number of iterations

	Returns the iterated distributions.
	"""	
	# transpose the transition matrix and the distributions
	# to transform the problem to the standard matrix-vector product setting
	p = p.transpose()
	d = d.transpose()

	return matrix_vector_iteration(p,d,k).transpose()

def matrix_vector_iteration(A,x,k): 
	""" Iterate some vectors with respect to a matrix.

	A: A matrix
	x: An ndarray of column vectors
	k: The number of iterations
	"""
	nvec = get_nvec(x)
	n = A.shape[0]

	if nvec == 1 or n*k < 1e5:
		return matrix_vector_iteration_local(A,x,k)

	# spawn worker processes and iterate distributions seperately		
	return matrix_vector_iteration_by_processes(A,x,k)

def matrix_vector_iteration_local(A,x,k):
	y = x.copy()

	# vector shape
	if x.ndim == 1:
		for j in xrange(k):
			y = A.dot(y)
	# matrix shape	
	else:	
		for i in xrange(get_nvec(x)):
			for j in xrange(k):
				y[:,i] = A.dot(y[:,i])

	return y

def matrix_vector_iteration_by_processes(A,x,k):
	# create a temporary directory to store the matrix and the vectors
	tmpdir = tempfile.mkdtemp()

	nvec = get_nvec(x)
	y = x.copy()

	save_matrix(tmpdir,A)
	for i in xrange(nvec):
		save_x(tmpdir,x,i)

	# start processes
	pool = Pool(processes=min(nvec,6))
	processes = []

	for i in xrange(nvec):
		processes.append( pool.apply_async(matrix_vector_iteration_process, (tmpdir,i,k)) ) 

	# fetch results (vector/matrix shape version)
	if x.ndim  == 1:
		processes[0].get()
		y = load_x(tmpdir,0)
	else:
		for i in xrange(nvec):
			processes[i].get()
			y[:,i] = load_x(tmpdir,i)

	pool.close()

	# remove temporary directory (with all it contains)
	shutil.rmtree(tmpdir)

	return y
	
def matrix_vector_iteration_process(tmpdir,i,k):
	P = load_matrix(tmpdir)
	x = load_x(tmpdir,i)

 	for j in xrange(k):
 		x = P.dot(x)

 	save_x(tmpdir,x,i)
 	return