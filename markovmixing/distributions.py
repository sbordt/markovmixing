""" Methods for dealing with probability distributions as row-ndarrays, 
such as the generation of some usefull distributions and distance measures between distributions.
"""
import numpy
import random

def is_distribution(mu,n=None):
	"""
	Check whether mu is a probability distribution. 

	x: a numpy ndarray

	n: the size of the space (optional, defaults to None)
	"""
	# essentialy has to be a single row
	if mu.ndim != 1 and numpy.prod(mu.shape) != mu.shape[1]:
		return False

	# needs to sum to 1 
	# (be generous with the tolerance, this is not a check for the numerics)  
	if numpy.abs(mu.sum()-1) > 1e-4:
		return False

	# also check that it matches the size of the state space
	if n != None:
		if mu.ndim != 1 and mu.shape[1] != n:
			return False
		elif mu.ndim == 1 and mu.shape[0] != n:
			return False

	return True

def total_variation(mu,nu):
	""" Return the total variation distance between the two distributions.
	"""
	return abs(mu-nu).sum()/2.

def relative_error(mu,nu):
	""" Return the "relative error" between the two distributions, that is
	the maximal relative error at any vertex.
	"""
	divisor = numpy.maximum(numpy.minimum(mu,nu), 1e-20)
	
	return numpy.divide(abs(mu-nu), divisor).max()

def delta_distribution(n,x=0):
	""" Create a dirac-delta distribution.

	n: size of the state space.

	x: vertex with probability mass of 1. Defaults to 0.
	"""
	d = numpy.zeros(n)
	d[x] = 1.
	return d

def random_delta_distributions(n,k=1):
	""" Draw k random dirac-delta distributions.

	n: size of the state space.
	
	k: number of distributions to draw. Defaults to 1. 
	"""
	d = numpy.zeros((k,n))

	indices = random.sample(xrange(n),k)

	for i in range(0,k):
		d[i,indices[i]] = 1

	return d

def uniform_distribution(n):
	"""
	Returns the uniform distribution on n vertices.
	"""
	return numpy.ones(n)/n

def graph_srw_stationary_distribution(A):
	"""
	Computes the stationary distribution for a srw on a graph
	from the graphs adjacency matrix, pi(x) = deg(x)/(2|E|).

	A: a numpy ndarray
	"""
	# get vertex degrees by summing over rows
	degrees = (A.sum(axis=1).flatten()*1.0)[0]

	return numpy.array((degrees/degrees.sum()).tolist()[0]) # hack to get a simple row-vector