import numpy
import random

def total_variation(my,nu):
	return abs(my-nu).sum()/2.

def relative_error(my,nu):
	divisor = numpy.maximum(numpy.minimum(my,nu), 1e-20)
	
	return numpy.divide(abs(my-nu), divisor).max()

def dirac_delta_distribution(n,x):
	d = numpy.zeros(n)
	d[x] = 1.
	return d

def random_dirac_delta_distributions(n,k=1):
	d = numpy.zeros((k,n))

	indices = random.sample(xrange(n),k)

	for i in range(0,k):
		d[i,indices[i]] = 1

	return d
