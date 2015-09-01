import markovmixing as mkm

def test_iteration():
	import numpy, time, random

	N = 10000
	k = 10000
	P = mkm.line_lazy_transition_matrix(N)
	P = P.transpose()
	P = P.tocsr()
	
	# single distribution
	x = mkm.delta_distribution(N,0)
	start = time.time()
	for i in xrange(k):
		x = P.dot(x)
	end = time.time()
	print "Python loop:"
	print end - start 
	print x

	x = mkm.delta_distribution(N,0)
	start = time.time()
	x = mkm.matrix_vector_iteration_local(P,x,k)
	end = time.time()
	print "Python local iteration:"
	print end - start 
	print x

	x = mkm.delta_distribution(N,0)
	start = time.time()
	x = mkm.matrix_vector_iteration_by_processes(P,x,k)
	end = time.time()
	print "Python iterating (multiple processes):"
	print end - start 
	print x

	P = P.transpose()
	x = mkm.delta_distribution(N,0)
	start = time.time()
	x = mkm.iterate_distributions(P,x,k)
	end = time.time()
	print "Generic Python iteration:"
	print end - start 
	print x
	P = P.transpose()

	# multiple distributions
	k = 10000
	nd = 10

	random.seed(0)
	x = mkm.random_delta_distributions(N,nd).transpose()
	start = time.time()
	x = mkm.matrix_vector_iteration_local(P,x,k)
	end = time.time()
	print "Python local iteration:"
	print end - start 
	print x

	random.seed(0)
	x = mkm.random_delta_distributions(N,nd).transpose()
	start = time.time()
	x = mkm.matrix_vector_iteration_by_processes(P,x,k)
	end = time.time()
	print "Python iterating (multiple processes):"
	print end - start 
	print x

	random.seed(0)
	P = P.transpose()
	x = mkm.random_delta_distributions(N,nd)
	start = time.time()
	x = mkm.iterate_distributions(P,x,k).transpose()
	end = time.time()
	print "Generic Python iteration:"
	print end - start 
	print x
	P = P.transpose()

if __name__=="__main__":
    test_iteration()



