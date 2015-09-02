import markovmixing as mkm
import numpy 

def tets_distributions():
	# not distributions
	d = numpy.ones((3,3))
	assert(mkm.is_distribution(d) == False)

	d = mkm.delta_distribution(10)
	d[1] = 1
	assert(mkm.is_distribution(d) == False)

	
	assert(mkm.is_distribution(mkm.delta_distribution(10),n=11) == False)
	assert(mkm.is_distribution(mkm.delta_distribution(10),n=10) == True)

	assert(mkm.relative_error(numpy.array([0.0001,1,1e-16]),numpy.array([0.001,1,1e-16]))>8.0)
	# print mkm.relative_error(numpy.array([0.0001,1,1e-16]),numpy.array([0.001,1,1e-16]))

if __name__=="__main__":
    tets_distributions()