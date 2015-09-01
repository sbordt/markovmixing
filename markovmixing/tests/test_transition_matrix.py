import markovmixing as mkm
import scipy.sparse as ssp
import numpy 

def tets_transition_matrix():
	assert(mkm.is_transition_matrix(numpy.eye(1)))
	assert(mkm.is_transition_matrix(numpy.eye(5)))
	assert(mkm.is_transition_matrix(numpy.ones((1,2,3))) == False)

	assert(mkm.is_transition_matrix(mkm.line_lazy_transition_matrix(100, p = 0.51)))

	A = numpy.ones((3,3))
	numpy.fill_diagonal(A,0)
	P = mkm.graph_nbrw_transition_matrix(ssp.dok_matrix(A))
	assert(mkm.is_transition_matrix(P))
	print P


if __name__=="__main__":
    tets_transition_matrix()