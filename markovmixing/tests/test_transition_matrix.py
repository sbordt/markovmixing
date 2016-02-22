import markovmixing as mkm
import scipy.sparse as ssp
import numpy 

def tets_transition_matrix():
	assert(mkm.is_transition_matrix(numpy.eye(1)))
	assert(mkm.is_transition_matrix(numpy.eye(5)))
	assert(mkm.is_transition_matrix(numpy.ones((1,2,3))) == False)

	assert(mkm.is_transition_matrix(mkm.line_lazy_transition_matrix(100, p = 0.51)))

	# graph_nbrw_transition_matrix
	A = numpy.ones((3,3))
	numpy.fill_diagonal(A,0)
	P = mkm.graph_nbrw_transition_matrix(A)
	assert(mkm.is_transition_matrix(P))
	print A
	print P

	# tree_nbrw_transition_matrix
	A = numpy.array([[0,1,0,0,0],
					 [1,0,1,1,0],
					 [0,1,0,0,0],
					 [0,1,0,0,1],
					 [0,0,0,1,0]])
	P = mkm.tree_nbrw_transition_matrix(A,0)
	assert(mkm.is_transition_matrix(P))
	print A
	print P

if __name__=="__main__":
    tets_transition_matrix()