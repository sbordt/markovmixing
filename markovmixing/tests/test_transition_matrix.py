import markovmixing as mkm
import numpy 

def tets_transition_matrix():
	assert(mkm.is_transition_matrix(numpy.eye(1)))
	assert(mkm.is_transition_matrix(numpy.eye(5)))
	assert(mkm.is_transition_matrix(numpy.ones((1,2,3))) == False)

	assert(mkm.is_transition_matrix(mkm.line_lazy_transition_matrix(100, p = 0.51)))



if __name__=="__main__":
    tets_transition_matrix()