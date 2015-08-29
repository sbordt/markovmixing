import markovmixing as mkm
from markovmixing import MarkovChain 

import numpy

def test_markov_chain():
	#  initialization and initial state
	n = 10
	mc = MarkovChain(mkm.line_lazy_transition_matrix(n))

	assert(mc.get_n() == n)
	assert(mc.stationary_distribution_known() == False)
	assert(mc.get_stationary_distribution() == None)
	assert(mc.get_num_distributions() == 0)

	# distributions
	mc.add_distributions(mkm.dirac_delta_distribution(n,0))
	assert (mc.get_distribution(0) == mkm.dirac_delta_distribution(n,0)).all()

	mc.add_distributions(mkm.random_dirac_delta_distributions(n,3))
	assert(mc.get_num_distributions() == 4)

	# iterations
	assert(mc.get_last_iteration_time(1) == 0)
	#mc.add_iteration(1,100,mkm.dirac_delta_distribution(n,5))
	#assert(mc.get_last_iteration_time(1) == 100)


	# iterate 
	mc.iterate_distributions([0],2)
	mc.iterate_distributions([0,1,3],5)

	mc.iterate_distributions_to_stationarity([0,2])
	mc.iterate_all_distributions_to_stationarity()

	# stationary distribution

	# mixing
	(x,tv) = mc.get_distribution_tv_mixing(1)
	print x
	print tv

	# print some stuff
	mc.print_info()

if __name__=="__main__":
    test_markov_chain()