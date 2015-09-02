import markovmixing as mkm
from markovmixing import MarkovChain 

import numpy

def test_markov_chain():
	#  initialization and initial state (take n>=10000 to challenge the numerics)
	n = 100
	mc = MarkovChain(mkm.line_lazy_transition_matrix(n))

	assert(mc.get_n() == n)
	assert(mc.stationary_distribution_known() == False)
	assert(mc.get_stationary_distribution() == None)
	assert(mc.num_distributions() == 0)

	# distributions
	mc.add_distributions(mkm.delta_distribution(n,0))
	assert (mc.get_distribution(0) == mkm.delta_distribution(n,0)).all()

	mc.add_distributions(mkm.random_delta_distributions(n,2))
	mc.add_distributions(mkm.delta_distribution(n,n-1))
	assert(mc.num_distributions() == 4)

	# iterations
	assert(mc.last_iteration_time(1) == 0)

	# iterate 
	mc.iterate_distributions([0],2) # this one will determine the stationary distribution
	assert(mc.last_iteration_time(0) == 2)
	assert(mc.next_iteration_time(0,1) == 2)

	mc.iterate_distributions([0,1,3],5)

	mc.iterate_distributions_to_stationarity([0,2])
	mc.iterate_all_distributions_to_stationarity()

	# stationary distribution

	# mixing
	(x,tv) = mc.distribution_tv_mixing(1)
	mc.compute_tv_mixing()

	# print some stuff
	mc.print_info()

if __name__=="__main__":
    test_markov_chain()