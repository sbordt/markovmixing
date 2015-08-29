import markovmixing as mkm
import scipy.sparse as ssp
import numpy 

class MarkovChain:
	"""Class for Markov chains.

	Any Markov chain is defined by its transition matrix.

	The transition matrix is being stored as a scipy sparse matrix.

	In addition to the transition matrix this class manages "distributions"
	and their "iterations". A "distribution" is an initial distribution 
	on the state space of the Markov chain. An "iteration" is the initial
	distribution after the Markov chain went t steps, or distribution*P^{t}.
	If everything goes right, the sequence of iterations should converge to the 
	stationary distribution.

	Distributions and iterations are stored in a dictionary, mapping t to the 
	iteration after time t, where t=0 corresponds to the original distribution.

	Does not store information such as where the Markov chain 
	comes from (i.e. whether it is random walk on a graph or not).
	"""

	def __init__(self, p):
		"""Initialize a Markov chain with a transition matrix

		p: the transition matrix, a numpy ndarray
		"""
		# verify that p is indeed a transition matrix
		if mkm.is_transition_matrix(p) == False:
			raise Exception('Matrix used to initialize a Markov chain is not a transition matrix')

		self.p = ssp.csr_matrix(p)
		self.n = p.shape[0]
		self.distributions = []
		self.sd = None


	########### General methods ###########

	def get_n(self):
		return self.n

	def get_transition_matrix(self):
		return self.p

	########### Methods to manage distributions ###########

	"""
		d: An ndarray where each row is a distribution
	"""
	def add_distributions(self,d):
		# vector shape
		if d.ndim == 1:
			new = {0: d}
			self.distributions.append(new)
		# matrix shape
		else:
			for i in range(d.shape[0]):
				new = {0: d[i,:]}
				self.distributions.append(new)

	def get_num_distributions(self):
		return len(self.distributions)

	def get_distribution(self,index):
		return self.distributions[index][0]

	def get_distribution_tv_mixing(self,index):
		if self.sd == None:
			raise Exception('cant determinde the mixing as long as the stationary distribution is unknown')

		x = []
		tv = []
		
		for idx,t in enumerate(self.get_iteration_times(index)):
			x.append(t)
			tv.append(mkm.total_variation(self.sd,self.get_iteration(index,t)))

		return (x,tv)

	########### Methods to manage iterations ###########

	def add_iteration(self,index,t,iteration):
		self.distributions[index][t] = iteration

	"""
		returns the dictionary of iterations
	"""
	def get_iterations(self,index):
		return self.distributions[index]

	def get_iteration_times(self,index):
		return sorted(self.distributions[index].keys())

	def get_last_iteration_time(self,index):
		return self.get_iteration_times(index)[-1]

	""" The number of iterations is on a per-distribution basis. 
	"""
	def get_num_iterations(self,index):
		return len(self.distributions[index])

	def get_iteration(self,index,t):
		return self.distributions[index][t]

	def get_last_iteration(self,index):
		return self.distributions[index][self.get_last_iteration_time(index)]

	########### Methods to manage the stationary distribution ###########

	def stationary_distribution_known(self):
		return self.sd != None

	def set_stationary_distribution(self,d):
		self.sd = d

	""" Returns stationary distribution or None if unknown
	"""
	def get_stationary_distribution(self):
		return self.sd

	def close_to_stationarity(self,dist):
		if self.sd == None:
			return False

		return mkm.total_variation(dist,self.sd) < 0.05

	########### Iterate the distributions! ###########

	""" Iterate a number of distributions k steps from their last iteration.

	This is the core method for iterating distributions invoked by 
	all other methods.

	It will automatically set the stationary distribution if it is previously unknown
	and reached during the iteration.

	indices: list containing indices of distributions to be iterated
	k: number of iterations to perform
	"""
	def iterate_distributions(self,indices,k):
		x = []
		for i in indices:
			x.append(self.get_last_iteration(i))

		# invoke the external method performing the iteration
		y = mkm.iterate_distributions(self.p,numpy.array(x),k)

		for idx, val in enumerate(indices):
			self.add_iteration(val,self.get_last_iteration_time(val)+k,y[idx,:])

		# did we find the stationary distribution?
		# determine this by checking the defintion of stationarity
		if self.sd == None:
			for i in indices:
				last = self.get_last_iteration(i)
				last_plus_one = self.p.transpose().dot(last).transpose()

				# 1e-6 is a good threshold?
				if mkm.relative_error(last,last_plus_one) < 1e-6:
					self.sd = last

	""" Iterate a number of distributions untill they coincide with the
	stationary distribution.

	indices: list containing indices of distributions to be iterated
	"""
	def iterate_distributions_to_stationarity(self,indices,recursion_first_call=True):
		import time

		if recursion_first_call:
			print "INFO: Iterating "+`len(indices)`+" distribution(s) for a Markov chain with n="+`self.get_n()`+"."

		# iterate only distributions that are not already close to stationarity
		# and determine the number of steps to iterate
		iteration_indices = []
		k = 1000000					# perform maximally 1 Mio. steps at once

		for i in indices:
			if not(self.close_to_stationarity(self.get_last_iteration(i))):
				iteration_indices.append(i)
				k = min(k,self.get_last_iteration_time(i))

		# break recursion
		if len(iteration_indices) == 0:
			return

		# iterate		
		k = max(k,1)

		start = time.time()
		self.iterate_distributions(iteration_indices,k)
		end = time.time()
		seconds = end-start

		print time.strftime("%d %b %H:%M", time.localtime())+": "+`k`+" step(s) completed (that took %(sec).2f seconds)." % {'sec': seconds}

		# recursion rulez
		self.iterate_distributions_to_stationarity(indices,recursion_first_call=False)	

	def iterate_all_distributions_to_stationarity(self):
		self.iterate_distributions_to_stationarity(range(self.get_num_distributions()))
		return

	########### And everything else ###########

	def print_info(self):
		print "The Markov chain has %d distributions with iterations saved at the following timesteps:" % (self.get_num_distributions())
		for d in range(self.get_num_distributions()):
			print self.get_iteration_times(d)

		print "The distributions:"
		for d in range(self.get_num_distributions()):
			print self.get_distribution(d)	

		print "The latest iterations are:"
		for d in range(self.get_num_distributions()):
			print self.get_last_iteration(d)

