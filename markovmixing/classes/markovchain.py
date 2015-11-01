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
		""" Returns the size of the state space.
		"""
		return self.n

	def get_transition_matrix(self):
		""" Returns the transition matrix.
		"""
		return self.p

	########### Methods to manage distributions ###########

	def add_distributions(self,d):
		""" Add a number of distributions to the Markov chain.

		A distribution in the sense of this class is an initial
		distribution on the state space.

			d: An ndarray where each row is a distribution
		"""
		# vector shape
		if d.ndim == 1:
			if  mkm.is_distribution(d) == False:
				raise Exception('not a probability distribution')

			new = {0: d}
			self.distributions.append(new)
		# matrix shape
		else:
			for i in range(d.shape[0]):
				if  mkm.is_distribution(d[i,:]) == False:
					raise Exception('not a probability distribution')

				new = {0: d[i,:]}
				self.distributions.append(new)

	def add_random_delta_distributions(self,k=1):
		""" Convenience method that adds random delta distribution 
		to the Markov chain.

			k: Number of distributions to add. Defaults to 1.
		"""
		self.add_distributions(mkm.random_delta_distributions(self.n, k))

	def num_distributions(self):
		""" Returns the number of distributions.
		"""
		return len(self.distributions)

	def get_distribution(self,index):
		""" Get a distribution. Distributions are indexed 0,...,num_distributions-1.
		"""
		return self.distributions[index][0]

	########### Methods to manage iterations ###########

	def add_iteration(self,index,t,iteration):
		""" Add an iteration for a distribution.

		In the terminology of this class the iteration for a distribution
		at time t is defined by distribution*P^{t}.

		index: index of the distribution
		t: iteration time
		iteration: the iteration
		"""
		if  mkm.is_distribution(iteration) == False:
			raise Exception('not a probability distribution')

		self.distributions[index][t] = iteration

	def get_iterations(self,index):
		"""Returns the dictionary of all iterations for a given
		distribution, including the distribution itself at t=0.

		index: index of the distribution
		"""
		return self.distributions[index]

	def get_iteration_times(self,index):
		""" Returns a sorted list containing all iteration times 
		for a given distribution, including t=0.

		index: index of the distribution
		"""
		return sorted(self.distributions[index].keys())

	def num_iterations(self,index):
		""" Returns the number of iterations for a given distribution.

		index: index of the distribution
		"""
		return len(self.distributions[index])

	def get_iteration(self,index,t):
		""" Returns an iterations for a given distribution if present.

		index: index of the distribution
		t: time at which the iteration is requested
		"""
		return self.distributions[index][t]

	def next_iteration_time(self,index,t):
		"""
		Get the next iteration time strictly after
		time 't' for distribution 'index'.

		index: index of the distribution
		"""
		times = ([ x for x in self.get_iteration_times(index) if x > t])

		if len(times) == 0:
			raise Exception('There is no iteration after time %d' % (d))

		return times[0]
		

	def last_iteration_time(self,index):
		""" Returns the last iteration time for a given distribution.

		index: index of the distribution
		"""
		return self.get_iteration_times(index)[-1]

	def get_last_iteration(self,index):
		""" Return the last iteration of a given distribution.

		index: index of the distribution
		"""
		return self.distributions[index][self.last_iteration_time(index)]

	########### Methods to manage the stationary distribution ###########

	def stationary_known(self):
		""" Is the stationary distribution known?
		"""
		return self.sd != None

	def set_stationary(self,d):
		""" Set the stationary distribution.

		d: the stationary distribution
		"""
		if  mkm.is_distribution(d) == False:
			raise Exception('not a probability distribution')

		self.sd = d

	def get_stationary(self):
		""" Returns the stationary distribution or None if unknown
		"""
		return self.sd

	def close_to_stationarity(self,dist):
		""" For a distribution dist, determine if it is "close"
		to the stationary distribution.

		Currently implemented as total variation distance < 0.05.
		"""
		if self.sd == None:
			return False

		if mkm.total_variation(dist,self.sd) < 0.05:
			return True

	########### Iterate the distributions! ###########

	def iterate_distributions(self,indices,k):
		""" Iterate a number of distributions k steps from their last iteration.

		This is the core method for iterating distributions to stationarity.

		It will automatically set the stationary distribution if it is previously unknown
		and reached during the iteration.

		indices: list containing indices of distributions to be iterated
		k: number of iterations to perform
		"""
		x = []
		for i in indices:
			x.append(self.get_last_iteration(i))

		# invoke the external method performing the iteration
		y = mkm.iterate_distributions(self.p,numpy.array(x),k)

		for idx, val in enumerate(indices):
			self.add_iteration(val,self.last_iteration_time(val)+k,y[idx,:])

		# did we find the stationary distribution?
		# determine this by checking the defintion of stationarity
		# selecting the number of iterations and the threshold is a numerical task
		if not(self.stationary_known()):
			for i in indices:
				last = self.get_last_iteration(i)
				last_iterated = mkm.iterate_distributions(self.p,last,100)
				
				# 1e-6 is a good threshold?
				if mkm.relative_error(last,last_iterated) < 1e-6:
					self.sd = last

	def iterate_distributions_to_stationarity(self,indices,recursion_first_call=True):
		""" Iterate a number of distributions untill they coincide with the
		stationary distribution.

		indices: list containing indices of distributions to be iterated
		"""
		import time

		# iterate only distributions that are not already close to stationarity
		# and determine the number of steps to iterate
		iteration_indices = []
		k = 1000000					# perform maximally 1 Mio. steps at once

		for i in indices:
			if not(self.close_to_stationarity(self.get_last_iteration(i))):
				iteration_indices.append(i)
				k = min(k,self.last_iteration_time(i))

		# nothing to do (break recursion)
		if len(iteration_indices) == 0:
			return

		if recursion_first_call:
			print "INFO: Iterating "+`len(indices)`+" distribution(s) to stationarity for a Markov chain with n="+`self.get_n()`+"."

		# iterate		
		k = max(k,1)

		start = time.time()
		self.iterate_distributions(iteration_indices,k)
		seconds = time.time()-start

		print time.strftime("%d %b %H:%M", time.localtime())+": "+`k`+" iteration step(s) completed (that took %(sec).2f seconds)." % {'sec': seconds}

		# recursion rulez
		self.iterate_distributions_to_stationarity(indices,recursion_first_call=False)	

	def iterate_all_distributions_to_stationarity(self):
		self.iterate_distributions_to_stationarity(range(self.num_distributions()))
		return

	def refine_iterations(self,indices,refine):
		"""
		Refine the iterations (that is add iterations in between existing iterations)
		until they meet a certain criterion.

		indices: list containing indices of distributions to be iterated
		
		refine: boolean function of two iterations where a return value of
		'True' means that an additional iteration in between the 
		given iterations is necessary
		"""
		import time

		print "INFO: Refining "+`len(indices)`+" distribution(s) for a Markov chain with n="+`self.get_n()`+"."
		print "INFO: Iterating distributions one after another."

		for i in indices:
			t = 0

			while t != self.last_iteration_time(i):
				next_t = self.next_iteration_time(i,t)

				while t+1 != next_t and refine(self.get_iteration(i,t), self.get_iteration(i,next_t)) == True:
					
					k = int((next_t-t)/2)

					start = time.time()
					y = mkm.iterate_distributions(self.p,self.get_iteration(i,t),k)
					seconds = time.time() - start
					print time.strftime("%d %b %H:%M", time.localtime())+": "+`k`+" iteration step(s) completed (that took %(sec).2f seconds)." % {'sec': seconds}

					self.add_iteration(i,t+k,y)
					
					next_t = self.next_iteration_time(i,t)

				t = next_t
		return

	########### Find mixing in total variation ###########

	def compute_tv_mixing(self,indices=None):
		""" Compute the mixing in total variation for a number of 
		distributions. 

		Automatically iterates the distributions to stationarity if necessary.

		indices: list containing indices of distributions for which to find the
		total variation mixing. Defaults to None (for all distributions).

		Returns nothing.
		"""
		if indices == None:
			indices = range(self.num_distributions()) 

		self.iterate_distributions_to_stationarity(indices)

		# want subsequent iterations to have a maximal difference of 0.1 units 
		# of total variation (as compared to the stationary distribution)		
		self.refine_iterations(indices, lambda x,y: abs(mkm.total_variation(x,self.get_stationary()) - mkm.total_variation(y,self.get_stationary())) > 0.1 )

	########### And everything else ###########

	def print_info(self):
		print "This is a Markov chain with n=%d. The transition matrix is:"  % (self.get_n())
		print self.get_transition_matrix()

		if self.stationary_known():
			print "The stationary distribution is:"
			print self.get_stationary()
		else:
			print "The stationary distribution in unknown."

		print "The Markov chain has %d distributions with iterations saved at the following timesteps:" % (self.num_distributions())
		for d in range(self.num_distributions()):
			print self.get_iteration_times(d)

		print "The distributions:"
		for d in range(self.num_distributions()):
			print self.get_distribution(d)	

		print "The latest iterations are:"
		for d in range(self.num_distributions()):
			print self.get_last_iteration(d)

	def distribution_tv_mixing(self,index):
		""" Returns a tupel (t,tv) that contains the distance in total variation
		to stationarity for the given distribution at all known times t.

		Iterates the distribution to stationarity if necessary.

		index: index of the distribution
		"""
		self.compute_tv_mixing(indices=[index])

		x = []
		tv = []
		
		for idx,t in enumerate(self.get_iteration_times(index)):
			x.append(t)
			tv.append(mkm.total_variation(self.sd,self.get_iteration(index,t)))

		return (x,tv)

	def plot_tv_mixing(self,index):
		""" Plots the total variation mixing for a given distribution.

		Iterates the distribution to stationarity if necessary.

		index: index of the distribution of wich the mixing should be plotted
		"""
		import matplotlib.pyplot as plt

		(x,tv) = self.distribution_tv_mixing(index)

		plt.plot(x, tv)
		plt.title("Convergence to the stationary distribution")
		plt.xlabel("n")
		plt.ylabel("Total variation distance to the stationary distribution")

		plt.show()

	def plot_iteration(self,index,t):
		""" Plots an iteration (a probability distribution). 

		The iteration has to exist.

		index: index of the distribution of which the iteration should be plotted
		t: iteration time
		"""
		import matplotlib.pyplot as plt

		iteration = self.get_iteration(index,t)

		plt.plot(numpy.arange(self.n), iteration)
		plt.title("Probability distribution after %d steps" % (t))
		plt.xlabel("Markov chain state space")
		plt.ylabel("Probabiliy")
		plt.ylim(0, 1.1*numpy.max(iteration))

		plt.show()

	def sample_path(self, x0, length):
		""" Sample a path of the Markov chain.

		x0 : initial state at t=0 
		length: length of the path

		Returns a list containg the path.
		"""
		path = [x0]
		r = numpy.random.random(length-1)

		for i in xrange(length-1):
			x = path[-1]
			d = self.p[x,:]
			cumsum = 0.

			for y in d.nonzero()[1]:
				cumsum = cumsum + d[0,y]
				
				if r[i] < cumsum:
					path.append(y)
					break
			
		return path

			