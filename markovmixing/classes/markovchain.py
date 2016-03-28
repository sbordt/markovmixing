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
		""" Returns an iteration for a given distribution if present.

		index: index of the distribution
		t: time at which the iteration is requested
		"""
		return self.distributions[index][t]

	def previous_iteration_time(self,index,t):
		""" Get the iteration time strictly before
		time 't' for distribution 'index'.

		index: index of the distribution
		"""
		times = ([ x for x in self.get_iteration_times(index) if x < t])

		if len(times) == 0:
			raise Exception('There is no iteration before time %d' % (t))

		return times[-1]

	def next_iteration_time(self,index,t):
		""" Get the next iteration time strictly after
		time 't' for distribution 'index'.

		index: index of the distribution
		"""
		times = ([ x for x in self.get_iteration_times(index) if x > t])

		if len(times) == 0:
			raise Exception('There is no iteration after time %d' % (t))

		return times[0]
		
	def last_iteration_time(self,index):
		""" Returns the last iteration time for a given distribution.

		index: index of the distribution
		"""
		return self.get_iteration_times(index)[-1]

	def closest_iteration_time(self,index,t):
		""" Returns a time where an iterations exists, as close
		as possible to t.

		index: index of the distribution
		"""
		if  t<0:
			raise Exception('t must be greater or equal than zero')

		for j,time in enumerate(self.get_iteration_times(index)):
			if t < time: # this can never happen for j=0
				prev_time = self.get_iteration_times(index)[j-1]

				if time-t<=t-prev_time:
					return time
				return prev_time

		return self.last_iteration_time(index)

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

	def close_to_stationarity(self,dist,tv_tol=0.05):
		""" For a distribution dist, determine if it is "close"
		to the stationary distribution.

		Currently implemented as total variation distance < tv_tol.

		tv_tol: Desired tolerance in total variation. Defaults to 0.05.
		"""
		if self.sd == None:
			return False

		if mkm.total_variation(dist,self.sd) < tv_tol:
			return True

	########### Iterate the distributions! ###########

	def iterate_distributions(self,indices,k,tv_tol=0.05):
		""" Iterate a number of distributions k steps from their last iteration.

		This is the core method for iterating distributions to stationarity.

		If the number of distributions is >= 3, it will automatically 
		set the stationary distribution, if unknown and found.

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

		# check if we found the stationary distribution
		if len(indices) >= 3:
			# first check the definition of stationarity for all iterations
			for i in indices:
				last = self.get_last_iteration(i)
				last_iterated = mkm.iterate_distributions(self.p,last,100)
						
				# is 1e-6 a good threshold? (however this can never be trusted -> cutoff)
				if mkm.relative_error(last,last_iterated) > 1e-6:
					return

			# check pairwise distance <= tol
			for i in indices:
				d_i = self.get_last_iteration(i)

				for j in indices:
					d_j = self.get_last_iteration(j)

					if mkm.total_variation(d_i, d_j) > tv_tol:
						return

			# take the arithmetic mean of all iterations as the stationary distribution
			stationary_candidate = numpy.zeros(self.n)

			for i in indices:
				stationary_candidate = stationary_candidate + self.get_last_iteration(i)

			stationary_candidate = stationary_candidate / len(indices)

			self.set_stationary(stationary_candidate)			

	def iterate_distributions_to_stationarity(self,indices,tv_tol=0.05,recursion_first_call=True):
		""" Iterate a number of distributions untill they coincide with the
		stationary distribution.

		indices: list containing indices of distributions to be iterated
		tv_tol: tolerance in total variation distance for the determination of convergence.
		Defaults to 0.05
		"""
		import time

		# if the stationary distribution is unknown, add two random delta distributions
		if recursion_first_call and not(self.stationary_known()) and len(indices)<3:
			self.add_random_delta_distributions(2)
			indices.extend([self.num_distributions()-2,self.num_distributions()-1])
	
			print "INFO: Added two random delta distributions, since the stationary distribution is unknown. They will help to find the stationary distribution."

		# iterate only distributions that are not already close to stationarity
		# and determine the number of steps to iterate
		iteration_indices = []
		k = 1000000					# at most 1 Mio. steps at once

		for i in indices:
			if not(self.close_to_stationarity(self.get_last_iteration(i), tv_tol=tv_tol)):
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
		self.iterate_distributions(iteration_indices,k,tv_tol=tv_tol)
		seconds = time.time()-start

		print time.strftime("%d %b %H:%M", time.localtime())+": "+`k`+" iteration step(s) completed (that took %(sec).2f seconds)." % {'sec': seconds}

		# recursion rulez
		self.iterate_distributions_to_stationarity(indices, tv_tol=tv_tol, recursion_first_call=False)	

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
		given iterations is necessary (i.e. refine(t1, iteration1, t2, iteration2))
		"""
		import time

		print "INFO: Refining "+`len(indices)`+" distribution(s) for a Markov chain with n="+`self.get_n()`+"."
		print "INFO: For multiple distributions, refinement might take longer than iterating to stationarity."

		# Iterating distributions one after another
		for i in indices:
			t = 0

			while t != self.last_iteration_time(i):
				next_t = self.next_iteration_time(i,t)

				while t+1 != next_t and refine(t, self.get_iteration(i,t), next_t, self.get_iteration(i,next_t)) == True:

					k = int((next_t-t)/2)

					start = time.time()
					y = mkm.iterate_distributions(self.p,self.get_iteration(i,t),k)
					seconds = time.time() - start

					self.add_iteration(i,t+k,y)
					
					next_t = self.next_iteration_time(i,t)

				t = next_t
		return

		print "INFO: Done."

	def assert_iteration(self,indices,t):
		""" For the distributions given by indices, assert that there exists
		an iteration at time t.

		indices: list with indices of distributions
		t: time 
		"""
		for i in indices:
			t0 = 0

			for t1 in self.get_iteration_times(i):
				if t1 > t0 and t1 <= t:
					t0 = t1

			if t0 != t:
				y = mkm.iterate_distributions(self.p,self.get_iteration(i,t0),t-t0)
				self.add_iteration(i,t,y)

	########### Methods to determine the mixing in total variation ###########

	def compute_tv_mixing(self,indices=None,convergence_tol=0.05,refinement_tol=0.1):
		""" Compute the mixing in total variation for a number of 
		distributions. 

		Automatically iterates the distributions to stationarity if necessary.

		indices: list containing indices of distributions for which to find the
		total variation mixing. Defaults to None (for all distributions).
		convergence_tol: tolerance in total variation distance for the determination of convergence.
		Defaults to 0.05. 
		refinement_tol: maximum distance in total variation between two iterations. If the 
		mixing is plotted, this corresponds to the smoothness of the graph. Defaults to 0.1.

		Returns nothing.
		"""
		if indices == None:
			indices = range(self.num_distributions()) 

		self.iterate_distributions_to_stationarity(indices, tv_tol=convergence_tol)

		# want subsequent iterations to have a maximal difference of refinement_tol units 
		# of total variation (as compared to the stationary distribution)		
		self.refine_iterations(indices, lambda t1,x1,t2,x2: abs(mkm.total_variation(x1,self.get_stationary()) - mkm.total_variation(x2,self.get_stationary())) > refinement_tol)

	def distribution_tv_mixing(self,index):
		""" Returns a tupel (t,tv) that contains the distance in total variation
		to stationarity for the given distribution at all known times t.

		index: index of the distribution
		"""
		x = []
		tv = []
		
		for idx,t in enumerate(self.get_iteration_times(index)):
			x.append(t)
			tv.append(mkm.total_variation(self.sd,self.get_iteration(index,t)))

		return (x,tv)

	########### Plot and video creation ###########

	def plot_tv_mixing(self,indices=None,y_tol=0.1,threshold=0.05,text=True):
		""" Plots the total variation mixing for a given number
		of distributions.

		Iterates the distributions to stationarity if necessary.

		The x-limit of the plot will be choosen such that the total varation distance
		to stationarity of all distributions is below threshold.

		indices: list with indices of distributions for which to plot the 
		mixing. Can also be an integer, indicating a single distribution.
		Defaults to None (for all distributions).
		
		y_tol: maximum y-distance between to data points on the same graph

		threshold: determines the x-limit of the plot. Defaults to 0.05.

		text: if the plot should contain axis labels and a title
		
		"""
		import matplotlib.pyplot as plt

		if indices == None:
			indices = range(self.num_distributions()) 

		if isinstance(indices, int):
			indices = [indices]

		# iterate distributions to stationarity, given the desired threshold
		self.compute_tv_mixing(indices, convergence_tol=threshold, refinement_tol=y_tol)

		# determine the x-limit of the plot
		xlim = 5

		for index in indices:
			for t in self.get_iteration_times(index):
				if t > xlim and mkm.total_variation(self.sd,self.get_iteration(index,t)) > threshold:
					xlim = t

		self.assert_iteration(indices, xlim)
		self.compute_tv_mixing(indices, convergence_tol=threshold, refinement_tol=y_tol) # need the refinement

		# plot
		for index in indices:
			x = []
			tv = []
					
			for t in self.get_iteration_times(index):
				if t > xlim:
					continue
				x.append(t)
				tv.append(mkm.total_variation(self.sd,self.get_iteration(index,t)))

			plt.plot(x, tv)

		plt.xlim(0, xlim)
		plt.ylim(0, 1)

		if text:
			plt.xlabel("t")
			plt.ylabel("Total variation distance to stationarity")

		plt.show()

	def plot_iteration(self,index,t):
		""" Plots an iteration (a probability distribution). 

		The iteration has to exist.

		index: index of the distribution of which the iteration should be plotted
		t: iteration time
		"""
		import matplotlib.pyplot as plt

		iteration = self.get_iteration(index,t)

		mkm.pyplot_bar(iteration)

		plt.title("Probability distribution after %d steps" % (t))
		plt.xlabel("Markov chain state space")
		plt.ylabel("Probabiliy")

		plt.tick_params(axis='x', which='both', bottom='off', top='off', labelbottom='off')
		
		plt.xlim(0, self.n)
		plt.ylim(0, 1.1*numpy.max(iteration))

		plt.show()

	def convergence_video(self,path,index,seconds):
		"""
		"""
		import matplotlib.pyplot as plt

		nframes = 100*seconds

		# first iterate the distribution to stationarity (if that has not been done already)
		self.iterate_distributions_to_stationarity([index])

		# we want the video to end once the stationary distribution is reached
		t_end = self.last_iteration_time(index)

		for t in self.get_iteration_times(index):
			if mkm.total_variation(self.get_iteration(index,t), self.get_stationary()) < 0.01:
				t_end = int(t*1.05)
				break

		frametime = t_end/float(nframes)

		# if possible, we want an iteration for every frame
		self.refine_iterations([index], lambda t1,x1,t2,x2: (t1<=t_end or t2<=t_end) and abs(t1-t2) > frametime)

		def frame(i):
			fig = plt.figure(figsize=(19.20, 10.80), dpi=100)

			# time of closest iteration
			t = self.closest_iteration_time(index,i*frametime)

			iteration = self.get_iteration(index,t)
			mkm.pyplot_bar(iteration)
		
			plt.title("Probability distribution after %d steps" % (t))
			plt.xlabel("Markov chain state space")
			plt.ylabel("Probabiliy")

			plt.tick_params(axis='x', which='both', bottom='off', top='off', labelbottom='off')
		
			plt.xlim(0, self.n)
			plt.ylim(0, 1.1*numpy.max(iteration))

			return fig

		mkm.matplotlib_plots_to_video(path, frame, nframes)

	########### Miscellaneous methods ###########

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

			