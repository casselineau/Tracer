import numpy as N

class Estimator(object):
	def __init__(self, n_sigmas=3.):
		self.value = 0.
		self.Qsum = 0.
		self.p = 0.
		self.CI = N.inf
		self.n_sigmas = n_sigmas

	def update(self, value, weight=1.):
		self.weight = weight
		self.Qsum += self.p/(self.p+self.weight)*(value-self.value)**2.
		self.value = (self.value*self.p+value)/(self.p+self.weight)
		self.CI = self.get_CI()
		self.p += self.weight

	def get_CI(self):
		# Returns the confidence interval value in relative terms ie. divided by the current estimation value.
		stdev = self.get_stdev()
		CI = N.array(self.n_sigmas*stdev/self.value)
		CI[stdev==0.] = 0.
		return CI

	def get_stdev(self):
		if self.p == 0.:
			return N.inf*N.ones(self.value.shape)
		return N.sqrt(self.Qsum/self.p)/N.sqrt(self.p+self.weight)

