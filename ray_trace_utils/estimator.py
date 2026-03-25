import numpy as N

class Estimator(object):
	'''
	A class to automate a weighted Welford algorithm for batches of samples (rays).
	'''
	def __init__(self, n_sigmas=3., relative_CI=True):
		self.mean = N.array([0.])
		self.M2 = N.array([0.])
		self.n = 0.
		self.n2 = 0.
		self.n_sigmas = n_sigmas
		self.relative_CI = relative_CI

	def update(self, values, num_samples):
		delta = values - self.mean
		if self.n == 0.:
			self.n += num_samples
			self.mean = num_samples * delta / self.n
			self.M2 = num_samples * delta * (values - self.mean)
		else:
			self.n += num_samples
			self.mean += num_samples * delta / self.n
			self.M2 += num_samples * delta * (values - self.mean)
		self.n2 += num_samples**2.

	def get_CI(self):
		# Returns the confidence interval value in relative terms if the self.relative_CI is True ie. divided by the current estimation value.
		if self.n == 0:
			return N.inf*N.ones(self.mean.shape)
		denom = (self.n-self.n2/self.n)
		while denom <= 0:
			return N.inf*N.ones(self.mean.shape)
		stdev = N.sqrt(self.M2/denom)
		CI = self.n_sigmas*stdev/N.sqrt(self.n**2/self.n2)
		if self.relative_CI:
			CI = CI/self.mean
		CI[stdev==0.] = 0.
		return CI

def MCRT_to_CI(fun, target_CI, num_samples, n_sigmas=3., *args, **kwargs):
	'''
	This function automates a Monte-Carlo simulation until a confidence interval is reached.
	:param fun: ray-tracing function
	:param num_samples: number of rays er iteration
	:param target_CI: targeted confidence interval
	:param n_sigmas: number of sigmas
	:return:
	'''
	estimator = Estimator(n_sigmas)
	while estimator.get_CI()>target_CI:
		samples = fun(num_samples=num_samples, *args, **kwargs)
		estimator.update(samples, num_samples=num_samples)
		print ('Mean: %5.1f, CI: %.5f -> %.5f \033[F'%(estimator.mean, estimator.get_CI(), target_CI))
	print (' ')
	return estimator

