import numpy as N
from scipy.interpolate import RegularGridInterpolator

class PW_linear_distribution(object):

	def __init__(self, xs, ys):
		self.xs = N.round(xs, decimals=8)
		self.ys = N.round(ys, decimals=8)
		self.a = (self.ys[1:] - self.ys[:-1])/(self.xs[1:] - self.xs[:-1])
		self.b = self.ys[:-1]-self.a*self.xs[:-1]
		self.integ = (xs[1:]-xs[:-1])*(ys[1:]+ys[:-1])/2.
		self.tot_integ = N.sum(self.integ)
		self.PDF_def = self.ys/self.tot_integ
		self.CDF_def = N.add.accumulate(N.hstack([[0], self.integ]))/self.tot_integ

	def find_slice(self, x):
		comp = (N.round(x, decimals=7)>N.vstack(self.xs)).T
		locs = N.sum(comp, axis=1)-1
		return locs

	def __call__(self, x):
		loc = self.find_slice(x)
		a, b = self.a[loc], self.b[loc]
		return a*x+b

	def PDF(self, x):
		return self(x)/self.tot_integ

	def CDF(self, x):
		loc = self.find_slice(x)
		return self.CDF_def[loc] + (x-self.xs[loc])*(self.PDF(x)+self.PDF_def[loc])/2.
		
	def sample(self, ns):
		x_samples = N.zeros(ns)
		R = N.random.uniform(size=ns)
		a = self.a/(2.*self.tot_integ)
		b = self.b/self.tot_integ
		c = self.CDF_def[:-1] - a*self.xs[:-1]**2-b*self.xs[:-1]
		slice_locs = N.logical_and((R >= N.vstack(self.CDF_def[:-1])), (R < N.vstack(self.CDF_def[1:])))
		for i in range(len(self.CDF_def)-1):
			if slice_locs[i].any():
				if a[i] == 0.:
					x_samples[slice_locs[i]] = self.xs[i] + (R[slice_locs[i]]-self.CDF_def[i]) / b[i]
				else:
					C = c[i]-R[slice_locs[i]]
					D = b[i]**2.-4.*a[i]*C
					x2 = (-b[i]+N.sqrt(D))/(2.*a[i])
					x_samples[slice_locs[i]] = x2
		weights = N.ones(len(R))
		return x_samples, weights

class PW_bilinear_distribution(object):
	def __init__(self, xu, yu, zs):
		'''
		Works with regular distributions, ie. there is a grid formed by xu and yu and we only give the axis coordinates of the grid. 
		xu, yu - Unique values of the grid coordinates
		zs - (len(xy), len(yu)) array of zs values.
		'''
		self.xu = xu
		self.yu = yu
		self.zs = zs
		
		# Calculate f_x, the distribution along x, integrating over y.
		f_x = N.zeros(len(xu))
		for i, x in enumerate(xu):
			f_x[i] = PW_linear_distribution(yu, zs[i]).tot_integ
		self.dist_x = PW_linear_distribution(xu, f_x)
		self.tot_integ = self.dist_x.tot_integ
		self.interpolator = RegularGridInterpolator((xu,yu), zs)
		
	def __call__(self, x, y):
		return self.interpolator((x, y))
		
	def PDF(self, x, y):
		return self(x, y)/self.tot_integ
		
	def sample(self, ns):
		# Sample dist_x
		x_samples, weights_x = self.dist_x.sample(ns)
		# With samples found for x, compute the the PDF for y given the x-sampled values and sample y.
		# It is faster here to use importance sampling over the xu values intervals. Thsi way we use a common sampling function for all of them and only need weighting
		y_samples = N.zeros(ns)
		weights = N.zeros(ns)
		for i in range(len(self.xu)-1):
			loc = N.logical_and(x_samples>=self.xu[i], x_samples<self.xu[i+1])
			if loc.any():
				x_mid = N.average(x_samples[loc])
				dist_y_s = PW_linear_distribution(self.yu, self(x_mid*N.ones(len(self.yu)), self.yu))
				# Sampling distribution
				y_samples[loc], weights_y_s = dist_y_s.sample(N.sum(loc))
				# Conditional PDF
				p_ygx = self.PDF(x_samples[loc], y_samples[loc])/self.dist_x.PDF(x_samples[loc])
				# Importance sampling
				weights[loc] = p_ygx*weights_y_s
		return x_samples, y_samples, weights
		
class PW_lincos_distribution(PW_linear_distribution):
	def __init__(self, xs, ys):
		'''
		xs and ys are the piecewise linear function values. The cos(xs) factor is added within the class
		'''
		PW_linear_distribution.__init__(self, xs, ys)
		self.a = (self.ys[1:] - self.ys[:-1])/(self.xs[1:] - self.xs[:-1])
		self.b = self.ys[:-1]-self.a*self.xs[:-1]
		self.integ = ys[1:]*N.sin(xs[1:])-ys[:-1]*N.sin(xs[:-1]) + self.a*(N.cos(xs[1:])-N.cos(xs[:-1])) # Obtained throught integration by parts. 
		self.tot_integ = N.sum(self.integ)
		self.PDF_def = self.ys*N.cos(self.xs)/self.tot_integ
		self.CDF_def = N.add.accumulate(N.hstack([[0], self.integ]))/self.tot_integ

	def __call__(self, x):
		loc = self.find_slice(x)
		return (self.a[loc]*x+self.b[loc])*N.cos(x)

	def CDF(self, x):
		loc = self.find_slice(x)
		return self.CDF_def[loc] + (self(x)*N.sin(x)-self.ys[loc]*N.sin(self.xs[loc]) + self.a[loc]*(N.cos(self(x))-N.cos(self.xs[loc])))
		
	def sample(self, ns):
		x_s, weights_s = super().sample(ns)
		weights = weights_s*self.PDF(x_s)
		weights /= (N.sum(weights/float(ns))) # takes care of rounding errors
		return x_s, weights
	
class BDRF_distribution_noinc(object):
	def __init__(self, th_u, phi_u, bdrf):
		'''
		Works with any piecewise linear distribution, ie. there is a grid formed by th_u and phi_u and we only give the axis coordinates of the grid. 
		th_u, phi_u - Unique values of the grid coordinates
		bdrf - (len(th_u), len(phi_u)) array of bdrf values.
		This implementation returns samples that include the cosine factor: the resulting weights are applied directly to the energy of the incident rays.
		This _noinc does not give any indication about the incident angle
		'''
		self.th_u = th_u
		self.phi_u = phi_u
		self.bdrf = bdrf
		
		# Calculate f_x, the distribution along x, integrating over y.
		f_th = N.zeros(len(th_u))
		for i, th in enumerate(th_u):
			f_th[i] = PW_linear_distribution(phi_u, bdrf[i]*N.cos(th)).tot_integ
		self.dist_th = PW_lincos_distribution(th_u, f_th/N.cos(th_u)) # we have to remove the cosine factor as it is added in the lincos distribution
		self.tot_integ = self.dist_th.tot_integ
		self.interpolator = RegularGridInterpolator((th_u, phi_u), bdrf)
		
	def __call__(self, th, phi):
		return self.interpolator((th, phi))*N.cos(th)
		
	def PDF(self, th, phi):
		return self(th, phi)/self.tot_integ
		
	def sample(self, ns):
		# Sample dist_th
		th_samples, weights_th = self.dist_th.sample(ns)
		# With samples found for x, compute the the PDF for y given the x-sampled values and sample y.
		# It is faster here to use importance sampling over the th_u values intervals. Thsi way we use a common sampling function for all of them and only need weighting
		phi_samples = N.zeros(ns)
		weights = N.zeros(ns)
		for i in range(len(self.th_u)-1):
			loc = N.logical_and(th_samples>=self.th_u[i], th_samples<self.th_u[i+1])
			if loc.any():
				th_mid = N.average(th_samples[loc])
				# Sampling distribution
				dist_phi_s = PW_linear_distribution(self.phi_u, self(th_mid*N.ones(len(self.phi_u)), self.phi_u) )
				phi_samples[loc], weights_phi_s = dist_phi_s.sample(N.sum(loc))
				# Conditional PDF
				p_phigth = self.PDF(th_samples[loc], phi_samples[loc])/self.dist_th.PDF(th_samples[loc])#weights_th[loc]#weights_phi_s # This is because these weights form the ps distribution are perfect inverse of the PDF
				# Importance sampling
				weights[loc] = p_phigth*weights_phi_s
		return th_samples, phi_samples, weights
		
class BDRF_distribution(object):
	# make a more afficient class that can deal with having an interpolator as an input.
	def __init__(self, interpolator):
		'''
		Works with any piecewise linear distribution.
		Input: interpolator able to return the piecewise linear interpolation of the BDRF. The interpolator contains all the information needed about the definition grid.
		This implementation returns samples that include the cosine factor: the resulting weights are applied directly to the energy of the incident rays.
		'''
		self.th_u, self.phi_u = interpolator.grid[2:]
		self.interpolator = interpolator
		
	def integrate(self, th_i, phi_i):
		# Calculate f_x, the distribution along x, integrating over y.
		f_th = N.zeros(len(self.th_u))
		for i, th in enumerate(self.th_u):
			f_th[i] = PW_linear_distribution(self.phi_u, self(th_i, phi_i, th, self.phi_u)).tot_integ
		self.dist_th = PW_lincos_distribution(self.th_u, f_th/N.cos(self.th_u)) # we have to remove the cosine factor as it is added in the lincos distribution
		self.tot_integ = self.dist_th.tot_integ
		
	def theoretical_histogram(self, th_i, phi_i):
		hist = N.zeros((len(self.th_u)-1,len(self.phi_u)-1))
		f_th = N.zeros((len(self.th_u), len(self.phi_u)-1))
		for i, th in enumerate(self.th_u):
			f_th[i] = PW_linear_distribution(self.phi_u, self(th_i, phi_i, th, self.phi_u)).integ
		for j, phi in enumerate(self.phi_u[:-1]):
			hist[:,j] = PW_lincos_distribution(self.th_u, f_th[:,j]/N.cos(self.th_u)).integ
		return N.cos(th_i)*hist
		
	def DHR(self, th_i, phi_i):
		return N.sum(self.theoretical_histogram(th_i, phi_i))
		
	def __call__(self, th_i, phi_i, th, phi):
		return self.interpolator((th_i, phi_i, th, phi))*N.cos(th)
		
	def PDF(self, th_i, phi_i, th, phi):
		return self(th_i, phi_i, th, phi)/self.tot_integ
		
	def sample(self, th_i, phi_i, ns):
		self.integrate(th_i, phi_i)
		# Sample dist_th
		th_samples, weights_th = self.dist_th.sample(ns)
		# With samples found for x, compute the the PDF for y given the x-sampled values and sample y.
		# It is faster here to use importance sampling over the th_u values intervals. This way we use a common sampling function for all of them and only need weighting
		phi_samples = N.zeros(ns)
		weights = N.zeros(ns)
		for i in range(len(self.th_u)-1):
			loc = N.logical_and(th_samples>=self.th_u[i], th_samples<self.th_u[i+1])
			if loc.any():
				th_mid = N.average(th_samples[loc])
				# Sampling distribution
				dist_phi_s = PW_linear_distribution(self.phi_u, self(th_i, phi_i, th_mid*N.ones(len(self.phi_u)), self.phi_u) )
				phi_samples[loc], weights_phi_s = dist_phi_s.sample(N.sum(loc))
				# Conditional PDF
				p_phigth = self.PDF(th_i, phi_i, th_samples[loc], phi_samples[loc])/self.dist_th.PDF(th_samples[loc])# This is because these weights form the ps distribution are perfect inverse of the PDF
				# Importance sampling
				weights[loc] = p_phigth*weights_phi_s
		return th_samples, phi_samples, weights

def pw_linear_importance_sampling(dist, ns):
	'''
	dist is the nonlinear piecewise 1D distribution we want to sample from.
	ns the number of samples
	'''
	sampling_dist = PW_linear_distribution(dist.xs, dist(dist.xs))
	x_s, weights_s = dist.sample(ns)
	weights = weights_s*dist.PDF(x_s)
	weights /= (N.sum(weights/float(ns))) # takes care of rounding errors
	return x_s, weights
	
def disk_sampling(r_ext, ns, normal_up=True):
	ths = N.random.uniform(size=ns)*2.*N.pi
	rs = N.sqrt(N.random.uniform(size=ns))*r_ext
	positions = N.vstack([rs*N.cos(ths), rs*N.sin(ths), N.zeros(ns)])
	normals = N.vstack([N.zeros(ns), N.zeros(ns), N.ones(ns)])
	if normal_up == False:
		normals = -normals
	return positions, normals
	
def cylinder_sampling(r_ext, h, ns, normal_in=False, volume=False):
	'''
	Uniformly samples a cylinder surface or volume if volume argument is True
	Surface sampling also returns surface normal vetcors at the sampled points locations.
	'''
	zs = N.random.uniform(size=ns)*h-h/2.
	if volume == False:
		ths = N.random.uniform(size=ns)*2.*N.pi
		positions = N.vstack([r_ext*N.cos(ths), r_ext*N.sin(ths), zs])
		normals = N.vstack([N.cos(ths), N.sin(ths), N.zeros(ns)])
		if normal_in == True:
			normals = -normals
		return positions, normals
	else:
		positions, normdisk = disk_sampling(r_ext, ns)
		positions[2] = zs
		return positions
	
def sphere_sampling(r_ext, ns, normal_in=False, volume=False):
	'''
	Uniformly samples a sphere surface or volume if volume argument is True
	Surface sampling also returns surface normal vectors at the sampled points locations.
	'''
	phis = N.random.uniform(size=ns)*2.*N.pi
	cosths = N.random.uniform(low=-1., high=1., size=ns) # cosine of polar angle uniformly distributed
	sinths = N.sqrt(1.-cosths**2)
	normals = N.vstack([sinths*N.cos(phis), sinths*N.sin(phis), cosths])
	if volume == False:
		positions = r_ext*normals
		if normal_in == True:
			normals = -normals
		return positions, normals
	else:
		r_s = r_ext*N.cuberoot(N.random.uniform(size=ns))
		positions = r_s*normals
		return positions
	
if __name__ == '__main__':
	import matplotlib.pyplot as plt
	from sys import path
	from time import time
	import cProfile
	
	test_cos = 0
	test_importance_sampling = 0
	test_sample = 0
	test_sampling_full_BDRF = 1

	if test_cos:
		ths = N.linspace(0, N.pi/2., 100)
		plt.figure()
		plt.subplot(121)
		plt.plot(ths, N.cos(ths), label=r'cos($\theta$)')
		C = N.pi/2.
		def f(x, e):
			return 1.-x**e
		def g(x):
			return (N.pi**2.-4.*x**2)/(N.pi**2.+x**2)
		e = 1.75
		plt.subplot(121)
		plt.plot(ths, f(ths/C,e), label=r'1-$(\frac{\theta}{\pi/2})^{%s}$'%e)
		plt.plot(ths, g(ths), label='Bhāskara')
		plt.legend()

		plt.subplot(122)
		plt.plot(ths, f(ths/C,e)-N.cos(ths), label=r'1-$(\frac{\theta}{\pi/2})^{%s}$'%e)
		plt.plot(ths, g(ths)-N.cos(ths), label='Bhāskara')
		plt.legend()
		def gamma(a):
			return N.prod(N.linspace(1, a-1, int(a)))
		def beta(a, b, c, th):
			return c*(th**(a-1.)*(1.-th)**(b-1.))
		#plt.plot(ths, beta(3, 6., 3., (1.-ths/(N.pi/2.))), label='$\Beta$ fit')

		plt.savefig(path[0]+'/cosine.png')

	if test_importance_sampling:
		
		plt.figure()
		n = 3
		ns = 100000
		thetas = N.linspace(0, N.pi/2., n)#N.array([0, N.pi/3., N.pi/2.])
		f = N.linspace(1., 0.1, n)**0.7
		
		dist = PW_lincos_distribution(thetas, f)
		thetas_samples, weights = dist.sample(ns)
		
		sampling_f = PW_linear_distribution(thetas, f*N.cos(thetas))
		
		plt.subplot(121, aspect='equal')
		thetas_p = N.linspace(0., N.pi/2., 1000)
		plt.plot(thetas_p, dist(thetas_p), label='$f(\theta)$')
		plt.plot(thetas_p, sampling_f(thetas_p), label='Sampling f')
		plt.plot(thetas_p, dist.PDF(thetas_p), label='PDF')
		plt.plot(thetas_p, sampling_f.PDF(thetas_p), 'g', label='Sampling PDF')
		plt.legend()

		plt.subplot(122, aspect='equal')

		plt.plot(thetas_p, dist.PDF(thetas_p), color='r', zorder=1000)
		plt.plot(thetas_p, sampling_f.PDF(thetas_p), color='g', zorder=1000)
		plt.hist(thetas_samples, weights=weights/ns, bins=100, density=True)

		plt.savefig(path[0]+'/test_importance_sampling.png')

	if test_sample:
		ns = int(1e6)

		xs = N.linspace(-1, 1, 10)
		ys = N.random.uniform(size=10)
		
		t0 = time()
		dist = PW_linear_distribution(xs, ys)
		xs_sampled, weights = dist.sample(ns)
		print('Linear:', time()-t0, 's')

		plt.figure(figsize=(8,4))
		plt.plot(xs, ys/N.trapz(x=xs, y=ys))	
		plt.hist(xs_sampled, density=True, label='Samples', bins=200)

		plt.legend()
		plt.savefig(path[0]+'/test_pw_linear.png')

		NX, NY = 10, 20
		xs = N.linspace(0., 1., NX)
		ys = N.linspace(0., 2., NY)
		X, Y = N.meshgrid(xs, ys)
		zs = (N.cos(5*X)+N.sin(10*Y)).T
		zs += 2*N.amin(zs)
		integ = (zs[:-1,:-1]+zs[:-1,1:]+zs[1:,1:]+zs[1:,:-1])/4.*2./((NX-1)*(NY-1))
		PDF_th = integ/N.sum(integ)

		t0 = time()
		dist = PW_bilinear_distribution(xs, ys, zs)
		x_samples, y_samples, weights = dist.sample(ns)
		#cProfile.run('sim()', sort='tottime')
		print('Bilinear:', time()-t0, 's')

		plt.figure(figsize=(8,3))

		plt.subplots_adjust(left=0.05, wspace=0.8, right=0.9)
		plt.subplot(131, title='Theoretical histogram', aspect='equal')
		plt.pcolormesh(xs, ys, PDF_th.T)

		plt.subplot(132, title='Numerical sampling', aspect='equal')
		hist, edgesx, edgesy = N.histogram2d(x_samples, y_samples, weights = weights/ns, bins=[xs, ys])
		plt.pcolormesh(xs, ys, hist.T, vmin=N.amin(PDF_th), vmax=N.amax(PDF_th))
		plt.colorbar(label='PDF')

		plt.subplot(133, title='Relative error', aspect='equal')
		rel_error = (hist-PDF_th)/PDF_th*100.
		plt.pcolormesh(xs, ys, rel_error.T, cmap='bwr', vmin=-N.amax(N.abs(rel_error)), vmax=N.amax(N.abs(rel_error)))
		plt.colorbar(label='Relative error (%)')

		plt.savefig(path[0]+'/test_bidirectional_pw_linear.png')
		
		plt.figure(figsize=(8,3))
		
		NX, NY = 10, 20
		xs = N.linspace(0., N.pi/2., NX)
		ys = N.linspace(0., 2.*N.pi, NY)
		X, Y = N.meshgrid(xs, ys)
		zs = N.ones(X.shape).T#((N.pi/2.-X)**2).T

		
		integ = (Y[1:,0]-Y[:-1,0])*N.vstack(((N.sin(xs[1:])-N.sin(xs[:-1]))))
		hist_th = integ/N.sum(integ)

		t0 = time()
		dist = BDRF_distribution(xs, ys, zs)
		x_samples, y_samples, weights = dist.sample(ns)
		#cProfile.run('sim()', sort='tottime')
		print('BDRF:', time()-t0, 's')

		plt.subplots_adjust(left=0.05, wspace=0.3, right=0.95)
		plt.subplot(131, title='Theoretical histogram', aspect='equal', projection='polar')
		plt.pcolormesh(ys, xs, hist_th, vmin=N.amin(hist_th), vmax=N.amax(hist_th))
		plt.colorbar(label='PDF')

		plt.subplot(132, title='Numerical sampling', aspect='equal', projection='polar')
		hist, edgesx, edgesy = N.histogram2d(x_samples, y_samples, weights=weights/ns, bins=[xs, ys])
		plt.pcolormesh(ys, xs, hist)#, vmin=N.amin(PDF_th), vmax=N.amax(PDF_th))
		plt.colorbar(label='PDF')

		plt.subplot(133, title='Relative error', aspect='equal', projection='polar')
		rel_error = (hist-hist_th)/hist_th*100.
		plt.pcolormesh(ys, xs, rel_error, cmap='bwr', vmin=-N.amax(N.abs(rel_error)), vmax=N.amax(N.abs(rel_error)))
		plt.colorbar(label='Relative error (%)')

		plt.savefig(path[0]+'/test_BDRF.png')
		
	if test_sampling_full_BDRF:
		from BDRF_models import regular_grid_Cook_Torrance
		from BDRF_analysis import bdrf_to_dhr
		m, R_Lam, alpha = 1.1, 0.5, 0.2
		ns = int(1e6)
		ares_rad = 5.*N.pi/180.
		# build BDRFs for the relevant wavelengths and incident angles
		npoints = int(N.ceil(N.pi/2./ares_rad))
		thetas_r, phis_r = N.linspace(0., N.pi/2., npoints), N.linspace(0., 2.*N.pi, npoints)
		thetas_i, phis_i = thetas_r, phis_r
		#if axisymmetric_i: # if the bdrf is axisymmetric in incidence angle, no need to do all the phi incident
		phis_i = phis_i[[0]]
		bdrfs = N.zeros((len(thetas_i), len(phis_i), len(thetas_r), len(phis_r)))
		dhr = N.zeros((len(thetas_i), len(phis_i)))
		for i, thi in enumerate(thetas_i):
			for j, phi in enumerate(phis_i):
				CT = regular_grid_Cook_Torrance(thetas_r_rad=thetas_r, phis_r_rad=phis_r, th_i_rad=thi, phi_i_rad=phi, m=m, R_dh_Lam=R_Lam, alpha=alpha)
				bdrfs[i,j] = CT[-1].reshape(len(thetas_r), len(phis_r))
				dhr[i,j] = bdrf_to_dhr(*CT)
		# build a linear interpolator
		points = (thetas_i, phis_i, thetas_r, phis_r)
		bdrf = RegularGridInterpolator(points, bdrfs) # This is the interpolator that can give us the PDFs we need to sample from. We want to sample from the reflected energy, not the bdrf.
		points = (thetas_i, phis_i)
		dhr = RegularGridInterpolator(points, dhr)
		
		
		t0 = time()
		dist = BDRF_distribution(bdrf)
		xs, ys = thetas_r, phis_r
		
		th_sam = 0.*N.pi/180.
		x_samples, y_samples, weights = dist.sample(th_sam, phis_i[0], ns)
		#cProfile.run('sim()', sort='tottime')
		print('BDRF:', time()-t0, 's')

		hist_th = dist.theoretical_histogram(th_sam, phis_i[0])

		ref = dist.DHR(th_sam, phis_i[0])

		plt.subplots_adjust(left=0.05, wspace=0.3, right=0.95)
		plt.subplot(131, title='Theoretical histogram', aspect='equal', projection='polar')
		plt.pcolormesh(ys, xs, hist_th, vmin=N.amin(hist_th), vmax=N.amax(hist_th))
		plt.colorbar(label='PDF')

		plt.subplot(132, title='Numerical sampling', aspect='equal', projection='polar')
		hist, edgesx, edgesy = N.histogram2d(x_samples, y_samples, weights=ref*weights/ns, bins=[xs, ys])
		plt.pcolormesh(ys, xs, hist)#, vmin=N.amin(PDF_th), vmax=N.amax(PDF_th))
		plt.colorbar(label='PDF')

		plt.subplot(133, title='Relative error', aspect='equal', projection='polar')
		rel_error = (hist-hist_th)/hist_th*100.
		plt.pcolormesh(ys, xs, rel_error, cmap='bwr', vmin=-N.amax(N.abs(rel_error)), vmax=N.amax(N.abs(rel_error)))
		plt.colorbar(label='Relative error (%)')

		plt.savefig(path[0]+'/test_BDRF_i.png')		
		
