import numpy as N

def Planck(wl, T):
	'''
	Planck sidtribution
	:param wl: wavelength in m
	:param T: Temperature in K
	:return: Planck distribution values for each wl
	'''
	h = 6.626070040e-34 # Planck constant
	c = 299792458. # Speed of light in vacuum
	k = 1.38064852e-23 # Boltzmann constant
	hc_kTwl = h*c/(k*T*wl)
	return (2.*h*c**2.)/(wl**5.)/(N.exp(hc_kTwl)-1.)

def dielectric_to_refractive(eps):
	'''
	eps - complex dielectic function
	/!\: Non-magnetic medium
	'''
	n = N.sqrt(0.5*(eps.real+N.sqrt(eps.real**2+eps.imag**2)))
	k = N.sqrt(0.5*(-eps.real+N.sqrt(eps.real**2+eps.imag**2)))

	m = n + 1j*k
	return m

def refractive_to_dielectric(m):
	'''
	m - complex refractive index
	/!\: Non-magnetic medium
	'''
	eps_p = (m.real)**2 - (m.imag)**2
	eps_pp = 2.*m.real*m.imag

	return eps_p + 1j*eps_pp

def Fresnel_dielectrics(n1, n2, theta1):
	'''
	Wikipedia...
	'''
	theta2 = N.arcsin(n1*N.sin(theta1)/n2)

	c1, c2 = N.cos(theta1), N.cos(theta2)
	R_s = N.abs((n1*c1-n2*c2)/(n1*c1+n2*c2))**2
	R_p = N.abs((n1*c2-n2*c1)/(n1*c2+n2*c1))**2

	return R_p, R_s, theta2

def lambda_to_freqs(lambdas):
	return 299.792458e6/lambdas
	
def lambda_to_angular_freqs(lambdas):
	return lambda_to_freqs(lambdas)/(2.*N.pi)
	
def Drude_Lorentz_model(lambdas, resonators):
	'''
	resonators: n by 3 array with n the number of resonators and [:,0] the plasma frequencies, [:1] the resonance frequencies and [:,3] the damping factors.
	'''
	freqs = N.vstack(lambda_to_angular_freqs(lambdas)) # freqs in Hz from lamdas in m

	omega_p = resonators[:,0]
	omega = resonators[:,1]
	gamma = resonators[:,2]
	eps = 1.+N.sum(omega_p**2/(omega**2-freqs**2-1j*gamma*freqs), axis=1)

	return eps

def fit_Drude_Lorentz_from_m(lambdas, m, n_res, metal=False):
	from scipy.optimize import minimize, dual_annealing
	'''
	Find way to find the peaks
	First fit thereal part
	Then fit the imaginary part
	'''

	eps = refractive_to_dielectric(m)
	
	#deps = (eps.imag[1:]-eps.imag[:-1])/(lambdas[1:]-lambdas[:-1])
	#signs = deps/N.abs(deps)
	#peaks = N.argwhere((signs[1:]*signs[:-1])<0)
	#lambdapeaks = (lambdas[1:]+lambdas[:-1])/2.
	#lambdapeaks = (lambdapeaks[peaks+1]+lambdapeaks[peaks])/2.
	#freqs_peak = lambda_to_freqs(lambdapeaks)
	#print(lambdapeaks, freqs_peak)

	def error(params, metal):
		omega_p = params[:n_res]
		omega = params[n_res:2*n_res]
		if metal:
			omega *= 0.
		gamma = params[2*n_res:]
		material = N.array([omega_p, omega, gamma]).T

		eps_fit = Drude_Lorentz_model(lambdas, material)
		diff = N.sum((eps_fit.real/eps.real)**2+(eps_fit.imag/eps.imag)**2)
		return diff
	#n = 0 
	x0=None
	#while n<10:
	res = dual_annealing(error, bounds=((1e14, 1e18),(0, 1e-10),(1e10, 1e18)), args=(metal,), maxiter=10000, x0=x0)
		#x0=res.x
		#n+=1
		#print(res)

	omega_p = N.array(res.x[:n_res])
	omega = N.array(res.x[n_res:2*n_res])
	if metal:
		omega *=0
	#gamma = N.zeros(n_res)
	gamma = res.x[2*n_res:]
	material = N.array([omega_p, omega, gamma]).T
	print(eps)
	print(Drude_Lorentz_model(lambdas, material))
	
	return material
	
def fresnel_to_attenuating(n1, m2, theta1):
	'''
	From Modest Chapter 2 -  The Interface between a Perfect Dielectric and
an Absorbing Medium.
	returns:
	parallel (p) and perpendicular (s) polarized reflectivities + transmission (refraction) angle
	'''
	b = (m2.real**2 - m2.imag**2- (n1*N.sin(theta1))**2)	
	a = N.sqrt(b**2 + 4.*(m2.real*m2.imag)**2 ) 

	p = N.sqrt(0.5*(a+b))
	q = N.sqrt(0.5*(a-b))

	theta2 = N.arctan(n1*N.sin(theta1)/p)

	R_s = ((n1*N.cos(theta1)-p)**2+q**2)/((n1*N.cos(theta1)+p)**2+q**2) # s is perpendicular to the plane of incidence	
	R_p = ((p-n1*N.sin(theta1)*N.tan(theta1))**2+q**2)/((p+n1*N.sin(theta1)*N.tan(theta1))**2+q**2)*R_s # p is parallel to the plane of incidence
	
	return R_p, R_s, theta2

def Fresnel_general(m1, m2, theta_1):
	'''
	From Born and Wolf, as written in https://www.sciencedirect.com/science/article/pii/S0022407306001932
	psi from: 
	'''

	m = m1/m2

	s1 = N.sin(theta_1)
	c1 = N.cos(theta_1)

	q = N.abs(N.sqrt(1.-(s1/m)**2))
	gamma = N.angle(N.sqrt(1.-(s1/m)**2))

	theta_r = N.arcsin(s1*m) 

	cr = N.cos(theta_r)

	R_p = N.abs((c1-m*cr)/(c1+m*cr))**2
	R_s = N.abs((cr-m*c1)/(cr+m*c1))**2

	psi = N.arctan(1./(1./N.tan(theta_r)).real)

	return R_p, R_s, theta_r, psi


def attenuation(path_lengths, k, lambda_0, energy):
	'''
	Calculates energy attenuation from the wavelength in vacuum (lambda 0), the absorption index (complex part of the complex refractive index, k) and theoath length.
	
	'''
	T = N.exp(-4.*N.pi*k/lambda_0*path_lengths)
	energy = T*energy
	
	return energy

if __name__ == '__main__':
	import matplotlib.pyplot as plt
	from sys import path
	'''
	# test cases for Fresnel equations
	# Modest 3E, ex 2-4:
	print('Modest 2-4:')
	n1, n2 = 2., 8./3.
	k1, k2 = 0, 0
	m1, m2 = complex(n1, k1), complex(n2, k2)
	theta1 = N.arccos(0.6)

	R_p, R_s, theta2 = Fresnel_dielectrics(m1, m2, theta1)
	print(R_s, R_p, theta2/N.pi*180.)

	R_p, R_s, theta2 = Fresnel_to_attenuating(m1, m2, theta1)
	print(R_s, R_p, theta2/N.pi*180.)

	R_p, R_s, theta2, psi = Fresnel_general(m1, m2, theta1)
	print(R_s, R_p, theta2/N.pi*180., psi/N.pi*180.)

	# Modest 3E example 2-5:
	n1, k1 = 2., 0.
	n2, k2 = 90., 90.
	m1, m2 = complex(n1, k1), complex(n2, k2)
	print('Modest 2-5:')
	R_p, R_s, theta2 = Fresnel_to_attenuating(m1, m2, theta1)
	print(R_s, R_p, theta2/N.pi*180.)

	R_p, R_s, theta2, psi = Fresnel_general(m1, m2, theta1)
	print(R_s, R_p, theta2/N.pi*180., psi/N.pi*180.)

	# Test Drude-Lorentz:
	copper = N.array([[8.21225411, 0., -0.030], [2.67481269, 0.291, -0.378], [3.49257006, 2.957, -1.056],[9.20868474, 5.300, -3.213],[8.65045191, 11.18, -4.305]]) # https://refractiveindex.info/?shelf=main&book=Cu&page=Rakic-LD
	lambdas = N.linspace(210e-9, 12.4e-6, 1000)
	
	eps = Drude_Lorentz_model(lambdas, copper)
	m = dielectric_to_refractive(eps)
	plt.figure()
	plt.plot(lambdas*1e6, m.real, label ='n')
	plt.plot(lambdas*1e6, m.imag, label ='k')
	plt.legend()
	plt.xlabel('${\lambda}$ ${\mu m}$')
	plt.ylabel('n, k')
	plt.savefig(path[0]+'/test_ref_index_copper.png')
	'''
	# test fitting with interpolation
	data = N.loadtxt('/media/ael/Flashy/backup_05-06-2021/Documents/Boulot/Material_properties/Rhodium.csv', skiprows=1)
	lambdas = data[:,0]*1e-6
	m = data[:,1] + 1j*data[:,2]
	eps = refractive_to_dielectric(m)
	from scipy.interpolate import interp1d
	m_func = interp1d(lambdas, m)

	plt.figure()

	rhodium = fit_Drude_Lorentz_from_m(lambdas, m, n_res=1, metal=True)
	print(rhodium)
	eps_fit = Drude_Lorentz_model(lambdas, rhodium)
	m_fit = dielectric_to_refractive(eps_fit)

	#print(rhodium, m_fit, m)
	print(m, m_fit)
	plt.subplot(121)
	plt.plot(lambdas*1e6, eps.real, label ='epsR')
	plt.plot(lambdas*1e6, eps.imag, label ='epsI')
	plt.plot(lambdas*1e6, eps_fit.real, label ='epsR_fit')
	plt.plot(lambdas*1e6, eps_fit.imag, label ='epsI_fit')
	plt.legend()
	plt.subplot(122)
	plt.plot(lambdas*1e6, m.real, label ='n_exp')
	plt.plot(lambdas*1e6, m.imag, label ='k_exp')
	plt.plot(lambdas*1e6, m_fit.real, label ='n_fit')
	plt.plot(lambdas*1e6, m_fit.imag, label ='k_fit')
	plt.legend()
	plt.xlabel('${\lambda}$ ${\mu m}$')
	plt.ylabel('n, k')
	plt.savefig(path[0]+'/test_ref_index_rhodium.png', dpi=600)
	'''
	# attenuation
	ks = N.hstack([0.01, 0.1, N.linspace(1., 10., 10), 20, 50, 100])
	wls = N.linspace(200e-9, 10e-6, 500)
	d = N.linspace(0.001e-6, 10e-6, 1000)
	energy = 1.
	plt.figure()
	for i,k in enumerate(ks):
		plt.plot(wls*1e6, -N.log(0.001)*wls/(4.*N.pi*k)*1e6, label=k)

	plt.legend()
	plt.xlabel('${\lambda}$ ${\mu m}$')
	plt.ylabel('${\mu m}$')
	#plt.pcolormesh(ks, wls*1e6, N.log(d_99))
	#plt.xscale('log')
	plt.yscale('log')
	#plt.colorbar()
	#plt.xlabel('k')
	#plt.ylabel('${\lambda}$ ${\mu m}$')
	plt.savefig(path[0]+'/99.9%_absorptance_distance.png')
	'''
