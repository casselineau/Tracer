# Common functions for optics laws, for use by surfaces etc.
#
# References:
# [1] http://en.wikipedia.org/wiki/Fresnel_equations
# [2] http://en.wikipedia.org/wiki/Snell%27s_law
# [3] Warren J. Smith, Modern Optical Engineering, 4th Ed., 2008; p. 208.

import numpy as N
import logging

def fresnel(ray_dirs, normals, n1, n2):
	"""Determines what ratio of the ray bundle is reflected and what is refracted, 
	and the performs the appropriate functions on them. Based on fresnel's equations, [1]
	
	Arguments: 
	ray_dirs - the directions of the ray bundle
	normals - a 3 by n array where for each of the n rays in ray_dirs, the 
		normal to the surface at the ray/surface intersection is given.
	n1 - refraction index of the material the ray is leaving
	n2 - refraction index of the material the ray is entering
	
	Returns:  
	R - the reflectance of a homogenously-polarized light ray with the given
		parameters.
	""" 
	theta_in = N.arccos(N.abs((normals*ray_dirs).sum(axis=0)))
	# Factor out common terms in Fresnel's equations:
	foo = N.cos(theta_in) 
	bar = N.sqrt(1 - (n1/n2 * N.sin(theta_in))**2)
	Rs = ((n1*foo - n2*bar)/(n1*foo + n2*bar))**2 
	Rp = ((n1*bar - n2*foo)/(n1*bar + n2*foo))**2

	# For now, assume no polarization and that the light contains an equal mix of 
	# s and p polarized light
	# R = ratio of reflected energy, T = ratio refracted (transmittance)
	R = (Rs + Rp)/2
	return R

def fresnel_conductor(ray_dirs, normals, lambdas, material, n1=1., m2=None):
	'''
	Fresnel reflections from a perfect dielectric to an attenuator (conductor). 
	Arguments: 
	ray_dirs - the directions of the ray bundle
	normals - a 3 by n array where for each of the n rays in ray_dirs, the 
		normal to the surface at the ray/surface intersection is given.
	lambdas - The wavelengths of the rays in the bundle
	material - an optical material instance (see the optical constants module)
	n1 - refraction index of the material the ray is leaving
	m2 - a forced complex refractive index
	
	Returns:  
	R_p, R_s - the parallel and perpendicular reflectances
	theta2 - the refraction angle in the material
	'''
	if m2 is None:
		m2 = material.m(lambdas)
	theta_in = N.arccos(N.abs((normals*ray_dirs).sum(axis=0)))
	R_p, R_s, theta2 = fresnel_to_attenuating(n1, m2, theta_in)
	return R_p, R_s, theta2
	
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

def polarised_reflections(ray_dirs, normals, R_p, R_s, E_p, E_s):
	'''
	not verified yet
	'''

	s_i = ray_dirs
	s_r = reflections(ray_dirs, normals) # reflection directions

	z = N.vstack([0.,0.,1.])

	c_i = N.cross(z.T, s_i.T).T
	h_i = c_i/N.linalg.norm(c_i,axis=0)
	c_r = N.cross(z.T, s_r.T).T
	h_r = c_r/N.linalg.norm(c_r,axis=0)

	v_i = N.cross(h_i.T, s_i.T).T
	v_r = N.cross(h_r.T, s_r.T).T

	hrsi = N.matmul(h_r, s_i)
	hisr = N.matmul(h_i, s_r)
	vrsi = N.matmul(v_r, s_i)
	visr = N.matmul(v_i, s_r)
	sisr4 = N.linalg.norm(N.cross(si.T, sr.T).T, axis=0)**4

	rho_ss = N.linalg.norm(vrsi*visr*R_s + hrsi*hisr*R_p, axis=0)**2/sisr4
	rho_ps = N.linalg.norm(hrsi*visr*R_s - vrsi*hisr*R_p, axis=0)**2/sisr4
	rho_sp = N.linalg.norm(vrsi*hisr*R_s - hrsi*vsr*R_p, axis=0)**2/sisr4
	rho_pp = N.linalg.norm(hrsi*hisr*R_s + vrsi*visr*R_p, axis=0)**2/sisr4

	in_pol = N.vstack([E_s, E_p])
	pol_mat = N.array([[rhoss, rho_ps],[rho_sp, rho_pp]])
	E_r_s, E_r_p = N.dot(pol_mat, in_pol)
	return E_r_p, E_r_s, s_r

def apparent_NK(m, alpha):
	'''
	https://www-sciencedirect-com.virtual.anu.edu.au/science/article/pii/S002240730500066X
	'''
	n2_k2 = m.real**2-m.imag**2
	N = N.sqrt(0.5*(n2_k2+N.sqrt(n2_k2**2+4.*(m.real*m.imag/N.cos(alpha))**2)))
	K = N.sqrt(N**2-n2_k2)
	return N, K

def generalised_fresnel(ray_dirs, normals, lambdas, material1, material2):
	'''
	https://www-sciencedirect-com.virtual.anu.edu.au/science/article/pii/S002240730500066X
	INCOMPLETE
	'''

	# if k/n <0.07: standard Fresnel
	# otherwise more
	logging.error("WIP")
	stop
	m1, m2 = material1.m(lambdas), material2.m(lambdas)
	N1, K1 = apparent_NK(m1, alpha1)
	N2, K2 = apparent_NK(m2, alpha2)

	theta_in = N.arccos(N.abs((normals*ray_dirs).sum(axis=0)))
	#R_p, R_s, theta2 = fresnel(m1, m2, theta_in)
	return R_p, R_s, theta2
	
			   
def reflections(ray_dirs, normals):  
	"""
	Generate directions of rays reflecting according to the reflection law.
	
	Arguments:
	ray_dirs - a 3 by n array where each column is the i-th of n ray directions
	normals - for each ray, the corresponding normal on the point where the ray
		intersects a surface, also 3 by n array.
	
	Returns: new ray directions as the result of reflection, 3 by n array.
	"""
	vertical = N.sum(ray_dirs*normals, axis=0)*normals # normal dot ray, really
	return ray_dirs - 2.*vertical

def refractions(n1, n2, ray_dirs, normals):
	"""Generates directions of rays refracted according to Snells's law (in its vector
	form, [2]
	
	Arguments: 
	n1, n2 - respectively the refractive indices of the medium the unrefracted ray
		travels in and of the medium the ray is entering.
	ray_dirs, normals - each a row of 3-component vectors (as an array) with the
		direction of incoming rays and corresponding normals at the points of
		incidence with the refracting surface.
	
	Returns:
	refracted - a boolean array stating which of the incoming rays has not
		undergone total internal reflection.
	refr_dirs - new ray directions as the result of refraction, for the non-TIR
		rays in the input bundle.
	"""
	# Broadcast all necessary arrays to the larger size required:
	n = N.broadcast_arrays(n2/n1, ray_dirs[0])[0]
	normals = N.broadcast_arrays(normals, ray_dirs)[0]
	cos1 = (normals*ray_dirs).sum(axis=0)
	refracted = cos1**2 >= 1. - n**2
	
	# Throw away totally-reflected rays.
	cos1 = cos1[refracted]
	ray_dirs = ray_dirs[:,refracted]
	normals = normals[:,refracted]
	n = n[refracted]
	
	refr_dirs = (ray_dirs - cos1*normals)/n
	cos2 = N.sqrt(1 - 1./n**2*(1. - cos1**2))
	refr_dirs += normals*cos2*N.where(cos1 < 0., -1, 1)

	return refracted, refr_dirs

def refr_idx_hartmann(wavelength, a, b, c, d, e):
	"""
	Calculate a material's refractive index corresponding to each given
	wavelength, using the Hartmann dispersion equation [3]:
	
	n(L) = a + b/(c - L) + d/(e - L)
	
	where L is the wavelength.
	"""
	return a + b/(c - wavelength) 

def attenuations(path_lengths, k, lambda_0, energy):
	'''
	Calculates energy attenuation from the wavelength in vacuum (lambda 0), the absorption index (complex part of the complex refractive index, k) and the path length.
	
	'''
	T = N.exp(-4.*N.pi*k/lambda_0*path_lengths)
	energy = T*energy
	return energy
	
def scattering(sigma, intersection_path_lengths, keep_path_lengths=False):
	'''
	Calculates randomly distributed scattereing path lengths for homogenous medium based on a scattering coefficient.
	Arguments:
	- sigma: scattering coefficient (m-1) if imaginary, the imag is the remaining distance to scattering because we have determined it earlier with a periodic BC
	Returns:
	 - boolean scattered, not scattered array
	 - Uniformly sampled path lengths to scattering event fro scattered rays
	'''

	realsig = N.isrealobj(sigma)
	if not realsig:
		path_lengths_left = sigma.imag
		sigma = sigma.real
	R = N.random.uniform(size=len(intersection_path_lengths))
	scattered_path_lengths = -N.log(R)/sigma
	scattered_path_lengths[sigma==0.] = intersection_path_lengths[sigma==0.]
	if not realsig:
		scattered_path_lengths[path_lengths_left != 0] = path_lengths_left[path_lengths_left != 0]
	scattered = scattered_path_lengths < intersection_path_lengths
	if keep_path_lengths:
		to_scatter = scattered_path_lengths - intersection_path_lengths
		to_scatter[scattered] = 0.
		return scattered, scattered_path_lengths, to_scatter
	else:
		return scattered, scattered_path_lengths


def R_from_n_k(n1, n2, thetas_in):
	n1, n2 = N.vstack(n1), N.vstack(n2)
	foo = N.cos(thetas_in)
	bar = N.sqrt(1. - (n1/n2 * N.sin(thetas_in))**2)
	Rs = ((n1*foo - n2*bar)/(n1*foo + n2*bar))**2
	Rp = ((n1*bar - n2*foo)/(n1*bar + n2*foo))**2
	R = (Rs + Rp) / 2.
	return N.ravel(thetas_in), R

if __name__ == '__main__':
	import matplotlib.pyplot as plt
	from sys import path

	# test fresnel with Modest's 3 ed. Figure 2-11
	lambdas= [3.1e-6]
	n1 = 1.
	m2 = 4.46 +1j*31.5

	thetas = N.linspace(0.,90.,150)
	ray_dirs = N.zeros((3, 150))
	ray_dirs[0] = -N.sin(thetas*N.pi/180.)
	ray_dirs[2] = -N.cos(thetas*N.pi/180.)

	normals = N.zeros((3, 150))
	normals[2] = 1.

	R_p, R_s, theta_2 = fresnel_conductor(ray_dirs, normals, lambdas, material='no', n1=n1, m2=m2)
	#polarised_reflections_conductor(ray_dirs, normals, R_s, R_p, E_s, E_p)

	plt.figure()
	plt.plot(thetas, R_p, color='b', label=r'$\rho_\parallel$')
	plt.plot(thetas, R_s, color='r', label=r'$\rho_\bot$')
	plt.plot(thetas, (R_p+R_s)/2., color='violet', label=r'$\overline{\rho}$')
	plt.xlabel(r'${\theta}$ (${^\circ}$)')
	plt.ylabel(r'$\rho$')
	plt.legend()
	
	plt.savefig(path[0]+'/test_polarised_fresnel_Al.png')
