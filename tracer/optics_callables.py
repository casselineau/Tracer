# -*- coding: utf-8 -*-
# A collection of callables and tools for creating them, that may be used for
# the optics-callable part of a Surface object.

from tracer import optics, ray_bundle, sources
from .spatial_geometry import rotation_to_z
import numpy as N
from scipy.interpolate import RegularGridInterpolator
#from BDRF_models import Cook_Torrance, regular_grid_Cook_Torrance
from tracer.ray_bundle import RayBundle
from ray_trace_utils.sampling import BDRF_distribution, Henyey_Greenstein
from ray_trace_utils.vector_manipulations import get_angle, rotate_z_to_normal
from tracer.spatial_geometry import rotz, general_axis_rotation
from abc import ABC, abstractmethod
from itertools import combinations

import sys, inspect

'''
#TODO:
Systematize creation based on keywords: WIP ideas
Accountant declaration: mix energy, direction and spectral accounting with keywords.
They have to be compatible to automatically subclass.
By defaults all accountants store ray positions for now as fluxmapping is the basic use of these.
- Energy accounting: Need to make these compatibles
	- Absorption accountant: counts what is absorbed locally. Receiver
	- Scattered accountant: counts what leaves the interaction. Scatterer
	- Spectral accountant: stores ray wavelengths. Spectro
	- Polychromatic accountant: stores ray spectra. Polychromatic
- Location accountant: stores the hits
- Directions accounting:
	- Direction accountant: incident direction stored. Directional
	- Bidirectional accountant: incident and scattered. Bidirectional
 
Accountants shortname examples to append at then end of the optical callable for automatic instancing:
Option: Use the capital letters instead of teh full name? 
SpectroDirectionalReceiver (SDR): stores the hit location, absorbed energy, direction and wavelength of the rays
BidirectionalScatterer (BS): stores the hit location, the incident and scattering direction of the rays.

Polychromatic optical managers: figure-out how to declare and maintain spectra throughout the simulations without 
storing excessive wavelength information
Spectra could be declared in the optics and used as reference. 
Ideally, the full simulation uses the exact same spectral description and each element has its own values for the 
relevant properties. This would make a spectrum Class useful as a single reference.

Optical behaviours: dictates how ras are handled
- Incident directions: 
	- nothing = isotropic property
	- Directional = Depends on Phi and theta
	- DirectionalAxisymmetric = depends on theta
- Reflected directions: 
	- Lambertian = diffuse
	- Specular = specular
	- Real = ideal normal modified with a gaussian error in a given angular range (bivariate or conical)
	- Directional = depends on phi and theta
	- DirectionalAxisymmetric = depends on theta
- Transmitted directions:
	- Nothing = opaque surface
	- Transparent = nothing happens, light goes through
	- Refractive = fresnel refraction
	- Absorbant = attenuation in the medium
	- Scattering = scattering in the medium
- Spectrum:
	- Nothing = Spectral band approximation, properties valid for all rays
	- Spectral = one wl per ray
	- Polychromatic = Rays carry full piecewise linear spectra
	
Then we can introduce more general wrappers:
- BSDF: informs about the whole behaviour.

How to do this:
- Make module hosted functions of:
 	- energy: attenuation, reflection, absorption, make them compatible with array operations for spectral and polychromatic operations
 	- normals: add error to normal vectors
 	- directions: specular reflections, refraction, bxdf sampling
- Use these functions in order to modify surface properties or energy output at the right location in the optics_callable calls. 
'''
class Transparent(object):
	"""
	Generates a function that simply intercepts rays but does not change any of their properties.
	
	Arguments:
	- /
	
	Returns:
	a function with the signature required by Surface.
	"""
	def __init__(self):
		pass
	
	def __call__(self, geometry, rays, selector):
		outg = rays.inherit(selector,
			vertices=geometry.get_intersection_points_global(),
			direction=rays.get_directions()[:,selector],
			energy=rays.get_energy()[selector],
			parents=selector)

		return outg


class Reflective(object):
	"""
	Generates a function that represents the optics of an opaque, absorptive
	surface with specular reflections.
	
	Arguments:
	absorptivity - the amount of energy absorbed before reflection.
	
	Returns:
	a function with the signature required by Surface.
	"""
	def __init__(self, absorptivity):
		self._abs = absorptivity
	
	def __call__(self, geometry, rays, selector):
		outg = rays.inherit(selector,
			vertices=geometry.get_intersection_points_global(),
			direction=optics.reflections(rays.get_directions()[:,selector], geometry.get_normals()),
			energy=rays.get_energy()[selector]*(1. - self._abs),
			parents=selector)
	
		if outg.has_property('spectra'):
			outg._spectra *= (1.-self._abs)

		return outg

class OneSidedReflective(Reflective):
	"""
	This optics manager behaves similarly to the ReflectiveReceiver class,
	but adds directionality. In this way a simple one-side receiver doesn't
	necessitate an extra surface in the back.
	"""
	def __call__(self, geometry, rays, selector):
		"""
		Rays coming from the "up" side are reflected like in a Reflective
		instance, rays coming from the "down" side have their energy set to 0.
		As usual, "up" is the surface's Z axis.
		"""
		outg = Reflective.__call__(self, geometry, rays, selector)
		energy = outg.get_energy()
		proj = N.sum(rays.get_directions()[:,selector] * geometry.up()[:,None], axis=0)
		energy[proj > 0] = 0
		outg.set_energy(energy)
		return outg

class Reflective_IAM(object):
	'''
	Generates a function that performs specular reflections from an opaque absorptive surface modified by the Incidence Angle Modifier from: Martin and Ruiz: https://pvpmc.sandia.gov/modeling-steps/1-weather-design-inputs/shading-soiling-and-reflection-losses/incident-angle-reflection-losses/martin-and-ruiz-model/. 
	'''
	def __init__(self, absorptivity, a_r):
		self._abs = absorptivity
		self.a_r = a_r
	
	def __call__(self, geometry, rays, selector):
		normals = geometry.get_normals()
		directions = rays.get_directions()[:,selector]
		vertical = N.sum(directions*normals, axis=0)*normals
		cos_theta_AOI = N.sqrt(N.sum(vertical**2, axis=0))
		outg = rays.inherit(selector,
			vertices=geometry.get_intersection_points_global(),
			direction=optics.reflections(directions, normals),
			energy=rays.get_energy()[selector]*(1. - self._abs*(1.-N.exp(-cos_theta_AOI/self.a_r))/(1.-N.exp(-1./self.a_r))),
			parents=selector)

		if outg.has_property('spectra'):
			outg._spectra *= (1. - self._abs*(1.-N.exp(-cos_theta_AOI/self.a_r))/(1.-N.exp(-1./self.a_r)))

		return outg

class Reflective_mod_IAM(object):
	'''
	Generates a function that performs specular reflections from an opaque absorptive surface modified by the Incidence Angle Modifier from: Martin and Ruiz: https://pvpmc.sandia.gov/modeling-steps/1-weather-design-inputs/shading-soiling-and-reflection-losses/incident-angle-reflection-losses/martin-and-ruiz-model/. 
	'''
	def __init__(self, absorptivity, a_r, c):
		self._abs = absorptivity
		self.a_r = a_r
		self.c = c
	
	def __call__(self, geometry, rays, selector):
		normals = geometry.get_normals()
		directions = rays.get_directions()[:,selector]
		vertical = N.sum(directions*normals, axis=0)*normals
		cos_theta_AOI = N.sqrt(N.sum(vertical**2, axis=0))
		outg = rays.inherit(selector,
			vertices=geometry.get_intersection_points_global(),
			direction=optics.reflections(directions, normals),
			energy=rays.get_energy()[selector]*(1. - self._abs*(1.-self.c*N.exp(-cos_theta_AOI/self.a_r))/(1.-N.exp(-1./self.a_r))),
			parents=selector)

		if outg.has_property('spectra'):
			outg._spectra *= (1. - self._abs*(1.-self.c*N.exp(-cos_theta_AOI/self.a_r))/(1.-N.exp(-1./self.a_r)))

		return outg


class Lambertian_IAM(object):
	'''
	Generates a function that performs diffuse reflections from an opaque absorptive surface modified by the Incidence Angle Modifier from: Martin and Ruiz: https://pvpmc.sandia.gov/modeling-steps/1-weather-design-inputs/shading-soiling-and-reflection-losses/incident-angle-reflection-losses/martin-and-ruiz-model/. 
	'''
	def __init__(self, absorptivity, a_r):
		self._abs = absorptivity
		self.a_r = a_r
	
	def __call__(self, geometry, rays, selector):
		normals = geometry.get_normals()
		directions = rays.get_directions()[:,selector]
		vertical = N.sum(directions*normals, axis=0)*normals
		cos_theta_AOI = N.sqrt(N.sum(vertical**2, axis=0))

		directs = sources.pillbox_sunshape_directions(len(selector), ang_range=N.pi/2.)
		directs = N.sum(rotation_to_z(normals.T) * directs.T[:,None,:], axis=2).T

		outg = rays.inherit(selector,
			vertices=geometry.get_intersection_points_global(),
			direction=directs,
			energy=rays.get_energy()[selector]*(1. - self._abs*(1.-N.exp(-cos_theta_AOI/self.a_r))/(1.-N.exp(-1./self.a_r))),
			parents=selector)

		if outg.has_property('spectra'):
			outg._spectra *= (1. - self._abs*(1.-N.exp(-cos_theta_AOI/self.a_r))/(1.-N.exp(-1./self.a_r)))

		return outg

class Lambertian_mod_IAM(object):
	'''
	Generates a function that performs diffuse reflections from an opaque absorptive surface modified by the Incidence Angle Modifier from: Martin and Ruiz: https://pvpmc.sandia.gov/modeling-steps/1-weather-design-inputs/shading-soiling-and-reflection-losses/incident-angle-reflection-losses/martin-and-ruiz-model/. 
	'''
	def __init__(self, absorptivity, a_r, c):
		self._abs = absorptivity
		self.a_r = a_r
		self.c = c
	
	def __call__(self, geometry, rays, selector):
		normals = geometry.get_normals()
		directions = rays.get_directions()[:,selector]
		vertical = N.sum(directions*normals, axis=0)*normals
		cos_theta_AOI = N.sqrt(N.sum(vertical**2, axis=0))

		directs = sources.pillbox_sunshape_directions(len(selector), ang_range=N.pi/2.)
		directs = N.sum(rotation_to_z(normals.T) * directs.T[:,None,:], axis=2).T

		outg = rays.inherit(selector,
			vertices=geometry.get_intersection_points_global(),
			direction=directs,
			energy=rays.get_energy()[selector]*(1. - self._abs*(1.-N.exp(-cos_theta_AOI**self.c/self.a_r))/(1.-N.exp(-1./self.a_r))),
			parents=selector)

		if outg.has_property('spectra'):
			outg._spectra *= (1. - self._abs*(1.-N.exp(-cos_theta_AOI**self.c/self.a_r))/(1.-N.exp(-1./self.a_r)))

		return outg

class Lambertian_directional_axisymmetric_piecewise(object):
	'''
	Generates a function that performs diffuse reflections off opaque surfaces whose angular absorptance (axisymmetrical) is interpolated from discrete angular values.
	'''
	def __init__(self, thetas, absorptance_th, specularity=0.):
		self.thetas = thetas # thetas are angle to the normal. Have to be defined between 0 and N.pi/2 rad and in increasing order. If not, the model returns teh closest values.
		self.abs_th = absorptance_th # angular apsorptance points
		self.specularity = specularity
	
	def __call__(self, geometry, rays, selector):
		normals = geometry.get_normals()
		directions = rays.get_directions()[:,selector]
		vertical = N.sum(directions*normals, axis=0)*normals
		directs = N.zeros(directions.shape)

		thetas_in = N.arccos(N.sqrt(N.sum(vertical**2, axis=0)))

		ang_abss = N.interp(thetas_in, self.thetas, self.abs_th)

		directs = sources.pillbox_sunshape_directions(len(selector), ang_range=N.pi/2.)
		directs = N.sum(rotation_to_z(normals.T) * directs.T[:,None,:], axis=2).T

		outg = rays.inherit(selector,
			vertices=geometry.get_intersection_points_global(),
			direction=directs,
			energy=rays.get_energy()[selector]*(1.-ang_abss),
			parents=selector)

		if outg.has_property('spectra'):
			outg._spectra *= (1.-ang_abss)

		return outg

class Lambertian_directional_axisymmetric_piecewise_spectral(object):
	'''
	Generates a function that performs diffuse reflections off opaque surfaces whose spectral angular absorptance (axisymmetrical) is interpolated from discrete angular and spectral values.
	'''
	def __init__(self, thetas, absorptance, wavelengths):
		thetas, wavelengths = N.unique(thetas), N.unique(wavelengths)
		absorptance = N.reshape(absorptance, (len(thetas), len(wavelengths)))
		points = (thetas, wavelengths)
		self.interpolator = RegularGridInterpolator(points, absorptance)

	def __call__(self, geometry, rays, selector):
		normals = geometry.get_normals()
		directions = rays.get_directions()[:,selector]
		vertical = N.sum(directions*normals, axis=0)*normals

		thetas_in = N.arccos(N.sqrt(N.sum(vertical**2, axis=0)))
		wavelengths = rays.get_wavelengths()[selector]

		ang_abss = self.interpolator(N.array([thetas_in, wavelengths]).T)

		directs = sources.pillbox_sunshape_directions(len(selector), ang_range=N.pi/2.)
		directs = N.sum(rotation_to_z(normals.T) * directs.T[:,None,:], axis=2).T

		outg = rays.inherit(selector,
			vertices=geometry.get_intersection_points_global(),
			direction=directs,
			energy=rays.get_energy()[selector]*(1.-ang_abss),
			parents=selector)
		return outg

class Lambertian_directional_axisymmetric_piecewise_Polychromatic(object):
	'''
	Generates a function that performs diffuse reflections off opaque surfaces whose spectral angular absorptance (axisymmetrical) is interpolated from discrete angular and spectral values.
	'''
	def __init__(self, thetas, absorptance, wavelengths):
		thetas, wavelengths = N.unique(thetas), N.unique(wavelengths)
		absorptance = N.reshape(absorptance, (len(thetas), len(wavelengths)))
		points = (thetas, wavelengths)
		self.interpolator = RegularGridInterpolator(points, absorptance)

	def __call__(self, geometry, rays, selector):
		normals = geometry.get_normals()
		directions = rays.get_directions()[:,selector]
		vertical = N.sum(directions*normals, axis=0)*normals
		wavelengths = rays.get_wavelengths()[:,selector] # wavelengths resolution of each spectrum
		thetas_in = N.arccos(N.sqrt(N.sum(vertical**2, axis=0)))
		points = N.array([N.tile(thetas_in, (wavelengths.shape[0], 1)), wavelengths]).T
		ang_abss = self.interpolator(points)

		spectra = rays.get_spectra()[:,selector] * (1.-ang_abss.T) # spectral power of each.
		energy = N.trapz(spectra, wavelengths, axis=0)
		
		directs = sources.pillbox_sunshape_directions(len(selector), ang_range=N.pi/2.)
		directs = N.sum(rotation_to_z(normals.T) * directs.T[:,None,:], axis=2).T
		
		outg = rays.inherit(selector,
			vertices=geometry.get_intersection_points_global(),
			direction=directs,
			energy=energy,
			wavelengths=wavelengths,
			spectra=spectra,
			parents=selector)
		return outg

class LambertianSpecular_directional_axisymmetric_piecewise(object):
	'''
	Generates a function that performs partly specular/diffuse reflections off opaque surfaces whose angular absorptance (axisymmetrical) is interpolated from discrete angular values. Specularity is constant n the incident angles.
	'''
	def __init__(self, thetas, absorptance_th, specularity=0.):
		self.thetas = thetas # thetas are angle to the normal. Have to be defined between 0 and N.pi/2 rad and in increasing order. If not, the model returns teh closest values.
		self.abs_th = absorptance_th # angular apsorptance points
		self.specularity = specularity
	
	def __call__(self, geometry, rays, selector):
		normals = geometry.get_normals()
		directions = rays.get_directions()[:,selector]
		vertical = N.sum(directions*normals, axis=0)*normals
		directs = N.zeros(directions.shape)

		thetas_in = N.arccos(N.sqrt(N.sum(vertical**2, axis=0)))
		ang_abss = N.interp(thetas_in, self.thetas, self.abs_th)

		specular = N.random.rand(len(selector))<self.specularity
		directs[:,specular] = optics.reflections(directions[:,specular], normals[:,specular])
		direct_lamb = sources.pillbox_sunshape_directions(N.sum(~specular), ang_range=N.pi/2.)
		directs[:,~specular] = N.sum(rotation_to_z(normals[:,~specular].T) * direct_lamb.T[:,None,:], axis=2).T

		outg = rays.inherit(selector,
			vertices=geometry.get_intersection_points_global(),
			direction=directs,
			energy=rays.get_energy()[selector]*(1.-ang_abss),
			parents=selector)
		return outg

class Lambertian_piecewise_Specular_directional_axisymmetric_piecewise(object):
	'''
	Generates a function that performs partly specular/diffuse reflections off opaque surfaces whose angular absorptance (axisymmetrical) is interpolated from discrete angular values. Specularity varies with the incident angles.
	'''
	def __init__(self, thetas, absorptance_th, specularity_th):
		self.thetas = thetas # thetas are angle to the normal. Have to be defined between 0 and N.pi/2 rad and in increasing order. If not, the model returns teh closest values.
		self.abs_th = absorptance_th # angular apsorptance
		self.spec_th = specularity_th # angular specularity
	
	def __call__(self, geometry, rays, selector):
		normals = geometry.get_normals()
		directions = rays.get_directions()[:,selector]
		vertical = N.sum(directions*normals, axis=0)*normals
		directs = N.zeros(directions.shape)

		thetas_in = N.arccos(N.sqrt(N.sum(vertical**2, axis=0)))

		ang_abss = N.interp(thetas_in, self.thetas, self.abs_th)
		ang_spec = N.interp(thetas_in, self.thetas, self.spec_th)

		specular = N.random.rand(len(selector))<ang_spec
		directs[:,specular] = optics.reflections(directions[:,specular], normals[:,specular])
		direct_lamb = sources.pillbox_sunshape_directions(N.sum(~specular), ang_range=N.pi/2.)
		directs[:,~specular] = N.sum(rotation_to_z(normals[:,~specular].T) * direct_lamb.T[:,None,:], axis=2).T

		outg = rays.inherit(selector,
			vertices=geometry.get_intersection_points_global(),
			direction=directs,
			energy=rays.get_energy()[selector]*(1.-ang_abss),
			parents=selector)
		return outg

perfect_mirror = Reflective(0)

class RealReflective(object):
	'''
	Generates a function that represents the optics of an opaque absorptive surface with specular reflections and realistic surface slope error. The surface slope error is considered equal in both x and y directions. The consequent distribution of standard deviation is described by a radial bivariate normal distribution law.

	Arguments:
	absorptivity - the amount of energy absorbed before reflection
	sigma - Standard deviation of the reflected ray in the local x and y directions. 
	
	Returns:
	Reflective - a function with the signature required by surface
	'''
	def __init__(self, absorptivity, sigma, bi_var=True):
		self._abs = absorptivity
		self._sig = sigma
		self.bi_var = bi_var

	def __call__(self, geometry, rays, selector):
		ideal_normals = geometry.get_normals()

		if self._sig > 0.:
			if self.bi_var == True:
				# Creates projection of error_normal on the surface (sin can be avoided because of very small angles).
				tanx = N.tan(N.random.normal(scale=self._sig, size=N.shape(ideal_normals[1])))
				tany = N.tan(N.random.normal(scale=self._sig, size=N.shape(ideal_normals[1])))

				normal_errors_z = (1./(1.+tanx**2.+tany**2.))**0.5
				normal_errors_x = tanx*normal_errors_z
				normal_errors_y = tany*normal_errors_z

			else:
				th = N.random.normal(scale=self._sig, size=N.shape(ideal_normals[1]))
				phi = N.random.uniform(low=0., high=2.*N.pi, size=N.shape(ideal_normals[1]))
				normal_errors_z = N.cos(th)
				normal_errors_x = N.sin(th)*N.cos(phi)
				normal_errors_y = N.sin(th)*N.sin(phi)

			normal_errors = N.vstack((normal_errors_x, normal_errors_y, normal_errors_z))

			# Determine rotation matrices for each normal:
			rots_norms = rotation_to_z(ideal_normals.T)
			if rots_norms.ndim==2:
				rots_norms = [rots_norms]

			# Build the normal_error vectors in the local frame.
			real_normals = N.zeros(N.shape(ideal_normals))
			for i in range(N.shape(real_normals)[1]):
				real_normals[:,i] = N.dot(rots_norms[i], normal_errors[:,i])

			real_normals_unit = real_normals/N.sqrt(N.sum(real_normals**2, axis=0))
		else:
			real_normals_unit = ideal_normals
		# Call reflective optics with the new set of normals to get reflections affected by 
		# shape error.
		outg = rays.inherit(selector,
			vertices = geometry.get_intersection_points_global(),
			direction = optics.reflections(rays.get_directions()[:,selector], real_normals_unit),
			energy = rays.get_energy()[selector]*(1 - self._abs),
			parents = selector)

		if outg.has_property('spectra'):
			outg._spectra *= (1.-self._abs)

		return outg

class OneSidedRealReflective(RealReflective):
	"""
	Adds directionality to an optics manager that is modelled to represent the
	optics of an opaque absorptive surface with specular reflections and realistic
	surface slope error.
	"""
	def __call__(self, geometry, rays, selector):
		outg = RealReflective.__call__(self, geometry, rays, selector)
		energy = outg.get_energy()
		proj = N.sum(rays.get_directions()[:,selector]*geometry.up()[:,None], axis = 0)
		energy[proj > 0] = 0 # projection up - set energy to zero
		outg.set_energy(energy) #substitute previous step into ray energy array
		return outg

class SemiLambertian(object):
	"""
	Represents the optics of an semi-diffuse surface, i.e. one that absrobs and reflects rays in a random direction if they come in a certain angular range and fully specularly if they come from a larger angle
	"""
	def __init__(self, absorptivity=0., angular_range=N.pi/2.):
		self._abs = absorptivity
		self._ar = angular_range
	
	def __call__(self, geometry, rays, selector):
		"""
		Arguments:
		geometry - a GeometryManager which knows about surface normals, hit
			points etc.
		rays - the incoming ray bundle (all of it, not just rays hitting this
			surface)
		selector - indices into ``rays`` of the hitting rays.
		"""
		directs = sources.pillbox_sunshape_directions(len(selector), self._ar)
		normals = geometry.get_normals()

		in_directs = rays.get_directions()[selector]
		angs = N.arccos(N.dot(in_directs, -normals)[2])
		glancing = angs>self._ar

		directs[~glancing] = N.sum(rotation_to_z(normals[~glancing].T) * directs[~glancing].T[:,None,:], axis=2).T
		directs[glancing] = optics.reflections(in_directs[glancing], normals[glancing])
	  
		energies = rays.get_energy()[selector]
		energies[~glancing] = energies[~glancing]*(1. - self._abs)
		
		outg = rays.inherit(selector,
			vertices=geometry.get_intersection_points_global(),
			energy=energies,
			direction=directs, 
			parents=selector)

		if outg.has_property('spectra'):
			outg._spectra[~glancing] *= (1.-self._abs)
		return outg

class Lambertian(object):
	"""
	Represents the optics of an ideal diffuse (lambertian) surface, i.e. one
	that reflects rays in a random direction (uniform distribution of
	directions in 3D, see tracer.sources.pillbox_sunshape_directions)
	"""
	def __init__(self, absorptivity=0.):
		self._abs = absorptivity
	
	def __call__(self, geometry, rays, selector):
		"""
		Arguments:
		geometry - a GeometryManager which knows about surface normals, hit
			points etc.
		rays - the incoming ray bundle (all of it, not just rays hitting this
			surface)
		selector - indices into ``rays`` of the hitting rays.
		"""
		directs = sources.pillbox_sunshape_directions(len(selector), ang_range=N.pi/2.)
		normals = geometry.get_normals()
		directs = N.sum(rotation_to_z(normals.T) * directs.T[:,None,:], axis=2).T

		outg = rays.inherit(selector,
			vertices=geometry.get_intersection_points_global(),
			energy=rays.get_energy()[selector]*(1. - self._abs),
			direction=directs, 
			parents=selector)

		if outg.has_property('spectra'):
			outg._spectra *= (1.-self._abs)

		return outg

class LambertianSpecular(object):
	"""
	Represents the optics of surface with mixed specular and diffuse characteristics. Specularity is the ratio of incident rays that are specularly reflected to the total number of rays incident on the surface.
	"""
	def __init__(self, absorptivity=0., specularity=0.5):
		self._abs = absorptivity
		self.specularity = specularity
	
	def __call__(self, geometry, rays, selector):
		"""
		Arguments:
		geometry - a GeometryManager which knows about surface normals, hit
			points etc.
		rays - the incoming ray bundle (all of it, not just rays hitting this
			surface)
		selector - indices into ``rays`` of the hitting rays.
		"""
		in_directs = rays.get_directions()[:,selector]
		normals = geometry.get_normals()
		directs = N.zeros(in_directs.shape)

		specular = N.random.rand(len(selector))<self.specularity

		directs[:,specular] = optics.reflections(in_directs[:,specular], normals[:,specular])
		direct_lamb = sources.pillbox_sunshape_directions(N.sum(~specular), ang_range=N.pi/2.)
		directs[:,~specular] = N.sum(rotation_to_z(normals[:,~specular].T) * direct_lamb.T[:,None,:], axis=2).T

		outg = rays.inherit(selector,
			vertices=geometry.get_intersection_points_global(),
			energy=rays.get_energy()[selector]*(1. - self._abs),
			direction=directs, 
			parents=selector)
		return outg
        

class LambertianSpecular_IAM(object):
	"""
	Represents the optics of surface with mixed specular and diffuse characteristics. Specularity is the ratio of incident rays that are specularly reflected to the total number of rays incident on the surface.
	"""
	def __init__(self, absorptivity=0., specularity=0.5, a_r=0.16):
		self._abs = absorptivity
		self.specularity = specularity
		self.a_r = a_r
	
	def __call__(self, geometry, rays, selector):
		"""
		Arguments:
		geometry - a GeometryManager which knows about surface normals, hit
			points etc.
		rays - the incoming ray bundle (all of it, not just rays hitting this
			surface)
		selector - indices into ``rays`` of the hitting rays.
		"""
		in_directs = rays.get_directions()[:,selector]
		normals = geometry.get_normals()
		directs = N.zeros(in_directs.shape)
		vertical = N.sum(directs*normals, axis=0)*normals
		cos_theta_AOI = N.sqrt(N.sum(vertical**2, axis=0))

		specular = N.random.rand(len(selector))<self.specularity

		directs[:,specular] = optics.reflections(in_directs[:,specular], normals[:,specular])
		direct_lamb = sources.pillbox_sunshape_directions(N.sum(~specular), ang_range=N.pi/2.)
		directs[:,~specular] = N.sum(rotation_to_z(normals[:,~specular].T) * direct_lamb.T[:,None,:], axis=2).T

		outg = rays.inherit(selector,
			vertices=geometry.get_intersection_points_global(),
			energy=rays.get_energy()[selector]*(1. - self._abs*(1.-N.exp(-cos_theta_AOI/self.a_r))/(1.-N.exp(-1./self.a_r))),
			direction=directs, 
			parents=selector)
            
		if outg.has_property('spectra'):
			outg._spectra *= (1. - self._abs*(1.-N.exp(-cos_theta_AOI/self.a_r))/(1.-N.exp(-1./self.a_r)))   
         
		return outg

"""
class BDRF_Cook_Torrance_isotropic(object):
	'''
	# Implements the Cook Torrance BDRF model using linear interpolationsand isotropic assumption
	#Directional absorptance is found by integration of the bdrf
	# Reflected direction by sampling of the normalised interpolated bdrf.
	'''
	def __init__(self, m, alpha, R_Lam, angular_res_deg=5., axisymmetric_i=True):

		ares_rad = angular_res_deg*N.pi/180.
		self._m = m
		self._alpha = alpha
		self.R_lam = R_Lam
		# build BDRFs for the relevant wavelengths and incident angles
		thetas_r, phis_r = N.linspace(0., N.pi/2., int(N.ceil(N.pi/2./ares_rad))), N.linspace(0., 2.*N.pi, int(N.ceil(N.pi/2./ares_rad)))
		thetas_i, phis_i = thetas_r, phis_r
		if axisymmetric_i: # if the bdrf is axisymmetric in incidence angle, no need to do all the phi incident
			phis_i = phis_i[[0,-1]]

		bdrfs = N.zeros((len(thetas_i), len(phis_i), len(thetas_r), len(phis_r)))
		for i, thi in enumerate(thetas_i):
			for j, phi in enumerate(phis_i):
				CT = regular_grid_Cook_Torrance(thetas_r_rad=thetas_r, phis_r_rad=phis_r, th_i_rad=thi, phi_i_rad=phi, m=m, R_dh_Lam=R_Lam, alpha=alpha)
				bdrfs[i,j] = CT[-1].reshape(len(thetas_r), len(phis_r))
		# build a linear interpolator
		points = (thetas_i, phis_i, thetas_r, phis_r)
		self.bdrf = BDRF_distribution(RegularGridInterpolator(points, bdrfs)) # This instance is a BDRF wilth all the necessary functions inclyuded for energy conservative sampling. /!\ Wavelength not included yet!				

	def get_incident_angles(self, directions, normals):
		vertical = N.sum(directions*normals, axis=0)*normals
		return N.arccos(N.sqrt(N.sum(vertical**2, axis=0)))
		
	def project_to_normals(self, directions, normals):
		return N.sum(rotation_to_z(normals.T) * directions.T[:,None,:], axis=2).T

	def __call__(self, geometry, rays, selector):
		# TODO: reflected direction orientation for non axisymmetric bdrf
		# Incident directions in the frame of reference of the geometry
		# find Normals
		normals = geometry.get_normals()
		# find theta_in
		directions = rays.get_directions()[:,selector]
		thetas_in = self.get_incident_angles(directions, normals)
		energy_out = rays.get_energy()[selector]
		# sample reflected directions:
		for i, theta_in in enumerate(thetas_in):
			# sample a reflected direction given theta_in		
			dhr = self.bdrf.DHR(theta_in, 0)
			theta_r, phi_r, weights = self.bdrf.sample(theta_in, 0, 1)
			energy_out[i] *= dhr*weights
			directions[:,i] = N.array([N.sin(theta_r)*N.cos(phi_r), N.sin(theta_r)*N.sin(phi_r), N.cos(theta_r)]).T
		### IMPORTANT: need to check for asymmetrical BRDF etc.
		outg = rays.inherit(selector,
			vertices=geometry.get_intersection_points_global(),
			energy=energy_out,
			direction=self.project_to_normals(directions, normals), 
			parents=selector)

		return outg
	#'''
"""
class PeriodicBoundary(object):
	'''
	The ray intersections incident on the surface are translated by a given period in the direction of the surface normal, creating a perdiodic boundary condition.
	'''
	def __init__(self, period):
		'''
		Argument:
		period: distance of periodic repetition. The ray positions are translated of period*normal vector for the next bundle, with same direction and energy.
		'''
		self.period = period
		
	def __call__(self, geometry, rays, selector):
		# This is done this way so that the rendering knows that there is no ray between the hit on th efirst BC and the new ray starting form the second. With this implementation, the outg rays are cancelled because their energy is 0 and only the outg2 are going forward.
		# set original outgoing energy to 0
		outg = rays.inherit(selector,
			vertices=geometry.get_intersection_points_global(),
			energy=N.zeros(len(selector)),
			direction=rays.get_directions()[:,selector], 
			parents=selector)
		# If the bundle is polychromatic, also get and cancel the spectra
		if rays.has_property('spectra'):
			spectra = rays.get_spectra()[:,selector]
			outg._spectra = N.zeros(spectra.shape)

		# Create new bundle with the updated positions and all remaining properties identical:
		outg2 = rays.inherit(selector, vertices=geometry.get_intersection_points_global()+self.period*geometry.get_normals(), parents=selector)

		# concatenate both bundles in one outgoing one
		outg = ray_bundle.concatenate_rays([outg, outg2])
		
		return outg

class RefractiveHomogenous(object):
	"""
	Represents the optics of a surface bordering homogenous media with 
	constant refractive index on each side. The specific index in which a
	refracted ray moves is determined by toggling between the two possible
	indices.
	"""
	def __init__(self, n1, n2, single_ray=True):
		"""
		Arguments:
		n1, n2 - scalars representing the homogenous refractive index on each
			side of the surface (order doesn't matter).
		single_ray - if True, only simulate a reflected or a refracted ray.
		"""
		self._ref_idxs = (n1, n2)
		self._single_ray = single_ray
	
	def toggle_ref_idx(self, current):
		"""
		Determines which refractive index to use based on the refractive index
		rays are currently travelling through.

		Arguments:
		current - an array of the refractive indices of the materials each of 
			the rays in a ray bundle is travelling through.
		
		Returns:
		An array of length(n) with the index to use for each ray.
		"""
		return N.where(current == self._ref_idxs[0], 
			self._ref_idxs[1], self._ref_idxs[0])
	
	def __call__(self, geometry, rays, selector):
		if len(selector) == 0:
			return ray_bundle.empty_bund()

		normals = geometry.get_normals()
		directions = rays.get_directions()
		
		# Check for refractions:
		n1 = rays.get_ref_index()[selector]
		n2 = self.toggle_ref_idx(n1)
		refr, out_dirs = optics.refractions(n1, n2, \
			directions[:,selector], normals)

		if not refr.any():
			return perfect_mirror(geometry, rays, selector)
			
		# Reflected energy:
		R = N.ones(len(selector))
		energy = rays.get_energy()
		R[refr] = optics.fresnel(directions[:,selector][:,refr],
			normals[:,refr], n1[refr], n2[refr])
		
		# The output bundle is generated by stacking together the reflected and
		# refracted rays in that order.
		inters = geometry.get_intersection_points_global()
		
		if self._single_ray:
			# Draw probability of reflection or refraction out of the reflected energy of refraction events:
			refl = N.random.uniform(size=R.shape)<=R
			# reflected rays are TIR OR rays selected to go to reflection
			sel_refl = selector[refl]
			sel_refr = selector[~refl]
			dirs_refr = N.zeros((3, len(selector)))
			dirs_refr[:,refr] = out_dirs
			dirs_refr = dirs_refr[:,~refl]
			
			reflected_rays = rays.inherit(sel_refl, vertices=inters[:,refl],
				direction=optics.reflections(
					directions[:,sel_refl],
					normals[:,refl]),
				energy=energy[selector][refl],
				parents=sel_refl)
		
			refracted_rays = rays.inherit(sel_refr, vertices=inters[:,~refl],
				direction=dirs_refr, parents=sel_refr,
				energy=energy[selector][~refl],
				ref_index=n2[~refl])
				
		else:
			reflected_rays = rays.inherit(selector, vertices=inters,
				direction=optics.reflections(
					directions[:,selector],
					normals),
				energy=energy[selector]*R,
				parents=selector)
		
			refracted_rays = rays.inherit(selector[refr], vertices=inters[:,refr],
				direction=out_dirs, parents=selector[refr],
				energy=energy[selector][refr]*(1 - R[refr]),
				ref_index=n2[refr])

		return reflected_rays + refracted_rays

class RefractiveAbsorbantHomogenous(RefractiveHomogenous):
	'''
	Same as RefractiveHomogenous but with absoption in the medium. This is an approximation where we only consider attenuation in the medium but not its influence on the fresnel coefficients.
	'''
	def __init__(self, n1, n2, k1, k2, single_ray=True, sigma=None):
		"""
		Arguments:
		n1, n2 - scalars representing the homogenous refractive index on each
			side of the surface (order doesn't matter).
		k1, k2 - extinction coefficients in each side of the surface. a1 corresponds to n1, a2 to n2.
		single_ray - if True, refraction or reflection is decided for each ray
			    based on the amount of reflection coefficient and only a single 
			    ray is launched in the next bundle. Otherwise, the ray is split into
			    two in the next bundle.
		"""
		RefractiveHomogenous.__init__(self, n1, n2, single_ray)
		self._ext_coeffs = (k1, k2)
		self._sigma = sigma

	def get_ext_coeff(self, current_n):
		return N.where(current_n == self._ref_idxs[0], 
			self._ext_coeffs[0], self._ext_coeffs[1])	

	def __call__(self, geometry, rays, selector):
		if len(selector) == 0:
			return ray_bundle.empty_bund()
			
		normals = geometry.get_normals()
		directions = rays.get_directions()
		energy = rays.get_energy()
		
		if self._sigma is not None:
			th = N.random.normal(scale=self._sigma, size=N.shape(normals[1]))
			phi = N.random.uniform(low=0., high=N.pi, size=N.shape(normals[1]))
			normal_errors = N.vstack((N.cos(th), N.sin(th)*N.cos(phi), N.sin(th)*N.sin(phi)))

			# Determine rotation matrices for each normal:
			rots_norms = rotation_to_z(normals.T)
			if rots_norms.ndim==2:
				rots_norms = [rots_norms]

			# Build the normal_error vectors in the local frame.
			for i in range(N.shape(normals)[1]):
				normals[:,i] = N.dot(rots_norms[i], normal_errors[:,i])
		
		# Check for refractions:
		n1 = rays.get_ref_index()[selector]
		n2 = self.toggle_ref_idx(n1)
		refr, out_dirs = optics.refractions(n1, n2, \
			directions[:,selector], normals)

		# The output bundle is generated by stacking together the reflected and
		# refracted rays in that order.
		inters = geometry.get_intersection_points_global()

		# attenuation in current medium:
		a = self.get_ext_coeff(n1)
		prev_inters = rays.get_vertices(selector)
		path_lengths = N.sqrt(N.sum((inters-prev_inters)**2, axis=0))
		wavelengths = rays.get_wavelengths(selector)

		energy = optics.attenuations(path_lengths=path_lengths, k=a, lambda_0=wavelengths, energy=energy[selector])

		if not refr.any():
			rays.set_energy(energy, selector=selector)
			return perfect_mirror(geometry, rays, selector)
		
		# Reflected energy:
		R = N.ones(len(selector))
		R[refr] = optics.fresnel(directions[:,selector][:,refr],
			normals[:,refr], n1[refr], n2[refr])

		if self._single_ray:
			# Draw probability of reflection or refraction out of the reflected energy of refraction events:
			refl = N.random.uniform(size=R.shape)<=R
			# reflected rays are TIR OR rays selected to go to reflection
			sel_refl = selector[refl]
			sel_refr = selector[~refl]
			dirs_refr = N.zeros((3, len(selector)))
			dirs_refr[:,refr] = out_dirs
			dirs_refr = dirs_refr[:,~refl]
			
			reflected_rays = rays.inherit(sel_refl, vertices=inters[:,refl],
				direction=optics.reflections(
					directions[:,sel_refl],
					normals[:,refl]),
				energy=energy[refl],
				parents=sel_refl)
		
			refracted_rays = rays.inherit(sel_refr, vertices=inters[:,~refl],
				direction=dirs_refr, parents=sel_refr,
				energy=energy[~refl],
				ref_index=n2[~refl])
			
		else:
			reflected_rays = rays.inherit(selector, vertices=inters,
				direction=optics.reflections(
					directions[:,selector],
					normals),
				energy=energy*R,
				parents=selector)
		
			refracted_rays = rays.inherit(selector[refr], vertices=inters[:,refr],
				direction=out_dirs, parents=selector[refr],
				energy=energy[refr]*(1 - R[refr]),
				ref_index=n2[refr])

		return reflected_rays + refracted_rays
		
class RefractiveScatteringHomogenous(RefractiveHomogenous):
	'''
	Same as RefractiveHomogenous but with sacttering in the medium.
	
	On interaction:
	1 - check if thee is any scattering
		1 - a) if scattering, perform the calculation of teh scattering directions using H-G.
	2 - If there is some non-scatterred light, it reaches a surface
		2 - a) Evaluate the refraction at that surface: rays can totally internally reflect 
		or refract 
		2 - b) If refraction event, choose wether to reflect or refract out of the surfcae 
		based on a random number
	3 - regroup rays and output bundle
	
	Currently scattering is handled using a scattering coefficient and a Henyey-Greenstein phase function.
	
	Arguments:
	n1, n2 - scalars representing the homogenous refractive index on each
			side of the surface (order doesn't matter).
	s_c1, s_c2 - Scattering coefficients of medium 1 and medium 2 in m-1
	g_HG - asymmetry factor for the Henyey-Greenstein phase function, from -1 to 1. 0 is anisotropic scattering.
	'''
	def __init__(self, n1, n2, s_c1, s_c2, g_HG, single_ray=True):
		RefractiveHomogenous.__init__(self, n1, n2, single_ray)
		
		self._s_cs = [s_c1, s_c2] #
		self.phase_function = Henyey_Greenstein(g_HG) # Henyey-Greenstein phase function parameter
		
	def toggle_media(self, current_ref_idx, current_s_c):
		"""
		Determines which refractive index to use based on the refractive index and scattering
		coefficient of the rays currently travelling through.

		Arguments:
		current_ref_idx, current_s_c - arrays of the refractive indices and scattering 
		coefficients of the materials each of the rays in a ray bundle is travelling through.
		
		Returns:
		An array of length(n) with the index to use for each ray.
		"""
		new_idx = self.toggle_ref_idx(current_ref_idx)
		new_s_c = N.where(current_s_c == self._s_cs[0], 
			self._s_cs[1], self._s_cs[0])

		return new_idx, new_s_c

	def __call__(self, geometry, rays, selector):
		if len(selector) == 0:
			return ray_bundle.empty_bund()

		# General raybundle variables
		normals = geometry.get_normals() # one normal per intresection found
		inters = geometry.get_intersection_points_global() # one vertex per intresection found
		directions = rays.get_directions() # one unit direction vector per ray
		energy = rays.get_energy() # value per ray
					
		# Check for scattering
		prev_inters = rays.get_vertices(selector)
		intersection_path_lengths = N.sqrt(N.sum((inters-prev_inters)**2, axis=0))
		s_cs = rays.get_scat_coeff(selector)
		
		# Determine which ray gets scattered:
		scat, scattered_path_lengths = optics.scattering(s_cs, intersection_path_lengths)
		sel_scat = selector[scat] # indices of the scattered rays in the whole incident bundle
		sel_nonscat = selector[~scat] # indices of the non-scattered rays in the whole incident bundle
		
		# The output bundle is generated by stacking together the scattered, reflected and
		# refracted rays in that order.
		
		# Scattered rays:
		if len(sel_scat):
			scat_vertices = prev_inters[:,scat] + scattered_path_lengths[scat]*directions[:, sel_scat]
			scat_ths, scat_phis = self.phase_function.sample(N.sum(scat))
			scat_directions = N.array([N.sin(scat_ths)*N.cos(scat_phis),
								N.sin(scat_ths)*N.sin(scat_phis), 
								N.cos(scat_ths)])
			scat_directions = rotate_z_to_normal(scat_directions, directions[:, sel_scat]) # rotate  with z pointing in directions
			
			scattered_rays = rays.inherit(sel_scat, vertices=scat_vertices,
							direction=scat_directions,
							energy=energy[sel_scat],
							parents=sel_scat)

		# Are there non-scattered rays?
		if len(sel_nonscat):
			# Check for refractions:
			n1 = rays.get_ref_index(sel_nonscat)
			s_cs = rays.get_scat_coeff(sel_nonscat)
			n2, s_c = self.toggle_media(n1, s_cs)
			refr, refr_dirs = optics.refractions(n1, n2,
				directions[:,sel_nonscat], normals[:,~scat]) # refr here notes non TIR situations
			
			nonscat_inters = inters[:,~scat]
			
			all_TIR = not refr.any()
			
			if all_TIR: # All TIR in refracted
				reflected_rays = rays.inherit(sel_nonscat,
					vertices=nonscat_inters,
					direction=optics.reflections(directions[:,sel_nonscat], normals[:,~scat]),
					energy=energy[sel_nonscat],
					parents=sel_nonscat)

			else:
				# Reflected energy:
				R = N.ones(len(sel_nonscat))
				R[refr] = optics.fresnel(directions[:,sel_nonscat][:,refr],
					normals[:,~scat][:,refr], n1[refr], n2[refr]) # TIR are left to 1 magically here.

				if self._single_ray:
					# Draw probability of reflection or refraction out of the reflected energy of refraction events:
					refl = N.random.uniform(size=R.shape)<=R
					
					# Reflected rays are TIR OR rays selected to go to reflection. 
					# Rays reflect all incident energy or transmit fully
					sel_refl = sel_nonscat[refl]
					sel_refr = sel_nonscat[~refl]

					reflected_rays = rays.inherit(sel_refl,
						vertices=nonscat_inters[:,refl],
						direction=optics.reflections(directions[:,sel_refl], normals[:,~scat][:,refl]),
						energy=energy[sel_refl],
						parents=sel_refl)

					dirs_refr = refr_dirs[:, ~refl[R<1.]]
					refracted_rays = rays.inherit(sel_refr,
						vertices=nonscat_inters[:,~refl],
						direction=dirs_refr,
						parents=sel_refr,
						energy=energy[sel_refr],
						ref_index=n2[~refl], 
						scat_coeff=s_c[~refl])

				else:
					reflected_rays = rays.inherit(sel_nonscat,
						vertices=inters,
						direction=optics.reflections(directions[:,sel_nonscat],normals),
						energy=energy*R,
						parents=sel_nonscat)

					refracted_rays = rays.inherit(selector[refr],
						vertices=inters[:,refr],
						direction=out_dirs,
						parents=selector[refr],
						energy=energy[refr]*(1 - R[refr]),
						ref_index=n2[refr],
						scat_coeff=s_c[refr])
										
		if len(sel_scat):
			if len(sel_nonscat):
				if all_TIR:
					ret_bundle = scattered_rays + reflected_rays
				else:
					ret_bundle = scattered_rays + reflected_rays + refracted_rays
			else:
				ret_bundle = scattered_rays
		else:
			if all_TIR:
				ret_bundle = reflected_rays
			else:
				ret_bundle = reflected_rays + refracted_rays
		return ret_bundle

class FresnelConductorHomogenous(object):
	'''
	Fresnel equation with a conductive medium instersected. The attenuation is total in a very short range in the intersected metal and refraction is not modelled. Only strictly valid for k2 >> 1 and in situations where the refracted ray is not interacting with the scene again (eg. not traversing thin metal volumes).
	'''
	def __init__(self, n1, material):
		"""
		Arguments:
		n1 - scalar representing the homogenous refractive index of a perfect dielectric (incident medium always as we assume skin depth absorption in the conductor).
		material - a material instance from the optical_constants module.
		"""
		self._n1 = n1
		self._material = material

	def __call__(self, geometry, rays, selector):
		if len(selector) == 0:
			return ray_bundle.empty_bund()
		
		inters = geometry.get_intersection_points_global()

		# Reflected energy:
		R_p, R_s, theta_ref = optics.fresnel_conductor(
				rays.get_directions()[:,selector],
				normals, 
				rays.get_wavelengths()[selector], 
				self._material, n1=self._n1)

		R = (R_p+R_s)/2. # randomly polarised reflection

		reflected_rays = rays.inherit(selector, vertices=inters,
				direction = optics.reflections(
				rays.get_directions()[:,selector],
				normals),
				energy = energy[selector]*R,
				parents = selector)
		
		return reflected_rays


class OpticsCallable(object):

	def __init__(self, optics, *args, **kwargs):
		"""
		Callable initialises a superclass for the optics that returns the processed bundle when called.
		Used for composition with the accountant classes.
		Arguments:
		optics - the optics manager class to actually use with input attributes (*args, **kwargs)
		"""
		self._opt = optics(*args, **kwargs)

	def __call__(self, geometry, rays, selector):
		newb = self._opt(geometry, rays, selector)
		return newb

class Accountant(ABC):
	def __init__(self):
		# Initialise counter
		self.reset()

	@abstractmethod
	def reset(self):
		# Reset counter
		pass

	@abstractmethod
	def count(self, geometry, rays, selector, new_bundle):
		# Accumulate data, done withing OpticsCallable __call__ method
		pass

	@abstractmethod
	def get_data(self):
		# Outputs counter
		return

class LocationAccountant(Accountant):
	def __init__(self):
		super().__init__()
		self.shorthand = 'Location'

	def reset(self):
		"""Clear the memory of hits (best done before a new trace)."""
		self._hits = []

	def count(self, geometry, rays, selector, new_bundle):
		self._hits.append(geometry.get_intersection_points_global())

	def get_data(self):
		"""
		Aggregate all hits from all stages of tracing into joined arrays.

		Returns:
		hits - the corresponding global coordinates for each hit-point.
		"""
		if not len(self._hits):
			return N.array([]).reshape(3,0)

		return N.hstack([h for h in self._hits if h.shape[1]])

class AbsorptionAccountant(Accountant):
	"""
	This optics manager remembers all of the locations where rays hit it
	in all iterations, and the energy absorbed from each ray.
	"""
	def __init__(self):
		super().__init__()
		self.shorthand = 'Absorber'

	def reset(self):
		"""Clear the memory of hits (best done before a new trace)."""
		self._absorbed = []

	def count(self, geometry, rays, selector, new_bundle):
		ein = rays.get_energy()[selector]
		eout = new_bundle.get_energy()
		self._absorbed.append(ein - eout)

	def get_data(self):
		"""
		Aggregate all hits from all stages of tracing into joined arrays.

		Returns:
		absorbed - the energy absorbed by each hit-point
		hits - the corresponding global coordinates for each hit-point.
		"""
		if not len(self._absorbed):
			return N.array([])

		return N.hstack([a for a in self._absorbed if len(a)])

class ReceptionAccountant(Accountant):
	"""
	This optics manager remembers all of the locations where rays hit it
	in all iterations, and the energy absorbed from each ray.
	"""
	def __init__(self):
		super().__init__()
		self.shorthand = 'Receptor'

	def reset(self):
		"""Clear the memory of hits (best done before a new trace)."""
		self._received = []

	def count(self, geometry, rays, selector, new_bundle):
		ein = rays.get_energy()[selector]
		self._received.append(ein)

	def get_data(self):
		"""
		Aggregate all hits from all stages of tracing into joined arrays.

		Returns:
		absorbed - the energy absorbed by each hit-point
		hits - the corresponding global coordinates for each hit-point.
		"""
		if not len(self._received):
			return N.array([])

		return N.hstack([a for a in self._received if len(a)])

class ScatteringAccountant(Accountant):
	'''
	in all iterations, and the energy that was scattered (here understood as not absorbed) for each ray.
	'''
	def __init__(self):
		super().__init__()
		self.shorthand = 'Scatterer'

	def reset(self):
		"""Clear the memory of hits (best done before a new trace)."""
		self._scattered = []
	
	def count(self, geometry, rays, selector, new_bundle):
		eout = new_bundle.get_energy()
		self._scattered.append(eout)
	
	def get_data(self):
		"""
		Aggregate all hits from all stages of tracing into joined arrays.
		
		Returns:
		_transmitted - the energy not absorbed by each hit-point
		"""
		if not len(self._scattered):
			return N.array([])
		
		return N.hstack([a for a in self._scattered if len(a)])

class DirectionAccountant(Accountant):
	"""
	This optics manager remembers all of the locations where rays hit it
	in all iterations, and the energy absorbed from each ray.
	"""
	def __init__(self):
		super().__init__()
		self.shorthand = 'Directional'

	def reset(self):
		"""Clear the memory of hits (best done before a new trace)."""
		self._directions = []
	
	def count(self, geometry, rays, selector, new_bundle):
		self._directions.append(rays.get_directions()[:,selector])

	def get_data(self):
		"""
		Aggregate all hits from all stages of tracing into joined arrays.
		
		Returns:
		super class method results followed by:
		directions - the corresponding unit vector directions for each hit-point.
		"""

		if not len(self._directions):
			return N.array([]).reshape(3,0)
		
		return N.hstack([d for d in self._directions if d.shape[1]])

class SpectralAccountant(Accountant):
	def __init__(self):
		super().__init__()
		self.shorthand = 'Spectral'

	def reset(self):
		"""Clear the memory of hits (best done before a new trace)."""
		self._wavelengths = []

	def count(self, geometry, rays, selector, new_bundle):
		self._wavelengths.append(rays.get_wavelengths()[selector])

	def get_data(self):
		"""
		Aggregate all hits from all stages of tracing into joined arrays.
		
		Returns:
		wavelengths - wavelength of each ray.
		"""
		if not len(self._absorbed):
			return N.array([])
		
		return N.hstack([w for w in self._wavelengths if len(w)])

class PolychromaticAccountant(Accountant):
	def __init__(self):
		super().__init__()
		self.shorthand = 'Polychromatic'

	def reset(self):
		"""Clear the memory of hits (best done before a new trace)."""
		self._wavelengths = []
		self._spectra = []
	
	def count(self, geometry, rays, selector, new_bundle):
		oldspectra = rays.get_spectra()[:,selector]
		self._wavelengths.append(new_bundle.get_wavelengths())
		self._spectra.append(oldspectra-new_bundle.get_spectra())
	
	def get_data(self):
		"""
		Returns:
		"""
		if not len(self._absorbed):
			return N.array([]), N.array([]).reshape(2,0)
		
		return N.concatenate([w for w in self._wavelengths], axis=-1), \
			N.concatenate([s for s in self._spectra], axis=-1)

'''
deprecated
class NormalAccountant(Accountant):
	"""
	"""
	def reset(self):
		"""Clear the memory of hits (best done before a new trace)."""
		self._normals = []
	
	def count(self, geometry, rays, selector, new_bundle):
		self._normals.append(geometry.get_normals())
	
	def get_data(self):
		"""
		Aggregate all hits from all stages of tracing into joined arrays.
		
		Returns:
		absorbed - the energy absorbed by each hit-point
		hits - the corresponding global coordinates for each hit-point.
		directions - the corresponding unit vector directions for each hit-point.
		"""
		if not len(self._absorbed):
			return N.array([]).reshape(3,0)
		
		return N.hstack([n for n in self._normals if n.shape[1]])
'''

class BiFacial(object):
	'''
	This optical manager separates the optical response between front side (+z in general, depending on the geometry manager) and back side properties.
	'''
	def __init__(self, OpticsCallable_front, OpticsCallable_back):
		self.OpticsCallable_front = OpticsCallable_front
		self.OpticsCallable_back = OpticsCallable_back

	def __call__(self, geometry, rays, selector):

		proj = N.around(N.sum(rays.get_directions()[:,selector]*geometry.up()[:,None], axis=0), decimals=6)
		back = proj > 0.
		outg = []

		if back.any():
			outg.append(self.OpticsCallable_back.__call__(geometry, rays, selector).inherit(N.nonzero(back)[0]))
		if ~back.all():
			outg.append(self.OpticsCallable_front.__call__(geometry, rays, selector).inherit(N.nonzero(~back)[0]))

		if len(outg)>1:
			outg = ray_bundle.concatenate_rays(outg)
		else: 
			outg = outg[0]

		return outg

	def get_all_hits(self):
		
		try:
			front_hits = self.OpticsCallable_front.get_all_hits()
		except:
			front_hits = []
		try:
			back_hits = self.OpticsCallable_back.get_all_hits()
		except:
			back_hits = []

		return front_hits, back_hits

	def reset(self):
		try:
			self.OpticsCallable_front.reset(self)
		except:
			pass
		try:
			self.OpticsCallable_back.reset(self)
		except:
			pass

# This stuff automatically generates the classes from optical callables using the relevant Accountants.
'''
def subclass_from_name(name, parent, optics_class):
	class NewClass(parent):
		optics_class = optics_class # this is weird but it works!
		def __init__(self, *args, **kwargs):
			parent.__init__(self, self.optics_class, *args, **kwargs)
	accountant_with_name(name, NewClass, parent)
'''

def accountant_with_name(name, class_template):
	classdict = {}
	for e in class_template.__dict__.items():
		classdict.update({e[0]:e[1]})
	globals()[name] = type(name, (OpticsCallable,), classdict)

def make_mixed_accountant_class(name, accountants, optics_class):
	class NewClass(OpticsCallable):
		optics_class = optics_class # this is very weird but it works! If we do not do this, we cannot pass the optics_class argument as a class name.
		def __init__(self, *args, **kwargs):
			OpticsCallable.__init__(self, self.optics_class, *args, **kwargs)
			self.accountants = accountants

		def __call__(self, geometry, rays, selector):
			new_bundle = super().__call__(geometry, rays, selector)
			for a in self.accountants:
				a.count(geometry, rays, selector, new_bundle)
			return new_bundle

		def reset(self):
			for a in self.accountants:
				a.reset()

		def get_all_hits(self):
			output = []
			for a in accountants:
				output.append(a.get_data())
			return output
	accountant_with_name(name, NewClass)
'''
def make_spectral_accountant_class(name, parent):
	class SpectralAccountant(parent):
		def __init__(self):
			parent.__init__(self)

		def reset(self):
			"""Clear the memory of hits (best done before a new trace)."""
			parent.reset(self)
			self._wavelengths = []

		def __call__(self, geometry, rays, selector):
			self._wavelengths.append(rays.get_wavelengths()[selector])
			newb = parent.__call__(self, geometry, rays, selector)
			return newb
			
		def get_data(self):
			"""
			Aggregate all hits from all stages of tracing into joined arrays.
			
			Returns:

			"""
			if not len(self._absorbed):
				return N.array([])
			
			return N.hstack([w for w in self._wavelengths if len(w)])

	accountant_with_name(name, SpectralAccountant, parent)

def make_polychromatic_accountant_class(name, parent):
	class PolychromaticAccountant(parent):
		def __init__(self, accountant):
			parent.__init__(self)

		def reset(self):
			"""Clear the memory of hits (best done before a new trace)."""
			parent.reset(self)
			self._wavelengths = []
			self._spectra = []

		def __call__(self, geometry, rays, selector):
			oldspectra = rays.get_spectra()[:,selector]
			newb = parent.__call__(self, geometry, rays, selector)
			self._wavelengths.append(newb.get_wavelengths())
			self._spectra.append(oldspectra-newb.get_spectra())
			return newb
			
		def get_data(self):
			"""
			Aggregate all hits from all stages of tracing into joined arrays.
			
			Returns:
			absorbed - the energy absorbed by each hit-point
			hits - the corresponding global coordinates for each hit-point.
			directions - the corresponding unit vector directions for each hit-point.
			wavelengths - values of wavelength that define teh spectrum carried by each ray
			spectra - normalised spectral power at the values defined by wavelengths.
			"""
			if not len(self._absorbed):
				return N.array([]), N.array([]).reshape(4,0), N.array([]).reshape(4,0)
			
			return N.hstack([a for a in self._absorbed]), \
				N.hstack([h for h in self._hits]), \
				N.hstack([d for d in self._directions]), \
				N.concatenate([w for w in self._wavelengths], axis=-1), \
				N.concatenate([s for s in self._spectra], axis=-1)
				
	accountant_with_name(name, PolychromaticAccountant, parent)
	'''
def make_accountant_classes(optical_class):
	for accs in mixed_accountants:
		names = [a().shorthand for a in accs]
		accountant_name = ''.join(names[::-1])
		newclass = optical_class.__name__ + accountant_name
		make_mixed_accountant_class(newclass, accs, optical_class)
		if accountant_name in aliases.keys():
			newclass = optical_class.__name__ + aliases[accountant_name]
			make_mixed_accountant_class(newclass, accs, optical_class)

# Make all accountant combinations to then use in making the optics callable classes with accountant composition
# Naming conventions: Start directional then location, then spectral and finish with energy accountants
# Eg.: DirectionalSpectralAbsorber
# opposite order of the output because it reads better when declaring
accountants_raw = Accountant.__subclasses__()
accountants = []
# TODO: decide this, split spectral and polychromatic, associate polychrimatic with energy processing and change order decided so far.
# Current order: accountants to respect get_all_hits() output convention: energy (absorbed or scattered), wavelengths or spectra, hits, directions
# Within each category, do alphabetical
accountant_types = ['Absorption', 'Reception', 'Scattering', 'Polychromatic', 'Spectral', 'Location', 'Direction']
for at in accountant_types:
	accountants.extend([a for a in accountants_raw if at in a.__name__])
# Aliases to make declaration easier for common accounatnts and ensure backward compatibility
aliases = {'LocationAbsorber':'Receiver', 'DirectionalLocationAbsorber':'Detector', 'LocationScatterer':'Transmitter'}
mixed_accountants = []
for i in range(1, len(accountants)):
	mix = (list(tup) for tup in combinations(accountants, i))
	# Polychromatic and spectral should be mutually exclusive
	for acc in mix:
		names = [a.__name__ for a in acc]
		if ('SpectralAccountant' in names) and ('PolychromaticAccountant' in names):
			continue
		mixed_accountants.append(acc)

# The following lines run the accountant making stuff upon loading the module.
not_optics = ['Accountant', 'BiFacial'] # to exclude these classes from the factory
# Find all classes from this module that are optics
names = inspect.getmembers(sys.modules[__name__], lambda member: inspect.isclass(member) and member.__module__ == __name__ and member.__name__ not in not_optics)
# Make all accountants combinations for each optics class
for name, obj in names:
	optics_class = locals()[name]
	make_accountant_classes(optics_class)

# vim: ts=4
