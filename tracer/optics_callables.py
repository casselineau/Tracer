# -*- coding: utf-8 -*-
# A collection of callables and tools for creating them, that may be used for
# the optics-callable part of a Surface object.

from . import optics, ray_bundle, sources
from .spatial_geometry import rotation_to_z
import numpy as N

class Reflective(object):
	"""
	Generates a function that represents the optics of an opaque, absorptive
	surface with specular reflections.
	
	Arguments:
	absorptivity - the amount of energy absorbed before reflection.
	
	Returns:
	refractive - a function with the signature required by Surface.
	"""
	def __init__(self, absorptivity):
		self._abs = absorptivity
	
	def __call__(self, geometry, rays, selector):

		outg = rays.inherit(selector,
			vertices=geometry.get_intersection_points_global(),
			direction=optics.reflections(rays.get_directions()[:,selector], geometry.get_normals()),
			energy=rays.get_energy()[selector]*(1. - self._abs),
			parents=selector)
		return outg

	def reset(self):
		pass

def Reflective_IAM(object):
	'''
	Generates a function that performs specular reflections from an opaque absorptive surface modified by the Incidence Angle Modifier from: Martin and Ruiz: https://pvpmc.sandia.gov/modeling-steps/1-weather-design-inputs/shading-soiling-and-reflection-losses/incident-angle-reflection-losses/martin-and-ruiz-model/. 
	'''
	def __init__(self, absorptivity, a_r):
		self._abs = absorptivity
		self.a_r = a_r
	
	def __call__(self, geometry, rays, selector):

		theta_AOI = N.arccos(N.dot(rays.get_directions()[:,selector], -geometry.get_normals()))
		outg = rays.inherit(selector,
			vertices=geometry.get_intersection_points_global(),
			direction=optics.reflections(rays.get_directions()[:,selector], geometry.get_normals()),
			energy=rays.get_energy()[selector]*(1. - self._abs*(1.-N.exp(-N.cos(theta_AOI)/a_r))/(1.-N.exp(-1./a_r))),
			parents=selector)
		return outg

	def reset(self):
		pass

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
				phi = N.random.uniform(low=0., high=N.pi, size=N.shape(ideal_normals[1]))
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

			#normal_errors = N.dot(geometry._working_frame[:3,:3], N.vstack((normal_errors_x, normal_errors_y, normal_errors_z)))
			#real_normals = ideal_normals + normal_errors
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
		return outg

	def reset(self):
		pass


class AbsorptionAccountant(object):
	"""
	This optics manager remembers all of the locations where rays hit it
	in all iterations, and the energy absorbed from each ray.
	"""
	def __init__(self, real_optics, absorptivity, **kwargs):
		"""
		Arguments:
		real_optics - the optics manager class to actually use. Expected to
			have the _abs protected attribute, and accept absorptivity as its
			only constructor argument (as in Reflective and
			LambertianReflector below).
		absorptivity - to be passed to a new real_optics object.
		"""
		self._opt = real_optics(absorptivity, **kwargs)
		"""
		if sigma==None:
			self._opt = real_optics(absorptivity)
		else:
			self._opt = real_optics(absorptivity, sigma, bi_var)
		"""
		self.reset()
	
	def reset(self):
		"""Clear the memory of hits (best done before a new trace)."""
		self._absorbed = []
		self._hits = []
	
	def __call__(self, geometry, rays, selector):
		self._absorbed.append(rays.get_energy()[selector]*self._opt._abs)
		self._hits.append(geometry.get_intersection_points_global())
		return self._opt(geometry, rays, selector)
	
	def get_all_hits(self):
		"""
		Aggregate all hits from all stages of tracing into joined arrays.
		
		Returns:
		absorbed - the energy absorbed by each hit-point
		hits - the corresponding global coordinates for each hit-point.
		"""
		if not len(self._absorbed):
			return N.array([]), N.array([]).reshape(3,0)
		
		return N.hstack([a for a in self._absorbed if len(a)]), \
			N.hstack([h for h in self._hits if h.shape[1]])

class DirectionAccountant(AbsorptionAccountant):
	"""
	This optics manager remembers all of the locations where rays hit it
	in all iterations, and the energy absorbed from each ray.
	"""
	def __init__(self, real_optics, absorptivity, **kwargs):
		"""
		Arguments:
		real_optics - the optics manager class to actually use. Expected to
			have the _abs protected attribute, and accept absorptivity as its
			only constructor argument (as in Reflective and
			LambertianReflector below).
		absorptivity - to be passed to a new real_optics object.
		"""
		AbsorptionAccountant.__init__(self, real_optics, absorptivity, **kwargs)
	
	def reset(self):
		"""Clear the memory of hits (best done before a new trace)."""
		AbsorptionAccountant.reset(self)
		self._directions = []
	
	def __call__(self, geometry, rays, selector):	
		AbsorptionAccountant.__call__(self, geometry, rays, selector)
		self._directions.append(rays.get_directions()[:,selector])
		return self._opt(geometry, rays, selector)
	
	def get_all_hits(self):
		"""
		Aggregate all hits from all stages of tracing into joined arrays.
		
		Returns:
		absorbed - the energy absorbed by each hit-point
		hits - the corresponding global coordinates for each hit-point.
		directions - the corresponding unit vector directions for each hit-point.
		"""
		if not len(self._absorbed):
			return N.array([]), N.array([]).reshape(3,0), N.array([]).reshape(3,0)
		
		return N.hstack([a for a in self._absorbed if len(a)]), \
			N.hstack([h for h in self._hits if h.shape[1]]), \
			N.hstack([d for d in self._directions if d.shape[1]])


class ReflectiveReceiver(AbsorptionAccountant):
	"""A wrapper around AbsorptionAccountant with a Reflective optics"""
	def __init__(self, absorptivity=1):
		AbsorptionAccountant.__init__(self, Reflective, absorptivity)

class ReflectiveDetector(DirectionAccountant):
	"""A wrapper around DirectionAccountant with a Reflective optics"""
	def __init__(self, absorptivity=0):
		DirectionAccountant.__init__(self, Reflective, absorptivity)
		
class OneSidedReflectiveReceiver(AbsorptionAccountant):
	"""A wrapper around AbsorptionAccountant with a Reflective optics"""
	def __init__(self, absorptivity=1):
		AbsorptionAccountant.__init__(self, OneSidedReflective, absorptivity)

class OneSidedReflectiveDetector(DirectionAccountant):
	"""A wrapper around DirectionAccountant with a Reflective optics"""
	def __init__(self, absorptivity=0):
		DirectionAccountant.__init__(self, OneSidedReflective, absorptivity)

class RealReflectiveReceiver(AbsorptionAccountant):
	"""A wrapper around AbsorptionAccountant with a RealReflective optics"""
	def __init__(self, absorptivity=0, sigma=0, bi_var=True):
		AbsorptionAccountant.__init__(self, RealReflective, absorptivity, sigma, bi_var)
		
class RealReflectiveDetector(DirectionAccountant):
	"""A wrapper around DirectionAccountant with a RealReflective optics"""
	def __init__(self, absorptivity=0, sigma=0, bi_var=True):
		DirectionAccountant.__init__(self, RealReflective, absorptivity, sigma=sigma, bi_var=bi_var)

class OneSidedRealReflectiveReceiver(AbsorptionAccountant):
	"""A wrapper around AbsorptionAccountant with a AbsorberRealReflector optics"""
	def __init__(self, absorptivity=0, sigma=0, bi_var=True):
		AbsorptionAccountant.__init__(self, OneSidedRealReflective, absorptivity, sigma=sigma, bi_var=bi_var)

class OneSidedRealReflectiveDetector(DirectionAccountant):
	"""A wrapper around DirectionAccountant with AbsorberRealReflector optics"""
	def __init__(self, absorptivity=0, sigma=0, bi_var=True):
		DirectionAccountant.__init__(self, OneSidedRealReflective, absorptivity, sigma=sigma, bi_var=bi_var)
		
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
		
class RefractiveHomogenous(object):
	"""
	Represents the optics of a surface bordering homogenous media with 
	constant refractive index on each side. The specific index in which a
	refracted ray moves is determined by toggling between the two possible
	indices.
	"""
	def __init__(self, n1, n2):
		"""
		Arguments:
		n1, n2 - scalars representing the homogenous refractive index on each
			side of the surface (order doesn't matter).
		"""
		self._ref_idxs = (n1, n2)
	
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
		
		n1 = rays.get_ref_index()[selector]
		n2 = self.toggle_ref_idx(n1)
		refr, out_dirs = optics.refractions(n1, n2, \
			rays.get_directions()[:,selector], geometry.get_normals())
		
		if not refr.any():
			return perfect_mirror(geometry, rays, selector)
		
		# Reflected energy:
		R = N.ones(len(selector))
		R[refr] = optics.fresnel(rays.get_directions()[:,selector][:,refr],
			geometry.get_normals()[:,refr], n1[refr], n2[refr])
		
		# The output bundle is generated by stacking together the reflected and
		# refracted rays in that order.
		inters = geometry.get_intersection_points_global()
		reflected_rays = rays.inherit(selector, vertices=inters,
			direction=optics.reflections(
				rays.get_directions()[:,selector],
				geometry.get_normals()),
			energy=rays.get_energy()[selector]*R,
			parents=selector)
		
		refracted_rays = rays.inherit(selector[refr], vertices=inters[:,refr],
			direction=out_dirs, parents=selector[refr],
			energy=rays.get_energy()[selector][refr]*(1 - R[refr]),
			ref_index=n2[refr])
		
		return reflected_rays + refracted_rays
		

class LambertianReflector(object):
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
		return outg

class LambertianReceiver(AbsorptionAccountant):
	"""A wrapper around AbsorptionAccountant with LambertianReflector optics"""
	def __init__(self, absorptivity=1.):
		AbsorptionAccountant.__init__(self, LambertianReflector, absorptivity)

class LambertianDetector(DirectionAccountant):
	"""A wrapper around DirectionAccountant with LambertianReflector optics"""
	def __init__(self, absorptivity=0.):
		DirectionAccountant.__init__(self, LambertianReflector, absorptivity)

class SemiLambertianReflector(object):
	"""
	Represents the optics of an semi-diffuse surface, i.e. one
	that absrobs and reflects rays in a random direction if they come in a certain angular range and fully specularly if they come from a larger angle
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
		return outg

class SemiLambertianReceiver(AbsorptionAccountant):
	"""A wrapper around AbsorptionAccountant with LambertianReflector optics"""
	def __init__(self, absorptivity=1., ang_range=N.pi/2.):
		AbsorptionAccountant.__init__(self, SemiLambertianReflector, absorptivity, ang_range)

class SemiLambertianDetector(DirectionAccountant):
	"""A wrapper around DirectionAccountant with LambertianReflector optics"""
	def __init__(self, absorptivity=0., ang_range=N.pi/2.):
		DirectionAccountant.__init__(self, SemiLambertianReflector, absorptivity, ang_range)

class IAMReceiver(AbsorptionAccountant):
	"""A wrapper around DirectionAccountant with Refective_IAM optics"""
	def __init__(self, absorptivity=1., ang_range=N.pi/2.):
		AbsorptionAccountant.__init__(self, Reflective_IAM, absorptivity, a_r)

class IAMdetector(AbsorptionAccountant):
	"""A wrapper around DirectionAccountant with Refective_IAM optics"""
	def __init__(self, absorptivity=1., ang_range=N.pi/2.):
		AbsorptionAccountant.__init__(self, Reflective_IAM, absorptivity, a_r)

