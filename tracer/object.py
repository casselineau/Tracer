# Defines an object class, where an object is defined as an assembly of surfaces.

import numpy as N
from tracer.spatial_geometry import general_axis_rotation
from tracer.assembly import Assembly

class AssembledObject(Assembly):
	""" Defines an assembly of surfaces as an object. The object has its own set of 
	coordinates such that each surface composing the object can be described in terms of
	the object's coordinate system, and thus the user can rotate or translate the entire
	object together as one piece.
	The object also tracks refractive indices as a ray bundle leaves or enters a new
	material.
	"""
	def __init__(self, surfs=None, bounds=None, location=None, rotation=None, transform=None):
		"""
		Attributes:
		surfaces - a list of Surface objects
		bounds - a list of Boundary objects that the surfaces are limited 
			by.
		location - Location of the object
		rotation - 3 by 3 rotation matrix of teh object
		transform - a 4x4 array representing the homogenous transformation 
			matrix of this object relative to the coordinate system of its 
			container (this can be overridden in the Assembly.add_object method). Transform overrides the location and rotation arguments.
		"""
		# Use the supplied values or some defaults:
		if surfs is None:
			self.surfaces = []
		else:
			self.surfaces = surfs
		
		if bounds is None:
			self.boundaries = []
		else:
			self.boundaries = bounds

		if not hasattr(self.surfaces, '__len__'): # take care of non listed single surface cases
			self.surfaces = [self.surfaces]
		if not hasattr(self.boundaries, '__len__'):
			self.boundaries = [self.boundaries]

		if transform is None:
			transform = N.eye(4)
			if rotation is not None:
				transform[:3,:3] = rotation
			if location is not None:
				transform[:3,3] = location

		self.set_transform(transform)

   
	def get_surfaces(self):
		return self.surfaces

	def add_surface(self, surface):
		"""Adds a surface to the object
		Arguments:  surface - a surface object
		"""
		self.surfaces.append(surface)
		self.transform_children()

	def add_boundary(self, boundary):
		"""Adds a boundary to the object. Surfaces not enclosed by the boundary
		sphere will not count as hit.
		Arguments: boundary shape object
		"""
		self.boundaries.append(boundary)
		self.transform_children()

	def get_boundaries(self):
		if self.boundaries is None:
			self.boundaries = []
		return self.boundaries
	
	def transform_children(self, assembly_transform=N.eye(4)):
		"""Transforms an object if the assembly is transformed""" 
		const_t = self.get_transform()
		for child in self.surfaces + self.boundaries:
			child.transform_frame(N.dot(assembly_transform, const_t))
	
	def own_rays(self, rays, surface_id):
		"""
		Decide which of the rays continue to propagate inside the object, so
		only the object's surfaces need be checked for intersection.
		This default implementation owns nothing.
		
		Arguments:
		rays - the RayBundle to check. 
		surface_id - the index of the surface which generated this bundle.
		
		Returns:
		a boolean array of length rays.get_num_rays() with False if not owned,
			True if owned.
		"""
		return N.zeros(rays.get_num_rays(), dtype=bool)
	
	def surfaces_for_next_iteration(self, rays, surface_id):
		"""
		Informs the ray tracer that some of the surfaces can be skipped in the
		next ireration for some of the rays.
		This default implementation marks all surfaces as relevant to all rays.
		
		Arguments:
		rays - the RayBundle to check. 
		surface_id - the index of the surface which generated this bundle.
		
		Returns:
		an array of size s by r for s surfaces in this object and r rays,
			stating whether ray i=1..r should be intersected with surface j=1..s
			in the next iteration.
		"""
		return N.ones((len(self.surfaces), rays.get_num_rays()), dtype=bool)

	def get_scene_graph(self,resolution=None, fluxmap=None, trans=False, vmin=None, vmax=None):
		n = self.get_scene_graph_transform()

		for sfc in self.surfaces:
			n.addChild(sfc.get_scene_graph(resolution, fluxmap, trans, vmin, vmax))

		return n


# vim: ts=4
