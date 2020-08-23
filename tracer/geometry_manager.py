# -*- coding: utf-8 -*-
# Define a base class for managing geometry of an optical surface.
# This is an abstract class defining the interface.
# TODO: explore abstract meta-classes in Python.

import numpy as N
import pivy.coin as coin

class GeometryManager(object):
	def find_intersections(self, frame, ray_bundle):
		"""
		First part of the trace protocol: tell the surface about the ray bundle
		to check. The subclass should respond with an array of parametric
		locations along the rays, where each rays intersects the surface.
		
		Arguments:
		frame - a 4x4 homogenous transform (array) representing the surface's
			frame in global coordinates.
		ray_bundle - a RayBundle instance with the information on the incoming
			bundle.
		"""
		# FIXME does find_intersections really need to include a 'frame' parameter?
		self._working_frame = frame
		self._working_bundle = ray_bundle
		
		# This must be extended to return the correct result!
		# This needs to be updated for use in Python3 - temporarily deactivated the check!

		#if isinstance(self, GeometryManager):
		#	raise TypeError("Find intersections must be extended by a base class")

	def up(self):
		"""
		Returns a single direction that is considered "up" in the woking frame
		(the Z axis) in global coordinates.
		"""
		return self._working_frame[:3,2]
	
	def done(self):
		"""
		Discard internal data structures. After calling done(), the information
		on the latest trace iteration is lost. 
		"""
		if hasattr(self, '_working_frame'):
			del self._working_frame
			del self._working_bundle
   
	def select_rays(self, idxs):
		"""
		Inform the surface that only the rays at indices `idxs` will be used.
		"""
		pass
	
	def get_normals(self):
		"""
		Return a 3 by n array with the normals to the surface at each of the
		previously selected hit points.
		"""
		pass
	
	def get_intersection_points_global(self):
		"""
		Return the intersection points of the previously selected rays, in
		global coordinates.
		"""
		pass

	def get_scene_graph(self, resolution=None):
		"""
		Any object that provides a nice QuadMesh from the previous code should be able to render in Coin3D with with the following...

		Prefer to allow objects to calculate a 'suitable' resolution based on size.
		"""
		return self.mesh(resolution)


