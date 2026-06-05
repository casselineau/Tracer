"""
Geometry managers based on a cylinder along the Z axis.

References:
.. [1] http://www.siggraph.org/education/materials/HyperGraph/raytrace/rtinter4.htm
"""

from .quadric import QuadricGM
import numpy as N
import logging

class InfiniteCylinder(QuadricGM):
	"""
	A cylindrical surface infinitely long on the Z axis.
	"""
	def __init__(self, diameter):
		self._R = diameter/2.
		QuadricGM.__init__(self)

	def _normals(self, verts, dirs):
		# Move to local coordinates
		hit = N.dot(N.linalg.inv(self._working_frame), N.vstack((verts.T, N.ones(verts.shape[0]))))
		dir_loc = N.dot(self._working_frame[:3,:3].T, dirs.T)
		
		# The local external normal is made from the X,Y components of the vertex:
		local_norm = N.vstack((hit[:2], N.zeros(hit.shape[1])))
		local_norm /= self._R

		# Choose whether the normal is inside or outside:
		local_norm[:, N.sum(local_norm[:2] * dir_loc[:2], axis=0) > 0.] *= -1.
		
		# Back to global coordinates:
		return N.dot(self._working_frame[:3,:3], local_norm)
	
	def get_ABC(self, bundle):
		"""
		Finds the coefficients of the quadratic equation for the intersection
		of a ray and the cylinder. See [1]_.
		
		Arguments:
		bundle - a RayBundle instance with rays for which to get the
			coefficients.
		
		Returns:
		A, B, C - satisfying A*t**2 + B*t + C = 0 such that the intersection
			points are at distance t from each ray's vertex.
		"""
		# Transform the the direction and position of the rays temporarily into the
		# frame of the paraboloid for calculations
		d = N.dot(self._working_frame[:3,:3].T, bundle.get_directions())
		v = N.dot(N.linalg.inv(self._working_frame), N.vstack((bundle.get_vertices(), N.ones(d.shape[1]))))[:3]
		
		A = N.sum(d[:2]**2, axis=0)
		B = 2.*N.sum(d[:2]*v[:2], axis=0)
		C = N.sum(v[:2]**2, axis=0) - self._R**2
		
		return A, B, C

class FiniteCylinder(InfiniteCylinder):
	"""
	This geometry manager represents a cylinder with a given height, centered
	on its origin, (so that the top Z is at height/2).
	"""
	def __init__(self, diameter, height, ang_range=[0., 2.*N.pi]):
		self._half_h = height/2.
		assert len(ang_range) == 2
		self._ang_range = ang_range
		InfiniteCylinder.__init__(self, diameter)
	
	def _select_coords(self, coords, prm):
		"""
		Choose between two intersection points on a quadric surface.
		This implementation extends QuadricGM's behaviour by not choosing
		intersections higher or lower than half the cylinder height.

		Arguments:
		coords - a 2 by 3 by n array whose each column is the global coordinates
			of one intersection point of a ray with the cylinder.
		prm - the corresponding parametric location on the ray where the
			intersection occurs.

		Returns:
		The index of the selected intersection, or None if neither will do.
		"""
		select = N.empty(prm.shape[1])
		select.fill(N.nan)

		proj = N.linalg.inv(self._working_frame)
		coords = N.concatenate((coords, N.ones((2,1,coords.shape[-1]))), axis=1)

		local = []
		for i in range(coords.shape[0]):
			local.append(N.dot(proj[:-1,:], coords[i]))
		#height= N.sum(proj[None,2,:,None] * N.concatenate((coords, N.ones((2,1,coords.shape[-1]))), axis=1), axis=1)
		local = N.array(local)

		height = local[:,2]
		angs = N.arctan2(local[:,1],local[:,0])
		angs[angs<0] = 2*N.pi+angs[angs<0]
		inside_height = (abs(height) <= self._half_h)
		inside_ang_range = N.logical_and(angs>=self._ang_range[0], angs<=self._ang_range[1])

		inside = N.logical_and(inside_height, inside_ang_range)
		positive = prm > 1e-6
		hitting = inside & positive
		select[N.logical_and(*hitting)] = 1
		one_hitting = N.logical_xor(*hitting)
		select[one_hitting] = N.nonzero(hitting.T[one_hitting,:])[1]

		return select
	
	def mesh(self, resolution = None):
		"""
		Represent the surface as a mesh in local coordinates. Uses cylindrical
		bins, i.e. the points are equally distributed by angle and axial 
		location, not by x,y.

		Arguments:
		resolution - in points per unit length (so the number of points 
			returned is O(height*pi*diameter*resolution**2))

		Returns:
		x, y, z - each a 2D array holding in its (i,j) cell the x, y, and z
			coordinate (respectively) of point (i,j) in the mesh.
		"""
		if resolution is None:
			resolution=40

		# note: current mesh has no detail in the axial direction, just start/end points -- JP
		h = N.linspace(-self._half_h, self._half_h, resolution+1)
		angs = N.linspace(self._ang_range[0], self._ang_range[1], resolution+1)

		x = N.tile(self._R*N.cos(angs), (len(h), 1))
		y = N.tile(self._R*N.sin(angs), (len(h), 1))
		z = N.tile(h[:,None], (1, len(angs)))

		return x, y, z

	def get_fluxmap(self, eners, local_coords, resolution):
		'''
		Cylindrical mesh based fluxmap for the dish.
		'''
		heights = local_coords[2]
		angs = N.arctan2(local_coords[1], local_coords[0])

		angs[angs<0.] = angs[angs<0.]+2.*N.pi

		h_bins = N.linspace(-self._half_h, self._half_h, resolution+1)
		ang_bins = N.linspace(self._ang_range[0], self._ang_range[1], resolution+1)		

		eners = N.histogram2d(heights, angs, bins=[h_bins, ang_bins], weights=eners)[0]

		dh = N.abs(N.repeat(N.vstack(h_bins[1:]-h_bins[:-1]), len(ang_bins)-1, axis=1)).T
		dangs = N.repeat(N.vstack(ang_bins[1:]-ang_bins[:-1]), len(h_bins)-1, axis=1)
		areas = dangs*self._R*dh

		flux = eners/areas

		return N.hstack(flux)
		
class RectCutCylinder(FiniteCylinder):

	def __init__(self, diameter, height, w, h):
		FiniteCylinder.__init__(self, diameter, height)
		self.half_dims = N.array([w/2., h/2.])
		if (N.sqrt(N.sum(self.half_dims**2))<=self._R).all():
			logging.error('Bad rectangular cut cylindershape, width and height too small')
			stop

	def _select_coords(self, coords, prm):
		select = N.empty(prm.shape[1])
		select.fill(N.nan)

		# Projects the hit coordinates in a local frame on the z axis.
		proj = N.linalg.inv(self._working_frame)
		coords = N.concatenate((coords, N.ones((2,1,coords.shape[-1]))), axis=1)
		local = []
		for i in range(coords.shape[0]):
			local.append(N.dot(proj[:-1,:], coords[i]))
		local = N.array(local)
		x, y, height = local[:,0], local[:,1], local[:,2]
		
		# Checks if the local_z-projected hit coords are in the actual height of the furstum
		# and if the parameter is positive so that the ray goes ahead.
		inside_height = (-self._half_h <= height) & (height <= self._half_h)
		absx, absy = N.abs(x), N.abs(y)
		inside_w = absx <= self.half_dims[0]
		inside_h = absy <= self.half_dims[1]
   
		inside = inside_height & inside_w & inside_h
		positive = prm > 1e-6
		hitting = inside & positive

		# Choses between the two intersections offered by the surface.
		select[N.logical_and(*hitting)] = True
		one_hitting = N.logical_xor(*hitting)
		select[one_hitting] = N.nonzero(hitting.T[one_hitting,:])[1]

		return select

	def mesh(self, resolution=40):
		"""
		Represent the surface as a mesh in local coordinates. Uses polar
		bins, i.e. the points are equally distributed by angle and radius,
		not by x,y.
		
		Arguments:
		resolution - in points per unit length (so the number of points 
			returned is O(A*resolution**2) for area A)
		
		Returns:
		x, y, z - each a 2D array holding in its (i,j) cell the x, y, and z
			coordinate (respectively) of point (i,j) in the mesh.
		"""
		if resolution is None:
			resolution = 20

		# frame corners:
		top_right_corner_angle = N.round(N.arctan2(*self.half_dims[::-1]), decimals=9)

		phis = N.array([top_right_corner_angle, N.pi-top_right_corner_angle,  N.pi+top_right_corner_angle, 2.*N.pi-top_right_corner_angle, 2.*N.pi+top_right_corner_angle])
		
		# intersection of circular perimeter and frame:
		r_beyond = self._R>=self.half_dims
		crossing_angle = []
		if r_beyond.any():
			if r_beyond[0]:
				crossing_angle.append(N.arccos(self.half_dims[0]/self._R))
			if r_beyond[1]: 
				crossing_angle.append(N.arcsin(self.half_dims[1]/self._R))
			crossing_angle = N.round(crossing_angle, decimals=10)
			crossing_angles_in = N.hstack([crossing_angle, N.pi-crossing_angle, N.pi+crossing_angle, 2.*N.pi-crossing_angle, 2.*N.pi+crossing_angle])		
			phis = N.hstack([phis, crossing_angles_in])
			
		# Sort angles
		phis = N.sort(phis)
		
		# make azimuth angles
		angs = []
		for i, p in enumerate(phis[:-1]):
			angresloc = N.ceil((phis[i+1]-p)/(2.*N.pi)*resolution)
			if angresloc < 2:
				angresloc = 2
			angs.append(N.linspace(p, phis[i+1], int(angresloc))[:-1])
		angs.append(phis[-1])
		angs = N.unique(N.round(N.hstack(angs), decimals=8))
		
		# Look at the distance of the frame point at each angle. If it is bigger than the radius, use the radius, otherwise, skipp as it is a hole and close the submesh.
		rs = N.zeros((len(angs), resolution+1))
		meshes = []
		startang = 0
		current_mesh = []
		for i, a in enumerate(angs):
			rx = N.abs(N.round(self.half_dims[0]/N.cos(a), decimals=5))
			ry = N.abs(N.round(self.half_dims[1]/N.sin(a), decimals=5))
			r = N.amin([rx, ry])

			if r==self._R:
				current_mesh.append(i)
				# Make and finish this mesh as we have a node where the rectangular border crosses.
				rs[i] = N.ones(resolution+1)*r
				if len(current_mesh)>1:
					x = rs[current_mesh].T*N.cos(angs[current_mesh])
					y = rs[current_mesh].T*N.sin(angs[current_mesh])
					z = N.tile(N.linspace(-self._half_h, self._half_h, x.shape[0]), (x.shape[1],1)).T

					meshes.append(x)
					meshes.append(y)
					meshes.append(z)
					current_mesh = [] 
			elif r>self._R:
				current_mesh.append(i)
				rs[i] = N.ones(resolution+1)*self._R
				
			if a == angs[-1]:
				x = rs[current_mesh].T*N.cos(angs[current_mesh])
				y = rs[current_mesh].T*N.sin(angs[current_mesh])
				z = N.tile(N.linspace(-self._half_h, self._half_h, x.shape[0]), (x.shape[1],1)).T
				meshes.append(x)
				meshes.append(y)
				meshes.append(z)
 
		# continuous mesh, well behaved shape
		if len(meshes) == 0: # there was no mesh break therefore and no mesh added
			x = rs.T*N.cos(angs)
			y = rs.T*N.sin(angs)
			z = N.tile(N.linspace(-self._half_h, self._half_h, x.shape[0]), (x.shape[1],1)).T

		return meshes


