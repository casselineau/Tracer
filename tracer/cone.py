# Implements a circular conical surface

import numpy as N
from tracer.quadric import QuadricGM
import logging

class InfiniteCone(QuadricGM):
	"""
	Implements the geometry of an infinite circular conical surface. That
	means the sloping side-walls of the cone, doesn't include the base.
	"""
	def __init__(self, c, a=0):
		"""		  
		Arguments: 
		c - cone gradient (r/h)
		a - position of cone apex on z axis
		
		Cone equation is x**2 + y**2 = (c*(z-a))**2
		
		Private attributes:													  
		c - cone gradient (r/h)
		a - position of cone apex on z axis
		""" 
		QuadricGM.__init__(self)
		self.c = float(c)
		self.a = float(a)

	def _normals(self, hits, directs):
		"""
		Finds the normal to the cone in a bunch of intersection points, by
		taking the derivative and rotating it. Used internally by quadric.
		
		Arguments:
		hits - the coordinates of intersections, as an n by 3 array.
		directs - directions of the corresponding rays, n by 3 array.
		"""
		# Transform the position and directions of the hits temporarily in the frame of 
		# the geometry for calculations
		hit = N.dot(N.linalg.inv(self._working_frame), N.vstack((hits.T, N.ones(hits.shape[0]))))
		dir_loc = N.dot(self._working_frame[:3,:3].T, directs.T)
		# Partial derivation of the 'hit' equations <=> normal directions	   
		partial_x = 2.*hit[0]
		partial_y = 2.*hit[1]
		partial_z = -2.*(hit[2] - self.a)*self.c**2.
		# Build local unit normal vector
		local_normal = N.vstack((partial_x, partial_y, partial_z))
		local_unit = local_normal/N.sqrt(N.sum(local_normal**2., axis=0))
		# Identify the orientation of the normal considering the incident orientation of the 
		# ray. Treat the specific case of the apex setting the normal to be
		# -1*dir_loc at that point.
		down = N.sum(dir_loc * local_unit, axis=0) > 1e-9
		apex = (hit[2] == self.a)
		local_unit[:,down] *= -1.
		local_unit[:,apex] = N.vstack((0.,0.,-1.))
		normals = N.dot(self._working_frame[:3,:3], local_unit)
		
		return normals  
	
	def get_ABC(self, ray_bundle):
		"""
		Determines the variables forming the relevant quadric equation, [1]
		"""
		# Transform the the direction and position of the rays temporarily into the
		# frame of the paraboloid for calculations
		d = N.dot(self._working_frame[:3,:3].T, ray_bundle.get_directions())
		v = N.dot(N.linalg.inv(self._working_frame), N.vstack((ray_bundle.get_vertices(), N.ones(d.shape[1]))))[:3]

		A = d[0]**2. + d[1]**2. - (self.c*d[2])**2.
		B = 2.*(v[0]*d[0] + v[1]*d[1] - self.c**2.*(v[2] - self.a)*d[2])
		C = v[0]**2. + v[1]**2. - (self.c*(v[2] - self.a))**2.

		return A, B, C

class FiniteCone(InfiniteCone):
	"""
	Implements a finite cone. Parameters are r (base radius) and h (cone
	height). The cone is aligned with the (positive) z axis, and the apex of the
	cone is at the origin.

	"""
	def __init__(self, r, h):
		if h <= 0. or r <= 0.:
			raise AttributeError	  
		self.h = float(h)
		self.r = float(r)
		c = self.r/self.h		
		InfiniteCone.__init__(self, c=c)
	
	def _select_coords(self, coords, prm):
		"""
		Refinement of QuadricGM._select_coords; we want to choose the correct
		intersection point for a set of rays and our *truncated* quadric cone
		surface.

		Arguments:
		coords - a 2 by 3 by n array whose each column is the global coordinates
			of one intersection point of a ray with the surface geometry.
		prm - a 2 by n array giving the parametric location on the
			ray where the intersection occurs.

		Returns:
		The index of the selected intersection, or None if neither will do.
		"""
		select = N.empty(prm.shape[1])
		select.fill(N.nan)

		# Project the hit coordinates on the cone generatrix:
		height = N.sum(N.linalg.inv(self._working_frame)[None,2,:,None] * N.concatenate((coords, N.ones((2,1,coords.shape[-1]))), axis=1), axis=1)
		
		# Check the valid hitting points:
		inside = (height >= 0) & (height <= self.h)
		positive = prm > 1e-9 # to account for flating point precision issues.

		# Choses between the two intersections offered by the surface.
		hitting = inside & positive
		select[N.logical_and(*hitting)] = 1
		one_hitting = N.logical_xor(*hitting)
		select[one_hitting] = N.nonzero(hitting.T[one_hitting,:])[1]

		return select
	
	def mesh(self, resolution=None):
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
			resolution = 40
		# Generate a circular-edge mesh using polar coordinates.
		rc = self.c * (self.h - self.a)
		rmin = rc/resolution
		rs = N.r_[rmin,rc]

		angres = 2*N.pi / resolution 

		# Make the circumferential points at the requested resolution.
		ang_end = 2*N.pi
		angs = N.r_[0:ang_end+angres:angres]

		x = N.outer(N.hstack([0, rs]),N.cos(angs))
		y = N.outer(N.hstack([0, rs]),N.sin(angs))
		z = self.a + 1/self.c * N.sqrt(x**2 + y**2)

		return x, y, z	  

class RectCutCone(FiniteCone):

	def __init__(self, r, h, wf, hf):
		FiniteCone.__init__(self, r, h)
		self.half_dims = N.array([wf/2., hf/2.])
		
	def select_coords(self, coords):
		select = N.empty(prm.shape[1])
		select.fill(N.nan)

		# local coordinates:
		proj = N.linalg.inv(self._working_frame)
		coords = N.concatenate((coords, N.ones((2,1,coords.shape[-1]))), axis=1)
		local = []
		for i in range(coords.shape[0]):
			local.append(N.dot(proj[:-1,:], coords[i]))
		local = N.array(local)
		x, y, height = local[:,0], local[:,1], local[:,2]

		# Check the valid hitting points:
		inside_height = (height >= 0) & (height <= self.h)
		absx, absy = N.abs(x), N.abs(y)
		inside_w = absx <= self.half_dims[0]
		inside_h = absy <= self.half_dims[1]
   
		inside = inside_height & inside_w & inside_h
		positive = prm > 1e-8 # to account for floating point precision issues.

		# Choses between the two intersections offered by the surface.
		hitting = inside & positive
		select[N.logical_and(*hitting)] = 1
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
		top_right_corner_angle = N.round(N.arctan2(*self.half_dims[::-1]), decimals=8)
		rmax = self.r

		phis = N.array([top_right_corner_angle, N.pi-top_right_corner_angle,  N.pi+top_right_corner_angle, 2.*N.pi-top_right_corner_angle, 2.*N.pi+top_right_corner_angle])
		# Find the azimuth angle of the intersections of the external and internal perimeters with the rectangular frame.

		# External:
		rmax_beyond = rmax>=self.half_dims
		crossing_angle = []
		if rmax_beyond.any():
			if rmax_beyond[0]:
				crossing_angle.append(N.arccos(self.half_dims[0]/rmax))
			if rmax_beyond[1]: 
				crossing_angle.append(N.arcsin(self.half_dims[1]/rmax))
			crossing_angle = N.round(crossing_angle, decimals=9)
			crossing_angles_out = N.hstack([crossing_angle, N.pi-crossing_angle, N.pi+crossing_angle, 2.*N.pi-crossing_angle, 2.*N.pi+crossing_angle])
			phis = N.hstack([phis, crossing_angles_out])
			
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
		angs = N.unique(N.round(N.hstack(angs), decimals=9))

		# Intersect the frame with the external radius and check which distance is smallest.
		# If rmin is equal to that minimum, it is the intersection point: the mesh should be broken and a new starting angle found
		rs = N.zeros((len(angs), resolution+1))
		meshes = []
		for i, a in enumerate(angs):
			rx = N.abs(N.round(self.half_dims[0]/N.cos(a), decimals=6))
			ry = N.abs(N.round(self.half_dims[1]/N.sin(a), decimals=6))
			r = N.amin([rx, ry, rmax])
			rs[i] = N.linspace(0., r, resolution+1)
				
		# continuous mesh, well behaved shape
		x = rs.T*N.cos(angs)
		y = rs.T*N.sin(angs)
		z = self.a + 1./self.c * N.sqrt(x**2 + y**2)
		meshes.append(x)
		meshes.append(y)
		meshes.append(z)

		return meshes


class ConicalFrustum(InfiniteCone):
	"""
	Implements a conical frustum from (z1,r1) to (z2,r2), along the z-axis.
	z1 must not equal z2; r1 must not equal r2; r1 and r2 must be positive.
	"""
	def __init__(self, z1, r1, z2, r2):
		r1 = float(r1)
		r2 = float(r2)
		z1 = float(z1)
		z2 = float(z2)
		if r1 <= 0. or r2 <= 0.:
			raise AttributeError
		if r1 == r2 or z1 == z2:
			raise AttributeError
		#if z1 > z2:
		#	raise AttributeError

		c = float((r2 - r1)/(z2 - z1))
		a = float((r2*z1 - r1*z2) / (r2 - r1))
		InfiniteCone.__init__(self, c=c, a=a)

		self.r1 = float(r1)
		self.r2 = float(r2)
		self.z1 = float(z1)
		self.z2 = float(z2)
		self.zmin, self.zmax = N.sort([z1, z2])
	
	def _select_coords(self, coords, prm):
		"""
		Refinement of QuadricGM._select_coords; we want to choose the correct
		intersection point for a set of rays and our *sliced* quadric cone
		surface.

		Arguments:
		coords - a 2 by 3 by n array whose each column is the global coordinates
			of one intersection point of a ray with the surface.
		prm - a 2 by n array giving the parametric location on the
			ray where the intersection occurs.

		Returns:
		The index of the selected intersection, or None if neither will do.
		"""
		select = N.empty(prm.shape[1])
		select.fill(N.nan)

		# Projects the hit coordinates in a local frame on the z axis.
		height = N.sum(N.linalg.inv(self._working_frame)[None,2,:,None] * N.concatenate((coords, N.ones((2,1,coords.shape[-1]))), axis=1), axis=1)

		# Checks if the local_z-projected hit coords are in the actual height of the furstum
		# and if the parameter is positive so that the ray goes ahead.
		inside = (self.zmin <= height) & (height <= self.zmax)
		positive = prm > 1e-6
		hitting = inside & positive

		# Choses between the two intersections offered by the surface.
		select[N.logical_and(*hitting)] = True
		one_hitting = N.logical_xor(*hitting)
		select[one_hitting] = N.nonzero(hitting.T[one_hitting,:])[1]

		return select

	def mesh(self, resolution=None):
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
		# Generate a circular-edge mesh using polar coordinates.	
		r1 = self.r1
		r2 = self.r2
		rs = N.r_[min(r1,r2),max(r1,r2)]

		if resolution is None:
			angres = 2.*N.pi / 40
		else:
			angres = 2.*N.pi / resolution
			
		# Make the circumferential points at the requested resolution.
		ang_end = 2.*N.pi
		angs = N.r_[0.:ang_end+angres:angres]

		x = N.outer(rs, N.cos(angs))
		y = N.outer(rs, N.sin(angs))
		z = self.a + 1./self.c * N.sqrt(x**2 + y**2)
		
		return x, y, z

class RectCutConicalFrustum(ConicalFrustum):

	def __init__(self, z1, r1, z2, r2, w, h):
		ConicalFrustum.__init__(self, z1, r1, z2, r2)
		self.half_dims = N.array([w/2., h/2.])
		self.zmin, self.zmax = N.sort([z1, z2])
		if (N.sqrt(N.sum(self.half_dims**2))<=N.amin([r1, r2])).all():
			logging.error('Bad rectangular cut frustum shape, width and height too small')
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
		inside_height = (self.zmin <= height) & (height <= self.zmax)
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

		# frame corners:
		top_right_corner_angle = N.round(N.arctan2(*self.half_dims[::-1]), decimals=6)
		rmin, rmax = N.amin([self.r1, self.r2]), N.amax([self.r1, self.r2])

		phis = N.array([top_right_corner_angle, N.pi-top_right_corner_angle,  N.pi+top_right_corner_angle, 2.*N.pi-top_right_corner_angle, 2.*N.pi+top_right_corner_angle])
		# Find the azimuth angle of the intersections of the external and internal radii of the frustum with the rectangular frame.
		# Internal:
		rmin_beyond = rmin>=self.half_dims
		crossing_angle = []
		if rmin_beyond.any():
			if rmin_beyond[0]:
				crossing_angle.append(N.arccos(self.half_dims[0]/rmin))
			if rmin_beyond[1]: 
				crossing_angle.append(N.arcsin(self.half_dims[1]/rmin))
			crossing_angle = N.round(crossing_angle, decimals=8)
			crossing_angles_in = N.hstack([crossing_angle, N.pi-crossing_angle, N.pi+crossing_angle, 2.*N.pi-crossing_angle, 2.*N.pi+crossing_angle])		
			phis = N.hstack([phis, crossing_angles_in])

		# External:
		rmax_beyond = rmax>=self.half_dims
		crossing_angle = []
		if rmax_beyond.any():
			if rmax_beyond[0]:
				crossing_angle.append(N.arccos(self.half_dims[0]/rmax))
			if rmax_beyond[1]: 
				crossing_angle.append(N.arcsin(self.half_dims[1]/rmax))
			crossing_angle = N.round(crossing_angle, decimals=8)
			crossing_angles_out = N.hstack([crossing_angle, N.pi-crossing_angle, N.pi+crossing_angle, 2.*N.pi-crossing_angle, 2.*N.pi+crossing_angle])
			phis = N.hstack([phis, crossing_angles_out])
			
		# Sort angles
		phis = N.sort(phis)

		# make azimuth angles
		angs = []
		for i, p in enumerate(phis[:-1]):
			angresloc = N.ceil((phis[i+1]-p)/(2.*N.pi)*(resolution+1))
			if angresloc < 2:
				angresloc = 2
			angs.append(N.linspace(p, phis[i+1], int(angresloc))[:-1])
		angs.append(phis[-1])
		angs = N.unique(N.round(N.hstack(angs), decimals=8))

		# Intersect the frame with the external radius and check which distance is smallest, then check if the rmin is larger. 
		# If rmin is equal to that minimum, it is the intersection point: the mesh should be broken and a new starting angle found
		rs = N.zeros((len(angs), resolution+1))
		meshes = []
		startang = 0
		current_mesh = []
		for i, a in enumerate(angs):
			rx = N.abs(N.round(self.half_dims[0]/N.cos(a), decimals=7))
			ry = N.abs(N.round(self.half_dims[1]/N.sin(a), decimals=7))
			r = N.amin([rx, ry, rmax])

			if r==rmin:
				current_mesh.append(i)
				# Make and finish this mesh as we have a node where the rectangular border crosses.
				rs[i] = N.ones(resolution+1)*rmin
				if len(current_mesh)>1:
					x = rs[current_mesh].T*N.cos(angs[current_mesh])
					y = rs[current_mesh].T*N.sin(angs[current_mesh])
					z = self.a + 1./self.c * N.sqrt(x**2 + y**2)
					meshes.append(x)
					meshes.append(y)
					meshes.append(z)
					current_mesh = [] 
			if r>rmin:
				current_mesh.append(i)
				rs[i] = N.linspace(rmin, r, resolution+1)
				
			if a == angs[-1]:
				x = rs[current_mesh].T*N.cos(angs[current_mesh])
				y = rs[current_mesh].T*N.sin(angs[current_mesh])
				z = self.a + 1./self.c * N.sqrt(x**2 + y**2)
				meshes.append(x)
				meshes.append(y)
				meshes.append(z)

		# continuous mesh, well behaved shape
		if len(meshes) == 0: # there was no mesh break therefore and no mesh added
			x = rs.T*N.cos(angs)
			y = rs.T*N.sin(angs)
			z = self.a + 1./self.c * N.sqrt(x**2 + y**2)
			meshes.append(x)
			meshes.append(y)
			meshes.append(z)

		return meshes

	
	#use get_scene_graph from GeometryManager, since we've implemented mesh.



