import numpy as N
from .quadric import QuadricGM

class FlatQuadricSurfaceGM(QuadricGM):
	"""Implements the geometry of an infinite "flat" quadratic surface. "Flat" refers to the fact that this surface will have only one solution in the local z plane."""
	def __init__(self, a=1., b=1., c=1., d=0., e=0., f=0.):
		"""			   
		Arguments: 
		a, b, c, d, e, f: z = ax**2 + by**2 + cxy + dx + ey + f
		
		""" 
		QuadricGM.__init__(self)
		self.a = a
		self.b = b
		self.c = c
		self.d = d
		self.e = e
		self.f = f

	def _normals(self, hits, directs):
		"""
		Finds the normal to the parabola in a bunch of intersection points, by
		taking the derivative and rotating it. Used internally by quadric.
		
		Arguments:
		hits - the coordinates of intersections, as an n by 3 array.
		directs - directions of the corresponding rays, n by 3 array.
		"""
		hit = N.dot(N.linalg.inv(self._working_frame), N.vstack((hits.T, N.ones(hits.shape[0]))))
		dir_loc = N.dot(self._working_frame[:3,:3].T, directs.T)

		partial_x = 2.*hit[0]*self.a+self.c*hit[1]+self.d
		partial_y = 2.*hit[1]*self.b+self.c*hit[0]+self.e
		partial_z = -1*N.ones(N.shape(hits)[0])
		
		local_normal = N.vstack((partial_x, partial_y, partial_z))
		local_unit = local_normal/N.sqrt(N.sum(local_normal**2, axis=0))

		down = N.sum(dir_loc * local_unit, axis=0) > 0.
		local_unit[:,down] *= -1

		normals = N.dot(self._working_frame[:3,:3], local_unit)

		return normals
 
	
	def get_ABC(self, ray_bundle):
		"""
		Determines the variables forming the relevant quadric equation, [1]
		"""
		# Transform the the direction and position of the rays temporarily into the
		# frame of the paraboloid for calculations
		d = N.dot(self._working_frame[:3,:3].T, ray_bundle.get_directions())
		v = N.dot(N.linalg.inv(self._working_frame), 
			N.vstack((ray_bundle.get_vertices(), N.ones(d.shape[1]))))[:3]

		A = self.a*d[0]**2. + self.b*d[1]**2. + self.c*d[0]*d[1]
		B = 2.*self.a*d[0]*v[0] + 2.*self.b*d[1]*v[1] + self.c*(v[0]*d[1]+v[1]*d[0]) + self.d*d[0] + self.e*d[1] - d[2]
		C = self.a*v[0]**2 + self.b*v[1]**2 + self.c*v[0]*v[1] + self.d*v[0] + self.e*v[1] + self.f - v[2]
		
		return A, B, C


class RectFlatQuadricSurfaceGM(FlatQuadricSurfaceGM):
	"""
	A quadratic surface with rectangular frame.
	"""
	def __init__(self, width, height, a=1., b=1., c=1., d=1., e=1., f=1.):
		FlatQuadricSurfaceGM.__init__(self, a, b, c, d, e, f)
		self._half_dims = N.c_[[width, height]]/2.
		self._w, self._h = width/2., height/2.

	def _select_coords(self, coords, prm):
		"""
		Choose between two intersection points on a quadric surface.
		This implementation extends QuadricGM's behaviour by not choosing
		intersections outside the rectangular aperture.
		
		Arguments:
		coords - a 2 by 3 by n array whose each column is the global coordinates
			of one intersection point of a ray with the sphere.
		prm - the corresponding parametric location on the ray where the
			intersection occurs.

		Returns:
		The index of the selected intersection, or None if neither will do.
		"""
		select = QuadricGM._select_coords(self, coords, prm)

		coords = N.concatenate((coords, N.ones((2,1,coords.shape[2]))), axis = 1)
		# assumed no additional parameters to coords, axis = 1
		local = N.sum(N.linalg.inv(self._working_frame)[None,:2,:,None] * \
			coords[:,None,:,:], axis=2)

		abs_x = abs(local[:,0,:])
		abs_y = abs(local[:,1,:])
		outside = abs_x > self._w
		outside |= abs_y > self._h
		inside = (~outside) & (prm > 1e-6)

		select[~N.logical_or(*inside)] = N.nan
		one_hit = N.logical_xor(*inside)
		select[one_hit] = N.nonzero(inside.T[one_hit,:])[1]

		return select

	def mesh(self, resolution=None):
		"""
		Represent the surface as a mesh in local coordinates.
		
		Arguments:
		resolution - in points per unit length (so the number of points 
			returned is O(A*resolution**2) for area A)
		
		Returns:
		x, y, z - each a 2D array holding in its (i,j) cell the x, y, and z
			coordinate (respectively) of point (i,j) in the mesh.
		"""
		if resolution == None:
			resolution = 40
		points = N.ceil(resolution*self._half_dims.reshape(-1)*2)
		points[points < 2] = 2 # At least the edges of the range.
		xs = N.linspace(-self._half_dims[0,0], self._half_dims[0,0], points[0])
		ys = N.linspace(-self._half_dims[1,0], self._half_dims[1,0], points[1])
		
		x, y = N.broadcast_arrays(xs[:,None], ys)
		z = self.a*x**2 + self.b*y**2 + self.c*x*y + self.d*x +self.e*y + self.f
		#print(">>> heliostat", x, y, z)
		return x, y, z
