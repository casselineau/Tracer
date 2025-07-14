# -*- coding: utf-8 -*-
# Implements the geometry of several types of flat surfaces.
# 
# Reference:
# [1]http://www.siggraph.org/education/materials/HyperGraph/raytrace/rayplane_intersection.htm

from numpy import linalg as LA
import numpy as N
from tracer.geometry_manager import GeometryManager

class FlatGeometryManager(GeometryManager):
	"""
	Implements the geometry of an infinite flat surface, an the XY plane of its
	local coordinates (so the local Z is the surface normal).
	"""
	def find_intersections(self, frame, ray_bundle):
		"""
		Register the working frame and ray bundle, calculate intersections
		and save the parametric locations of intersection on the surface.
		Algorithm taken from [1].
		
		Arguments:
		frame - the current frame, represented as a homogenous transformation
			matrix stored in a 4x4 array.
		ray_bundle - a RayBundle object with the incoming rays' data.
		
		Returns:
		A 1D array with the parametric position of intersection along each of
			the rays. Rays that missed the surface return +infinity.
		"""
		GeometryManager.find_intersections(self, frame, ray_bundle)
		
		d = ray_bundle.get_directions()
		v = ray_bundle.get_vertices() - frame[:3,3][:,None]
		n = ray_bundle.get_num_rays()
		
		# Vet out parallel rays:
		dt = N.dot(d.T, frame[:3,2])
		unparallel = N.abs(dt) > 1e-7
		
		# `params` holds the parametric location of intersections along the ray 
		params = N.empty(n)
		params.fill(N.inf)
		
		vt = N.dot(frame[:3,2], v[:,unparallel])
		params[unparallel] = -vt/dt[unparallel]
		
		# Takes into account a negative depth
		# Note that only the 3rd row of params is relevant here!
		negative = params < 1e-7
		params[negative] = N.inf
		
		if hasattr(self, '_params'):
			self._params = N.hstack([self._params, params])
		else:
			self._params = params
		if hasattr(self, '_backside'):
			self._backside = N.hstack([self._backside, dt > 0.])
		else:
			self._backside = dt > 0.

		return self._params
		
	def select_rays(self, idxs):
		"""
		Inform the geometry manager that only the given rays are to be used,
		so that internal data size is kept small.
		
		Arguments: 
		idxs - an index array stating which rays of the working bundle
			are active.
		"""
		self._idxs = idxs # For slicing ray bundles etc.
		self._backside = N.nonzero(self._backside[idxs])[0]
		
		v = self._working_bundle.get_vertices()[:,idxs]
		d = self._working_bundle.get_directions()[:,idxs]
		p = self._params[idxs]
		del self._params
		
		# Global coordinates on the surface:
		self._global = v + p[None,:]*d
	
	def get_normals(self):
		"""
		Report the normal to the surface at the hit point of selected rays in
		the working bundle.
		"""
		norms = N.tile(self._working_frame[:3,2].copy()[:,None], (1, len(self._idxs)))
		norms[:,self._backside] *= -1
		return norms
	
	def get_intersection_points_global(self):
		"""
		Get the ray/surface intersection points in the global coordinates.
		
		Returns:
		A 3-by-n array for 3 spatial coordinates and n rays selected.
		"""
		return self._global
	
	def done(self):
		"""
		Discard internal data structures. This should be called after all
		information on the latest bundle's results have been extracted already.
		"""
		GeometryManager.done(self)
		if hasattr(self, '_global'):
			del self._global
		if hasattr(self, '_idxs'):
			del self._idxs
		if hasattr(self, '_backside'):
			del self._backside

class FiniteFlatGM(FlatGeometryManager):
	"""
	Calculates intersection points before select_rays(), so that those outside
	the aperture can be dropped, and on select_rays trims it.
	"""
	def __init__(self):
		FlatGeometryManager.__init__(self)
	
	def find_intersections(self, frame, ray_bundle):
		"""
		Register the working frame and ray bundle, calculate intersections
		and save the parametric locations of intersection on the surface.
		Algorithm taken from [1].
		
		In this class, global- and local-coordinates of intersection points
		are calculated and kept. _global is handled in select_rays(), but
		_local must be taken care off by subclasses.
		
		Arguments:
		frame - the current frame, represented as a homogenous transformation
			matrix stored in a 4x4 array.
		ray_bundle - a RayBundle object with the incoming rays' data.
		
		Returns:
		A 1D array with the parametric position of intersection along each of
			the rays. Rays that missed the surface return +infinity.
		"""
		ray_prms = FlatGeometryManager.find_intersections(self, frame, ray_bundle)
		v = self._working_bundle.get_vertices() 
		d = self._working_bundle.get_directions()
		p = self._params

		del self._params
		
		# Global coordinates on the surface:
		glob = v + p[None,:]*d
		if hasattr(self, '_global'):
			self._global = N.concatenate([self._global, glob], axis=-1)
		else:
			self._global = glob

		# above we ignore invalid values. Those rays can't be selected anyway.
		# Local should be deleted by children in their find_intersections.
		
		if hasattr(self, '_local'):
			self._local = N.concatenate([self._local, N.dot(N.linalg.inv(self._working_frame),
			N.vstack((glob, N.ones(glob.shape[1]))))], axis=-1)
		else:	
			self._local = N.dot(N.linalg.inv(self._working_frame),
			N.vstack((glob, N.ones(glob.shape[1]))))

		return ray_prms
	
	def select_rays(self, idxs):
		"""
		Inform the geometry manager that only the given rays are to be used,
		so that internal data size is kept small.
		
		Arguments: 
		idxs - an index array stating which rays of the working bundle
			are active.
		"""
		self._idxs = idxs

		if hasattr(self, "_backside"):
			self._backside = N.nonzero(self._backside[idxs])[0]
		else:
			self._backside = N.array([], dtype=int)

		self._global = self._global[:,idxs].copy()

class RectPlateGM(FiniteFlatGM):
	"""
	Trims the infinite flat surface by marking rays whose intersection with
	the surface are outside the given width and height.
	"""
	def __init__(self, width, height):
		"""
		Arguments:
		width - the extent along the x axis in the local frame (sets self._w)
		height - the extent along the y axis in the local frame (sets self._h)
		"""
		if width <= 0:
			raise ValueError("Width must be positive")
		if height <= 0:
			raise ValueError("Height must be positive")
		
		self.width = width
		self.height = height
		self._half_dims = N.c_[[width, height]]/2.

		FiniteFlatGM.__init__(self)
		
	def find_intersections(self, frame, ray_bundle):
		"""
		Extends the parent flat geometry manager by discarding in advance
		impact points outside a centered rectangle.
		"""
		ray_prms = FiniteFlatGM.find_intersections(self, frame, ray_bundle)
		ray_prms[N.any(N.abs(self._local[:2]) > self._half_dims, axis=0)] = N.inf
		del self._local
		return ray_prms
	
	def mesh(self, resolution):
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
		#points = N.ceil(resolution*self._half_dims.reshape(-1)*2)
		#points[points < 2] = 2 # At least the edges of the range.
		xs = N.linspace(-self._half_dims[0,0], self._half_dims[0,0], resolution+1)
		ys = N.linspace(-self._half_dims[1,0], self._half_dims[1,0], resolution+1)
		
		x, y = N.broadcast_arrays(xs[:,None], ys)
		z = N.zeros_like(x)

		return x, y, z

	def get_fluxmap(self, eners, local_coords, resolution):
		'''
		Cartesian mesh for flat rectangular surfaces
		'''
		xs = N.linspace(-self._half_dims[0,0], self._half_dims[0,0], resolution+1)
		ys = N.linspace(-self._half_dims[1,0], self._half_dims[1,0], resolution+1)

		eners = N.histogram2d(local_coords[0], local_coords[1], bins=[xs, ys], weights=eners)[0]
		dxs = N.tile(N.abs(xs[1:]-xs[:-1]), (len(ys)-1,1))
		dys = N.tile(N.abs(ys[1:]-ys[:-1]), (len(xs)-1,1)).T
		areas = dxs*dys
		
		flux = eners/areas
		
		return N.hstack(flux)

class ExtrudedRectPlateGM(RectPlateGM):

	def __init__(self, width, height, extr_center, extr_width, extr_height):
		'''
		Extension of the rectangular plate GM to support a rectangular extrusion.
		'''
		RectPlateGM.__init__(self, width, height)
		self.extr_center = extr_center
		self.extr_half_dims = N.c_[[extr_width, extr_height]]/2.

		assert(((extr_center+self.extr_half_dims)<self._half_dims).all()) 

	def find_intersections(self, frame, ray_bundle):
		"""
		Extends the parent flat geometry manager by discarding in advance
		impact points inside the extrusion and outside the extent of teh plate.
		"""
		ray_prms = FiniteFlatGM.find_intersections(self, frame, ray_bundle)
		ray_prms[N.any(N.abs(self._local[:2]) > self._half_dims, axis=0)] = N.inf
		ray_prms[N.all(N.abs(self._local[:2]-self.extr_center) < self.extr_half_dims, axis=0)] = N.inf
		del self._local
		return ray_prms

	def mesh(self, resolution):
		"""
		Represent the surface as a mesh in local coordinates.
		
		Arguments:
		resolution - in points per unit length (so the number of points 
			returned is O(A*resolution**2) for area A)
		
		Returns:
		x, y, z - each a 2D array holding in its (i,j) cell the x, y, and z
			coordinate (respectively) of point (i,j) in the mesh.
		As the surface is picewise defined, the mesh returns a series of arrays.
		"""
		if resolution == None:
			resolution = 40

		# Bottom:
		xs_bot = N.linspace(-self._half_dims[0,0], self._half_dims[0,0], resolution+1)
		ys_bot = N.linspace(-self._half_dims[1,0], self.extr_center[1]-self.extr_half_dims[1,0], resolution+1)
		x_bot, y_bot = N.broadcast_arrays(xs_bot[:,None], ys_bot)
		z_bot = N.zeros_like(x_bot)

		# Middle left:
		xs_mid_left = N.linspace(-self._half_dims[0,0], self.extr_center[0]-self.extr_half_dims[0,0], resolution+1)
		ys_mid_left = N.linspace(self.extr_center[1]-self.extr_half_dims[1,0], self.extr_center[1]+self.extr_half_dims[1,0], resolution+1)

		x_mid_left, y_mid_left = N.broadcast_arrays(xs_mid_left[:,None], ys_mid_left)
		z_mid_left = N.zeros_like(x_mid_left)

		# Middle right:
		xs_mid_right = N.linspace(self.extr_center[0]+self.extr_half_dims[0,0], self._half_dims[0,0], resolution+1)
		ys_mid_right = N.linspace(self.extr_center[1]-self.extr_half_dims[1,0], self.extr_center[1]+self.extr_half_dims[1,0], resolution+1)

		x_mid_right, y_mid_right = N.broadcast_arrays(xs_mid_right[:,None], ys_mid_right)
		z_mid_right = N.zeros_like(x_mid_right)

		# Top:
		xs_top = N.linspace(-self._half_dims[0,0], self._half_dims[0,0], resolution+1)
		ys_top = N.linspace(self.extr_center[1]+self.extr_half_dims[1,0], self._half_dims[1,0], resolution+1)

		x_top, y_top = N.broadcast_arrays(xs_top[:,None], ys_top)
		z_top = N.zeros_like(x_top)

		return x_bot, y_bot, z_bot, x_mid_left, y_mid_left, z_mid_left, x_mid_right, y_mid_right, z_mid_right, x_top, y_top, z_top
	  
	def get_fluxmap(self,eners, local_coords, resolution):
		# Bottom:
		xs_bot = N.linspace(-self._half_dims[0,0], self._half_dims[0,0], resolution+1)
		ys_bot = N.linspace(-self._half_dims[1,0], self.extr_center[1]-self.extr_half_dims[1,0], resolution+1)
		eners_bot = N.histogram2d(local_coords[0], local_coords[1], bins=[xs_bot, ys_bot], weights=eners)[0]
		dxs_bot = N.tile(N.abs(xs_bot[1:]-xs_bot[:-1]), (len(ys_bot)-1,1))
		dys_bot = N.tile(N.abs(ys_bot[1:]-ys_bot[:-1]), (len(xs_bot)-1,1)).T
		areas_bot = dxs_bot*dys_bot
		flux_bot = eners_bot/areas_bot
		# Middle left:
		xs_mid_left = N.linspace(-self._half_dims[0,0], self.extr_center[0]-self.extr_half_dims[0,0], resolution+1)
		ys_mid_left = N.linspace(self.extr_center[1]-self.extr_half_dims[1,0], self.extr_center[1]+self.extr_half_dims[1,0], resolution+1)
		eners_mid_left = N.histogram2d(local_coords[0], local_coords[1], bins=[xs_mid_left, ys_mid_left], weights=eners)[0]
		dxs_mid_left = N.tile(N.abs(xs_mid_left[1:]-xs_mid_left[:-1]), (len(ys_mid_left)-1,1))
		dys_mid_left = N.tile(N.abs(ys_mid_left[1:]-ys_mid_left[:-1]), (len(xs_mid_left)-1,1)).T
		areas_mid_left = dxs_mid_left*dys_mid_left
		flux_mid_left = eners_mid_left/areas_mid_left
		# Middle right:
		xs_mid_right = N.linspace(self.extr_center[0]+self.extr_half_dims[0,0], self._half_dims[0,0], resolution+1)
		ys_mid_right = N.linspace(self.extr_center[1]-self.extr_half_dims[1,0], self.extr_center[1]+self.extr_half_dims[1,0], resolution+1)
		eners_mid_right = N.histogram2d(local_coords[0], local_coords[1], bins=[xs_mid_right, ys_mid_right], weights=eners)[0]
		dxs_mid_right = N.tile(N.abs(xs_mid_right[1:]-xs_mid_right[:-1]), (len(ys_mid_right)-1,1))
		dys_mid_right = N.tile(N.abs(ys_mid_right[1:]-ys_mid_right[:-1]), (len(xs_mid_right)-1,1)).T
		areas_mid_right = dxs_mid_right*dys_mid_right
		flux_mid_right = eners_mid_right/areas_mid_right
		# Top:
		xs_top = N.linspace(-self._half_dims[0,0], self._half_dims[0,0], resolution+1)
		ys_top = N.linspace(self.extr_center[1]+self.extr_half_dims[1,0], self._half_dims[1,0], resolution+1)
		eners_top = N.histogram2d(local_coords[0], local_coords[1], bins=[xs_top, ys_top], weights=eners)[0]
		dxs_top = N.tile(N.abs(xs_top[1:]-xs_top[:-1]), (len(ys_top)-1,1))
		dys_top = N.tile(N.abs(ys_top[1:]-ys_top[:-1]), (len(xs_top)-1,1)).T
		areas_top = dxs_top*dys_top
		flux_top = eners_top/areas_top

		return N.hstack(flux_bot), N.hstack(flux_mid_left), N.hstack(flux_mid_right), N.hstack(flux_top)

class RoundPlateGM(FiniteFlatGM):
	"""
	Trims the infinite flat surface by marking as missing the rays falling
	outside the given external radius or inside the internal radius.
	"""
	def __init__(self, Re, Ri = None):
		"""
		Arguments:
		Re - the plate's external radius
		Ri - the plate's internal radius
		"""
		if Re <= 0.:
			raise ValueError("Radius must be positive")
		if Ri != None:
			if Ri >= Re:
				logging.error('Inner Radius must be lower than the outer one. Ri: {} Re: {}'.format(Ri, Re))
				raise ValueError("Inner Radius must be lower than the outer one")
			if Ri <= 0.:
				logging.error('Radius must be positive. Ri: {} Re: {}'.format(Ri, Re))
				raise ValueError("Radius must be positive")
		
		self._Ri = Ri	   
		self._Re = Re
		FiniteFlatGM.__init__(self)
	
	def find_intersections(self, frame, ray_bundle):
		"""
		Extends the parent flat geometry manager by discarding in advance
		impact points outside a centered circle.
		"""
		ray_prms = FiniteFlatGM.find_intersections(self, frame, ray_bundle)
		ray_prms[N.sum(self._local[:2]**2., axis=0) > self._Re**2.] = N.inf
		if self._Ri != None:
			ray_prms[N.sum(self._local[:2]**2., axis=0) < self._Ri**2.] = N.inf
		del self._local
		return ray_prms
	
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
		# Make the circumferential points at the requested resolution.
		if resolution == None:
			resolution = 40
		angs = N.r_[0.:2.*N.pi+2.*N.pi/resolution:2.*N.pi/resolution]
		if self._Ri != None:
			rs = self._Ri+(self._Re-self._Ri)/resolution*N.arange(0, resolution+1)
		else:
			rs = self._Re/resolution*N.arange(0, resolution+1)
   
		x = N.outer(rs, N.cos(angs))
		y = N.outer(rs, N.sin(angs))
		z = N.zeros_like(x)

		return x, y, z

	def get_fluxmap(self, eners, local_coords, resolution):
		if resolution == None:
			resolution = 40
		local_rads = N.sqrt(N.sum(local_coords[:2]**2, axis=0))
		local_angs = N.arctan2(local_coords[0], local_coords[1])
		local_angs[local_angs<0.] = local_angs[local_angs<0.]+2.*N.pi

		angs = N.r_[0.:2.*N.pi+2.*N.pi/resolution:2.*N.pi/resolution]
		if self._Ri != None:
			rs = self._Ri+(self._Re-self._Ri)/resolution*N.arange(0, resolution+1)
		else:
			rs = self._Re/resolution*N.arange(0, resolution+1)

		eners = N.histogram2d(local_rads, local_angs, bins=[rs, angs], weights=eners)[0]
		drs = N.tile(N.abs(rs[1:]-rs[:-1]), (len(angs)-1,1)).T
		ravgs = N.tile((rs[1:]+rs[:-1])/2., (len(angs)-1,1)).T
		dangs = N.tile(N.abs(angs[1:]-angs[:-1]), (len(rs)-1,1))
		areas = drs*ravgs*dangs
		
		flux = eners/areas

		return N.hstack(flux)


class StraightCutRoundPlateGM(RoundPlateGM):
	def __init__(self, Re, x_cut=None):
		RoundPlateGM.__init__(self, Re)
		self.x_cut = x_cut

	def find_intersections(self, frame, ray_bundle):
		ray_prms = FiniteFlatGM.find_intersections(self, frame, ray_bundle)
		ray_prms[N.sum(self._local[:2]**2., axis=0) > self._Re**2.] = N.inf
		if self._Ri != None:
			ray_prms[N.sum(self._local[:2]**2., axis=0) < self._Ri**2.] = N.inf
		ray_prms[self._local[0]>self.x_cut] = N.inf
		del self._local
		return ray_prms

	def mesh(self, resolution):
		if resolution == None:
			resolution = 30
		else:
			resolution = int(N.ceil((resolution)/3.)*3)

		dangcut = N.arccos(self.x_cut/self._Re)
		if dangcut<(N.pi/2.):

			angcut1 = N.arange(0., dangcut, dangcut/(resolution/3))
			angdisk = N.arange(dangcut, 2.*N.pi-dangcut, 2.*(N.pi-dangcut)/(resolution/3))
			angcut2 = N.linspace(2.*N.pi-dangcut, 2.*N.pi, (resolution/3)+1)

			rs = self._Re/resolution*N.arange(0, resolution+1)

			xs = N.linspace(0., self.x_cut, resolution+1)

			xcut1 = N.tile(xs, (len(angcut1),1)).T
			xcut2 = N.tile(xs, (len(angcut2),1)).T
			ycut1 = N.vstack(xs)*N.tan(angcut1)
			ycut2 = N.vstack(xs)*N.tan(angcut2)
			zcut1 = N.zeros_like(xcut1)
			zcut2 = N.zeros_like(xcut2)

			xdisk = N.outer(rs, N.cos(angdisk))
			ydisk = N.outer(rs, N.sin(angdisk))
			zdisk = N.zeros_like(xdisk)
		
			x = N.concatenate((xcut1, xdisk, xcut2), axis=1)
			y = N.concatenate((ycut1, ydisk, ycut2), axis=1)
			z = N.concatenate((zcut1, zdisk, zcut2), axis=1)

		else:
			angs = N.linspace(dangcut, 2.*N.pi-dangcut,resolution+1)
			x = N.zeros((resolution+1,resolution+1))
			y = N.zeros_like(x)
			z = N.zeros_like(x)
			for i in xrange(x.shape[0]):
				rs = N.linspace(self.x_cut/N.cos(angs[i]), self._Re, resolution+1)
				x[i] = rs*N.cos(angs[i])
				y[i] = rs*N.sin(angs[i])
		return x, y, z

	def get_fluxmap(self, eners, local_coords, resolution):

		if resolution == None:
			resolution = 30
		else:
			resolution = int(N.ceil(resolution/3.)*3)

		flux = N.zeros(resolution**2)

		if len(eners) == 0:
			return flux

		local_rads = N.sqrt(N.sum(local_coords[:2]**2, axis=0))
		local_angs = N.arctan2(local_coords[1], local_coords[0])
		local_angs[local_angs<0.] = local_angs[local_angs<0.]+2.*N.pi

		dangcut = N.arccos(self.x_cut/self._Re)

		if dangcut<(N.pi/2.):

			angcut1 = N.arange(0., dangcut, dangcut/(resolution/3))
			angdisk = N.arange(dangcut, 2.*N.pi-dangcut, 2.*(N.pi-dangcut)/(resolution/3))
			angcut2 = N.linspace(2.*N.pi-dangcut, 2.*N.pi, (resolution/3)+1)

			angs = N.hstack((angcut1, angdisk, angcut2))

			rs = N.linspace(0., self._Re, resolution+1)
	
			xs = N.linspace(0., self.x_cut, resolution+1)

			# Treat differently the disk region and the cut region	
			# Disk region: bin according to radii and angle
			disk = angs[resolution/3:2*resolution/3+1]
			enersdisk = N.histogram2d(local_rads, local_angs, bins=[rs, disk], weights=eners)[0]
			drs = N.tile(rs[1:]-rs[:-1], (len(disk)-1,1)).T
			ravgs = N.tile((rs[1:]+rs[:-1])/2., (len(disk)-1,1)).T
			dangs = N.tile(N.abs(disk[1:]-disk[:-1]), (len(rs)-1,1))
			areas = drs*ravgs*dangs
			fluxdisk = N.hstack(enersdisk/areas)
		
			# Cut region: bin according to x coord and angle
			cut1 = angs<=dangcut
			enerscut1 = N.histogram2d(local_coords[0], local_angs, bins=[xs, angs[cut1]], weights=eners)[0]
			dxs = N.tile(xs[1:]-xs[:-1], (len(angs[cut1])-1,1))
			dys = (xs[:-1]*N.vstack(N.tan(angs[cut1][:-1]))+xs[1:]*N.vstack(N.tan(angs[cut1][1:])/2.))
			areas = N.abs(dxs*dys)
			fluxcut1 = N.hstack(enerscut1/areas.T)
		
			cut2 = angs>=(2.*N.pi-dangcut)
			enerscut2 = N.histogram2d(local_coords[0], local_angs, bins=[xs, angs[cut2]], weights=eners)[0]
			dxs = N.tile(xs[1:]-xs[:-1], (len(angs[cut2])-1,1))
			dys = (xs[:-1]*N.vstack(N.tan(angs[cut2][:-1]))+xs[1:]*N.vstack(N.tan(angs[cut2][1:])/2.))

			areas = N.abs(dxs*dys)
			fluxcut2 = N.hstack(enerscut2/areas.T)

			for i in xrange(len(flux)/3):
				idx = resolution/3
				flux[resolution*i:resolution*i+idx] = fluxcut1[idx*i:idx*(i+1)]
				flux[resolution*i+idx:resolution*i+2*idx] = fluxdisk[idx*i:idx*(i+1)]
				flux[resolution*i+2*idx:resolution*i+3*idx] = fluxcut2[idx*i:idx*(i+1)]

		else:
			flux = N.zeros(resolution**2)
			angs = N.linspace(dangcut, 2.*N.pi-dangcut, resolution+1)
			x,y,z = self.mesh(resolution)

			xA = x[:-1,:-1]
			xB = x[:-1,1:]
			xC = x[1:,1:]
			xD = x[1:,:-1]
			yA = y[:-1,:-1]
			yB = y[:-1,1:]
			yC = y[1:,1:]
			yD = y[1:,:-1]
			a = N.sqrt((xB-xA)**2+(yB-yA)**2)
			b = N.sqrt((xC-xB)**2+(yC-yB)**2)
			c = N.sqrt((xD-xC)**2+(yD-yC)**2)
			d = N.sqrt((xA-xD)**2+(yA-yD)**2)

			p = N.sqrt((xC-xA)**2+(yC-yA)**2)
			q = N.sqrt((xD-xB)**2+(yD-yB)**2)

			# Quadrilateral area:
			areas = 0.25*N.sqrt(4.*p**2*q**2-(b**2+d**2-a**2-c**2)**2)

			# Add the disk cap that is unaccounted for to the last element
			areas[:,-1] += (angs[1:]-angs[:-1])/2.*self._Re**2-b[:,-1]/2.*self._Re*N.cos(N.arcsin(b[:,-1]/(2.*self._Re)))

			# Binning:
			for i in xrange(int(resolution)):
				# Separations lines equation coefficients:
				a_seps = N.tile((y[i+1]-y[i])/(x[i+1]-x[i]), (local_coords.shape[1],1))
				b_seps = y[i]-a_seps*x[i]
				# Equation of the line from the origin goping through the hit coordinate:
				local_a = local_coords[1]/local_coords[0]
				# Intersection with the "radial" separations:
				local_inters_x = b_seps/(N.vstack(local_a)-a_seps)
				local_inters_x[N.isnan(local_inters_x)] = self.x_cut
				local_inters_y = N.vstack(local_a)*local_inters_x
				inter_rads = N.vstack(N.sqrt(local_inters_x**2+local_inters_y**2))
								
				in_wedge = N.logical_and((local_angs>=angs[i]), (local_angs<angs[i+1]))

				if in_wedge.any():
					inter_rads[:,-1] = self._Re # to grab the hits that are beyond the last separation but before the end of the disk.
					in_bins = N.logical_and((N.vstack(local_rads)>=inter_rads[:,:-1]),(N.vstack(local_rads)<inter_rads[:,1:]))

					#flux[i*resolution:(i+1)*resolution] = N.sum(N.vstack(eners)*in_bins, axis=0)/areas[i]
					flux[i:resolution**2:resolution] = N.sum(N.vstack(eners)*in_bins, axis=0)/areas[i]
		return flux
