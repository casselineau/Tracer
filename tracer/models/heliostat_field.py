"""
Manage a field of flat or parabolic heliostats aimed at a solar tower. The tower
is assumed to be at the origin, and the aiming is done by giving the sun's
azimuth and elevation.

The local coordinates system assumes that +x is East and +y is North.

References:
.. [1] http://www.flickr.com/photos/8242576@N06/2652388885

"""
import numpy as N

from ..assembly import Assembly
from .one_sided_mirror import rect_one_sided_mirror, rect_para_one_sided_mirror, flat_quad_one_sided_mirror
from ..spatial_geometry import rotx, roty, rotz, general_axis_rotation
from ..object import AssembledObject

class RotationAxis(AssembledObject):
	def __init__(self,  axis=None):
		self.axis = axis
		AssembledObject.__init__(self)

	def get_rotation_axis(self):
		return N.dot(self.get_rotation()[:3,:3], self.axis)

class HeliostatField(Assembly):
	def __init__(self, positions, width, height, absorptivity, sigma, bi_var=True, focal_lengths=None, quad_params=None, MCRT_option='fast', rotation_axes_pos=N.array([[0.,0.,0.],[0.,0.,0.]]), rotation_axes_vec=N.array([[0.,0.,1.],[1.,0.,0.]])):
		"""
		Generates a field of heliostats, each being a rectangular one-sided
		mirror, initially pointing downward - for safety reasons, of course :)
		
		Arguments:
		positions - an (n,3) array, each row has the location of one heliostat.
		width, height - The width
		and height, respectively, of each heliostat. apsorpt - part of incident energy absorbed by the heliostat.
		sigma - Heliostats surface slope error
		bi_var - If true, the slope error is a gaussian bi-variate on x and y, if false, it is an axi-symmetrical radial gaussian error.
		focal_lengths - the focal lengths of mirrors. If None, the mirrors are flat or quadric.
		quad_params - if not None, it is an array of quadric parameters for a RectFlatQuadricSurfaceGM instance: each line is [a, b, c, d, e] the coefficients of the flat quadratic
		surface.
		MCRT_option - if 'fast' does not stra any information of the helsostat hits and energy
		rotation_axes_pos - rotation axes reference positions with respect to heliostat nominal position (could be
		ground anchor point for example). Important: these positions are for now fixed. This works well for sound tracking axes design: ie. where the secondary axis is coplanar with the primary axis. If it is not the case, the secondary axis moves when the primariy axis acts, and one would theoretically need to obtain that new position to perform exact tracking as the center of teyh mirrro is not where it was when we calculated the tracking angles.
		rotation_axes_vec - rotation unit vectors in 3D coordinates
		facet_offset - offset of the facet from the reference position of secondary axis at 0 angle on both axes
		"""
		self._pos = positions
		#face_down = rotx(N.pi)[:3,:3]

		if focal_lengths==None:
			focal_lengths = [None]*positions.shape[0]
		if quad_params==None:
			quad_params = [None]*positions.shape[0]
		if not(hasattr(absorptivity, '__len__')):
			absorptivity = N.ones(positions.shape[0])*absorptivity
		
		self._heliostats = []
		self.rotation_axes_pos = rotation_axes_pos
		axes_offset = rotation_axes_pos[1]-rotation_axes_pos[0]
		for p in range(positions.shape[0]):
			primary_axis = RotationAxis(axis=rotation_axes_vec[0])
			secondary_axis = RotationAxis(axis=rotation_axes_vec[1])

			assert(not((focal_lengths[p] != None) and (quad_params[p] != None)))
			if (focal_lengths[p] == None) and (quad_params[p] == None):
				mirror = rect_one_sided_mirror(width, height, absorptivity[p], sigma, bi_var, MCRT_option)
			elif focal_lengths[p] != None: 
				mirror = rect_para_one_sided_mirror(width, height, focal_lengths[p], absorptivity[p], sigma, bi_var, MCRT_option)
			else:
				mirror = flat_quad_one_sided_mirror(width, height, quad_params[p], absorptivity[p], sigma, bi_var, MCRT_option)
			mirror.set_location(axes_offset)
			facet = Assembly(objects=[mirror, secondary_axis], location=rotation_axes_pos[0])
			hstat = Assembly(objects=[primary_axis], subassemblies=[facet], location=positions[p])

			self._heliostats.append(hstat)

		Assembly.__init__(self, subassemblies=self._heliostats)

	def get_heliostats(self):
		"""Access the list of one-sided mirrors representing the heliostats"""
		return self._heliostats
	
<<<<<<< Updated upstream
	def aim_to_sun(self, azimuth, zenith, aim_points=None, aim_vectors=None, tracking='azimuth_elevation', tracking_error=None, tracking_limits_primary_axis=None, tracking_limits_secondary_axis=None):
=======
	def set_aim_height(self, h):
		"""Change the verical position of the tower's target."""
		self._th = h
	
	def track_sun(self, azimuth, zenith, aim_points=None, aim_vectors=None, tracking='azimuth_elevation', tracking_error=None, tracking_limits_primary_axis=None, tracking_limits_secondary_axis=None):
>>>>>>> Stashed changes
		"""
		Aim the heliostats in a direction that brings the incident energy to
		the receiver location.
		
		Arguments:
		azimuth - the sun's azimuth, in radians from North, clockwise.
		zenith - angle created between the solar vector and the Z axis, 
			in radians.
		aim_points - if not None, defines an aim point for each heliostat
		aim_vectors - if not None, defines an aiming direction  for each heliostat
		tracking - 'azimuth_elevation'; 'titl_roll': tracking actuation method. 
		"""
		sun_vec = solar_vector(azimuth, zenith)
		
		if aim_points is None:
			if aim_vectors is None:
				raise('aim-points or aiming vectors have to be set')
			else:
				aim_vec = aim_vectors
			aim_vec /= N.sqrt(N.sum(aim_vec**2, axis=1)[:,None])
			trac = sun_vec + aim_vec
			trac /= N.sqrt(N.sum(trac**2, axis=1)[:,None])
		else:
			aim_points -= self._pos+N.sum(self.rotation_axes_pos, axis=0)
			aim_points /= N.sqrt(N.sum(aim_points**2, axis=1)[:,None])
			trac = sun_vec + aim_points
			trac /= N.sqrt(N.sum(trac**2, axis=1)[:,None])

		ang_err_1 = 0.
		ang_err_2 = 0.

		if tracking_limits_primary_axis == None:
			tracking_limits_primary_axis = [-N.pi, N.pi]
		if tracking_limits_secondary_axis == None:
			tracking_limits_secondary_axis = [-N.pi, N.pi]

		if tracking == 'azimuth_elevation':
			trac_az = N.arctan2(trac[:,1], trac[:,0])
			trac_ze = N.arccos(trac[:,2])
			for hidx in range(self._pos.shape[0]):
				if tracking_error != None:
					ang_err_1 = N.random.normal(scale=tracking_error)
					ang_err_2 = N.random.normal(scale=tracking_error)
				ang_az = trac_az[hidx]+ang_err_1
				ang_ze = trac_ze[hidx]+ang_err_2
				if ang_az<-N.pi:
					ang_az += N.pi
				if ang_az>N.pi:
					ang_az -= N.pi
				if ang_az<tracking_limits_primary_axis[0] or ang_az>tracking_limits_primary_axis[1]:
					print(ang_az, 'is outside of tracking limits')
					continue		
				elif ang_ze<tracking_limits_secondary_axis[0] or ang_ze>tracking_limits_secondary_axis[1]:
					print(ang_ze, 'is outside of tracking limits')
					continue

				facet = self._heliostats[hidx].get_assemblies()[0]
				prim_axis = self._heliostats[hidx].get_local_objects()[0]
				# get primary axis
				az_axis = prim_axis.get_rotation_axis()
				# make rotation matrix
				prim_rot = general_axis_rotation(az_axis, N.pi/2.+ang_az)
				# rotate heliostat (including secondary axis) around primary axis
				facet.set_rotation(prim_rot)
				# get secondary axis
				mirror, sec_axis = facet.get_objects()
				el_axis = sec_axis.get_rotation_axis()
				# make rotation matrix
				sec_rot = general_axis_rotation(el_axis, ang_ze)
				# rotate facet around secondary axis
				mirror.set_rotation(sec_rot)


				'''az_rot = rotz(N.pi/2+ang_az)
				ze_rot = rotx(ang_ze)
				# rotate secondary axis
				trans = N.dot(az_rot, ze_rot)
				trans[:3,3] = self._pos[hidx]
				
				self._heliostats[hidx].set_transform(trans)'''

		elif tracking == 'tilt_roll':
			hstat_tilt = N.arctan2(hstat[:,1],hstat[:,2])
			hstat_roll = N.arcsin(hstat[:,0])
			for hidx in range(self._pos.shape[0]):
				if tracking_error != None:
					ang_err_1 = N.random.normal(scale=tracking_error)
					ang_err_2 = N.random.normal(scale=tracking_error)
				ang_tilt = hstat_tilt[hidx]+ang_err_1
				ang_roll = hstat_roll[hidx]+ang_err_2

				if ang_tilt<tracking_limits_primary_axis[0] or ang_tilt>tracking_limits_primary_axis[1]:
					continue		
				if ang_roll<tracking_limits_secondary_axis[0] or ang_roll>tracking_limits_secondary_axis[1]:
					continue
				tilt_rot = rotx(-ang_tilt)
				roll_rot = roty(ang_roll)
				rot = N.dot(tilt_rot[:3,:3], roll_rot[:3,:3])
			
				self._heliostats[hidx].set_rotation(rot)

		Assembly.__init__(self,
					  subassemblies=self._heliostats)  # there is an issue between assemblies that forces us to re-initilise the assmebly after aiming. I am not sure what the problem is.
	def get_tracking_vectors(self):
		heliostats = self.get_heliostats()
		tracking_vectors = []
		for h in range(len(heliostats)):
			tracking_vectors.append(N.dot(heliostats[h].get_rotation(), N.vstack([0.,0.,1.])))
		return tracking_vectors

def solar_vector(azimuth, zenith):
	"""
	Calculate the solar vector using zenith and azimuth.
	
	Arguments:
	azimuth - the sun's azimuth, in radians, from North increasing towards to the East
	zenith - angle created between the solar vector and the Z axis, 
		in radians.
	
	Returns: a 3-component 1D array with the solar vector.
	"""
	azimuth = N.pi/2.-azimuth
	if azimuth<0.: azimuth += 2*N.pi
	sun_x = N.sin(zenith)*N.cos(azimuth)
	sun_y = N.sin(zenith)*N.sin(azimuth)
	sun_z = N.cos(zenith)

	sun_vec = N.r_[sun_x, sun_y, sun_z] 

	return sun_vec

def radial_stagger(start_ang, end_ang, az_space, rmin, rmax, r_space):
	"""
	Calculate positions of heliostats in a radial-stagger field. This is a
	common way to arrange heliostats.
	
	Arguments:
	start_ang, end_ang - the angle in radians CW from the X axis that define
		the field's boundaries.
	az_space - the azimuthal space between two heliostats, in [rad]
	rmin, rmax - the boundaries of the field in the radial direction.
	r_space - the space between radial lines of heliostats.
	
	Returns:
	An array with an x,y row for each heliostat (shape n,2)
	"""
	rs = N.r_[rmin:rmax:r_space]
	angs = N.r_[start_ang:end_ang:az_space/2]
	
	# 1st stagger:
	xs1 = N.outer(rs[::2], N.cos(angs[::2])).flatten()
	ys1 = N.outer(rs[::2], N.sin(angs[::2])).flatten()
	
	# 2nd staggeer:
	xs2 = N.outer(rs[1::2], N.cos(angs[1::2])).flatten()
	ys2 = N.outer(rs[1::2], N.sin(angs[1::2])).flatten()
	
	xs = N.r_[xs1, xs2]
	ys = N.r_[ys1, ys2]
	
	return N.vstack((xs, ys)).T
