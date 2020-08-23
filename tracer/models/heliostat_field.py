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

class HeliostatField(Assembly):
	def __init__(self, positions, width, height, absorptivity, aim_height, sigma, bi_var=True, focal_lengths=None, quad_params=None, MCRT_option='fast'):
		"""
		Generates a field of heliostats, each being a rectangular one-sided
		mirror, initially pointing downward - for safety reasons, of course :)
		
		Arguments:
		positions - an (n,3) array, each row has the location of one heliostat.
		width, height - The width and height, respectively, of each
			heliostat.
		apsorpt - part of incident energy absorbed by the heliostat.
		aim_height - the height (Z coordinate) of the target for aiming
		sigma - Heliostats surface slope error
		bi_var - If true, the slope error is a gaussian bi-variate on x and y, if false, it is an axi-symmetrical radial gaussian error.
		focal_lengths - the focal lengths of mirrors. If None, the mirrors are flat or quadric.
		quad_params - if not None, it is an array of quadric parameters for a RectFlatQuadricSurfaceGM instance: each line is [a, b, c, d, e] the coefficients of the flat quadratic surface. 
		"""
		self._pos = positions
		self._th = aim_height
		face_down = rotx(N.pi)
		tower_ht = N.array([0,0,self._th])

		if focal_lengths==None:
			focal_lengths = [None]*positions.shape[0]
		if quad_params==None:
			quad_params = [None]*positions.shape[0]
		if not(hasattr(absorptivity, '__len__')):
			absorptivity = N.ones(positions.shape[0])*absorptivity
		
		self._heliostats = []

		for p in range(positions.shape[0]):
			assert(not((focal_lengths[p] != None) and (quad_params[p] != None)))
			if (focal_lengths[p] == None) and (quad_params[p] == None):
				hstat = rect_one_sided_mirror(width, height, absorptivity[p], sigma, bi_var, MCRT_option)	
			elif focal_lengths[p] != None: 
				hstat = rect_para_one_sided_mirror(width, height, focal_lengths[p], absorptivity[p], sigma, bi_var, MCRT_option)
			else:
				hstat = flat_quad_one_sided_mirror(width, height, quad_params[p], absorptivity[p], sigma, bi_var, MCRT_option)
	
			trans = face_down.copy()
			trans[:3,3] = positions[p]
			hstat.set_transform(trans)
			self._heliostats.append(hstat)

		Assembly.__init__(self, objects=self._heliostats)

	def get_heliostats(self):
		"""Access the list of one-sided mirrors representing the heliostats"""
		return self._heliostats
	
	def set_aim_height(self, h):
		"""Change the verical position of the tower's target."""
		self._th = h
	
	def aim_to_sun(self, azimuth, zenith, aim_points=None, tracking='azimuth_elevation', tracking_error=None, tracking_limits_primary_axis=None, tracking_limits_secondary_axis=None):
		"""
		Aim the heliostats in a direction that brings the incident energy to
		the tower.
		
		Arguments:
		azimuth - the sun's azimuth, in radians from North, clockwise.
		zenith - angle created between the solar vector and the Z axis, 
			in radians.
		tracking - 'azimuth_elevation'; 'titl_roll': tracking actuation method. 
		"""
		sun_vec = solar_vector(azimuth, zenith)
		if aim_points == None:
			aim_point = -self._pos 
			aim_point[:,2] += self._th
			aim_point /= N.sqrt(N.sum(aim_point**2, axis=1)[:,None])
			hstat = sun_vec + aim_point
			hstat /= N.sqrt(N.sum(hstat**2, axis=1)[:,None])
		else:
			aim_points -= self._pos 
			aim_points /= N.sqrt(N.sum(aim_points**2, axis=1)[:,None])
			hstat = sun_vec + aim_points
			hstat /= N.sqrt(N.sum(hstat**2, axis=1)[:,None])

		ang_err_1 = 0.
		ang_err_2 = 0.

		if tracking_limits_primary_axis == None:
			tracking_limits_primary_axis = [-N.pi, N.pi]
		if tracking_limits_secondary_axis == None:
			tracking_limits_secondary_axis = [-N.pi, N.pi]

		if tracking == 'azimuth_elevation':
			hstat_az = N.arctan2(hstat[:,1], hstat[:,0])
			hstat_el = N.arccos(hstat[:,2])
			for hidx in range(self._pos.shape[0]):
				if tracking_error != None:
					ang_err_1 = N.random.normal(scale=tracking_error)
					ang_err_2 = N.random.normal(scale=tracking_error)
				ang_az = hstat_az[hidx]+ang_err_1-N.pi/2.
				ang_el = -hstat_el[hidx]+ang_err_2
				if ang_az<tracking_limits_primary_axis[0] or ang_az>tracking_limits_primary_axis[1]:
					continue		
				elif -ang_el<tracking_limits_secondary_axis[0] or -ang_el>tracking_limits_secondary_axis[1]:
					continue
				az_rot = rotz(ang_az)
				elev_rot = rotx(ang_el)		
				trans = N.dot(az_rot, elev_rot)
				trans[:3,3] = self._pos[hidx]
			
				self._heliostats[hidx].set_transform(trans)

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
