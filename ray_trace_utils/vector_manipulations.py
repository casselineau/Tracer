import numpy as N
from tracer.spatial_geometry import general_axis_rotation

def get_angles(v1, v2, signed=False):
	'''
	v1 - (3,n)
	v2 - (3)
	'''

	if len(v1.shape)<=1:
		return get_angle(v1, v2, signed)

	proj = N.dot(v1.T, v2)
	v2s = N.tile(v2, (v1.shape[1],1)).T
	costheta = proj.T/(N.sqrt(N.sum(v1**2, axis=0))*N.sqrt(N.sum(v2s**2, axis=0)))
	angs = N.arccos(costheta)
	if signed == True:
		sign = N.sign(proj)
		sign[sign==0] = 1.
		angs = sign*angs

	return angs
	
def get_angle(v1, v2, signed=False):
	'''
	v1 - (3)
	v2 - (3)
	'''
	proj = N.dot(v1.T, v2)
	costheta = proj/(N.sqrt(N.sum(v1**2))*N.sqrt(N.sum(v2**2)))
	angs = N.arccos(costheta)
	if signed == True:
		sign = N.sign(proj)
		if sign == 0.:
			sign = 1.
		angs = sign*angs
		
	return angs
	
def axes_and_angles_between(vecs, normals):
	'''
	Determine the plane normal betwene each vecs and respective normals, and estimates the angle to go from vecs to normals while rotating around the normal.
	'''
	if len(vecs.shape)>1:
		normals = N.tile(normal, (vecs.shape[1],1)).T
		axes = get_plane_normals(vecs.T, normals.T) # axis of rotation ie normal of the plane formed by both vectors
		angles = get_angles(vecs, normals, signed=False) # angle between +z in directiosn referential and normals on the plane defined earlier
	else:
		axes = get_plane_normals(vecs.T, normal.T) # axis of rotation ie normal of the plane formed by both vectors
		angles = get_angle(vecs, normal, signed=False) # angle between +z in directiosn referential and normals on the plane defined earlier

	return axes, angles
	
def rotate_z_to_normal(vecs, normals):
	'''
	Rotate vecs so that they consider normals as their +z. The rotation matrix is established so that it is the minimal rotation along the plane formed between each direction and their respective normal unlike the rotate_to_z alternative in the tracer.spatial_geometry module.
	'''
	zs = N.zeros((vecs.shape))
	zs[2] = 1.
	axes, angles = axes_and_angles_between(zs, normals)
	# rotate +z to normals
	for i, d in enumerate(vecs.T):
		if angles[i] != 0.:
			rot = general_axis_rotation(axes[:,i], angles[i])
			vecs[:,i] = N.dot(rot, d)

	return vecs

def project_on_plane(v1, normal):
	# projects v1 on the plane defined by the normal
	normals = N.tile(normal, (1,v1.shape[1]))
	proj = N.dot(v1.T, normal)
	proj = N.tile(proj/(N.sqrt(N.sum(normal**2, axis=0))**2), (1,v1.shape[0])).T
	proj = v1-proj*normals
	return proj

def get_plane_normals(v1, v2):
	# returns the normal to a plane defined by v1 and v2
	plane = N.cross(v1, v2).T
	plane /= N.sqrt(N.sum(plane**2, axis=0))
	return plane

def get_az_el_th(dirs, norm):
	z = N.array([0.,0.,1.])
	el_plane = N.vstack(get_plane_normal(N.hstack(norm), z))
	# Project dirs on el_plane
	proj = project_on_plane(dirs, el_plane)
	# Get the elevation angle
	el = get_angle(proj, norm, signed=True)
	# Project on the plane of the bank
	proj = project_on_plane(dirs, norm)
	# Project on the Elevation plane to find teh sign of the azimuth angle:
	sign = N.sign(N.dot(proj.T, el_plane)).T
	# Project on the azimuthal plane
	proj = project_on_plane(dirs, N.vstack(z))
	# Get azimuth angle
	az = sign*get_angle(proj, norm, signed=True)
	# Get the angle with the normal
	theta = get_angle(dirs, norm)
	return az, el, theta

def get_bank_th_ph(dirs, norm):
	'''
	Used to get the external facing normal of simplified pipe banks in cylindrical receivers.
	'''
	# Get theta
	theta = get_angle(dirs, norm)
	# Get phi from local x, perpendicular to the pipes direction.
	y_loc = N.array([0.,0.,1.])
	x_loc = get_plane_normal(y_loc, N.hstack(norm))
	proj = project_on_plane(dirs, norm)
	phi = get_angle(proj, N.vstack(x_loc))
	return theta, phi

def AABB(vecs):
	'''
	Axes-Aligned Bounding-Box determination
	Arguments:
	- vecs - (3, N)
	Returns:
	- minimum_point and maximum_point of the box, ie. minimum coordinates and maximum coordinates
	'''
	minimum_point = N.amin(vecs, axis=1)
	maximum_point = N.amax(vecs, axis=1)
	return minimum_point, maximum_point

