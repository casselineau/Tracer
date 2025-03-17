import numpy as N
from tracer.spatial_geometry import general_axis_rotation


def get_angles(v1, v2, signed=False):
	'''
	v1 - (3,n)
	v2 - (3,n)
	'''
	v2 = v2[:,0]
	if len(v2.shape)<=1:
		if len(v1.shape)<=1:
			return get_angle(v1, v2, signed)
		else:
			costheta = N.vecdot(v1.T, v2)
			angs = N.arccos(costheta)
	else:
		costheta = N.vecdot(v1.T, v2.T)
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
	
def axes_and_angles_between(vecs, normal):
	'''
	Determine the normal of a plane forme by two vectors (each vecs and the respective normals), and estimates the angle to go from vecs to normals while rotating around this normal.
	vecs (3,n)
	normal (3) or (3,n)
	'''
	if len(vecs.shape)>1:
		axes = get_plane_normals(vecs.T, normal.T) # axis of rotation ie normal of the plane formed by both vectors
		angles = get_angles(vecs, normal, signed=False) # angle between +z in directiosn referential and normals on the plane defined earlier
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

