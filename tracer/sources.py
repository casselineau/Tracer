"""
This module contains functions to create some frequently-used light sources.
Each function returns a RayBundle instance that represents the distribution of
rays expected from that source.

References:
.. [1] Monte Carlo Ray Tracing, Siggraph 2003 Course 44


TODO:
Systematize source declarations:
- ray vertices from surface/volume sampling -> could be obtains in the ray_trace_utils.sampling module or, potentally, adding sampling functions to the geometry managers.
- ray directions directions from normalised directional radiance distributions -> using sampling functions but keeping all radiance preoccupations here.
	- For surface emissions: cosine weighted
	- for volume emisison, no cosine weighting?
- ray energy from total source power or Planckian thermal emisison source 
"""

from numpy import random, linalg as LA
import numpy as N
from tracer.ray_bundle import RayBundle, concatenate_rays
from tracer.spatial_geometry import *
from ray_trace_utils.vector_manipulations import rotate_z_to_normal
from ray_trace_utils.electromagnetics import Planck

def single_ray_source(position, direction, flux=None):
	'''
	Establishes a single ray source originating from a definned point on a defined exact 
	direction for the purpose of testing single ray behviours.

	Arguments:
	position - column 3-array with the ray's starting position.
	direction - a 1D 3-array with the unit average direction vector for the
				bundle.
	flux - if not None, the energy transported by the ray.

	Returns:
	A Raybundle object with the corresponding characteristics.
	'''
	directions = N.tile(direction[:,None],1)
	directions /= N.sqrt(N.sum(directions**2, axis=0))
	singray = RayBundle(vertices = position, directions = directions)
	singray.set_energy(flux*N.ones(1))
	return singray

def lambertian_directions_sampling(num_rays, ang_range, normals=None):
	# Diffuse divergence from +Z:
	# development based on eq. 2.12  from [1]
	xi1 = random.uniform(low=0., high=2.*N.pi, size=num_rays) # Phi
	xi2 = random.uniform(size=num_rays) # Rtheta
	sinsqrt = N.sin(ang_range)*N.sqrt(xi2)
	dirs = N.vstack((N.cos(xi1)*sinsqrt, N.sin(xi1)*sinsqrt , N.sqrt(1.-sinsqrt**2.)))
	if normals is not None:
		dirs = rotate_z_to_normal(dirs, normals)
	return dirs

def pillbox_sunshape_directions(num_rays, ang_range):
	"""	Calculates directions for a ray bundles with ``num_rays`` rays, distributed
	as a pillbox sunshape shining toward the +Z axis, and deviating from it by
	at most ang_range, such that if all rays have the same energy, the flux
	distribution comes out right.
	
	Arguments:
	num_rays - number of rays to generate directions for.
	ang_range - in radians, the maximum deviation from +Z.
	
	Returns:
	A (3, num_rays) array whose each column is a unit direction vector for one
		ray, distributed to match a pillbox sunshape.
	"""
	return lambertian_directions_sampling(num_rays, ang_range)

def bivariate_directions(num_rays, ang_range_hor, ang_range_vert):
	"""
	Calculates directions for a ray bundles with ``num_rays`` rays, distributed
	as uniform bi-variate distribution shining toward the +Z axis and deviating from it by
	at most ang_range_hor on the zx plane and ang_range_vert on the yz plane such that if all rays have the same energy, the flux
	distribution comes out right.
	
	Arguments:
	num_rays - number of rays to generate directions for.
	ang_range_hor - in radians, the maximum deviation from +Z on the zx plane.
	ang_range_vert - in radians, the maximum deviation from +Z on the yz plane.
	
	Returns:
	A (3, num_rays) array whose each column is a unit direction vector for one
		ray, distributed to match a uniform bi-variate distribution.
	"""
	# Diffuse divergence from +Z:
	# development based on eq. 2.12  from [1]
	'''
	xi1 = N.random.uniform(low=-1., high=1., size=num_rays)
	xi2 = N.random.uniform(low=-1., high=1., size=num_rays)
	
	theta_hor = N.arcsin(N.sin(ang_range_hor)*N.sqrt(xi1))
	theta_vert = N.arcsin(N.sin(ang_range_vert)*N.sqrt(xi2))

	xa = N.sin(theta_hor)
	ya = N.sin(theta_vert)
	za = N.sqrt(1.-N.sin(theta_hor)**2.-N.sin(theta_vert)**2.)

	a = N.vstack((xa, ya, za))
	'''
	return a

def edge_rays_directions(num_rays, ang_range):
	"""
	Calculates directions for a ray bundles with ``num_rays`` rays, distributed
	as a pillbox sunshape shining toward the +Z axis, and deviating from it by
	at most ang_range, such that if all rays have the same energy, the flux
	distribution comes out right.
	
	Arguments:
	num_rays - number of rays to generate directions for.
	ang_range - in radians, the maximum deviation from +Z.
	
	Returns:
	A (3, num_rays) array whose each column is a unit direction vector for one
		ray, distributed to match a pillbox sunshape.
	"""
	# Diffuse divergence from +Z:
	# development based on eq. 2.12  from [1]
	xi1 = random.uniform(high=2.*N.pi, size=num_rays)
	sin_th = N.ones(num_rays)*N.sin(ang_range)
	a = N.vstack((N.cos(xi1)*sin_th, N.sin(xi1)*sin_th , N.cos(N.ones(num_rays)*ang_range)))

	return a

def disk_bundle(num_rays,  center,  direction,  radius, ang_range, flux=None, radius_in=0., angular_span=[0.,2.*N.pi], x_cut=None, procs=1, rays_direction=None):
	"""
	Generates a ray bundle emanating from a disk, with each surface element of 
	the disk having the same ray density. The rays all point at directions uniformly 
	distributed between a given angle range from a given direction.
	Setting of the bundle's energy is left to the caller.
	
	Arguments:
	num_rays - number of rays to generate.
	center - a column 3-array with the 3D coordinate of the disk's center
	direction - a 1D 3-array with the unit average direction vector for the
		bundle.
	radius - of the disk.
	ang_range - in radians, the maximum deviation from <direction>.
	flux - if not None, the ray bundle's energy is set such that each ray has
		an equal amount of energy, and the total energy is flux*pi*radius**2
	radius_in - Inner radius if the disc is pierced
	angular_span - wedge of the disc to consider
	
	Returns: 
	A RayBundle object with the above characteristics set.
	"""

	radius = float(radius)
	radius_in = float(radius_in)
	a = pillbox_sunshape_directions(num_rays, ang_range)
		
	# Rotate to a frame in which <direction> is Z:
	if rays_direction == None:
		rays_direction = direction
	perp_rot = rotation_to_z(rays_direction)		
	directions = N.sum(perp_rot[...,None] * a[None,...], axis=1)
	# Locations:
	# See [1]
	xi1 = random.uniform(size=num_rays)
	thetas = random.uniform(low=angular_span[0], high=angular_span[1], size=num_rays)
	rs = N.sqrt(radius_in**2.+xi1*(radius**2.-radius_in**2.))
	xs = rs * N.cos(thetas)
	ys = rs * N.sin(thetas)

	# Rotate locations to the plane defined by <direction>:
	vertices_local = N.vstack((xs, ys, N.zeros(num_rays)))
	if x_cut != None:
		vertices_local = vertices_local[:,xs<x_cut]
		missing_rays = num_rays-vertices_local.shape[1]
		while missing_rays>0:
			xi1 = random.uniform(size=2*missing_rays)
			thetas = random.uniform(low=angular_span[0], high=angular_span[1], size=2*missing_rays)
			rs = N.sqrt(radius_in**2.+xi1*(radius**2.-radius_in**2.))
			xs = rs * N.cos(thetas)
			ys = rs * N.sin(thetas)
			vertices_local = N.concatenate((vertices_local, N.vstack((xs, ys, N.zeros(2*missing_rays)))), axis=1)
			vertices_local = vertices_local[:,vertices_local[0]<x_cut]
			missing_rays = num_rays-vertices_local.shape[1]
		if missing_rays<0:
			vertices_local = vertices_local[:,:num_rays]

	vertices_global = N.dot(perp_rot, vertices_local)
	rayb = RayBundle(vertices=vertices_global + center, directions=directions)
	if flux != None:
		cosangle = 2.*N.sin(N.sqrt(N.sum((rays_direction-direction)**2))/2.)
		rayb.set_energy(N.pi*(radius**2.-radius_in**2.)/num_rays*flux*N.ones(num_rays)*N.cos(cosangle))
	else:
		rayb.set_energy(N.ones(num_rays)/float(num_rays)/procs)
	return rayb

def rect_bundle(num_rays, center, direction, x, y, ang_range, flux=None, procs=1):

	a = pillbox_sunshape_directions(num_rays, ang_range)

	# Rotate to a frame in which <direction> is Z:
	perp_rot = rotation_to_z(direction)
	directions = N.sum(perp_rot[...,None] * a[None,...], axis=1)

	xs = random.uniform(low=-x/2., high=x/2., size=num_rays)
	ys = random.uniform(low=-y/2., high=y/2., size=num_rays)

	if (direction == N.array([0,0,-1])).all():
		xs, ys = ys, xs

	# Rotate locations to the plane defined by <direction>:
	vertices_local = N.vstack((ys, xs, N.zeros(num_rays)))
	vertices_global = N.dot(perp_rot, vertices_local)

	rayb = RayBundle(vertices=vertices_global + center, directions=directions)
	if flux != None:
		rayb.set_energy(x*y/num_rays*flux*N.ones(num_rays))
	else:
		rayb.set_energy(N.ones(num_rays)/float(num_rays)/procs)
	return rayb

#def bivariate_rect_bundle(num_rays, center, direction, x, y, ang_range_vert, ang_range_hor, flux=None):

def oblique_solar_rect_bundle(num_rays, center, source_direction, rays_direction, x, y, ang_range, flux=None, procs=1, wavelength=None, ref_index=None):
	a = pillbox_sunshape_directions(num_rays, ang_range)
	# Rotate to a frame in which <direction> is Z:
	perp_rot = rotation_to_z(rays_direction)
	directions = N.sum(perp_rot[...,None] * a[None,...], axis=1)

	xs = random.uniform(low=-x/2., high=x/2., size=num_rays)
	ys = random.uniform(low=-y/2., high=y/2., size=num_rays)

	if (source_direction == N.array([0,0,-1])).all():
		xs, ys = ys, xs

	# Rotate locations to the plane defined by <direction>:
	vertices_local = N.vstack((ys, xs, N.zeros(num_rays)))
	perp_rot = rotation_to_z(source_direction)
	vertices_global = N.dot(perp_rot, vertices_local)
	if wavelength is not None:
		wavelengths = N.repeat(wavelength, num_rays)
		rayb = RayBundle(vertices=vertices_global + center, directions=directions, wavelengths=wavelengths)
	else:
		rayb = RayBundle(vertices=vertices_global + center, directions=directions)
	
	if flux != None:
		cosangle = 2.*N.arcsin(0.5*N.sqrt(N.sum((rays_direction-source_direction)**2)))
		rayb.set_energy(x*y/num_rays*flux*N.ones(num_rays)*N.cos(cosangle))
	else:
		rayb.set_energy(N.ones(num_rays)/float(num_rays)/procs)

	if wavelength is not None:
		rayb._create_property('wavelengths', N.ones(num_rays)*wavelength)
	
	if ref_index is not None:
		rayb._create_property('ref_index', N.ones(num_rays)*ref_index)

	return rayb

def edge_rays_bundle(num_rays,  center,  direction,  radius, ang_range, flux=None, radius_in=0.):

	radius = float(radius)
	radius_in = float(radius_in)
	a = edge_rays_directions(num_rays, ang_range)
		
	# Rotate to a frame in which <direction> is Z:
	perp_rot = rotation_to_z(direction)
	directions = N.sum(perp_rot[...,None] * a[None,...], axis=1)
	# Locations:
	# See [1]
	xi1 = random.uniform(size=num_rays)
	thetas = random.uniform(high=2.*N.pi, size=num_rays)
	rs = N.sqrt(radius_in**2.+xi1*(radius**2.-radius_in**2.))
	xs = rs * N.cos(thetas)
	ys = rs * N.sin(thetas)

	# Rotate locations to the plane defined by <direction>:
	vertices_local = N.vstack((xs, ys, N.zeros(num_rays)))
	vertices_global = N.dot(perp_rot, vertices_local)

	rayb = RayBundle(vertices=vertices_global + center, directions=directions)
	if flux != None:
		rayb.set_energy(N.pi*(radius**2.-radius_in**2.)/num_rays*flux*N.ones(num_rays))
	return rayb

def buie_distribution(num_rays, CSR, pre_process_CSR=True):
	
	# Angles of importance:
	theta_dni = 4.65e-3 # rad
	theta_tot = 43.6e-3 # rad

	# Polar angle array:
	thetas = N.zeros(num_rays)

	# Discrete random ray directions generation according to Buie sunshape
	# Step 1: integration over the whole Sunshape: 
	nelem = 210

	theta_int = N.linspace(0., theta_dni, nelem+1)
	phi_dni_int = N.cos(0.326*theta_int*1e3)/N.cos(0.308*theta_int*1e3)#*N.sin(theta_int)

	integ_phi_dni = 0.5*(phi_dni_int[:-1]*N.cos(theta_int[:-1])*N.sin(theta_int[:-1])+phi_dni_int[1:]*N.cos(theta_int[1:])*N.sin(theta_int[1:]))*(theta_int[1:]-theta_int[:-1])

	if CSR == 0.:
		integ_phi = N.sum(integ_phi_dni)
	else:
		if pre_process_CSR:
			if CSR<=0.1:
				CSR = -2.245e+03*CSR**4.+5.207e+02*CSR**3.-3.939e+01*CSR**2.+1.891e+00*CSR+8e-03
			else:
				CSR = 1.973*CSR**4.-2.481*CSR**3.+0.607*CSR**2.+1.151*CSR-0.020
		# Buie Sunshape parameters:
		kappa = 0.9*N.log(13.5*CSR)*CSR**(-0.3)
		gamma = 2.2*N.log(0.52*CSR)*CSR**(0.43)-0.1
		integ_phi_csr = 1e-6*N.exp(kappa)/(gamma+2.)*((theta_tot*1000.)**(gamma+2.)-(theta_dni*1000.)**(gamma+2.))
		integ_phi = N.sum(integ_phi_dni)+integ_phi_csr

	# Step 2: pdf-cdf and random variate declaration
	PDF_DNI = integ_phi_dni/integ_phi
	CDF_DNI = N.add.accumulate(N.hstack(([0],PDF_DNI)))

	# Step 3: polar angle determination: 
	thetas = N.zeros(num_rays)
	R_thetas = N.random.uniform(size=num_rays)
	for i in range(len(CDF_DNI)-1):
		slice_loc = N.logical_and((R_thetas >= CDF_DNI[i]), (R_thetas < CDF_DNI[i+1]))
		A = phi_dni_int[i]*N.cos(theta_int[i])*N.sin(theta_int[i])
		B = phi_dni_int[i+1]*N.cos(theta_int[i+1])*N.sin(theta_int[i+1])
		C = 2.*N.sum(integ_phi_dni)*(R_thetas[slice_loc]-CDF_DNI[i])*(theta_int[i+1]-theta_int[i])
		R = -(-A*theta_int[i+1]+B*theta_int[i]+N.sqrt(((theta_int[i]-theta_int[i+1])*A)**2.+C*(B-A)))/(A-B)
		thetas[slice_loc] = R

	aureole = R_thetas >= CDF_DNI[-1]

	if CSR>0.:
		thetas[aureole] = ((R_thetas[aureole]-1.)*((gamma+2.)/(10.**(3.*gamma)*N.exp(kappa))*N.sum(integ_phi_dni)-theta_dni**(gamma+2.))+R_thetas[aureole]*theta_tot**(gamma+2.))**(1./(gamma+2.))

	# Generate directions:
	xi1 = random.uniform(high=2.*N.pi, size=num_rays)
	sin_th = N.sin(N.hstack(thetas))
	directions = N.vstack((N.cos(xi1)*sin_th, N.sin(xi1)*sin_th , N.cos(thetas)))

	return directions

def sunshape_to_ray_directions(angles, norm_intensity, num_rays):
	num_rays = int(num_rays)
	thetas = N.zeros(num_rays)
	R_thetas = N.random.uniform(size=num_rays)
	# Integration over full linear intervals:
	integ_n_flux = 0.5*(norm_intensity[:-1]*N.cos(angles[:-1])*N.sin(angles[:-1])+norm_intensity[1:]*N.cos(angles[1:])*N.sin(angles[1:]))*(angles[1:]-angles[:-1])
	# Numerical PDF and CDF
	PDF = integ_n_flux/N.sum(integ_n_flux)
	CDF = N.add.accumulate(N.hstack(([0.],PDF)))
	# theta angles within the intervals:	
	for i in range(len(CDF)-1):
		slice_loc = N.logical_and((R_thetas >= CDF[i]), (R_thetas < CDF[i+1]))
		A = norm_intensity[i]*N.cos(angles[i])*N.sin(angles[i])
		B = norm_intensity[i+1]*N.cos(angles[i+1])*N.sin(angles[i+1])
		if A==B:
			thetas[slice_loc] = angles[i]+N.sum(integ_n_flux)*(R_thetas[slice_loc]-CDF[i])/A
		else:
			C = 2.*N.sum(integ_n_flux)*(R_thetas[slice_loc]-CDF[i])*(angles[i+1]-angles[i])
			R = -(-A*angles[i+1]+B*angles[i]+N.sqrt(((angles[i]-angles[i+1])*A)**2.+C*(B-A)))/(A-B)
			thetas[slice_loc] = R

	phis = N.random.uniform(high=2.*N.pi, size=num_rays)
	sin_th = N.sin(N.hstack(thetas))
	directions = N.vstack((N.cos(phis)*sin_th, N.sin(phis)*sin_th , N.cos(thetas)))
	return directions

def buie_sunshape(num_rays, center, direction, radius, CSR, flux=None, pre_process_CSR=True, rays_direction=None):
	'''
	Generate a ray bundle according to Buie et al.: "Sunshape distributions for terrestrial simulations." Solar Energy 74 (2003) 113-122 (DOI: 10.1016/S0038-092X(03)00125-7).

	Arguments:
	num_rays - number of rays in the bundle
	center - position of the source center
	direction - direction of the normal to the source disc.
	radius - radius of the source disc
	CSR - Circumsolar ratio, fraction of the incoming solar energy which incident angle is greater than the angle subtended by the solar disc.
	flux - horizontal radiation density in W/m2
	pre_process_CSR=True - Use or not the polynomial pre-processing to get better I/O match using the Buie sunshape
	rays_direction=None - If the general direction of propagation of the source is different from the source normal.

	Returns:
	A raybundle object with the above characteristics set.
	'''

	# Rays vertices (start positions):
	xv1 = random.uniform(size=num_rays)
	phiv = random.uniform(high=2.*N.pi, size=num_rays)
	rs = radius*N.sqrt(xv1)
	xs = rs * N.cos(phiv)
	ys = rs * N.sin(phiv)

	# Source surface area:
	S = N.pi*radius**2.

	# Rays escaping direction setup:
	if rays_direction == None:
		rays_direction = direction

	# Uniform ray energy:
	cosangle = 2.*N.sin(N.sqrt(N.sum((rays_direction-direction)**2))/2.)
	energy = N.ones(num_rays)*flux*S/num_rays*N.cos(cosangle)

	# Buie sunshape directions:
	a = buie_distribution(num_rays, CSR, pre_process_CSR)

	# Rotate to a frame in which <rays_direction> is Z:
	perp_rot = rotation_to_z(rays_direction)
	directions = N.sum(perp_rot[...,None] * a[None,...], axis=1)

	# Rotate to a frame in which <direction> is Z:
	perp_rot = rotation_to_z(direction)

	# Rotate locations to the plane defined by <direction>:
	vertices_local = N.vstack((xs, ys, N.zeros(num_rays)))
	vertices_global = N.dot(perp_rot, vertices_local)
	
	rayb = RayBundle(vertices=vertices_global+center, directions=directions, energy=energy)

	return rayb

def rect_buie_sunshape(num_rays, center, direction, width, height, CSR, flux=None, pre_process_CSR=True, rays_direction=None):
	'''
	Generate a ray bundle according to Buie et al.: "Sunshape distributions for terrestrial simulations." Solar Energy 74 (2003) 113-122 (DOI: 10.1016/S0038-092X(03)00125-7).

	Arguments:
	num_rays - number of rays in the bundle
	center - position of the source center
	direction - direction of the normal to the source disc.
	radius - radius of the source disc
	CSR - Circumsolar ratio, fraction of the incoming solar energy which incident angle is greater than the angle subtended by the solar disc.
	flux - horizontal radiation density in W/m2
	pre_process_CSR=True - Use or not the polynomial pre-processing to get better I/O match using the Buie sunshape
	rays_direction=None - If the general direction of propagation of the source is different from the source normal.

	Returns:
	A raybundle object with the above characteristics set.
	'''

	# Rays vertices (start positions):
	xs = width*(random.uniform(size=num_rays)-0.5)
	ys = height*(random.uniform(size=num_rays)-0.5)

	# Source surface area:
	S = width*height

	# Rays escaping direction setup:
	if rays_direction is None:
		rays_direction = direction

	# Uniform ray energy:
	cosangle = 2.*N.sin(N.sqrt(N.sum((rays_direction-direction)**2))/2.)
	energy = N.ones(num_rays)*flux*S/num_rays*N.cos(cosangle)

	# Buie sunshape directions:
	a = buie_distribution(num_rays, CSR, pre_process_CSR)

	# Rotate to a frame in which <rays_direction> is Z:
	perp_rot = rotation_to_z(rays_direction)
	directions = N.sum(perp_rot[...,None] * a[None,...], axis=1)

	# Rotate to a frame in which <direction> is Z:
	perp_rot = rotation_to_z(direction)

	# Rotate locations to the plane defined by <direction>:
	vertices_local = N.vstack((xs, ys, N.zeros(num_rays)))
	vertices_global = N.dot(perp_rot, vertices_local)
	
	rayb = RayBundle(vertices=vertices_global+center, directions=directions, energy=energy)

	return rayb


def regular_square_bundle(num_rays, center, direction, width):
	"""
	Generate a ray bundles whose rays are equally spaced along a square grid,
	and all pointing in the same direction.
	
	Arguments:
	num_rays - number of rays to generate.
	center - a column 3-array with the 3D coordinate of the disk's center
	direction - a 1D 3-array with the unit direction vector for the bundle.
	width - of the square of starting points.
	
	Returns: 
	A RayBundle object with the above charachteristics set.
	"""
	rot = rotation_to_z(direction)
	directions = N.tile(direction[:,None], (1, num_rays))
	range = N.s_[-width:width:float(2*width)/N.sqrt(num_rays)]
	xs, ys = N.mgrid[range, range]
	vertices_local = N.array([xs.flatten(),  ys.flatten(),  N.zeros(len(xs.flatten()))])
	vertices_global = N.dot(rot,  vertices_local)

	rayb = RayBundle()
	rayb.set_vertices(vertices_global + center)
	rayb.set_directions(directions)
	return rayb

def triangular_bundle(num_rays, A, B, C, direction=None, ang_range=N.pi/2., flux=None, procs=1):
	"""
	Triangular ray-casting surface. A, B and C are 3D coordinates of the vertices. Right hand rule determines the normal vector direction.
	Arguments:
	- num_rays: the number of rays 
	- A: The first summit of the triangle and its anchor point.
	- AB and AC the vertices of the sides of the triangle in its plane of reference.
	- direction: The direction around which rays are escaping the source. If None, the direction is the normal.
	- ang_range: the angular range of the rays emitted by the source

	Returns: 
	- A ray bundle object for tracing
	"""
	# Triangle ray vertices:
	# Declare random numbers:
	r1 = N.vstack(N.random.uniform(size=num_rays))
	r2 = N.vstack(N.random.uniform(size=num_rays))
	
	AB = B-A	
	AC = C-A
	sqrtr1 = N.sqrt(r1)
	vertices = (A+sqrtr1*(1.-r2)*AB+r2*sqrtr1*AC).T # Triangle point picking

	# Local referential directions:
	a = pillbox_sunshape_directions(num_rays, ang_range)
	# Normal vector:
	normal = N.cross(AB, AC)
	normal = normal/N.sqrt(N.sum(normal**2))

	if direction is None:
		direction = normal
	
	# Rotate to a frame in which <direction> is direction:
	rot = rotation_to_z(direction)
	directions = N.sum(rot[...,None] * a[None,...], axis=1)

	rayb = RayBundle()

	rayb.set_vertices(vertices)
	rayb.set_directions(directions)

	# Heron's formula for triangle surface area
	l1 = N.sqrt(N.sum(AB**2))
	l2 = N.sqrt(N.sum(AC**2))
	l3 = N.sqrt(N.sum((-AB+AC)**2))
	s = (l1+l2+l3)/2.
	area = N.sqrt(s*(s-l1)*(s-l2)*(s-l3))
	if flux != None:
		cosangle = 2.*N.arcsin(0.5*N.sqrt(N.sum((direction-normal)**2)))
		rayb.set_energy(area/num_rays*flux*N.ones(num_rays)*N.cos(cosangle))
	else:
		rayb.set_energy(N.ones(num_rays)/float(num_rays)/procs)

	return rayb

def trapezoid_bundle(num_rays, A, B, C, direction=None, ang_range=N.pi/2., flux=None, procs=1):
	"""
	Regular trapezoid ray-casting surface.
	ABCD must be placed to follow the perimeter of the quadrilateral. AB is the first base and CD is the second base.
	Arguments:
	- num_rays: the number of rays cast
	- A: the first point of the trapezoid.
	- B: second vertex forming AB the first base.
	- C: third vertex forming AC the first diagonal. D is obtained by symmetry.
	- direction: The around which the rays escape the source. If None: the normal of the surface with respect to the right hand rule.
	- ang_range: the angular range of the rays emitted by the source
	Returns:
	- A ray-bundle object to ray-trace
	"""
	AB = B-A
	AC = C-A
	# Separate into two triangles ABC and ACD and calculate their respective areas:
	l1 = N.sqrt(N.sum(AB**2))
	l2 = N.sqrt(N.sum(AC**2))
	cos_theta = N.dot(AC, AB)/(l1*l2)
	cB = AB*(1.-1./l1*l2*cos_theta)
	CD = -(AB-2.*cB)
	AD = AC+CD
	D = A+AD
	l3 = N.sqrt(N.sum(AD**2))
	l4 = N.sqrt(N.sum((-AB+AC)**2))
	l5 = N.sqrt(N.sum((-AC+AD)**2))
	# Area calculated using Heron's formula
	s1 = (l1+l2+l4)/2.
	s2 = (l2+l3+l5)/2.
	area_ABC = N.sqrt(s1*(s1-l1)*(s1-l2)*(s1-l4))
	area_ACD = N.sqrt(s2*(s2-l2)*(s2-l3)*(s2-l5))
	# Calculate how many rays per triangle are needed considering the number of rays asked for:
	num_rays_ABC = int(area_ABC/(area_ABC+area_ACD)*num_rays)
	num_rays_ACD = num_rays-num_rays_ABC

	# Get a ray-bundle for each triangle and concatenate them:
	rayb_ABC = triangular_bundle(num_rays_ABC, A, B, C, direction, ang_range, flux)
	rayb_ACD = triangular_bundle(num_rays_ACD, A, C, D, direction, ang_range, flux)
	rayb = concatenate_rays([rayb_ABC,rayb_ACD])
	if flux == None:
		rayb.set_energy(N.ones(num_rays)/float(num_rays)/procs)

	return rayb

def vf_frustum_bundle(num_rays, r0, r1, depth, center, direction, flux=None , rays_in=True, angular_span=[0.,2.*N.pi], angular_range=N.pi/2.):
	'''
	Generate a frustum shaped lambertian source with randomly situated rays to compute view factors. The overall energy of the bundle is 1.

	Arguments:
	num_rays - number of rays to generate.
	center - a column 3-array with the 3D coordinate of the center of one of the bases.
	r0 - The radius of the frustum base which center coordinate has been given.
	r1 - the radius of the frustum at the other base location.
	depth - the depth of the frustum.
	direction - The orientation of the overall bundle. 
			   Positive if in the same direction as the depth.
	rays_in - True if rays are fired towards the axis of the frustum.
	angular_span - wedge of the shape to consider.

	Returns:
	A raybundle object with the above characteristics set.
	'''
	r0 = float(r0)
	r1 = float(r1)
	depth = float(depth)

	num_rays = int(num_rays)

	dir_flat = pillbox_sunshape_directions(num_rays, angular_range)

	c = (r1-r0)/depth

	R = random.uniform(size=num_rays)
	
	'''
	if r0<r1:
		zs = depth*N.sqrt(R)
	else:
		zs = depth*(1.-N.sqrt(R))
	'''
	rs = N.sqrt((r1**2.-r0**2.)*R+r0**2.)
	zs = (rs-r0)/((r1-r0)/depth)

	phi_s = random.uniform(low=angular_span[0], high=angular_span[1], size=num_rays)

	xs = rs * N.cos(phi_s)
	ys = rs * N.sin(phi_s)

	theta_s = N.arctan(c)
	theta_rot = -N.pi/2.+theta_s
	yrot = roty(theta_rot)[:3,:3]
	local_unit = N.zeros((N.shape(dir_flat)))
	for t in range(N.shape(dir_flat)[1]):
		rotd = N.dot(yrot, dir_flat[:,t])
		zrot = rotz(phi_s[t])[:3,:3]
		local_unit[:,t] = N.dot(zrot, rotd)

	if rays_in == False:
		local_unit = -1.*local_unit

	vertices_local = N.vstack((xs, ys, zs))

	perp_rot = rotation_to_z(direction)
	vertices_global = N.dot(perp_rot, vertices_local)
	directions = N.dot(perp_rot, local_unit)

	if flux == None:
		energy = N.ones(num_rays)/float(num_rays)
	else:
		area = (angular_span[1]-angular_span[0])*(r1+r0)/2.*N.sqrt(abs(r1-r0)**2.+depth**2.)
		energy = N.ones(num_rays)*flux*area/float(num_rays)

	rayb = RayBundle(vertices = vertices_global+center, directions = directions, energy = energy)

	return rayb

def vf_cylinder_bundle(num_rays, rc, lc, center, direction, flux=None, rays_in=True, angular_span=[0.,2.*N.pi], ang_range=N.pi/2.):
	'''
	Generate a cylinder shaped lambertian source with randomly situated rays to compute view factors. The overall energy of the bundle is 1.

	Arguments:
	num_rays - number of rays to generate.
	center - a column 3-array with the 3D coordinate of the mid heigh on the central axis..
	rc - The radius of the cylinder.
	lc - the length of the cylinder.
	direction - the direction of outgoing rays as projected on the cylinder axis. 
			   Positive if in the same direction as lc.
	rays_in - True if rays are fired towards the axis of the frustum.
	angular_span - wedge of the shape to consider.

	Returns:
	A raybundle object with the above characteristics set.
	'''
	rc = float(rc)
	lc = float(lc)
	num_rays = int(num_rays)

	zs = lc*random.uniform(size=num_rays)-lc/2.

	phi_s = random.uniform(low=angular_span[0], high=angular_span[1], size=num_rays)

	xs = rc * N.cos(phi_s)
	ys = rc * N.sin(phi_s)

	dir_flat = pillbox_sunshape_directions(num_rays, ang_range)

	yrot = roty(-N.pi/2.)[:3,:3]
	local_unit = N.zeros((N.shape(dir_flat)))
	for t in range(N.shape(dir_flat)[1]):
		zrot = rotz(phi_s[t])[:3,:3]
		rot = N.dot(zrot, yrot)
		local_unit[:,t] = N.dot(rot, dir_flat[:,t])

	if rays_in == False:
		local_unit = -local_unit

	vertices_local = N.vstack((xs, ys, zs))
	perp_rot = rotation_to_z(direction)
	vertices_global = N.dot(perp_rot, vertices_local)
	directions = N.dot(perp_rot, local_unit)

	if flux == None:
		energy = N.ones(num_rays)/float(num_rays)
	else:
		area = rc*(angular_span[1]-angular_span[0])*lc
		energy = N.ones(num_rays)*flux*area/float(num_rays)

	rayb = RayBundle(vertices = vertices_global+center, directions = directions, energy = energy)

	return rayb

def spectral_band_axisymmetrical_thermal_emission_source(positions, normals, area, thetas, band_emittance, T, nrays, band, ref_index):
	'''
	Returns a RayBundle instance describing a thermal emitter with given directional emissivities in a given spectral band.

	Arguments:
	positions 	ray positions
	normals  	normals to the surface at teh ray positions.
	thetas 	 	angles at which emittances are given
	band_emittance 	if a number, the band hemispherical emittance,
					if a 1D array of the length of thetas, the directional band emittances
	T  			Temperature of the emitter in K
	nrays 		Number of rays to trace
	band		A list of 2 values delimiting the spectral band
	'''
	from ray_trace_utils.sampling import PW_lincossin_distribution
	# Build axisymmetrical emissions profile
	# Integrate the emittance
	wls = N.linspace(band[0], band[1], int((band[1]-band[0])/1e-9))
	bb_spectral_radiance_in_band = Planck(wls, T)
	bb_spectral_radiance_in_band_integrated = N.trapezoid(bb_spectral_radiance_in_band, wls)
	bb_spectral_radiance_in_band_mid_wl = (bb_spectral_radiance_in_band[1:] + bb_spectral_radiance_in_band[:-1]) / 2. * (wls[1:] - wls[:-1])
	mid_wls = (wls[1:] + wls[:-1]) / 2.
	wl_avg = N.sum(mid_wls * bb_spectral_radiance_in_band_mid_wl) / bb_spectral_radiance_in_band_integrated
	source_spectral_radiance = band_emittance*bb_spectral_radiance_in_band_integrated

	# Sample the emmissions profile distribution to get directions and energy
	thetas_rays, weights = PW_lincossin_distribution(thetas, source_spectral_radiance).sample(nrays)
	source_exitance = N.trapezoid(source_spectral_radiance*N.cos(thetas)*N.sin(thetas), thetas)*2.*N.pi
	phis_rays = N.random.uniform(size=nrays)*2.*N.pi
	directions = N.array([N.sin(thetas_rays)*N.cos(phis_rays), N.sin(thetas_rays)*N.sin(phis_rays), N.cos(thetas_rays)])
	# Rotate to make z the normals
	directions = rotate_z_to_normal(directions, normals)
	# for i,d in enumerate(directions.T):
	#	directions[:,i] = N.dot(rotation_to_z(normals[:,i]), d)
	energy = weights/N.sum(weights)*source_exitance*area
	rayb = RayBundle(vertices=positions, directions=directions, energy=energy)
<<<<<<< HEAD
	wl_avg = N.sum(wls*bb_spectral_radiance_in_band)/N.sum(bb_spectral_radiance_in_band)
	rayb._create_property('wavelengths', N.ones(nrays)*wl_avg)
	rayb._create_property('ref_index', N.ones(nrays))
=======
	rayb._create_property('wavelengths', N.ones(nrays)*wl_avg)
	rayb._create_property('ref_index', N.ones(nrays)*ref_index)
>>>>>>> e7aefa278279c11136a2aefaa422a8cc642452bf
	return rayb
