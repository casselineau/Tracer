"""
This module contains functions to create some frequently-used light sources.
Each function returns a RayBundle instance that represents the distribution of
rays expected from that source.

References:
.. [1] Monte Carlo Ray Tracing, Siggraph 2003 Course 44
"""

from numpy import random, linalg as LA
import numpy as N
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D



from .ray_bundle import RayBundle
from .spatial_geometry import *

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

def pillbox_sunshape_directions(num_rays, ang_range):
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
    xi2 = random.uniform(size=num_rays)
    theta = N.arcsin(N.sin(ang_range)*N.sqrt(xi2))

    sin_th = N.sin(theta)
    a = N.vstack((N.cos(xi1)*sin_th, N.sin(xi1)*sin_th , N.cos(theta)))

    return a

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

def solar_disk_bundle(num_rays,  center,  direction,  radius, ang_range, flux=None, radius_in=0.):
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
    
    Returns: 
    A RayBundle object with the above characteristics set.
    """

	# FIXME why should 'center' be a column vector... that's just annoying.

    radius = float(radius)
    radius_in = float(radius_in)
    a = pillbox_sunshape_directions(num_rays, ang_range)
        
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

def solar_rect_bundle(num_rays, center, direction, x, y, ang_range, flux=None):

    a = pillbox_sunshape_directions(num_rays, ang_range)

    # Rotate to a frame in which <direction> is Z:
    perp_rot = rotation_to_z(direction)
    directions = N.sum(perp_rot[...,None] * a[None,...], axis=1)

    xs = random.uniform(low=-x/2., high=x/2., size=num_rays)
    ys = random.uniform(low=-y/2., high=y/2., size=num_rays)

    if direction == [0,0,-1]:
        xs, ys = ys, xs

    # Rotate locations to the plane defined by <direction>:
    vertices_local = N.vstack((ys, xs, N.zeros(num_rays)))
    vertices_global = N.dot(perp_rot, vertices_local)

    rayb = RayBundle(vertices=vertices_global + center, directions=directions)
    if flux != None:
        rayb.set_energy(x*y/num_rays*flux*N.ones(num_rays))
    return rayb

#def bivariate_rect_bundle(num_rays, center, direction, x, y, ang_range_vert, ang_range_hor, flux=None):



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

def buie_sunshape(num_rays, center, direction, radius, CSR, flux=None):
    '''
    Generate a ray bundle according to Buie et al.: "Sunshape distributions for terrestrial simulations." Solar Energy 74 (2003) 113-122 (DOI: 10.1016/S0038-092X(03)00125-7).

    Arguments:
    num_rays - number of rays in the bundle
    center - position of the source center
    direction - direction of the normal to the source disc.
    radius - radius of the source disc
    CSR - Circumsolar ratio, fraction of the incoming solar energy which incident angle is greater than the angle subtended by the solar disc.
    flux - horizontal radiation density in W/m2

    Returns:
    A raybundle object with the above characteristics set.
    '''
    # Angles of importance:
    theta_dni = 4.65e-3 # rad
    theta_tot = 43.6e-3 # rad

    # Rays vertices (start positions):

    xv1 = random.uniform(size=num_rays)
    phiv = random.uniform(high=2.*N.pi, size=num_rays)
    rs = radius*N.sqrt(xv1)
    xs = rs * N.cos(phiv)
    ys = rs * N.sin(phiv)

    # Source surface area:
    S = N.pi*radius**2.

    # Uniform ray energy:
    energy = N.ones(num_rays)*flux*S/num_rays

    # Buie Sunshape parameters:
    kappa = 0.9*N.log(13.5*CSR)*CSR**(-0.3)
    gamma = 2.2*N.log(0.52*CSR)*CSR**(0.43)-0.1

    # Discrete random ray directions generation according to Buie sunshape
    # Step 1: integration over the whole Sunshape: 
    theta_dni_int = N.linspace(0.,theta_dni,num=800)
    theta_csr_int = N.linspace(theta_dni,theta_tot,num=800)
    thetas_int = N.hstack((theta_dni_int,theta_csr_int))

    integ_phi_dni = N.zeros(len(theta_dni_int))
    integ_phi_csr = N.zeros(len(theta_csr_int))

    for i in xrange(len(theta_dni_int)-1):
        theta_int = N.linspace(theta_dni_int[i], theta_dni_int[i+1], num=400)
        phi_dni_int = N.cos(0.326*theta_int*1000.)/N.cos(0.308*theta_int*1000.)
        integ_phi_dni[i] = 2.*N.pi*N.trapz(phi_dni_int*theta_int*1000.,theta_int*1000.)
    for i in xrange(len(theta_csr_int)-1):
        integ_phi_csr[i] = 2.*N.pi*N.exp(kappa)/(gamma+2.)*((theta_csr_int[i+1]*1000.)**(gamma+2.)-(theta_csr_int[i]*1000.)**(gamma+2.))

    integ_phi = N.sum(integ_phi_dni)+N.sum(integ_phi_csr)
    integ_phis = N.hstack((integ_phi_dni, integ_phi_csr))

    theta = []

    # Step 2: random declaration slice by slice according to the Buie sunshape:
    for i in xrange(len(theta_dni_int)-1):
        theta.append(N.random.uniform(low=theta_dni_int[i], high=theta_dni_int[i+1], size=N.round(integ_phi_dni[i]/integ_phi*num_rays)))
    for i in xrange(len(theta_csr_int)-1):
        theta.append(N.random.uniform(low=theta_csr_int[i], high=theta_csr_int[i+1], size=N.round(integ_phi_csr[i]/integ_phi*num_rays)))
    theta= N.hstack(theta)
    # re-arange if rounding of the number of rays per interval gives too much or not enough rays in total.
    if len(theta) < num_rays:
        theta = N.hstack((theta,N.random.uniform(low=N.sin(0.), high=N.sin(theta_tot), size=num_rays-len(theta))))
    if len(theta) > num_rays:
        theta = theta[:num_rays]

    # Generate directions:
    xi1 = random.uniform(high=2.*N.pi, size=num_rays)
    sin_th = N.sin(N.hstack(theta))
    a = N.vstack((N.cos(xi1)*sin_th, N.sin(xi1)*sin_th , N.cos(theta)))

    # Identify dni and aureole cases:
    dni = theta <= theta_dni
    aureole = ~dni

    # Rotate to a frame in which <direction> is Z:
    perp_rot = rotation_to_z(direction)
    directions = N.sum(perp_rot[...,None] * a[None,...], axis=1)

    # Rotate locations to the plane defined by <direction>:
    vertices_local = N.vstack((xs, ys, N.zeros(num_rays)))
    vertices_global = N.dot(perp_rot, vertices_local)
    
    rayb = RayBundle(vertices = vertices_global+center, directions = directions, energy = energy)
    csr_calc = N.sum(energy[aureole])/N.sum(energy)

    return rayb#, csr_calc, theta, integ_phis, thetas_int, energy

def square_bundle(num_rays, center, direction, width):
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

def vf_frustum_bundle(num_rays, r0, r1, depth, center, direction, rays_in=True, procs=1):
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

    Returns:
    A raybundle object with the above characteristics set.
    '''
    r0 = float(r0)
    r1 = float(r1)
    depth = float(depth)
    num_rays = float(num_rays)

    dir_flat = pillbox_sunshape_directions(num_rays, N.pi/2.)

    c = (r1-r0)/depth

    R = random.uniform(size=num_rays)

    zs = (-r0+N.sqrt(r0**2.+R*(r1**2.-r0**2.)))/c

    phi_s = 2.*N.pi*random.uniform(size=num_rays)
    rs = r0+c*zs
    xs = rs * N.cos(phi_s)
    ys = rs * N.sin(phi_s)

    theta_s = N.arctan(c)

    yrot = roty(theta_s-N.pi/2.)[:3,:3]
    local_unit = N.zeros((N.shape(dir_flat)))
    for t in xrange(N.shape(dir_flat)[1]):
        zrot = rotz(phi_s[t])[:3,:3]
        rot = N.dot(zrot, yrot)
        local_unit[:,t] = N.dot(rot, dir_flat[:,t])

    if rays_in == False:
        local_unit = -local_unit

    vertices_local = N.vstack((xs, ys, zs))
    perp_rot = rotation_to_z(direction)
    vertices_global = N.dot(perp_rot, vertices_local)
    directions = N.dot(perp_rot, local_unit)

    energy = N.ones(num_rays)/num_rays/procs
    rayb = RayBundle(vertices = vertices_global+center, directions = directions, energy = energy)

    return rayb

def vf_cylinder_bundle(num_rays, rc, lc, center, direction, rays_in=True, procs=1):
    '''
    Generate a cylinder shaped lambertian source with randomly situated rays to compute view factors. The overall energy of the bundle is 1.

    Arguments:
    num_rays - number of rays to generate.
    center - a column 3-array with the 3D coordinate of the center of one of the bases.
    rc - The radius of the cylinder.
    lc - the length of the cylinder.
    direction - the direction of outgoing rays as projected on the cylinder axis. 
               Positive if in the same direction as lc.
    rays_in - True if rays are fired towards the axis of the frustum.

    Returns:
    A raybundle object with the above characteristics set.
    '''
    rc = float(rc)
    lc = float(lc)
    num_rays = float(num_rays)

    zs = lc*random.uniform(size=num_rays)

    phi_s = 2.*N.pi*random.uniform(size=num_rays)

    xs = rc * N.cos(phi_s)
    ys = rc * N.sin(phi_s)

    dir_flat = pillbox_sunshape_directions(num_rays, N.pi/2.)

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
    '''
    plt.hist(vertices_local[2,:]/(N.sqrt(vertices_local[0,:]**2.+vertices_local[1,:]**2.)))
    plt.show()
    '''
    energy = N.ones(num_rays)/num_rays/procs

    rayb = RayBundle(vertices = vertices_global+center, directions = directions, energy = energy)

    return rayb


# vim: et:ts=4
