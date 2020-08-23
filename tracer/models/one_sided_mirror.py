# -*- coding: utf-8 -*-
# Implement an object with a front reflective surface and an opaque back.

from ..object import AssembledObject
from ..surface import Surface
from ..flat_surface import RectPlateGM
from ..paraboloid import ParabolicDishGM, RectangularParabolicDishGM
from ..quadratic_surface import RectFlatQuadricSurfaceGM
from .. import optics_callables as opt

from numpy import r_
import numpy as N
import types
import math #

def surfaces_for_next_iteration(self, rays, surface_id):
	"""
	Informs the ray tracer that some of the surfaces can be skipped in the
	next ireration for some of the rays.
	This implementation marks all surfaces as irrelevant to all rays.
	
	Arguments:
	rays - the RayBundle to check. 
	surface_id - the index of the surface which generated this bundle.
	
	Returns:
	an array of size s by r for s surfaces in this object and r rays,
		stating whether ray i=1..r should be intersected with surface j=1..s
		in the next iteration.
	"""
	return N.zeros((len(self.surfaces), rays.get_num_rays()), dtype=N.bool)

def rect_one_sided_mirror(width, height, absorptivity=0, sigma=0., bi_var=True, option=None, location=None, rotation=None):
	"""
	construct an object with two surfaces: one on the XY plane, that is
	specularly reflective, and one slightly below (negative z), that is opaque.
	
	Arguments:
	width - the extent along the x axis in the local frame.
	height - the extent along the y axis in the local frame.
	absorptivity - the ratio of energy incident on the reflective side that's
		not reflected back.
	"""
	if option == 'fast':
		surf = Surface(RectPlateGM(width, height),
				   opt.OneSidedRealReflective(absorptivity, sigma, bi_var), location=location, rotation=rotation)
	else:
		surf = Surface(RectPlateGM(width, height),
			opt.OneSidedRealReflectiveDetector(absorptivity, sigma, bi_var), 							location=location, rotation=rotation)
	obj = AssembledObject(surfs=[surf])

	return obj


def rect_para_one_sided_mirror(width, height, focal_length, absorptivity=0., sigma=0., bi_var=True, option=None, location=None, rotation=None):

	if option == 'fast':
		surf = Surface(RectangularParabolicDishGM(width, height, focal_length),
				   opt.OneSidedRealReflective(absorptivity, sigma, bi_var), location=location, rotation=rotation)
	else:
		surf = Surface(RectangularParabolicDishGM(width, height, focal_length),
			opt.OneSidedRealReflectiveDetector(absorptivity, sigma, bi_var), 							location=location, rotation=rotation)

	surf.set_location(surf.get_location()-N.array([0,0,(width/2.)**2.*surf.get_geometry_manager().a+(height/2.)**2.*surf.get_geometry_manager().b])) # to have the aperture as the reference.
	obj = AssembledObject(surfs = [surf])
	obj.surfaces_for_next_iteration = types.MethodType(
		surfaces_for_next_iteration, obj, obj.__class__)
	return obj

def flat_quad_one_sided_mirror(width, height, quad_params, absorptivity=0., sigma=0., bi_var=True, option=None, location=None, rotation=None):

	a, b, c, d, e, f = quad_params
	if option == 'fast':
		surf = Surface(RectFlatQuadricSurfaceGM(width, height, a, b, c, d, e, f),
				   opt.OneSidedRealReflective(absorptivity, sigma, bi_var), location=location, rotation=rotation)
	elif option == 'receiver':
		surf = Surface(RectFlatQuadricSurfaceGM(width, height, a, b, c, d, e, f),
			opt.OneSidedRealReflectiveReceiver(absorptivity, sigma, bi_var), 							location=location, rotation=rotation)
	else:
		surf = Surface(RectFlatQuadricSurfaceGM(width, height, a, b, c, d, e, f),
			opt.OneSidedRealReflectiveDetector(absorptivity, sigma, bi_var), 							location=location, rotation=rotation)

	surf.set_location(surf.get_location()-N.array([0,0,(width/2.)**2.*surf.get_geometry_manager().a+(height/2.)**2.*surf.get_geometry_manager().b])) # to have the aperture as the reference.
	obj = AssembledObject(surfs = [surf])
	obj.surfaces_for_next_iteration = types.MethodType(
		surfaces_for_next_iteration, obj, obj.__class__)
	return obj

def one_sided_receiver(width, height, absorptivity=1, location=None, rotation=None):
	"""
	construct an object with two surfaces: one on the XY plane, that is
	specularly reflective, and one slightly below (negative z), that is opaque.
	The reflective surface is by default also apaque; it is a ReflectiveReceiver
	object, so all hits can be obtained after a trace using that surface's 
	get_all_hits() method.
	
	Arguments:
	width - the extent along the x axis in the local frame.
	height - the extent along the y axis in the local frame.
	absorptivity - the ratio of energy incident on the reflective side that's
		not reflected back.
	
	Returns:
	front - the receiving surface
	obj - the AssembledObject containing both surfaces
	"""
	front = Surface(RectPlateGM(width, height), opt.OneSidedReflectiveReceiver(absorptivity))

	obj = AssembledObject(surfs=[front])
	
	return obj
