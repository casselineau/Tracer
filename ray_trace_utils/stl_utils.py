import numpy as N
from sys import path
from tracer.polygon import FlatSimplePolygonGM
from tracer.triangular_face import TriangularFace
from tracer.surface import *
from tracer.object import *
from tracer.boundary_shape import BoundaryBox
from tracer.spatial_geometry import roty, rotz
from .vector_manipulations import AABB, get_plane_normal

def cylinder_verts_faces(radius, length, angbins=24, lbins=10, capped=False):
	'''
	Makes cylinder vertices and faces arrays to build stl geometries
	'''
	angbins = angbins+1
	lbins = lbins+1
	thetas = N.linspace(0., 2.*N.pi, angbins)
	ls = N.linspace(-length/2., length/2., lbins)
	verts = []
	faces = []

	if capped:
		# Face:
		verts.append((0,0,ls[0]))
		xs = radius*N.cos(thetas)
		ys = radius*N.sin(thetas)
		zs = ls[0]
		for j in range(len(xs)):
			verts.append((xs[j], ys[j], zs))
		
		for i in range(len(thetas)-1):
			A = 0
			B = i+1
			if i+2 == len(thetas):
				C = 1
			else:
				C = i+2
			faces.append((A,B,C))

	flatverts = len(verts)

	# Cylinder:
	for i in range(len(ls)):
		xs = radius*N.cos(thetas)
		ys = radius*N.sin(thetas)
		zs = N.ones(len(thetas))*ls[i]
		for j in range(len(xs)):
			verts.append((xs[j], ys[j], zs[j]))
	for i in range(len(verts)-flatverts-angbins):
		A = i+flatverts
		B = i+1+flatverts
		if (i+1)%angbins:
			C = i+angbins+flatverts
			D = i+1+angbins+flatverts
		else:
			C = i-(angbins-1)+flatverts
			D = i+1-(angbins-1)+flatverts
		faces.append((A,B,C))
		faces.append((B,D,C))

	if capped:
		cylverts = len(verts)
		# Face:
		verts.append((0,0,ls[-1]))
		xs = radius*N.cos(thetas)
		ys = radius*N.sin(thetas)
		zs = ls[-1]
		for j in range(len(xs)):
			verts.append((xs[j], ys[j], zs))
		
		for i in range(len(thetas)-1):
			A = 0+cylverts
			B = i+1+cylverts
			if i+2 == len(thetas):
				C = 1+cylverts
			else:
				C = i+2+cylverts
			faces.append((A,B,C))

	return N.array(verts), N.array(faces)

def disc_verts_faces(radius, radius_in=0., radbins=10, angbins=24):
	'''
	Makes disc vertices and faces arrays to build stl geometries
	'''

	rads = N.linspace(radius_in, radius, radbins)
	thetas = N.linspace(0., 2.*N.pi, angbins)
	zs = 0.

	verts = []
	faces = []

	if radius_in == 0.:
		verts.append((0,0,0))
		for i in range(len(rads)-1):
			xs = rads[i+1]*N.cos(thetas)
			ys = rads[i+1]*N.sin(thetas)
			for j in range(len(xs)):
				verts.append((xs[j], ys[j], zs))
	else:
		for i in range(len(rads)):
			xs = rads[i]*N.cos(thetas)
			ys = rads[i]*N.sin(thetas)
			for j in range(len(xs)):
				verts.append((xs[j], ys[j], zs))

	for i in range(len(rads)-1):
		for j in range(len(thetas)-1):
			if (i==0) and (radius_in==0):
				A = 0
				B = i*len(thetas)+1
				if (i+2) == len(thetas):
					C = 1
				else:
					C = i+2
				faces.append((A,B,C))
			else:
				A = i*len(thetas)+j
				if (j+1) == len(thetas):
					B = i*len(thetas)+1
				else:
					B = A+1
				C = A + len(thetas)
				D = B + len(thetas)
				faces.append((A,B,C))
				faces.append((B,D,C))

	return N.array(verts), N.array(faces)

def rectangle_verts_faces(lx, ly, xbins, ybins):
	'''
	Makes rectangles vertices and faces arrays to build stl geometries
	'''
	xbins, ybins = xbins+1, ybins+1
	lxs, lys = N.linspace(-lx/2., lx/2., xbins), N.linspace(-ly/2., ly/2., ybins)

	verts = []
	faces = []

	for i in range(len(lxs)):
		for j in range(len(lys)):
			verts.append((lxs[i], lys[j], 0.))

	for i in range(len(lxs)-1):
		for j in range(len(lys)-1):
			A = i*ybins+j
			B = A+ybins
			C = A+1
			D = B+1
			faces.append((A,B,C))
			faces.append((B,D,C))

	return N.array(verts), N.array(faces)

def load_stl(stl_file):
	'''
	Load a STL file and creates a list of triangles.
	'''
	from stl import mesh
	data = mesh.Mesh.from_file(stl_file)
	triangles = []
	for t in range(len(data.vectors)):
		triangles.append(data.vectors[t])
	return N.array(triangles)

def make_stl(verts, faces, filename):
	'''
	Makes a STL file from a list of vertices and faces.
	'''
	from stl import mesh
	data = mesh.Mesh(N.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
	for i, f in enumerate(faces):
		for j in range(3):
			data.vectors[i][j] = verts[f[j],:]
	data.save(filename)

def stl_to_tracer_geom(triangles, option='polygon'):
	'''
	Arguments:
	- triangles: list of n triangle arrays, with each array having in axis 0 the axes and in axis 1 the coordinates
	- option: 'polygon' or 'triangle' for the geometry manager. Triangles are not faster than polygons so far...
	Returns:
	- geoms: list of geometry objects for Tracer
	- locs: list of location vectors for Tracer
	- rots: list of rotation matrices for Tracer
	'''
	geoms, locs, rots = [], [], []
 
	for i, triangle in enumerate(triangles):
		A, B, C = triangle
		normal = get_plane_normal(B-A, C-B)
		phi_n = N.arctan2(normal[1], normal[0])
		th_n = N.arccos(normal[2])
		back_rot = N.dot(roty(-th_n), rotz(-phi_n))[:3,:3]
		Aa, Ba, Ca = A-A, B-A, C-A
		Ap, Bp, Cp = N.dot(back_rot, N.array([Aa, Ba, Ca]).T).T
		profile = N.array([[Ap[0], Bp[0], Cp[0]], [Ap[1], Bp[1], Cp[1]]])
		# Make planar profile
		if option == 'polygon':
			geoms.append(FlatSimplePolygonGM(profile))
		if option == 'triangle':
			tri_profile = N.zeros((3,2))
			tri_profile[:2] = profile[:,1:]
			geoms.append(TriangularFace(tri_profile))
		locs.append(A)
		phi_n = N.arctan2(normal[1],normal[0])
		th_n = N.arccos(normal[2])
		rots.append(N.dot(rotz(phi_n), roty(th_n))[:3,:3])

	return geoms, locs, rots

def make_stl_tracer_object(triangles, optics, optics_args, option='polygon'):
	'''
	Takes in triangle vertices an optics callable or function name and optics arguments and builds the assembled object in Tracer format.
	'''
	from tracer import optics_callables
	surfs, bounds = [], []
	geoms, locs, rots = stl_to_tracer_geom(triangles, option=option)

	for i, geom in enumerate(geoms):
		opt = optics(**optics_args)
		surf = Surface(geometry=geom, optics=opt, location=locs[i], rotation=rots[i], fixed_color=(0,1,0))
		surfs.append(surf)
		bounds.append(BoundaryBox(AABB(triangles[i].T)))

	return AssembledObject(surfs=surfs, bounds=bounds)

def load_stl_into_tracer(stl_file, optics, optics_args, option='polygon'):
	triangles = load_stl(stl_file)
	return make_stl_tracer_object(triangles, optics, optics_args, option)


if __name__ == "__main__":
	pass
