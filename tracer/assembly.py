# Defines an assembly class, where an assembly is defined as a collection of AssembledObjects.

import operator
import numpy as N

from tracer.spatial_geometry import general_axis_rotation
from tracer.has_frame import HasFrame

class Assembly(HasFrame):
	"""
	Defines an assembly of objects or sub-assemblies.
	
	Attributes:
	_objects - a list of the objects the assembly contains
	_assemblies - a list of the sub assemblies the assembly contains
	"""
	def __init__(self, objects=None, subassemblies=None, location=None, rotation=None):
		"""
		Arguments:
		objects (optional) - a list of AssembledObject instances that are part
			of this assembly.
		subassemblies (optional) - a list of Assembly instances to be
			transformed together with this assembly.
		location, rotation - passed on to HasFrame.
		"""
		if objects is None:
			objects = []
		self._objects = objects

		if subassemblies is None:
			subassemblies = []
		self._assemblies = subassemblies
		
		HasFrame.__init__(self, location, rotation)

	def global_to_local(self, points):
		"""
		Transform a set of points in the global coordinates back into the frame
		used during tracing.
		
		Arguments:
		points - a 3 x n array for n 3D points
		
		returns:
		local - a 3 x n array with the respective points in local coordinates.
		"""
		proj = N.round(N.linalg.inv(self._temp_frame), decimals=9)
		return N.dot(proj, N.vstack((points, N.ones(points.shape[1]))))

	def get_local_objects(self):
		"""
		Get the list of objects belonging directly to this assembly, without
		querying the child assemblies.
		"""
		return self._objects

	def get_assemblies(self):
		return self._assemblies
	
	def get_objects(self):
		"""
		Generates a list of AssembledObject instances belonging to this assembly
		or its subassemblies.
		"""
		return N.hstack([[]+os for os in [asm.get_objects() for asm in self._assemblies] if os] + self._objects).tolist()
		
	def get_surfaces(self):  
		"""
		Generates a list of surface objects out of all the surfaces in the
		objects and subassemblies belonging to this assembly.
		
		The surfaces are guarantied to be in the order that each object returns
		them, and the objects are guarantied to be ordered the same as in 
		self.get_objects()
		"""
		surfaces = [surface for obj in self.get_objects() for surface in obj.get_surfaces()]
		return surfaces

	def add_object(self, object, transform=None):
		"""
		Adds an object to the assembly.
		
		Arguments: 
		objects - the AssembledObject to add
		transform - the transformation matrix (as an array object) that describes 
			the object in the coordinate system of the Assembly
		"""
		self._objects.append(object)
		if transform is not None:
			object.set_transform(transform)
		self.transform_children()

	def add_assembly(self, assembly, transform=None):
		"""Adds an assembly to the current assembly.
		
		Arguments:
		assembly - the assembly object to add
		transform - the transformation matrix (as an array object) that describes the 
			new assembly in the coordinate system of the current assembly
		"""

		self._assemblies.append(assembly)
		if transform is not None:
			assembly.set_transform(transform)
		self.transform_children()

	def set_rotation(self, rotation):
		"""
		A recursive version of the parent's set_rotation. Changes the rotation
		part of the assembly's transform, and updates the assembly's children's
		transform accordingly.
		
		Arguments:
		rotation - a 3x3 rotation matrix.
		"""
		HasFrame.set_rotation(self, rotation)
		self.transform_children()
	
	def set_location(self, location):
		"""
		A recursive version of the parent's set_rotation. Changes the location
		part of the assembly's transform, and updates the assembly's children's
		transform accordingly.
		
		Arguments:
		location - a 3-component location vector.
		"""
		HasFrame.set_location(self, location)
		self.transform_children()
	
	def set_transform(self, transform):
		HasFrame.set_transform(self, transform)
		self.transform_children()

	def transform_children(self, assembly_transform=N.eye(4)):
		"""
		Transforms the entire assembly
		
		Arguments:
		assembly_transform - the transformation into the parent assembly containing the 
			current assembly
		"""
		const_t = self.get_transform()
		for obj in self._assemblies + self._objects:
			obj.transform_children(N.dot(assembly_transform, const_t))
			
	def reset_all_optics(self):
		for s in self.get_surfaces():
			if hasattr(s.get_optics_manager(), 'reset'):
				s.get_optics_manager().reset()

	def get_scene_graph(self,resolution, fluxmap, trans, vmin, vmax, bounding_boxes):
		from pivy import coin
		n0 = coin.SoSeparator()
		n = self.get_scene_graph_transform()

		for obj in self._assemblies:
			n.addChild(obj.get_scene_graph(resolution, fluxmap, trans, vmin, vmax, bounding_boxes))

		boundaries = []
		for obj in self._objects:
			n.addChild(obj.get_scene_graph(resolution, fluxmap, trans, vmin, vmax))
			boundaries += obj.get_boundaries()

		if bounding_boxes:
			for b in boundaries:
				coords = []
				nedges = []
				if b:
					minp, maxp = b._AABB
					# base
					coords.extend([(minp[0], minp[1], minp[2]),\
								  (maxp[0], minp[1], minp[2]),\
								  (maxp[0], maxp[1], minp[2]),\
								  (minp[0], maxp[1], minp[2]),\
								  (minp[0], minp[1], minp[2])])
					nedges.append(5)
					# columns
					coords.extend([(minp[0], minp[1], minp[2]),\
								  (minp[0], minp[1], maxp[2]),\
								  (maxp[0], minp[1], minp[2]),\
								  (maxp[0], minp[1], maxp[2]),\
								  (maxp[0], maxp[1], minp[2]),\
								  (maxp[0], maxp[1], maxp[2]),\
								  (minp[0], maxp[1], minp[2]),\
								  (minp[0], maxp[1], maxp[2])])
					nedges.extend((2,2,2,2))
					# roof
					coords.extend([(minp[0], minp[1], maxp[2]),\
								  (maxp[0], minp[1], maxp[2]),\
								  (maxp[0], maxp[1], maxp[2]),\
								  (minp[0], maxp[1], maxp[2]),\
								  (minp[0], minp[1], maxp[2])])
					nedges.append(5)

					coords = N.reshape(N.ravel(coords), (-1,3))
					nedges = N.ravel(nedges)

					bb = coin.SoSeparator()

					ma1 = coin.SoMaterial()
					ma1.diffuseColor = (0.,0.,0.)
					bb.addChild(ma1)

					ds = coin.SoDrawStyle()
					ds.style = ds.LINES
					ds.lineWidth = 1
					bb.addChild(ds)

					co = coin.SoCoordinate3()
					co.point.setValues(0, len(coords), coords)
					bb.addChild(co)

					ls = coin.SoLineSet()
					ls.numVertices.setValues(0, len(nedges), nedges)
					bb.addChild(ls)

					n0.addChild(bb)
		n0.addChild(n)
		return n0
		
# vim: ts=4
