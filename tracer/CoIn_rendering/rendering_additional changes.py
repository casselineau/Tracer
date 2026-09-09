#SOGUI_BINDING="Quarter"

from pivy import coin
from pivy.sogui import *

import numpy as N
import sys

class Renderer():
	'''	__________________________________________________________________________________________________
	Rendering:

	Renders the scene. Offers the option to highlight specific rays according to the number of times they have been 	reflected. 
	/!\ Information needs updating /!\

Reference:
[1] The inventor Mentor, Josie Wernecke (https://webdocs.cs.ualberta.ca/~graphics/books/mentor.pdf)	__________________________________________________________________________________________________

	'''
	def __init__(self, sim):
		self.sim = sim

		# Scene axis label
		length = 1
		self.r = coin.SoSeparator()
		st = coin.SoDrawStyle()
		st.lineWidth=3
		self.r.addChild(st)
		data = {'x':(1,0,0), 'y':(0,1,0), 'z':(0,0,1)}
		tra = coin.SoTransparencyType()

		#text_ref = coin.SoText2()
		#self.r.addChild(text_ref)
		#text_ref.string = 'Tracer rendering'

		for k in data:

			vx,vy,vz = data[k]
			vec = (length*vx, length*vy, length*vz)		

			s1 = coin.SoSeparator()
			la = coin.SoLabel()
			la.label = k
			s1.addChild(la)
			tr1 = coin.SoTranslation()
			tr1.translation = vec
			s1.addChild(tr1)
			self.r.addChild(s1)		

			s2 = coin.SoSeparator()
			tr2 = coin.SoTransform()
			tr2.translation.setValue(data[k])
			s2.addChild(tr2)
			matxt = coin.SoMaterial()
			matxt.diffuseColor = data[k]
			s2.addChild(matxt)
			txaxis = coin.SoText2()	  
			txaxis.string = k	   
			s2.addChild(txaxis)
			self.r.addChild(s2)

			ma = coin.SoMaterial()
			ma.diffuseColor = data[k]
			self.r.addChild(ma)

			co = coin.SoCoordinate3()
			co.point.setValues(0,2,[(0,0,0),vec])
			self.r.addChild(co)

			ls = coin.SoLineSet()
			ls.numVertices.setValues(0,1,[2])
			self.r.addChild(ls)

	def show(self):
		win = SoGui.init(sys.argv[0])
		viewer = SoGuiExaminerViewer(win)
		bgcol = coin.SbColor(.9*255,.9*255,.8*255)
		viewer.setBackgroundColor(bgcol)
		viewer.quarterwidget.sorendermanager.getGLRenderAction().setTransparencyType(coin.SoTransparencyType.DELAYED_BLEND)
		viewer.setSceneGraph(self.r)
		viewer.setTitle("Examiner Viewer")
		viewer.show()
		SoGui.mainLoop()

	def geom(self, resolution=None, fluxmap=None, trans=False, vmin=None, vmax=None, bounding_boxes=None):
		"""
		Method to draw the geometry of the scene to a Coin3D scenegraph.
		"""

		ls1 = coin.SoDirectionalLight()
		self.r.addChild(ls1)
		ls1.direction = (0,0,1)
		ls1.color=(1,1,1)

		ls2 = coin.SoDirectionalLight()
		self.r.addChild(ls2)
		ls2.direction = (0,1,0)
		ls2.color=(1,1,1)

		ls3 = coin.SoDirectionalLight()
		self.r.addChild(ls3)
		ls3.direction = (1,0,0)
		ls3.color=(1,1,1)

		ls4 = coin.SoDirectionalLight()
		self.r.addChild(ls4)
		ls4.direction = (0,0,-1)
		ls4.color=(1,1,1)

		ls5 = coin.SoDirectionalLight()
		self.r.addChild(ls5)
		ls5.direction = (0,-1,0)
		ls5.color=(1,1,1)

		ls6 = coin.SoDirectionalLight()
		self.r.addChild(ls6)
		ls6.direction = (-1,0,0)
		ls6.color=(1,1,1)

		self.r.addChild(self.sim._asm.get_scene_graph(resolution, fluxmap, trans, vmin, vmax, bounding_boxes))

	def show_geom(self, resolution=None, fluxmap=None, trans=False, vmin=None, vmax=None, bounding_boxes=None):
		self.geom(resolution, fluxmap, trans, vmin, vmax, bounding_boxes)
		self.show()

	def rays(self, escaping_len=.02, max_rays=None, resolution=None):
		"""
		Method to draw the rays to a Coin3D scenegraph. Needs to be called after a raytrace has been performed.
		"""

		tree = self.sim.tree
		no = coin.SoSeparator()
		
		# loop through the reflection sequences?
		co = [] # regular lines
		pos = [] # 2D level text position

		lentree = tree.num_bunds()

		for level in range(lentree):

			start_rays = tree[level]
			sv = start_rays.get_vertices()
			sd = start_rays.get_directions()
			se = start_rays.get_energy()
			if max_rays is None:
				max_rays = len(se)
			else:
				max_rays = N.amin([max_rays, len(se)])

			if tree.num_bunds() ==1:
				parents = []
				shown = N.arange(max_rays)
			elif level == tree.num_bunds() - 1:
				parents = []
				shown = N.nonzero(parents_of_shown)[0]
			else:
				end_rays = tree[level + 1]
				ev = end_rays.get_vertices()
				parents = end_rays.get_parents()
				if level==0:
					shown = N.random.choice(len(se), size=max_rays, replace=False)
					parents_of_shown = N.in1d(parents, shown)
				else:
					shown = N.nonzero(parents_of_shown)[0]
					parents_of_shown = N.in1d(parents, shown)

			# loop through individual rays in the selection
			for ray in shown:
				if se[ray] <= self.sim.minener:
					# ignore rays with starting energy smaller than energy cutoff
					continue
		
				if ray in parents:
					# Has a hit on another surface
					first_childs = N.nonzero(ray == parents)[0]
					c1 = sv[:,ray]
					for cs in first_childs:
						c2 = ev[:,cs]
						# if the ray is not on the direction, it is a boundary condition affected ray and should not be represented
						dir_vecs = N.round((c2-c1)/N.sqrt(N.sum((c2-c1)**2)), decimals=6)
						dir_ray = N.round(sd[:,ray], decimals=6)
						if (dir_vecs == dir_ray).all():
							co += [(c1[0],c1[1],c1[2]), (c2[0],c2[1],c2[2])]

				else:
					l = escaping_len
					# Escaping ray.
					c1 = sv[:,ray]
					c2 = sv[:,ray] + sd[:,ray]*l
					co += [(c1[0],c1[1],c1[2]), (c2[0],c2[1],c2[2])]

			color=(int((1.-float(level)/lentree)*255),0,0)
			
			no1 = coin.SoSeparator()

			ma1 = coin.SoMaterial()
			ma1.diffuseColor = coin.SbColor(color)
			no1.addChild(ma1)

			ds = coin.SoDrawStyle()
			ds.style = ds.LINES
			ds.lineWidth = 1
			no1.addChild(ds)

			coor = coin.SoCoordinate3()
			coor.point.setValues(0, len(co), co)
			no1.addChild(coor)

			ls = coin.SoLineSet()
			ind = [2] * int(len(co)/2)
			ls.numVertices.setValues(0, len(ind), ind)
			no1.addChild(ls)

			no.addChild(no1)

		self.r.addChild(no)

		# ---- store segments for exporting as OBJ ----
		self._ray_segments = []
		for i in range(0, len(co), 2):
			p1 = co[i]
			p2 = co[i+1]
			self._ray_segments.append((p1, p2))


	def show_rays(self, escaping_len=.02, max_rays=None, resolution=None, fluxmap=None, trans=False, vmin=None, vmax=None, bounding_boxes=None, only_rays=False, create_only=False):
		self.rays(escaping_len, max_rays, resolution)
		if not only_rays:
			self.geom(resolution, fluxmap, trans, vmin, vmax, bounding_boxes)

		if not create_only:
			self.show()


	def export_inventor(self, filename="scene.iv"):
		"""
		Export the current Coin3D scene graph (self.r) to an Open Inventor file.

		Call geom()/rays()/show_rays() first so the geometry you care about
		is already attached to self.r.
		"""
		wa = coin.SoWriteAction()
		out = wa.getOutput()
		if not out.openFile(filename):
			raise IOError("Could not open %s for writing" % filename)
		# ASCII is easier to inspect and many tools accept it
		out.setBinary(False)
		wa.apply(self.r)
		out.closeFile()

	def export_rays_as_obj(self, filename="rays.obj"):
		"""
		Export all ray segments as an OBJ file containing only line (l) primitives.

		Requires that rays() or show_rays() has been called before, so that
		self._ray_segments is populated.
		"""
		if not hasattr(self, "_ray_segments"):
			raise RuntimeError(
				"Ray segments not available. Call rays() or show_rays() before export_rays_as_obj()."
			)

		with open(filename, "w") as f:
			f.write("# OBJ export of ray segments from Renderer\n")

			vidx = 1
			for p1, p2 in self._ray_segments:
				x1, y1, z1 = p1
				x2, y2, z2 = p2
				f.write(f"v {x1} {y1} {z1}\n")
				f.write(f"v {x2} {y2} {z2}\n")
				f.write(f"l {vidx} {vidx+1}\n")
				vidx += 2

	def tessellate_scene(self):
		"""
		Replace primitive shapes in self.r with tesselated IndexedFaceSet geometry.
		Works for SoSphere, SoCylinder, SoCone, SoCube.
		After running this, export_vrml() or export() will produce
		real triangle meshes Blender can import.
		"""
		from pivy.coin import (
			SoSearchAction,
			SoVertexProperty, SoIndexedFaceSet
		)

		# Step 1: find all shapes
		search = SoSearchAction()
		search.setType(coin.SoShape.getClassTypeId())
		search.setInterest(SoSearchAction.ALL)
		search.apply(self.r)
		results = search.getPaths()

		if results is None:
			return

		# Step 2: for each shape, tessellate using CallbackAction
		for path in results:
			shape = path.getTail()

			# CallbackAction collects triangles
			va = []
			ia = []

			cb = coin.SoCallbackAction()

			def triangle_cb(userdata, action, v1, v2, v3):
				p1 = v1.getPoint()
				p2 = v2.getPoint()
				p3 = v3.getPoint()

				base = len(va)
				va.extend([p1, p2, p3])
				ia.extend([base, base+1, base+2])

			cb.addTriangleCallback(shape.getTypeId(), triangle_cb, None)
			cb.apply(shape)

			if not va:
				continue

			# Build face set
			vp = SoVertexProperty()
			vp.vertex.setValues(0, len(va), va)

			fs = SoIndexedFaceSet()
			fs.vertexProperty = vp
			fs.coordIndex.setValues(0, len(ia), ia)

			parent = path.getNodeFromTail(1)
			idx = parent.findChild(shape)
			parent.replaceChild(idx, fs)



	def export_obj_mesh(self, filename="scene.obj"):
		"""
		Export all Coin shapes in self.r as a single triangulated OBJ mesh.

		This uses SoCallbackAction + triangle callbacks to grab whatever
		Coin would actually render (spheres, cylinders, cubes, etc.),
		and writes them out as vertices/faces.
		"""

		cb = coin.SoCallbackAction()

		vertices = []      # list of (x, y, z)
		faces = []         # list of (i1, i2, i3), 1-based OBJ indices
		index_map = {}     # maps (x, y, z) -> index

		def get_index(pt):
			key = (pt[0], pt[1], pt[2])
			idx = index_map.get(key)
			if idx is None:
				idx = len(vertices) + 1      # OBJ is 1-based
				vertices.append(key)
				index_map[key] = idx
			return idx

		def triangle_cb(userdata, action, v1, v2, v3):
			# v1, v2, v3 are SoPrimitiveVertex
			p1 = v1.getPoint()
			p2 = v2.getPoint()
			p3 = v3.getPoint()

			i1 = get_index(p1)
			i2 = get_index(p2)
			i3 = get_index(p3)

			faces.append((i1, i2, i3))

		# Register callback for *all* shapes
		cb.addTriangleCallback(coin.SoShape.getClassTypeId(), triangle_cb, None)

		# Apply to the whole scene graph
		cb.apply(self.r)

		# Now write the OBJ
		with open(filename, "w") as f:
			f.write("# OBJ export from Coin/Pivy scene\n")
			for x, y, z in vertices:
				f.write(f"v {x} {y} {z}\n")
			f.write("o scene\n")
			for i1, i2, i3 in faces:
				f.write(f"f {i1} {i2} {i3}\n")

	
	def export_vrml(self, filename="scene.wrl"):
		action = coin.SoToVRML2Action()
		action.apply(self.r)
		vrml_root = action.getVRML2SceneGraph()

		wa = coin.SoWriteAction()
		out = wa.getOutput()
		if not out.openFile(filename):
			raise IOError("Could not open %s for writing" % filename)
		out.setBinary(False)
		wa.apply(vrml_root)
		out.closeFile()


	def export(self, filename, format=None):
		"""
		Small convenience wrapper.

		Examples:
			renderer.export("scene.iv")       # Open Inventor
			renderer.export("scene.wrl")       # VRML2
			renderer.export("rays.obj")       # OBJ with lines
		"""
		if format is None:
			ext = filename.split(".")[-1].lower()
		else:
			ext = format.lower()

		if ext in ("iv", "inventor"):
			self.export_inventor(filename)
		elif ext == "obj":
			self.export_obj_mesh(filename)
		elif ext == "wrl":
			self.export_vrml(filename)
		else:
			raise ValueError(f"Unknown export format: {ext}")
