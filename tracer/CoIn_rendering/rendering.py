#SOGUI_BINDING="SoQt"


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
		bgcol = coin.sbColor(.9,.9,.8) # if rgb: from 0 to 255, if rgba, values form 0 to 1
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

			color=(1-float(level)/lentree,0,0)

			no1 = coin.SoSeparator()

			ma1 = coin.SoMaterial()
			ma1.diffuseColor = color
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

	def show_rays(self, escaping_len=.02, max_rays=None, resolution=None, fluxmap=None, trans=False, vmin=None, vmax=None, bounding_boxes=None, only_rays=False):
		self.rays(escaping_len, max_rays, resolution)
		if not only_rays:
			self.geom(resolution, fluxmap, trans, vmin, vmax, bounding_boxes)
		self.show()

