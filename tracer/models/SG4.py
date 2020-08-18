'''
SG4 dish object to be used as optics model in a scene.
'''

import numpy as N
from tracer.surface import *
from tracer.quadric import *
from tracer.paraboloid import *
from tracer.optics_callables import *
from tracer.object import *
from tracer.assembly import *
from tracer.spatial_geometry import *

class SG4(Assembly):
	'''
	- dishDiameter: diameter of the dish concentrator in (m).
	- dishFocus: focal distance of the dish concentrator in (m).
	- absDish: absorptivity of the dish mirrors.
	- sigma: surface slope error of the outer layer of the dish concentrator in (rad).
	- dishDiameter_in: inner diameter of the dish in (m).
	- sigma_in: surface slope error of the inner layer of the dish concentrator in (rad).
	'''

	def __init__(self, dishDiameter, dishFocus, absMirrors, sigma, dishDiameter_in=20., sigma_in=1.95e-3):

		aperture_area = (dishDiameter/2.)**2.*N.pi
		effective_area = 489.

		self.dishDiameter = dishDiameter
		self.dishFocus = dishFocus
		self.absDish = 1.-(1.-absMirrors)*effective_area/aperture_area
		self.sigma = sigma

		Assembly.__init__(self, objects=None, subassemblies=None, location=None, rotation=None)

		DISH = AssembledObject(surfs=[Surface(ParabolicDishGM(self.dishDiameter, self.dishFocus), RealReflectiveReceiver(self.absDish, self.sigma))])
		DISH2 = AssembledObject(surfs=[Surface(ParabolicDishGM(dishDiameter_in, self.dishFocus), RealReflectiveReceiver(self.absDish, sigma_in))], transform=translate(z=0.0001))

		self.add_object(DISH)
		self.add_object(DISH2)

	def get_all_hits(self):
	
		surfs = self.get_surfaces()

		hits = []
		abs = []

		for i in range(len(surfs)):
			abs.append(surfs[i].get_optics_manager().get_all_hits()[0])
			hits.append(surfs[i].get_optics_manager().get_all_hits()[1])

		self.abs = N.hstack(abs)
		self.hits = N.hstack(hits)
		
		self.total_abs = N.sum(self.abs)

		return self.hits, self.abs

	

