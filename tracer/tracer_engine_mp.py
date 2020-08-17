import time
import numpy as N
from pathos.multiprocessing import ProcessingPool as Pool
from tracer.sources import *
from tracer.tracer_engine import *
from tracer.assembly import *
from copy import copy

class TracerEngineMP(TracerEngine):
	'''
	Famework for multi-processing using the tracer engine as is. 
	Requires pathos: https://github.com/uqfoundation/pathos

	Inheritance is broken by the multiprocessing pool and rebuilt on the tree and self._asm
	The original assembly needs to be reallocated after the simulation to be able to get the values stored in the optical managers with previously defined objects.

	Not the cleanest or finest implementation. Could be blended with the original engine and use the very same api. It works.
	'''
	def multi_ray_sim(self, sources, procs=1, minener=1e-10, reps=1000, tree=True):
		self.minener = minener # minimum energy threshold
		self.reps = reps # stop iteration after this many ray bundles were generated (i.e. 
					# after the original rays intersected some surface this many times).
		self.tree_switch = tree
		# The multiprocessing raytracing method to call from the original engine.
		if len(sources) != procs:
			raise Exception('Number of sources and processors do not agree')

		# Creates a pool of processes and makes them raytrace one different source each. The resm list returned is a list of copies of the original engine post raytrace.
		timetrace = time.clock()
		pool = Pool(processes=procs)
		def trace(source):
			self.ray_tracer(source, self.reps, self.minener, self.tree_switch)
			return self
		resm = pool.map(trace, sources)

		del pool
		
		timetrace = time.clock() - timetrace
		#print 'Raytrace time: ', timetrace,'s'

		timepost = time.clock()

		# New general tree:
		for eng in range(len(resm)):

			# Get and regroup results in one tree and assembly only:
			if eng == 0 : # Initialise with the first engine
				if self.tree_switch == True:
					self.tree =  resm[eng].tree
				self._asm = resm[eng]._asm
			else:
				if self.tree_switch == True:
					eng_bunds = resm[eng].tree._bunds
					for b in range(len(eng_bunds)):
						if b > 0: # if it is not the starting bundle (emanating from the source) add to the parents indices according to the size of the general tree bundle size.
							eng_bunds[b]._parents = eng_bunds[b]._parents+next_parents_adjust
						if b == len(self.tree._bunds): # If the bundle number is over the existing limit in the general tree, append it to increase the general tree size.
							self.tree.append(eng_bunds[b])
						else: # concatenate the bundle with its existing counterpart in the general tree
							next_parents_adjust = self.tree._bunds[b].get_num_rays() # to adjust the index of parents before changing the total size of the general tree bundle.
							self.tree._bunds[b] = concatenate_rays([self.tree._bunds[b], eng_bunds[b]])

				# Next loop is to get the optics callable objects and copy regroup their values without asumptions about what they are.
				subas_engine = resm[eng]._asm.get_assemblies()
				if len(subas_engine):
					for a in range(len(subas_engine)):
						objs_subas = subas_engine[a].get_local_objects()
						for o in range(len(objs_subas)):
							surfs_object = objs_subas[o].get_surfaces()
							for s in range(len(surfs_object)):
								for k in list(surfs_object[s]._opt.__dict__.keys()):
									if k != '_opt':
										if hasattr(surfs_object[s]._opt.__dict__[k], '__len__'):
											[self._asm._assemblies[a]._objects[o].surfaces[s]._opt.__dict__[k].append(q) for q in surfs_object[s]._opt.__dict__[k]]

				objs_engine = resm[eng]._asm.get_local_objects()
				if len(objs_engine):
					for o in range(len(objs_engine)):
						surfs_object = objs_engine[o].get_surfaces()
						for s in range(len(surfs_object)):
							for k in list(surfs_object[s]._opt.__dict__.keys()):
								if k != '_opt':
									[self._asm._objects[o].surfaces[s]._opt.__dict__[k].append(q) for q in surfs_object[s]._opt.__dict__[k]]

		# We need the next part to reshape everything to the right array format.
		asm_subas = self._asm.get_assemblies()
		if len(asm_subas):
			for a in range(len(asm_subas)):
				objs_subas = asm_subas[a].get_local_objects()
				for o in range(len(objs_subas)):
					surfs_object = objs_subas[o].get_surfaces()
					for s in range(len(surfs_object)):
						for k in list(surfs_object[s]._opt.__dict__.keys()):
							if k != '_opt':
								if hasattr(surfs_object[s]._opt.__dict__[k], '__len__'):
									if len(surfs_object[s]._opt.__dict__[k])>0:
										if k == '_absorbed':
											self._asm._assemblies[a]._objects[o].surfaces[s]._opt.__dict__[k] = [N.hstack(self._asm._assemblies[a]._objects[o].surfaces[s]._opt.__dict__[k])]
										else:	
											self._asm._assemblies[a]._objects[o].surfaces[s]._opt.__dict__[k] = [N.column_stack(self._asm._assemblies[a]._objects[o].surfaces[s]._opt.__dict__[k])]
								#else:
								#	self._asm._assemblies[a]._objects[o].surfaces[s]._opt.__dict__[k] = []
	
		asm_objs = self._asm.get_local_objects()
		if len(asm_objs):
			for o in range(len(asm_objs)):
				surfs_object = asm_objs[o].get_surfaces()
				for s in range(len(surfs_object)):
					for k in list(surfs_object[s]._opt.__dict__.keys()):
						if k != '_opt':
							if hasattr(surfs_object[s]._opt.__dict__[k], '__len__'):
								if len(surfs_object[s]._opt.__dict__[k])>0:
									if k == '_absorbed':
										self._asm._objects[o].surfaces[s]._opt.__dict__[k] = [N.hstack(self._asm._objects[o].surfaces[s]._opt.__dict__[k])]
									else:	
										self._asm._objects[o].surfaces[s]._opt.__dict__[k] = [N.column_stack(self._asm._objects[o].surfaces[s]._opt.__dict__[k])]
							#else:
							#	self._asm._objects[o].surfaces[s]._opt.__dict__[k] = []	
						#print 'general assembly',' object ', o,' surface ', s,' number of rays: ', len(self._asm._objects[o].surfaces[s].get_optics_manager().get_all_hits()[0])
		del resm
		timepost2 = time.clock()-timepost
		#print 'Post processing reassociation time: ', timepost2,'s'

