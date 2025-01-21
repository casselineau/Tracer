# Implements a tracer engine class

import numpy as N
from tracer.ray_bundle import RayBundle, concatenate_rays
from tracer.trace_tree import RayTree
from tracer.accel_tree import KdTree
import gc
import time
import logging

class TracerEngine():
	"""
	Tracer Engine implements that actual ray tracing. It keeps track of the number
	of objects, and determines which rays intersected which object.
	"""
	def __init__(self, parent_assembly, loglevel = logging.DEBUG):
		"""
		Arguments:
		parent_assembly - the highest level assembly
		
		Attributes:
		_asm - the Assembly instance containing the model to trace through.
		"""
		self._asm = parent_assembly
		self.loglevel = loglevel

	def intersect_ray(self, bundle, surfaces, surf_relevancy):
		"""
		Finds the first surface intersected by each ray.
		
		Arguments:
		bundle - the RayBundle instance holding incoming rays.
		surfaces - the list of surfaces in the main assembly
		surf_relevancy - a binarry array of (nsurf, nrays) shape that states whether surface s needs to test intersection with ray r.
		
		Returns:
		earliest_surf - a nray length list with the indices of teh first surface intersected.
		surf_relevancy - a binary array of (nsurf, nrays) shape that states whether surface s needs to test intersection with ray r.
		"""
		nrays = bundle.get_num_rays()
		rays_mins = N.ones(nrays)*N.inf
		earliest_surf = -1*N.ones(nrays, dtype=int)

		# Bounce rays off each object
		for surf_num, surf in enumerate(surfaces):
			rays_rel = N.copy(surf_relevancy[surf_num])
			if rays_rel.any():
				# If all rays are relevant all the bundle goes into in_rays
				if rays_rel.all():
					in_rays = bundle
				else: # ...Otherwise, the bundle inherits the relevant only
					in_rays = bundle.inherit(rays_rel)   
			else: # if not a relevant surface for intersection, next surface:
				continue
				
			# Fills the stack assigning rays to surfaces hit.
			surf_stack = surf.register_incoming(in_rays) # find intersections using surface geometry manager
			surf_stack[surf_stack==0.] = N.inf # if intersection distance is 0, set this as not an intersection
			earlier_hit = surf_stack < rays_mins[rays_rel] # compare intersection distance with this surface with rays_mins relevant for this surface to find if the currently detected hit is the earliest.
			if earlier_hit.any():
				rays_rel[rays_rel] = earlier_hit
				rays_mins[rays_rel] = surf_stack[earlier_hit]
				earliest_surf[rays_rel] = surf_num
		return earliest_surf, surf_relevancy

	def intersect_ray_accel_seq(self, bundle, surfaces, surf_relevancy):
		"""
		Finds the first surface intersected by each ray.
		
		Arguments:
		bundle - the RayBundle instance holding incoming rays.
		surfaces - the list of surfaces in the main assembly
		surf_relevancy - a lots of nsurf elements. In each s element of this main list, a list of nseqs elements describing sequential ray intersection tests. In each seq element of that list, the indices of the rays that shoudl be intersected.
		
		Returns:
		earliest_surf - a nray length list with the indices of teh first surface intersected.
		owned_rays - a list of nsurf elements witth, in each element, the list fo indiced of the rays that went to be registered for intersection in the surface.		
		
		Esplaination:
		We check, in order of intersections, all the rays that are relevant to each surface, as given in surfs-relevancy, thanks to the progressive traversal of the acceleration tree.
		First we loop through all surfaces for first order intersection, then second order etc until no surface has rays left to simulate. After each order, ray mins are compard to find the first surface hit. 
		If there is a hit, we prevent this ray form being registered for intersection in following iterations (this is where we should gain intersection time). 
		"""
		nrays = bundle.get_num_rays()
		rays_mins = N.ones(nrays)*N.inf
		earliest_surf = -1*N.ones(nrays, dtype=int)
		rays_not_done = N.ones(nrays, dtype=bool)
		hits = N.zeros(nrays, dtype=bool)
		owned_rays = [[] for _ in range(len(surfaces))]
		# length of intersection sequences per surface:
		seqs = [len(s) for s in surf_relevancy]

		# Maximum length
		seqmax = N.amax(seqs)		
		# Loop through sequence.
		for seq in range(seqmax):
			# Loop through surfaces
			for surf_num, surf in enumerate(surfaces):
				if seq<seqs[surf_num]:
					rays_in_seq = [r for r in surf_relevancy[surf_num][seq] if rays_not_done[r]]
					
					if len(rays_in_seq)==0:
						continue
					else:
						# the bundle inherits the relevant only
						in_rays = bundle.inherit(rays_in_seq)
						owned_rays[surf_num].extend(rays_in_seq)
						
						# Bounce rays off each object
						# Fills the stack assigning rays to surfaces hit.
						surf_stack = surf.register_incoming(in_rays) # find intersections using surface geometry manager
						surf_stack[surf_stack==0.] = N.inf # if intersection distance is 0, set this as not an intersection
						earlier_hit = surf_stack < rays_mins[rays_in_seq] # compare intersection distance with this surface with rays_mins relevant for this surface to find if the currently detected hit is the earliest.
						if earlier_hit.any():
							hit_idx = [r for i,r in enumerate(rays_in_seq) if earlier_hit[i]]
							rays_mins[hit_idx] = surf_stack[earlier_hit]
							earliest_surf[hit_idx] = surf_num
							hits[hit_idx] = True
			# With all surfaces done for this sequence, update the list of rays that are done.
			rays_not_done[hits] = False
			hits[:] = False
		return earliest_surf, owned_rays

	def ray_tracer(self, bundle, reps=100, min_energy=1e-10, tree=True, accel=False, Kd_Tree=None, **kwargs):
		"""
		Creates a ray bundle or uses a reflected ray bundle, and intersects it
		with all objects, uses intersect_ray(). Based on the intersections,
		generates an outgoing ray in accordance with way the incoming ray
		reflects or refracts off any surfaces.
		
		Arguments:
		bundle - the initial incoming bundle
		reps - stop iteration after this many ray bundles were generated (i.e. 
			after the original rays intersected some surface this many times).
		min_energy - the minimum energy the rays have to have continue tracking
			them; rays with a lower energy are discarded. A float.
		tree - a list  used for track parent rays and child rays. Each element
			of the list is a ray bundle created after one iteration of the 
			tracer. Each bundle contains an array listing the parent ray in the
			previous bundle (see ray_bundle.py). When a ray branches, the child
			rays point back to the same index representing that same parent ray.
			Otherwise, the index of each ray points to the ray in the previous
			branch.tree. If True, register each bundle in self.tree, otherwise only
			register the last bundle.
		accel - Enables Kd_Tree acceleration if True.
		Kd_Tree - pre-determined acceleration structure given to avoid the tree buidling.
		
		Returns: 
		A tuple containing an array of vertices and an array of the the direcitons
		of the last outgoing raybundle (note that the vertices of the new bundle are the 
		intersection points of the previous incoming bundle)
		
		NB: the order of the rays within the arrays may change, but they are tracked
		by the ray tree
		"""

		self.reps = reps
		self.minener = min_energy
		self.tree = RayTree()
		bund = bundle
		if tree is True:
			self.tree.append(bund)

		# A list of surfaces and their matching objects:
		surfaces = self._asm.get_surfaces()
		objects = self._asm.get_objects()

		num_surfs = len(surfaces)
		num_rays = bund.get_num_rays()

		if accel:
			if Kd_Tree is None:
				max_depth = 8+1.3*N.log(num_surfs)
				logging.log(self.loglevel, 'Maximum Kd tree depth %i'%max_depth)
				min_leaf = 1
				fast = False
				if accel == 'fast':
					fast = True

				if 'min_leaf' in kwargs:
					self.Kd_Tree = KdTree(self._asm, max_depth, loglevel = self.loglevel, fast = fast, **kwargs)
				else:
					self.Kd_Tree = KdTree(self._asm, max_depth, loglevel = self.loglevel, fast = fast, min_leaf = 1, **kwargs)
			else:
				self.Kd_Tree = Kd_Tree
		else: # these are legacy arrays from the original code from Y. Meller. The objective is that the objects, through their own_rays and surface_for_next_iteration methods, drive the next bundle restrictions. It requires object-specific interaction managementm which is less relevant if we use a generic binary tree type acceleration.
			surfs_per_obj = [len(obj.get_surfaces()) for obj in objects]
			surfs_until_obj = N.hstack((N.r_[0], N.add.accumulate(surfs_per_obj)))
			surf_ownership = N.repeat(N.arange(len(objects)), surfs_per_obj)
			ray_ownership = -1*N.ones(num_rays)
			surfs_relevancy = N.ones((num_surfs, num_rays), dtype=bool)

		for i in range(reps):
			t0 = time.time()
			if accel:
				if accel == 'lightweight':
					any_inter, surfs_relevancy = self.Kd_Tree.traversal(bund, lightweight=True)
					t0 = time.time()
					if any_inter:
						front_surf, owned_rays = self.intersect_ray_accel_seq(bund, surfaces, surfs_relevancy)
				else:
					any_inter, surfs_relevancy = self.Kd_Tree.traversal(bund)
					t0 = time.time()
					if any_inter:
						front_surf, owned_rays = self.intersect_ray(bund, surfaces, surfs_relevancy)
			else:
				front_surf, owned_rays = self.intersect_ray(bund, surfaces, surfs_relevancy)#self.intersect_ray_old(bund, surfaces, objects, surf_ownership, ray_ownership, surfs_relevancy)
				
			outg = []
			record = []
			weak_ray_pos = []
			if not accel:
				out_ray_own = []
				new_surfs_relevancy = []

			for surf_idx in range(num_surfs):

				intersections = front_surf[owned_rays[surf_idx]] == surf_idx # -1 are excluded automagically here
				if not any(intersections):
					surfaces[surf_idx].done()
					continue
					
				if accel == 'lightweight': # as we do sequential ray-trace, we need to give the full intersected bundle back to the surface to be able to select the right rays and calculate the outgoing rays. Ideally this is done at the surface level automatically in the future...
					surfbun = bund.inherit(selector=owned_rays[surf_idx])
					surfaces[surf_idx].update_current_bundle(surfbun)
				surfaces[surf_idx].select_rays(N.nonzero(intersections)[0])
				new_outg = surfaces[surf_idx].get_outgoing()
				
				# Fix parent indexing to refer to the full original bundle:
				if accel == 'lightweight':
					parents = N.array(owned_rays[surf_idx])[new_outg.get_parents()]
				else:
					parents = N.nonzero(owned_rays[surf_idx])[0][new_outg.get_parents()]
				new_outg.set_parents(parents)

				# Add to record before culling low_energy rays
				record.append(new_outg)

				# Delete rays with negligible energies
				delete = new_outg.get_energy() <= min_energy
				weak_ray_pos.append(delete)
				if delete.any():
					new_outg = new_outg.delete_rays(N.nonzero(delete)[0])
				surfaces[surf_idx].done()

				# Aggregate outgoing bundles from all the objects
				outg.append(new_outg)

				if not accel:
					# Add new ray-ownership information to the total list:
					obj_idx = surf_ownership[surf_idx]
					surf_rel_idx = surf_idx - surfs_until_obj[obj_idx]
					object_owns_outg = objects[obj_idx].own_rays(new_outg, surf_rel_idx)
					out_ray_own.append(N.where(object_owns_outg, obj_idx, -1))

					# Add new surface-relevancy information, saying which surfaces
					# of the full list of surfaces must be checked next. This is
					# somewhat memory-intensize and requires optimization.
					surf_relev = N.ones((num_surfs, new_outg.get_num_rays()), dtype=bool)
					surf_relev[surf_ownership == obj_idx] = \
						objects[obj_idx].surfaces_for_next_iteration(new_outg, surf_rel_idx)
					new_surfs_relevancy.append(surf_relev)					

			bund = concatenate_rays(outg)

			if tree:
				# stores parent branch for purposes of ray tracking
				record = concatenate_rays(record)
				if record.get_num_rays() != 0:
					weak_ray_pos = N.hstack(weak_ray_pos)
					record = bund + record.inherit(N.nonzero(weak_ray_pos)[0])
					self.tree.append(record)
				#gc.collect() # This was found useful to avoid memory error when using large bundles and/or broadband sources.
				# This is not useful in Python3
			if bund.get_num_rays() == 0:
				# All rays escaping
				logging.log(self.loglevel, 'Ray bundle depleted')
				break

			t1 = time.time()-t0
			if not accel:
				surfs_relevancy = N.hstack(new_surfs_relevancy)
			else:
				logging.log(self.loglevel, f'trace time {t1} s')

		if not tree:
			# Save only the last bundle. Don't bother moving weak rays to end.
			record = concatenate_rays(record)
			self.tree.append(record)
		if bund.get_num_rays() != 0:
			logging.warning(f'{bund.get_num_rays()} rays left at the end of the simulation')
			logging.warning(f'Remaining energy in last bundle: {N.sum(bund.get_energy())/N.sum(bundle.get_energy())*100.}%')
		return bund.get_vertices(), bund.get_directions()


# vim: ts=4
