import numpy as N

from tracer.surface import *
from tracer.quadric import *
from tracer.cone import *
from tracer.cylinder import *
from tracer.flat_surface import *
from tracer.assembly import *
from tracer.optics_callables import *
from tracer.object import *
from tracer.spatial_geometry import *
from tracer.sources import *
from tracer.tracer_engine_mp import *
from tracer.CoIn_rendering.rendering import *

#import matplotlib.pyplot as plt

import time

class RTVF():
	'''
	General class for view factor raytraces.
	num_rays - number of rays fired per bundle
	precision - confidence interval threshold on view factor values for each element and for the combination rule between all elements of the scene.
	precision_rec - independent confidence interval threshold on the reciprocity rule between all elements of the scene.
	'''
	def __init__(self, num_rays=10000, precision=0.01, precision_option='absolute', precision_rec=None):
		self.num_rays = num_rays
		self.precision = precision
		if precision_rec ==None:
			self.precision_rec = precision
		else:	
			self.precision_rec = precision_rec
		self.stdev = N.inf
		self.precision_option = precision_option

	def reset_opt(self):
		'''
		Optics reset script to be able to raytrace again the same object without having energy already stored on the surfaces.
		'''
		for S in self.A.get_surfaces():
			S.get_optics_manager().reset()
	
	def test_precision(self):
		'''
		Routine that gathers the results of the VF binning after a full geometry reaytrace and checks the standard deviations of: 
		1. The value of the actual VF_estimator (passes averaged VF) to check each VF matrix element on its independent variation
		2. The combination rule; taking as arbitrary average 0 in the standard deviation formula.
		Afterwards both standard deviation matrices are tested on the maximum self.precision value and the result is inserted in the progress matrix used in the while loop for each geometry. This progress matrix shuts off the raytrace on elements that already hit their precision target in order to speed up teh calculation. A minimum number of iterations is set in order to avoid unlucky breacks at the start of the routine due to statistical bias on low number of passes.
		'''

		r = N.vstack(self.ray_counts)
		p = N.vstack(self.p)
		p_1 = p-r

		# Areas
		Ai = N.ones(N.shape(self.VF_esperance))*N.vstack(self.areas)

		# Online weighted standard deviation calculation variable
		self.Qsum = self.Qsum + r*p_1/p*(self.VF-self.VF_esperance)**2.
		self.stdev_VF = 3.*N.sqrt(self.Qsum/(p-1.))/N.sqrt(p)

		# Expected value = weighted average
		self.VF_esperance = (self.VF_esperance*p_1+self.VF*r)/p

		# Reciprocity rule variables
		AiFij = self.VF_esperance*Ai
		AjFji = AiFij.T
		self.VF_reciprocity = N.abs(AiFij-AjFji)

		# Evaluation of the relative precision of the evaluation
		if self.precision_option == 'absolute':
			stdev_test = self.stdev_VF <= self.precision/2.

			tas = self.stdev_VF*Ai
			reciprocity_test = (tas+(tas.T)) <= self.precision_rec


		elif self.precision_option == 'relative':
			rel_stdev = (self.stdev_VF/self.VF_esperance)
			rel_stdev[N.isnan(rel_stdev)] = 0.
			stdev_test = rel_stdev <= self.precision # relative precision on the VFs

			tas = Ai*self.stdev_VF
			rel_rec = (tas+(tas.T))/AiFij
			rel_rec[N.isnan(rel_rec)] = 0.
			rel_rec[N.isinf(rel_rec)] = 0.

			reciprocity_precision = rel_rec <= self.precision_rec

			# Minimum precision condition for surfaces with negligible contribution: (this is to speed up the process when some surfaces have a small surface and a very small view factor to another one)
			minimum_AF_test = AiFij < N.vstack((self.precision_rec*N.amax(AiFij, axis=1)))

			reciprocity_test = N.logical_or(reciprocity_precision, minimum_AF_test)

		# Summation rule
		summ_test = N.abs(N.sum(self.VF_esperance, axis=1)-1.) < self.precision

		# Minimum precision condition for surfaces with negligible contribution: (this is to speed up the process when some surfaces have a small surface and a very small view factor to another one)
		#minimum_VF_test = AiFij < self.precision 
		# Simulation progress switch to determine which surfaces need to cast more rays
		self.progress = N.logical_not(N.logical_and(summ_test, N.logical_and(stdev_test, reciprocity_test)))

		#print 'Progress:', N.hstack(N.argwhere((self.progress==True).any(axis=1)))
		#print 'Stdev:', N.hstack(N.argwhere((stdev_test==False).any(axis=1)))
		#print 'Reciprocity:', N.hstack(N.argwhere((reciprocity_test==False).any(axis=1)))
		#print 'Summation:', N.hstack(N.argwhere((summ_test==False).any(axis=1)))
		'''
		for f in xrange(N.shape(AiFij)[0]):
			if (reciprocity_test[f]==False).any():
				print N.amax(AiFij[f,reciprocity_test[f]==False])/N.amax(AiFij[f])
		'''

class FONaR_RTVF(RTVF):
	'''
	FONaR view factor calculation class.
	Needs to manually add the aperture inthe child class.
	Imports directy a nice FONaR object with the binning scheme array etc. The receiver type is 'Surround' for the axisymmetrical concept and 'Polar' for a polar field facing cavity. This is important for the ray casting directions and handling of passive surfaces.
	'''
	def __init__(self, Assembly, binning_scheme, areas, num_rays=10000, precision=0.01, precision_option='absolute', precision_rec=None, receiver_type='Surround', procs=8):
		RTVF.__init__(self, num_rays, precision, precision_option, precision_rec)
		self.receiver_type = receiver_type
		self.binning_scheme = binning_scheme
		self.areas = areas
		
		self.t0 = time.clock()

		self.VF = N.zeros((N.shape(binning_scheme)[0], N.shape(binning_scheme)[0]))
		self.progress = N.ones(N.shape(self.VF), dtype=N.bool)

		# Standard deviation computation variables.
		self.VF_esperance = N.zeros(N.shape(self.VF))
		self.Qsum = N.zeros(N.shape(self.VF))
		self.stdev_VF = N.zeros(N.shape(self.VF))
		self.stdev_reciprocity = N.zeros(N.shape(self.VF))

		self.p = N.zeros(N.shape(self.VF)[0])

		self.ray_counts = N.ones(len(areas))*self.num_rays

		vf_tracer = TracerEngineMP(Assembly)
		vf_tracer.itmax = 1 # stop iteration after this many ray bundles were generated (i.e. after the original rays intersected some surface this many times).
		vf_tracer.minener = 1e-15 # minimum energy threshold
		stable_stats = N.zeros(N.shape(self.progress))
		stat_cond = 2

		while (self.progress==True).any() or (stable_stats<stat_cond).any():
			self.ray_counts = N.zeros(N.shape(self.VF)[0])
			tp = time.clock()
			for i in xrange(N.shape(self.VF)[0]):
				if (self.progress[i,:]==True).any() or (stable_stats[i,:]<stat_cond).any():
					disc = binning_scheme[i,1,1] == binning_scheme[i,1,0]
					up = binning_scheme[i,1,1] > binning_scheme[i,1,0]

					if receiver_type == 'Surround':
						apbottom = i==0
						apcyl = i==1
						aptop = i==2
						if i < 3:
							rays_in = True
						elif up:
							rays_in = False
						else:
							rays_in = True

					elif receiver_type == 'Polar':
						ap = i==0
						if up:
							rays_in = True
						else:
							rays_in = False

					S = []
					for pr in xrange(procs):
						source = self.gen_source(binning_scheme[i], num_rays, rays_in, procs)
						if disc:
							if i>0:
								outwards = binning_scheme[i,2,1] > binning_scheme[i,2,0]
								if receiver_type == 'Surround':		
									if aptop:
										source._directions = -source._directions
									elif outwards:
										if ~apbottom:
											source._directions = -source._directions
								if receiver_type == 'Polar':
									if ~outwards:
										source._directions = -source._directions
						S.append(source)

					vf_tracer.multi_ray_sim(S, procs=procs)
					self.A = vf_tracer._asm # due to multiprocessing inheritance break
					#if i%10==0.:
					#	print 'Surface ',i,'/', N.shape(self.VF)[0]
					'''
					if i == 0:#(N.shape(self.VF)[0]-(31+32)):# N.shape(self.VF)[0]-1:
						view = Renderer(vf_tracer)
						view.show_rays()
						#view.show_geom()
						stop
					#'''
					self.alloc_VF(i)
					self.ray_counts[i] = self.num_rays

			self.p += self.ray_counts
			self.test_precision()
			#N.set_printoptions(formatter={'float': '{: 0.2f}'.format})

			#print N.round(self.VF, decimals=2)
			print '		Progress:', N.sum(self.progress),'/', N.size(self.progress),',', N.sum(N.sum(self.progress, axis=1)>0),'/', N.shape(self.progress)[0],'; Pass duration:', time.clock()-tp, 's'

			stable_stats[self.progress==False] += 1

		t1=time.clock()-self.t0
		print '	VF calculation time:',t1,'s'


	def gen_source(self, ahr, num_rays, rays_in, procs):
		center = N.vstack([0,0,ahr[1,0]])
		if ahr[2,0] == ahr[2,1]:
			S = vf_cylinder_bundle(num_rays=num_rays, rc=ahr[2,0], lc=ahr[1,1]-ahr[1,0], center=center, direction=N.array([0,0,N.sign(ahr[1,1]-ahr[1,0])]), rays_in=rays_in, procs=procs, angular_span=ahr[0])
		elif ahr[1,0]==ahr[1,1]:
			S = solar_disk_bundle(num_rays=num_rays, center=center, direction=N.array([0,0,1]), radius=ahr[2,1], ang_range=N.pi/2., procs=procs, radius_in=ahr[2,0], angular_span=ahr[0])
		else:
			if ahr[1,1] > ahr[1,0]:
				if ahr[2,0]>ahr[2,1]:
					#center = N.vstack([0,0,ahr[1,1]])
					S = vf_frustum_bundle(num_rays=num_rays, r0=ahr[2,0], r1=ahr[2,1], depth=ahr[1,1]-ahr[1,0], center=center, direction=N.array([0,0,1]), rays_in=rays_in, procs=procs, angular_span=ahr[0])
				else:
					S = vf_frustum_bundle(num_rays=num_rays, r0=ahr[2,0], r1=ahr[2,1], depth=ahr[1,1]-ahr[1,0], center=center, direction=N.array([0,0,1]), rays_in=rays_in, procs=procs, angular_span=ahr[0])
			else:
				if ahr[2,0]>ahr[2,1]:
					center = N.vstack([0,0,ahr[1,1]])
					S = vf_frustum_bundle(num_rays=num_rays, r0=ahr[2,1], r1=ahr[2,0], depth=N.abs(ahr[1,1]-ahr[1,0]), center=center, direction=N.array([0,0,1]), rays_in=rays_in, procs=procs, angular_span=ahr[0])
				else:
					S = vf_frustum_bundle(num_rays=num_rays, r0=ahr[2,0], r1=ahr[2,1], depth=N.abs(ahr[1,1]-ahr[1,0]), center=center, direction=N.array([0,0,-1.]), rays_in=rays_in, procs=procs, angular_span=ahr[0])

		return S

	def alloc_VF(self, n):

		if self.receiver_type == 'Surround':
			ap_index = 3

			AP = self.A.get_objects()[0]
			ENV_bot = self.A.get_objects()[1]
			ABS = self.A.get_objects()[2]
			ENV_top = self.A.get_objects()[3]

			# Aperture: 3 surfaces
			Aperture_bot_abs, Aperture_bot_hits = AP.get_surfaces()[0].get_optics_manager().get_all_hits()
			Aperture_cyl_abs, Aperture_cyl_hits = AP.get_surfaces()[1].get_optics_manager().get_all_hits()
			Aperture_top_abs, Aperture_top_hits = AP.get_surfaces()[2].get_optics_manager().get_all_hits()

			# Absorber surfaces
			ABS_surfs = ABS.get_surfaces()
			Absorber_abs, Absorber_hits = ABS_surfs[0].get_optics_manager().get_all_hits()
			for s in xrange(1,len(ABS_surfs)):
				abso, hits = ABS_surfs[s].get_optics_manager().get_all_hits()
				Absorber_abs = N.concatenate((Absorber_abs, abso))
				Absorber_hits = N.concatenate((Absorber_hits, hits), axis=1)

			if len(ENV_bot.get_surfaces())>1:
				# Envelope bot cylinder: 
				Envelope_bot_abs, Envelope_bot_hits = ENV_bot.get_surfaces()[0].get_optics_manager().get_all_hits()
				# Envelope top cylinder:
				Envelope_top_abs, Envelope_top_hits = ENV_top.get_surfaces()[0].get_optics_manager().get_all_hits()
				# Regroup envelopes
				Envelope_abs = N.concatenate((Envelope_bot_abs, Envelope_top_abs))
				Envelope_hits = N.concatenate((Envelope_bot_hits, Envelope_top_hits), axis=1)

				# Regroup all receiver hits
				Receiver_abs = N.concatenate((Envelope_abs, Absorber_abs))
				Receiver_hits = N.concatenate((Envelope_hits, Absorber_hits), axis=1)
			else:
				Receiver_abs = Absorber_abs
				Receiver_hits = Absorber_hits

			# Aperture:
			self.VF[n,0] = N.sum(Aperture_bot_abs)
			self.VF[n,1] = N.sum(Aperture_cyl_abs)
			self.VF[n,2] = N.sum(Aperture_top_abs)

		elif self.receiver_type == 'Polar':

			# Aperture:
			ap_index = 1
			AP = self.A.get_objects()[0]
			self.VF[n,0] = N.sum(AP.get_surfaces()[0].get_optics_manager().get_all_hits()[0])

			# Rest of the absorber:
			ABS = self.A.get_objects()[1:]
	
			absos = []
			hitss = []

			for o in xrange(len(ABS)):
				abso, hits = ABS[o].get_surfaces()[0].get_optics_manager().get_all_hits()
				absos.append(abso)
				hitss.append(hits)
			Receiver_abs = N.hstack(absos)
			Receiver_hits = N.concatenate(hitss, axis=1)

		# Get the angles out of the cartesian coordinates:
		angles_receiver = N.arctan2(Receiver_hits[1],Receiver_hits[0])
		angles_receiver[angles_receiver<0.] = angles_receiver[angles_receiver<0.]+2.*N.pi

		# Get the radii from the cartesian coordinates:
		radii_receiver = N.around(N.sqrt(Receiver_hits[0]**2.+Receiver_hits[1]**2.),decimals=9)

		for i in xrange(N.shape(self.VF)[0]-ap_index):

			ahr = self.binning_scheme[i+ap_index]

			ang0 = ahr[0,0]
			ang1 = ahr[0,1]
			h0 = N.around(ahr[1,0], decimals=9)
			h1 = N.around(ahr[1,1], decimals=9)
			r0 = N.around(ahr[2,0], decimals=9)
			r1 = N.around(ahr[2,1], decimals=9)

			if r0>r1:
				r0,r1 = r1,r0
			if h0>h1:
				h0,h1 = h1,h0

			# Some floating point errors around here,
			hit_in_ang = N.logical_and(angles_receiver>=ang0, angles_receiver<=ang1)

			if h0==h1:
				hit_in_h = N.logical_and(Receiver_hits[2]>=h0, Receiver_hits[2]<=h1)
			else:
				hit_in_h = N.logical_and(Receiver_hits[2]>=h0, Receiver_hits[2]<=h1)

			if r0==r1:
				hit_in_r = N.logical_and(radii_receiver>=r0, radii_receiver<=r1)	
			else:
				hit_in_r = N.logical_and(radii_receiver>=r0, radii_receiver<=r1)


			hit_in_bin = hit_in_ang*hit_in_h*hit_in_r

			hit_abs = N.sum(Receiver_abs[hit_in_bin])

			self.VF[n,i+ap_index] = hit_abs

			# This is to speed up the process: we empty the hits containers as we assign and consequently browse through the remaining ones faster. We break the loop as soon as the containers are empty
			Receiver_abs = Receiver_abs[~hit_in_bin]
			Receiver_hits = Receiver_hits[:,~hit_in_bin]
			angles_receiver = angles_receiver[~hit_in_bin]
			radii_receiver = radii_receiver[~hit_in_bin]

			if len(Receiver_abs) == 0:
				'zeroleft'
				break

		self.reset_opt()


class Two_N_parameters_cavity_RTVF(RTVF):
	'''
	A class for 2N parameters axisymmetrical cavities composed of frusta and a cone.

	apertureRadius - Radius of the aperture of the geometry
	frustaRadii - List of the successive radii of the frusta, starting from the aperture and following the profile of the geometry.
	frustaDepths - List of the depths of the frusta, starting from the aperture and following the profile of the geometry.
	coneDepth - Depth of the cone to close the geometry.
	el_FRUs - A list describing the discretisation of the frusta in the scene. Each frustum [i] is discretised into el_FRUs[i] elements of equal depths.
	el_CON - The number of discretisation elements of equal depths used for the conical part of the receiver.
	'''
	def __init__(self, apertureRadius, frustaRadii, frustaDepths, coneDepth, el_FRUs, el_CON, num_rays=10000, precision=0.01):

		RTVF.__init__(self, num_rays, precision)

		self.apertureRadius = apertureRadius
		self.frustaRadii = frustaRadii
		self.frustaDepths = frustaDepths
		self.coneDepth = coneDepth
		self.el_FRUs = el_FRUs
		self.el_CON = el_CON
		procs = 8 # number of CPUs to be used

		self.t0=time.clock()

		self.VF = N.zeros((1+N.sum(el_FRUs)+el_CON, 1+N.sum(el_FRUs)+el_CON))
		self.progress = N.ones(N.shape(self.VF), dtype=N.bool)

		# Standard deviation computation variables.
		self.VF_esperance = N.zeros(N.shape(self.VF))
		self.Qsum = N.zeros(N.shape(self.VF))
		self.stdev_VF = N.zeros(N.shape(self.VF))
		self.stdev_reciprocity = N.zeros(N.shape(self.VF))

		areas = N.zeros(N.shape(self.VF)[0])
		self.p = N.zeros(N.shape(self.VF)[0])

		A = Assembly() # VF scene assembly

		if type(el_FRUs)==int:
			el_FRUs = N.asarray([el_FRUs])
			self.el_FRUs = el_FRUs
		if type(el_CON)==int:
			el_CON = N.asarray([el_CON])
			self.el_CON = el_CON

		# Areas calculations: ___________________________________________________________________
		areas[0] = N.pi*apertureRadius**2. # Aperture

		if apertureRadius==frustaRadii[0]: # Cylinder
			areas[1:1+el_FRUs[0]] = N.pi*2.*frustaRadii[0]*N.sqrt((frustaDepths[0]/el_FRUs[0])**2)
		else: # 1st frustum
			L = N.sqrt((frustaDepths[0])**2+(frustaRadii[0]-apertureRadius)**2)/el_FRUs[0]
			radii = N.hstack([apertureRadius, apertureRadius+(N.arange(el_FRUs[0])+1)*(frustaRadii[0]-apertureRadius)/el_FRUs[0]])
			areas[1:1+el_FRUs[0]] = N.pi*(radii[:-1]+radii[1:])*L

		for k in xrange(1,len(el_FRUs)): # next frusta
			if self.frustaRadii[k-1]==self.frustaRadii[k]:
				areas[1+N.sum(el_FRUs[:k+1])-el_FRUs[k]:1+N.sum(el_FRUs[:k+1])] = 2.*N.pi*frustaRadii[k]*N.sqrt((frustaDepths[k]/el_FRUs[k])**2)
			else:
				L = N.sqrt(frustaDepths[k]**2+(frustaRadii[k]-frustaRadii[k-1])**2)/el_FRUs[k]
				radii = N.hstack([frustaRadii[k-1],frustaRadii[k-1]+(N.arange(el_FRUs[k])+1)*(frustaRadii[k]-frustaRadii[k-1])/el_FRUs[k]])
				areas[1+N.sum(el_FRUs[:k+1])-el_FRUs[k]:1+N.sum(el_FRUs[:k+1])] = N.pi*(radii[:-1]+radii[1:])*L

		radii = N.hstack([frustaRadii[-1],frustaRadii[-1]+(N.arange(el_CON)+1)*(-frustaRadii[-1])/el_CON])
		areas[1+N.sum(el_FRUs):1+N.sum(el_FRUs)+el_CON] = N.pi*(radii[:-1]+radii[1:])*N.sqrt(coneDepth**2+frustaRadii[-1]**2)/el_CON # Cone

		self.areas = areas

		#_______________________________________________________________________________________
		#self.ray_counts = N.ones(len(areas))*int(self.num_rays/len(areas))
		self.ray_counts = N.ones(len(areas))*int(self.num_rays)

		# Build the geometry:___________________________________________________________________
		max_depth = N.sum(frustaDepths)

		AP = AssembledObject(surfs=[Surface(RoundPlateGM(Re=apertureRadius), LambertianReceiver(absorptivity=1.))], transform = None)

		FRU = []

		# 1st frustum:
		if apertureRadius==frustaRadii[0]: # Cylinder
			frustum = AssembledObject(surfs=[Surface(FiniteCylinder(diameter=frustaRadii[0]*2., height=frustaDepths[0]), LambertianReceiver(absorptivity=1.))], transform=translate(z=frustaDepths[0]/2.))
		elif frustaDepths[0] == 0.: # flat plate
			print 'FLAT!'
			frustum = AssembledObject(surfs=[Surface(RoundPlateGM(Re=apertureRadius, Ri=frustaRadii[0]), LambertianReceiver(absorptivity=1.))], transform=translate(z=N.sum(frustaDepths[:i])))
		else: # frustum
			frustum = AssembledObject(surfs=[Surface(ConicalFrustum(z1=0., r1=apertureRadius, z2=frustaDepths[0], r2=frustaRadii[0]), LambertianReceiver(absorptivity=1.))], transform=None)
		FRU.append(frustum)
		# next frusta:
		for i in xrange(1,len(frustaRadii)):
			if frustaRadii[i-1]==frustaRadii[i]:
				frustum = AssembledObject(surfs=[Surface(FiniteCylinder(diameter=frustaRadii[i]*2., height=frustaDepths[i]), LambertianReceiver(absorptivity=1.))], transform=translate(z=N.sum(frustaDepths[:i])+frustaDepths[i]/2.))
			elif frustaDepths[i] < 0.:
				frustum = AssembledObject(surfs=[Surface(ConicalFrustum(z1=0., r1=frustaRadii[i-1], z2=-frustaDepths[i], r2=frustaRadii[i]), LambertianReceiver(absorptivity=1.))], transform=N.dot(translate(z=N.sum(frustaDepths[:i])),rotx(N.pi)))
			elif frustaDepths[i] > 0.:
				frustum = AssembledObject(surfs=[Surface(ConicalFrustum(z1=0., r1=frustaRadii[i-1], z2=frustaDepths[i], r2=frustaRadii[i]), LambertianReceiver(absorptivity=1.))], transform=translate(z=N.sum(frustaDepths[:i])))
			else:
				frustum = AssembledObject(surfs=[Surface(RoundPlateGM(Re=frustaRadii[i-1], Ri=frustaRadii[i]), LambertianReceiver(absorptivity=1.))], transform=translate(z=N.sum(frustaDepths[:i])))
			FRU.append(frustum)

		# Cone section:
		if coneDepth>0.: # == cone depth > 0: Outgoing cone
			trc = N.dot(rotx(N.pi), translate(z=-(max_depth+coneDepth))) # Cone frame transformation
			CON = AssembledObject(surfs=[Surface(FiniteCone(r=frustaRadii[-1], h=coneDepth), LambertianReceiver(absorptivity=1.))], transform=trc)
			rays_cone=True
		elif coneDepth == 0.: # Round flat plates
			CON = AssembledObject(surfs=[Surface(RoundPlateGM(Re=frustaRadii[-1]), LambertianReceiver(absorptivity=1.))], transform=translate(z=max_depth))
			rays_cone=True
		else: # coneDepth < 0 Inward cone
			CON = AssembledObject(surfs=[Surface(FiniteCone(r=frustaRadii[-1], h=-coneDepth), LambertianReceiver(absorptivity=1.))], transform=translate(z=max_depth+coneDepth))
			rays_cone=False

		A.add_object(AP)
		for i in xrange(len(FRU)):
			A.add_object(FRU[i])
		A.add_object(CON)

		el_FRUs = self.el_FRUs
		el_CON = self.el_CON

		vf_tracer = TracerEngineMP(A)
		vf_tracer.itmax = 1 # stop iteration after this many ray bundles were generated (i.e. after the original rays intersected some surface this many times).
		vf_tracer.minener = 1e-10 # minimum energy threshold
		stable_stats = 0
		while (self.progress==True).any() or stable_stats<2:

			tp = time.clock()

			if self.ray_counts[0] != 0.:

				SA = []
				for p in xrange(procs):
					SA.append(solar_disk_bundle(self.ray_counts[0]/procs, center=N.vstack([0,0,0]), direction=N.array([0,0,1]), radius=apertureRadius, ang_range=N.pi/2., flux=1./(N.pi*self.apertureRadius**2.)/procs))

				vf_tracer.multi_ray_sim(SA, procs = procs)

				#view = Renderer(vf_tracer)
				#view.show_rays()
				self.A = vf_tracer._asm #due to multiprocessing inheritance break
				self.alloc_VF(0)

			for elf in xrange(int(el_FRUs[0])):
				if self.ray_counts[elf+1] != 0.:
					center = N.vstack([0,0,elf*frustaDepths[0]/el_FRUs[0]])
					r0 = apertureRadius+elf*(frustaRadii[0]-apertureRadius)/el_FRUs[0]
					r1 = apertureRadius+(elf+1)*(frustaRadii[0]-apertureRadius)/el_FRUs[0]
					depth = frustaDepths[0]/el_FRUs[0]
					num_rays = self.ray_counts[elf+1]
					rays_in = True
					S = []
					for p in xrange(procs):
						S.append(self.gen_source(num_rays/procs, r0, r1, depth, center, rays_in, procs=procs))
					vf_tracer.multi_ray_sim(S, procs=procs)

					self.A = vf_tracer._asm #due to multiprocessing inheritance break
					self.alloc_VF(elf+1)


			for n in xrange(1,len(el_FRUs)):
				for elf in xrange(int(el_FRUs[n])):
					if self.ray_counts[1+N.sum(el_FRUs[:n+1])-el_FRUs[n]+elf] != 0.:
						center = N.vstack([0,0,N.sum(frustaDepths[:n])+elf*frustaDepths[n]/el_FRUs[n]])
						r0 = frustaRadii[n-1]+elf*(frustaRadii[n]-frustaRadii[n-1])/el_FRUs[n]
						r1 = frustaRadii[n-1]+(elf+1)*(frustaRadii[n]-frustaRadii[n-1])/el_FRUs[n]
						depth = frustaDepths[n]/el_FRUs[n]
						num_rays = self.ray_counts[1+N.sum(el_FRUs[:n+1])-el_FRUs[n]+elf]

						if frustaDepths[n] < 0.:
							rays_in = False
						else:
							rays_in = True

						S = []
						for p in xrange(procs):
							S.append(self.gen_source(num_rays/procs, r0, r1, depth, center, rays_in, procs=procs))
						vf_tracer.multi_ray_sim(S, procs=procs)
						#view = Renderer(vf_tracer)
						#view.show_rays()

						self.A = vf_tracer._asm #due to multiprocessing inheritance break

						self.alloc_VF(1+N.sum(el_FRUs[:n+1])-el_FRUs[n]+elf)

			for elc in xrange(int(el_CON)):
				if self.ray_counts[N.sum(el_FRUs)+elc+1] != 0.:
					center = N.vstack([0,0,N.sum(frustaDepths)+coneDepth*elc/el_CON])
					r0 = frustaRadii[-1]+elc*(-frustaRadii[-1])/el_CON
					r1 = frustaRadii[-1]+(elc+1)*(-frustaRadii[-1])/el_CON
					depth = coneDepth/el_CON
					num_rays = self.ray_counts[N.sum(el_FRUs)+elc+1]
					rays_in = rays_cone

					S = []
					for p in xrange(procs):
						S.append(self.gen_source(num_rays/procs, r0, r1, depth, center, rays_in, procs=procs))
					vf_tracer.multi_ray_sim(S, procs=procs)

					self.A = vf_tracer._asm #due to multiprocessing inheritance break

					self.alloc_VF(N.sum(el_FRUs)+elc+1)


			self.p += self.ray_counts
			self.test_precision()
			print '		Progress:', N.sum(self.progress),'/', len(self.progress),'; Pass duration:', time.clock()-tp, 's'
			if N.sum(self.progress) ==0:
				stable_stats +=1
			'''
			if self.p[0]>5e6:
				self.stdev_store = N.array(self.stdev_store)
				for h in xrange(N.shape(self.stdev_store)[1]):
					for v in xrange(N.shape(self.stdev_store)[2]):
						plt.plot(N.arange(0, self.p[0], self.ray_counts[0]), self.stdev_store[:,h,v], label= str(h)+str(v))
				plt.legend()
				plt.show()
		self.stdev_store = N.array(self.stdev_store)
		for h in xrange(N.shape(self.stdev_store)[1]):
			for v in xrange(N.shape(self.stdev_store)[2]):
				plt.plot(N.arange(0, self.p[0], self.ray_counts[0]), self.stdev_store[:,h,v], label= str(h)+str(v))
		plt.legend()
		plt.show()
			'''
		t1=time.clock()-self.t0
		print '	VF calculation time:',t1,'s'

	def gen_source(self, num_rays, r0, r1, depth, center, rays_in, procs):
		'''
		Generate a source for a specific element		
		'''
		if r0==r1:
			S = vf_cylinder_bundle(num_rays=num_rays, rc=r0, lc=depth, center=center, direction=N.array([0,0,1]), rays_in=rays_in, procs=procs)
		elif depth==0.:
			S = solar_disk_bundle(num_rays=num_rays, center=center, direction=N.array([0,0,N.sign(r1-r0)]), radius=r0, ang_range=N.pi/2., radius_in=r1, procs=procs)
		else:
			S = vf_frustum_bundle(num_rays=num_rays, r0=r0, r1=r1, depth=depth, center=center, direction=N.array([0,0,1]), rays_in=rays_in, procs=procs)

		return S

	def alloc_VF(self, n):
		'''
		get hits in the scene and bin them in VF matrix.
		'''
		apertureRadius = self.apertureRadius
		frustaRadii = self.frustaRadii
		frustaDepths = self.frustaDepths
		coneDepth = self.coneDepth
		el_FRUs = self.el_FRUs
		el_CON = self.el_CON

		# Gather hits and absorbed radiative power
		self.AP = self.A.get_objects()[0]
		self.FRU = self.A.get_objects()[1:N.sum(len(el_FRUs))+1]
		self.CON = self.A.get_objects()[N.sum(len(el_FRUs))+1:]

		Aperture_abs, Aperture_hits = self.AP.get_surfaces()[0].get_optics_manager().get_all_hits()
		'''
		Frustum_abs = []
		Frustum_hits = []
		for i in xrange(len(el_FRUs)):
			Fru_abs, Fru_hits = self.FRU[i].get_surfaces()[0].get_optics_manager().get_all_hits()
			Frustum_abs.append(N.asarray(Fru_abs))
			Frustum_hits.append(N.asarray(Fru_hits))
		'''
		Cone_abs, Cone_hits = self.CON[0].get_surfaces()[0].get_optics_manager().get_all_hits()

		heights = N.add.accumulate(N.hstack([0, frustaDepths]))
		rads = N.hstack([apertureRadius, frustaRadii])


		# VF allocation to a nxn VF matrix. Convention is to go from the aperture to the back of the shape following the axi-symmetric profile line. First loop is for geometrical shapes and second one for the discretisation of each shape.
		for j in xrange(len(el_FRUs)+2):

			if j == 0:
				self.VF[n,j] = N.sum(Aperture_abs)

			elif j <= len(el_FRUs):
				Fru_abs, Fru_hits = self.FRU[j-1].get_surfaces()[0].get_optics_manager().get_all_hits()
				fru_hits_r = N.around(N.sqrt(Fru_hits[0]**2.+Fru_hits[1]**2.), decimals=9)
				fru_hits_h = N.around(Fru_hits[2], decimals=9)

				for i in xrange(int(el_FRUs[j-1])):
					#frustum_el_base = N.sum(frustaDepths[:j])-frustaDepths[j-1]+i*frustaDepths[j-1]/el_FRUs[j-1]
					#frustum_el_top = N.sum(frustaDepths[:j])-frustaDepths[j-1]+(i+1)*frustaDepths[j-1]/el_FRUs[j-1]

					frustum_h_base = heights[j-1]+i*(heights[j]-heights[j-1])/el_FRUs[j-1]
					frustum_h_top = heights[j-1]+(i+1)*(heights[j]-heights[j-1])/el_FRUs[j-1]

					frustum_r_min = rads[j-1]+i*(rads[j]-rads[j-1])/el_FRUs[j-1]
					frustum_r_max = rads[j-1]+(i+1)*(rads[j]-rads[j-1])/el_FRUs[j-1]

					if frustum_h_base>frustum_h_top:
						frustum_h_base, frustum_h_top = frustum_h_top, frustum_h_base
					if frustum_r_min>frustum_r_max:
						frustum_r_min, frustum_r_max = frustum_r_max, frustum_r_min		

					#print frustum_h_base, frustum_h_top, frustum_r_base, frustum_r_top
					in_frustum_el_h = N.logical_and(fru_hits_h>=frustum_h_base, fru_hits_h<=frustum_h_top)
					in_frustum_el_r = N.logical_and(fru_hits_r>=frustum_r_min, fru_hits_r<=frustum_r_max)
					
					in_frustum_el = N.logical_and(in_frustum_el_h, in_frustum_el_r)

					self.VF[n,i+1+N.sum(el_FRUs[:j])-el_FRUs[j-1]] = N.sum(Fru_abs[in_frustum_el])

			else:
				for i in xrange(int(el_CON)):

					r1 = frustaRadii[-1]-i*frustaRadii[-1]/float(el_CON)
					r2 = frustaRadii[-1]-(i+1)*frustaRadii[-1]/float(el_CON)

					cone_hits_radii = N.sqrt(Cone_hits[0]**2+Cone_hits[1]**2)
					in_cone_el = N.logical_and(cone_hits_radii<r1, cone_hits_radii>=r2)
		
					self.VF[n,i+1+N.sum(el_FRUs)] = N.sum(Cone_abs[in_cone_el])
		
		self.reset_opt()


class Four_parameters_cavity_RTVF(Two_N_parameters_cavity_RTVF):
	'''
	Wrapper around the Two_N_parameters_cavity_RTVF class to a 4 parameters cavity.
	ref: "Open cavity receiver geometry influence on radiative losses" (DOI:10.13140/2.1.3845.5048)
	'''
	def __init__(self, apertureRadius, apertureDepth, coneRadius, coneDepth, el_FRU, el_CON, num_rays, precision):
		Two_N_parameters_cavity_RTVF.__init__(self, apertureRadius, [coneRadius], [apertureDepth], coneDepth, el_FRU, el_CON, num_rays, precision)

