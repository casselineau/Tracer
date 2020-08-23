'''
A tower/heliostat field example.

Imports the coordinaes of the Sandia NSTTF facility.

Blocking and shading are identified while ray-traciong and stored for each heliostat in respective blocking and shading arrays.
'''
from tracer.CoIn_rendering.rendering import *

from tracer.ray_bundle import RayBundle, concatenate_rays
from tracer.sources import solar_disk_bundle, buie_sunshape
from tracer.assembly import Assembly
from tracer.spatial_geometry import rotx, roty, rotz, rotation_to_z
from tracer.tracer_engine import TracerEngine

from tracer.tracer_engine_mp import TracerEngineMP

from tracer.models.one_sided_mirror import one_sided_receiver
from tracer.models.heliostat_field import HeliostatField, radial_stagger, solar_vector

import numpy as N
import time
from scipy.constants import degree
import math
import matplotlib.pyplot as plt
import pickle


class TowerScene():

	# Location of the sun:
	sun_az = 0
	sun_zenith = 35.05 #34.96

	sun_vec = solar_vector(sun_az*degree, sun_zenith*degree)
	hstat_normals = N.zeros((218,3))
	
	# import custom coordinate file
	pos = N.loadtxt("/home/charles/Documents/Tracer/examples/sandia_hstat_coordinates.csv", delimiter=',')
		
	# Field-based calculations for source size parameters
	#===================================================
	t_pos = pos.T
	xc_min = t_pos[0][N.argmin(t_pos[0])]
	xc_max = t_pos[0][N.argmax(t_pos[0])]
	yc_min = t_pos[1][N.argmin(t_pos[1])]
	yc_max = t_pos[1][N.argmax(t_pos[1])]

	x_dist = xc_max - xc_min
	y_dist = yc_max - yc_min

	xc_cent = (xc_min + xc_max) / 2
	yc_cent = (yc_min + yc_max) / 2
	field_centre = N.r_[xc_cent, yc_cent, 0]
	#===================================================
	
	def __init__(self):
		self.gen_plant() 

	def gen_rays(self, num_rays, flux=1000.):
		#========================
		individual_source = False
		#========================

		if individual_source:
			# Pillbox source on a per-heliostat basis
			radius = 1.20 * math.sqrt(2 * 3.405**2)

			direction = N.array(-self.sun_vec)

			ray_list = []
			num_surfs = self.pos.shape[0]
			for i in xrange(num_surfs):
				centre = N.c_[50 * self.sun_vec + self.pos[i]]
				#rayb = solar_disk_bundle(num_rays/num_surfs, centre, direction, radius, 4.65e-3, flux)
				rayb = buie_sunshape(num_rays/num_surfs, centre, direction, radius, CSR=0.01, flux=flux)	
				ray_list.append(rayb)

			rays = concatenate_rays(ray_list)
			del ray_list
			
		else:
			# Large pillbox sunshape source disc source covering entire field area:
			radius = 1.10 * math.sqrt((self.x_dist/2)**2 + (self.y_dist/2)**2)

			self.source_area = N.pi * radius**2

			centre = N.c_[300*self.sun_vec + self.field_centre]
			direction = N.array(-self.sun_vec)

			#rays = solar_disk_bundle(num_rays, centre, direction, radius, 4.65e-3, flux)
			rays = buie_sunshape(num_rays, centre, direction, radius, CSR=0.01, flux=flux, pre_process_CSR=False)	
				
		return rays
	
	def gen_plant(self, width=6.1, height=6.1, absorptivity=0.04, aim_height=60.,  sigma_xy=1e-3, rec_w=11., rec_h=11.):

		self.pos[:,1] = self.pos[:,1]-4. # correction for the true position of the plate on the tower.
		self.width = width
		self.height = height
		self.absorptivity = absorptivity
		self.field = HeliostatField(self.pos, width, height, absorptivity, aim_height,  sigma_xy)
		self.rec_w = rec_w
		self.rec_h = rec_h
		rec, recobj = one_sided_receiver(self.rec_w, self.rec_h)
		rec_trans = rotx(N.pi/-2)
		rec_trans[2,3] = self.field._th

		# Evaluating just the receiver
		recobj.set_transform(rec_trans)
	
		self.plant = Assembly(objects=[recobj], subassemblies=[self.field])	  
	
	def aim_field(self):
		hstat_az, hstat_elev = self.field.aim_to_sun(self.sun_az*degree, self.sun_zenith*degree)

		return hstat_az, hstat_elev

	def calculate_area(self, hstat_az, hstat_elev):
		'''
		Calculates the heliostats areas as seen from the source, necessary for shading calculations.
		'''
		# CONVERSION
		# sun_vec az	0	   -45	 -90	 +-180	   +90	 +45
		# hstat_az	  -90	 -45	 0	   +90		 +-180   -135
		hstat_az = -hstat_az - N.pi/2
		
		for i in xrange(len(self.pos)):
			self.hstat_normals[i] = solar_vector(hstat_az[i], hstat_elev[i])
		
		self.hstat_proj_areas = [0]*len(self.pos)
		for i in xrange(len(self.pos)):
			self.hstat_proj_areas[i] = (6.1**2) * abs(N.dot(-self.sun_vec, self.hstat_normals[i]))
	
	def trace(self, num_rays=1e5, nbins_w=50., nbins_h=50.):
		'''
		Raytrace method.

		Raytraces successive bundles and stores the resultsogf the shading, blicking, incoming radiative power on the heliostats and the fluxmap on the receiver.
		'''
		# Generate a large ray bundle using [a radial stagger much denser
		# than the field] a Buie sunshape with radius equal to the longest
		# dimension of the field.

		#=============
		render = False
		#=============
		
		sun_vec = solar_vector(self.sun_az*degree, self.sun_zenith*degree)
		
		bundlesize = 1e4
		iters = int(num_rays/bundlesize)

		# Results bins:
		incoming = N.zeros(len(self.pos))
		prev_incoming = N.zeros(len(self.pos))
		incoming_Q = N.zeros(len(self.pos))
		incoming_stdev = N.zeros(len(self.pos))

		shading = N.ones(len(self.pos))
		prev_shading = N.zeros(len(self.pos))
		shading_Q = N.zeros(len(self.pos))
		shading_stdev = N.zeros(len(self.pos))

		blocking = N.zeros(len(self.pos))
		prev_blocking = N.zeros(len(self.pos))
		blocking_Q= N.zeros(len(self.pos))
		blocking_stdev= N.zeros(len(self.pos))

		timer_mcrt = 0. 
		timer_postprocess = 0.

		# Receiver bins:

		dlw = self.rec_w/nbins_w
		dlh = self.rec_h/nbins_h
		bins_w = N.arange(-self.rec_w/2.,self.rec_w/2.+dlw, dlw)
		bins_h = N.arange(-self.rec_h/2.,self.rec_h/2.+dlh, dlh)
		bins = [bins_w, bins_h]
		self.bins = bins
		fluxmap = N.zeros((len(bins_w)-1,len(bins_h)-1))

		# Raytrace:
		mcrt = time.clock()
		e = TracerEngineMP(self.plant)
		procs = 1
		e.minener = 1e-10
		timer_mcrt += time.clock()-mcrt
		hits_helios=0
		i=0

		#while hits_helios < num_rays:
		for i in xrange(iters):			
			print ' '
			print ' '
			print 'ITERATION ', i+1, ' of ', iters 
			#print hits_helios, 'hits out of ', num_rays
			mcrt = time.clock()
			# Perform the trace:
			sources = []
			self.flux = 1000.
			for s in xrange(procs):
				sources.append(self.gen_rays(num_rays=bundlesize/float(procs), flux=self.flux/float(procs)))
			e.multi_ray_sim(sources=sources, procs=procs)
			self.plant = e._asm
			self.field._heliostats = self.plant._assemblies[0].get_surfaces()
			self.rec = self.plant._objects[0].get_surfaces()[0]

			timer_mcrt += time.clock()-mcrt
			postprocess = time.clock()

			# Render:
			if render:
				trace_scene = Renderer(e)
				trace_scene.show_rays(resolution=10)

			# Get the energy and location of all hits using optics manager
			en, pts = self.rec.get_optics_manager().get_all_hits()
			x, y = self.rec.global_to_local(pts)[:2]

			# FLUX MAP OPERATIONS
			#===========================================================================
			H, xbins, ybins = N.histogram2d(x, y, bins, weights=en/(dlw*dlh)*1e-3)
			extent = [ybins[0], ybins[-1], xbins[-1], xbins[0]]

			fluxmap = (fluxmap*float(i)+H)/(i+1.)
			#===========================================================================
		
			# BLOCKAGE and SHADING
			#===========================================================================
			# Detect blockage and look for the parents of the blocked rays. Identify from which heliostats the parents come and associate the blockage losses to the heliostats where blockage is suffered.
			
			hz = (e.tree._bunds[1].get_vertices()[2]) < (self.field._th-self.rec_h/2.)
			hits_helios += N.sum(hz)
			print 'Useful rays:', hits_helios
			# Get the 3rd bundle (after 2 hits):
			bund_2 = e.tree._bunds[2].get_vertices()
			bund_2_ener = e.tree._bunds[2].get_energy()

			# Remove receiver hits from the bundle to get only hits on heliostats:
			bund_2_helio_hits = N.ravel(N.nonzero(bund_2[2] < (self.field._th-self.rec_h/2.)))
			bund_2_bloc = bund_2[:, bund_2_helio_hits]

			# Get the bundle emitting the blocked rays and isolate the blocked rays:
			bund_1_helio_blocs = e.tree._bunds[2].get_parents()[bund_2_helio_hits]
			bund_1 = e.tree._bunds[1].get_vertices()
			bund_1_ener = e.tree._bunds[1].get_energy()
			bund_1_bloc = bund_1[:, bund_1_helio_blocs]

			# Screen the field to find where blocked rays originate:
			for h in xrange(len(self.pos)):
				# Get the information from the optics manager of the heliostat:
				abs_hstats, hits_hstats, dirs_hstats = self.field._heliostats[h].get_optics_manager().get_all_hits()

				blocs = []
				hit_0s = []

				# Check if any hits:
				if len(hits_hstats)!=0:
					# Screen through every hit:
					for r in xrange(N.shape(hits_hstats)[1]):
						# Is the hit a ray that will be blocked or a blocked ray?
						bloc = N.nonzero(hits_hstats[0,r] == bund_1_bloc[0])[0]
						# Next "if" is because if there are no valid hits the bloc returns an empty array or to isolate each hit in case of 2 hits matching.
						if len(bloc)>0:
							for b in xrange(len(bloc)):
								# If sthe first coordinate matches, do the rest of them?
								if (hits_hstats[:,r]==N.ravel(bund_1_bloc[:,bloc[b]])).all():
									# If so add the blocked energy to the result bin.
									blocs.append(bund_1_helio_blocs[bloc[b]])

						else:
							hit_0 = N.nonzero(hits_hstats[0,r] == bund_1[0])[0]
							if len(hit_0)>0:
								for s in xrange(len(hit_0)):
									if (hits_hstats[:,r]==N.ravel(bund_1[:,hit_0[s]])).all():			
										hit_0s.append(e.tree._bunds[1].get_parents()[hit_0[s]])
				prev_blocking[h] = blocking[h]

				# Monte-Carlo sampling:
				blocking[h] = (blocking[h]*i+N.sum(bund_1_ener[blocs]))/(i+1.)		

				# Shading is the theoretical energy hitting subtracted by the energy absorbed without the backside blocking.
				prev_incoming[h] = incoming[h]
				# Monte-Carlo sampling:
				incoming[h] = (incoming[h]*i+N.sum(e.tree._bunds[0].get_energy()[hit_0s]))/(i+1.)

				prev_shading[h] = shading[h]
				# Monte-Carlo sampling:
				shading[h] = (shading[h]*i+self.flux*self.hstat_proj_areas[h]-incoming[h])/(i+1.)
	
			# Streamlined stats variable:
			incoming_Q = incoming_Q+i/(i+1.)*(incoming-prev_incoming)**2.
			blocking_Q = blocking_Q+i/(i+1.)*(blocking-prev_blocking)**2.
			shading_Q = shading_Q+i/(i+1.)*(shading-prev_shading)**2.

			# Standard deviatiosn updates:
			if i>0:
				incoming_stdev = N.sqrt(incoming_Q/i)
				blocking_stdev = N.sqrt(blocking_Q/i)
				shading_stdev = N.sqrt(shading_Q/i)

			print 'Shading=', N.sum(shading)
			print 'Blockage=', N.sum(blocking)

			timer_postprocess += time.clock()-postprocess

			print 'timer_mcrt: ', timer_mcrt/60., 'min'
			print 'timer_postprocess: ', timer_postprocess/60., 'min'

			print 'Peak flux (kW/m2):', N.amax(fluxmap)
			print 'AVG flux (kW/m2): ', N.sum(fluxmap)/(N.shape(fluxmap)[0]*N.shape(fluxmap)[1])
			print 'Total radiative power (kW): ', N.sum(fluxmap*(11./50.)**2)

			i+=1
		
			#===========================================================================
			e.tree._bunds = []
			for clear in xrange(len(e._asm.get_surfaces())):
				e._asm.get_surfaces()[clear].get_optics_manager().reset()
			#===========================================================================
			del(self.plant)
		results = {'positions':self.pos, 'blocking':blocking, 'blocking_stdev':blocking_stdev, 'shading':shading, 'shading_stdev':shading_stdev, 'incoming':incoming, 'incoming_stdev':incoming_stdev, 'fluxmap':fluxmap, 'extent':extent, 'width':self.width, 'height':self.height, 'absorptivity':self.absorptivity, 'rec_width':self.rec_w, 'rec_height':self.rec_h, 'rec_bins':self.bins}
		filesave = open('/home/charles/Documents/Boulot/These/Heliostat field/Sandia_data','w')
		pickle.dump(results, filesave)
		filesave.close()

if __name__ == '__main__':
	scene = TowerScene()
	hstat_az, hstat_elev = scene.aim_field()
	scene.calculate_area(hstat_az, hstat_elev)
	total_time = time.time()
	scene.trace(num_rays=20000000.)
	total_time = time.time()-total_time
	print 'Simulation total time: ', total_time/60., 'min'

