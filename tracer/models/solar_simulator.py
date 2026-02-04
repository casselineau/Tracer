import numpy as N
from tracer.assembly import Assembly
from tracer.object import AssembledObject
from tracer.surface import Surface
from tracer.flat_surface import RectPlateGM
from tracer.sphere_surface import SphericalGM
from tracer.ray_bundle import *
from ray_trace_utils.sampling import cylinder_sampling, sphere_sampling, PW_linear_distribution
from ray_trace_utils.vector_manipulations import axes_and_angles_between, rotate_z_to_normal
from tracer.spatial_geometry import general_axis_rotation
from tracer.ellipsoid import *
from tracer.optics_callables import RealReflective, TransparentTransmitter
from tracer.tracer_engine import TracerEngine
from tracer.CoIn_rendering.rendering import *
from tracer.sources import isotropic_directions_sampling
from sys import path
from ray_trace_utils.estimator import Estimator
from copy import copy
import pathlib

class Target(AssembledObject):
	def __init__(self, width, height, location, normal, binx, biny):
		'''
		This is a target class for fluxmapping in whioch the self.fluxmap is also an estimator, meaning that it has moving window statistics implemented on teh fluxmap.
		Parameters:
		- width, height: dimensions of the target
		- location: Position of the target
		- normal: Surface normal of the target
		- binx, biny: pyplot-bins-like arguments to pass to the targets 2D histogram functions.
		'''
		geom = RectPlateGM(binx[-1]-binx[0], biny[-1]-biny[0])
		opt = TransparentTransmitter()
		axis, ang = axes_and_angles_between(N.array([0.,0.,1.]), normal)
		rotation_target = general_axis_rotation(axis, ang)
		AssembledObject.__init__(self, surfs=[Surface(geometry=geom, optics=opt)], location=location, rotation=rotation_target)
		self.binx, self.biny = binx, biny
		self.areas = N.vstack(binx[1:]-binx[:-1])*(biny[1:]-biny[:-1])
		self.fluxmap = Estimator() # this means that the fluxmap instance can perform sliding window statistics to evaluate CIs.
		
	def evaluate_fluxmap(self):
		'''
		Calculate the fluxmap of the target and resets the poptical manager for the next iteration.
		'''
		surf = self.get_surfaces()[0]
		opt = surf.get_optics_manager()
		
		abso, hits = opt.get_all_hits()
		hits = surf.global_to_local(hits)
		opt.reset()

		powermap, edges = N.histogramdd(hits[:2].T, bins=(self.binx, self.biny), weights=abso)
		self.fluxmap.update(powermap/self.areas)

		return self.fluxmap

class SolarSimulator(Assembly):
	'''
	A solar simulator class. The solar simulator is composed of modules.		
	'''
	def __init__(self, modules_positions, modules_directions, modules_dicts, targets, homegenizer=None):
		'''
		Parameters:
		- modules_positions: list (or array) of M length 3 (or shape(M,3)) vectors with each vector giving the position of the module first focus in 3D space.
		- modules_directions: list(array) of M length 3 (or shape(M,3)) unit vectors with each unit vector giving the aiming direction (unit direction vector along the optical axis going from the first to the second focus of the ellipsoids) of the module in 3D space.
		- modules_dicts: dictionaries of parameters needed to instantiate modules.
		'''
		self.modules = []
		for i in range(len(modules_positions)):
			self.modules.append(SolarSimulatorModule(**modules_dicts[i], first_focus_location=modules_positions[i], aiming_vector=modules_directions[i]))
		self.targets = targets
		objects = targets
		if homogenizer is not None:
			self.homogenizer = homogenizer
			objects += self.homogenizer
		Assembly.__init__(self, subassemblies=self.modules, objects=objects)
	
	def simulate(self, nrays, part_load=1., lamp_mapper=False, rendering=False, ray_batch=1e4, save_dir= path[0]+'/Fluxmaps'):
		'''
		This simulate method creates a range of targets then performs a ray-trace with the given number of rays per module.
		The targets fluxmaps are saved in '.csv' file and plotted as colormaps in '.png' in a Fluxmaps folder that needs to be created beforehand.
		Arguments:
		- nrays: number of rays traced
		- lamp_mapper: optional argument. If set to True, makes a spherical surface around the lamp arc to perform flux mapping.
		'''
		# Make sure that teh dir exists
		pathlib.Path(save_dir).mkdir(parents=True, exist_ok=True) 
		
		# Sequential ray-trace simulation
		ray_batch = int(N.amin([ray_batch, nrays]))
		for i in range(int(nrays/ray_batch)):
			# Re-build engine
			engine = TracerEngine(self)
			sources = []
			for m in self.modules:
				sources.append(m.fire_lamp(ray_batch, lamp_mapper=lamp_mapper, part_load=part_load))
			source = concatenate_rays(sources)
			engine.ray_tracer(source)
			print ("%i rays simulated per reflector"%((i+1)*ray_batch))
			if rendering:
				Renderer(engine).show_rays(fluxmap=True, resolution=40, max_rays=1000, vmin=0)#, trans=True)
			# Gather fliuxmaps
			for j,target in enumerate(self.targets):
				target.evaluate_fluxmap()
				fluxmap = target.fluxmap.value
				CI = target.fluxmap.CI

				# save the fluxmaps
				with open(save_dir+'/fluxmap_%i.csv'%(j), 'w') as fo:
					fo.write('bins_x,')
					for e in target.binx:
						fo.write(str(e)+',')
					fo.write('\n')
					fo.write('bins_y,')
					for e in target.biny:
						fo.write(str(e)+',')
					fo.write('\n')

					for l in range(fluxmap.shape[0]):
						for f in fluxmap[l]:
							fo.write(str(f)+',')
						fo.write('\n')
						
				# save the CI maps
				with open(save_dir+'/confidence_intervals_%i.csv'%(j), 'w') as fo:
					fo.write('bins_x,')
					for e in target.binx:
						fo.write(str(e)+',')
					fo.write('\n')
					fo.write('bins_y,')
					for e in target.biny:
						fo.write(str(e)+',')
					fo.write('\n')

					for l in range(CI.shape[0]):
						for f in CI[l]:
							fo.write(str(f)+',')
						fo.write('\n')
						
class SolarSimulatorModule(Assembly):
	'''
	A module class for solar simulator including an ellipsoidal reflector and a plasma arc lamp source.
	'''
	def __init__(self, a, b, c, zlim, reflectivity=0.9, slope_error=2.5e-3, bi_var=False, lampdict={'model':'Bader', 'P_elec':2.5e3, 'eff_el':0.6, 'r_c':7.5e-4, 'l_c':4.5e-3, 'theta_CDF':'/media/ael/Flashy/backup_05-06-2021/Documents/Boulot/Projects/Solar_simulator/CDF_theta.dat'}, first_focus_location=N.array([0,0,0]), aiming_vector=N.array([0,0,1])):
		'''
		Parameters:
		- a, b and c are the parameters of the ellipsoid shape: the semi axes in x, y and z directions
		- zlim is a list of the truncation distancesof the ellipsoidal shape along the z axis.
		- lampdict is a dictionary that determines the lamp model used. the 'model' key determines the type of model used and the rest of the dict contains the arguments names and values. The currently implemented models are:
			'Bader': https://doi.org/10.1115/1.4028702
			'Zhu': https://doi.org/10.1016/j.apenergy.2020.115165
			More information aboiut teh required arguments for each model is available in the respective classes.
			The lamp position and direction is ovreriddent by the next two arguments of this class
		- first_focus_location: (3) vector location of the first focal point, the position of reference of thsi system.
		- aiming_vector: (3) unit vector giving the oriented direction of the module (along a line that joins the first and second focii of the ellipsoid.
		'''
		# We place the 0 at the lamp mid-arc location and then the reflector, aligned along the z direction, is adjusted with +z pointing at the target and -z pointing at the reflector.
		self.location = first_focus_location
		self.aiming_vector = aiming_vector

		axis, angle = axes_and_angles_between(vecs=N.array([0,0,1.]), normal=N.array(aiming_vector))
		self.rotation = general_axis_rotation(axis, angle)
			
		objects = []
		self.reflector = SimulatorReflector(a, b, c, zlim, self.location, self.rotation, reflectivity, slope_error, bi_var)
		objects.append(self.reflector)
		Assembly.__init__(self, objects=objects)
		model = lampdict['model']
		lamp_params = copy(lampdict)
		lamp_params.pop('model')
		self.lamp = eval('SimulatorLamp%s(**lamp_params)'%(model))

	def fire_lamp(self, nrays, lamp_mapper=False, part_load=1.):
		'''
		Fires rays from the lamp
		Arguments:
		- nrays: number of rays to sample
		- lamp_mapper: optionally adds a spherical fluxmap surface around the lamp to check what the distribution of the rays looks like.
		- part_load: a part-load factor on the arc output.
		'''
		# Optionally makes the spherical flux sensor around the lamp. 
		if lamp_mapper:
			geom = SphericalGM(radius=2.*self.lamp.l_c)
			opt = TransparentTransmitter()
			lamp_mapper = AssembledObject(surfs=[Surface(geometry=geom, optics=opt)], location=self.location, rotation=self.rotation)
			self.add_object(lamp_mapper)
		# Create lamp source
		source = self.lamp.generate_rays(nrays, part_load=part_load)
		# Reposition the source in general referential:
		# Rotate vertices in local referential, the center of the cylinder is 0 and move them to self.loc
		source.set_vertices(rotate_z_to_normal(source.get_vertices(), normal=self.aiming_vector) + 			N.vstack(self.location))
		# Rotate direction vectors with direction indicating the new +z
		source.set_directions(rotate_z_to_normal(source.get_directions(), normal=self.aiming_vector))
		return source

class SimulatorReflector(AssembledObject):
	def __init__(self, a, b, c, zlim, location, rotation, reflectivity=0.9, slope_error=2.5e-3, bi_var=False):
		'''
		An ellipsoid reflector class defined by semi-axes.
		Parameters:
		- a, b and c are the parameters of the ellipsoid shape: the semi-axes in x, y and z directions
		- zlim is a list of the truncation distances of the ellipsoidal shape along the z axis.
		- location, rotation: the location of the first focus and rotation of the reflector
		- slope_error: the "conical" or axisymmetrical slope error of the reflector surface.
		'''
		excentricity = N.sqrt(1.-a**2/c**2)
		half_focal_dist = c*excentricity
		geom = EllipsoidGM(a, b, c, zlim=zlim)
		opt = RealReflective(absorptivity=1.-reflectivity, sigma=slope_error, bi_var=bi_var)
		AssembledObject.__init__(self, surfs=[Surface(geometry=geom, optics=opt, location=N.array([0.,0.,half_focal_dist]))], location=location, rotation=rotation)
		self.excentricity = excentricity
		self.focal_dist = 2.*half_focal_dist


class SimulatorLampBader():
	def __init__(self, P_elec=2.5e3, eff_el=0.6, r_c=7.5e-4, l_c=4.5e-3, theta_CDF='/media/ael/Flashy/backup_05-06-2021/Documents/Boulot/Projects/Solar_simulator/CDF_theta.dat', location=[0,0,0], direction=[0,0,1]):
		'''
		Parameters:
		- P_elec: electrical power of the lamp
		- eff_el: electrical efficiency ie. conversion of eletrical into radiant power.
		- r_c, l_c: radius and length of the cylindrical volume used to model teh lamp arc.
		- theta_CDF: a file descriing the cumulative distribution funtion of the light emissions as a function fo theta, a polar angle spanning 90 to -90 degrees, 0 degrees being set on the ++z vector.
		- location: a (3) vector describing the position of the center of the arc (center of the cylinder) in local referential. (0,0,0) is assuemd to be the first focus of the ellipsoid in the SimulatorModule class.
		- direction: a (3) unit vector describing the direction of the lamp optical axis.
		'''
		self.P = eff_el*P_elec # lamp power
		self.r_c = r_c
		self.l_c = l_c
		self.loc = location # location of the lamp in local referential
		self.dir = direction # the axis of the lamp, with -z towards the reflector and +z towrds the target
		data = N.loadtxt(theta_CDF)
		integs = N.diff(data[:,1])
		dths = N.diff(data[:,0])
		PDF = integs/dths
		self.ths, self.ths_PDF = data[:-1,0]+dths/2., PDF
		
	
	def generate_rays(self, n_rays, part_load=1.):
		'''
		Creates a source in local referential.
		Arguments:
		- n_rays: number of rays
		- part_load: a multiplication factor on the source power to assuma part load operations
		'''
		P = self.P*part_load
		# vertices
		vertices = cylinder_sampling(ns=n_rays, r_ext=self.r_c, h=self.l_c, volume=True)
		# directions
		thetas, weights = PW_linear_distribution(xs=self.ths, ys=self.ths_PDF).sample(n_rays) # piecewise linear sampling of theta
		sinths = N.sin(thetas)
		phis = N.random.uniform(size=n_rays)*2.*N.pi
		directions = N.vstack([sinths*N.cos(phis), sinths*N.sin(phis), N.cos(thetas)])
		# Rotate vertices in local referential, the center of the cylinder is 0 and move them to self.loc
		vertices = rotate_z_to_normal(vertices, normal=self.dir) + N.vstack(self.loc)
		# Rotate direction vectors with direction indicating the new +z
		directions = rotate_z_to_normal(directions, normal=self.dir)
		# ray bundle
		lampsource = RayBundle(vertices=vertices, directions=directions, energy=N.ones(n_rays)*P/n_rays)
		return lampsource
		

class SimulatorLampZhu():
	def __init__(self, P_elec=7e3, eff_el=0.6, alpha_s=0.3, beta_c1=0.0412, gamma_c2=0.6588, location=[0,0,0], direction=[0,0,1]):
		'''
		Parameters:
		- P_elec: electrical power of the lamp
		- eff_el: electrical efficiency ie. conversion of eletrical into radiant power.
		- alpha_s: Fraction of the power that is assigned to the sperical arc at the cathode tip.
		- beta_c1: Fraction of the power that is assigend to the inner cylinder surface
		- gamma_c2: Fraction of the power that is assigend to the outer cylinder surface
		- location: a (3) vector describing the position of the center of the arc (center of the cylinder) in local referential. (0,0,0) is assuemd to be the first focus of the ellipsoid in the SimulatorModule class.
		- direction: a (3) unit vector describing the direction of the lamp optical axis.
		'''
		# https://www.sciencedirect.com/science/article/pii/S0306261920306772
		self.a_s = alpha_s
		self.b_c1 = beta_c1
		self.g_c2 = gamma_c2
		self.r_s = (0.5e-3)/2.
		self.r_c2 = (2e-3)/2.
		self.l_c = 10e-3
		self.P = eff_el*(alpha_s+beta_c1+gamma_c2)*P_elec # lamp power
		self.loc = location # location of the lamp with respect to the theoretical focla point.
		self.dir = direction # the axis of the lamp, with -z towards the reflector and +z towrds the target

	def generate_rays(self, n_rays, part_load=1.):
		'''
		Creates a source in local referential.
		Arguments:
		- n_rays: number of rays
		- part_load: a multiplication factor on the source power to assuma part load operations
		'''
		P = part_load*self.P
		nrays_s, nrays_c1, nrays_c2 = int(n_rays*self.a_s), int(n_rays*self.b_c1), int(n_rays*self.g_c2)

		verts_s, norms_s = sphere_sampling(r_ext=self.r_s, ns=nrays_s, normal_in=False)
		verts_s[2] -= self.l_c/2.-self.r_s
		dirs_s = isotropic_directions_sampling(num_rays=nrays_s, ang_range=N.pi/2., normals=norms_s)#sphere_sampling(r_ext=1., ns=nrays_s, normal_in=False)[1]
		energy_s = N.ones(nrays_s)*P*self.a_s/nrays_s
		sphere = RayBundle(vertices=verts_s, directions=dirs_s, energy=energy_s)
		verts_c1, norms_c1 = cylinder_sampling(r_ext=self.r_s, h=self.l_c, ns=nrays_c1, normal_in=False)
		#verts_c1[2] += self.l_c/2.-self.r_s
		dirs_c1 = isotropic_directions_sampling(num_rays=nrays_c1, ang_range=N.pi/2., normals=norms_c1)#sphere_sampling(r_ext=1., ns=nrays_c1, normal_in=False)[1]
		energy_c1 = N.ones(nrays_c1)*P*self.b_c1/nrays_c1
		cylinder_1 = RayBundle(vertices=verts_c1, directions=dirs_c1, energy=energy_c1)
		verts_c2, norms_c2 = cylinder_sampling(r_ext=self.r_c2, h=self.l_c, ns=nrays_c2, normal_in=False)
		#verts_c2[2] += self.l_c/2.-self.r_s
		dirs_c2 = isotropic_directions_sampling(num_rays=nrays_c2, ang_range=N.pi/2., normals=norms_c2)#sphere_sampling(r_ext=1., ns=nrays_c2, normal_in=False)[1]
		energy_c2 = N.ones(nrays_c2)*P*self.g_c2/nrays_c2
		cylinder_2 = RayBundle(vertices=verts_c2, directions=dirs_c2, energy=energy_c2)

		lampsource = concatenate_rays([sphere, cylinder_1, cylinder_2])
		# Rotate vertices in local referential, the center of the cylinder is 0 and move them to self.loc
		lampsource.set_vertices(rotate_z_to_normal(lampsource.get_vertices(), normal=self.dir) + N.vstack(self.loc))
		# Rotate direction vectors with direction indicating the new +z
		lampsource.set_directions(rotate_z_to_normal(lampsource.get_directions(), normal=self.dir))
				
		return lampsource
