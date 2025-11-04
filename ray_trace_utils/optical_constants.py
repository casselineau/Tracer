import numpy as N
from ray_trace_utils.electromagnetics import Drude_Lorentz_model, dielectric_to_refractive
from scipy.interpolate import interp1d
from os import walk

Sopra_data_loc = '/'.join(__file__.split('/')[:-1])+'/Sopra_Data'
other_materials_loc = '/'.join(__file__.split('/')[:-1])+'/other_material_data'

def get_from_Sopra(material):
	'''
	Automatically creates an optical_material subclass with the name given in material (a string) and the properties of this material in the Sopra database, then returns a classe instance of this material.
	The material string argument is not case sensitive.
	
	'''
	class Newmat(optical_material):
		'''
		We use the decorator to declare data from Sopra database directly. Requires the instance to be created with source='Sopra' argument.
		'''
		@optical_material.source_check
		def __init__(self):
			optical_material.__init__(self, self.l_min, self.l_max)

		@optical_material.check_valid
		@optical_material.check_m_source
		def m(self, lambdas):
			pass

	classdict = {}
	for e in Newmat.__dict__.items():
		classdict.update({e[0]:e[1]})
	globals()[material] = type(material, (optical_material,), classdict)
	return eval(material)(source='Sopra')


class optical_material(object):

	def __init__(self, l_min, l_max, force_bounds=False):
		self.l_min, self.l_max = l_min, l_max
		self.force_bounds = force_bounds

	@staticmethod
	def check_valid(prop):
		def wrapper_check_valid(self, lambdas):
			if hasattr(lambdas,'__len__'):
				l_low = (lambdas<self.l_min).any()
				l_high = (lambdas>self.l_max).any()
			else:
				l_low = (lambdas<self.l_min)
				l_high = (lambdas>self.l_max)
			if l_low or l_high:
				print ("Wavelength outside of data or model range of validity")
				print (self.l_min, self.l_max)
				print (l_low, l_high)
				valid = N.logical_and(lambdas>=self.l_min, lambdas<=self.l_max)
				data = N.ones(len(lambdas))*N.nan
				data[valid] = prop(self, lambdas[valid])
				return data
			return prop(self, lambdas)
		return wrapper_check_valid

	@staticmethod
	def check_m_source(prop):
		def wrapper_m_source(self, lambdas):
			if hasattr(self, 'm_func'):
				return self.m_func(lambdas)
			else:
				return prop(self, lambdas)
		return wrapper_m_source

	@staticmethod
	def source_check(init):
		def wrapper_source_check(self, source):
			if source == 'Sopra':
				filename = self.__class__.__name__.upper()+'.txt'
				data = N.loadtxt(Sopra_data_loc+'/'+filename, skiprows=1, delimiter=',')
				lambdas, m = data[:,0]*1e-9, data[:,1] + 1j*data[:,2]
				self.l_min, self.l_max = N.amin(lambdas), N.amax(lambdas)
				self.m = interp1d(lambdas, m)
				init(self)
				'''
				# Not tested yet
				if source == 'Other_materials':
					filenames = next(walk(other_materials_loc), (None, None, []))[2]  # [] if no file
					filenames = filenames[self.__class__.__name__.upper() in f for f in filenames]
					lambdas_list, m_list = [], []
					for f in filenames:
						data = N.loadtxt(other_materials_loc+'/'+f, skiprows=1, delimiter=',')
						lambdas_list.append(data[:,0])
						m_list.append = data[:,1] + 1j*data[:,2]
					lambdas = 				
					self.l_min, self.l_max = N.amin(lambdas), N.amax(lambdas)
					self.m = interp1d(lambdas, m)
					init(self)
				'''
			else:
				return init(self)
		return wrapper_source_check

class Al2O3(optical_material):

	def __init__(self):
		data = N.loadtxt(other_materials_loc+'/Al2O3_Querry-o.csv', skiprows=1, delimiter=',')
		lambdas, m = data[:,0], data[:,1] + 1j*data[:,2]
		l_min, l_max = N.amin(lambdas), N.amax(lambdas)
		self.m_func = interp1d(lambdas, m)
		optical_material.__init__(self, l_min, l_max)

	@optical_material.check_valid
	def m(self, lambdas):
		'''
		Optical constants used by the optical manager in tracer
		'''
		return self.m_func(lambdas)


class OpticalMaterialFromFile(optical_material):
	def __init__(self, filename, wavelength_col=0, n_col=1, k_col=2, wavelength_unit='nm'):
		data = N.loadtxt(filename, skiprows=1, delimiter=',', usecols=(wavelength_col, n_col, k_col))

		unit_factor = 1
		if wavelength_unit == 'nm':
			unit_factor = 1e-9
		elif wavelength_unit == 'um':
			unit_factor = 1e-6
		elif wavelength_unit == 'm':
			unit_factor = 1
		else:
			raise ValueError("Invalid wavelength unit. Use 'nm', 'um', or 'm'.")

		lambdas, m = data[:,0]*unit_factor, data[:,1] + 1j*data[:,2]
		l_min, l_max = N.amin(lambdas), N.amax(lambdas)
		self.m_func = interp1d(lambdas, m)
		optical_material.__init__(self, l_min, l_max)

	@optical_material.check_valid
	def m(self, lambdas):
		'''
		Optical constants used by the optical manager in tracer
		'''
		return self.m_func(lambdas)


class Air(optical_material):

	def __init__(self):
		# simplified air/vaccum placeholder returning 1.
		optical_material.__init__(self, 200e-9, 10e-6)

	@optical_material.check_valid
	def m(self, lambdas):
		'''
		Optical constants used by the optical manager in tracer
		'''
		return 1.
		
class SiO2(optical_material):

	@optical_material.source_check
	def __init__(self, source=None):
		l_min, l_max = 210e-9, 6.7e-6
		optical_material.__init__(self, l_min, l_max)

	@optical_material.check_valid
	@optical_material.check_m_source
	def m(self, lambdas):
		'''
		Optical constants used by the optical manager in tracer
		'''
		n, k = N.zeros(len(lambdas)), N.zeros(len(lambdas))
		# Malitson from https://refractiveindex.info/?shelf=main&book=SiO2&page=Malitson
		Malitson = N.logical_and(lambdas>=self.l_min, lambdas<6.7e-6)
		L2 = (lambdas[Malitson]*1e6)**2
		n[Malitson] = N.sqrt(1.+0.6961663*L2/(L2-0.0684043**2)+0.4079426*L2/(L2-0.1162414**2)+0.8974794*L2/(L2-9.896161**2))
		k[Malitson] = 0.
		return n+1j*k

class Cu(optical_material):

	@optical_material.source_check
	def __init__(self):
		l_min, l_max = 207e-9, 12.4e-6
		optical_material.__init__(self, l_min, l_max)

	@optical_material.check_valid
	@optical_material.check_m_source
	def m(self, lambdas):
		'''
		Optical constants used by the optical manager in tracer
		'''
		# https://refractiveindex.info/?shelf=main&book=Cu&page=Rakic-LD
		copper = N.array([[8.21225411, 0., -0.030], [2.67481269, 0.291, -0.378], [3.49257006, 2.957, -1.056], [9.20868474, 5.300, -3.213], [8.65045191, 11.18, -4.305]])
		return dielectric_to_refractive(Drude_Lorentz_model(lambdas, copper))

class Al(optical_material):

	@optical_material.source_check
	def __init__(self):
		l_min, l_max = 62e-9, 248e-6
		optical_material.__init__(self, l_min, l_max)

	@optical_material.check_valid
	@optical_material.check_m_source
	def m(self, lambdas):
		'''
		Optical constants used by the optical manager in tracer
		'''
		# https://refractiveindex.info/?shelf=main&book=Cu&page=Rakic-LD
		aluminium = N.array([[10.83334709, 7.13714865, 3.34962983,  6.10331602, 2.59461211],[0., 0.333, 0.312, 1.351, 3.382],[-0.047, -0.333, -0.312, -1.351, -3.382]]).T

		return dielectric_to_refractive(Drude_Lorentz_model(lambdas, aluminium))

class Ti(optical_material):

	@optical_material.source_check
	def __init__(self):
		l_min, l_max = 248e-9, 31e-6
		optical_material.__init__(self, l_min, l_max)

	@optical_material.check_valid
	@optical_material.check_m_source
	def m(self, lambdas):
		'''
		Optical constants used by the optical manager in tracer
		'''
		# https://refractiveindex.info/?shelf=main&book=Cu&page=Rakic-LD
		titanium = N.array([[2.8045189961916823, 6.912058007569092, 4.570080010240521, 3.1524509036621016, 0.23053004142627484], [0.0, 2.276, 2.518, 1.663, 1.762], [-0.082, -2.276, -2.518, -1.663, -1.762]]).T

		return dielectric_to_refractive(Drude_Lorentz_model(lambdas, titanium))

class Rh(optical_material):

	@optical_material.source_check
	def __init__(self):
		data = N.loadtxt('/media/ael/Flashy/backup_05-06-2021/Documents/Boulot/Material_properties/Rhodium.csv', skiprows=1)
		lambdas, m = data[:,0]*1e-6, data[:,1] + 1j*data[:,2]
		l_min, l_max = N.amin(lambdas), N.amax(lambdas)
		self.m_func = interp1d(lambdas, m)
		optical_material.__init__(self, l_min, l_max)

	@optical_material.check_valid
	@optical_material.check_m_source
	def m(self, lambdas):
		'''
		Optical constants used by the optical manager in tracer
		'''
		return self.m_func(lambdas)

class Ta(optical_material):
	'''
	Mixed sources to cover a large spectrum. data from refractiveindex.com
	Until 2.48 um: W. S. M. Werner, K. Glantschnig, C. Ambrosch-Draxl. Optical constants and inelastic electron-scattering data for 17 elemental metals. J. Phys Chem Ref. Data 38, 1013-1092 (2009)
	Then: M. A. Ordal, R. J. Bell, R. W. Alexander, L. A. Newquist, M. R. Querry. Optical properties of Al, Fe, Ti, Ta, W, and Mo at submillimeter wavelengths. Appl. Opt. 27, 1203-1209 (1988)
	'''
	def __init__(self):
		data = N.loadtxt(other_materials_loc+'/Ta.csv', skiprows=1, delimiter=',')
		lambdas, m = data[:,0], data[:,1] + 1j*data[:,2]
		l_min, l_max = N.amin(lambdas), N.amax(lambdas)
		self.m_func = interp1d(lambdas, m)
		optical_material.__init__(self, l_min, l_max)

	@optical_material.check_valid
	def m(self, lambdas):
		'''
		Optical constants used by the optical manager in tracer
		'''
		return self.m_func(lambdas)

class W(optical_material):
	'''
	Mixed sources to cover a large spectrum. data from refractiveindex.com
	Until 2.48 um: W. S. M. Werner, K. Glantschnig, C. Ambrosch-Draxl. Optical constants and inelastic electron-scattering data for 17 elemental metals. J. Phys Chem Ref. Data 38, 1013-1092 (2009)
	Then: M. A. Ordal, R. J. Bell, R. W. Alexander, L. A. Newquist, M. R. Querry. Optical properties of Al, Fe, Ti, Ta, W, and Mo at submillimeter wavelengths. Appl. Opt. 27, 1203-1209 (1988)
	'''
	def __init__(self):
		data = N.loadtxt(other_materials_loc+'/W.csv', skiprows=1, delimiter=',')
		lambdas, m = data[:,0], data[:,1] + 1j*data[:,2]
		l_min, l_max = N.amin(lambdas), N.amax(lambdas)
		self.m_func = interp1d(lambdas, m)
		optical_material.__init__(self, l_min, l_max)

	@optical_material.check_valid
	def m(self, lambdas):
		'''
		Optical constants used by the optical manager in tracer
		'''
		return self.m_func(lambdas)

if __name__ == '__main__':
	from sys import path
	import matplotlib.pyplot as plt
	from matplotlib.lines import Line2D

	plot_metals = 0
	test_autogen = 1

	if plot_metals:
		# Metals
		plt.figure()
		lambdas = N.linspace(250e-9, 10e-6, 200)
		metals = ['Cu', 'Al', 'Ti', 'Rh']
		colors = ['orangered', '0.5', 'b', 'g']
		for i, met in enumerate(metals):
			metal = globals()[met]()
			n = metal.m(lambdas)
			plt.plot(lambdas*1e6, n.real, color=colors[i], label=met)
			plt.plot(lambdas*1e6, n.imag, color=colors[i], linestyle='--')

		han = [Line2D([],[],color='k'), Line2D([],[], linestyle='--',color='k')]
		lab = ['${n}$', '${k}$']
		leg2 = plt.legend(han, lab, loc=2, bbox_to_anchor=(0.16,1))

		plt.legend(title='Metals', loc=2)
		plt.gca().add_artist(leg2)
		plt.xlabel('${\lambda}$ ${\mu m}$')
		plt.ylabel('${n,k}$')
		plt.savefig(path[0]+'/refractive_indices_metals.png')

	if test_autogen:
		mat = Ta('Sopra')
