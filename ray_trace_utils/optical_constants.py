import numpy as N
from ray_trace_utils.electromagnetics import Drude_Lorentz_model, dielectric_to_refractive
from scipy.interpolate import interp1d

Sopra_data_loc = '/'.join(__file__.split('/')[:-1])+'/Sopra_Data'

def get_from_Sopra(material):

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
				return N.nan
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
			else:
				return init(self)
		return wrapper_source_check
				
		
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
		# https://refractiveindex.info/?shelf=main&book=Rh&page=Weaver
		return self.m_func(lambdas)

class Ta(optical_material):
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
