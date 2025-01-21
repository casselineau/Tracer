'''
__________________________________________________________________________________________________
*********** 2 N Parameters cavity receiver  ************
__________________________________________________________________________________________________
'''
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

#from freesteam import *
from Tube_materials import *

from emissive_losses.emissive_losses import *
from emissive_losses.view_factors_3D import *

class TwoNparamcav(Assembly):
	'''
	TwoNparamcav object initializes the scene asembly for the geometrical and optical parameters given.
	Several methods can be called to perform energy balance calculations.
	- self.bin_hits(): Assigns the resulst of a ray-trace to the receiver surface elements.
	- self.VF_sim(): Receiver view factor calculation method.
	- self.energy_balance(): Evaluates the energy balance of the receiver using an iterative solution scheme based on the following two methods.
	- self.temperature_guess(): Evaluates the temperature of the tubes considering water/steam flow conditions, incident ratiation and thermal emissions.
	- self.emi_sim(): performs an emissive losses calculation and returns the total emissive losses.
	'''
	def __init__(self, apertureRadius, frustaRadii, frustaDepths, coneDepth, eps_wall, absReceiver, emsReceiver, aperture_position=0., envelope_radius=None, envelope_depth=None):
		'''
		Initialisation of the geometry and surface properties of the scene.

		The concentrator is a dish, the receiver is defined by a set of 2N parameters and housed in a cylindrical case.

		Arguments:
		- apertureRadius: radius of the aperture in (m).
		- frustaRadii: list of radii for the frusta elements in (m).
		- frustaDepths: list of depths for the frusta elements in (m).
		- coneDepth: depth of the cone element closing the geometry in (m).
		- eps_wall: minimum thickness of the insulation layer around the receiver in (m).
		- absReceiver: absorptivity of the walls of the cavity in the visible wavelength region. Can be a single value for a constant wall absorptivity or an array/list for a description of absorptivities element per element.
		- emsReceiver: emissivity of the walls of the cavity in the thermal emissions wavelength region. Can be a single value for a constant wall emissivity or an array/list for a description of emissivities element per element.

		- aperture_position: position of the receiver aperture plane on the z axis, axis of symmetry of the receiver and dish in (m).
		- envelope_radius: overrides the wall thickness based determination of the cylindrical envelope radius in (m).
		- envelope_depth: overrides the wall thickness based determination of the cylindrical envelope depth in (m).
		'''
		Assembly.__init__(self, objects=None, subassemblies=None, location=None, rotation=None)

		self.apertureRadius = apertureRadius
		self.frustaRadii = frustaRadii
		self.frustaDepths = frustaDepths # frusta depths
		self.coneDepth = coneDepth # cone depth

		if (type(absReceiver)==float) or (type(absReceiver)==int):
			self.absReceiver = N.hstack((1.,N.ones(len(frustaRadii)+1)*absReceiver))
		else:
			self.absReceiver = absReceiver
		assert(len(self.absReceiver)==(len(frustaRadii)+2))
		self.emsReceiver = emsReceiver

		Envelope = Assembly()
		self.Active_zone = Assembly()

		self.aperture_position = aperture_position

		# Envelope ----------------------------------------
		trr = translate(z=aperture_position) # Receiver frame transformation

		# Envelope maximum radius is the maximum between the aperture outer radius (to allow for pancakes) and the radii of frusta +the wall insulation thickness.
		if envelope_radius == None:		
			self.envelope_radius = N.amax(N.hstack([self.apertureRadius,N.array(self.frustaRadii)+eps_wall]))
		else:
			self.envelope_radius = envelope_radius
		# Envelope maximum depth is the sum of all depths of the frusta + the wall insulation thickness. If the cone goes out, its depth has to be added too.
		if envelope_depth == None:
			max_depth = N.amax(N.add.accumulate(N.hstack([self.frustaDepths, self.coneDepth])))
			self.envelope_depth = max_depth+eps_wall
		else: 
			self.envelope_depth = envelope_depth

		if self.envelope_radius != self.apertureRadius:
			Envelope_front = AssembledObject(surfs=[Surface(RoundPlateGM(Ri=self.apertureRadius, Re=self.envelope_radius), LambertianReceiver(1))], transform=None)
			Envelope.add_object(Envelope_front)

		Envelope_back = AssembledObject(surfs=[Surface(RoundPlateGM(Re=self.envelope_radius), LambertianReceiver(1))], transform=translate(z=self.envelope_depth))
		Envelope_cylinder = AssembledObject(surfs=[Surface(FiniteCylinder(diameter=2.*self.envelope_radius, height=self.envelope_depth), LambertianReceiver(1))], transform=translate(z=self.envelope_depth/2.))

		Envelope.add_object(Envelope_back)
		Envelope.add_object(Envelope_cylinder)

		# Receiver inner surface ----------------------------------------
		if self.frustaRadii == []:
			# Cone
			if self.coneDepth < 0.: # self.coneDepth < 0 Inward cone
				CON = AssembledObject(surfs=[Surface(FiniteCone(r=self.apertureRadius, h=-self.coneDepth), LambertianReceiver(self.absReceiver[-1]))], transform=translate(z=N.sum(frustaDepths)+self.coneDepth))

			elif self.coneDepth == 0.: # Round flat plates
				CON = AssembledObject(surfs=[Surface(RoundPlateGM(Re=self.apertureRadius), LambertianReceiver(self.absReceiver[-1]))], transform=translate(z=N.sum(frustaDepths)))

			else:	 # == cone depth > 0: Outgoing cone
				trc = N.dot(rotx(N.pi), translate(z=-N.sum(frustaDepths)-self.coneDepth)) # Cone frame transformation
				CON = AssembledObject(surfs=[Surface(FiniteCone(r=self.apertureRadius, h=self.coneDepth), LambertianReceiver(self.absReceiver[-1]))], transform=trc)
		
		else:
			# Cone
			if self.coneDepth < 0.: # self.coneDepth < 0 Inward cone
				CON = AssembledObject(surfs=[Surface(FiniteCone(r=self.frustaRadii[-1], h=-self.coneDepth), LambertianReceiver(self.absReceiver[-1]))], transform=translate(z=N.sum(frustaDepths)+self.coneDepth))

			elif self.coneDepth == 0.: # Round flat plates
				CON = AssembledObject(surfs=[Surface(RoundPlateGM(Re=self.frustaRadii[-1]), LambertianReceiver(self.absReceiver[-1]))], transform=translate(z=N.sum(frustaDepths)))

			else:	 # == cone depth > 0: Outgoing cone
				trc = N.dot(rotx(N.pi), translate(z=-N.sum(frustaDepths)-self.coneDepth)) # Cone frame transformation
				CON = AssembledObject(surfs=[Surface(FiniteCone(r=self.frustaRadii[-1], h=self.coneDepth), LambertianReceiver(self.absReceiver[-1]))], transform=trc)

			FRU = []
			# 1st frustum:
			if self.apertureRadius==self.frustaRadii[0]: # Cylinder
				frustum = AssembledObject(surfs=[Surface(FiniteCylinder(diameter=self.apertureRadius*2, height=self.frustaDepths[0]), LambertianReceiver(self.absReceiver[1]))], transform=translate(z=self.frustaDepths[0]/2.))
			elif self.frustaDepths[0] == 0.:
				frustum = AssembledObject(surfs=[Surface(RoundPlateGM(Re=self.apertureRadius, Ri=self.frustaRadii[0]), LambertianReceiver(self.absReceiver[1]))], transform=translate(z=N.sum(self.frustaDepths[:0])))
			else:
				frustum = AssembledObject(surfs=[Surface(ConicalFrustum(z1=0., r1=self.apertureRadius, z2=self.frustaDepths[0], r2=self.frustaRadii[0]), LambertianReceiver(absorptivity=self.absReceiver[1]))], transform=None)
			FRU.append(frustum)
			# next frusta:
			for i in xrange(1,len(frustaRadii)):
				if self.frustaRadii[i-1]==self.frustaRadii[i]: # Cylinder
					frustum = AssembledObject(surfs=[Surface(FiniteCylinder(diameter=self.frustaRadii[i-1]*2, height=self.frustaDepths[i]), LambertianReceiver(self.absReceiver[1+i]))], transform=translate(z=N.sum(self.frustaDepths[:i])+self.frustaDepths[i]/2.))
				elif self.frustaDepths[i] < 0.:
					frustum = AssembledObject(surfs=[Surface(ConicalFrustum(z1=0., r1=self.frustaRadii[i-1], z2=-self.frustaDepths[i], r2=self.frustaRadii[i]), LambertianReceiver(absorptivity=self.absReceiver[1+i]))], transform=N.dot(translate(z=N.sum(self.frustaDepths[:i])),rotx(N.pi)))
				elif self.frustaDepths[i] > 0.:
					frustum = AssembledObject(surfs=[Surface(ConicalFrustum(z1=0., r1=self.frustaRadii[i-1], z2=self.frustaDepths[i], r2=self.frustaRadii[i]), LambertianReceiver(absorptivity=self.absReceiver[1+i]))], transform=translate(z=N.sum(self.frustaDepths[:i])))
				else:
					frustum = AssembledObject(surfs=[Surface(RoundPlateGM(Re=self.frustaRadii[i-1], Ri=self.frustaRadii[i]), ReflectiveReceiver(self.absReceiver[-1]))], transform=translate(z=N.sum(self.frustaDepths[:i])))

				FRU.append(frustum)
				
			# Receiver Assembly ----------------------------------------

			self.CON = CON
			self.FRU = FRU
			for i in xrange(len(FRU)):
				self.Active_zone.add_object(FRU[i])
		self.Active_zone.add_object(CON)	
		self.add_assembly(Envelope, transform=trr)
		self.add_assembly(self.Active_zone, transform=trr)

	def VF_sim(self, bins_frusta, bins_cone, num_rays, precision):
		'''
		Calculates view factors and areas for a Two_N_parameters_cavity instance.
		
		Arguments:
		- bins_frusta: array/list of the number of elements per frusta sections used in the radiative power binning process. Every bin on a single frustum element has the same depth on the z axis (symmetry axis of the receiver geometry).
		- bins_cone: number of elements per frusta sections used in the radiative power binning process. Every bin on the cone section has the same depth on the z axis (symmetry axis of the receiver geometry).
		- precision: precision required for the view factor calculations.

		Returns:
		- self.VF: view factor matrix.
		- self.areas: array containing the areas of each of the bins in (m2).
		'''
		# Emissivity array shaping. Aperture value has to be 1 for an open receiver. Takes place before the VF calculation to avoid spending useless time on this if the array is not shaped properly.
		if (type(self.emsReceiver)!=N.ndarray) and (type(self.emsReceiver)!=list):
			self.emsReceiver = N.hstack(N.append(1.,N.ones(N.sum(N.array(bins_frusta))+bins_cone)*self.emsReceiver))
		#assert(len(self.emsReceiver)==N.sum(N.array(bins_frusta))+bins_cone+1)

		# Calculate geometry view factor
		vfcase = Two_N_parameters_cavity_RTVF(self.apertureRadius, self.frustaRadii, self.frustaDepths, self.coneDepth, el_FRUs=bins_frusta, el_CON=bins_cone, num_rays=num_rays, precision=precision)

		self.bins_frusta = bins_frusta
		self.bins_cone = bins_cone
		self.VF = vfcase.VF_esperance
		self.areas = vfcase.areas

		return self.VF, self.areas, self.bins_frusta, self.bins_cone

	def bin_hits(self):

		int_walls = self.Active_zone.get_surfaces()
		self.bin_abs = N.zeros(len(self.areas)-1)
		index = 0 # to track which element of the profile we are dealing with in absolute terms.

		# Bin Frusta:
		for i in xrange(len(self.frustaDepths)):
			# Adjust the hits detector position to take into account the placement of the receiver in comparision with the focal plane.
			#z1 = N.sum(self.frustaDepths[:i+1])-self.frustaDepths[i]+self.aperture_position
			#z2 = N.sum(self.frustaDepths[:i+1])+self.aperture_position

			if i == 0:
				r1 = self.apertureRadius
			else:
				r1 = self.frustaRadii[i-1]
			r2 = self.frustaRadii[i]

			abs, hits = int_walls[i].get_optics_manager().get_all_hits()

			hits = int_walls[i].global_to_local(hits)
			heights = N.around(hits[2], decimals=9)

			if self.frustaDepths[i] < 0.:
				heights = -heights

			rads = N.around(N.sqrt(hits[0]**2.+hits[1]**2.), decimals=9)
	
			for j in xrange(self.bins_frusta[i]):
				index+=1
				z1bin = j*self.frustaDepths[i]/self.bins_frusta[i]
				z2bin = (j+1)*self.frustaDepths[i]/self.bins_frusta[i]

				r1bin = N.around(r1+j*(r2-r1)/self.bins_frusta[i], decimals=9)
				r2bin = N.around(r1+(j+1)*(r2-r1)/self.bins_frusta[i], decimals=9)

				if z1bin>z2bin:
					z1bin,z2bin = z2bin,z1bin

				if r1bin>r2bin:
					r1bin,r2bin = r2bin,r1bin

				test_depth = N.logical_and(heights>z1bin, heights<=z2bin)
				test_rads = N.logical_and(rads>=r1bin, rads<=r2bin)

				self.bin_abs[index-1] = N.sum(abs[N.logical_and(test_depth, test_rads)])

		# Bin the cone:
		abs, hits = int_walls[-1].get_optics_manager().get_all_hits()
		hits = N.around(hits, decimals=9)
		hits = int_walls[i].global_to_local(hits)

		for i in xrange(self.bins_cone):
			index+=1
			r1 = self.frustaRadii[-1]*(1.-float(i)/self.bins_cone)
			r2 = self.frustaRadii[-1]*(1.-(i+1.)/self.bins_cone)
			cone_hits_radii = N.around(N.sqrt(hits[0]**2+hits[1]**2), decimals=9)
			section = N.logical_and(cone_hits_radii<=r1, cone_hits_radii>=r2)
			self.bin_abs[index-1] = N.sum(abs[section])

		return self.bin_abs

	def temperature_guess(self, T_in, p_in, T_out, tube_diameters_in, tube_diameters_out, tube_conductivity, emissions_guess, coating_thickness, coating_conductivity, tube_roughness, uconvloss, passive=None, tube_material=None):
		'''
		Makes a first guess on temperature profiles approximating the enthalpy gain of the water/steam mixture to be equal to flux input on the tubes external walls. The tube walls are coated with a selective coating. Default arguments are for Pyromark2500(R).

		Arguments:
		- T_in: Inlet temperature of the water  in (K).
		- p_in: Inlet pressure of the water in (pa).
		- T_out: Outlet temeprature of the water in (K).
		- tube_diameters_in: inner diameter of the tubes in (m).
		- tube_diameters_out: outer diameter of the tubesin (m).
		- tube_conductivity: thermal conductivity of teh tubes in (W/mK).
		- emissions_guess: emissive losses guess to account for thermal emissions in (W).
		- passive: array of the indices of the adiabatic surfaces in the cavity.
		- coating_thickness: thickness of the coating layer on the tubes in (m).
		- coating_conductivity: coating thermal conductivity in (W/mK).

		Returns:
		- strings 'good_geom' or 'bad_geom' depending on the mass flow guess to meet the input/output arguments and amount of actually going in the receiver. This is a quick hack to prevent issues with receivers forcing in the required inpu/output by lowering the mass flow too much/putting it negative, thus impacting the enthalpy guess... and basically screwing-up the convergence process for non-performing geometries.
		'''
		# Get active surfaces net radiative power
		active = N.ones(len(self.areas), dtype = bool)
		active[0] = 0. # aperture

		if passive != None:
			active[passive] = 0.

		# Check the tube_diameters arrays or make one:
		tube_diameters_in = N.hstack([tube_diameters_in])
		tube_diameters_out = N.hstack([tube_diameters_out])
		if len(tube_diameters_in)!=(N.sum(active)+1):
			if len(tube_diameters_in)==1:
				tube_diameters_in = N.ones(N.sum(active)+1)*tube_diameters_in[0]
			else:
				stop
			if len(tube_diameters_out)==1:
				tube_diameters_out = N.ones(N.sum(active)+1)*tube_diameters_out[0]
			else:
				stop

		# Equivalent tube sections radii:
		R_in = (tube_diameters_in[:-1]+tube_diameters_in[1:])/4.
		R_out = (tube_diameters_out[:-1]+tube_diameters_out[1:])/4.
		self.R_in = R_in
		self.R_out = R_out

		# Get tubes lengths and positions:
		tube_lengths = N.array(self.areas[active]/(2.*R_out))
		self.tube_lengths = tube_lengths+2.*N.pi*(R_out-R_in)
		tube_positions = N.add.accumulate(N.hstack([0,self.tube_lengths]))

		# Initialise pressures and steam quality at each position along the flow path:
		self.p = N.ones(len(tube_positions))*p_in
		self.qual = N.zeros(len(tube_positions)-1)
		self.v = N.zeros(len(tube_positions))

		# Correlation single phase:
		def single_phase_u(Re, Pr, f_F, k, tube_D):
			if Re<1e4:
			#if Re<1e5: # STG code error
				# Gnielinski:
				return ((Re-1000.)*Pr*(f_F*k/(2.*tube_D)))/(1.+12.7*(Pr**(2./3.)-1.)*N.sqrt(f_F/2.))
			else:
				# Petukhov
				return (Re*Pr*(f_F*k/(2.*tube_D))/(1.07+12.7*(Pr**(2./3.)-1.)*N.sqrt(f_F/2.)))

		# Evaluate convective losses and qnets:
		Qconvloss = uconvloss*self.areas[1:]*(self.T[1:]-self.T[0])
		qnets = self.bin_abs[active[1:]]-emissions_guess[active[1:]]-Qconvloss[active[1:]]

		# Get starting enthalpy via Freesteam and initialise enthalpy array:
		h_in = steam_pT(p_in,T_in).h
		h_out = steam_pT(self.p[-1], T_out).h
		hs_p = h_in+N.add.accumulate(N.hstack([0,qnets])/N.sum(qnets))*(h_out-h_in)
		self.h = N.ones(len(qnets)+1)*h_in

		# Enthalpy convergence loop:
		conv_h = N.ones(len(hs_p))*N.inf
		iterh = 0
		while (conv_h>1e-9).any():
			# Evaluate the mass-flow:
			self.m = N.sum(qnets)/(h_out-h_in)

			# Evaluate convective losses and qnets:
			Qconvloss = uconvloss*self.areas[1:]*(self.T[1:]-self.T[0])
			qnets = 	self.bin_abs[active[1:]]-emissions_guess[active[1:]]-Qconvloss[active[1:]]

			# Initialise internal convective heat trasnfer coefficient:
			uconv = N.zeros(len(tube_diameters_in)-1)

			# Go through the flow-path, actualise the pressures and evaluate the heat transfer coefficients.
			for i in xrange(len(tube_positions)-1):

				# Evaluate the steam properties:
				steam_state = steam_ph(self.p[i], hs_p[i])
				rho = steam_state.rho
				Cp = steam_state.cp
				x = steam_state.x

				Tsat = Tsat_p(self.p[i])
				steam_L = steam_Tx(Tsat, 0.)
				steam_G = steam_Tx(Tsat, 1.)
				h_LG = steam_G.h-steam_L.h	

				qual = (hs_p[i]-steam_L.h)/h_LG
				v = self.m/(rho*N.pi*(tube_diameters_in[i]/2.)**2.)
				if qual<=0.:
					mu = steam_state.mu
					k = steam_state.k
					# Calculate Reynolds, Prandtl, Darcy and Fanning friction factors
					Re = rho*v*tube_diameters_in[i]/mu
					Pr = mu/(k/Cp)

				S = N.log(Re/(1.816*N.log(1.1*Re/(N.log(1.+1.1*Re)))))
				f_D = (-2.*N.log10(tube_roughness/(3.71*tube_diameters_in[i])+2.18*S/Re))**(-2.) # Brkic using Lambert W-function approximation to solve Colebrook's implicit equation.
				f_F = 0.25*f_D
				
				# Calculate heat transfer coefficient:
				uconv[i] = single_phase_u(Re, Pr, f_F, k, tube_diameters_in[i])
			
				rho_L = steam_L.rho
				rho_G = steam_G.rho

				mult = 1. # Carey Friction factor multiplier

				if ((qual>0.) and (qual<1.)):
					mu_L = steam_L.mu
					mu_G = steam_G.mu
					#v_L = self.m/(rho_L*N.pi*(tube_diameters_in[i]/2.)**2.)
					#Re_L = rho_L*v_L*tube_diameters_in[i]/mu_L
					#mult = (1.+(mu_L/mu_G-1.)*qual)**(-0.25) # Carey Friction factor multiplier
					#f_D = 4.*mult*(1.58*N.log(Re_L)-3.28)**(-2) # Correlation in Kandlikar (less precise presumably) which gives significantly lower heat transfer coefficients in the pre-dryout region

					if (qual<0.8):

						k_L = steam_L.k
						Cp_L = steam_L.cp
						v_L = self.m/(rho_L*N.pi*(tube_diameters_in[i]/2.)**2.)

						h_LG = steam_G.h-steam_L.h

						Re_L = rho_L*v_L*tube_diameters_in[i]/mu_L
						#Re_L = rho_L*v_L*(1.-qual)*tube_diameters_in[i]/mu_L # from Notes from Jose. My interpretation is all liquid in the tube (saturated then).
						Pr_L = mu_L/(k_L/Cp_L)
						S_L = N.log(Re_L/(1.816*N.log(1.1*Re_L/(N.log(1.+1.1*Re_L)))))
						f_F_L = 0.25*((-2.*N.log10(tube_roughness/(3.71*tube_diameters_in[i])+2.18*S_L/Re_L))**(-2.))
						#f_F_L = (1.58*N.log(Re_L)-3.28)**(-2) # Correlation in Kandlikar (less precise presumably) which gives significantly lower heat transfer coefficients in the pre-dryout region
						# Kandlikar
						Co = (rho_G/rho_L)**0.5*((1.-qual)/qual)**0.8

						if i == 0:
							Bo = 0.
						else:
							Bo = qnets[i]/(N.pi*R_in[i]*tube_lengths[i])/(rho*v*h_LG)
							#Bo = qnets[i]/(N.pi*R_out[i]*tube_lengths[i])/(rho*v*h_LG) # Changed to the outer flux to compare with the STG version

						uconv_L = single_phase_u(Re_L, Pr_L, f_F_L, k_L, tube_diameters_in[i])

						uconvNB = uconv_L*(0.6683*Co**(-0.2)+1058.*Bo**0.7)*(1.-qual)**0.8
						uconvCB = uconv_L*(1.136*Co**(-0.9)+667.2*Bo**0.7)*(1.-qual)**0.8

						uconv[i] = N.amax([uconvNB, uconvCB])
				
					elif qual<0.9: # Here the validity of the correlation is changed from the STG version.				
						# Groeneveld
						a = 1.09e-3
						b = 0.989
						c = 1.41
						d = -1.15

						Y = 1.-0.1*((rho_L/rho_G-1.)*(1.-qual))**0.4

						k_G = steam_G.k
						Cp_G = steam_G.cp
						v_G = self.m/(rho_G*N.pi*(tube_diameters_in[i]/2.)**2.)

						Re_G = rho_G*v_G*tube_diameters_in[i]/mu_G
						Pr_G = mu_G/(k_G/Cp_G)

						uconv[i] = a*(Re_G*(qual+rho_G/rho_L*(1.-qual)))**b*Pr_G**c*Y**d*k_G/tube_diameters_in[i]

				# Calculate pressure drop for the next element:
				#f_D = 4.*f_F # For using the friction factor formula advised by Kandlikar
				dp = f_D*self.tube_lengths[i]/(2.*R_in[i])*rho*v**2./2.
				steam_next = steam_ph(self.p[i+1], self.h[i+1])
				rho_next = steam_next.rho
				v_next = self.m/(rho_next*N.pi*(tube_diameters_in[i+1]/2.)**2.)
				self.p[i+1] = self.p[i]+rho*v**2./2.-rho_next*v_next**2./2.-dp
				self.v[i] = v

				# Store dryness fractions;
				self.qual[i] = qual

				# Re-evaluate enthalpies:
				hs_p[i+1] = hs_p[i]+qnets[i]/self.m


			# Final velocity storage:
			self.v[i+1] = v_next

			# Evaluate the enthalpy at the outlet:
			#h_out = steam_pT(self.p[-1], T_out).h
			h_out = hs_p[-1]

			#FIXME: need a more reliable convergence insurance
			if self.m < 0.01:
				print 'bad_geom'
				return 'bad_geom'

			# Evaluate convergence:
			conv_h = N.abs((self.h-hs_p)/self.h)
			self.h = (self.h+hs_p)/2.

			iterh += 1
			if iterh>100:
				print conv_h
				print self.h
				stop

		# Get the tube elements properties:
		self.uconv = uconv

		# Get temperatures from enthalpies via Freesteam
		T_guess_fluid = N.zeros(len(self.h))
		T_guess_fluid[0] = T_in
		for i in xrange(1,len(self.h)):
			T_guess_fluid[i] = steam_ph(self.p[i],self.h[i]).T

		# Central differences scheme:
		self.T_guess_fluid = (T_guess_fluid[:-1]+T_guess_fluid[1:])/2.

		T_guess_wall = N.zeros(len(self.areas)-1)

		Rconv = 1./(N.pi*tube_lengths*R_in*self.uconv)
		Rcond = 1./(N.pi*tube_lengths)*(N.log(R_out/R_in)/tube_conductivity+N.log((R_out+coating_thickness)/R_out)/coating_conductivity)

		T_guess_wall[active[1:]] = self.T_guess_fluid + qnets*(Rconv+Rcond)

		self.T_wall_in = self.T_guess_fluid + qnets*(Rconv)

		self.T_guess = T_guess_wall
		self.tube_positions = tube_positions
		self.Q_conv_loss = Qconvloss

		assert (self.T_guess==float('inf')).any()==False, str(self.T_guess)+str(T_guess_wall)+str(emissions_guess)+str(self.m)+str([self.apertureRadius, self.frustaRadii, self.frustaDepths, self.coneDepth])
		return 'good_geom'

	def emi_sim(self, Tamb, Trec, VF, areas, inc_radiation=None):
		'''
		Method to calculate the emissive losses of a Two_N_parameters_cavity instance. Calls a radiosity method using the view factors matrix calculated previously

		Arguments:
		- Tamb: temperature of the environment in (K), used for the aperture in the radiosity method.
		- Trec: array/list of temperatures of the surfaces of the receiver in (K). If only one value is declared, set all receiver surfaces to this single temperature value.
		- VF: View factors matrix.
		- areas: Areas of the elemenst of the view factor matrix (m2).
		- inc_radiation: array/list of incoming radiative power on the elements (W). If not None overrides the radiosity problem temperature boundary condition only where the inc_radiation value is not equal to 0.

		Returns: 
		- self.emissive_losses: overall emissive losses through the aperture
		The method also stores the temperatures and net emitted radiative flux and net radiative powers in the thermal wavelength region (semi-gray body assumption):
			- self.q: Net thermal radiative flux (W/m2) per element.
			- self.Q: Net thermal radiative power (W) per element.
			- self.T: Temeprature of each element (K).
		'''
		if inc_radiation != None:
			inc_radiation = N.hstack((N.nan, inc_radiation)) # To take into account the aperture in the radiosity system.
		# Solve radiosity problem
		if (Trec==None) and (Tamb!=None):
			Trec = N.nan*N.ones(VF.shape[0]-1)
		T = N.hstack((Tamb, Trec))

		AA,bb,J,Eb,T,q,Q = radiosity_RTVF(VF, areas, self.emsReceiver, T, inc_radiation)
		self.q = q
		self.Q = Q
		self.T = T
		return J,T,Q

	def energy_balance(self, Tamb, Trec_in, p_in, Trec_out, tube_diameters_in, tube_diameters_out, tube_conductivity, coating_thickness = 45e-6, coating_conductivity = 1.2, tube_roughness=45e-6, uconvloss=30., passive=None, tube_material=None):
		'''
		Method to simulate the radiative efficiency of a Two_N_parameters_cavity receiver with a realistic evaluation of the temepratures of the walls using fluid properties and the heat exchange model from the temperature_guess() method.

		Arguments:
		- Trec_in: Inlet temperature of the water  in (K).
		- p_in: Inlet pressure of the water in (pa).
		- Trec_out: Outlet temeprature of the water in (K).
		- tube_diameters_in: inner diameter of the tubes in (m).
		- tube_diameters_out: outer diameter of the tubesin (m).
		- tube_conductivity: thermal conductivity of the tubes in (W/mK).
		- passive: array of the indices of the adiabatic surfaces in the cavity.
		- tube_material: If given, it is a tube material instance that can reture temperature varying thermal conductivity. Overrides the tube_conductivity argument.

		Returns:
		The temperature of the elements, an array of zeros if the candidates are net energy destructors.
		'''
		# Iterate to find wall temperatures and respect the energy balance:
		if type(self.emsReceiver)==float:
			self.emsReceiver = N.hstack((1.,N.ones(len(self.areas)-1)*self.emsReceiver))

		self.T = N.linspace(Trec_in, Trec_out, len(self.areas))#N.ones(len(self.areas))*Trec_out
		self.T[0] = Tamb

		if tube_material!=None:
			tube_conductivity = tube_material.k(self.T[1:-self.bins_cone])

		emissions = N.ones(len(self.areas))
		convergence = N.ones(len(emissions))
		iterQ = 0

		threshold = 0.0001
		while (convergence>threshold).any():

			result_T_guess = self.temperature_guess(Trec_in, p_in, Trec_out, tube_diameters_in, tube_diameters_out, tube_conductivity, emissions[1:], coating_thickness, coating_conductivity, tube_roughness, uconvloss, passive)
			if result_T_guess == 'bad_geom': # discard 'bad_geom' geometries.
				self.T_guess = N.ones(len(self.areas))*Trec_in
				break

			self.rad_passive = None
			if passive != None:
				self.rad_passive = N.zeros(N.shape(self.bin_abs))
				self.rad_passive.fill(N.nan)
				self.rad_passive[passive] = (self.bin_abs[passive]-self.Q_conv_loss[passive])/self.areas[1:][passive]
				self.T_guess[passive] = N.nan

			self.emi_sim(Tamb=Tamb, Trec=self.T_guess, VF=self.VF, areas=self.areas, inc_radiation=self.rad_passive)

			self.T_guess[convergence[1:]>threshold] = self.T[1:][convergence[1:]>threshold]
			self.T_fluid = self.T_guess_fluid

			if tube_material!=None:
				tube_conductivity = tube_material.k((self.T_guess[:-self.bins_cone]+self.T_fluid)/2.)

			convergence = N.abs((self.Q-emissions)/emissions)#N.abs((self.Q-emissions)/self.Q)
			if iterQ>30:
				convergence = N.abs((self.Q-emissions))
				threshold = 100.
				print iterQ
				#print convergence

			emissions = (self.Q+emissions)/2.
			iterQ += 1

		print 'Final T guess:', self.T
		return result_T_guess
