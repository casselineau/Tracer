from math import pi, sqrt, tan
import numpy as N
import scipy.linalg as S

def radiosity_RTVF(VF, areas, eps, T=None, inc_radiation=None, q_net=None):
	'''
	Solve the radiosity problem depending on the geometry and a specified boundary condition.

	Arguments:
	VF - view factor matrix of the geometry calculated
	areas - areas of the view factor elements (m2)
	eps - emissivity of the surfaces, can be an array decribing descrete emissivity values or a single number applicable to all surfaces except the aperture.
	Tamb - ambiant temperature (K)
	Twall - wall boundary condition, can be an array decribing descrete temperature values or a single number applicable to all surfaces except the aperture (K)
	inc_radiation - specifies the ratiative power coming to the surface (W/m2) and is used if no temperature is given to determine the thermal emissions.
	q_net - specifies the net radiative heat enforced in the surface. q_net is subtracted from inc_radiation if the latter is declared.

	Returns:
	J - Radiosities (W/m2)
	E - black body emission flux (W/m2)
	T - Temperatures of the elements (K)
	q - Net radiative flux (W/m2) fraction in the surface after absorption and emission.
	Q - Net radiative power (W)
	''' 
	A = areas
	n = N.shape(VF)[0]
	sigma = 5.6677e-8 # Stefan-Boltzman constant

	if len(eps) != len(areas):
		raise AttributeError
	if (T == None) and (inc_radiation == None):
		raise AttributeError
	# The radiosity problem is formulated as [AA][J]=[bb], and the following matrices are
	# defined:
	AA = N.zeros((n,n)) 
	bb = N.zeros(n)
	AA[N.diag_indices(n)] = 1.
	# Boundary conditions. Error raised if none or both of flux and tempreature are fixed at the boundary.
	if (inc_radiation!=None) and (T!=None):
		# If no BC in any element, raise error:
		if N.logical_and(N.isnan(T), N.isnan(inc_radiation)).any():
			raise AttributeError('At least one element has no boundary condition for radiosity')
		# If BC has double definition, raise error:
		if N.logical_and(~N.isnan(T), ~N.isnan(inc_radiation)).any():
			raise AttributeError('At least one element has two boundary condition definitions for radiosity')
	if (inc_radiation != None): # Flux
		bb[~N.isnan(inc_radiation)] += inc_radiation[~N.isnan(inc_radiation)]
		AA[~N.isnan(inc_radiation)] += -VF[~N.isnan(inc_radiation)]
	else:  # Temperature
		bb[~N.isnan(T)] += eps*sigma*T[~N.isnan(T)]**4.
		AA[~N.isnan(T)] += -VF[~N.isnan(T)]*(1.-N.vstack(eps[~N.isnan(T)]))
	if q_net != None: # net heat removal from teh surface always applies
		bb[~N.isnan(q_net)] -= q_net[~N.isnan(q_net)]

	if (N.isnan(bb).any()):
		raise AttributeError('Wrong right hand side')
	if N.isnan(AA).any():
		print AA
		stop

	# Matrix inversion:
	J = N.linalg.solve(AA, bb)

	# Calculate element-wise flux density q and total flux Q.
	q = N.zeros(n)
	Q = N.zeros(n)
	E = N.zeros(n)

	for i in range(n):
		if ~N.isnan(T[i]):
			E[i] = sigma*T[i]**4.
			if eps[i] != 1.:
				q[i] = eps[i]/(1.-eps[i])*(E[i]-J[i]) #(W/m2)
			else:
				q[i] = E[i]-N.sum(VF[i,:]*J) # (W/m2)

		elif ~N.isnan(inc_radiation[i]):
			q[i] = bb[i]
			T[i] = (1./sigma*(J[i]+(1.-eps[i])/eps[i]*q[i]))**0.25

	E = sigma*T**4.
	Q = A*q #(W)

	return AA,bb,J,E,T,q,Q

