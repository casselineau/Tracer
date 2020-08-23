import FONaR

#test = TwoNparamcav(0.5, [0.5,1.5, 0.7], [0.1,0.4,0.6], -0.4, 0.8, 0.6, 22., 13.4, 0.6, 4e-3)


#VF1 = Four_parameters_cavity_RTVF(0.5,0.4,0.2,-0.2,5,5, num_rays=5000, precision=0.001).VF_estimator

'''
def Two_N_parameters_RTVF(apertureRadius, frustaRadii, frustaDepths, coneDepth, el_FRUs, el_CON, num_rays=5000, precision=0.005, cav=None):
'''
#VFx = Two_N_parameters_cavity_RTVF(1., [2.5,1.5,1.], [0.5,1.5,1.], 0.4, [4,4,4], 1, num_rays=15000, precision=0.005).VF_esperance
#VFx = Cylinder_cavity_RTVF(1e-2, 1e-2, [1e-2,1e-2,1e-2], num_rays=100, precision=0.001).VF_estimator

#rec = FONaR(radii, heights, envelope_thickness, absorptivity_absorber, emissivity_absorber, absorptivity_envelope, emissivity_envelope, location=None, rotation=None)
vfs = rec.VF_sim(num_rays=5000, precision=0.05)

#plt.show()
#VF0 = Uniform_cylinder_cavity_RTVF(0.083/2., 0.083/2., 3,3 , num_rays=10000, precision=1e-3).VF_estimator
#VF1 = Cylinder_cavity_RTVF(0.083/2., 0.083/2., [1,1,1], num_rays=10000, precision=1e-3).VF_estimator


#VF1[1:,:] = VF1[1:,:][::-1,:]
#VF1[:,1:] = VF1[:,1:][:,::-1]
#print areas-areas1
#print VF0-VF1

#VF2[1:,:] = VF2[1:,:][::-1,:]
#VF2[:,1:] = VF2[:,1:][:,::-1]
#print areas2
#print VF2

