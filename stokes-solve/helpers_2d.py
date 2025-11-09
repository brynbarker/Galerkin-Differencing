import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML

from linear_basis.stokes.shape_functions import *

#integrators

def get_phi_gauss(h,n):
	[p,w] = np.polynomial.legendre.leggauss(n)
	vals = {}
	ind = 0
	for y_shift in [0,-h]:
		for x_shift in [0,-h]:
			v = np.zeros((n,n))
			for j in range(n):
				for i in range(n):
					v[j,i] = phi_2d(h/2*p[j]+h/2+x_shift,h/2*p[i]+h/2+y_shift)
			vals[ind] = v
			ind += 1
	return vals,p,w
		

def get_dphi_gauss(h,n):
	[p,w] = np.polynomial.legendre.leggauss(n)
	p = h/2*(p+1)
	vals = {}
	ind = 0
	for y_shift in [0,-h]:
		for x_shift in [0,-h]:
			v = np.zeros((n,n))
			for j in range(n):
				for i in range(n):
					v[j,i] = grad_phi(p[j]+x_shift,p[i]+y_shift)
			vals[ind] = v
			ind += 1

	return vals,p,w

def fast_gauss(f,p,w,v,n):
	outer = 0.
	for j in range(n):
		inner = 0.
		for i in range(n):
			inner += w[i]*f(p[j],p[i])*v[j,i]
		outer += w[j]*inner
	return outer*h*h/4

def gauss(f,a,b,c,d,n):
	xmid, ymid = (a+b)/2, (c+d)/2
	xscale, yscale = (b-a)/2, (d-c)/2
	[p,w] = np.polynomial.legendre.leggauss(n)
	outer = 0.
	for j in range(n):
		inner = 0.
		for i in range(n):
			inner += w[i]*f(xscale*p[j]+xmid,yscale*p[i]+ymid)
		outer += w[j]*inner
	return outer*xscale*yscale

def local_stiffness(h,p,qpn=5,y0=0,y1=1):
	y0 *= h
	y1 *= h
	xl,yl = p[0]+1,p[1]+1
	id_to_ind = {ID:[int(ID/xl),ID%xl] for ID in range(xl*yl)}
	K = np.zeros((xl*yl,xl*yl))

	for test_id in range(xl*yl):

		test_ind = id_to_ind[test_id]
		grad_phi_test = lambda x,y: grad_phi_ref(p,x,y,h,test_ind)

		for trial_id in range(test_id,xl*yl):

			trial_ind = id_to_ind[trial_id]
			grad_phi_trial = lambda x,y: grad_phi_ref(p,x,y,h,trial_ind)

			func = lambda x,y: grad_phi_trial(x,y) @ grad_phi_test(x,y)
			val = gauss(func,0,h,y0,y1,qpn)

			K[test_id,trial_id] += val
			K[trial_id,test_id] += val * (test_id != trial_id)
	return K

def local_mass(h,p,qpn=5,y0=0,y1=1):
	xl,yl = p[0]+1,p[1]+1
	id_to_ind = {ID:[int(ID/xl),ID%xl] for ID in range(xl*yl)}
	M = np.zeros((xl*yl,xl*yl))

	y0 *= h
	y1 *= h

	for test_id in range(xl*yl):

		test_ind = id_to_ind[test_id]
		phi_test = lambda x,y: phi_2d_ref(p,x,y,h,test_ind)

		for trial_id in range(test_id,xl*yl):

			trial_ind = id_to_ind[trial_id]
			phi_trial = lambda x,y: phi_2d_ref(p,x,y,h,trial_ind)

			func = lambda x,y: phi_trial(x,y) * phi_test(x,y)
			val = gauss(func,0,h,y0,y1,qpn)

			M[test_id,trial_id] += val
			M[trial_id,test_id] += val * (test_id != trial_id)
	return M

def local_zero_mean(h,p,qpn=5,y0=0,y1=1):
	xl,yl = p[0]+1,p[1]+1
	id_to_ind = {ID:[int(ID/xl),ID%xl] for ID in range(xl*yl)}
	G = np.zeros((xl*yl))

	y0 *= h
	y1 *= h

	for trial_id in range(xl*yl):

		trial_ind = id_to_ind[trial_id]
		phi_trial = lambda x,y: phi_2d_ref(p,x,y,h,trial_ind)

		val = gauss(phi_trial,0,h,y0,y1,qpn)

		G[trial_id] += val
	return G

