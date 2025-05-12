import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML

from shape_functions_2d import *

#visualization helpers

def vis_constraints(C,dofs):
	fig = plt.figure(figsize=(10,15))
	constrained = np.where(np.diag(C)==0)[0]
	h = dofs[0].h

	#to_plot = [h*1.5,h,2*h]
	#titles = ['Coarse/Fine','Fine/Fine','Coarse/Coarse']
	#plt.plot([-1.5*h,1+1.5*h],[-.1,-.1],'grey',alpha=.2)
	#plt.plot([-2*h,1+2*h],[.1,.1],'grey',alpha=.2)
	for i,ind in enumerate(constrained):
		dof_inds = np.nonzero(C[ind])[0]
		f_y = dofs[ind].y
		for c_ind in dof_inds:
			c_y = dofs[c_ind].y
	
			if dofs[ind].h != dofs[c_ind].h:
				plt.plot([-.11,.11],[f_y,f_y],'grey',alpha=.2)
				plt.plot([-.1,.1],[f_y,c_y],'C'+str(i),alpha=.8)
				plt.scatter([-.1,.1],[f_y,c_y],c='k')

	plt.xlim(-.11,.11)
	plt.ylim(-1.5*h,1+1.5*h)
	#plt.title(titles[_])
	plt.title('interface constraints')
	plt.show()

	fig = plt.figure(figsize=(10,10))
	for i,ind in enumerate(constrained):
		dof_inds = np.nonzero(C[ind])[0]
		d0x,d0y = dofs[ind].x,dofs[ind].y
		for c_ind in dof_inds:
			d1x,d1y = dofs[c_ind].x,dofs[c_ind].y
	
			if dofs[ind].h == dofs[c_ind].h:
				if d0y < 0: color = 'C1'
				if d0y == 1: color = 'C2'
				if d0y > 1: color = 'C3'
				plt.scatter([d0x],[d0y],c=color)
				plt.scatter([d1x],[d1y],c=color,alpha=.2)

	plt.title('periodic constraints')
	plt.show()

#integrators

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

def local_stiffness(h,qpn=5):
	K = np.zeros((16,16))
	id_to_ind = {ID:[int(ID/4),ID%4] for ID in range(16)}

	for test_id in range(16):

		test_ind = id_to_ind[test_id]
		grad_phi_test = lambda x,y: grad_phi3_ref(x,y,h,test_ind)

		for trial_id in range(test_id,16):

			trial_ind = id_to_ind[trial_id]
			grad_phi_trial = lambda x,y: grad_phi3_ref(x,y,h,trial_ind)

			func = lambda x,y: grad_phi_trial(x,y) @ grad_phi_test(x,y)
			val = gauss(func,0,h,0,h,qpn)

			K[test_id,trial_id] += val
			K[trial_id,test_id] += val * (test_id != trial_id)
	return K

def local_mass(h,qpn=5):
	M = np.zeros((16,16))
	id_to_ind = {ID:[int(ID/4),ID%4] for ID in range(16)}

	for test_id in range(16):

		test_ind = id_to_ind[test_id]
		phi_test = lambda x,y: phi3_2d_ref(x,y,h,test_ind)

		for trial_id in range(test_id,16):

			trial_ind = id_to_ind[trial_id]
			phi_trial = lambda x,y: phi3_2d_ref(x,y,h,trial_ind)

			func = lambda x,y: phi_trial(x,y) * phi_test(x,y)
			val = gauss(func,0,h,0,h,qpn)

			M[test_id,trial_id] += val
			M[trial_id,test_id] += val * (test_id != trial_id)
	return M

