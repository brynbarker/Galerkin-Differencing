import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML

from shape_functions_phi1 import *

#visualization helpers

def c_map(v):
	if v == 1:
		return 'C0'
	elif v == 1/4:
		return 'C1'
	elif v == 1/2:
		return 'C2'
	elif v == 3/4:
		return 'C3'
	elif v == 1/8:
		return 'C4'
	elif v == 3/8:
		return 'C5'
	else:
		print(v)
		return 'k'

def vis_constraints(C,dofs,fine_ghosts,gridtype=None):
	if gridtype == 'horiz'
		h_ghosts = fine_ghosts
		v_ghosts = []
	if gridtype == 'vert'
		h_ghosts = []
		v_ghosts = fine_ghosts
	if gridtype == 'corner'
		h_ghosts = fine_ghosts[0]
		v_ghosts = fine_ghosts[1]

	#fig = plt.figure(figsize=(15,10))
	h = dofs[0].h

	flags = {'C0':True,'C1':True,'C2':True,
			 'C3':True,'C4':True,'C5':True,'k':True}
	labels = {'C0':'1','C1':'1/4','C2':'1/2',
              'C3':'3/4','C4':'1/8','C5':'3/8','k':'other'}

	for j,ind in enumerate(h_ghosts):
		dof_inds = np.nonzero(C[ind])[0]

		f_x,f_y = dofs[ind].x,dofs[ind].y
		for c_ind in dof_inds:
			c_x,c_y = dofs[c_ind].x, dofs[c_ind].y
			if f_x==c_x:
				tmp_x = .4
			else:
				tmp_x = .4-abs(f_x-c_x)
	
			if dofs[ind].h != dofs[c_ind].h:
				c = c_map(C[ind,c_ind])
				if flags[c]:
					plt.plot([.5,tmp_x],[f_y,c_y],c=c,label=labels[c],lw=5)
					flags[c] = False
				else:
					plt.plot([.5,tmp_x],[f_y,c_y],c=c,lw=5)
				plt.scatter([.5,tmp_x],[f_y,c_y],c='k')
		
	for j,ind in enumerate(v_ghosts):
		dof_inds = np.nonzero(C[ind])[0]

		f_x = dofs[ind].x
		f_x,f_y = dofs[ind].x,dofs[ind].y
		for c_ind in dof_inds:
			c_x,c_y = dofs[c_ind].x, dofs[c_ind].y
			if f_y==c_y:
				tmp_y = .4
			else:
				tmp_y = .4-abs(f_y-c_y)
	
			if dofs[ind].h != dofs[c_ind].h:
				c = c_map(C[ind,c_ind])
				if flags[c]:
					plt.plot([f_x,c_x],[.5,tmp_y],c=c,label=labels[c],lw=5)
					flags[c] = False
				else:
					plt.plot([f_x,c_x],[.5,tmp_y],c=c,lw=5)
				plt.scatter([f_x,c_x],[.5,tmp_y],c='k')

	plt.legend(fontsize=30)
	plt.show()

#animators

def animate_2d(data,size,figsize=(10,10),yesdot=True):

	frame = [[.5,1,1,0,1,1,0,0,.5,.5,],[0,0,.5,.5,.5,1,1,0,0,1]]
	fig,ax = plt.subplots(figsize=figsize)
	if yesdot:
		ax.set_xlim(-.2,1.2)
		ax.set_ylim(-.2,1.2)
	else:
		ax.set_xlim(frame[0][0],frame[0][-1])
		ax.set_ylim(min(data[0][0][1])-.1,max(data[0][0][1])+.1)
	
	line, = ax.plot(frame[0],frame[1],'lightgrey')
	blocks = []
	if yesdot: dot, = ax.plot([],[],'ko',linestyle='None')
	for i in range(size):
		block, = ax.plot([],[])
		blocks.append(block)

	def update(n):
		if yesdot: blocks_n, dots_n = data[n]
		else: blocks_n = data[n]
		line.set_data(frame[0],frame[1])
		for i in range(size):
			if i < len(blocks_n):
				blocks[i].set_data(blocks_n[i][0],blocks_n[i][1])
			else:
				blocks[i].set_data([],[])
		if yesdot: dot.set_data(dots_n[0],dots_n[1])
		to_return = [line]+blocks
		if yesdot: to_return += [dot]
		return to_return
	interval = 400 if yesdot else 100
	ani = FuncAnimation(fig, update, frames=len(data), interval=interval)
	plt.close()
	return HTML(ani.to_html5_video())

	

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

def local_stiffness(h,qpn=5,half=-1,I=False):
	K = np.zeros((4,4))
	id_to_ind = {ID:[int(ID/2),ID%2] for ID in range(4)}

	y0 = h/2*(half>0)
	y1 = h-h/2*(half==0)

	if I:
		y0,y1 = 0, 3/4*h

	for test_id in range(4):

		test_ind = id_to_ind[test_id]
		grad_phi_test = lambda x,y: grad_phi1_ref(x,y,h,test_ind,I)

		for trial_id in range(test_id,4):

			trial_ind = id_to_ind[trial_id]
			grad_phi_trial = lambda x,y: grad_phi1_ref(x,y,h,trial_ind,I)

			func = lambda x,y: grad_phi_trial(x,y) @ grad_phi_test(x,y)
			val = gauss(func,0,h,y0,y1,qpn)

			K[test_id,trial_id] += val
			K[trial_id,test_id] += val * (test_id != trial_id)
	return K

def local_mass(h,qpn=5,half=-1,I=False):
	M = np.zeros((4,4))
	id_to_ind = {ID:[int(ID/2),ID%2] for ID in range(4)}

	y0 = h/2*(half>0)
	y1 = h-h/2*(half==0)
        
	if I:
		y0,y1 = 0, 3/4*h

	for test_id in range(4):

		test_ind = id_to_ind[test_id]
		phi_test = lambda x,y: phi1_2d_ref(x,y,h,test_ind,I)

		for trial_id in range(test_id,4):

			trial_ind = id_to_ind[trial_id]
			phi_trial = lambda x,y: phi1_2d_ref(x,y,h,trial_ind,I)

			func = lambda x,y: phi_trial(x,y) * phi_test(x,y)
			val = gauss(func,0,h,y0,y1,qpn)

			M[test_id,trial_id] += val
			M[trial_id,test_id] += val * (test_id != trial_id)
	return M

