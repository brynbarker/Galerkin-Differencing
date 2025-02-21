import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML

from linear_basis.mac_grid.shape_functions import *

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
	if gridtype == 'horiz':
		h_ghosts = fine_ghosts
		v_ghosts = [[],[]]
	if gridtype == 'vert':
		h_ghosts = [[],[]]
		v_ghosts = fine_ghosts
	if gridtype == 'corner':
		h_ghosts = [fine_ghosts[0],fine_ghosts[1]]
		v_ghosts = [fine_ghosts[2],fine_ghosts[3]]

	#fig = plt.figure()
	fig = plt.figure(figsize=(20,20))
	h = dofs[0].h

	flags = {'C0':True,'C1':True,'C2':True,
			 'C3':True,'C4':True,'C5':True,'k':True}
	labels = {'C0':'1','C1':'1/4','C2':'1/2',
              'C3':'3/4','C4':'1/8','C5':'3/8','k':'other'}

	for i,scale in enumerate([1,-1]):
		for ind in h_ghosts[i]:
			if C[ind,ind] != 1.:
				dof_inds = np.nonzero(C[ind])[0]

				f_x,f_y = dofs[ind].x,dofs[ind].y
				for c_ind in dof_inds:
					c_x,c_y = dofs[c_ind].x, dofs[c_ind].y
					og = [c_x,c_y]
					if f_x - c_x > .5: c_x += 1
					if f_y - c_y > .5: c_y += 1
					if c_x - f_x > .5: c_x -= 1
					if c_y - f_y > .5: c_y -= 1
					cmark = '^' if (og[1]==c_y) else '*'
					if f_x==c_x:
						c_x -= scale*h/2
					if dofs[ind].h != h:
						c = c_map(C[ind,c_ind])
						if flags[c]:
							plt.plot([f_x,c_x],[f_y,c_y],c=c,label=labels[c],lw=1)
							flags[c] = False
						else:
							plt.plot([f_x,c_x],[f_y,c_y],c=c,lw=1)
						plt.plot([f_x],[f_y],c='k',ls='',marker='o')
						plt.plot([c_x],[c_y],c='k',ls='',marker=cmark)
		
	for i,scale in enumerate([-1,1]):
		for ind in v_ghosts[i]:
			if C[ind,ind] != 1.:
				dof_inds = np.nonzero(C[ind])[0]

				f_x,f_y = dofs[ind].x,dofs[ind].y
				for c_ind in dof_inds:
					c_x,c_y = dofs[c_ind].x, dofs[c_ind].y
					og = [c_x,c_y]
					if f_y - c_y > .5: c_y += 1
					if f_x - c_x > .5: c_x += 1
					cmark = '^' if (og[0]==c_x) else '*'
		
					if dofs[ind].h != h:
						c = c_map(C[ind,c_ind])
						if flags[c]:
							plt.plot([f_x,c_x],[f_y,c_y],c=c,label=labels[c],lw=1)
							flags[c] = False
						else:
							plt.plot([f_x,c_x],[f_y,c_y],c=c,lw=1)
						plt.plot([f_x],[f_y],c='k',ls='',marker='o')
						plt.plot([c_x],[c_y],c='k',ls='',marker=cmark)

	plt.legend(fontsize=20)
	return fig

def vis_periodic(C,dofs,gridtype):
	h = dofs[0].h/2
	fig = plt.figure()
	col = ['k','grey']
	c = [['C0','C2'],['C1','C3']]
	m = ['^','o']
	for ind,row in enumerate(C):
		if row[ind]==0 and max(row)==1:
			fine = dofs[ind].h==h
			g_x,g_y = dofs[ind].x,dofs[ind].y
			x_shft = g_y/abs(g_y)*h/3
			if gridtype=='vert': x_shft=0
			dof_inds = np.nonzero(row)[0]
			for c_ind in dof_inds:
				if C[ind,c_ind] != 1:
					print(ind,c_ind,C[ind,c_ind])
				if dofs[ind].h == dofs[c_ind].h:
					c_x,c_y = dofs[c_ind].x,dofs[c_ind].y
					plt.plot([g_x+x_shft,c_x+x_shft],[g_y,c_y],col[row[c_ind]==1])
					plt.scatter(c_x+x_shft,c_y,color=c[fine][1],marker=m[fine])
					plt.scatter(g_x+x_shft,g_y,color=c[fine][0],marker=m[fine])
	return fig
				

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

def get_phi_gauss(h,n):
	[p,w] = np.polynomial.legendre.leggauss(n)
	vals = {}
	ind = 0
	for y_shift in [0,-h]:
		for x_shift in [0,-h]:
			v = np.zeros((n,n))
			for j in range(n):
				for i in range(n):
					v[j,i] = phi1_2d(h/2*p[j]+h/2+x_shift,h/2*p[i]+h/2+y_shift)
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
					v[j,i] = grad_phi1(p[j]+x_shift,p[i]+y_shift)
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

def local_stiffness(h,qpn=5,y0=0,y1=1):
	K = np.zeros((4,4))
	id_to_ind = {ID:[int(ID/2),ID%2] for ID in range(4)}

	y0 *= h
	y1 *= h

	for test_id in range(4):

		test_ind = id_to_ind[test_id]
		grad_phi_test = lambda x,y: grad_phi1_ref(x,y,h,test_ind)#,I)

		for trial_id in range(test_id,4):

			trial_ind = id_to_ind[trial_id]
			grad_phi_trial = lambda x,y: grad_phi1_ref(x,y,h,trial_ind)#,I)

			func = lambda x,y: grad_phi_trial(x,y) @ grad_phi_test(x,y)
			val = gauss(func,0,h,y0,y1,qpn)

			K[test_id,trial_id] += val
			K[trial_id,test_id] += val * (test_id != trial_id)
	return K

def local_mass(h,qpn=5,y0=0,y1=1):
	M = np.zeros((4,4))
	id_to_ind = {ID:[int(ID/2),ID%2] for ID in range(4)}

	y0 *= h
	y1 *= h

	for test_id in range(4):

		test_ind = id_to_ind[test_id]
		phi_test = lambda x,y: phi1_2d_ref(x,y,h,test_ind)#,I)

		for trial_id in range(test_id,4):

			trial_ind = id_to_ind[trial_id]
			phi_trial = lambda x,y: phi1_2d_ref(x,y,h,trial_ind)#,I)

			func = lambda x,y: phi_trial(x,y) * phi_test(x,y)
			val = gauss(func,0,h,y0,y1,qpn)

			M[test_id,trial_id] += val
			M[trial_id,test_id] += val * (test_id != trial_id)
	return M

