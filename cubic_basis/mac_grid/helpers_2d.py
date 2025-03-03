import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML

from cubic_basis.mac_grid.shape_functions_2d import *

#visualization helpers
v1, v3 = phi3(1/2,1), phi3(3/2,1)
v11 = v1*v1
v13 = v1*v3
v33 = v3*v3

v3_1 = v3/v1
v33_1 = v33/v1

v14, v34, v54, v74 = phi3(1/4,1), phi3(3/4,1), phi3(5/4,1), phi3(7/4,1)
def c_map(v):
	if v == v1:
		return 'C0'
	elif v == v3:
		return 'C1'
	elif v == -1:
		return 'C4'
	elif v == -v1/v3:#-v3_1:
		return 'C5'
	elif v == v11/v3:#v33_1:
		return 'C2'
	elif v == v1/v3:#v3_1:
		return 'C3'
	elif v == v74:
		return 'C6'
	elif v == v14:
		return 'C7'
	elif v == v34:
		return 'C8'
	elif v == v54:
		return 'C9'
	elif v == 1:
		return 'k'
	else:
		print(v)
		return 'red'

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

	flags = {'C0':True,'C1':True,'C2':True,'C7':True,'C8':True,'C9':True,
			 'C3':True,'C4':True,'C5':True,'C6':True,'k':True,'red':True}
	labels = {'C0':r'$\phi_3(h/2)$','C1':r'$\phi_3(3h/2)$','C4':r'$-1$',
              'C5':r'$-\phi_3(3h/2)/\phi_3(h/2)$','C2':r'$\phi_3(3h/2)\phi_3(3h/2)/\phi_3(h/2)$','C3':r'$\phi_3(3h/2)/\phi_3(h/2)$',
			  'C6':r'$\phi_3(7h/4)$','C7':r'$\phi_3(h/4)$','C8':r'$\phi_3(3h/4)$','C9':r'$\phi_3(5h/4)$','k':'1','red':'other'}
	done = set()
	for i,scale in enumerate([1,-1]):
		for ind in h_ghosts[i]:
			if C[ind,ind] != 1. and ind not in done:
				done.add(ind)
				dof_inds = np.nonzero(C[ind])[0]

				f_x,f_y = dofs[ind].x,dofs[ind].y
				for c_ind in dof_inds:
					c_x,c_y = dofs[c_ind].x, dofs[c_ind].y
					og = [c_x,c_y]
					if f_x - c_x > .5: c_x += 1
					if f_y - c_y > .5: c_y += 1
					if c_x - f_x > .5: c_x -= 1
					if c_y - f_y > .5: c_y -= 1
					cmark = '^' if (og[0]==c_x and og[1]==c_y) else '*'
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
			if C[ind,ind] != 1. and ind not in done:
				done.add(ind)
				dof_inds = np.nonzero(C[ind])[0]

				f_x,f_y = dofs[ind].x,dofs[ind].y
				for c_ind in dof_inds:
					c_x,c_y = dofs[c_ind].x, dofs[c_ind].y
					og = [c_x,c_y]
					if f_y - c_y > .5: c_y += 1
					if f_x - c_x > .5: c_x += 1
					if c_y - f_y > .5: c_y -= 1
					if c_x - f_x > .5: c_x -= 1
					cmark = '^' if (og[0]==c_x and og[1]==c_y) else '*'
		
					if dofs[ind].h != h:
						c = c_map(C[ind,c_ind])
						if flags[c]:
							plt.plot([f_x,c_x],[f_y,c_y],c=c,label=labels[c],lw=1)
							flags[c] = False
						else:
							plt.plot([f_x,c_x],[f_y,c_y],c=c,lw=1)
						plt.plot([f_x],[f_y],c='k',ls='',marker='o')
						plt.plot([c_x],[c_y],c='k',ls='',marker=cmark)

	#for i,scale in enumerate([1,-1]):
	#	for ind in h_ghosts[i]:
	#		dof_inds = np.nonzero(C[ind])[0]

	#		f_x,f_y = dofs[ind].x,dofs[ind].y
	#		for c_ind in dof_inds:
	#			c_x,c_y = dofs[c_ind].x, dofs[c_ind].y
	#			if f_x==c_x or f_x-c_x==1:
	#				tmp_x = 1-i/2+scale*.2
	#			else:
	#				#print('ok x=.5',f_x,c_x)
	#				tmp_x = 1-i/2+scale*.2*(1+abs(f_x-c_x))
	#				pass
	#			if dofs[ind].h != dofs[c_ind].h:
	#				c = c_map(C[ind,c_ind])
	#				#if abs(f_y-c_y)>5*h: c_y = 1-c_y
	#				if flags[c]:
	#					plt.plot([1-i/2,tmp_x],[f_y,c_y],c=c,label=labels[c],lw=1)
	#					flags[c] = False
	#				else:
	#					plt.plot([1-i/2,tmp_x],[f_y,c_y],c=c,lw=1)
	#				plt.plot([1-i/2],[f_y],c='k',ls='',marker='o')
	#				plt.plot([tmp_x],[c_y],c='k',ls='',marker='^')
	#	
	#for i,scale in enumerate([-1,1]):
	#	for ind in v_ghosts[i]:
	#		dof_inds = np.nonzero(C[ind])[0]

	#		f_x,f_y = dofs[ind].x,dofs[ind].y
	#		for c_ind in dof_inds:
	#			c_x,c_y = dofs[c_ind].x, dofs[c_ind].y
	#			if f_y==c_y or f_y-c_y==1:
	#				tmp_y = .5+i/2+scale*.2
	#				xsft = 0
	#			else:
	#				print('ok y=.5',f_y,c_y)
	#				tmp_y = c_y#1-i/2+scale*.2*(1+abs(f_y-c_y))
	#				xsft = 0#-.1
	#	
	#			if dofs[ind].h != dofs[c_ind].h:
	#				c = c_map(C[ind,c_ind])
	#				#if abs(f_x-c_x)>5*h: c_x = 1-c_x
	#				if flags[c]:
	#					plt.plot([f_x,c_x+xsft],[.5+i/2,tmp_y],c=c,label=labels[c],lw=1)
	#					flags[c] = False
	#				else:
	#					plt.plot([f_x,c_x+xsft],[.5+i/2,tmp_y],c=c,lw=1)
	#				plt.plot([f_x],[.5+i/2],c='k',ls='',marker='o')
	#				plt.plot([c_x+xsft],[tmp_y],c='k',ls='',marker='^')

	plt.legend(fontsize=20)
	return fig
	#plt.show()

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
				#if C[ind,c_ind] != 1:
				#	skip=True#print(ind,c_ind,C[ind,c_ind])
				if dofs[ind].h == dofs[c_ind].h and C[ind,c_ind]==1:
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
	K = np.zeros((16,16))
	id_to_ind = {ID:[int(ID/4),ID%4] for ID in range(16)}

	y0 *= h
	y1 *= h 

	for test_id in range(16):

		test_ind = id_to_ind[test_id]
		grad_phi_test = lambda x,y: grad_phi3_ref(x,y,h,test_ind)

		for trial_id in range(test_id,16):

			trial_ind = id_to_ind[trial_id]
			grad_phi_trial = lambda x,y: grad_phi3_ref(x,y,h,trial_ind)

			func = lambda x,y: grad_phi_trial(x,y) @ grad_phi_test(x,y)
			val = gauss(func,0,h,y0,y1,qpn)

			K[test_id,trial_id] += val
			K[trial_id,test_id] += val * (test_id != trial_id)
	return K

def local_mass(h,qpn=5,y0=0,y1=1):
	M = np.zeros((16,16))
	id_to_ind = {ID:[int(ID/4),ID%4] for ID in range(16)}

	y0 *= h
	y1 *= h

	for test_id in range(16):

		test_ind = id_to_ind[test_id]
		phi_test = lambda x,y: phi3_2d_ref(x,y,h,test_ind)

		for trial_id in range(test_id,16):

			trial_ind = id_to_ind[trial_id]
			phi_trial = lambda x,y: phi3_2d_ref(x,y,h,trial_ind)

			func = lambda x,y: phi_trial(x,y) * phi_test(x,y)
			val = gauss(func,0,h,y0,y1,qpn)

			M[test_id,trial_id] += val
			M[trial_id,test_id] += val * (test_id != trial_id)
	return M

