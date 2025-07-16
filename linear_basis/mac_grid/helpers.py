import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy import sparse
from IPython.display import HTML

from linear_basis.mac_grid.shape_functions import *

#visualization helpers

def add_constraint(r,c,d,newr,newc,newdval):
	if not isinstance(newr,list):
		newr = list(newr.flatten())	
	if not isinstance(newc,list):
		newc = list(newc.flatten())	
	assert len(newr)==len(newc)
	return r+newr, c+newc, d+[newdval]*len(newr)

def indswap(l,i_old,i_new):
	for (i,l_ind) in enumerate(l):
		if l_ind == i_old:
			l[i] = i_new
	return l

def full_swap_and_set(r,c,d,mm,td):
	carr = np.array(c)
	ids = np.arange(len(mm))
	mismask = (mm!=ids)*(mm!=-1)
	for j,jmap in zip(ids[mismask],mm[mismask]):
		carr[carr==j] = jmap

	C = list(carr)
	nonneg = mm!=-1
	C += list(mm[nonneg])
	r += list(ids[nonneg])
	d += [1]*sum(nonneg)

	c_small = np.array(C)
	for i,td in enumerate(td):
		c_small[c_small==td] = i

	return r,list(C),d,list(c_small)

def inddel(r,c,d,ind):
	n = len(r)
	to_pop = [j for j in range(n) if r[j]==ind]
	for j in to_pop[::-1]:
		r.pop(j)
		c.pop(j)
		d.pop(j)
	return r,c,d

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

def get_all_gauss(qpn,x0,x1,y0,y1,z0,z1,p):
	id_to_ind = {ID:[int(ID/2)%2,ID%2,int(ID/4)] for ID in range(8)}
	val_list = []
	for test_id in range(8):
		test_ind = id_to_ind[test_id]
		phi_test = lambda x,y,z: phi1_3d_ref(x,y,z,1,test_ind)
		vals = gauss_vals(phi_test,x0,x1,y0,y1,z0,z1,qpn,p)
		val_list.append(vals)
	return val_list
		
def compute_gauss(qpn):	
	[p,w] = np.polynomial.legendre.leggauss(qpn)
	reg = get_all_gauss(qpn,0,1,0,1,0,1,p)
	interface = []
	doms = [[1],[0],[3],[2],[1,3],[1,2],[0,3],[0,2]]
	for dom_change in doms:
		dom = [0,1,0,1]
		for d in dom_change:
			dom[d] = .5
		y0,y1,z0,z1 = dom
		vals = get_all_gauss(qpn,0,1,y0,y1,z0,z1,p)
		interface.append(vals)
	return reg,interface,p,w

def gauss_vals(f,a,b,c,d,q,r,n,p):
	xmid, ymid, zmid = (a+b)/2, (c+d)/2, (q+r)/2
	xscale, yscale, zscale = (b-a)/2, (d-c)/2, (r-q)/2
	vals = np.zeros((n,n,n))
	outer = 0.
	for j in range(n):
		for i in range(n):
			for k in range(n):
				vals[i,j,k] = f(xscale*p[j]+xmid,yscale*p[i]+ymid,zscale*p[k]+zmid)
	return vals

def super_quick_gauss(vals0,vals1,a,b,c,d,q,r,n,w):
	xscale, yscale, zscale = (b-a)/2, (d-c)/2, (r-q)/2
	outer = 0.
	for j in range(n):
		middle = 0.
		for i in range(n):
			inner = 0.
			for k in range(n):
				inner += w[k]*vals0[i,j,k]*vals1[i,j,k]
			middle += w[i]*inner
		outer += w[j]*middle
	return outer*xscale*yscale*zscale
	
def super_quick_gauss_error(vals0,vals1,a,b,c,d,q,r,n,w):
	xscale, yscale, zscale = (b-a)/2, (d-c)/2, (r-q)/2
	outer = 0.
	for j in range(n):
		middle = 0.
		for i in range(n):
			inner = 0.
			for k in range(n):
				inner += w[k]*(vals0[i,j,k]-vals1[i,j,k])**2
			middle += w[i]*inner
		outer += w[j]*middle
	return outer*xscale*yscale*zscale

def quick_gauss(f,vals,a,b,c,d,q,r,n,p,w):
	xmid, ymid, zmid = (a+b)/2, (c+d)/2, (q+r)/2
	xscale, yscale, zscale = (b-a)/2, (d-c)/2, (r-q)/2
	outer = 0.
	for j in range(n):
		middle = 0.
		for i in range(n):
			inner = 0.
			for k in range(n):
				inner += w[k]*f(xscale*p[j]+xmid,yscale*p[i]+ymid,zscale*p[k]+zmid)*vals[i,j,k]
			middle += w[i]*inner
		outer += w[j]*middle
	return outer*xscale*yscale*zscale

def gauss(f,a,b,c,d,q,r,n):
	xmid, ymid, zmid = (a+b)/2, (c+d)/2, (q+r)/2
	xscale, yscale, zscale = (b-a)/2, (d-c)/2, (r-q)/2
	[p,w] = np.polynomial.legendre.leggauss(n)
	outer = 0.
	for j in range(n):
		middle = 0.
		for i in range(n):
			inner = 0.
			for k in range(n):
				inner += w[k]*f(xscale*p[j]+xmid,yscale*p[i]+ymid,zscale*p[k]+zmid)
			middle += w[i]*inner
		outer += w[j]*middle
	return outer*xscale*yscale*zscale

def local_stiffness(h,qpn=5,y0=0,y1=1,z0=0,z1=1):
	K = np.zeros((8,8))
	id_to_ind = {ID:[int(ID/2)%2,ID%2,int(ID/4)] for ID in range(8)}

	y0 *= h
	y1 *= h
	z0 *= h
	z1 *= h

	for test_id in range(8):

		test_ind = id_to_ind[test_id]
		grad_phi_test = lambda x,y,z: grad_phi1_ref(x,y,z,h,test_ind)

		for trial_id in range(test_id,8):

			trial_ind = id_to_ind[trial_id]
			grad_phi_trial = lambda x,y,z: grad_phi1_ref(x,y,z,h,trial_ind)

			func = lambda x,y,z: grad_phi_trial(x,y,z) @ grad_phi_test(x,y,z)
			val = gauss(func,0,h,y0,y1,z0,z1,qpn)

			K[test_id,trial_id] += val
			K[trial_id,test_id] += val * (test_id != trial_id)
	return K

def local_mass(h,qpn=5,y0=0,y1=1,z0=0,z1=1):
	M = np.zeros((8,8))
	id_to_ind = {ID:[int(ID/2)%2,ID%2,int(ID/4)] for ID in range(8)}

	y0 *= h
	y1 *= h
	z0 *= h
	z1 *= h


	for test_id in range(8):

		test_ind = id_to_ind[test_id]
		phi_test = lambda x,y,z: phi1_3d_ref(x,y,z,h,test_ind)

		for trial_id in range(test_id,8):

			trial_ind = id_to_ind[trial_id]
			phi_trial = lambda x,y,z: phi1_3d_ref(x,y,z,h,trial_ind)

			func = lambda x,y,z: phi_trial(x,y,z) * phi_test(x,y,z)
			val = gauss(func,0,h,y0,y1,z0,z1,qpn)

			M[test_id,trial_id] += val
			M[trial_id,test_id] += val * (test_id != trial_id)
	return M

