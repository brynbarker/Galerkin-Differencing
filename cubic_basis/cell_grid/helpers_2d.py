import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML

from cubic_basis.cell_grid.shape_functions_2d import *

def replace(rows,cols,data,i_out,i_in):
	# i_out is set to i_in but i_in is a ghost
	new_rows,new_cols,new_data = [],[],[]
	for (r,c,d) in zip(rows,cols,data):
		if r == i_in:
			new_rows.append(i_out)
			new_cols.append(c)
			new_data.append(d)
	return rows+new_rows, cols+new_cols, data+new_data

def indswap(l,i_old,i_new):
	for (i,l_ind) in enumerate(l):
		if l_ind == i_old:
			l[i] = i_new
	return l

def inddel(r,c,d,ind):
	n = len(r)
	to_pop = [j for j in range(n) if r[j]==ind]
	for j in to_pop[::-1]:
		r.pop(j)
		c.pop(j)
		d.pop(j)
	return r,c,d

#integrators
def get_all_gauss(qpn,x0,x1,y0,y1,p):
	id_to_ind = {ID:[int(ID/4),ID%4] for ID in range(16)}
	val_list = []
	for test_id in range(16):
		test_ind = id_to_ind[test_id]
		phi_test = lambda x,y: phi3_2d_ref(x,y,1,test_ind)
		vals = gauss_vals(phi_test,x0,x1,y0,y1,qpn,p)
		val_list.append(vals)
	return val_list
		
def compute_gauss(qpn):	
	[p,w] = np.polynomial.legendre.leggauss(qpn)
	reg = get_all_gauss(qpn,0,1,0,1,p)
	interface = [reg]
	doms = [[3],[2],[1],[1,3],[1,2],[0],[0,3],[0,2]]
	for dom_change in doms:
		dom = [0,1,0,1]
		for d in dom_change:
			dom[d] = .5
		x0,x1,y0,y1 = dom
		vals = get_all_gauss(qpn,x0,x1,y0,y1,p)
		interface.append(vals)
	return interface,p,w

def gauss_vals(f,a,b,c,d,n,p):
	xmid, ymid = (a+b)/2, (c+d)/2
	xscale, yscale = (b-a)/2, (d-c)/2
	vals = np.zeros((n,n))
	outer = 0.
	for j in range(n):
		for i in range(n):
			vals[i,j] = f(xscale*p[j]+xmid,yscale*p[i]+ymid)
	return vals

def super_quick_gauss(vals0,vals1,a,b,c,d,n,w):
	xscale, yscale = (b-a)/2, (d-c)/2
	outer = 0.
	for j in range(n):
		inner = 0.
		for i in range(n):
			inner += w[i]*vals0[i,j]*vals1[i,j]
		outer += w[j]*inner
	return outer*xscale*yscale
	
def super_quick_gauss_error(vals0,vals1,a,b,c,d,n,w):
	xscale, yscale, zscale = (b-a)/2, (d-c)/2
	outer = 0.
	for j in range(n):
		inner = 0.
		for i in range(n):
			inner += w[i]*(vals0[i,j]-vals1[i,j])**2
		outer += w[j]*inner
	return outer*xscale*yscale

def quick_gauss(f,vals,a,b,c,d,n,p,w):
	xmid, ymid = (a+b)/2, (c+d)/2
	xscale, yscale = (b-a)/2, (d-c)/2
	outer = 0.
	for j in range(n):
		inner = 0.
		for i in range(n):
			inner += w[i]*f(xscale*p[j]+xmid,yscale*p[i]+ymid)*vals[i,j]
		outer += w[j]*inner
	return outer*xscale*yscale

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

def local_stiffness(h,qpn=5,xside=None,yside=None):
	bounds = {None:[0,h],0:[0,h/2],1:[h/2,h]}
	x0,x1 = bounds[xside]
	y0,y1 = bounds[yside]

	K = np.zeros((16,16))
	id_to_ind = {ID:[int(ID/4),ID%4] for ID in range(16)}

	for test_id in range(16):

		test_ind = id_to_ind[test_id]
		grad_phi_test = lambda x,y: grad_phi3_ref(x,y,h,test_ind)

		for trial_id in range(test_id,16):

			trial_ind = id_to_ind[trial_id]
			grad_phi_trial = lambda x,y: grad_phi3_ref(x,y,h,trial_ind)

			func = lambda x,y: grad_phi_trial(x,y) @ grad_phi_test(x,y)
			val = gauss(func,x0,x1,y0,y1,qpn)

			K[test_id,trial_id] += val
			K[trial_id,test_id] += val * (test_id != trial_id)
	return K

def local_mass(h,qpn=5,xside=None,yside=None):
	bounds = {None:[0,h],0:[0,h/2],1:[h/2,h]}
	x0,x1 = bounds[xside]
	y0,y1 = bounds[yside]


	M = np.zeros((16,16))
	id_to_ind = {ID:[int(ID/4),ID%4] for ID in range(16)}

	for test_id in range(16):

		test_ind = id_to_ind[test_id]
		phi_test = lambda x,y: phi3_2d_ref(x,y,h,test_ind)

		for trial_id in range(test_id,16):

			trial_ind = id_to_ind[trial_id]
			phi_trial = lambda x,y: phi3_2d_ref(x,y,h,trial_ind)

			func = lambda x,y: phi_trial(x,y) * phi_test(x,y)
			val = gauss(func,x0,x1,y0,y1,qpn)

			M[test_id,trial_id] += val
			M[trial_id,test_id] += val * (test_id != trial_id)
	return M

