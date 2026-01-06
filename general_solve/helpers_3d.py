import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML

from cubic_basis.cell_grid.shape_functions_3d import *

def replace(rows,cols,data,i_out,i_in):
	mask = np.array(rows)==i_in
	
	new_cols = np.array(cols)[mask]
	new_data = np.array(data)[mask]
	new_rows = [i_out]*len(new_cols)
	return rows+new_rows, cols+list(new_cols), data+list(new_data)
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

def get_all_gauss(qpn,x0,x1,y0,y1,z0,z1,p):
	id_to_ind = {ID:[int(ID/4)%4,ID%4,int(ID/16)] for ID in range(64)}
	val_list = []
	for test_id in range(64):
		test_ind = id_to_ind[test_id]
		phi_test = lambda x,y,z: phi3_3d_ref(x,y,z,1,test_ind)
		vals = gauss_vals(phi_test,x0,x1,y0,y1,z0,z1,qpn,p)
		val_list.append(vals)
	return val_list

		
def compute_gauss(qpn):	
	[p,w] = np.polynomial.legendre.leggauss(qpn)
	phi_ops = get_all_gauss(qpn,0,1,0,1,0,1,p)
	return phi_ops,p,w

def gauss_vals(f,a,b,c,d,q,r,n,p):
	xmid, ymid, zmid = (a+b)/2, (c+d)/2, (q+r)/2
	xscale, yscale, zscale = (b-a)/2, (d-c)/2, (r-q)/2
	vals = np.zeros((n,n,n))
	for j in range(n):
		for i in range(n):
			for k in range(n):
				vals[i,j,k] = f(xscale*p[j]+xmid,yscale*p[i]+ymid,zscale*p[k]+zmid)
	return vals

def gauss_debug(f,g,a,b,c,d,q,r,n):
	xmid, ymid, zmid = (a+b)/2, (c+d)/2, (q+r)/2
	xscale, yscale, zscale = (b-a)/2, (d-c)/2, (r-q)/2
	[p,w] = np.polynomial.legendre.leggauss(n)
	outer = 0.
	fvals = np.zeros((n,n,n))
	gvals = np.zeros((n,n,n))
	for j in range(n):
		middle = 0.
		for i in range(n):
			inner = 0.
			for k in range(n):
				fval = f(xscale*p[j]+xmid,yscale*p[i]+ymid,zscale*p[k]+zmid)
				gval = g(xscale*p[j]+xmid,yscale*p[i]+ymid,zscale*p[k]+zmid)
				inner += w[k]*fval*gval#f(xscale*p[j]+xmid,yscale*p[i]+ymid,zscale*p[k]+zmid)
				fvals[i,j,k] = fval
				gvals[i,j,k] = gval
			middle += w[i]*inner
		outer += w[j]*middle
	return outer*xscale*yscale*zscale, fvals,gvals,p

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

def local_stiffness(qpn=5):

	K = np.zeros((64,64))
	id_to_ind = {ID:[int(ID/4)%4,ID%4,int(ID/16)] for ID in range(64)}

	for test_id in range(64):

		test_ind = id_to_ind[test_id]
		grad_phi_test = lambda x,y,z: grad_phi3_ref(x,y,z,1,test_ind)

		for trial_id in range(test_id,64):

			trial_ind = id_to_ind[trial_id]
			grad_phi_trial = lambda x,y,z: grad_phi3_ref(x,y,z,1,trial_ind)

			func = lambda x,y,z: grad_phi_trial(x,y,z) @ grad_phi_test(x,y,z)
			val = gauss(func,0,1,0,1,0,1,qpn)

			K[test_id,trial_id] += val
			K[trial_id,test_id] += val * (test_id != trial_id)
	return K

def local_mass(qpn=5):

	M = np.zeros((64,64))
	id_to_ind = {ID:[int(ID/4)%4,ID%4,int(ID/16)] for ID in range(64)}

	for test_id in range(64):

		test_ind = id_to_ind[test_id]
		phi_test = lambda x,y,z: phi3_3d_ref(x,y,z,1,test_ind)

		for trial_id in range(test_id,64):

			trial_ind = id_to_ind[trial_id]
			phi_trial = lambda x,y,z: phi3_3d_ref(x,y,z,1,trial_ind)

			func = lambda x,y,z: phi_trial(x,y,z) * phi_test(x,y,z)
			val = gauss(func,0,1,0,1,0,1,qpn)

			M[test_id,trial_id] += val
			M[trial_id,test_id] += val * (test_id != trial_id)
	return M

