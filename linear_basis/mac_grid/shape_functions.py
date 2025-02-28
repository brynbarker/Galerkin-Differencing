
import numpy as np
import matplotlib.pyplot as plt

def phi1(x,h):
	if -h < x <= 0:
		return 1+1/h*x
	elif 0 < x <= h:
		return 1-1/h*x
	else:
		return 0
	
def phi1_dx(x,h):
	if -h < x <= 0:
		return 1/h
	elif 0 < x <= h:
		return -1/h
	else:
		return 0

def phi1_3d(x,y,z,h):
	return phi1(x,h)*phi1(y,h)*phi1(z,h)

def grad_phi1(x,y,z,h):
	phi_j = phi1(x,h)
	dphi_j_dx = phi1_dx(x,h)
	phi_i = phi1(y,h)
	dphi_i_dy = phi1_dx(y,h)
	phi_k = phi1(z,h)
	dphi_k_dz = phi1_dx(z,h)
	return np.array([phi_i*phi_k*dphi_j_dx,phi_j*phi_k*dphi_i_dy,phi_i*phi_j*dphi_k_dz])

def grad_phi1_eval(x_in,y_in,z_in,h,x0,y0,z0):
	x,y,z = x_in-x0,y_in-y0,z_in-z0
	return grad_phi1(x,y,z,h)

def phi1_3d_eval(x_in,y_in,z_in,h,x0,y0,z0):
	x,y,z = x_in-x0,y_in-y0,z_in-z0
	return phi1_3d(x,y,z,h)

def phi1_3d_ref(x_ref,y_ref,z_ref,h,ind):
	i,j,k = ind
	x,y,z = x_ref-h*j,y_ref-h*i,z_ref-h*k
	return phi1_3d(x,y,z,h)

def grad_phi1_ref(x_ref,y_ref,z_ref,h,ind):
	i,j,k = ind
	x,y,z = x_ref-h*j,y_ref-h*i,z_ref-h*k
	return grad_phi1(x,y,z,h)
