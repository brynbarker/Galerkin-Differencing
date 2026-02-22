
import numpy as	np
import matplotlib.pyplot as	plt

def	phi1(x,h):
	if -h <	x <= 0:
		return 1+1/h*x
	elif 0 < x <= h:
		return 1-1/h*x
	else:
		return 0

def	phi2(x,h):
	if h < x <=	2*h:
		return (x-h)*(x-2*h)/2/h**2
	elif 0 < x <= h:
		return (x+h)*(x-h)/-1/h**2
	elif -h	< x	<= 0:
		return (x+2*h)*(x+h)/2/h**2
	else:
		return 0

def	phi2L(x,h):
	if 0 < x <=	h:
		return (x-h)*(x-2*h)/2/h**2
	elif -h	< x	<= 0:
		return (x+h)*(x-h)/-1/h**2
	elif -2*h <	x <= -h:
		return (x+2*h)*(x+h)/2/h**2
	else:
		return 0

	
def	phi1_dx(x,h):
	if -h <	x <= 0:
		return 1/h
	elif 0 < x <= h:
		return -1/h
	else:
		return 0

def	phi2_dx(x,h):
	if h < x <=	2*h:
		return (2*x-3*h)/2/h**2
	elif 0 < x <= h:
		return (2*x)/-1/h**2
	elif -h	< x	<= 0:
		return (2*x+3*h)/2/h**2
	else:
		return 0

def	phi2L_dx(x,h):
	if 0 < x <=	h:
		return (2*x-3*h)/2/h**2
	elif -h	< x	<= 0:
		return (2*x)/-1/h**2
	elif -2*h <	x <= -h:
		return (2*x+3*h)/2/h**2
	else:
		return 0

def	phi3(x,h):
	if -2*h	< x	<= -h:
		return (x+3*h)*(x+2*h)*(x+h)/6/h**3
	elif -h	< x	<= 0:
		return -(x+2*h)*(x+h)*(x-h)/2/h**3
	elif 0 < x <= h:
		return (x+h)*(x-h)*(x-2*h)/2/h**3
	elif h < x <= 2*h:
		return -(x-h)*(x-2*h)*(x-3*h)/6/h**3
	else:
		return 0

def	phi3_dx(x,h):
	if -2*h	< x	<= -h:
		return (11*h**2+12*h*x+3*x**2)/6/h**3
	elif -h	< x	<= 0:
		return (h**2-4*h*x-3*x**2)/2/h**3
	elif 0 < x <= h:
		return -(h**2+4*h*x-3*x**2)/2/h**3
	elif h < x <= 2*h:
		return -(11*h**2-12*h*x+3*x**2)/6/h**3
	else:
		return 0

func_map = {1:phi1,2:phi2,3:phi3}
dx_map = {1:phi1_dx,2:phi2_dx,3:phi3_dx}

def	phi_2d(ords,x,y,h):
	comp_0 = func_map[ords[0]](x,h)
	comp_1 = func_map[ords[1]](y,h)
	return comp_0 *	comp_1

def	dphi_2d(ords,x,y,h):
	comp_0 = func_map[ords[0]](x,h)
	comp_1 = func_map[ords[1]](y,h)
	comp_0_dx =	dx_map[ords[0]](x,h)
	comp_1_dx =	dx_map[ords[1]](y,h)
	return np.array([comp_1*comp_0_dx,comp_0*comp_1_dx])

def	phi_2d_eval(ords,x_in,y_in,h,x0,y0):
	x,y	= x_in-x0,y_in-y0
	return phi_2d(ords,x,y,h)

def	dphi_2d_eval(ords,x_in,y_in,h,x0,y0):
	x,y	= x_in-x0,y_in-y0
	return dphi_2d(ords,x,y,h)

def	phi_2d_ref(ords,x_ref,y_ref,h,ind):
	i,j	= ind
	xL,	yL = int(ords[0]/2), int(ords[1]/2)
	x,y	= x_ref+h*(xL-j),y_ref+h*(yL-i)
	return phi_2d(ords,x,y,h)

def	dphi_2d_ref(ords,x_ref,y_ref,h,ind):
	i,j	= ind
	xL,	yL = int(ords[0]/2), int(ords[1]/2)
	x,y	= x_ref+h*(xL-j),y_ref+h*(yL-i)
	return dphi_2d(ords,x,y,h)

def	phi_3d(ords,x,y,z,h):
	comp_0 = func_map[ords[0]](x,h)
	comp_1 = func_map[ords[1]](y,h)
	comp_2 = func_map[ords[2]](z,h)
	return comp_0 *	comp_1 * comp_2

def	dphi_3d(ords,x,y,z,h):
	comp_0 = func_map[ords[0]](x,h)
	comp_1 = func_map[ords[1]](y,h)
	comp_2 = func_map[ords[2]](z,h)
	comp_0_dx =	dx_map[ords[0]](x,h)
	comp_1_dx =	dx_map[ords[1]](y,h)
	comp_2_dx =	dx_map[ords[2]](z,h)
	return np.array([comp_0_dx*comp_1*comp_2,
				     comp_0*comp_1_dx*comp_2,
					 comp_0*comp_1*comp_2_dx])

def	phi_3d_eval(ords,x_in,y_in,z_in,h,x0,y0,z0):
	x,y,z	= x_in-x0,y_in-y0,z_in-z0
	return phi_3d(ords,x,y,z,h)

def	dphi_3d_eval(ords,x_in,y_in,z_in,h,x0,y0,z0):
	x,y,z	= x_in-x0,y_in-y0,z_in-z0
	return dphi_3d(ords,x,y,z,h)

def	phi_3d_ref(ords,x_ref,y_ref,z_ref,h,ind):
	i,j,k = ind
	xL,	yL, zL = int(ords[0]/2), int(ords[1]/2), int(ords[2]/2)
	x,y,z = x_ref+h*(xL-j),y_ref+h*(yL-i),z_ref+h*(zL-k)
	return phi_3d(ords,x,y,z,h)

def	dphi_3d_ref(ords,x_ref,y_ref,z_ref,h,ind):
	i,j,k = ind
	xL,	yL, zL = int(ords[0]/2), int(ords[1]/2), int(ords[2]/2)
	x,y,z = x_ref+h*(xL-j),y_ref+h*(yL-i),z_ref+h*(zL-k)
	return dphi_3d(ords,x,y,z,h)

def	_get_phi_refs(ords,dim):
	if dim == 2:
		my_phi = lambda	x_ref,y_ref,h,ind: phi_2d_ref(ords,x_ref,y_ref,h,ind)
		my_dphi	= lambda x_ref,y_ref,h,ind:	dphi_2d_ref(ords,x_ref,y_ref,h,ind)
		return my_phi, my_dphi
	if dim == 3:
		my_phi = lambda	x_ref,y_ref,z_ref,h,ind: phi_3d_ref(ords,x_ref,y_ref,z_ref,h,ind)
		my_dphi	= lambda x_ref,y_ref,z_ref,h,ind:	dphi_3d_ref(ords,x_ref,y_ref,z_ref,h,ind)
		return my_phi, my_dphi

#def phi3_2d(x,y,h):
	#return	phi3(x,h)*phi3(y,h)

#def dphi3_2d(x,y,h):
	#phi_j = phi3(x,h)
	#dphi_j_dx = phi3_dx(x,h)
	#phi_i = phi3(y,h)
	#dphi_i_dy = phi3_dx(y,h)
	#return	np.array([phi_i*dphi_j_dx,phi_j*dphi_i_dy])

#def dphi3_2d_eval(x_in,y_in,h,x0,y0):
	#x,y = x_in-x0,y_in-y0
	#return	dphi3_2d(x,y,h)

#def phi3_2d_eval(x_in,y_in,h,x0,y0):
	#x,y = x_in-x0,y_in-y0
	#return	phi3_2d(x,y,h)

#def phi3_2d_ref(x_ref,y_ref,h,ind):
	#i,j = ind
	#x,y = x_ref+h*(1-j),y_ref+h*(1-i)
	#return	phi3_2d(x,y,h)

#def dphi3_2d_ref(x_ref,y_ref,h,ind):
	#i,j = ind
	#x,y = x_ref+h*(1-j),y_ref+h*(1-i)
	#return	dphi3_2d(x,y,h)

#def phi3_3d(x,y,z,h):
	#return	phi3(x,h)*phi3(y,h)*phi3(z,h)

#def dphi3_3d(x,y,z,h):
	#phi_j = phi3(x,h)
	#dphi_j_dx = phi3_dx(x,h)
	#phi_i = phi3(y,h)
	#dphi_i_dy = phi3_dx(y,h)
	#phi_k = phi3(z,h)
	#dphi_k_dz = phi3_dx(z,h)
	#return	np.array([phi_i*phi_k*dphi_j_dx,
					 #phi_j*phi_k*dphi_i_dy,
					 #phi_j*phi_i*dphi_k_dz])

#def dphi3_3d_eval(x_in,y_in,z_in,h,x0,y0,z0):
	#x,y,z = x_in-x0,y_in-y0,z_in-z0
	#return	dphi3_3d(x,y,z,h)

#def phi3_3d_eval(x_in,y_in,z_in,h,x0,y0,z0):
	#x,y,z = x_in-x0,y_in-y0,z_in-z0
	#return	phi3_3d(x,y,z,h)

#def phi3_3d_ref(x_ref,y_ref,z_ref,h,ind):
	#i,j,k = ind
	#x,y,z = x_ref+h*(1-j),y_ref+h*(1-i),z_ref+h*(1-k)
	#return	phi3_3d(x,y,z,h)

#def dphi3_3d_ref(x_ref,y_ref,z_ref,h,ind):
	#i,j,k = ind
	#x,y,z = x_ref+h*(1-j),y_ref+h*(1-i),z_ref+h*(1-k)
	#return	dphi3_3d(x,y,z,h)
