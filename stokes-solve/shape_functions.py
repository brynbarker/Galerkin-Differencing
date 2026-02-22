
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

def	phi3(x,h):
	if h < x <=	2*h:
		return -(x-h)*(x-2*h)*(x-3*h)/6/h**3
	elif 0 < x <= h:
		return (x+h)*(x-h)*(x-2*h)/2/h**3
	elif -h	< x	<= 0:
		return -(x+2*h)*(x+h)*(x-h)/2/h**3
	elif -2*h <	x <= -h:
		return (x+3*h)*(x+2*h)*(x+h)/6/h**3
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

def	phi_2d(p,x,y,h):
	return func_map[p[0]](x,h)*func_map[p[1]](y,h)
	 
def	grad_phi(p,x,y,h):
	phi_j =	func_map[p[0]](x,h)
	dphi_j_dx =	dx_map[p[0]](x,h)
	phi_i =	func_map[p[1]](y,h)
	dphi_i_dy =	dx_map[p[1]](y,h)
	return np.array([phi_i*dphi_j_dx,phi_j*dphi_i_dy])

def	grad_phi_eval(p,x_in,y_in,h,x0,y0):
	x,y	= x_in-x0,y_in-y0
	return grad_phi(p,x,y,h)

def	phi_2d_eval(p,x_in,y_in,h,x0,y0):
	x,y	= x_in-x0,y_in-y0
	return phi_2d(p,x,y,h)

def	phi_2d_ref(p,x_ref,y_ref,h,ind):
	i,j	= ind
	xL,	yL = int(p[0]/2), int(p[1]/2)
	x,y	= x_ref+h*(xL-j),y_ref+h*(yL-i)
	return phi_2d(p,x,y,h)

def	grad_phi_ref(p,x_ref,y_ref,h,ind):
	i,j	= ind
	xL,	yL = int(p[0]/2), int(p[1]/2)
	x,y	= x_ref+h*(xL-j),y_ref+h*(yL-i)
	return grad_phi(p,x,y,h)
	   
def partial_div_phi_ref(p, x_ref, y_ref, h, dim, ind):
	i,j	= ind
	xL,	yL = int(p[0]/2), int(p[1]/2)
	xy	= x_ref+h*(xL-j),y_ref+h*(yL-i)

	part1 = dx_map[p[dim]](xy[dim],h) 
	part2 = func_map[p[1-dim]](xy[1-dim],h)

	return part1*part2
