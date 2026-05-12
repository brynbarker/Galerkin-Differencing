
import numpy as	np

#GLOBAL
from general_solve import globals

### 1d local functions

def	phi0(x,h):
	if -h/2	<= x <=	h/2:
		return 1
	return 0

def	phi1(x,h):
	if -h <	x <= 0:
		return 1+1/h*x
	elif 0 < x <= h:
		return 1-1/h*x
	else:
		return 0

def	phi2(x,h):
	# return bspline3(x,h)
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
	# return cubic(x,h)
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

def bspline2(x,h):
	if -h <= x < 0:
		return (h+x)/h
	elif 0 <= x	< h:
		return (h-x)/h
	else: return 0

def	bspline3(x,h):
	if -h <= x < 0:
		return (x**2/2+x*h+h**2/2)/h**2
	elif 0 <= x	< h:
		return (-x**2+x*h+h**2/2)/h**2
	elif h <= x	< 2*h:
		return (x**2/2-2*x*h+2*h**2)/h**2
	else: return 0

def	bspline4(x,h):
	"""The bspline-4 kernel.
	"""
	if -2*h <= x < -h:
		return (x**3+6*x**2*h+12*x*h**2+8*h**3)/6/h**3
	elif -h <= x < 0:
		return (-3*x**3-6*x**2*h+4*h**3)/6/h**3
	elif 0 <= x < h:
		return (3*x**3-6*x**2*h+4*h**3)/6/h**3
	elif h <= x < 2*h:
		return (-x**3+6*x**2*h-12*x*h**2+8*h**3)/6/h**3
	else:
		return 0


### 1d local function derivatives

def	phi0_dx(x,h):
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

def bspline2_dx(x,h):
	return phi1_dx(x,h)

def	bspline3_dx(x,h):
	"""The bspline-3 kernel.
	"""
	if -h <= x < 0:
		return (x+h)/h**2
	elif 0 <= x	< h:
		return (-2*x+h)/h**2
	elif h <= x	< 2*h:
		return (x-2*h)/h**2
	else: return 0
	
def bspline4_dx(x,h):
	if -2*h <= x < -h:
		return (3*x**2+12*x*h+12*h**2)/6/h**3
	elif -h <= x < 0:
		return (-9*x**2-12*x*h)/6/h**3
	elif 0 <= x < h:
		return (9*x**2-12*x*h)/6/h**3
	elif h <= x < 2*h:
		return (-3*x**2+12*x*h-12*h**2)/6/h**3
	else:
		return 0


### map orders to functions

lag_func_map = {0:phi0,1:phi1,2:phi2,3:phi3}
lag_dx_map = {0:phi0_dx,1:phi1_dx,2:phi2_dx,3:phi3_dx}

spline_func_map = {0:phi0,1:bspline2,2:bspline3,3:bspline4}
spline_dx_map = {0:phi0_dx,1:bspline2_dx,2:bspline3_dx,3:bspline4_dx}

func_map = {True:lag_func_map,False:spline_func_map}
dx_map = {True:lag_dx_map,False:spline_dx_map}

### 2d local functions

def	phi_2d(ords,x,y,h):
	comp_0 = func_map[globals.LAG][ords[0]](x,h)
	comp_1 = func_map[globals.LAG][ords[1]](y,h)
	return comp_0 *	comp_1


### 2d local function derivatives

def	dphi_2d(ords,x,y,h):
	comp_0 = func_map[globals.LAG][ords[0]](x,h)
	comp_1 = func_map[globals.LAG][ords[1]](y,h)
	comp_0_dx =	dx_map[globals.LAG][ords[0]](x,h)
	comp_1_dx =	dx_map[globals.LAG][ords[1]](y,h)
	return np.array([comp_1*comp_0_dx,comp_0*comp_1_dx])


### 2d global functions

def	phi_2d_eval(ords,x_in,y_in,h,x0,y0):
	x,y	= x_in-x0,y_in-y0
	return phi_2d(ords,x,y,h)

def	dphi_2d_eval(ords,x_in,y_in,h,x0,y0):
	x,y	= x_in-x0,y_in-y0
	return dphi_2d(ords,x,y,h)

### 2d reference functions

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

### return relevant functions

def	_get_phi_refs(ords,dim):
	if dim == 2:
		my_phi = lambda	x_ref,y_ref,h,ind: phi_2d_ref(ords,x_ref,y_ref,h,ind)
		my_dphi	= lambda x_ref,y_ref,h,ind:	dphi_2d_ref(ords,x_ref,y_ref,h,ind)
		return my_phi, my_dphi
	if dim == 3:
		my_phi = lambda	x_ref,y_ref,z_ref,h,ind: phi_3d_ref(ords,x_ref,y_ref,z_ref,h,ind)
		my_dphi	= lambda x_ref,y_ref,z_ref,h,ind: dphi_3d_ref(ords,x_ref,y_ref,z_ref,h,ind)
		return my_phi, my_dphi

def	get_phi_2d_ref_xys(ords,h,ind):
	i,j	= ind
	xL,	yL = int(ords[0]/2), int(ords[1]/2)
	xshft,yshft	= h*(xL-j),h*(yL-i)
	return xshft,yshft

### 3d everything

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
	i,j,k =	ind
	xL,	yL,	zL = int(ords[0]/2), int(ords[1]/2), int(ords[2]/2)
	x,y,z =	x_ref+h*(xL-j),y_ref+h*(yL-i),z_ref+h*(zL-k)
	return phi_3d(ords,x,y,z,h)

def	dphi_3d_ref(ords,x_ref,y_ref,z_ref,h,ind):
	i,j,k =	ind
	xL,	yL,	zL = int(ords[0]/2), int(ords[1]/2), int(ords[2]/2)
	x,y,z =	x_ref+h*(xL-j),y_ref+h*(yL-i),z_ref+h*(zL-k)
	return dphi_3d(ords,x,y,z,h)
