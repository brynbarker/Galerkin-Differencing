
import numpy as	np
from decimal import Decimal, getcontext
getcontext().prec=15

#GLOBAL
from general_solve import globals

### explicit transform functions
def phi_trap(x,y,h,x0,y0,xi0,eta0,map_type):
	A,B,C = [1,2,-1] if map_type < 2 else [-1,1,1]


	xi = (x-x0)/h
	eta = (y-y0-A/2*(x-x0))/(B*h+C*(x-x0))

	phi0 = bspline3(xi-xi0,1)
	phi1 = bspline2(eta-eta0,1)
	return phi0*phi1

def AB0_to_eta(A,B,A0,B0):
	if A0+B0 == 0:
		return 0,0
	mod = B0-A0
	prod = A*A0+B*B0
	if A0+B0 < 0:
		return mod*prod,mod
	return mod*(prod-1),mod

def AB0_to_deta(A0,B0,coefs):
	if A0+B0==0:
		return 0,0
	elif B0 == 0:
		return coefs[0],coefs[1]
	else:
		return coefs[2],coefs[3]

def bspline3_triangle_point(xi):
	return bspline3(xi,1)+(xi-1)*bspline3(xi+1,1)+(1-xi)*bspline3(xi-1,1)

def dbspline3_triangle_point(xi):
	part0 = bspline3_dx(xi,1)
	part1a = bspline3(xi+1,1)
	part1b = (xi-1)*bspline3_dx(xi+1,1)
	part2a = -bspline3(xi-1,1)
	part2b = (1-xi)*bspline3_dx(xi-1,1)
	return part0+part1a+part1b+part2a+part2b


def dphi_tri(x,y,h,bary_coefs,x0,y0,map_type,x1=None,y1=None,check=False):

	xp,yp,coef0,coef1,coef2,coef3,tmp = bary_coefs
	dA_dx, dA_dy, dB_dx, dB_dy = coef0, coef1, coef2, coef3

	A0 = coef0*(x0-xp) + coef1*(y0-yp)
	B0 = coef2*(x0-xp) + coef3*(y0-yp)
	other0 = (A0+B0) if map_type < 2 else 1-(A0+B0)
	A = coef0*(x-xp) + coef1*(y-yp)
	B = coef2*(x-xp) + coef3*(y-yp)

	# print(A0,B0,A,B)
	other = A+B
	dother_dx = dA_dx+dB_dx
	dother_dy = dA_dy+dB_dy
	if map_type > 1:
		other = 1-other
		dother_dx *= -1
		dother_dy *= -1

	if map_type %2 == 0:
		xi = other
		xi0 = other0
		eta,deta = AB0_to_eta(A,B,A0,B0)
		
		phi0 = bspline3(xi-xi0,1)
		if A0+B0 != 0:
			dphi0_dxi = bspline3_dx(xi-xi0,1)
		else:
			dphi0_dxi = dbspline3_triangle_point(xi)
		phi1 = bspline2(eta,1)
		dphi1_deta = bspline2_dx(eta,1)*deta

		dxi_dx,dxi_dy = dother_dx,dother_dy
		deta_dx,deta_dy = AB0_to_deta(A0,B0,bary_coefs[2:-1])
	else:
		eta = other
		eta0 = other0
		xi,dxi = AB0_to_eta(A,B,A0,B0)
		
		phi0 = bspline2(xi,1)
		dphi0_dxi = bspline2_dx(xi,1)*dxi
		phi1 = bspline3(eta-eta0,1)
		dphi1_deta = bspline3_dx(eta-eta0,1)

		deta_dx,deta_dy = dother_dx,dother_dy
		dxi_dx,dxi_dy = AB0_to_deta(A0,B0,bary_coefs[2:-1])

	dphi0_dx = dphi0_dxi*dxi_dx
	dphi0_dy = dphi0_dxi*dxi_dy

	dphi1_dx = dphi1_deta*deta_dx
	dphi1_dy = dphi1_deta*deta_dy

	dphi_dx = dphi0_dx*phi1 + dphi1_dx*phi0
	dphi_dy = dphi0_dy*phi1 + dphi1_dy*phi0
	dphi = np.array([dphi_dx,dphi_dy],dtype=np.float128)

	# if check:
	# 	return eta,dphi1_deta#dphi0_dxi,dphi1_deta#phi0,phi1#,,dxi_dx,dxi_dy,deta_dx,deta_dy

	if x1 is None:
		return dphi
		# return np.array([dphi_dx,dphi_dy])

	_dphi = dphi_tri(x,y,h,bary_coefs,x1,y1,map_type)

	return dphi @ _dphi


def dphi_trap(x,y,h,x0,y0,xi0,eta0,map_type,xi1=None,eta1=None):
	A,B,C = [1,2,-1] if map_type < 2 else [-1,1,1]

	if map_type %2 == 0:
		xi = (x-x0)/h
		eta = (y-y0-A/2*(x-x0))/(B*h+C*(x-x0))

		dxi_dx = 1/h
		deta_dx = (-A/2*(B*h+C*(x-x0))-C*x*(y-y0-A/2*(x-x0)))/(B*h+C*(x-x0))**2
		dxi_dy = 0
		deta_dy = 1/(B*h+C*(x-x0))

		phi0 = bspline3(xi-xi0,1)
		phi1 = bspline2(eta-eta0,1)

		dphi0_dxi = bspline3_dx(xi-xi0,1)
		dphi1_deta = bspline2_dx(eta-eta0,1)
	else:
		eta = (y-y0)/h
		xi = (x-x0-A/2*(y-y0))/(B*h+C*(y-y0))

		deta_dy = 1/h
		dxi_dy = (-A/2*(B*h+C*(y-y0))-C*y*(x-x0-A/2*(y-y0)))/(B*h+C*(y-y0))**2
		deta_dx = 0
		dxi_dx = 1/(B*h+C*(y-y0))

		phi0 = bspline2(xi-xi0,1)
		phi1 = bspline3(eta-eta0,1)

		dphi0_dxi = bspline2_dx(xi-xi0,1)
		dphi1_deta = bspline3_dx(eta-eta0,1)

	dphi0_dx = dphi0_dxi*dxi_dx
	dphi0_dy = dphi0_dxi*dxi_dy

	dphi1_dx = dphi1_deta*deta_dx
	dphi1_dy = dphi1_deta*deta_dy

	dphi_dx = dphi0_dx*phi1 + dphi1_dx*phi0
	dphi_dy = dphi0_dy*phi1 + dphi1_dy*phi0
	dphi = np.array([dphi_dx,dphi_dy],dtype=np.float128)

	if xi1 is None:
		return dphi

	_dphi = dphi_trap(x,y,h,x0,y0,xi1,eta1,map_type)
	return dphi @ _dphi


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
