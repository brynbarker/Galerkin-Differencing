
import numpy as	np
import matplotlib.pyplot as	plt

### functions mapping interface/convex combination information

def interface_to_convex(loc,dof_loc,h,inter_d):
	convex_d = {}
	for comp in range(2):
		comp_bool,comp_alpha,comp_even = False,None,None
		for pair in inter_d[comp]:
			inter,lims,sgn = pair
			dist = abs(loc[1-comp]-inter)
			if (lims[0]< loc[comp] < lims[1]) and (0 < dist < h):
				comp_bool = True
				comp_alpha = sgn*dist

				tmp = dof_loc[comp] if dof_loc[comp]>0 else h-dof_loc[comp]
				comp_even = int(tmp/h)%2 == 0

		convex_d[comp] = (comp_bool,comp_even,comp_alpha)
	return convex_d

def ref_interface_to_convex(ref_inter_d):
	convex_d = {}
	for comp in range(2):
		comp_bool,comp_alpha,comp_even = False,None,None
		for pair in ref_inter_d[comp]:
			inter,sgn = pair
			dist = inter+sgn*loc[1-comp]
			if 0 < dist < h:
				comp_bool = True
				comp_alpha = sgn*dist

				tmp = dof_loc[comp] if dof_loc[comp]>0 else h-dof_loc[comp]
				comp_even = int(tmp/h)%2 == 0

		convex_d[comp] = (comp_bool,comp_even,comp_alpha)
	return convex_d

### 1d local functions

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
	modx = abs(x)/h
	r =	modx + 2.0
	r2 = r**2
	r3 = r2*r

	if modx	<= 1.0:
		return 1.0/6.0 * (3*r3 -24*r2 +	60*r - 44)
	elif modx <= 2.0:
		return 1.0/6.0 * (-r3 +	12*r2 -	48*r + 64)
	else:
		return 0.0

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

def linear(x,h,half=False):
	scale = 1. if not half else bspline2(h/2,h)
	return bspline2(x,h)*scale

def	quadratic(x,h,half=False,sgn=1):
	"""bspline-3 kernel	for	dof	x_i
	with support from [x_{i-1},x_{i+2}]
	with cell width	h
	"""
	scale = 1. if not half else bspline3(sgn*h/2,h)
	return bspline3(x,h)*scale

def	cubic(x,h,half=False):
	"""bspline-4 kernel	for	dof	x_i
	with support from [x_{i-2},x_{i+2}]
	with cell width	h
	"""
	scale = 1. if not half else bspline4(h/2,h)
	return bspline4(x,h)*scale

### 1d local function derivatives

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
	return bspline3_dx(x,h)
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

def linear_dx(x,h):
	return phi1_dx(x,h)

def quadratic_dx(x,h):
	return phi2_dx(x,h)

def cubic_dx(x,h):
	return bspline4_dx(x,h)

### 1d local split functions

def quadratic_split(x,h,even,half=False,sgn=1):
	if even:
		if -h <= x < h/2:
			return quadratic(x+7*h/2,2*h,half,sgn)
		elif h/2 <= x < 2*h:
			return quadratic(x-5*h/2,2*h,half,sgn)
		else:
			return 0
	else:
		if -h <= x < 2*h:
			return quadratic(x+h/2,2*h,half,sgn)
		else:
			return 0

def linear_split(x,h,even,half=False):
	if even:
		if -h <= x < -h/2:
			return linear(x-h/2,2*h,half)
		elif -h/2 <= x < h/2:
			return linear(x+3*h/2,2*h,half)
		elif h/2 <= x < h:
			return linear(x-5*h/2,2*h,half)
		else:
			return 0
	else:
		if -h <= x < h/2:
			return linear(x+h/2,2*h,half)
		elif h/2 <= x < h:
			return linear(x-3*h/2,2*h,half)
		else:
			return 0

def cubic_split(x,h,even,half=False):
	if even:
		if -2*h <= x < -3*h/2:
			return cubic(x+11*h/2,2*h,half)
		elif -3*h <= x < h/2:
			return cubic(x-5*h/2,2*h,half)
		elif h/2 <= x < 2*h:
			return cubic(x+3*h/2,2*h,half)
		else:
			return 0
	else:
		if -2*h <= x < 3*h/2:
			return cubic(x+h/2,2*h,half)
		elif 3*h/2 <= x < 2*h:
			return cubic(x-7*h/2,2*h,half)
		else:
			return 0

def split_phi1(x,h,even,half=False):
	return linear_split(x,h,even,half)

def split_phi2(x,h,even,half=False,sgn=1):
	if even:
		if -h <= x < h/2:
			return phi2(x+7*h/2,2*h)
		elif h/2 <= x < 2*h:
			return phi2(x-5*h/2,2*h)
		else:
			return 0
	else:
		if -h <= x < 2*h:
			return phi2(x+h/2,2*h)
		else:
			return 0
	return quadratic_split(x,h,even,half,sgn)

def split_phi3(x,h,even,half=False):
	if even:
		if -2*h <= x < -3*h/2:
			return phi3(x+11*h/2,2*h)
		elif -3*h/2 <= x < h/2:
			return phi3(x-5*h/2,2*h)
		elif h/2 <= x < 2*h:
			return phi3(x+3*h/2,2*h)
		else:
			return 0
	else:
		if -2*h <= x < 3*h/2:
			return phi3(x+h/2,2*h)
		elif 3*h/2 <= x < 2*h:
			return phi3(x-7*h/2,2*h)
		else:
			return 0

	return cubic_split(x,h,even,half)

### 1d local split function derivatives

def quadratic_split_dx(x,h,even):
	if even:
		if -h <= x < h/2:
			return quadratic_dx(x+7*h/2,2*h)
		elif h/2 <= x < 2*h:
			return quadratic_dx(x-5*h/2,2*h)
		else:
			return 0
	else:
		if -h <= x < 2*h:
			return quadratic_dx(x+h/2,2*h)
		else:
			return 0

def linear_split_dx(x,h,even):
	if even:
		if -h <= x < -h/2:
			return linear_dx(x-h/2,2*h)
		elif -h/2 <= x < h/2:
			return linear_dx(x+3*h/2,2*h)
		elif h/2 <= x < h:
			return linear_dx(x-5*h/2,2*h)
		else:
			return 0
	else:
		if -h <= x < h/2:
			return linear_dx(x+h/2,2*h)
		elif h/2 <= x < h:
			return linear_dx(x-3*h/2,2*h)
		else:
			return 0

def cubic_split_dx(x,h,even):
	if even:
		if -2*h <= x < -3*h/2:
			return cubic_dx(x+11*h/2,2*h)
		elif -3*h <= x < h/2:
			return cubic_dx(x-5*h/2,2*h)
		elif h/2 <= x < 2*h:
			return cubic_dx(x+3*h/2,2*h)
		else:
			return 0
	else:
		if -2*h <= x < 3*h/2:
			return cubic_dx(x+h/2,2*h)
		elif 3*h/2 <= x < 2*h:
			return cubic_dx(x-7*h/2,2*h)
		else:
			return 0

def dsplit_phi1(x,h,even):
	return linear_split_dx(x,0,h,even)

def dsplit_phi2(x,h,even):
	return quadratic_split_dx(x,0,h,even)

def dsplit_phi3(x,h,even):
	return cubic_split_dx(x,0,h,even)

### map orders to functions

func_map = {0:phi0,1:phi1,2:phi2,3:phi3}
# func_map = {0:phi0,1:linear,2:quadratic,3:cubic}
split_map = {0:None,1:split_phi1,2:split_phi2,3:split_phi3}
dsplit_map = {0:None,1:dsplit_phi1,2:dsplit_phi2,3:dsplit_phi3}
dx_map = {0:phi0_dx,1:phi1_dx,2:phi2_dx,3:phi3_dx}

### 2d local functions

def	phi_2d(ords,x,y,h):
	comp_0 = func_map[ords[0]](x,h)
	comp_1 = func_map[ords[1]](y,h)
	return comp_0 *	comp_1

def phi_2d_interface(ords,x,y,h,conv_d):
	x_bool,x_even,x_alpha = conv_d[0]
	y_bool,y_even,y_alpha = conv_d[1]

	if x_bool:
		alph = abs(x_alpha)
		comp_0a = func_map[ords[0]](x,h)
		comp_0b = split_map[ords[0]](x,h,x_even)
		comp_0 = comp_0a*alph + comp_0b*(1-alph)
	else:
		comp_0 = func_map[ords[0]](x,h)
	if y_bool:
		alph = abs(y_alpha)
		comp_1a = func_map[ords[1]](y,h)
		comp_1b = split_map[ords[1]](y,h,y_even)
		comp_1 = comp_1a*alph + comp_1b*(1-alph)
	else:
		comp_1 = func_map[ords[1]](y,h)
	return comp_0 * comp_1

### 2d local function derivatives

def	dphi_2d(ords,x,y,h):
	comp_0 = func_map[ords[0]](x,h)
	comp_1 = func_map[ords[1]](y,h)
	comp_0_dx =	dx_map[ords[0]](x,h)
	comp_1_dx =	dx_map[ords[1]](y,h)
	return np.array([comp_1*comp_0_dx,comp_0*comp_1_dx])

def dphi_2d_interface(ords,x,y,h,conv_d):
	x_bool,x_even,x_alpha = conv_d[0]
	y_bool,y_even,y_alpha = conv_d[1]

	if x_bool:
		alph = abs(x_alpha)
		comp_0a = func_map[ords[0]](x,h)
		comp_0b = split_map[ords[0]](x,h,x_even)
		comp_0 = comp_0a*alph + comp_0b*(1-alph)
		
		comp_0a_dx = dx_map[ords[0]](x,h)
		comp_0b_dx = dsplit_map[ords[0]](x,h,x_even)

		dalph = 1/h if x_alpha>0 else -1/h
		comp_0_dy = dalph*(comp_0a-comp_0b)
		comp_0_dx = comp_0a_dx*x_alpha + comp_0b_dx*(1-x_alpha)
		comp_0_dy = 0
	else:
		comp_0 = func_map[ords[0]](x,h)
		comp_0_dx =	dx_map[ords[0]](x,h)
	if y_bool:
		alph = abs(y_alpha)
		comp_1a = func_map[ords[1]](y,h)
		comp_1b = split_map[ords[1]](y,h,y_even)
		comp_1 = comp_1a*alph + comp_1b*(1-alph)

		comp_1a_dy = dx_map[ords[1]](y,h)
		comp_1b_dy = dsplit_map[ords[1]](y,h,y_even)

		dalph = 1/h if y_alpha>0 else -1/h
		comp_1_dx = dalph*(comp_1a-comp_1b)
		comp_1_dy = comp_1a_dy*alph + comp_1b_dy*(1-alph)
	else:
		comp_1 = func_map[ords[1]](y,h)
		comp_1_dy =	dx_map[ords[1]](y,h)
		comp_1_dx = 0

	return np.array([comp_1*comp_0_dx+comp_0*comp_1_dx,
				     comp_1*comp_0_dy+comp_0*comp_1_dy])

### 2d global functions

def	phi_2d_eval(ords,x_in,y_in,h,x0,y0,inter_d=None):
	x,y	= x_in-x0,y_in-y0
	if inter_d is None:
		return phi_2d(ords,x,y,h)

	convex_d = interface_to_convex([x_in,y_in],[x0,y0],h,inter_d)
	return phi_2d_interface(ords,x,y,h,convex_d)

def	dphi_2d_eval(ords,x_in,y_in,h,x0,y0,inter_d=None):
	x,y	= x_in-x0,y_in-y0
	if inter_d is None:
		return dphi_2d(ords,x,y,h)

	convex_d = interface_to_convex([x_in,y_in],[x0,y0],h,inter_d)
	return dphi_2d_interface(ords,x,y,h,convex_d)

### 2d reference functions

def	phi_2d_ref(ords,x_ref,y_ref,h,ind,ref_inter_d=None):
	i,j	= ind
	xL,	yL = int(ords[0]/2), int(ords[1]/2)
	x,y	= x_ref+h*(xL-j),y_ref+h*(yL-i)

	if ref_inter_d is None:
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
		my_dphi	= lambda x_ref,y_ref,z_ref,h,ind:	dphi_3d_ref(ords,x_ref,y_ref,z_ref,h,ind)
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
