
import numpy as	np
import matplotlib.pyplot as	plt

def phi0(x,h):
	if 0 <= x <= h:
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

def phi0_dx(x,h):
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

def phi1_interface(y,h,s,coll=True):
    if y > h or y <= h: return 0
    fine = phi1(y,h)
    
    if coll:
        coarse = phi1(y/2,h)
    else:
        if -h < y <= 0:
            coarse = phi3((y-h)/2,h)
        else:
            coarse = phi3((y+h)/2,h)
    return s*fine + (1-s)*coarse

def phi1_interface_dy(y,h,s,coll=True):
    if y > h or y <= h: return 0
    d_fine = phi1_dx(y,h)
    
    if coll:
        d_coarse = phi1_dx(y/2,h)
    else:
        if -h < y <= 0:
            d_coarse = phi1_dx((y-h)/2,h)/2
        else:
            d_coarse = phi1_dx((y+h)/2,h)/2
    return s*d_fine + (1-s)*d_coarse

def phi1_interface_dx(y,h,s,coll=True):
    if y > h or y <= -h: return 0
    fine = phi1(y,h)
    
    if coll:
        coarse = phi1(y/2,h)
    else:
        if -h < y <= 0:
            coarse = phi1((y-h)/2,h)
        else:
            coarse = phi1((y+h)/2,h)
    return (fine - coarse)/h

def phi2_interface(y,h,s,coll=True):
    if y > 2*h or y <= -h: return 0
    fine = phi2(y,h)
    
    if coll:
        coarse = phi2(y/2,h)
    else:
        if -h < y <= h:
            coarse = phi2((y+3*h)/2,h)
        else:
            coarse = phi2((y-h)/2,h)
    return s*fine + (1-s)*coarse

def phi2_interface_dy(y,h,s,coll=True):
    if y > 2*h or y <= -h: return 0
    d_fine = phi2_dx(y,h)
    
    if coll:
        d_coarse = phi2_dx(y/2,h)
    else:
        if -h < y <= h:
            d_coarse = phi2_dx((y+3*h)/2,h)/2
        else:
            d_coarse = phi2_dx((y-h)/2,h)/2
    return s*d_fine + (1-s)*d_coarse

def phi2_interface_dx(y,h,s,coll=True):
    if y > 2*h or y <= -h: return 0
    fine = phi2(y,h)
    
    if coll:
        coarse = phi2(y/2,h)
    else:
        if -h < y <= h:
            coarse = phi2((y+3*h)/2,h)
        else:
            coarse = phi2((y-h)/2,h)
    return (fine - coarse)/h

def phi3_interface(y,h,s,coll=True):
    if y > 2*h or y <= -2*h: return 0
    fine = phi3(y,h)
    
    if coll:
        coarse = phi3(y/2,h)
    else:
        if -2*h < y <= -h:
            coarse = phi3((y-h)/2,h)
        elif -h < y <= 0:
            coarse = phi3((y-3*h)/2,h)
        elif 0 < y <= h:
            coarse = phi3((y+3*h)/2,h)
        else:
            coarse = phi3((y+h)/2,h)
    return s*fine + (1-s)*coarse

def phi3_interface_dy(y,h,s,coll=True):
    if y > 2*h or y <= -2*h: return 0
    d_fine = phi3_dx(y,h)
    
    if coll:
        d_coarse = phi3_dx(y/2,h)
    else:
        if -2*h < y <= -h:
            d_coarse = phi3_dx((y-h)/2,h)/2
        elif -h < y <= 0:
            d_coarse = phi3_dx((y-3*h)/2,h)/2
        elif 0 < y <= h:
            d_coarse = phi3_dx((y+3*h)/2,h)/2
        else:
            d_coarse = phi3_dx((y+h)/2,h)/2
    return s*d_fine + (1-s)*d_coarse

def phi3_interface_dx(y,h,s,coll=True):
    if y > 2*h or y <= -2*h: return 0
    fine = phi3(y,h)
    
    if coll:
        coarse = phi3(y/2,h)
    else:
        if -2*h < y <= -h:
            coarse = phi3((y-h)/2,h)
        elif -h < y <= 0:
            coarse = phi3((y-3*h)/2,h)
        elif 0 < y <= h:
            coarse = phi3((y+3*h)/2,h)
        else:
            coarse = phi3((y+h)/2,h)
    return (fine - coarse)/h

func_map = {1:phi1,2:phi2,3:phi3}
dx_map = {1:phi1_dx,2:phi2_dx,3:phi3_dx}
inter_map = {1:phi1_interface,2:phi2_interface,3:phi3_interface}
d_other_inter_map = {1:phi1_interface_dx,2:phi2_interface_dx,3:phi3_interface_dx}
dinter_map = {1:phi1_interface_dy,2:phi2_interface_dy,3:phi3_interface_dy}

def	phi_2d(ords,x,y,h):
	comp_0 = func_map[ords[0]](x,h)
	comp_1 = func_map[ords[1]](y,h)
	return comp_0 *	comp_1

def phi_2d_inter(ords,x,y,h,xinter,yinter,s,coll):
	sx,sy = s
	collx,colly = coll
	if yinter:
		comp_0 = inter_map[ords[0]](x,h,sy,collx)
	else:
		comp_0 = func_map[ords[0]](x,h)
	if xinter:
		comp_1 = inter_map[ords[1]](y,h,sx,colly)
	else:
		comp_1 = func_map[ords[1]](y,h)
	return comp_0 *	comp_1

def	dphi_2d(ords,x,y,h):
	comp_0 = func_map[ords[0]](x,h)
	comp_1 = func_map[ords[1]](y,h)
	comp_0_dx =	dx_map[ords[0]](x,h)
	comp_1_dx =	dx_map[ords[1]](y,h)
	return np.array([comp_1*comp_0_dx,comp_0*comp_1_dx])

def dphi_2d_inter(ords,x,y,h,xinter,yinter,s,coll):
	sx,sy = s
	collx,colly = coll
	if yinter:
		comp_0 = inter_map[ords[0]](x,h,sy,collx)
		comp_0_dx =	dinter_map[ords[0]](x,h,sy,collx)
		comp_0_dy = d_other_inter_map[ords[0]](x,h,sy,collx)
	else:
		comp_0 = func_map[ords[0]](x,h)
		comp_0_dx =	dx_map[ords[0]](x,h)
		comp_0_dy = 0

	if xinter:
		comp_1 = inter_map[ords[1]](y,h,sx,colly)
		comp_1_dy =	dinter_map[ords[1]](y,h,sx,colly)
		comp_1_dx = d_other_inter_map[ords[1]](y,h,sx,colly)
	else:
		comp_1 = func_map[ords[1]](y,h)
		comp_1_dy =	dx_map[ords[1]](y,h)
		comp_1_dx = 0

	grad = np.array([
				comp_0*comp_1_dx+comp_0_dx*comp_1,
				comp_0*comp_1_dy+comp_0_dy*comp_1])
	return grad


def	phi_2d_eval(ords,x_in,y_in,h,x0,y0):
	x,y	= x_in-x0,y_in-y0
	return phi_2d(ords,x,y,h)

def	dphi_2d_eval(ords,x_in,y_in,h,x0,y0):
	x,y	= x_in-x0,y_in-y0
	return dphi_2d(ords,x,y,h)

def phi_2d_eval_inter(ords,x_in,y_in,h,x0,y0,xinter=None,yinter=None):
	if xinter is not None:
		xbool = True
		sx = abs(xinter-x_in)/h
		colly = 1-int(y0/h)%2
	else:
		xbool = False
		sx, colly = None,None
	if yinter is not None:
		ybool = True
		sy = abs(yinter-y_in)/h
		collx = 1-int(x0/h)%2
	else:
		ybool = False
		sy, collx = None,None
	x,y	= x_in-x0,y_in-y0
	return phi_2d_inter(ords,x,y,h,xbool,ybool,[sx,sy],[collx,colly])

def dphi_2d_eval_inter(ords,x_in,y_in,h,x0,y0,xinter=None,yinter=None):
	if xinter is not None:
		xbool = True
		sx = abs(xinter-x_in)/h
		colly = 1-int(y0/h)%2
	else:
		xbool = False
		sx, colly = None,None
	if yinter is not None:
		ybool = True
		sy = abs(yinter-y_in)/h
		collx = 1-int(x0/h)%2
	else:
		ybool = False
		sy, collx = None,None
	x,y	= x_in-x0,y_in-y0
	return dphi_2d_inter(ords,x,y,h,xbool,ybool,[sx,sy],[collx,colly])


def quad_ind_colloc(quad,i,j):
	xcoll = (j%2) == (1 - quad%2)
	ycoll = (i%2) == (1-int(quad/2))
	return [xcoll,ycoll]

def	phi_2d_ref(ords,x_ref,y_ref,h,ind):
	i,j	= ind
	xL,	yL = int(ords[0]/2), int(ords[1]/2)
	x,y	= x_ref+h*(xL-j),y_ref+h*(yL-i)
	return phi_2d(ords,x,y,h)

def phi_2d_ref_inter(ords,x_ref,y_ref,h,ind,xinter,yinter):
	i,j	= ind
	xL,	yL = int(ords[0]/2), int(ords[1]/2)
	x,y	= x_ref+h*(xL-j),y_ref+h*(yL-i)

	coll = [j%2==1, i%2==1]
	return phi_2d_inter(ords,x,y,h,xinter,yinter,[x_ref,y_ref],coll)

def	dphi_2d_ref(ords,x_ref,y_ref,h,ind):
	i,j	= ind
	xL,	yL = int(ords[0]/2), int(ords[1]/2)
	x,y	= x_ref+h*(xL-j),y_ref+h*(yL-i)
	return dphi_2d(ords,x,y,h)

def dphi_2d_ref_inter(ords,x_ref,y_ref,h,ind,xinter,yinter):
	i,j	= ind
	xL,	yL = int(ords[0]/2), int(ords[1]/2)
	x,y	= x_ref+h*(xL-j),y_ref+h*(yL-i)

	coll = [j%2==1, i%2==1]
	return dphi_2d_inter(ords,x,y,h,xinter,yinter,[x_ref,y_ref],coll)

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

def _get_phi_refs_inter(ords):
	def myphi_xonly(x_ref,y_ref,h,ind):
		return phi_2d_ref_inter(ords,x_ref,y_ref,h,ind,True,False)
	def myphi_yonly(x_ref,y_ref,h,ind):
		return phi_2d_ref_inter(ords,x_ref,y_ref,h,ind,False,True)
	def myphi_corner(x_ref,y_ref,h,ind):
		return phi_2d_ref_inter(ords,x_ref,y_ref,h,ind,True,True)

	def mydphi_xonly(x_ref,y_ref,h,ind):
		return dphi_2d_ref_inter(ords,x_ref,y_ref,h,ind,True,False)
	def mydphi_yonly(x_ref,y_ref,h,ind):
		return dphi_2d_ref_inter(ords,x_ref,y_ref,h,ind,False,True)
	def mydphi_corner(x_ref,y_ref,h,ind):
		return dphi_2d_ref_inter(ords,x_ref,y_ref,h,ind,True,True)

	myphis = [myphi_xonly,myphi_yonly,myphi_corner]
	mydphis = [mydphi_xonly,mydphi_yonly,mydphi_corner]
	return myphis,mydphis
