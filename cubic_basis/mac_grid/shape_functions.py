
import numpy as np
import matplotlib.pyplot as plt

def phi3(x,h):
    if -2*h < x <= -h:
        return (x+3*h)*(x+2*h)*(x+h)/6/h**3
    elif -h < x <= 0:
        return -(x+2*h)*(x+h)*(x-h)/2/h**3
    elif 0 < x <= h:
        return (x+h)*(x-h)*(x-2*h)/2/h**3
    elif h < x <= 2*h:
        return -(x-h)*(x-2*h)*(x-3*h)/6/h**3
    else:
        return 0

def phi3_interface(y,h_in):
	h = 3/4*h_in
	return phi3(y,h)

def phi3_dx(x,h):
    if -2*h < x <= -h:
        return (11*h**2+12*h*x+3*x**2)/6/h**3
    elif -h < x <= 0:
        return (h**2-4*h*x-3*x**2)/2/h**3
    elif 0 < x <= h:
        return -(h**2+4*h*x-3*x**2)/2/h**3
    elif h < x <= 2*h:
        return -(11*h**2-12*h*x+3*x**2)/6/h**3
    else:
        return 0

def phi3_interface_dx(y,h_in):
	h = 3/4*h_in
	return phi3_dx(y,h)

def phi3_2d(x,y,h,I=False):
    if I:
        return phi3(x,h)*phi3_interface(y,h)
    return phi3(x,h)*phi3(y,h)

def grad_phi3(x,y,h,I=False):
    phi_j = phi3(x,h)
    dphi_j_dx = phi3_dx(x,h)
    if I:
        phi_i = phi3_interface(y,h)
        dphi_i_dy = phi3_interface_dx(y,h)
    else:
        phi_i = phi3(y,h)
        dphi_i_dy = phi3_dx(y,h)
    return np.array([phi_i*dphi_j_dx,phi_j*dphi_i_dy])

def grad_phi3_eval(x_in,y_in,h,x0,y0,I=False):
    x,y = x_in-x0,y_in-y0
    return grad_phi3(x,y,h,I)

def phi3_2d_eval(x_in,y_in,h,x0,y0,I=False):
    x,y = x_in-x0,y_in-y0
    return phi3_2d(x,y,h,I)

def phi3_2d_ref(x_ref,y_ref,h,ind,I=False):
    i,j = ind
    x,y = x_ref+h*(1-j),y_ref+h*(1-i)*(1-I*1/4)
    return phi3_2d(x,y,h,I)

def grad_phi3_ref(x_ref,y_ref,h,ind,I=False):
    i,j = ind
    x,y = x_ref+h*(1-j),y_ref+h*(1-i)*(1-I*1/4)
    return grad_phi3(x,y,h,I)
