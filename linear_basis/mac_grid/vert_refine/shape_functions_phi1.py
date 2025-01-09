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

def phi1_interface(x,h,L):
    if L == 0 and (0 < x <= 3/4*h): #coarse
        return 1-4/3/h*x
    elif L == 1 and (-3/2*h < x <= 0): #fine
        return 1+2/3/h*x
    else:
        return phi1(x,h)
        
def phi1_interface_dx(x,h,L):
    if L == 0 and (0 < x <= 3/4*h): #coarse
        return -4/3/h
    elif L == 1 and (-3/2*h < x <= 0): #fine
        return 2/3/h
    else:
        return phi1_dx(x,h)
        
def phi1_2d(x,y,h,L=None):
    if L is not None:
        return phi1(x,h)*phi1_interface(y,h,L)
    return phi1(x,h)*phi1(y,h)

def grad_phi1(x,y,h,L=None):
    phi_j = phi1(x,h)
    dphi_j_dx = phi1_dx(x,h)
    if L is not None:
        phi_i = phi1_interface(y,h,L)
        dphi_i_dy = phi1_dx_interface(y,h,L)
    else:
        phi_i = phi1(y,h)
        dphi_i_dy = phi1_dx(y,h)
    return np.array([phi_i*dphi_j_dx,phi_j*dphi_i_dy])

def grad_phi1_eval(x_in,y_in,h,x0,y0,L=None):
    x,y = x_in-x0,y_in-y0
    return grad_phi1(x,y,h,L)

def phi1_2d_eval(x_in,y_in,h,x0,y0,L=None):
    x,y = x_in-x0,y_in-y0
    return phi1_2d(x,y,h,L)

def phi1_2d_ref(x_ref,y_ref,h,ind,L=None):
    i,j = ind
    x,y = x_ref-h*j,y_ref-h*i
    return phi1_2d(x,y,h,L)

def grad_phi1_ref(x_ref,y_ref,h,ind,L=None):
    i,j = ind
    x,y = x_ref-h*j,y_ref-h*i
    return grad_phi1(x,y,h,L)
