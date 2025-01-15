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

def phi1_2d(x,y,h):
    return phi1(x,h)*phi1(y,h)

def grad_phi1(x,y,h):
    phi_j = phi1(x,h)
    dphi_j_dx = phi1_dx(x,h)
    phi_i = phi1(y,h)
    dphi_i_dy = phi1_dx(y,h)
    return np.array([phi_i*dphi_j_dx,phi_j*dphi_i_dy])

def grad_phi1_eval(x_in,y_in,h,x0,y0):
    x,y = x_in-x0,y_in-y0
    return grad_phi1(x,y,h)

def phi1_2d_eval(x_in,y_in,h,x0,y0):
    x,y = x_in-x0,y_in-y0
    return phi1_2d(x,y,h)

def phi1_2d_ref(x_ref,y_ref,h,ind):
    i,j = ind
    x,y = x_ref-h*j,y_ref-h*i
    return phi1_2d(x,y,h)

def grad_phi1_ref(x_ref,y_ref,h,ind):
    i,j = ind
    x,y = x_ref-h*j,y_ref-h*i
    return grad_phi1(x,y,h)
