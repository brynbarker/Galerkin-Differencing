
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

def phi3_2d(x,y,h):
    return phi3(x,h)*phi3(y,h)

def dphi3_2d(x,y,h):
    phi_j = phi3(x,h)
    dphi_j_dx = phi3_dx(x,h)
    phi_i = phi3(y,h)
    dphi_i_dy = phi3_dx(y,h)
    return np.array([phi_i*dphi_j_dx,phi_j*dphi_i_dy])

def dphi3_2d_eval(x_in,y_in,h,x0,y0):
    x,y = x_in-x0,y_in-y0
    return dphi3_2d(x,y,h)

def phi3_2d_eval(x_in,y_in,h,x0,y0):
    x,y = x_in-x0,y_in-y0
    return phi3_2d(x,y,h)

def phi3_2d_ref(x_ref,y_ref,h,ind):
    i,j = ind
    x,y = x_ref+h*(1-j),y_ref+h*(1-i)
    return phi3_2d(x,y,h)

def dphi3_2d_ref(x_ref,y_ref,h,ind):
    i,j = ind
    x,y = x_ref+h*(1-j),y_ref+h*(1-i)
    return dphi3_2d(x,y,h)

def phi3_3d(x,y,z,h):
    return phi3(x,h)*phi3(y,h)*phi3(z,h)

def dphi3_3d(x,y,z,h):
    phi_j = phi3(x,h)
    dphi_j_dx = phi3_dx(x,h)
    phi_i = phi3(y,h)
    dphi_i_dy = phi3_dx(y,h)
    phi_k = phi3(z,h)
    dphi_k_dz = phi3_dx(z,h)
    return np.array([phi_i*phi_k*dphi_j_dx,
                     phi_j*phi_k*dphi_i_dy,
                     phi_j*phi_i*dphi_k_dz])

def dphi3_3d_eval(x_in,y_in,z_in,h,x0,y0,z0):
    x,y,z = x_in-x0,y_in-y0,z_in-z0
    return dphi3_3d(x,y,z,h)

def phi3_3d_eval(x_in,y_in,z_in,h,x0,y0,z0):
    x,y,z = x_in-x0,y_in-y0,z_in-z0
    return phi3_3d(x,y,z,h)

def phi3_3d_ref(x_ref,y_ref,z_ref,h,ind):
    i,j,k = ind
    x,y,z = x_ref+h*(1-j),y_ref+h*(1-i),z_ref+h*(1-k)
    return phi3_3d(x,y,z,h)

def dphi3_3d_ref(x_ref,y_ref,z_ref,h,ind):
    i,j,k = ind
    x,y,z = x_ref+h*(1-j),y_ref+h*(1-i),z_ref+h*(1-k)
    return dphi3_3d(x,y,z,h)
