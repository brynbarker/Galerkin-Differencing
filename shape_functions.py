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

def phi3_interface_eval(y_in,h,y0,s):
    y = y_in-y0
    coll = 1-int(y0/h)%2
    return phi3_interface(y,h,s,coll)
     

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

def phi3_2d(x,y,h,s=None,interface=False,coll=False):
    if interface:
        return phi3(x,h)*phi3_interface(y,h,s,coll)
    return phi3(x,h)*phi3(y,h)

def grad_phi3(x,y,h,s=None,interface=False,coll=True):
    phi_j = phi3(x,h)
    dphi_j_dx = phi3_dx(x,h)
    if interface:
        phi_i = phi3_interface(y,h,s,coll)

        dphi_i_dx = phi3_interface_dx(y,h,s,coll)

        dphi_i_dy = phi3_interface_dy(y,h,s,coll)
        return np.array([phi_j*dphi_i_dx+phi_i*dphi_j_dx,phi_j*dphi_i_dy])
    phi_i = phi3(y,h)
    dphi_i_dy = phi3_dx(y,h)
    return np.array([phi_i*dphi_j_dx,phi_j*dphi_i_dy])

def grad_phi3_eval(x_in,y_in,h,x0,y0):
    x,y = x_in-x0,y_in-y0
    dist = x_in-0.5
    s = dist/h
    coll = 1-int(y0/h)%2
    interface = 0 < dist <= h
    return grad_phi3(x,y,h,s,interface,coll)

def phi3_2d_eval(x_in,y_in,h,x0,y0):
    x,y = x_in-x0,y_in-y0
    dist = x_in-0.5
    s = dist/h
    coll = 1-int(y0/h)%2
    interface = 0 < dist <= h
    return phi3_2d(x,y,h,s,interface,coll)

def phi3_2d_ref(x_ref,y_ref,h,ind,interface=False,top=False):
    i,j = ind
    x,y = x_ref+h*(1-j),y_ref+h*(1-i)
    coll = (j+top)%2
    s = x_ref/h
    return phi3_2d(x,y,h,s,interface,coll)
