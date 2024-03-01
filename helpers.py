import numpy as np
import matplotlib.pyplot as plt

#visualization helpers
#animators
#integrators

def gauss(f,a,b,c,d,n):
    xmid, ymid = (a+b)/2, (c+d)/2
    xscale, yscale = (b-a)/2, (d-c)/2
    [p,w] = np.polynomial.legendre.leggauss(n)
    outer = 0.
    for j in range(n):
        inner = 0.
        for i in range(n):
            inner += w[i]*f(xscale*p[j]+xmid,yscale*p[i]+ymid)
        outer += w[j]*inner
    return outer*xscale*yscale

def local_stiffness(h,interface=False,top=False,qpn=5):
    K = np.zeros((16,16))
    id_to_ind = {ID:[int(ID/4),ID%4] for ID in range(16)}

    for test_id in range(16):

        test_ind = id_to_ind[test_id]
        grad_phi_test = lambda x,y: grad_phi3_ref(x,y,h,test_ind,interface,top)

        for trial_id in range(i,16):

            trial_ind = id_to_ind[trial_id]
            grad_phi_trial = lambda x,y: grad_phi3_ref(x,y,h,trial_ind,interface,top)

            func = lambda x,y: grad_phi_trial(x,y) @ grad_phi_test(x,y)
            val = gauss(func,0,h,0,h,qpn)

            K[test_ind,trial_ind] += val
            K[trial_ind,test_ind] += val * (test_ind != trial_ind)
    return K

