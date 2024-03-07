import numpy as np
import matplotlib.pyplot as plt
from matplotlip.animation import FuncAnimation

#visualization helpers
#animators

def animate_2d(frame,data,size):
    fig,ax = plt.subplots(figsize=(10,10))

    line, = ax.plot(frame[0],frame[1],'lightgrey')
    blocks, dots = [], []
    for i in range(size):
        block, = ax.plot([],[])
        dot, = ax.plot([],[],c='k',marker='o')
        block.append(block)
        dots.append(dot)

    def update(n):
        blocks_n, dots_n = data[n]
        line.set_data(frame[0],frame[1],'lightgrey')
        for i in range(size):
            if i < len(blocks_n):
                blocks[i].set_data(blocks_n[i][0],blocks_n[i][1])
                dots[i].set_data(dots_n[i][0],dots_n[i][1])
            else:
                blocks[i].set_data([],[])
                dots[i].set_data([],[])
        return [line]+blocks+dots

    ani = FuncAnimation(fig, update, frames=len(data), interval=20)
    return ani

        

    

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

