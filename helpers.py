import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML

#visualization helpers

def vis_3d_vals(l_func,l_a,l_b):
    vis_data = []
    for (f,a,b) in zip(l_func,l_a,l_b):
        xdom = np.linspace(a[0],b[0])
        ydom = np.linspace(a[1],b[1])
        X,Y = np.meshgrid(xdom,ydom)
        F = np.array([f(x,y) for (x,y) in \
                      zip(X.flatten(),Y.flatten())]).reshape(X.shape)
        vis_data.append([X,Y,F])
    return vis_data


def vis_3d_multiple(figs,rotate=False,extras=None):
    num = len(figs)
    if not rotate:
        fig,axs = plt.subplots(1,num,
                              figsize=(15,5),
                              subplot_kw={"projection":"3d"})
    else:
        fig,axs = extras
    for ax,(f,a,b) in zip(axs,figs):
        data = vis_3d_vals(f,a,b)
        count = len(data)
        for (X,Y,F) in data:
            if count == 1:
                surf = ax.plot_surface(X,Y,F,cmap="PiYG")
            else:
                surf = ax.plot_surface(X,Y,F,alpha=.5)
    if not rotate:
        plt.show()
    else:
        return fig,




def vis_3d(l_func,l_a,l_b,rotate=False,extras=None):
    count = len(l_func)
    if not rotate:
        fig,ax = plt.subplots(subplot_kw={"projection":"3d"})
    else:
        fig,ax = extras
    data = vis_3d_vals(l_func,l_a,l_b)
    for (X,Y,F) in data:
        if count == 1:
            surf = ax.plot_surface(X,Y,F,cmap="PiYG")
        else:
            surf = ax.plot_surface(X,Y,F,alpha=.5)
    if not rotate:
        plt.show()
    else:
        return fig,




#animators

def vis_3d_rotate(figs):
    num = len(figs)
    fig,axs = plt.subplots(1,num,
                          figsize=(2*num,3),
                          subplot_kw={"projection":"3d"})
    if num == 1:
        l_func,l_a,l_b = figs[0]
        init = lambda _=None: vis_3d(l_func,l_a,l_b,True,[fig,axs])
    else:
        init = lambda _=None: vis_3d_multiple(figs,True,[fig,axs])

    def animate(i):
        axs.view_init(elev=20., azim=i)
        return fig,

    ani = FuncAnimation(fig,animate,init_func=init,frames=360,interval=20)
    plt.close()
    return HTML(ani.to_html5_video())

def animate_2d(frames,data,size,figsize=(10,10),yesdot=True):
    fig,ax = plt.subplots(figsize=figsize)
    frame = frames[0]
    ax.set_xlim(frame[0][0],frame[0][-1])
    ax.set_ylim(min(data[0][0][1])-.1,max(data[0][0][1])+.1)
    
    line, = ax.plot(frame[0],frame[1],'lightgrey')
    blocks, dots = [], []
    for i in range(size):
        block, = ax.plot([],[])
        if yesdot: dot, = ax.plot([],[],c='k',marker='o')
        blocks.append(block)
        if yesdot: dots.append(dot)

    def update(n):
        if yesdot: blocks_n, dots_n = data[n]
        else: blocks_n = data[n]
        if len(frames) > 1: frame = frames[n]
        else: frame = frames[0]
        line.set_data(frame[0],frame[1])
        for i in range(size):
            if i < len(blocks_n):
                blocks[i].set_data(blocks_n[i][0],blocks_n[i][1])
                if yesdot: dots[i].set_data(dots_n[i][0],dots_n[i][1])
            else:
                blocks[i].set_data([],[])
                if yesdot: dots[i].set_data([],[])
        to_return = [line]+blocks
        if yesdot: to_return += dots
        return to_return

    ani = FuncAnimation(fig, update, frames=len(data), interval=100)
    plt.close()
    return HTML(ani.to_html5_video())

        

    

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

def local_mass(h,interface=False,top=False,qpn=5):
    M = np.zeros((16,16))
    id_to_ind = {ID:[int(ID/4),ID%4] for ID in range(16)}

    for test_id in range(16):

        test_ind = id_to_ind[test_id]
        phi_test = lambda x,y: phi3_2d_ref(x,y,h,test_ind,interface,top)

        for trial_id in range(i,16):

            trial_ind = id_to_ind[trial_id]
            phi_trial = lambda x,y: phi3_2d_ref(x,y,h,trial_ind,interface,top)

            func = lambda x,y: phi_trial(x,y) * phi_test(x,y)
            val = gauss(func,0,h,0,h,qpn)

            M[test_ind,trial_ind] += val
            M[trial_ind,test_ind] += val * (test_ind != trial_ind)
    return M

