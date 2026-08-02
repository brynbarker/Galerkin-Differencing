import numpy as	np
import matplotlib.pyplot as	plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML
from scipy.integrate import dblquad

def check_continuity(var,full=False,reps=1):
    eps = 1e-3
    if full:
        def get_max_diff(vals):
            Lvals = vals[::2]
            Rvals = vals[1::2]
            diffs = [abs(rval-lval) for (rval,lval) in zip(Lvals,Rvals)]
            return max(diffs)
        pair_ids = []
        nodes = np.linspace(0,1,4*var.N+1)
        dom = []
        for n in nodes:
            L = max(n-1e-15,0)
            R = min(n+1e-15,1)
            dom.append(L)
            dom.append(R)
            
        doms=  [dom,dom]
        # doms = [np.linspace(0,1,4*var.N+1),np.linspace(0,1,4*var.N+1)]
    else:
        if var.zigzag:
            doms = [np.linspace(.25-var.h,.25+var.h),np.linspace(.75-var.h,.75+var.h)]
        else:
            doms = [np.linspace(.25-eps,.25+eps),np.linspace(.75-eps,.75+eps)]
    rand_x = np.random.random(len(var.constraints.true_dofs))
    rand_u = var.constraints.spC.dot(rand_x)
    rand_sol = var.sol(rand_u)
    plt.figure(figsize=(50,10))
    for i,dom in enumerate(doms):
        j_titles = {k:[] for k in range(8)}
        for jj,other in enumerate(np.linspace(0,1,reps*8+2)[1:-1]):
            j = int(jj/reps)
            plt.subplot(2,8,8*i+j+1)
            mymin,mymax = 1e10,-1e10
            if not full or i == 0:
                rand_vals_0 = [rand_sol([x,other]) for x in dom]
                if full:
                    max_0 = get_max_diff(rand_vals_0)
                else:
                    max_0 = max([abs(rand_vals_0[i+1]-rand_vals_0[i]) for i in range(len(dom)-1)])
                plt.plot(dom,rand_vals_0,label='y = '+str(round(other,3)),lw=3)
            else:
                rand_vals_0,max_0 = [],None
            if not full or i == 1:
                rand_vals_1 = [rand_sol([other,x]) for x in dom]
                if full:
                    max_1 = get_max_diff(rand_vals_1)
                else:
                    max_1 = max([abs(rand_vals_1[i+1]-rand_vals_1[i]) for i in range(len(dom)-1)])
                plt.plot(dom,rand_vals_1,label='x = '+str(round(other,3)),lw=3)
            else:
                rand_vals_1,max_1 = [],None
            mymin = min(mymin,min(rand_vals_0+rand_vals_1))
            mymax = max(mymax,max(rand_vals_0+rand_vals_1))
            if full:
                plt.plot([.75,.75],[mymin,mymax],'k:')
                plt.plot([.25,.25],[mymin,mymax],'k:')
                if var.zigzag:
                    plt.plot([.75-var.h/2,.75-var.h/2],[mymin,mymax],'k:')
                    plt.plot([.25+var.h/2,.25+var.h/2],[mymin,mymax],'k:')
            else:
                if i:plt.plot([.75,.75],[mymin,mymax],'k:')
                else:plt.plot([.25,.25],[mymin,mymax],'k:')
                if var.zigzag:
                    if i:plt.plot([.75-var.h/2,.75-var.h/2],[mymin,mymax],'k:')
                    else:plt.plot([.25+var.h/2,.25+var.h/2],[mymin,mymax],'k:')
            plt.xticks([min(doms[i]),.25+.5*i,max(doms[i])])
            if full:
                vl = max_0 if i==0 else max_1
                j_titles[j].append(round(vl,4))
                plt.title(max(j_titles[j]),fontsize=20)
            else:
                plt.title([round(max_0,3),round(max_1,3)],fontsize=20)
                plt.legend()
    plt.show()

def	matvis(m):
	if not isinstance(m,list):
		m = [m]

	fig,ax = plt.subplots(1,len(m),figsize=(len(m)*5,5))
	for id,mat in enumerate(m):
		vism = mat.copy()
		vism[vism == 0]	= np.nan
		if len(m) > 1:
			cax = ax[id].matshow(vism)
		else:
			cax = ax.matshow(vism)
		fig.colorbar(cax)
	plt.show()

def	gauss(f,a,b,c,d,qpn):
	xmid, ymid = (a+b)/2, (c+d)/2
	xscale,	yscale = (b-a)/2, (d-c)/2
	[p,w] =	np.polynomial.legendre.leggauss(qpn)
	outer =	0.
	for	j in range(qpn):
		inner =	0.
		for	i in range(qpn):
			inner += w[i]*f(xscale*p[j]+xmid,yscale*p[i]+ymid)
		outer += w[j]*inner
	return outer*xscale*yscale

def	gauss_1d(f,a,b,qpn):
	xmid = (a+b)/2
	xscale = (b-a)/2
	[p,w] =	np.polynomial.legendre.leggauss(qpn)
	inner =	0.
	for	i in range(qpn):
		inner += w[i]*f(xscale*p[i]+xmid)
	return inner*xscale

def rank(A, atol=1e-13, rtol=0):
    """Estimate the rank (i.e. the dimension of the nullspace) of a matrix.

    The algorithm used by this function is based on the singular value
    decomposition of `A`.

    Parameters
    ----------
    A : ndarray
        A should be at most 2-D.  A 1-D array with length n will be treated
        as a 2-D with shape (1, n)
    atol : float
        The absolute tolerance for a zero singular value.  Singular values
        smaller than `atol` are considered to be zero.
    rtol : float
        The relative tolerance.  Singular values less than rtol*smax are
        considered to be zero, where smax is the largest singular value.

    If both `atol` and `rtol` are positive, the combined tolerance is the
    maximum of the two; that is::
        tol = max(atol, rtol * smax)
    Singular values smaller than `tol` are considered to be zero.

    Return value
    ------------
    r : int
        The estimated rank of the matrix.

    See also
    --------
    numpy.linalg.matrix_rank
        matrix_rank is basically the same as this function, but it does not
        provide the option of the absolute tolerance.
    """

    A = np.atleast_2d(A)
    s = np.linalg.svd(A, compute_uv=False)
    tol = max(atol, rtol * s[0])
    rank = int((s >= tol).sum())
    return rank


def nullspace(A, atol=1e-13, rtol=0):
    """Compute an approximate basis for the nullspace of A.

    The algorithm used by this function is based on the singular value
    decomposition of `A`.

    Parameters
    ----------
    A : ndarray
        A should be at most 2-D.  A 1-D array with length k will be treated
        as a 2-D with shape (1, k)
    atol : float
        The absolute tolerance for a zero singular value.  Singular values
        smaller than `atol` are considered to be zero.
    rtol : float
        The relative tolerance.  Singular values less than rtol*smax are
        considered to be zero, where smax is the largest singular value.

    If both `atol` and `rtol` are positive, the combined tolerance is the
    maximum of the two; that is::
        tol = max(atol, rtol * smax)
    Singular values smaller than `tol` are considered to be zero.

    Return value
    ------------
    ns : ndarray
        If `A` is an array with shape (m, k), then `ns` will be an array
        with shape (k, n), where n is the estimated dimension of the
        nullspace of `A`.  The columns of `ns` are a basis for the
        nullspace; each element in numpy.dot(A, ns) will be approximately
        zero.
    """

    A = np.atleast_2d(A)
    u, s, vh = np.linalg.svd(A)
    tol = max(atol, rtol * s[0])
    nnz = (s >= tol).sum()
    ns = vh[nnz:].conj().T
    return ns

def vis_dofs(var,rtype=None):
    frame = [[1,1,0,0,1],[0,1,1,0,0]]
    frames = [frame]
    if rtype is not None:
        if rtype == 'vstripe':
             frames.append([[.25,.25,.75,.75],[0,1,1,0]])
        elif rtype == 'hstripe':
             frames.append([[0,1,1,0],[.25,.25,.75,.75]])
        elif rtype == 'square':
             frames.append([[.25,.25,.75,.75,.25],[.75,.25,.25,.75,.75]])
         
    fig,ax = plt.subplots(figsize=(5,5))
    ax.set_xlim(-4*var.h,1+4*var.h)
    ax.set_ylim(-4*var.h,1+4*var.h)

    size = var.integrator.prod*2

    blocks,lines = [],[]
    for frame in frames:
        line, = ax.plot(frame[0],frame[1],'lightgrey')
        lines.append(line)
    for _ in range(size):
        block, = ax.plot([],[])
        blocks.append(block)
    dot, = ax.plot([],[],'ko',linestyle='None')

    def update(n):
        p_id = n >= var.constraints.dof_id_shift
        dof_id = n - var.constraints.dof_id_shift*p_id
        dof = var.mesh.patches[p_id].get_dof(dof_id)
        els = list(dof.elements.values())
        for i in range(len(lines)):
            lines[i].set_data(frames[i][0],frames[i][1])
        dot.set_data([dof.x],[dof.y])
        for i in range(size):
            if i < len(dof.elements):
                e = els[i]
                blocks[i].set_data(e.to_plot[0],e.to_plot[1])
            else:
                blocks[i].set_data([],[])
        return [lines,blocks,dot]
    interval = 400
    ani = FuncAnimation(fig, update, frames=var.constraints.size, interval=interval)
    plt.close()
    return HTML(ani.to_html5_video())

def vis_elements(var,rtype=None):
    frame = [[1,1,0,0,1],[0,1,1,0,0]]
    frames = [frame]
    if rtype is not None:
        if rtype == 'vstripe':
             frames.append([[.25,.25,.75,.75],[0,1,1,0]])
        elif rtype == 'hstripe':
             frames.append([[0,1,1,0],[.25,.25,.75,.75]])
        elif rtype == 'square':
             frames.append([[.25,.25,.75,.75,.25],[.75,.25,.25,.75,.75]])
    fig,ax = plt.subplots(figsize=(5,5))
    ax.set_xlim(-4*var.h,1+4*var.h)
    ax.set_ylim(-4*var.h,1+4*var.h)

    lines = []
    for frame in frames:
        line, = ax.plot(frame[0],frame[1],'lightgrey')
        lines.append(line)
    eline, = ax.plot([],[])
    dot, = ax.plot([],[],'ko',linestyle='None')

    coarse_el_count = len(var.mesh.patches[0].elements)
    def update(n):
        p_id = n >= coarse_el_count
        e_id = n - p_id * coarse_el_count
        e = var.mesh.patches[p_id].get_el(e_id)
        for i in range(len(lines)):
            lines[i].set_data(frames[i][0],frames[i][1])
        eline.set_data(e.to_plot[0],e.to_plot[1])
        xs,ys = [],[]
        for dof in e.dof_list:
            xs.append(dof.x)
            ys.append(dof.y)
        dot.set_data(xs,ys)
        return [lines,eline,dot]
    interval = 400
    el_count = len(var.mesh.patches[0].elements)+len(var.mesh.patches[1].elements)
    ani = FuncAnimation(fig, update, frames=el_count, interval=interval)
    plt.close()
    return HTML(ani.to_html5_video())

def fill_to_int_bounds(xs,y0,y1):
    if len(xs) == 2:
        sgn = 1 if y0[0]<y0[1] else -1
        a,b = xs[0],xs[1]
        c = lambda v: sgn*(y0[1]-y0[0])/(b-a)*(v-y0[0])+xs[0]
        d = lambda v: sgn*(y1[0]-y1[0])/(a-b)*(v-y0[-1])+xs[-1]
        flip = False
    else:
        sgn = 1 if y0[0]==y0[1] else -1
        a,b = y0[1],y1[1]
        c = lambda v: sgn*(xs[1]-xs[0])/(b-a)*(v-y0[0])+xs[0]
        d = lambda v: sgn*(xs[-1]-xs[-2])/(a-b)*(v-y0[-1])+xs[-1]
        flip = True
    return a,b,c,d,flip



def check_quadrature(var):
     for p in var.mesh.patches:
        issues = {}
        for e_id in p.elements:
            issues[e_id] =  {}
            e = p.elements[e_id]
            for j,dof in enumerate(e.dof_list):
                my_phi = lambda x,y: dof.phi([x,y],el=e)
                if e.regular:
                    vals = var.integrator._evaluate_func_at_points(
                          my_phi,e.bounds)
                    my_integral = var.integrator._compute_product_integral(
                         vals,e.vol)
                    a,b,c,d = e.bounds

                    func = lambda y,x: dof.phi([x,y])
                    sp_integral,err_est = dblquad(func,a,b,c,d)
                    if (abs(my_integral-sp_integral)) > 1e-14:
                        issues[e_id][j] = (my_integral,sp_integral)

                else:
                    # my_phi = lambda x,y:x**3
                    vals = var.interface_map.evaluate_func_on_element(e,my_phi,ret_array=False)
                    jdet = var.interface_map.evaluate_func_on_element(e,e.Jt_det,ret_array=False,ref=True)
                    prod = 0
                    for key in vals:
                        prod += vals[key]*jdet[key]
                    my_integral = var.interface_map._compute_product_integral(e.tri,prod)
                    xs,y0,y1 = e.fill(0)
                    a,b,c,d,flip = fill_to_int_bounds(xs,y0,y1)
                    if flip:
                        func=lambda x,y:dof.phi([x,y],el=e)
                    else:
                        func = lambda y,x:dof.phi([x,y],el=e)
                    sp_integral, err_est = dblquad(func,a,b,c,d)
                    true_sol = e.h**2/8 if e.tri else 3*e.h**2/8
                    if (abs(my_integral-sp_integral)) > 1e-14:
                        print(e.tri,flip,abs(my_integral-sp_integral),my_integral,sp_integral,true_sol,sep='\t')
                    # else:
                    #      print(e.tri,'err = {}'.format(abs(my_integral-sp_integral)))


                     
            if len(issues[e_id]) > 0:
                 plt.plot(e.to_plot[0],e.to_plot[1])