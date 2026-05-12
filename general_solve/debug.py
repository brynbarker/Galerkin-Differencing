import numpy as	np
import matplotlib.pyplot as	plt

def check_continuity(var):
    eps = 1e-3
    doms = [np.linspace(.25-eps,.25+eps),np.linspace(.75-eps,.75+eps)]
    rand_x = np.random.random(len(var.constraints.true_dofs))
    rand_u = var.constraints.spC.dot(rand_x)
    rand_sol = var.sol(rand_u)
    plt.figure(figsize=(50,10))
    for j,other in enumerate(np.linspace(.15,.85,8)):
        for i,dom in enumerate(doms):
            plt.subplot(2,8,8*i+j+1)
            mymin,mymax = 1e10,-1e10
            rand_vals_0 = [rand_sol([x,other]) for x in dom]
            plt.plot(dom,rand_vals_0,label='y = '+str(round(other,3)),lw=3)
            rand_vals_1 = [rand_sol([other,x]) for x in dom]
            plt.plot(dom,rand_vals_1,label='x = '+str(round(other,3)),lw=3)
            mymin = min(mymin,min(rand_vals_0+rand_vals_1))
            mymax = max(mymax,max(rand_vals_0+rand_vals_1))
            if i:plt.plot([.75,.75],[mymin,mymax],'k:')
            else:plt.plot([.25,.25],[mymin,mymax],'k:')
            plt.xticks([min(doms[i]),.25+.5*i,max(doms[i])])
            plt.title(str(round(other,3)),fontsize=20)
            # plt.legend()
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