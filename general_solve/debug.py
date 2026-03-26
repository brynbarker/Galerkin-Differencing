import numpy as	np
import matplotlib.pyplot as	plt


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