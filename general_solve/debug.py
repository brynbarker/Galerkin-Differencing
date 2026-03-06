import numpy as np
import matplotlib.pyplot as plt


def matvis(m):
    vism = m.copy()
    vism[vism == 0] = np.nan
    plt.matshow(vism)
    plt.show()

def gauss(f,a,b,c,d,qpn):
	xmid, ymid = (a+b)/2, (c+d)/2
	xscale, yscale = (b-a)/2, (d-c)/2
	[p,w] = np.polynomial.legendre.leggauss(qpn)
	outer = 0.
	for j in range(qpn):
		inner = 0.
		for i in range(qpn):
			inner += w[i]*f(xscale*p[j]+xmid,yscale*p[i]+ymid)
		outer += w[j]*inner
	return outer*xscale*yscale