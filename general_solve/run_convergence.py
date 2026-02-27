import numpy as np
from general_solve.variable import Solver

doflocs = ['node','cell','xside','yside']
rtypes = ['uniform','stripe','square']
rnames = {'uniform':[None],
		  'stripe':['vertfinecenter',
					'vertcoarsecenter',
					'horzfinecenter',
					'horzcoarsecenter'],
		  'square':['finecenter','coarsecenter']}

u0 = lambda	x: np.sin(2*np.pi*x)+np.cos(2*np.pi*x)
u =	lambda x,y:	u0(x) +	u0(y)
f_lap =	lambda x,y:	-4*np.pi**2*u(x,y)
f_helm = lambda x,y: f_lap(x,y) + u(x,y)

def	run_it(dofloc_ops=doflocs,rtype='uniform',rname_ops=None,ord_ops=[1,2,3],sys='lap'):
	if not isinstance(dofloc_ops,list):
		dofloc_ops=[dofloc_ops]
	if not isinstance(rname_ops,list):
		if rname_ops is None:
			rname_ops = rnames[rtype]
		else:
			rname_ops=[rname_ops]
	if not isinstance(ord_ops,list):
		ord_ops=[ord_ops]

	all_rates =	{}
	Nvals =	np.array([16,32,64])

	print('dofloc\trtype\t\torder\tL2 rates\tLinf rates\n'+'-'*60)
	for	dofloc in dofloc_ops:
		l2rates	= {}
		linfrates =	{}
		for	col,rname in enumerate(rname_ops):
			l2_tmp = []
			linf_tmp = []
			for	row,ord	in enumerate(ord_ops):
				L2,Linf	= [],[]
				L2rs,Linfrs	= [],[]

				for	N in Nvals:
					s =	Solver(N,2,dofloc,rtype,
							   rname=rname,u=u,ords=[ord,ord])
					if sys == 'lap':
						s.solve_poisson(f=f_lap,disp=False)
						L2.append(s.lap.err)
						Linf.append(s.lap.Linf_err)
					if sys == 'helm':
						s.solve_helmholtz(f=f_helm,disp=False)
						L2.append(s.helm.err)
						Linf.append(s.helm.Linf_err)

					del	s

					if N>16:
						l2rate = L2[-2]/L2[-1]
						linfrate = Linf[-2]/Linf[-1]
						myord = ord if N==32 else ' '
						myrn = rname if (row==0 and myord==ord) else '\t'
						mydl = dofloc if (col==0 and myrn==rname) else '  '
						print('{}\t{}\t{}\t{}\t\t{}'.format(
							mydl,myrn,myord,round(l2rate,3),round(linfrate,3)))
						L2rs.append(l2rate)
						Linfrs.append(linfrate)
				l2_tmp.append(L2rs)
				linf_tmp.append(Linfrs)
			l2rates[rname] = l2_tmp
			linfrates[rname] = linf_tmp
		all_rates[dofloc] =	(l2rates,linfrates)
	return all_rates