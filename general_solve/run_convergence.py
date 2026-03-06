import numpy as np
import matplotlib.pyplot as plt
from general_solve.variable import SingleComponentVariable as Var

doflocs = ['node','cell','xside','yside']
rtypes = ['uniform','stripe','square']
rnames = {'uniform':['no'],
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

	d_ops = {'dofs':dofloc_ops,'refs':rname_ops,'ords':ord_ops}

	all_l2_rates =	{}
	all_linf_rates =	{}
	Nvals =	np.array([16,32,64])

	print('dofloc\trtype\t\torder\tL2 rates\tLinf rates\n'+'-'*60)
	for	dofloc in dofloc_ops:
		l2rates	= {}
		linfrates =	{}
		for	col,rname in enumerate(rname_ops):
			l2_tmp = {}
			linf_tmp = {}
			for	row,ord	in enumerate(ord_ops):
				L2,Linf	= [],[]
				L2rs,Linfrs	= [],[]

				for	N in Nvals:
					s =	Var(N,2,dofloc,rtype,
							   rname=rname,var=u,ords=[ord,ord])
					if sys == 'lap':
						s.solve_poisson(f=f_lap,disp=False)
						L2.append(s.operators['lap'].err)
						Linf.append(s.operators['lap'].Linf_err)
					if sys == 'helm':
						s.solve_helmholtz(f=f_helm,disp=False)
						L2.append(s.operators['helm'].err)
						Linf.append(s.operators['helm'].Linf_err)

					del	s

					if N>16:
						l2rate = L2[-2]/L2[-1]
						linfrate = Linf[-2]/Linf[-1]
						myord = ord if N==32 else ' '
						myrn = rname if (row==0 and myord==ord) else '\t'
						mydl = dofloc if (col==0 and myrn==rname) else '  '
						if myrn == 'no': myrn = 'uniform  '
						print('{}\t{}\t{}\t{}\t\t{}'.format(
							mydl,myrn,myord,round(l2rate,3),round(linfrate,3)))
						L2rs.append(l2rate)
						Linfrs.append(linfrate)
				l2_tmp[ord]= (L2,L2rs)
				linf_tmp[ord] = (Linf,Linfrs)
			l2rates[rname] = l2_tmp
			linfrates[rname] = linf_tmp
		all_l2_rates[dofloc] =	l2rates
		all_linf_rates[dofloc] = linfrates
	return all_l2_rates, all_linf_rates, d_ops

def plot_it(d_results,d_ops,ord_shift=1):

	Ns = np.array([16,32,64,128])
	labels = [None,r'$h^1$',r'$h^2$',r'$h^3$',r'$h^4$']
	keys = ['dofs','ords','refs']
	err_names = [r'$|u|_{L_\infty}$',r'$|u|_{L_2}$']
	add_ons = [('',' centered',None),('p = ','',None),('',' refinement',8)]
	lens = [len(d_ops[key]) for key in keys]
	order = list(np.argsort(lens))
	[dof_i,ord_i,ref_i] = [order.index(j) for j in range(3)]

	title_addons = add_ons[order[0]]
	col_addons = add_ons[order[1]]
	row_addons = add_ons[order[2]]

	for lowest_it,l_name in enumerate(d_ops[keys[order[0]]]):
		col_count = lens[order[1]]
		row_count = lens[order[2]]

		fig = plt.figure(figsize=(5*col_count,5*row_count))


		for row_id,r_name in enumerate(d_ops[keys[order[2]]]):
			for col_id,c_name in enumerate(d_ops[keys[order[1]]]):
				k_ops = [l_name,c_name,r_name]
				spot = col_count*row_id+col_id+1
				plt.subplot(row_count,col_count,spot)
				errs = d_results[k_ops[dof_i]][k_ops[ref_i]][k_ops[ord_i]][0]
				myNs = Ns[:len(errs)]

				N_order = k_ops[ord_i]+ord_shift
				plt.loglog(myNs,1/myNs**N_order,label=labels[N_order])
				plt.loglog(myNs,errs,label=err_names[ord_shift])
				plt.legend(fontsize=15)
				if col_id == 0:
					plt.ylabel(row_addons[0]+str(r_name)[:row_addons[-1]]+row_addons[1],
									fontsize=15)
				if row_id == 0:
					plt.title(col_addons[0]+str(c_name)[:col_addons[-1]]+col_addons[1],
									fontsize=15)

		plt.suptitle(title_addons[0]+str(l_name)[:title_addons[-1]]+title_addons[1],
									fontsize=15)
		plt.show()
