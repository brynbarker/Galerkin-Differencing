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
	all_l1_rates =	{}
	all_linf_rates =	{}
	Nvals =	np.array([8,16,32])#,64])

	print('dofloc\trtype\t\torder\tL2 rates\tL1 rates\tLinf rates\n'+'-'*80)
	for	dofloc in dofloc_ops:
		l2rates	= {}
		l1rates	= {}
		linfrates =	{}
		for	col,rname in enumerate(rname_ops):
			l2_tmp = {}
			l1_tmp = {}
			linf_tmp = {}
			for	row,ord	in enumerate(ord_ops):
				L2,L1,Linf	= [],[],[]
				L2rs,L1rs,Linfrs	= [],[],[]

				for	N in Nvals:
					# print(N,dofloc,rtype,rname,ord)
					s =	Var(N,2,dofloc,rtype,
							   rname=rname,var=u,ords=[ord,ord])

					if sys == 'lap':
						s.solve_poisson(f=f_lap,disp=False)
						L2.append(s.operators["lap"].L2)
						L1.append(s.operators["lap"].L1)
						Linf.append(s.operators["lap"].Linf)
					if sys == 'helm':
						s.solve_helmholtz(f=f_helm,disp=False)
						L2.append(s.operators["helm"].L2)
						L1.append(s.operators["helm"].L1)
						Linf.append(s.operators["helm"].Linf)

					del	s

					if N>Nvals[0]:
						l2rate = L2[-2]/L2[-1]
						l1rate = L1[-2]/L1[-1]
						linfrate = Linf[-2]/Linf[-1]
						myord = ord if N==Nvals[1] else ' '
						myrn = rname if (row==0 and myord==ord) else '\t'
						mydl = dofloc if (col==0 and myrn==rname) else '  '
						if myrn == 'no': myrn = 'uniform  '
						print('{}\t{}\t{}\t{}\t\t{}\t\t{}'.format(
							mydl,myrn,myord,round(l2rate,3),round(l1rate,3),round(linfrate,3)))
						L2rs.append(l2rate)
						L1rs.append(l1rate)
						Linfrs.append(linfrate)
				l2_tmp[ord]= (L2,L2rs)
				l1_tmp[ord]= (L1,L1rs)
				linf_tmp[ord] = (Linf,Linfrs)
			l2rates[rname] = l2_tmp
			l1rates[rname] = l1_tmp
			linfrates[rname] = linf_tmp
		all_l2_rates[dofloc] =	l2rates
		all_l1_rates[dofloc] =	l1rates
		all_linf_rates[dofloc] = linfrates
	return [all_linf_rates, all_l1_rates, all_l2_rates], d_ops

def plot_it(d_results,d_ops,filename):

	Ns = np.array([16,32,64,128])
	labels = [None,r'$h^1$',r'$h^2$',r'$h^3$',r'$h^4$']
	keys = ['dofs','ords','refs']
	err_names = [r'$|u|_{L_\infty}$',r'$|u|_{L_1}$',r'$|u|_{L_2}$']
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
				for j,(d_res,err_name) in enumerate(zip(d_results,err_names)):
					errs = d_res[k_ops[dof_i]][k_ops[ref_i]][k_ops[ord_i]][0]
					myNs = Ns[:len(errs)]
					if j == 2:
						plt.loglog(myNs,errs,label=err_name,lw=5)
					else:
						plt.loglog(myNs,errs,label=err_name)

				N_order = k_ops[ord_i]
				plt.loglog(myNs,1/myNs**N_order,label=labels[N_order])
				plt.loglog(myNs,1/myNs**(N_order+1),label=labels[N_order+1],lw=5)
				plt.legend(fontsize=15)
				if col_id == 0:
					plt.ylabel(row_addons[0]+str(r_name)[:row_addons[-1]]+row_addons[1],
									fontsize=15)
				if row_id == 0:
					plt.title(col_addons[0]+str(c_name)[:col_addons[-1]]+col_addons[1],
									fontsize=15)

		plt.suptitle(title_addons[0]+str(l_name)[:title_addons[-1]]+title_addons[1],
									fontsize=15)
		if filename is not None:
			plt.savefig(filename,dpi=300)
		else:
			plt.show()

name_map = {'vertfine': 'Vertical Fine Center',
			'vertcoarse': 'Vertical Coarse Center',
			'horzfine': 'Horizontal Fine Center',
			'horzcoarse': 'Horizontal Coarse Center','uniform':'Uniform',
			'fine': 'Fine Center','cell':'Cell','xside':'X-Side','yside': 'Y-Side',
			'coarse': 'Coarse Center','stripe':'Striped','square':'Square'}
label_map = {1:r'$h^1$',2:r'$h^2$'}
Ns = np.array([4,8,16])
def check_convergence(dofloc,rtype,rname,ords,funcs,proj=False,helm=False):
	ufunc,ffunc = funcs
	errs,L1_errs,Linf_errs = [],[],[]
	for N in Ns:
		var = Var(N,dofloc=dofloc,rtype=rtype,rname=rname,ords=ords,var=ufunc)
		if proj:
			var.solve_projection(disp=False)
		elif helm:
			var.solve_helmholtz(ffunc,disp=False)
		else:
			var.solve_poisson(ffunc,disp=False)
		l2,l1,linf = var.curr_errs
		errs.append(l2)
		L1_errs.append(l1)
		Linf_errs.append(linf)

	prob = 'Projection' if proj else 'Poisson'
	prob = 'Helmholtz' if helm else prob 
	subtype = prob if rtype=='uniform' else name_map[rname[:-6]]
	titles = [prob,subtype]
	# title_str = '{} p{}{} {} {} convergence'.format(subtype+rtype,
											    #   ords[0],ords[1],dofloc,prob)
	return titles,errs,L1_errs,Linf_errs

def display_convergence(rtype_outputs,bigtitle):
	prob_count = len(rtype_outputs[0])
	r_count = len(rtype_outputs)

	if r_count == 1:
		fig,ax = plt.subplots(1,prob_count,figsize=(5*prob_count,5*r_count))
	else:
		fig,ax = plt.subplots(prob_count,r_count,figsize=(5*r_count,5*prob_count))
	for i in range(prob_count):
		for j in range(r_count):
			if prob_count == 1 and r_count == 1:
				axi = ax
			elif prob_count == 1:
				axi = ax[j]
			elif r_count == 1:
				axi = ax[i]
			else:
				axi = ax[i,j]
			titles, errs, L1_errs, Linf_errs = rtype_outputs[j][i]
			rates = np.mean([errs[0]/errs[1],errs[1]/errs[2]])
			l1_rates = np.mean([L1_errs[0]/L1_errs[1],L1_errs[1]/L1_errs[2]])
			linf_rates = np.mean([Linf_errs[0]/Linf_errs[1],Linf_errs[1]/Linf_errs[2]])
			conv_level = int(np.log(round(rates,0))/np.log(2))

			if i == 0 or r_count == 1: axi.set_title(titles[-1])
			if j == 0 and r_count > 1: axi.set_ylabel(titles[0])
			if i == prob_count-1 or r_count==1: axi.set_xlabel('N')
			axi.loglog(Ns,errs,label=r'$L_2$')
			axi.loglog(Ns,Linf_errs,':',label=r'$L_\infty$')
			axi.loglog(Ns,1/Ns**conv_level,label=label_map[conv_level])
			axi.loglog(Ns,L1_errs,':',label=r'$L_1$')

			axi.annotate('slope = {}'.format(round(rates,3)),(np.log(Ns[1]),np.log(errs[1])))
			axi.annotate('slope = {}'.format(round(linf_rates,3)),(np.log(Ns[1]),np.log(Linf_errs[1])))
			axi.annotate('slope = {}'.format(round(l1_rates,3)),(np.log(Ns[1]),np.log(L1_errs[1])))
			axi.legend()
	bigtitle = 'Convergence for {}-Centered Data with P = {} Shape Functions and {} Refinement'.format(
		name_map[bigtitle[0]],bigtitle[1],name_map[bigtitle[2]]
	)
	plt.suptitle(bigtitle)
	plt.show()

u_x_only = lambda x,y: u0(x)
f_x_only = lambda x,y: -4*np.pi**2*u_x_only(x,y)

u_y_only = lambda x,y: u0(y)
f_y_only = lambda x,y: -4*np.pi**2*u_y_only(x,y)

fx_helm = lambda x,y: f_x_only(x,y)+u_x_only(x,y)
fy_helm = lambda x,y: f_y_only(x,y)+u_y_only(x,y)
d_ords = {'cell':[0,0],'xside':[1,0],'yside':[0,1]}
d_funcs = {'xside':[u_x_only,f_x_only],
		   'yside':[u_y_only,f_y_only]}
d_funcs_h = {'xside':[u_x_only,fx_helm],
		   'yside':[u_y_only,fy_helm]}

def run_all(dofloc,low_ord=0):
	ords = d_ords[dofloc]
	ords = [ord+low_ord for ord in ords]

	if low_ord == 0:
		funcs = d_funcs[dofloc]
		funcs_h = d_funcs_h[dofloc]
	else:
		funcs = [u,f_lap]
		funcs_h = [u,f_helm]
	for rtype in rtypes:
		rtype_outputs = []
		for rname in rnames[rtype]:
			problems = []
			problems.append(check_convergence(dofloc,rtype,rname,ords,[u,u],True))
			if max(ords) > 0:
				if rtype != 'square':
					problems.append(check_convergence(dofloc,rtype,rname,ords,funcs,False))
				problems.append(check_convergence(dofloc,rtype,rname,ords,funcs_h,helm=True))
			rtype_outputs.append(problems)
		display_convergence(rtype_outputs,[dofloc,ords,rtype])