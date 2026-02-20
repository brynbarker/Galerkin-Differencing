import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
import scipy.sparse.linalg as sla
from general_solve.mesh import Mesh
from general_solve.integration import Integrator
from general_solve.simple_solve import LaplaceOperator,ProjectionOperator
from general_solve.constraints import ConstraintOperator

class Solver:
	def __init__(self,N,dim,dofloc,rtype,rname=None,u=None,ords=[3,3],qpn=5,dirichlet=False):
		self.N = N
		self.dim = dim
		self.ufunc = u
		self.ffunc = None #needs to be overwritten 
		self.qpn = qpn
		self.integrator = Integrator(qpn,dim,ords)
		self.ords = ords

		self.mesh = Mesh(N,dim,ords,dofloc,rtype,rname)
		self.h = self.mesh.h

		self.lap = LaplaceOperator(self.mesh,self.integrator,mu=1)
		self.mass = ProjectionOperator(self.mesh,self.integrator)
		self.constraints = ConstraintOperator(self.mesh,dirichlet=dirichlet)

		self.dirichlet = dirichlet
		self._build_zero_mean()

		self.U_true = None
		self.u_vals = None
	
	def _build_zero_mean(self):
		if self.dirichlet:
			self.constraints._setup_dirichlet(self.ufunc)
			return

		myGs = []
		for patch in self.mesh.patches:
			num_dofs = len(patch.dofs)
			G = np.zeros(num_dofs)

			for e in patch.elements.values():
				vol = (e.h)**self.dim
				for test_id,dof in enumerate(e.dof_list):
					phi_vals = self.integrator.phi_vals[test_id]
					for (quad,phi_val) in zip(e.quads,phi_vals):
						if quad:
							val = self.integrator._compute_product_integral(phi_val,volume=vol)
							G[dof.ID] += val
			myGs.append(G)

		Gd = list(np.hstack(myGs))
		size = len(Gd)
		Gr = list(np.arange(size))
		Gc = [0]*size
		self.spG = sparse.coo_array((Gd,(Gr,Gc)),shape=(size,1)).tocsc()

	def sol(self, interpolants=None):

		if interpolants is None:
			return

		def solution(loc):
			e,dof_shift = self.mesh.loc_to_el(loc)
			val = 0
			for local_id, dof in enumerate(e.dof_list):
				val += interpolants[dof.ID+dof_shift]*dof.phi(loc)
			return val
		return solution

	def error(self,U):
		if self.u_vals == None:
			tmp = [{},{}]
			for p_id,p in enumerate(self.mesh.patches):
				dof_shift = self.constraints.dof_id_shift*p_id
				for e in p.elements.values():
					vol = (e.h)**self.dim
					u_vals_e = self.integrator._evaluate_func_on_element(self.ufunc,e.bounds)
					tmp[p_id][e.ID] = u_vals_e
			self.u_vals = tmp
		l2_err = 0.
		for p_id,p in enumerate(self.mesh.patches):
			dof_shift = self.constraints.dof_id_shift*p_id
			for e in p.elements.values():
				vol = (e.h/2)**self.dim
				for q_id,q_bool in enumerate(e.quads):
					u_vals = self.u_vals[p_id][e.ID][q_id]
					if q_bool:
						uh_vals = 0
						for local_id, dof in enumerate(e.dof_list):
							phi_vals = self.integrator.phi_vals[local_id][q_id]
							uh_vals += U[dof.ID+dof_shift]*phi_vals
						l2_err += self.integrator._compute_error_integral(u_vals,uh_vals,vol)
		return np.sqrt(l2_err)

	def evaluate_on_grid(self,func):
		tmp = []
		for p in self.mesh.patches:
			for lookup_id in p.dofs:
				dof = p.dofs[lookup_id]
				if self.dim == 2:
					tmp.append(func(dof.x,dof.y))
				if self.dim == 3:
					tmp.append(func(dof.x,dof.y,dof.z))
		return np.array(tmp)


	def Linf_error(self,U):
		if self.U_true is None:
			tmp = []
			for p in self.mesh.patches:
				for lookup_id in p.dofs:
					dof = p.dofs[lookup_id]
					if self.dim == 2:
						tmp.append(self.ufunc(dof.x,dof.y))
					if self.dim == 3:
						tmp.append(self.ufunc(dof.x,dof.y,dof.z))
			self.U_true = np.array(tmp)
		Utmp = (U-self.U_true)[self.constraints.true_dofs]

		return np.linalg.norm(Utmp)


	def solve_simple(self,f,op,disp=True):
		op._build_force(f)
		op._build_system()

		C = self.constraints.spC
		if self.dirichlet:
			lhs = C.T @ op.spB @ C
			rhs = C.T.dot(op.F-op.spB.dot(self.constraints.dirichlet))
			solver = sla.cg
		else:
			sparse1 = sparse.coo_array(
				([1],([0],[0])),shape=(1,1)).tocsc()
			full_sys = sparse.bmat(np.array(
				[[op.spB,self.spG],
				 [self.spG.T,None]]),format='csc')
			padded_C = sparse.bmat(np.array(
				[[C,None],
				 [None,sparse1]]),format='csc')
			lhs = padded_C.T @ full_sys @ padded_C
			padded_F = np.hstack((op.F,np.array([0.])))
			rhs = padded_C.T.dot(padded_F)
			solver = sla.gmres

		x = np.linalg.solve(lhs.todense(),rhs)
		# try:
		# 	# x = np.linalg.solve(lhs.todense(),rhs)
		# 	if self.dirichlet:
		# 		x,conv = solver(lhs,rhs,rtol=1e-13)
		# 		assert conv == 0
		# 	else:
		# 		x = np.linalg.solve(lhs.todense(),rhs)
		# except:
		# 	try:
		# 		x = np.linalg.solve(lhs.todense(),rhs)
		# 	except:
		# 		print('Solver did not converge, check self.totest')
		# 		self.totest = [lhs,rhs]
		# 		return


		if self.dirichlet:
			U = C.dot(x) + self.constraints.dirichlet
		else:
			tmp = padded_C.dot(x)
			if disp:
				print('lambda = {}'.format(tmp[-1]))
			U = tmp[:-1]

		op.set_U(U)
		err = self.error(U)
		Linf_err = self.Linf_error(U)
		op.set_error(err)
		op.set_error(Linf_err,Linf=True)

		self.solve_system = [lhs,rhs,padded_C,U]

		if disp:
			print('L2 error     = {}'.format(err))
			print('Linf error   = {}'.format(Linf_err))

	def solve_poisson(self,f=None,disp=True):
		if self.ffunc is None:
			assert f is not None
		if f is None:
			f = self.ffunc
		self.solve_simple(f,self.lap,disp)

	def solve_projection(self,disp=True):
		self.solve_simple(self.ufunc,self.mass,disp)


	def vis_dof_sol(self,U):
		self.mesh.vis_dof_sol(U)


	#def solve(self):
	#	print('virtual not overwritten')

	#def vis_constraints(self):
	#	fig,ax = plt.subplots(1,figsize=(16,24))
	#	markers = np.array([['s','^'],['v','o']])
	#	cols = {-1/16:'C0',9/16:'C1',1:'C2',1/4:'C3',3/4:'C4'}
	#	flags = {-1/16:False,9/16:False,1:False,1/4:False,3/4:False}
	#	labs = {-1/16:'-1/16',9/16:'9/16',1:'1',1/4:'1/4',3/4:'3/4'}
	#	axshow = []
	#	for ind,b in enumerate(self.Id):
	#		if b:
	#			row = self.C_full[ind]
	#			dof = self.mesh.dofs[ind]
	#			x,y = dof.x,dof.y
	#			if dof.h==self.h:
	#				for cind,val in enumerate(row):
	#					if abs(val)>1e-12:
	#						cdof = self.mesh.dofs[cind]
	#						cx,cy = cdof.x,cdof.y
	#						if cdof.h!=dof.h or val==-1:
	#							if cx-x > .5: tx=cx-1
	#							elif x-cx>.5: tx=cx+1
	#							else: tx=cx
	#							if cy-y > .5: ty=cy-1
	#							elif y-cy>.5: ty=cy+1
	#							else: ty=cy

	#							if tx==x: tx-=self.h
	#							
	#							m = markers[int(ty==cy),int(tx==cx)]
	#							ax.scatter([x],[y],color='k',marker='o')
	#							ax.scatter([tx],[ty],color='k',marker=m)
	#							if flags[val]==False:
	#								ax.plot([x,tx],[y,ty],color=cols[val],label=labs[val])
	#								flags[val] = True
	#							else:
	#								ax.plot([x,tx],[y,ty],color=cols[val])
	#						else:
	#							assert val==1
	#						
	#						
	#				
	#				
	#	ax.legend()
	#	plt.show()
	#	return

	##def vis_constraints(self):
	##	if self.C is not None:
	##		vis_constraints(self.C,self.mesh.dofs)
	##	else:
	##		print('Constraints have not been set')
	#def vis_dof_sol(self,proj=False,err=False,fltr=False,fval=.9,dsp=False,myU=None,onlytrue=False):
	#	U = self.U
	#	if myU is not None: U=myU
	#	id0,x0,y0,c0 = [],[], [], []
	#	id1,x1,y1,c1 = [],[], [], []
	#	for dof in self.mesh.dofs.values():
	#		#if onlytrue and dof.ID in self.true_dofs:
	#		if dof.ID in self.true_dofs:
	#			if dof.h == self.h:
	#				id1.append(dof.ID)
	#				x1.append(dof.x)
	#				y1.append(dof.y)
	#				val = U[dof.ID]
	#				if err: val = abs(val-self.ufunc(dof.x,dof.y))
	#				c1.append(val)


	#			else:
	#				id0.append(dof.ID)
	#				x0.append(dof.x)
	#				y0.append(dof.y)
	#				val = U[dof.ID]
	#				if err: val = abs(val-self.ufunc(dof.x,dof.y))
	#				c0.append(val)
	#	
	#	m = ['o' for v in c1]+['^' for v in c0]
	#	
	#	if fltr:
	#		mx = max(c0)
	#		msk = np.array(c0)>fval*mx
	#		id0 = np.array(id0)[msk]
	#		x0 = np.array(x0)[msk]
	#		y0 = np.array(y0)[msk]
	#		c0 = np.array(c0)[msk]
	#		vals = np.array([x0,y0,c0]).T
	#		if dsp:print(vals)

	#		mx = max(c1)
	#		msk = np.array(c1)>fval*mx
	#		id1 = np.array(id1)[msk]
	#		x1 = np.array(x1)[msk]
	#		y1 = np.array(y1)[msk]
	#		c1 = np.array(c1)[msk]
	#		vals = np.array([x1,y1,c1]).T
	#		if dsp:print(vals)
	#		


	#	fig,ax = plt.subplots(1,2,figsize=(10,5))
	#	plot1 = ax[0].scatter(x0,y0,marker='^',c=c0,cmap='jet')
	#	fig.colorbar(plot1,location='left')
	#	#ax[0].set_xlim(-1.5*self.h,.5+1.5*self.h)
	#	#ax[0].set_ylim(-1.5*self.h,1+1.5*self.h)
	#	#plt.show()

	#	plot2 = ax[1].scatter(x1,y1,marker='o',c=c1,cmap='jet')
	#	fig.colorbar(plot2,location='left')
	#	#ax[1].set_xlim(-1.5*self.h,1+1.5*self.h)
	#	#ax[1].set_ylim(-1.5*self.h,1+1.5*self.h)
	#	plt.show()

	#	if fltr and dsp: return id0,id1

	#def vis_mesh(self,corner=False,retfig=False):

	#	fig = plt.figure()
	#	mk = ['^','o']
	#	for ind,dof in enumerate(self.mesh.dofs.values()):
	#		m = mk[dof.h==self.mesh.h]
	#		c = 'C0' if ind in self.true_dofs else 'C1'
	#		cind = 2*(ind in self.true_dofs)+((ind in self.true_dofs)==(dof.h==self.mesh.h))
	#		c = 'C'+str(cind)
	#		alpha = 1 if ind in self.true_dofs else .5
	#		plt.scatter(dof.x,dof.y,marker=m,color=c,alpha=alpha)

	#	plt.show()
	#	return

	#def vis_dofs(self):
	#	frame = [[.5,1,1,0,0,.5,.5],[0,0,1,1,0,0,1]]
	#	fig,ax = plt.subplots(figsize=(10,10))
	#	ax.set_xlim(-3*self.h,1+3*self.h)
	#	ax.set_ylim(-3*self.h,1+3*self.h)

	#	size = 16#int((self.p+1)**2)
	
	#	line, = ax.plot(frame[0],frame[1],'lightgrey')
	#	blocks = []
	#	for _ in range(size):
	#		block, = ax.plot([],[])
	#		blocks.append(block)
	#	dot, = ax.plot([],[],'ko',linestyle='None')

	#	def update(n):
	#		dof = self.mesh.dofs[n]
	#		els = list(dof.elements.values())
	#		line.set_data(frame[0],frame[1])
	#		dot.set_data(dof.x,dof.y)
	#		for i in range(size):
	#			if i < len(dof.elements):
	#				e = els[i]
	#				blocks[i].set_data(e.plot[0],e.plot[1])
	#			else:
	#				blocks[i].set_data([],[])
	#		return [line,blocks,dot]
	#	interval = 400
	#	ani = FuncAnimation(fig, update, frames=len(self.mesh.dofs), interval=interval)
	#	plt.close()
	#	return HTML(ani.to_html5_video())
	#	frame = [[.5,1,1,0,0,.5,.5],[0,0,1,1,0,0,1]]
	#	data = []
	#	for dof in self.mesh.dofs.values():
	#		blocks = []
	#		dots = [[dof.x],[dof.y]]
	#		for e in dof.elements.values():
	#			blocks.append(e.plot)
	#		data.append([blocks,dots])

	#	return animate_2d([frame],data,16)

	#def vis_elements(self):
	#	frame = [[.5,1,1,0,0,.5,.5],[0,0,1,1,0,0,1]]
	#	fig,ax = plt.subplots(figsize=(10,10))
	#	ax.set_xlim(-3*self.h,1+3*self.h)
	#	ax.set_ylim(-3*self.h,1+3*self.h)
	
	#	line, = ax.plot(frame[0],frame[1],'lightgrey')
	#	eline, = ax.plot([],[])
	#	dot, = ax.plot([],[],'ko',linestyle='None')

	#	def update(n):
	#		e = self.mesh.elements[n]
	#		line.set_data(frame[0],frame[1])
	#		eline.set_data(e.plot[0],e.plot[1])
	#		xs,ys = [],[]
	#		for dof in e.dof_list:
	#			xs.append(dof.x)
	#			ys.append(dof.y)
	#		dot.set_data(xs,ys)
	#		return [line,eline,dot]
	#		#	if i < len(e.dof_list):
	#		#		blocks[i].set_data(e.dof_list[i].x,e.dof_list.y
	#		#		blocks[i].set_data(blocks_n[i][0],blocks_n[i][1])
	#		#	else:
	#		#		blocks[i].set_data([],[])
	#		#if yesdot: dot.set_data(dots_n[0],dots_n[1])
	#		#to_return = [line]+blocks
	#		#if yesdot: to_return += [dot]
	#		#return to_return
	#	interval = 400
	#	ani = FuncAnimation(fig, update, frames=len(self.mesh.elements), interval=interval)
	#	plt.close()
	#	return HTML(ani.to_html5_video())
	#	frame = [[.5,1,1,0,0,.5,.5],[0,0,1,1,0,0,1]]
	#	data = []
	#	for e in self.mesh.elements:
	#		center = [(e.dom[1]+e.dom[0])/2,(e.dom[-1]+e.dom[-2])/2]
	#		plt.plot(e.plot[0],e.plot[1],'grey')
	#		plt.plot(center[0],center[1],'k.')


	#def xy_to_e(self,x,y):
	#	n_x_els = [self.N/2,self.N]
 #       
	#	x -= (x==1)*1e-12
	#	y -= (y==1)*1e-12
	#	fine = True if x >= 0.5 else False
	#	x_ind = int((x-fine*.5)/((2-fine)*self.h))
	#	y_ind = int(y/((2-fine)*self.h))
	#	el_ind = fine*self.mesh.n_coarse_els+y_ind*n_x_els[fine]+x_ind
	#	e = self.mesh.elements[int(el_ind)]
	#	assert x >= min(e.plot[0]) and x <= max(e.plot[0])
	#	assert y >= min(e.plot[1]) and y <= max(e.plot[1])
	#	return e

	#def sol(self, weights=None):

	#	if weights is None:
	#		assert self.solved
	#		weights = self.U

	#	def solution(x,y):
	#		e = self.xy_to_e(x,y)

	#		id_to_ind = {ID:[int(ID/4),ID%4] for ID in range(16)}
	#		val = 0
	#		for local_id, dof in enumerate(e.dof_list):
	#			local_ind = id_to_ind[local_id]
	#			val += weights[dof.ID]*phi3_2d_eval(x,y,dof.h,dof.x,dof.y)
	#		
	#		return val
	#	return solution

	#def error(self,qpn=5):
	#	uh = self.sol()
	#	l2_err = 0.
	#	for e in self.mesh.elements:
	#		func = lambda x,y: (self.ufunc(x,y)-uh(x,y))**2
	#		val = gauss(func,e.x,e.x+e.h,e.y,e.y+e.h,qpn)
	#		l2_err += val
	#	return np.sqrt(l2_err)
#
#class Laplace(Solver):
#	def __init__(self,N,u,f,qpn=5):
#		super().__init__(N,u,f,qpn)
#		self._setup_constraints()
#
#	def solve(self):
#		self._build_stiffness()
#		self._build_force()
#		LHS = self.C.T @ self.K @ self.C
#		RHS = self.C.T @ (self.F )#- self.K @ self.dirichlet)
#		#LHS = self.C_rect.T @ self.K @ self.C_rect
#		#RHS = self.C_rect.T @ (self.F - self.K @ self.dirichlet)
#		x = la.solve(LHS,RHS)
#		self.U = self.C @ x #+ self.dirichlet
#		self.solved = True
#
#
#class Projection(Solver):
#	def __init__(self,N,u,qpn=5):
#		super().__init__(N,u,u,qpn)
#
#	def solve(self):
#		self._build_mass()
#		self._build_force()
#		self._setup_constraints()
#		LHS = self.C.T @ self.M @ self.C
#		RHS = self.C.T @ (self.F - self.M @ self.dirichlet)
#		#LHS = self.C_rect.T @ self.M @ self.C_rect
#		#RHS = self.C_rect.T @ (self.F - self.M @ self.dirichlet)
#		x = la.solve(LHS,RHS)
#		self.U = self.C @ x + self.dirichlet
#		self.solved = True
#		return x
#

