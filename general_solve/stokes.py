import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg as scla
import scipy.sparse.linalg as sla
from general_solve.element import PseudoElement
from general_solve.differential_operators import *
from general_solve.variable import SingleComponentVariable, MultiComponentVariable
from general_solve.shape_functions import *

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

class Pressure(SingleComponentVariable):
	def __init__(self,N,dim=2,dofloc='cell',
			  rtype='uniform',rname=None,var=None,
			  ords=[1,1],qpn=3):
		super().__init__(N,dim,dofloc,rtype,rname,var,ords,qpn)

	def compute_divergence(self,l_dphivals,test_size,el_map):
		self.div_op = self.setup_divergence(
							  l_dphivals,el_map,test_size)
		# self.div_op = DivergenceOperator(self.mesh,self.integrator,
								#    l_dphivals,el_map,test_size)
		# self.div_op._build_system()

	def compute_errors(self,P):
		p_L2 = self.error(P)
		p_Linf = self.Linf_error(P)

		self.L2_err = p_L2
		self.Linf_err = p_Linf

		return p_L2,p_Linf


class Velocity(MultiComponentVariable):
	def __init__(self, N, dim=2, doflocs=['xside','yside'], 
			  rtype='uniform', rname=None, 
			  vars=[None,None], ords=[1,1], qpn=3,mu=1):
		super().__init__(N, dim, doflocs, 
				   rtype, rname, vars, ords, qpn)

		self.spC = sparse.bmat(np.array(
			[[self.u.constraints.spC,None],
			 [None,self.v.constraints.spC]]),
			 format='csc')

		self.lap_u_op = self.u.setup_laplace()
		self.lap_v_op = self.v.setup_laplace()

	def compute_errors(self,U,V):
		u_L2 = self.u.error(U,raw=True)
		v_L2 = self.v.error(V,raw=True)
		u_Linf = self.u.Linf_error(U,raw=True)
		v_Linf = self.v.Linf_error(V,raw=True)

		joined_uv_Linf = np.hstack([u_Linf,v_Linf])

		self.L2_err = np.sqrt(u_L2+v_L2)
		self.Linf_err = np.linalg.norm(joined_uv_Linf)

		return [np.sqrt(u_L2),np.sqrt(v_L2)],[np.linalg.norm(u_Linf),np.linalg.norm(v_Linf)]


class StokesFlow:
	def __init__(self,N,dim=2,rtype='uniform',
				 rname=None,vars=[None,None,None],
				 ords=[1,1,1],qpn=3,mu=1):
		self.N = N
		self.dim = dim
		self.pressure = Pressure(N,dim,qpn=qpn,rtype=rtype,
						   rname=rname,var=vars[-1],
						   ords=[ords[-1],ords[-1]])
		self.velocity = Velocity(N,dim,qpn=qpn,rtype=rtype,
						   rname=rname,vars=vars[:-1],
						   ords=ords[:-1],mu=mu)

		self._setup_coupling()
		self._setup_operators()
		self._setup_system()
		self.F = None

	def _build_force(self,forces):
		if self.F is not None:
			return 

		myFs = []
		vars = [self.velocity.u,self.velocity.v]
		for var,ffunc in zip(vars,forces):
			for patch in var.mesh.patches:
				num_dofs = len(patch.dofs)
				F = np.zeros(num_dofs)

				for e in patch.elements.values():
					vol = (e.h/2)**self.dim
					fvals = var.integrator._evaluate_func_on_element(ffunc,e.bounds)
					for q_id,(quad,f_val) in enumerate(zip(e.quads,fvals)):
						if quad:
							for test_id,dof in enumerate(e.dof_list):
								phi_val = var.integrator.phi_vals[q_id][test_id]
								val = var.integrator._compute_product_integral(phi_val,f_val,vol)
								F[dof.ID] += val
				myFs.append(F)

		# add rhs for second line of equation (0)
		myFs.append(np.zeros(self.sizes[-1]))
		self.F = np.hstack(myFs)

	def _setup_coupling(self):
		usize = [len(p.dofs) for p in self.velocity.u.mesh.patches]
		vsize = [len(p.dofs) for p in self.velocity.v.mesh.patches]
		psize = [len(p.dofs) for p in self.pressure.mesh.patches]
		self.sizes = [sum(usize),sum(vsize),sum(psize)]
		self.full_sizes = [usize,vsize,psize]
		self.test_sizes = psize

		self.C_sizes = [len(self.velocity.u.constraints.true_dofs),
				  		len(self.velocity.v.constraints.true_dofs),
						len(self.pressure.constraints.true_dofs)]

		shift_vecs = np.eye(self.dim)
		d_element_map = {}
		for p_id,patch in enumerate(self.pressure.mesh.patches):
			u_patch = self.velocity.u.mesh.patches[p_id]
			v_patch = self.velocity.v.mesh.patches[p_id]
			for e_pr in patch.elements.values():
				new_el = PseudoElement()
				for shift_dim,vel_patch in enumerate([u_patch,v_patch]):
					shift_vec = shift_vecs[shift_dim]*patch.h/2
					for ind,scale in enumerate([-1,1]):
						new_loc = e_pr.loc + scale*shift_vec+1e-6
						# if not (0<= new_loc[shift_dim]<=1):
						try:
							e_vel = vel_patch._get_element_from_loc(new_loc)
							new_el.add_dof_ids(shift_dim,ind,e_vel.dof_ids)
						except:
							new_el.add_dof_ids(shift_dim,ind)
				d_element_map[e_pr] = new_el

		l_dphi_vals = [self.velocity.u.integrator.dphi_vals,
					   self.velocity.v.integrator.dphi_vals]
		self.pressure.compute_divergence(l_dphi_vals,
								   self.velocity.u.integrator.prod,
								   d_element_map)

		self.divergence = self.pressure.div_op.spA


	def _split_vec(self,vec):
		if vec.size == sum(self.sizes):
			uv_split = self.sizes[-1]
			vp_split = sum(self.sizes[:-1])
		else:
			uv_split = self.C_sizes[-1]
			vp_split = sum(self.C_sizes[:-1])

		U = vec[:uv_split]
		V = vec[uv_split:vp_split]
		P = vec[vp_split:]
		return [U,V,P]


	def _merge_vec(self,vecs):
		if vecs[0].size in self.sizes:
			vec = np.zeros(sum(self.sizes))
			uv_split = self.sizes[-1]
			vp_split = sum(self.sizes[:-1])
		else:
			vec = np.zeros(sum(self.C_sizes))
			uv_split = self.C_sizes[-1]
			vp_split = sum(self.C_sizes[:-1])

		vec[:uv_split] = vecs[0].flatten()
		vec[uv_split:vp_split] = vecs[1].flatten()
		vec[vp_split:] = vecs[2].flatten()
		return vec

	def _setup_operators(self):
		self.laplace = sparse.bmat(np.array(
			[[self.velocity.lap_u_op.spA, None],
			 [None, self.velocity.lap_v_op.spA]]),
			 format='csc')

	def _setup_system(self):
		# build constraint matrix
		self.C = sparse.bmat(np.array(
			[[self.velocity.spC,None],
			 [None,self.pressure.constraints.spC]]),
			format='csc')

		self.A = sparse.bmat(np.array(
			[[self.laplace,self.divergence.T],
			 [self.divergence, None]]),
			 format='csc')

		self.sys = self.C.T @ self.A @ self.C

		zlist = [self.velocity.u.Z.T,
			  	 self.velocity.v.Z.T,
				 self.pressure.Z.T]
		Z = np.hstack(zlist).T
		self.mean_values = [sum(z.flatten()) for z in zlist]
		self.zTc = self.C.T.dot(Z)
		return

		# need to take care of the nullspace

	def solve(self,forces,disp=True):
		self._build_force(forces)

		rhs = self.C.T.dot(self.F)
		l_rhs = self._split_vec(rhs)
		for i in range(3):
			f_proj = sum(l_rhs[i])/l_rhs[i].size

			# check for nonzero projections into nullspace
			if abs(f_proj) > 1e-12: 
				l_rhs[i] = l_rhs[i] - f_proj

		rhs = self._merge_vec(l_rhs)

		# x_star, conv = sla.gmres(self.sys,rhs,rtol=1e-13)
		# assert conv==0
		x_star = np.linalg.solve(self.sys.todense(),rhs)

		l_x = self._split_vec(x_star)
		l_zTc = self._split_vec(self.zTc)
		for i in range(3):
			alpha = (l_zTc[i].T@l_x[i]) / self.mean_values[i]
			l_x[i] = l_x[i] - alpha
		x = self._merge_vec(l_x)

		sol_vec = self.C.dot(x)
		[U,V,P] = self._split_vec(sol_vec)
		self.sol_vec=sol_vec

		vel_L2_errs, vel_Linf_errs = self.velocity.compute_errors(U,V)
		pr_L2, pr_Linf = self.pressure.compute_errors(P)
		

		if disp:
			print('L2 Errors:\tVelocity: {}\tPressure: {}'.format(round(self.velocity.L2_err,5),round(pr_L2,5)))
			print('Linf Errors:\tVelocity: {}\tPressure: {}'.format(round(self.velocity.Linf_err,5),round(pr_Linf,5)))
			self.view_sol([U,V,P],err=True)
		
		

	def view_sol(self,vecs,err=False,true_list=None):

		self.velocity.u.vis_dof_sol(vecs[0],err,true_list)
		self.velocity.v.vis_dof_sol(vecs[1],err,true_list)
		self.pressure.vis_dof_sol(vecs[2],err,true_list)

	def get_dof(self,full_system_index):
		vec_length = sum(self.sizes[:-1])
		if full_system_index < self.sizes[0]:
			comp = 0
			dof = self.velocity.u.constraints.get_dof(full_system_index)
		elif full_system_index < vec_length:
			comp = 1
			dof = self.velocity.v.constraints.get_dof(
				full_system_index-self.sizes[0])
		else:
			comp = 2
			dof = self.pressure.constraints.get_dof(
				full_system_index-vec_length)
		return dof,comp

	def test_laplace(self,u=None,f=None):
		if u is None or f is None:
			u =	lambda x,y:	x**4+y**4
			f =	lambda x,y:	12*x**2+12*y**2

		err0 = self.velocity.u.solve_laplace_truncation(u,f)
		err1 = self.velocity.v.solve_laplace_truncation(u,f)
		return (err0,err1)

	def test_divergence(self,uv=None,f=None):
		if uv is None or f is None:
			u = lambda x,y: np.sin(2*np.pi*y)+x**3
			v = lambda x,y: np.cos(2*np.pi*x)+y**3
			f0 = lambda x,y: 3*x**2
			f1 = lambda x,y: 3*y**2
		else:
			u,v = uv
			f0,f1 = f
		errs = self.pressure.solve_div_truncation(
				[self.velocity.u,self.velocity.v],[u,v],[f0,f1])
		return errs

	def test_operators(self):
		def res(errs):
			if errs[0]<1e-12 and errs[1]<1e-12:
				return 'Passed'
			return 'Failed'
		test_names = ['Laplace','Divergence']
		tests = [self.test_laplace,self.test_divergence]
		for name,test in zip(test_names,tests):
			print('{} Test\t:   {}'.format(name,res(test())))

	def check_schur_null(self):
		
		dense_div, dense_lap = self.divergence.todense(), self.laplace.todense()
		
		schur = dense_div @ scla.inv(dense_lap) @ dense_div.T
		ns_schur = scla.null_space(schur)
		n,m = ns_schur.shape
		for vec in range(m):
			self.pressure.vis_dof_sol(ns_schur[:,vec])
		return schur
