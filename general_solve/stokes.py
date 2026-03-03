import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg as scla
import scipy.sparse.linalg as sla
from general_solve.element import PseudoElement, PseudoIntegrator, PseudoElementVel
from general_solve.differential_operators import *
from general_solve.variable import SingleComponentVariable, MultiComponentVariable
from general_solve.mesh import PseudoMesh
from general_solve.shape_functions import *
from time import time

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
		self.div_op = DivergenceOperator(self.mesh,self.integrator,
								   l_dphivals,el_map,test_size)
		self.div_op._build_system()

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

		self.lap_u_op = LaplaceOperator(
			self.u.mesh,self.u.integrator,mu=mu)
		self.lap_u_op._build_system()

		self.lap_v_op = LaplaceOperator(
			self.v.mesh,self.v.integrator,mu=mu)
		self.lap_v_op._build_system()

		self.vec = [self.u,self.v]

	def compute_divergence(self,test_integrators,el_map,test_sizes):

		self.du_dx_op = DerivativeOperator(
			self.u.mesh,self.u.integrator,
			test_integrators[0], el_map, test_sizes,0)
		self.du_dx_op._build_system()

		self.dv_dy_op = DerivativeOperator(
			self.v.mesh,self.v.integrator,
			test_integrators[1], el_map, test_sizes,1)
		self.dv_dy_op._build_system()

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

		self._setup_coupling(N,rtype,rname)
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
					for test_id,dof in enumerate(e.dof_list):
						phi_vals = var.integrator.phi_vals[test_id]
						fvals = var.integrator._evaluate_func_on_element(ffunc,e.bounds)
						for (quad,phi_val,f_val) in zip(e.quads,phi_vals,fvals):
							if quad:
								val = var.integrator._compute_product_integral(phi_val,f_val,vol)
								F[dof.ID] += val
				myFs.append(F)

		# add rhs for second line of equation (0)
		myFs.append(np.zeros(self.sizes[-1]))
		self.F = np.hstack(myFs)

	def _setup_coupling(self,N,rtype,rname):
		usize = [len(p.dofs) for p in self.velocity.u.mesh.patches]
		vsize = [len(p.dofs) for p in self.velocity.v.mesh.patches]
		psize = [len(p.dofs) for p in self.pressure.mesh.patches]
		self.sizes = [sum(usize),sum(vsize),sum(psize)]
		self.full_sizes = [usize,vsize,psize]
		self.test_sizes = psize

		self.C_sizes = [len(self.velocity.u.constraints.true_dofs),
				  		len(self.velocity.v.constraints.true_dofs),
						len(self.pressure.constraints.true_dofs)]
		
		start = time()
		quadrant_finder = PseudoMesh(N,rtype=rtype,rname=rname)
		self.coarse_quads = quadrant_finder.coarse_quads
		self.fine_quads = quadrant_finder.fine_quads
		self.div_lookup = self.pressure.integrator.get_div_vals(
			[self.velocity.u.integrator.dphi_vals,
			self.velocity.v.integrator.dphi_vals],
			self.velocity.u.integrator.prod)

		lookup_pux = self.div_lookup[0]
		lookup_pvy = self.div_lookup[1]
		dx_blocks,dy_blocks = [],[]
		for level,quad_level in enumerate([self.coarse_quads,self.fine_quads]):
			dx_r, dx_c, dx_d = [],[],[]
			dy_r, dy_c, dy_d = [],[],[]
			for quad_set in quad_level:
				for quad_id in quad_set:
					loc = quad_set[quad_id]
					p_el,pl = self.pressure.mesh.loc_to_el(loc)
					u_el,ul = self.velocity.u.mesh.loc_to_el(loc)
					v_el,vl = self.velocity.v.mesh.loc_to_el(loc)
					for trial_id,trial_dof in enumerate(p_el.dof_list):
						mylen = len(u_el.dof_ids)
						dx_r += [trial_dof.ID]*mylen
						dy_r += [trial_dof.ID]*mylen

						dx_c += u_el.dof_ids
						dy_c += v_el.dof_ids

						dx_d += list(lookup_pux[quad_id][trial_id])
						dy_d += list(lookup_pvy[quad_id][trial_id])

			sp_dx = sparse.coo_array((dx_d,(dx_r,dx_c)),
							shape=(self.full_sizes[-1][level],self.full_sizes[0][level]))
			sp_dy = sparse.coo_array((dy_d,(dy_r,dy_c)),
							shape=(self.full_sizes[-1][level],self.full_sizes[1][level]))
			dx_blocks.append(sp_dx)
			dy_blocks.append(sp_dy)

		self.dx = sparse.bmat(np.array(
				[[dx_blocks[0],None],
				[None,dx_blocks[1]]]),
				format='csc')
		self.dy = sparse.bmat(np.array(
				[[dy_blocks[0],None],
				[None,dy_blocks[1]]]),
				format='csc')
		self.divergence3 = sparse.bmat(np.array(
				[[dx_blocks[0],None,dy_blocks[0],None],
				[None,dx_blocks[1],None,dy_blocks[1]]]),
				format='csc')
		print('this divergence time: {}'.format(time()-start))


		shift_vecs = np.eye(self.dim)
		self.d_element_map = {}
		for dim,velocity_component in enumerate(self.velocity.vec):
			for p_id,patch in enumerate(velocity_component.mesh.patches):
				pressure_patch = self.pressure.mesh.patches[p_id]
				shift_vec = shift_vecs[dim]*patch.h/2
				for e_vel in patch.elements.values():
					pL_loc = e_vel.loc - shift_vec
					pR_loc = e_vel.loc + shift_vec

					e_pL = pressure_patch._get_element_from_loc(pL_loc)
					e_pR = pressure_patch._get_element_from_loc(pR_loc)

					new_el = PseudoElement(dim)
					new_el.add_dof_list(e_pL.dof_list,0)
					new_el.add_dof_list(e_pR.dof_list,1)
					self.d_element_map[e_vel] = new_el
		self.element_map = lambda e: self.d_element_map[e]

		start = time()
		p_element_map = {}
		for p_id,patch in enumerate(self.pressure.mesh.patches):
			u_patch = self.velocity.u.mesh.patches[p_id]
			v_patch = self.velocity.v.mesh.patches[p_id]
			for e_pr in patch.elements.values():
				new_el = PseudoElementVel()
				for shift_dim,vel_patch in enumerate([u_patch,v_patch]):
					shift_vec = shift_vecs[shift_dim]*patch.h/2
					for ind,scale in enumerate([-1,1]):
						new_loc = e_pr.loc + scale*shift_vec+1e-6
						if not (0<= new_loc[shift_dim]<=1):
							new_el.add_dof_ids(shift_dim,ind)
						else:
							e_vel = vel_patch._get_element_from_loc(new_loc)
							new_el.add_dof_ids(shift_dim,ind,e_vel.dof_ids)
				p_element_map[e_pr] = new_el

		l_dphi_vals = [self.velocity.u.integrator.dphi_vals,
					   self.velocity.v.integrator.dphi_vals]
		self.pressure.compute_divergence(l_dphi_vals,
								   self.velocity.u.integrator.prod,p_element_map)

		self.divergence4 = self.pressure.div_op.spA
		print('pressure div comp = {}'.format(time()-start))

		self.d_test_integrator = {}
		for shift_dim in range(2):
			test_integrator = PseudoIntegrator(
				self.pressure.ords, self.pressure.integrator.prod,
				shift_dim,self.pressure.integrator.phi_vals)
			self.d_test_integrator[shift_dim] = test_integrator

	def check_divergence(self,h):
		p_size = self.pressure.integrator.prod
		u_size = self.velocity.u.integrator.prod
		v_size = self.velocity.v.integrator.prod

		p_map = self.pressure.integrator.id_map
		u_map = self.velocity.u.integrator.id_map
		v_map = self.velocity.v.integrator.id_map

		p_ords = self.pressure.ords
		u_ords = self.velocity.u.ords
		v_ords = self.velocity.v.ords

		p_quad_x_shifts = [h/2,-h/2,h/2,-h/2]
		p_quad_y_shifts = [h/2,h/2,-h/2,-h/2]
		quad_starts = [(0,0),(h/2,0),(0,h/2),(h/2,h/2)]

		local_divs = {}

		for q_id in range(4):
			local_dx = np.zeros((p_size,u_size))
			local_dy = np.zeros((p_size,v_size))
			x0,y0 = quad_starts[q_id]
			x1,y1 = x0+h/2,y0+h/2
			for trial_id in range(p_size):
				trial_ind = p_map[trial_id]
				xshft = p_quad_x_shifts[q_id]
				yshft = p_quad_y_shifts[q_id]
				phi0 = lambda x,y: phi_2d_ref(p_ords,x+xshft,y,h,trial_ind)
				phi1 = lambda x,y: phi_2d_ref(p_ords,x,y+yshft,h,trial_ind)
				for test_id in range(u_size):
					test_ind_u = u_map[test_id]
					test_ind_v = v_map[test_id]
					dphi0 = lambda x,y: dphi_2d_ref(u_ords,x,y,h,test_ind_u)[0]
					dphi1 = lambda x,y: dphi_2d_ref(v_ords,x,y,h,test_ind_v)[1]

					func0 = lambda x,y: phi0(x,y)*dphi0(x,y)
					func1 = lambda x,y: phi1(x,y)*dphi1(x,y)
					val0 = gauss(func0,x0,x1,y0,y1,3)
					val1 = gauss(func1,x0,x1,y0,y1,3)
					local_dx[trial_id,test_id] += val0
					local_dy[trial_id,test_id] += val1
			local_divs[q_id] = (local_dx,local_dy)

		return local_divs

	def set_up_divergence(self):
		start = time()
		local_divs = self.check_divergence(self.velocity.u.h)
		dx_blocks,dy_blocks = [],[]

		for p_id,patch in enumerate(self.velocity.u.mesh.patches):
			dx_r, dx_c, dx_d = [],[],[]
			for e_id in patch.elements:
				el = patch.elements[e_id]
				p_el = self.element_map(el)
				for q_id,quad in enumerate(el.quads):
					if quad:
						for trial_id,p_dof in enumerate(p_el.get_dof_list(q_id)):
							dx_r += [p_dof.ID]*len(el.dof_list)
							dx_c += el.dof_ids
							dx_d += list(local_divs[q_id][0][trial_id])
			sp_dx = sparse.coo_array((dx_d,(dx_r,dx_c)),
							shape=(self.full_sizes[-1][p_id],self.full_sizes[0][p_id]))
			dx_blocks.append(sp_dx)
		for p_id,patch in enumerate(self.velocity.v.mesh.patches):
			dy_r, dy_c, dy_d = [],[],[]
			for e_id in patch.elements:
				el = patch.elements[e_id]
				p_el = self.element_map(el)
				for q_id,quad in enumerate(el.quads):
					if quad:
						for trial_id,p_dof in enumerate(p_el.get_dof_list(q_id)):
							dy_r += [p_dof.ID]*len(el.dof_list)
							dy_c += el.dof_ids
							dy_d += list(local_divs[q_id][1][trial_id])
			sp_dy = sparse.coo_array((dy_d,(dy_r,dy_c)),
							shape=(self.full_sizes[-1][p_id],self.full_sizes[1][p_id]))
			dy_blocks.append(sp_dy)
		self.pux = sparse.bmat(np.array(
				[[dx_blocks[0],None],
				[None,dx_blocks[1]]]),
				format='csc')
		self.pvy = sparse.bmat(np.array(
				[[dy_blocks[0],None],
				[None,dy_blocks[1]]]),
				format='csc')
		print('compute div comp = {}'.format(time()-start))


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

		# self.velocity.compute_divergence(self.d_test_integrator,
		# 			self.element_map,self.test_sizes)

		# self.divergence2 = sparse.bmat(np.array(
		# 	[[self.velocity.du_dx_op.spA, 
		# 	  self.velocity.dv_dy_op.spA]]),
		# 	 format='csc')

		self.set_up_divergence()
		self.divergence = sparse.bmat(np.array(
			[[self.pux, 
			  self.pvy]]),
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


		# checklist = [self.pressure.constraints.spC,
		# 	   self.velocity.u.constraints.spC,
		# 	   self.velocity.v.constraints.spC,
		# 	   self.velocity.spC,
		# 	   self.velocity.lap_u_op.spA,
		# 	   self.velocity.lap_v_op.spA,
		# 	   self.laplace,self.velocity.div_u_op.spA,
		# 	   self.velocity.div_v_op.spA,self.divergence]

		self.sys = self.C.T @ self.A @ self.C

		zlist = [self.velocity.u.Z.T,
			  	 self.velocity.v.Z.T,
				 self.pressure.Z.T]
		Z = np.hstack(zlist).T
		self.mean_values = [sum(z.flatten()) for z in zlist]
		self.zTc = self.C.T.dot(Z)
		return

		# need to take care of the nullspace

	def solve(self,forces):
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
		print(vel_L2_errs,self.pressure.error(P))
		print(vel_Linf_errs,self.pressure.Linf_error(P))

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


