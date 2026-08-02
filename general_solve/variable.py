import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
from scipy import linalg as scla
import scipy.sparse.linalg as sla
from general_solve.mesh import Mesh
from general_solve.integration import Integrator
from general_solve.differential_operators import DifferentialOperator,LaplaceOperator,ProjectionOperator,DivergenceOperator
from general_solve.constraints import ConstraintOperator
from general_solve.couple_interface_levels import InterfaceMapping,Empty
from general_solve.shape_functions import phi_2d_ref
# import krylov

from general_solve import globals

class subtract_null(sla.LinearOperator):
	def __init__(self,sys,size):
		self.shape = (size,size)
		self.sys = sys
		self.dtype = sys.dtype

	def _matvec(self,x):
		return self.matvec(x)

	def matvec(self,x):
		x_proj = sum(x)/x.size
		return self.sys.dot(x-x_proj)

def corner_pin(lhs,rhs,ghost_inds,true_vec):
	sz = len(ghost_inds)
	old_sz = rhs.size
	true_inds = [i for i in range(old_sz) if i not in ghost_inds]

	new_lhs = np.zeros((sz,sz))
	new_vec = np.zeros((sz))

	pinned_sys = np.zeros((sz,old_sz-sz))
	pinned_vec = np.zeros((old_sz-sz))

	new_rhs = np.zeros((sz))

	for i,ghost_ind in enumerate(ghost_inds):
		new_rhs[i] = rhs[ghost_ind]
		for j,ghost_ind_2 in enumerate(ghost_inds):
			new_lhs[i,j] = lhs[ghost_ind,ghost_ind_2]
		for j,true_ind in enumerate(true_inds):
			pinned_sys[i,j] = lhs[ghost_ind,true_ind]
			if i==0:
				pinned_vec[j] = true_vec[true_ind]

	new_vec = np.linalg.solve(new_lhs,new_rhs-pinned_sys@pinned_vec)
	return new_vec, new_lhs,new_rhs,pinned_sys,pinned_vec



class MultiComponentVariable:
	def __init__(self,N,dim=2,doflocs=['node','node'],
			  rtype='uniform',rname=None,vars=[None,None],
			  ords=[1,1],qpn=3):
		self.u = SingleComponentVariable(
			N,dim,doflocs[0],rtype,rname,vars[0],ords,qpn)
		self.v = SingleComponentVariable(
			N,dim,doflocs[1],rtype,rname,vars[1],ords[::-1],qpn)

class SingleComponentVariable:
	def __init__(self,N,dim=2,dofloc='node',
			  rtype='uniform',rname=None,var=None,
			  ords=[1,1],qpn=None,ghost_off=False,zigzag=False):
		if qpn is None: qpn = max(ords)+1

		self.zigzag = zigzag
		
		ord_incomp = False
		if min(ords)==0 and dofloc=='node':
			ord_incomp = True
		elif ords[0]==0 and dofloc=='xside':
			ord_incomp = True
		elif ords[1]==0 and dofloc=='yside':
			ord_incomp = True
		if ord_incomp:
			raise ValueError('incompatible p value and dof location')
		self.N = N
		self.dim = dim
		self.varfunc = var
		self.qpn = qpn
		self.integrator = Integrator(qpn,dim,ords)
		self.ords = ords
		self.curr_sol = None
		self.curr_errs = [None,None]

		self.mesh = Mesh(N,dim,ords,dofloc,rtype,rname,ghost_off=ghost_off,
				   			zigzag=zigzag)
		self.mesh.set_quadrature(self.integrator)
		self.h = self.mesh.h

		if zigzag:
			self.interface_map = InterfaceMapping(
									self.mesh,self.integrator,self.ords)
			traps = self.interface_map.trapezoid_elements
			tris = self.interface_map.triangle_elements
			self.mesh.add_zigzag_elements(traps,tris)

			# zz_fills = self.interface_map.zigzag_ghost_fills

			for p in self.mesh.patches:
				for dof_id in p.dofs:
					if p.dofs[dof_id].interface:
						p.dofs[dof_id].update()#set_phi()
		else:
			self.interface_map = Empty

		self.constraints = ConstraintOperator(self.mesh)#,zz_fills)

		self.k = None
		ops = ['mass','lap','helm','div','grad']
		self.operators = {op:None for op in ops}

		self.true_sol_vec = None
		self.true_var_quad_vals = None
		self.interior_list = []
		self.view_list = []

		# self._setup_mean_value()

	def _setup_mean_value(self):
		myZs = []
		for patch in self.mesh.patches:
			num_dofs = len(patch.dofs)
			myZ = np.zeros((num_dofs,1))

			for e in patch.elements.values():
				for quad_id,quad in enumerate(e.quads):
					if quad:
						for test_id,dof in enumerate(e.dof_list):
							if e.regular:
								phi_val = self.integrator.phi_vals[quad_id][test_id]
								val = self.integrator._compute_product_integral(phi_val,volume=e.vol)
							else:
								val = self.interface_map.compute_func_integral(e,test_id,1,quad_id)
							myZ[dof.ID,0] += val
			myZs.append(myZ)

		self.Z = np.vstack(myZs)
		# vol = 1/2**self.dim
		# for e in self.mesh.zigzag_elements:
		# 	for quad_id,quad in enumerate(e.quads):
		# 		if quad:
		# 			j_det = e.get_usub_det(quad_id)
		# 			for test_id,dof in enumerate(e.dof_list):
		# 				phi_val = self.integrator.phi_vals[quad_id][test_id]
		# 				# j_det = e.J_dets[quad_id]
		# 				val = self.integrator._compute_product_integral(phi_val,volume=vol,jdet=j_det)
		# 				self.Z[e.dof_ids[test_id],0] += val

		self.zTc = self.constraints.spC.T.dot(self.Z)[:,0]
		self.mean_value = sum(self.Z)

	def sol(self, interpolants=None):

		if interpolants is None:
			return

		def solution(loc):
			e,dof_shift = self.mesh.loc_to_el(loc)
			val = 0
			for dof in set(e.dof_list):
				dof_ID = dof.ID + (dof.h != self.h)*dof_shift
				val += interpolants[dof_ID]*dof.phi(loc,el=e,glob=True)
			# if e.regular:
			# 	for dof in e.dof_list:
			# 		val += interpolants[dof.ID+dof_shift]*dof.phi(loc,e)
			# else:
			# 	xi,eta = e.inv_transform(loc[0],loc[1])
			# 	for local_id,dof in enumerate(e.dof_list):
			# 		global_id = e.dof_ids[local_id]
			# 		val += interpolants[global_id]*dof.phi([xi,eta],e,local_id)#phi_2d_ref(self.ords,xi,eta,1,id_map[local_id])
			return val
		return solution

	def _get_true_vals_at_quad_points(self):
		if self.true_var_quad_vals == None:
			tmp = [{},{}]
			for p_id,p in enumerate(self.mesh.patches):
				for e in p.elements.values():
					if e.regular:
						true_var_vals_e = self.integrator._evaluate_func_on_element(
							self.varfunc,e.bounds)
					else:
						true_var_vals_e = self.interface_map.evaluate_func_on_element(
							e,self.varfunc)
					tmp[p_id][e.global_ID] = true_var_vals_e
			# for e in self.mesh.zigzag_elements:
			# 	true_var_vals_e = self.integrator._evaluate_func_on_element(self.varfunc,
			# 								local_bounds,wrap=e.transform)
			# 	tmp[2][e.global_ID] = true_var_vals_e

			self.true_var_quad_vals = tmp
 
	def error(self,sol_vec):
		if self.true_var_quad_vals == None:
			self._get_true_vals_at_quad_points()

		d_phi_ops = [self.integrator.phi_vals,self.interface_map.phi_vals]
		errs = np.zeros(3)
		norms = [2,1,'inf']
		c_dof_shift = self.constraints.dof_id_shift
		for p_id,p in enumerate(self.mesh.patches):
			dof_shift = c_dof_shift*p_id
			for e in p.elements.values():
				d_phi = d_phi_ops[0] if e.regular else d_phi_ops[1][e.tri][e.map_type]
				if e.regular: vol = e.h/2**self.dim
				for q_id,q_bool in enumerate(e.quads):
					if q_bool:
						var_vals = self.true_var_quad_vals[p_id][e.global_ID][q_id]
						varh_vals = np.zeros_like(var_vals)
						for local_id, dof in enumerate(e.dof_list):
							phi_vals = d_phi[q_id][local_id]
							varh_vals += sol_vec[dof.ID+dof_shift]*phi_vals
						
						for j in range(3):
							if e.regular:
								errs[j] = self.integrator._compute_error_integral(
												var_vals,varh_vals,vol,norms[j],errs[j])
							else:
								errs[j] = self.interface_map.compute_error_integral(
									e,var_vals,varh_vals,q_id,norms[j],errs[j])


		errs[0] = np.sqrt(errs[0])
		return errs

	def evaluate_on_domain(self,func):
		tmp = []
		for e in self.mesh.all_elements:
			x,y = e.mid
			try:
				tmp.append(func(x,y))
			except:
				tmp.append(func([x,y]))
		return np.array(tmp)

	def solve_simple_system(self,f,op,disp=True,helm=False,proj=False):
		if helm:
			self.operators['mass']._build_system()
			self.operators['lap']._build_system()
			part0 = -1/self.operators['lap'].mu*self.operators['lap'].spA
			part1 = self.k**2*self.operators['mass'].spA
			spA = part0+part1
		else:
			op._build_system()
			spA = op.spA
		op._build_force(f)

		C = self.constraints.spC
		lhs = C.T @ spA @ C

		rhs = C.T.dot(op.F)

				
		if not proj and not helm and sum(self.ords)==1:
			self.my_ns = self.constraints.construct_null_space()
			for v in self.my_ns:
				rhs -= (v@ rhs)*v



		self.sp_lhs = lhs
		self.lhs = lhs.todense()
		self.rhs = rhs

		ns = scla.null_space(self.lhs).T
		print(ns.shape)
		for vec in ns:
			before = vec@rhs
			rhs -= before*vec
			print(before,vec@rhs)

		try:
			# assert False
			x_star,conv = sla.gmres(lhs,rhs,rtol=1e-14)
			assert conv == 0
		except:
			print('krylov issue')
			x_star = np.linalg.solve(lhs.todense(),rhs)
			self.totest = [lhs,rhs]

		self.x_star = x_star
		if proj:
			self.x = x_star
		else:
			alpha = (self.zTc @ x_star) / self.mean_value
			print(alpha)
			self.x = x_star - alpha 
		for vec in ns:
			# rhs -= (vec@rhs)*vec
			print(vec@self.x,vec@x_star)

		coef_vec = C.dot(self.x)
		approx_on_domain = self.evaluate_on_domain(self.sol(coef_vec))
		self.sol_on_domain = self.evaluate_on_domain(self.varfunc)

		op.set_solution_coef_vector(coef_vec)
		# op.set_solution_at_dofs(sol_vec)
		errs = self.error(coef_vec)
		op.set_error(errs)

		# self.curr_sol_at_dofs = sol_vec
		self.curr_approx_sol = approx_on_domain
		self.curr_coefs = coef_vec
		self.curr_errs = errs

		if disp:
			print('L2 error     = {}'.format(errs[0]))
			print('L1 error     = {}'.format(errs[1]))
			print('Linf error   = {}'.format(errs[2]))

	def solve_poisson(self,f,mu=1,disp=True):
		if self.operators['lap'] is None:
			self.operators['lap'] = LaplaceOperator(
				self.mesh,self.integrator,self.interface_map,mu=mu)
		return self.solve_simple_system(f,self.operators['lap'],disp)

	def solve_projection(self,disp=True):
		if self.operators['mass'] is None:
			self.operators['mass'] = ProjectionOperator(
				self.mesh,self.integrator,self.interface_map)
		return self.solve_simple_system(self.varfunc,self.operators['mass'],disp,proj=True)

	def solve_helmholtz(self,f,k=1,disp=True):
		if self.operators['lap'] is None:
			self.operators['lap'] = LaplaceOperator(
				self.mesh,self.integrator,self.interface_map)
		if self.operators['mass'] is None:
			self.operators['mass'] = ProjectionOperator(
				self.mesh,self.integrator,self.interface_map)
		if self.operators['helm'] is None:
			self.operators['helm'] = DifferentialOperator(
				self.mesh,self.integrator,self.interface_map)
		self.k = k
		self.solve_simple_system(f,self.operators['helm'],disp,True)

	def solve_dx(self,u_var,ufunc,ffunc,deriv_op=None):
		tmp = DifferentialOperator(self.mesh,self.integrator,self.interface_map)
		tmp._build_force(ffunc)

		U = u_var.evaluate_on_grid(ufunc)
		if deriv_op is None:
			deriv_op = self.operators['div'].pux.spA
		lhs = deriv_op.dot(U)
		return lhs, tmp.F

	def solve_dy(self,v_var,vfunc,ffunc,deriv_op=None):
		tmp = DifferentialOperator(self.mesh,self.integrator,self.interface_map)
		tmp._build_force(ffunc)

		V = v_var.evaluate_on_grid(vfunc)
		if deriv_op is None:
			deriv_op = self.operators['div'].pvy.spA
		lhs = deriv_op.dot(V)
		return lhs, tmp.F

	def solve_div_truncation(self,vars,var_funcs,ffuncs):
		u_lhs,u_F = self.solve_dx(vars[0],var_funcs[0],
							ffuncs[0])
		v_lhs,v_F = self.solve_dy(vars[1],var_funcs[1],
							ffuncs[1])
		err_u = np.linalg.norm(u_lhs-u_F)
		err_v = np.linalg.norm(v_lhs-v_F)
		return err_u,err_v

	def vis_dof_sol(self,approx_vec,locs=None,err=False,log=True,lines=False,ave_only=False):
		if err and locs is None:
			sol_vec = abs(self.sol_on_domain-approx_vec)
		elif locs is not None:
			approx_func = self.sol(approx_vec)
			sol_vec = np.array([approx_func(loc) for loc in locs])
			if err:
				true_vec = np.array([self.varfunc(x,y) for (x,y) in locs])
				sol_vec = abs(true_vec-sol_vec)
		else:
			sol_vec = approx_vec
			log = False
		if ave_only:
			return np.min(sol_vec),np.mean(sol_vec),np.max(sol_vec)
		self.mesh.vis_dof_sol(sol_vec,locs=locs,log=log,lines=lines)

	def setup_laplace(self,mu=1):
		if self.operators['lap'] is None:
			self.operators['lap'] = LaplaceOperator(
				self.mesh,self.integrator,self.interface_map,mu=mu)
		self.operators['lap']._build_system()
		return self.operators['lap']

	def solve_laplace_truncation(self,var_func,ffunc):
		if len(self.view_list)==0:
			pad = 0#2*self.h
			shift = len(self.mesh.patches[0].dofs)
			for p_id in range(2):
				for dof in self.mesh.patches[p_id].dofs.values():
					if pad<dof.x<1-pad and pad<dof.y<1-pad:
						full_id = dof.ID+p_id*shift
						if full_id in self.constraints.true_dofs:
							self.view_list.append(full_id)

		C = self.constraints.spC
		lap = self.operators['lap']
		lap._build_force(ffunc)
		rhs = C.T.dot(lap.F)

		U = self.evaluate_on_grid(var_func)
		lhs = (C.T @ lap.spA).dot(U)

		self.vis_dof_sol(C.dot(lhs),true_list=self.view_list)
		Clhs = C.dot(lhs)
		denselap = lap.spA.todense()
		check = True
		for index,val in enumerate(Clhs):
			if check:
				if abs(val) < 1e-10:
					dof = self.constraints.get_dof(index)
					if 2*self.h<dof.x<1-2*self.h:
						if 2*self.h<dof.y<1-2*self.h:
							check = False
							print((dof.x/self.h,dof.y/self.h),val)
							for influenced,val2 in enumerate(denselap[index]):
								if abs(val2)>1e-8:
									dof2 = self.constraints.get_dof(influenced)
									print('\t\t',(dof2.x/self.h,dof2.y/self.h),val2)

		for index,val in enumerate(Clhs):
			if abs(val) > 1e-8:
				dof = self.constraints.get_dof(index)
				print(dof.ID,(dof.x/self.h,dof.y/self.h),val)
				for influenced,val2 in enumerate(denselap[index]):
					if abs(val2)>1e-8:
						dof2 = self.constraints.get_dof(influenced)
						print('\t\t',dof2.ID,(dof2.x/self.h,dof2.y/self.h),val2)



		
		return np.linalg.norm(C.dot(lhs-rhs)[self.view_list])

	def setup_divergence(self,l_dphivals,el_map,
					     local_test_size,test_sizes):
		if self.operators['div'] is None:
			self.operators['div'] = DivergenceOperator(
						self.mesh,self.integrator,self.interface_map,
					    l_dphivals,el_map,
						local_test_size,test_sizes)
		self.operators['div']._build_system()
		return self.operators['div']