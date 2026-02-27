import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg as scla
import scipy.sparse.linalg as sla
from general_solve.element import PseudoElement, PseudoIntegrator
from general_solve.differential_operators import *
from general_solve.variable import SingleComponentVariable, MultiComponentVariable

class Pressure(SingleComponentVariable):
	def __init__(self,N,dim=2,dofloc='cell',
			  rtype='uniform',rname=None,var=None,
			  ords=[1,1],qpn=3):
		super().__init__(N,dim,dofloc,rtype,rname,var,ords,qpn)
		
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

		self.div_u_op = DivergenceOperator(
			self.u.mesh,self.u.integrator,
			test_integrators[0], el_map, test_sizes)
		self.div_u_op._build_system()

		self.div_v_op = DivergenceOperator(
			self.v.mesh,self.v.integrator,
			test_integrators[1], el_map, test_sizes)
		self.div_v_op._build_system()

	def compute_errors(self,U,V):
		u_L2 = self.u.error(U,raw=True)
		v_L2 = self.v.error(V,raw=True)
		u_Linf = self.u.Linf_error(U,raw=True)
		v_Linf = self.v.Linf_error(V,raw=True)

		joined_uv_Linf = np.hstack([u_Linf,v_Linf])

		self.L2_err = np.sqrt(u_L2+v_L2)
		self.Linf_err = np.linalg.norm(joined_uv_Linf)



class StokesFlow:
	def __init__(self,N,dim=2,rtype='uniform',
				 rname=None,vars=[None,None],
				 ords=[1,1,1],qpn=3,mu=1):
		self.N = N
		self.dim = dim
		self.ufunc, self.vfunc = vars
		self.pressure = Pressure(N,dim,qpn=qpn,
						   rtype=rtype,rname=rname,
						   ords=[ords[-1],ords[-1]])
		self.velocity = Velocity(N,dim,qpn=qpn,
						   rtype=rtype,rname=rname,
						   ords=ords[:-1],mu=mu)

		self._setup_coupling()
		self._setup_operators()
		self._setup_system()

	def _build_force(self,forces):
		if self.F is not None:
			return 

		myFs = []

		for var,ffunc in zip([self.u, self.v],forces):
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
		myFs.append(np.zeros(len(self.pressure.mesh.dofs)))
		self.F = np.hstack(myFs)

	def _setup_coupling(self):
		usize = [len(p.dofs) for p in self.velocity.u.mesh.patches]
		vsize = [len(p.dofs) for p in self.velocity.v.mesh.patches]
		psize = [len(p.dofs) for p in self.pressure.mesh.patches]
		self.sizes = [sum(usize),sum(vsize),sum(psize)]
		self.test_sizes = psize

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

		self.d_test_integrator = {}
		for shift_dim in range(2):
			test_integrator = PseudoIntegrator(
				self.pressure.ords, self.pressure.integrator.prod,
				shift_dim,self.pressure.integrator.phi_vals)
			self.d_test_integrator[shift_dim] = test_integrator

	def _setup_operators(self):
		self.laplace = sparse.bmat(np.array(
			[[self.velocity.lap_u_op.spA, None],
			 [None, self.velocity.lap_v_op.spA]]),
			 format='csc')

		self.velocity.compute_divergence(self.d_test_integrator,
					self.element_map,self.test_sizes)

		self.divergence = sparse.bmat(np.array(
			[[self.velocity.div_u_op.spA], 
			 [self.velocity.div_v_op.spA]]),
			 format='csc')
		
	def _setup_system(self):
		# build constraint matrix
		self.C = sparse.bmat(np.array(
			[[self.velocity.spC,None],
			 [None,self.pressure.constraints.spC]]),
			format='csc')

		self.A = sparse.bmat(np.array(
			[[self.laplace,self.divergence],
			 [self.divergence.T, None]]),
			 format='csc')


		checklist = [self.pressure.constraints.spC,
			   self.velocity.u.constraints.spC,
			   self.velocity.v.constraints.spC,
			   self.velocity.spC,
			   self.velocity.lap_u_op.spA,
			   self.velocity.lap_v_op.spA,
			   self.laplace,self.velocity.div_u_op.spA,
			   self.velocity.div_v_op.spA,self.divergence]

		self.sys = self.C.T @ self.A @ self.C

		Z = np.hstack([self.velocity.u.Z.T,
			  		   self.velocity.v.Z.T,
					   self.pressure.Z.T]).T
		self.mean_value = sum(Z)
		self.zTc = self.C.T.dot(Z)
		return

		# need to take care of the nullspace

	def solve(self,forces):
		self._build_force(forces)

		rhs = self.C.T.dot(self.F)

		f_proj = sum(rhs)/rhs.size
		if abs(f_proj) > 1e-12: #subtract off nullspace
			rhs -= f_proj

		x_star, conv = sla.gmres(self.sys,rhs,rtol=1e-13)
		assert conv==0

		alpha = (self.zTc @ x_star) / self.mean_value

		x = x_star - alpha
		sol_vec = self.C.dot(x)

		vel_length = sum(self.sizes[:-1])
		U = sol_vec[:self.sizes[0]]
		V = sol_vec[self.sizes[0]:vel_length]
		P = sol_vec[vel_length:]

		err = self.velocity.compute_errors(U,V)
