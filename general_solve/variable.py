import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
from scipy import linalg as scla
import scipy.sparse.linalg as sla
from general_solve.mesh import Mesh
from general_solve.integration import Integrator
from general_solve.differential_operators import DifferentialOperator,LaplaceOperator,ProjectionOperator
from general_solve.constraints import ConstraintOperator

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
			  ords=[1,1],qpn=3):
		self.N = N
		self.dim = dim
		self.varfunc = var
		self.qpn = qpn
		self.integrator = Integrator(qpn,dim,ords)
		self.ords = ords

		self.mesh = Mesh(N,dim,ords,dofloc,rtype,rname)
		self.h = self.mesh.h

		self.constraints = ConstraintOperator(self.mesh)

		self.k = None
		ops = ['mass','lap','helm','div','grad']
		self.operators = {op:None for op in ops}

		self.true_sol_vec = None
		self.true_var_vals = None
		
		self._setup_mean_value()

	def _setup_mean_value(self):
		myZs = []
		for patch in self.mesh.patches:
			num_dofs = len(patch.dofs)
			myZ = np.zeros((num_dofs,1))

			for e in patch.elements.values():
				vol = (e.h)**self.dim
				for test_id,dof in enumerate(e.dof_list):
					phi_vals = self.integrator.phi_vals[test_id]
					for (quad,phi_val) in zip(e.quads,phi_vals):
						if quad:
							val = self.integrator._compute_product_integral(phi_val,volume=vol)
							myZ[dof.ID,0] += val
			myZs.append(myZ)

		self.Z = np.vstack(myZs)

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

	def error(self,sol_vec,raw=False):
		if self.true_var_vals == None:
			tmp = [{},{}]
			for p_id,p in enumerate(self.mesh.patches):
				dof_shift = self.constraints.dof_id_shift*p_id
				for e in p.elements.values():
					vol = (e.h)**self.dim
					true_var_vals_e = self.integrator._evaluate_func_on_element(self.varfunc,e.bounds)
					tmp[p_id][e.ID] = true_var_vals_e
			self.true_var_vals = tmp
		l2_err = 0.
		for p_id,p in enumerate(self.mesh.patches):
			dof_shift = self.constraints.dof_id_shift*p_id
			for e in p.elements.values():
				vol = (e.h/2)**self.dim
				for q_id,q_bool in enumerate(e.quads):
					var_vals = self.true_var_vals[p_id][e.ID][q_id]
					if q_bool:
						varh_vals = 0
						for local_id, dof in enumerate(e.dof_list):
							phi_vals = self.integrator.phi_vals[local_id][q_id]
							varh_vals += sol_vec[dof.ID+dof_shift]*phi_vals
						l2_err += self.integrator._compute_error_integral(var_vals,varh_vals,vol)
		if raw: return l2_err
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


	def Linf_error(self,sol_vec,raw=False):
		if self.true_sol_vec is None:
			tmp = []
			for p in self.mesh.patches:
				for lookup_id in p.dofs:
					dof = p.dofs[lookup_id]
					if self.dim == 2:
						tmp.append(self.varfunc(dof.x,dof.y))
					if self.dim == 3:
						tmp.append(self.varfunc(dof.x,dof.y,dof.z))
			self.true_sol_vec = np.array(tmp)
		raw_err = (sol_vec-self.true_sol_vec)[self.constraints.true_dofs]

		if raw:
			return raw_err

		return np.linalg.norm(raw_err)

	def solve_simple_system(self,f,op,disp=True,helm=False):
		op._build_force(f)

		if helm:
			self.operators['mass']._build_system()
			self.operators['lap']._build_system()
			part0 = -1/self.operators['lap'].mu*self.operators['lap'].spA
			part1 = self.k**2*self.operators['mass'].spA
			spA = part0+part1
		else:
			op._build_system()
			spA = op.spA

		C = self.constraints.spC
		lhs = C.T @ spA @ C

		rhs = C.T.dot(op.F)

		f_proj = sum(rhs)/rhs.size
		if abs(f_proj) > 1e-12:
			print('f in null')
			rhs -= f_proj

		try:
			x_star,conv = sla.cg(lhs,rhs,rtol=1e-13)
			assert conv == 0
		except:
			x_star = np.linalg.solve(lhs.todense(),rhs)
			print('Krylov Solver did not converge, check self.totest')
			self.totest = [lhs,rhs]

		zTcx_star = self.Z.T @ C.dot(x_star)
		alpha = zTcx_star[0] / sum(self.Z)

		x = x_star - alpha

		sol_vec = C.dot(x)

		op.set_solution_vector(sol_vec)
		err = self.error(sol_vec)
		Linf_err = self.Linf_error(sol_vec)
		op.set_error(err)
		op.set_error(Linf_err,Linf=True)

		if disp:
			print('L2 error     = {}'.format(err))
			print('Linf error   = {}'.format(Linf_err))

	def solve_poisson(self,f,mu=1,disp=True):
		if self.operators['lap'] is None:
			self.operators['lap'] = LaplaceOperator(
				self.mesh,self.integrator,mu=mu)
		self.solve_simple_system(f,self.operators['lap'],disp)

	def solve_projection(self,disp=True):
		if self.operators['mass'] is None:
			self.operators['mass'] = ProjectionOperator(
				self.mesh,self.integrator)
		self.solve_simple_system(self.varfunc,self.operators['mass'],disp)

	def solve_helmholtz(self,f,k=1,disp=True):
		if self.operators['lap'] is None:
			self.operators['lap'] = LaplaceOperator(
				self.mesh,self.integrator)
		if self.operators['mass'] is None:
			self.operators['mass'] = ProjectionOperator(
				self.mesh,self.integrator)
		if self.operators['helm'] is None:
			self.operators['helm'] = DifferentialOperator(
				self.mesh,self.integrator)
		self.k = k
		self.solve_simple_system(f,self.operators['mass'],disp,True)

	def vis_dof_sol(self,sol_vec,err=False,true_list=None):
		if err:
			sol_vec = abs(self.true_sol_vec-sol_vec)
		self.mesh.vis_dof_sol(sol_vec,true_list=true_list)
