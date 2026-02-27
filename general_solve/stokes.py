import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg as scla
import scipy.sparse.linalg as sla
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

		self.lap_u = LaplaceOperator(
			self.u.mesh,self.u.integrator,mu=mu)

		self.lap_v = LaplaceOperator(
			self.v.mesh,self.v.integrator,mu=mu)

		self.laplace = sparse.bmat(np.array(
			[[self.lap_u, None],[None, self.lap_v]]))

	def compute_divergence(self,pressure,el_map,test_sizes):

		self.div_u = DivergenceOperator(
			self.u.mesh,self.u.integrator,
			pressure.integrator, el_map, test_sizes)

		self.div_v = DivergenceOperator(
			self.v.mesh,self.v.integrator,
			pressure.integrator, el_map, test_sizes)

		self.divergence = sparse.bmat(np.array(
			[[self.div_u, None],[None, self.div_v]]))

class StokesFlow:
	def __init__(self,N,dim=2,rtype='uniform',
				 rname=None,vars=[None,None],
				 forces=[None,None],
				 ords=[1,1,1],qpn=3,mu=1):
		self.ufunc, self.vfunc = vars
		self.pressure = Pressure(N,dim,qpn=qpn,
						   rtype=rtype,rname=rname,
						   ords=[ords[-1],ords[-1]])
		self.velocity = Velocity(N,dim,qpn=qpn,
						   rtype=rtype,rname=rname,
						   ords=ords[:-1],mu=mu)

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

	def _setup_system(self):
		# build constraint matrix
		self.C = sparse.bmat(np.array(
			[[self.u.constraints.spC,None,None],
			 [None,self.v.constraints.spC,None],
			 [None,None,self.pressure.constraints.spC]]),
			format='csc')

		self.A = sparse.bmat(np.array(
			[[self.laplace,    self.divergence.T],
			 [self.divergence, None]]))

		lhs = self.C.T @ self.A @ self.C
		rhs = self.C.T.dot(self.F)