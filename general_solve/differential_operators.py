import numpy as np
from scipy import sparse


class DifferentialOperator:
	def __init__(self,mesh,integrator):
		self.mesh = mesh
		self.integrator = integrator
		self.dim = mesh.dim

		self.lookup = None # needs to be overwritten
		self.blocks = []

		self.spA = None
		self.sol_vec = None
		self.F = None
		self.err = None
		
		self.element_map = lambda e: e
		self.test_sizes = [len(p.dofs) for p in mesh.patches]

	def _get_blocks(self,map=None,test_sizes=None):
		if len(self.blocks) == self.dim:
			return 

		if self.lookup is None:
			print('operator quantities not specified')
			return

		for test_size,patch in zip(self.test_sizes,self.mesh.patches):
			size = len(patch.dofs)
			Ar, Ac, Ad = [],[],[]

			for e in patch.elements.values():
				test_e = self.element_map(e)
				for id,quad in enumerate(e.quads):
					if quad:
						for test_id,dof in enumerate(test_e.get_dof_list(id)):
							Ar += [dof.ID]*len(e.dof_ids)
							Ac += e.dof_ids
							Ad += list(self.lookup[id][test_id])
			spA = sparse.coo_array((Ad,(Ar,Ac)),shape=(test_size,size))
			self.blocks.append(spA)

	def _build_system(self,scale0=1,scale1=1):
		if self.spA is not None:
			return 

		self._get_blocks()

		self.spA = sparse.bmat(np.array(
			[[self.blocks[0]*scale0,None],
			 [None,self.blocks[1]*scale1]]),format='csc')

	def _build_force(self,ffunc):
		if self.F is not None:
			return 

		myFs = []

		for patch in self.mesh.patches:
			num_dofs = len(patch.dofs)
			F = np.zeros(num_dofs)

			for e in patch.elements.values():
				vol = (e.h/2)**self.dim
				for test_id,dof in enumerate(e.dof_list):
					phi_vals = self.integrator.phi_vals[test_id]
					fvals = self.integrator._evaluate_func_on_element(ffunc,e.bounds)
					for (quad,phi_val,f_val) in zip(e.quads,phi_vals,fvals):
						if quad:
							val = self.integrator._compute_product_integral(phi_val,f_val,vol)
							F[dof.ID] += val
			myFs.append(F)

		self.F = np.hstack(myFs)

	def set_solution_vector(self,sol_vec):
		self.sol_vec = sol_vec 

	def set_error(self,err,Linf=False):
		if Linf:
			self.Linf_err = err
		else:
			self.err = err

	def set_sys_to_check(self,sys):
		self.cTkc = sys


class LaplaceOperator(DifferentialOperator):
	def __init__(self,mesh,integrator,mu):
		super().__init__(mesh,integrator)
		self.mu = mu

		self.lookup = integrator.get_k_vals()

	def _build_system(self):
		super()._build_system(scale0=self.mu,scale1=self.mu)

	def _build_force(self, ffunc):
		super()._build_force(ffunc)
		self.F *= -1

class ProjectionOperator(DifferentialOperator):
	def __init__(self,mesh,integrator):
		super().__init__(mesh,integrator)
		scales = [p.h**self.mesh.dim for p in self.mesh.patches]
		self.scale0 = scales[0]
		self.scale1 = scales[1]

		self.lookup = integrator.get_m_vals()

	def _build_system(self):
		super()._build_system(scale0=self.scale0,scale1=self.scale1)

class DivergenceOperator(DifferentialOperator):
	def __init__(self,mesh,integrator,test_integrator,el_map,test_sizes):
		super().__init__(mesh,integrator)
		self.element_map = el_map
		self.test_sizes = test_sizes
		self.test_integrator = test_integrator

		self.lookup = integrator.get_div_vals(test_integrator)

class GradientOperator(DivergenceOperator):
	def __init__(self, mesh, integrator, test_integrator, map, test_sizes):
		super().__init__(mesh, integrator, test_integrator, map, test_sizes)
