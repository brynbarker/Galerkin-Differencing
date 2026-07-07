import numpy as np
from scipy import sparse
from general_solve import shape_functions as sf

class DifferentialOperator:
	def __init__(self,mesh,integrator,coupler):
		self.mesh = mesh
		self.integrator = integrator
		self.coupler = coupler
		self.dim = mesh.dim

		self.lookup = None # needs to be overwritten
		self.blocks = []

		self.spA = None
		self.sol_vec = None
		self.F = None
		self.err = None
		
		self.element_map = lambda e: e
		self.test_sizes = [len(p.dofs) for p in mesh.patches]

		# if mesh.zigzag:# is not None:
		# 	self.zigzag = True
		# 	# self.set_zigzag(mesh.zigzag_elements)
		# 	self.get_zigzag_lookup = mesh.get_zigzag_lookup_vals
		# else:
		# 	self.zigzag = False
		

	# def set_zigzag(self,extra_elements):
	# 	self.zigzag_elements = extra_elements

	def _get_blocks(self):
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
				e_lookup = self.lookup if e.regular else self.zigzag_lookup[e.tri][e.map_type]
				for trial_id,dof in enumerate(e.dof_list):
					vals = 0.
					for q_id,quad in enumerate(e.quads):
						if quad:
							test_ids = test_e.get_dof_ids(q_id)
							vals += e_lookup[q_id][trial_id]
							# vals += self.lookup[id][trial_id]
					Ar += [dof.ID]*len(test_ids)
					Ac += test_ids
					Ad += list(vals)#(self.lookup[id][trial_id])
			spA = sparse.coo_array((Ad,(Ar,Ac)),shape=(size,test_size))
			self.blocks.append(spA)

	def _build_system(self,scale0=1,scale1=1):
		if self.spA is not None:
			return 

		self._get_blocks()

		self.spA = sparse.bmat(np.array(
			[[self.blocks[0]*scale0,None],
			 [None,self.blocks[1]*scale1]]),format='csc')

	def _build_force(self,ffunc):
		# if self.F is not None:
			# return 

		myFs = []

		self.d_dofid_to_e_list = {0:{},1:{}}
		for p_id,patch in enumerate(self.mesh.patches):
			num_dofs = len(patch.dofs)
			F = np.zeros(num_dofs)


			for e in patch.elements.values():
				vol = (e.h/2)**self.dim
				if e.regular:
					fvals = self.integrator._evaluate_func_on_element(ffunc,e.bounds)
				else:
					fvals = self.coupler.evaluate_func_on_element(e,ffunc)
				for test_id,dof in enumerate(e.dof_list):
					for quad_id,quad in enumerate(e.quads):
						if quad:
							f_val = fvals[quad_id]

							if e.regular:
								phi_val = self.integrator.phi_vals[quad_id][test_id]
								val = self.integrator._compute_product_integral(phi_val,f_val,vol)
							else:
								val = self.coupler.compute_func_integral(e,test_id,f_val,quad_id)
							F[dof.ID] += val
			myFs.append(F)

		self.F = np.hstack(myFs)

	def set_solution_coef_vector(self,coef_vec):
		self.coef_vec = coef_vec 

	def set_solution_at_dofs(self,sol_at_dofs_vec):
		self.sol_at_dofs_vec = sol_at_dofs_vec 

	def set_error(self,errs):
		self.errs = errs
		self.L2 = errs[0]
		self.L1 = errs[1]
		self.Linf = errs[2]

	def set_sys_to_check(self,sys):
		self.cTkc = sys

class LaplaceOperator(DifferentialOperator):
	def __init__(self,mesh,integrator,coupler,mu=1):
		super().__init__(mesh,integrator,coupler)
		self.mu = mu 

		self.lookup = integrator.get_k_vals()
		self.zigzag_lookup = coupler.get_lookup_vals(True)

	def _build_system(self):
		super()._build_system(scale0=self.mu,scale1=self.mu)

	def _build_force(self, ffunc):
		super()._build_force(ffunc)
		self.F *= -1

class ProjectionOperator(DifferentialOperator):
	def __init__(self,mesh,integrator,coupler):
		super().__init__(mesh,integrator,coupler)
		scales = [p.h**self.mesh.dim for p in self.mesh.patches]
		self.scale0 = scales[0]
		self.scale1 = scales[1]

		self.lookup = integrator.get_m_vals()
		self.zigzag_lookup = coupler.get_lookup_vals(False)

	def _build_system(self):
		super()._build_system(scale0=self.scale0,scale1=self.scale1)

class DerivativeOperator(DifferentialOperator):
	def __init__(self,mesh,integrator,coupler,el_map,test_sizes,comp):
		super().__init__(mesh,integrator,coupler)

		def el_map_comp(e):
			new_e = el_map[e]
			new_e.set_comp(comp)
			return new_e

		self.element_map = el_map_comp
		self.test_sizes = test_sizes
		self.comp = comp

		scales = [p.h for p in self.mesh.patches]
		self.scale0 = scales[0]
		self.scale1 = scales[1]

		self.lookup = None
		self.zigzag_lookup = None

	def set_lookup(self,lookup_d):
		self.lookup = lookup_d

	def _build_system(self):
		super()._build_system(scale0=self.scale0,scale1=self.scale1)


class DivergenceOperator:
	def __init__(self,mesh,integrator,l_dphivals,el_map,
			     local_test_size,test_sizes):
		self.diff_ops = []

		self.pux = DerivativeOperator(mesh,integrator,el_map,test_sizes[0],0)
		self.pvy = DerivativeOperator(mesh,integrator,el_map,test_sizes[1],1)

		self.lookup = integrator.get_div_vals(l_dphivals,local_test_size)
		self.pux.set_lookup(self.lookup[0])
		self.pvy.set_lookup(self.lookup[1])


	def _build_system(self):
		self.pux._build_system()
		self.pvy._build_system()

		self.spA = sparse.bmat(np.array(
			 [[self.pux.spA,self.pvy.spA]]),
			 format='csc')


		