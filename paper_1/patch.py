import numpy as np
from paper_1.element import Element
from paper_1.dof import DoF

refinement_type = {'uniform':0,
				   'finecenter':1,
				   'coarsecenter':2}

class Patch:
	def __init__(self,N,dim,refinement_info,dtype,level=0):
		self.N = (1+level)*N
		self.h = 1/self.N
		self.dim = dim
		self.info = refinement_info[:-1]
		self.xlen = refinement_info[-1]
		self.dofs = {}
		self.elements = {}
		self.periodic_pairs = {}
		self.dirichlet_dofs = []
		self.cell = dtype=='cell'
		self.node = dtype=='node'

		self.alt_dof = {}
		self.alt_el = {}

		self._setup()

	def get_dof(self,non_lookup_id):
		lookup_id = self.alt_dof[non_lookup_id]
		dof = self.dofs[lookup_id]
		assert dof.ID == non_lookup_id
		return dof

	def get_el(self,non_lookup_id):
		lookup_id = self.alt_el[non_lookup_id]
		el = self.elements[lookup_id]
		assert el.ID == non_lookup_id
		return el 
		
	def _get_lookup_id_from_ind(self,ind):
		if self.dim == 2:
			i,j = ind
			return i*self.xlen+j
		elif self.dim == 3:
			i,j,k = ind
			return k*self.xlen*self.xlen+i*self.xlen+j

	def _get_lookup_id_from_loc(self,loc):
		shft = 1 if self.node else 3/2
		if self.dim == 2:
			[j,i] = [int(x/self.h+shft) for x in loc]
			ind = [i,j]
		elif self.dim == 3:
			[j,i,k] = [int(x/self.h+shft) for x in loc]
			ind = [i,j,k]

		return self._get_lookup_id_from_ind(ind)

	def _get_element_from_loc(self,loc):
		
		loc = [x - (x==1)*1e-12 for x in loc]
		el_lookup_id = self._get_lookup_id_from_loc(loc)
		e =	self.elements[el_lookup_id]
		e.check_loc(loc)
		return e

	def _get_periodic_pair(self,loc):
		def get_shift(x):
			if x < 0:
				return 1
			elif x >= 1:
				return -1
			return 0
		shifts = [get_shift(x) for x in loc]
		pair_loc = [x+shft for (x,shft) in zip(loc,shifts)]
		return self._get_lookup_id_from_loc(pair_loc)

	def _setup(self):
		d_info,e_info,i_info = self.info

		for id in range(len(d_info[0])):
			ind,loc,per,bc = d_info[0][id],d_info[1][id],d_info[2][id],d_info[3][id]
			newdof = DoF(id,self.dim,ind,loc,self.h)
			lookup_id = self._get_lookup_id_from_ind(ind)
			self.dofs[lookup_id] = newdof
			self.alt_dof[id] = lookup_id
			if per:
				pair_lookup_id = self._get_periodic_pair(loc)
				self.periodic_pairs[lookup_id] = pair_lookup_id
			if bc:
				self.dirichlet_dofs.append(lookup_id)
			

		for id in range(len(e_info[0])):
			ind,loc,quads = e_info[0][id],e_info[1][id],e_info[2][id]
			newel = Element(id,self.dim,ind,loc,self.h)
			newel.set_support(quads)
			dof_lookup_id = self._get_lookup_id_from_loc(loc)
			strt = dof_lookup_id-1-self.xlen
			newel.add_dofs(strt,self.xlen)
			el_lookup_id = self._get_lookup_id_from_ind(ind)
			self.elements[el_lookup_id] = newel
			self.alt_el[id] = el_lookup_id

		for e in self.elements.values():
			e.update_dofs(self.dofs)

		self._setup_interface()

	def _setup_interface(self):
		inds,ghosts = self.info[-1]
		self.interface_dofs = []
		self.interface_ghosts = []
		self.interface_points = []
		for (ind,ghost_loc) in zip(inds,ghosts):
			dof_lookup_id = self._get_lookup_id_from_ind(ind)
			if ghost_loc is not None:
				self.interface_ghosts.append(dof_lookup_id)
				### FIND CLOSEST POINT FOR EVALUATION
				self.interface_points.append(ghost_loc)
			else:
				self.interface_dofs.append(dof_lookup_id)

		self.ghost_count = sum(self.interface_ghosts)

	def evaluate_interface_points(self,eval_points):
		evals = np.zeros((len(eval_points),len(self.interface_dofs)))

		for i,loc in enumerate(eval_points):
			for j,dof_id in enumerate(self.interface_dofs):
				dof = self.dofs[dof_id]
				evals[i,j] = dof.phi(loc)
		return evals

	def evaluate_interface_ghosts(self):
		if len(self.interface_ghosts) == 0:
			return None

		ghosts = []
		for loc,dof_id in zip(self.interface_points,self.interface_ghosts):
			dof = self.dofs[dof_id]
			val = dof.phi(loc)
			assert abs(val)>1e-12
			ghosts.append(val)
		return ghosts




	def check_evaluate_interface_ghosts(self):
		if len(self.interface_ghosts) == 0:
			return None

		tmp = len(self.interface_ghosts)
		check_arr = np.zeros((tmp,tmp))
		for i,loc in enumerate(self.interface_points):
			for j,dof_id in enumerate(self.interface_ghosts):
				dof = self.dofs[dof_id]
				val = dof.phi(loc)
				check_arr[i,j] = val
		return check_arr

