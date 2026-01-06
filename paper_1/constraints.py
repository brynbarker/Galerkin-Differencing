import matplotlib.pyplot as plt
import numpy as np
from scipy import sparse

class ConstraintOperator:
	def __init__(self,mesh,dirichlet=False):
		self.mesh = mesh
		self.patches = mesh.patches
		self.dof_id_shift = len(self.patches[0].dofs)

		self.size = self.dof_id_shift+len(self.patches[1].dofs)
		self.Id = np.zeros(self.size)
		self.dirichlet_Id_update = False
		if dirichlet:
			self.dirichlet = np.zeros((self.size))
		else:
			self.dirichlet = None

		self.Cr = []
		self.Cc = []
		self.Cd = []

		self._set_patches(dirichlet)
		self._setup_boundary(dirichlet)
		self._setup_interface()
		self._construct_matrix()

	def _global_dof_id(self,dof_id,p_id):
		# input dof id relative to patch NOT LOOKUP IDS
		if isinstance(dof_id,list):
			output = []
			for d_id in dof_id:
				output.append(self._global_dof_id(d_id,p_id))
			return output
		return dof_id + self.dof_id_shift*p_id

	def _local_dof_id(self,dof_id):
		# input dof id for full system NOT LOOKUP IDS
		if isinstance(dof_id,list):
			output = []
			for d_id in dof_id:
				output.append(self._local_dof_id(d_id))
			return output
		if dof_id >= self.dof_id_shift:
			return (dof_id-self.dof_id_shift,1)
		else:
			return (dof_id,0)

	def _set_patches(self,dirichlet=False):

		self.gpatch = None
		for p_id,p in enumerate(self.patches):
			if len(p.periodic_pairs)>0:
				self.bpatch = p_id
				assert len(self.patches[1-p_id].periodic_pairs) == 0

				if dirichlet:
					self.l_dirichlet = p.dirichlet_dofs

				else:
					self.d_periodic = {}
					for key in p.periodic_pairs:
						val = p.periodic_pairs[key]
						self.d_periodic[p.dofs[key].ID] = p.dofs[val].ID

			if len(p.interface_ghosts) > 0:
				self.gpatch = p_id
				self.nongpatch = 1-p_id
				assert len(self.patches[1-p_id].interface_ghosts) == 0
				self.interface_points = p.interface_points
				ghost_list = [p.dofs[lookup_id].ID for lookup_id in p.interface_ghosts]
				self.ghost_list = self._global_dof_id(ghost_list,p_id)
		
		return

	def _setup_boundary(self,dirichlet=False):
		if dirichlet: return self._setup_dirichlet()
		for per_dof_id in self.d_periodic:
			per_dof_id_pair = self.d_periodic[per_dof_id]
			per_id,per_pair_id = self._global_dof_id([per_dof_id,per_dof_id_pair],self.bpatch)
			self.Cr.append(per_id)
			self.Cc.append(per_pair_id)
			self.Cd.append(1)
			self.Id[per_id] = 1
		return

	def _setup_dirichlet(self,ufunc=None):
		if ufunc is None and self.dirichlet_Id_update == True: return
		for dir_dof_lookup_id in self.l_dirichlet:
			global_id = self._global_dof_id(dir_dof_lookup_id,self.bpatch)
			if ufunc is not None:
				dof = self.patches[self.bpatch].dofs[dir_dof_lookup_id]
				if dof.dim == 2:
					uval = ufunc(dof.x,dof.y)
				else:
					uval = ufunc(dof.x,dof.y,dof.z)
				self.dirichlet[global_id] = uval
			
			if not self.dirichlet_Id_update:
				self.Id[global_id] = 1
		self.dirichlet_Id_update = True
		return
	
	def _setup_interface(self):
		if self.gpatch is None:
			return

		ghost_vals = self.patches[self.gpatch].evaluate_interface_ghosts()

		for p_id,p in enumerate(self.patches):
			p_dof_ids = [p.dofs[lookup_id].ID for lookup_id in p.interface_dofs]
			p_interface_dofs = self._global_dof_id(p_dof_ids,p_id)
			evals = p.evaluate_interface_points(self.interface_points)

			sgn = 1. if p_id==self.nongpatch else -1.

			for g_id,g_dof_id in enumerate(self.ghost_list):
				if p_id == 0:
					self.Id[g_dof_id] = 1

				g_val = ghost_vals[g_id]
				mask = abs(evals[g_id]) > 1e-12
				local_dofs = list(np.array(p_interface_dofs)[mask])
				local_vals = (evals[g_id])[mask]

				self.Cr += [g_dof_id]*sum(mask)
				self.Cc += local_dofs
				self.Cd += list(sgn*local_vals/g_val)
		
	def _construct_matrix(self):
		self.true_dofs = list(np.where(self.Id==0)[0])
		for true_ind in self.true_dofs:
			self.Cr.append(true_ind)
			self.Cc.append(true_ind)
			self.Cd.append(1.)

		spC_full =	sparse.coo_array((self.Cd,(self.Cr,self.Cc)),shape=(self.size,self.size))
		c_data = {}
		for	i,r	in enumerate(spC_full.row):
			tup	= (spC_full.col[i],spC_full.data[i])
			if r in	c_data.keys():
				c_data[r].append(tup)
			else:
				c_data[r] =	[tup]
		self.C_full = c_data

		Cc_array = np.array(self.Cc)
		masks =	[]
		for	true_dof in	self.true_dofs:
			masks.append(Cc_array==true_dof)
		for	j,mask in enumerate(masks):
			Cc_array[mask] = j
		Cc = list(Cc_array)

		num_true = len(self.true_dofs)
		self.tocheck = [Cc]
		self.spC = sparse.coo_array((self.Cd,(self.Cr,Cc)),shape=(self.size,num_true)).tocsc()


	def vis(self):
		color = 0
		fig,ax = plt.subplots(1,2,figsize=(20,10))
		for r in self.C_full:
			pairs = self.C_full[r]
			ghost_id,gpatch = self._local_dof_id(r)
			ghost = self.patches[gpatch].get_dof(ghost_id)
			for (c,v) in pairs:
				if v == 1:
					assert len(pairs)==1
					assert c in self.true_dofs
					c_id,cpatch = self._local_dof_id(c)
					cdof = self.patches[cpatch].get_dof(c_id)
					mycolor = 'C'+str(color % 10)
					color += 1
					ax[0].plot(ghost.x,ghost.y,'o',ms=10,c=mycolor,fillstyle='none')
					ax[0].plot(cdof.x,cdof.y,'.',c=mycolor)
					ax[0].plot([ghost.x,cdof.x],[ghost.y,cdof.y],c=mycolor)


				else:
					c_id,cpatch = self._local_dof_id(c)
					cdof = self.patches[cpatch].get_dof(c_id)
					if c not in self.true_dofs:
						print(ghost.loc,cdof.loc)
						print(ghost.h,cdof.h)
					
					
