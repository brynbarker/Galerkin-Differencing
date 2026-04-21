import matplotlib.pyplot as plt
import numpy as np
from scipy import sparse
from scipy import linalg as scla
from drawarrow import ax_arrow

from general_solve.debug import matvis

class ConstraintOperator:
	def __init__(self,mesh):
		self.mesh = mesh
		self.patches = mesh.patches
		self.dof_id_shift = len(self.patches[0].dofs)

		self.size = self.dof_id_shift+len(self.patches[1].dofs)
		self.Id = np.zeros(self.size)
		self.bpatch = []
		self.d_periodic = {}

		self.Cr = []
		self.Cc = []
		self.Cd = []

		self._set_patches()
		self._setup_interface()
		self._setup_boundary()
		self._construct_matrix()

	def _global_dof_id(self,dof_id,p_id):
		# input dof id relative to patch NOT LOOKUP IDS
		if isinstance(dof_id,list):
			output = []
			for d_id in dof_id:
				output.append(self._global_dof_id(d_id,p_id))
			return output
		return dof_id + self.dof_id_shift*p_id

	def construct_null_proj_op(self,comp):
		proj_ops = []
		split = len(self.true_dofs)
		for index,dof_id in enumerate(self.true_dofs):
			if dof_id >= self.dof_id_shift:
				split = index
				break

		sep_true_dofs = [self.true_dofs[:split],self.true_dofs[split:]]
		for level in range(2):
			p = self.mesh.patches[level]
			if len(p.dofs)>0:
				full_sum_arr = p.col_sum if comp else p.row_sum

				sum_arr = full_sum_arr[:,sep_true_dofs[level]]

				normed_sum_arr = sum_arr / sum_arr.sum(axis=1).reshape((-1,1))
				true_size = len(sep_true_dofs[level])

				proj_op = np.eye(true_size) - sum_arr.T @ normed_sum_arr
			else:
				proj_op = None

			proj_ops.append(proj_op)
		return proj_ops, split



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

	def get_dof(self,dof_id): # this input is the dof id used in C
		local_id,p_id = self._local_dof_id(dof_id)
		return self.patches[p_id].get_dof(local_id)

	def _set_patches(self):

		self.gpatch = None
		self.ghost_list = []
		for p_id,p in enumerate(self.patches):
			if len(p.periodic_pairs)>0:
				self.bpatch.append(p_id)

				for key in p.periodic_pairs:
					key_ID = p.dofs[key].ID
					val = p.periodic_pairs[key]
					val_ID = p.dofs[val].ID
					global_key_ID,global_val_ID = self._global_dof_id([key_ID,val_ID],p_id)
					self.d_periodic[global_key_ID] = global_val_ID
					# self.d_periodic[p.dofs[key].ID] = p.dofs[val].ID

			if len(p.interface_ghosts) > 0:
				self.gpatch = p_id
				self.nongpatch = 1-p_id
				assert len(self.patches[1-p_id].interface_ghosts) == 0
				self.interface_points = p.interface_points
				ghost_list = [p.dofs[lookup_id].ID for lookup_id in p.interface_ghosts]
				self.ghost_list = self._global_dof_id(ghost_list,p_id)
		
		return

	def _setup_boundary(self):
		# for per_dof_id in self.d_periodic:
		for per_id in self.d_periodic:
			if self.Id[per_id] == 0:
				# skip if this is a interface already set up
				per_pair_id = self.d_periodic[per_id]
				# per_dof_id_pair = self.d_periodic[per_dof_id]
				# per_id,per_pair_id = self._global_dof_id([per_dof_id,per_dof_id_pair],self.bpatch)
				self.Cr.append(per_id)
				self.Cc.append(per_pair_id)
				self.Cd.append(1)
				self.Id[per_id] = 1
		return



	def _setup_interface(self):

		if self.gpatch is None:
			return

		def swap_periodic(non_ghost_global_ids):
			to_return = []
			for global_id in non_ghost_global_ids:
				if global_id in self.d_periodic:
					to_return.append(self.d_periodic[global_id])
				else:
					to_return.append(global_id)
			return to_return

		ghost_vals_arr = self.patches[self.gpatch].evaluate_interface_ghosts()
		ghost_vals_inv = np.linalg.inv(ghost_vals_arr)

		for p_id,p in enumerate(self.patches):
			p_dof_local_ids = [p.dofs[lookup_id].ID for lookup_id in p.interface_dofs]
			p_interface_dof_ids_tmp = self._global_dof_id(p_dof_local_ids,p_id)
			p_interface_dof_ids = swap_periodic(p_interface_dof_ids_tmp)
			nonghost_evals = p.evaluate_interface_points(self.interface_points)
			evals = ghost_vals_inv @ nonghost_evals

			sgn = 1. if p_id==self.nongpatch else -1.

			for g_id_tmp,g_dof_id in enumerate(self.ghost_list):
				if g_dof_id in self.d_periodic:
					g_id = self.ghost_list.index(self.d_periodic[g_dof_id])
				else:
					g_id = g_id_tmp
				if p_id == 0: # just do this once
					self.Id[g_dof_id] = 1

				mask = abs(evals[g_id]) > 1e-12
				local_dofs = list(np.array(p_interface_dof_ids)[mask])
				local_vals = (evals[g_id])[mask]

				self.Cr += [g_dof_id]*sum(mask)
				self.Cc += local_dofs
				self.Cd += list(sgn*local_vals)

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
		# return
		self.spC = sparse.coo_array((self.Cd,(self.Cr,Cc)),shape=(self.size,num_true)).tocsc()

		# sums = self.spC.sum(axis=1)

	def vis_boundary(self):
		color = 0
		fig,ax = plt.subplots(1,2,figsize=(20,10))
		for global_id,gbool in enumerate(self.Id):
			if gbool:
				if global_id not in self.ghost_list:

					ghost_id,gpatch = self._local_dof_id(global_id)
					ghost = self.patches[gpatch].get_dof(ghost_id)
					mycolor = 'C'+str(color % 10)
					color += 1
					ax[gpatch].plot(ghost.x,ghost.y,'o',ms=10,c=mycolor,fillstyle='none')


					pairs = self.C_full[global_id]
					assert len(pairs) == 1
					c,v = pairs[0]
					assert v == 1
					assert c in self.true_dofs
					c_id,cpatch = self._local_dof_id(c)
					assert gpatch == cpatch
					cdof = self.patches[cpatch].get_dof(c_id)
					ax[gpatch].plot(cdof.x,cdof.y,'.',c=mycolor)
					# ax[gpatch].plot([ghost.x,cdof.x],[ghost.y,cdof.y],c=mycolor)

					ax_arrow(tail_position=(ghost.x,ghost.y),
						head_position=(cdof.x,cdof.y),
						ax=ax[gpatch],color=mycolor,radius=.4)

		plt.show()
	
	def vis_one_constraint(self,global_id,rtype=None):
		color = 0
		fig,ax = plt.subplots(1,1,figsize=(10,10))
		if rtype is not None:
			if rtype=='stripe':
				plt.plot([.25,.25,.75,.75],[0,1,1,0],'k')
			elif rtype=='square':
				plt.plot([.25,.75,.75,.25,.25],[.25,.25,.75,.75,.25],'k')
		val_list = []
		assert global_id in self.ghost_list
		pairs = self.C_full[global_id]
		ghost_id,gpatch = self._local_dof_id(global_id)
		assert gpatch==1
		ghost = self.patches[gpatch].get_dof(ghost_id)
		mycolor = 'C'+str(color % 10)
		color += 1
		myval_list = []
		minx,maxx = 1,0
		miny,maxy = 1,0
		for (c,v) in pairs:
			val_list.append(v)
			myval_list.append(v)
			c_id,cpatch = self._local_dof_id(c)
			cdof = self.patches[cpatch].get_dof(c_id)
			print(((cdof.x-ghost.x)/(self.mesh.h/(1+cpatch)),(cdof.y-ghost.y)/(self.mesh.h/(1+cpatch))),v)
			if c not in self.true_dofs:
				print(ghost.loc,cdof.loc)
				print(ghost.h,cdof.h)

			# mycolor = 'C'+str(color % 10)
			# color += 1
			cm = '.' if cpatch==1 else 'o'
			ax.plot(ghost.x,ghost.y,'o',ms=10,c=mycolor,fillstyle='none')
			ax.plot(cdof.x,cdof.y,cm,c=mycolor)
			# ax[1].plot([ghost.x,cdof.x],[ghost.y,cdof.y],c=mycolor)
			ax_arrow(tail_position=(ghost.x,ghost.y),
				head_position=(cdof.x,cdof.y),
				ax=ax,color=mycolor,radius=.4)
			minx = min(minx,cdof.x)
			miny = min(miny,cdof.y)
			maxx = max(maxx,cdof.x)
			maxy = max(maxy,cdof.y)
		try:
			assert abs(sum(myval_list)-1)<1e-12
		except:
			print((ghost.x,ghost.y),sum(myval_list),len(pairs))
		plt.show()
		return val_list


	def vis_interface(self,mylevel=None,rtype=None):
		issue_spots = []
		color = 0
		fig,ax = plt.subplots(2,1,figsize=(10,20))
		val_list = []
		full_ghost_list = self.ghost_list
		if rtype is not None:
			for i in range(2):
				if rtype=='stripe':
					ax[i].plot([.25,.25,.75,.75],[0,1,1,0],'k')
				elif rtype=='square':
					ax[i].plot([.25,.75,.75,.25,.25],[.25,.25,.75,.75,.25],'k')
		for ghost_id_global in full_ghost_list:
			pairs = self.C_full[ghost_id_global]
			ghost_id,gpatch = self._local_dof_id(ghost_id_global)
			assert gpatch==1
			ghost = self.patches[gpatch].get_dof(ghost_id)
			mycolor = 'C'+str(color % 10)
			color += 1
			ax[1].plot([.25,.25,.75,.75,.25],[.25,.75,.75,.25,.25],'grey')
			myval_list = []
			for (c,v) in pairs:
				if len(pairs) == 1:#if v == 1:
					# assert len(pairs)==1
					myval_list.append(v)
					assert c in self.true_dofs
					c_id,cpatch = self._local_dof_id(c)
					cdof = self.patches[cpatch].get_dof(c_id)
					# mycolor = 'C'+str(color % 10)
					# color += 1
					ax[0].plot(ghost.x,ghost.y,'o',ms=10,c=mycolor,fillstyle='none')
					ax[0].plot(cdof.x,cdof.y,'.',c=mycolor)
					ax[0].plot([ghost.x,cdof.x],[ghost.y,cdof.y],c=mycolor)


				else:
					val_list.append(v)
					myval_list.append(v)
					c_id,cpatch = self._local_dof_id(c)
					cdof = self.patches[cpatch].get_dof(c_id)
					if c not in self.true_dofs:
						print(ghost.loc,cdof.loc)
						print(ghost.h,cdof.h)

					# mycolor = 'C'+str(color % 10)
					# color += 1
					ax[1].plot(ghost.x,ghost.y,'o',ms=10,c=mycolor,fillstyle='none')

					if mylevel is None or mylevel==cpatch:
						ax[1].plot(cdof.x,cdof.y,'.',c=mycolor)
						# ax[1].plot([ghost.x,cdof.x],[ghost.y,cdof.y],c=mycolor)
						ax_arrow(tail_position=(ghost.x,ghost.y),
							head_position=(cdof.x,cdof.y),
							ax=ax[1],color=mycolor,radius=.4)
			try:
				assert abs(sum(myval_list)-1)<1e-12
			except:
				issue_spots.append(ghost_id_global)
				print(ghost_id_global,(ghost.x,ghost.y),sum(myval_list),len(pairs))
		plt.show()

		for global_id in issue_spots:
			self.vis_one_constraint(global_id)
		return val_list, issue_spots

	def view_true_dofs(self):
		plt.plot([0,0,1,1,0],[0,1,1,0,0],'lightgrey')
		if self.size != self.dof_id_shift:
			plt.plot([.25,.25,.75,.75,.25],[.25,.75,.75,.25,.25],'lightgrey')
		for global_id in self.true_dofs:
			local_id,p_id = self._local_dof_id(global_id)
			dof = self.patches[p_id].get_dof(local_id)
			m = '.' if p_id else 'o'
			ms = 5 if p_id else 5
			plt.plot(dof.x,dof.y,'C'+str(1-p_id)+m,ms=ms)
		plt.show()

					
					
