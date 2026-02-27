import matplotlib.pyplot as plt
import numpy as np
from scipy import sparse
from drawarrow import ax_arrow

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

	def _locate_corners(self):
		if self.size == self.dof_id_shift:
			return # uniform grid

		# this is for order 3 and 2
		extraps = {3:[4,-6,4,-1], 2:[3,-3,1]}
		extraps = [2,-1]

		def corner_sides(x,y):
			xlo = x < .25
			xhi = x > .75

			ylo = y < .25
			yhi = y > .75

			if not ((xlo or xhi) and (ylo or yhi)):
				print('false alarm')
				return False,None

			return True, [xhi,yhi] # 0 if low and 1 if hi

		self.corners = {}

		# we also want to store all dofs that corners depend on
		corner_deps = []

		for p_id,p in enumerate(self.patches):
			corner_dofs = [p.dofs[lookup] for lookup in p.corners]

			# we have four corners, let's just do this manually for now
			for c_dof in corner_dofs:
				check, sides = corner_sides(c_dof.x,c_dof.y)
				if check:
					start_inds = [c_dof.j,c_dof.i]
					extrap_inds = {0:[],1:[]}
					for dim,(start,side) in enumerate(zip(start_inds,sides)):
						sgn = -1 if side else 1
						for step in range(2):#p.ords[dim]+1):
							extrap_inds[dim].append(int(start+sgn*(step+1)))
					
					dof_list, weight_list = [],[]
					# for x_weight,j in zip(extraps[p.ords[0]],extrap_inds[0]):
					for x_weight,j in zip(extraps,extrap_inds[0]):
						# for y_weight,i in zip(extraps[p.ords[1]],extrap_inds[1]):
						for y_weight,i in zip(extraps,extrap_inds[1]):
							weight = x_weight*y_weight
							dof_lookup_id = p._get_lookup_id_from_ind([i,j])
							dof_local_id = p.dofs[dof_lookup_id].ID
							global_id = self._global_dof_id(dof_local_id,p_id)

							dof_list.append(global_id)
							weight_list.append(weight)
					
					c_global_id = self._global_dof_id(c_dof.ID,p_id)
					self.corners[c_global_id] = (dof_list,weight_list)
					corner_deps += dof_list

		self.corner_deps = list(set(corner_deps))



	def _setup_interface(self):

		self._locate_corners()

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

		corner_mods = {}

		# ghost_vals = self.patches[self.gpatch].evaluate_interface_ghosts()
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

				if g_dof_id in self.corner_deps:
					corner_mods[g_dof_id] = (local_dofs,sgn*local_vals)

		# now let's try setting the corners
		for corner_id in self.corners:
			if corner_id in self.ghost_list:
				print('false alarm')
				pass
			else:
				self.Id[corner_id] = 1
				dof_list,weight_list = self.corners[corner_id]
				for dof_id,weight in zip(dof_list,weight_list):

					if dof_id in corner_mods:
						# this is a little messy
						mod_dofs,mod_vals = corner_mods[dof_id]
						for mod_dof_id,mod_val in zip(mod_dofs,mod_vals):
							self.Cr.append(corner_id)
							self.Cc.append(mod_dof_id)
							self.Cd.append(mod_val*weight)
					else:
						self.Cr.append(corner_id)
						self.Cc.append(dof_id)
						self.Cd.append(weight)

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
	
	def vis_one_constraint(self,global_id):
		color = 0
		fig,ax = plt.subplots(1,1,figsize=(10,10))
		ax.plot([.25,.25,.75,.75,.25],[.25,.75,.75,.25,.25],'k')
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
		plt.xlim(.95*minx,1.05*maxx)
		plt.ylim(.95*miny,1.05*maxy)
		plt.show()
		return val_list


	def vis_interface(self):
		issue_spots = []
		color = 0
		fig,ax = plt.subplots(1,2,figsize=(20,10))
		val_list = []
		for ghost_id_global in self.ghost_list:
			pairs = self.C_full[ghost_id_global]
			ghost_id,gpatch = self._local_dof_id(ghost_id_global)
			assert gpatch==1
			ghost = self.patches[gpatch].get_dof(ghost_id)
			mycolor = 'C'+str(color % 10)
			color += 1
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

					
					
