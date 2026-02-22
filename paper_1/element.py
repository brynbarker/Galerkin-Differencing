
class Element:
	def __init__(self,ID,dim,inds,loc,h):
		self.ID = ID
		self.dim = dim
		self.h = h
		self.loc = loc
		self.ind = inds
		if dim == 2:
			self.i,self.j = inds
			self.x,self.y = loc
			self.k,self.z = None, None
		elif dim == 3:
			self.i,self.j,self.k = inds
			self.x,self.y,self.z = loc
		else:
			raise ValueError('dim must be 2 or 3')

		self.bounds = []
		for x in loc:
			self.bounds.append(x)
			self.bounds.append(x+h)

		self.dof_lookup_ids = [] # lookup ids
		self.dof_ids = [] # not lookup ids
		self.dof_list = []
		self.fine = False
		self.interface = False
		self.dom = [[coord,coord+h] for coord in loc]

	def add_dofs(self,strt,lens):
		# these are lookup ids!
		if len(self.dof_ids) != 0:
			return
		if self.dim == 2:
			return self.add_dofs_2d(strt,lens)
		else:
			return self.add_dofs_3d(strt,lens)

	def add_dofs_2d(self,strt,xlen):
		for ii in range(4):
			for jj in range(4):
				self.dof_lookup_ids.append(strt+xlen*ii+jj)
		return

	def add_dofs_3d(self,strt,xlen):
		for	kk in range(4):
			for	ii in range(4):
				for	jj in range(4):
					self.dof_lookup_ids.append(strt+jj+ii*xlen+kk*xlen*xlen)
		return

	def update_dofs(self,dofs):
		if len(self.dof_list) != 0:
			return
		for dof_lookup_id in self.dof_lookup_ids:
			dof = dofs[dof_lookup_id]
			dof.add_element(self)
			self.dof_list.append(dof)
			self.dof_ids.append(dof.ID)
		return

	def set_fine(self):
		self.fine = True

	def set_support(self,quads):
		self.quads = quads


	def set_interface(self):
		self.interface = True

	def check_loc(self,loc):
		for d in range(self.dim):
			assert loc[d] >= self.dom[d][0] and loc[d] <= self.dom[d][1]
			
