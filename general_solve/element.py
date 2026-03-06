class Element:
	def __init__(self,ID,dim,inds,loc,h,ords):#=[3,3]):
		self.ID = ID
		self.dim = dim
		self.h = h
		self.loc = loc
		self.ind = inds
		self.ords = ords
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
		for ii in range(self.ords[1]+1):
			for jj in range(self.ords[0]+1):
				self.dof_lookup_ids.append(strt+xlen*ii+jj)
		return

	def add_dofs_3d(self,strt,xlen):
		for	kk in range(self.ords[2]+1):
			for	ii in range(self.ords[1]+1):
				for	jj in range(self.ords[0]+1):
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

	def _set_lines(self):
		self.lines = []
		m = [[[[0,1],[2,3]],[[0,2],[1,3]]]]
		for comp in range(2):
			for half in range(2):
				dir = int(self.quads[m[comp][half][0]])-int(self.quads[m[comp][half][1]])
				if dir != 0:
					self.lines.append((comp,half,dir))

	def set_interface(self):
		self.interface = True

	def check_loc(self,loc):
		for d in range(self.dim):
			assert loc[d] >= self.dom[d][0] and loc[d] <= self.dom[d][1]
	
	def get_dof_ids(self,id=None):
		return self.dof_ids

class PseudoElement:
	def __init__(self):
		self.dof_id_lists = {0:{},1:{}}

		self.comp = None

	def set_comp(self,comp):
		self.comp = comp

	def add_dof_ids(self,dim,q_id,dof_ids=None):
		self.dof_id_lists[dim][q_id] = dof_ids

	def get_dof_ids(self,q_id,dim=None):
		if dim is None:
			dim = self.comp
		if dim == 0:
			return self.dof_id_lists[dim][q_id%2]
		else:
			return self.dof_id_lists[dim][int(q_id/2)]
