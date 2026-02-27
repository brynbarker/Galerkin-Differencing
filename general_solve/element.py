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


	def set_interface(self):
		self.interface = True

	def check_loc(self,loc):
		for d in range(self.dim):
			assert loc[d] >= self.dom[d][0] and loc[d] <= self.dom[d][1]

	def get_dof_list(self,id=None):
		return self.dof_list
			

class PseudoElement:
	def __init__(self,shift_dim):
		self.shift_dim = shift_dim
		self.dof_list = {}

	def add_dof_list(self,dofs,id):
		self.dof_list[id] = dofs

	def get_dof_list(self,id):
		if self.shift_dim == 0:
			return self.dof_list[id%2]
		else:
			return self.dof_list[int(id/2)]

class PseudoIntegrator:
	def __init__(self,ords,prod,shift_dim,og_phi_vals):
		self.shift_dim = shift_dim
		self.ord_string = '{}{}'.format(
			ords[0],ords[1])
		self.prod = prod
		self.phi_vals = {
			test_id:[] for test_id in range(prod)}

		self._setup_quad_map()
		self._setup_phi_vals(og_phi_vals)

	def _setup_quad_map(self):
		self.d_quad_map	= {}
		for quad_id in range(4):
			if self.shift_dim == 0:
				side = quad_id%2
			else:
				side = int(quad_id/2)
			shift = 1+self.shift_dim
			shift = -shift if side else shift
			self.d_quad_map[quad_id] = quad_id+shift

	def _setup_phi_vals(self,og_phi_vals):
		for test_id in range(self.prod):
			for quad_id in range(4):
				pquad_id = self.d_quad_map[quad_id]
				self.phi_vals[test_id].append(
					og_phi_vals[test_id][pquad_id])
