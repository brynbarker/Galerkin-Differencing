import numpy as np

class Element:
	def __init__(self,ID,dim,inds,loc,h,ords,cart=True):
		for index in range(dim):
			if ords[index] == 0:
				loc[index] -= h/2
		self.ID = ID
		self.dim = dim
		self.h = h
		self.loc = loc
		self.ind = inds
		self.ords = ords
		self.regular = cart
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
		if not self.regular:
			self._compute_jacobian()
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
	
	def get_dof_ids(self,id=None):
		return self.dof_ids

class TrapElement(Element):
	def __init__(self, ID, dim, inds, loc, h, ords):
		super().__init__(ID, dim, inds, loc, h, ords, cart=False)

	def set_corners(self,corner_nodes):
		xa,xb,xc,xd = [n.x for n in corner_nodes]
		ya,yb,yc,yd = [n.y for n in corner_nodes]
		self.x = xa
		self.y = ya

		x_coefs = [xa,xb-xa,xc-xa,xd+xa-xb-xc]
		y_coefs = [ya,yb-ya,yc-ya,yd+ya-yb-yc]

		if 0 in x_coefs:
			slopes = [(yb-ya)/(xb-xa), (yd-yc)/(xd-xc)]
			self.slant = lambda x,y: (slopes[0]*(x-xa)+ya < y < slopes[1]*(x-xc)+yc)
			self.parralel = lambda x,y: xa < x < xb
		else:
			slopes = [(xc-xa)/(yc-ya), (xb-xd)/(yb-yd)]
			self.slant = lambda x,y: (slopes[0]*(y-ya)+xa < x < slopes[1]*(y-yd)+xd)
			self.parralel = lambda x,y: ya < y < yc

		self.check_loc = lambda x,y: self.parralel(x,y) and self.slant(x,y)

		def my_transform(x,y):
			if 0 in x_coefs:
				xi = (x-x_coefs[0])/x_coefs[1]
				eta = (y-y_coefs[0]-y_coefs[1]*xi)/(y_coefs[-2]+y_coefs[-1]*xi)
			elif 0 in y_coefs:
				eta = (y-y_coefs[0])/y_coefs[1]
				xi = (x-x_coefs[0]-x_coefs[1]*xi)/(x_coefs[-2]+x_coefs[-1]*eta)
			return xi*self.h,eta*self.h
		self.transform = my_transform

		self.corners = corner_nodes

	def _compute_jacobian(self):
		x_coefs = [n.x for n in self.corners]
		y_coefs = [n.y for n in self.corners]
		all_coefs = [x_coefs,y_coefs]

		def jac(xi,eta):
			vars = [xi,eta]
			J = np.zeros((2,2))
			for i,var in enumerate(vars):
				for j,coefs in enumerate(all_coefs):
					c0 = coefs[i+1]-coefs[0]
					c1 = coefs[3]-coefs[2-i]
					J[i,j] = c0*(1-var)+c1*var
			Jinv = np.array([[J[1,1],-J[0,1]],[-J[1,0],J[0,0]]])
			Jdet = J[0,0]*J[1,1]-J[0,1]*J[1,0]
			return Jdet*Jinv
		self.jac = jac

class TriElement(Element):
	def __init__(self, ID, dim, inds, loc, h, ords):
		super().__init__(ID, dim, inds, loc, h, ords, cart=False)

	def set_corners(self,corner_nodes):
		xa,xb,xc = [n.x for n in corner_nodes]
		ya,yb,yc = [n.y for n in corner_nodes]
		self.x = xa
		self.y = ya

		if xb == xc:
			slopes = [(yb-ya)/(xb-xa), (yc-ya)/(xc-xa)]
			self.slant = lambda x,y: (slopes[0]*(x-xa)+ya < y < slopes[1]*(x-xc)+yc)
			self.parallel = lambda x,y: xa < x < xb

			A,B,C = xb-xa, (yb-ya)/(xb-xa), (yc-ya)/(xc-xa)
			self.J = lambda xi,eta: np.array([[A,0],[A*(eta*(C-B)+B),A*xi*(C-B)]])
		else:
			slopes = [(xb-xa)/(yb-ya), (xc-xa)/(yc-ya)]
			self.slant = lambda x,y: (slopes[0]*(y-ya)+xa < x < slopes[1]*(y-yc)+xc)
			self.parallel = lambda x,y: ya < y < yb

			A,B,C = yb-ya, (xb-xa)/(yb-ya), (xc-xa)/(yc-ya)
			self.J = lambda xi,eta: np.array([[A*eta*(C-B),A*(xi*(C-B)+B)],[0,A]])

		self.check_loc = lambda x,y: self.parralel(x,y) and self.slant(x,y)

		def my_transform(x,y):
			if xb == xc:
				eta = ((y-ya)/(x-xa) - B)/(C-B)
				xi = (x-xa)/A
			else:
				xi = ((x-xa)/(y-ya) - B)/(C-B)
				eta = (y-ya)/A
			return xi*self.h,eta*self.h
		self.transform = my_transform

		self.corners = corner_nodes

	def _compute_jacobian(self):
		def jac(xi,eta):
			myJ = self.J[xi,eta]
			Jinv = np.array([[myJ[1,1],-myJ[0,1]],[-myJ[1,0],myJ[0,0]]])
			Jdet = myJ[0,0]*myJ[1,1]-myJ[0,1]*myJ[1,0]
			return Jdet*Jinv
		self.jac = lambda xi,eta: jac(xi,eta)

class PseudoElement:
	def __init__(self):
		self.dof_id_lists = {0:{},1:{}}

		self.comp = None
		self.p0 = False

	def set_comp(self,comp):
		self.comp = comp

	def set_const(self):
		self.p0 = True

	def add_dof_ids(self,dim,q_id,dof_ids=None):
		self.dof_id_lists[dim][q_id] = dof_ids

	def get_dof_ids(self,q_id,dim=None):
		if dim is None:
			dim = self.comp

		if self.p0:
			return self.dof_id_lists[dim][q_id]
		if dim == 0:
			return self.dof_id_lists[dim][q_id%2]
		else:
			return self.dof_id_lists[dim][int(q_id/2)]
