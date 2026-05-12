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

class NonCartElement(Element):
	def __init__(self,ID,ords,corner_nodes):
		inds = corner_nodes[0].ind
		loc = corner_nodes[0].loc
		h = corner_nodes[1].h
		dim = corner_nodes[0].dim

		self.dof_len = (ords[0]+1)*(ords[1]+1)
		super().__init__(ID, dim, inds, loc, h, ords, cart=False)

		if corner_nodes[1].x == corner_nodes[-1].x:
			if corner_nodes[0].x < corner_nodes[1].x:
				self.map_type = 0
			else:
				self.map_type = 1
		else:
			assert corner_nodes[1].y == corner_nodes[-1].y
			if corner_nodes[0].y < corner_nodes[1].y:
				self.map_type = 2
			else:
				self.map_type = 3

		self.corners = corner_nodes
		self.dof_list = [c for c in corner_nodes]
		self.dof_ids = []
		self.local_dof_ids = [c.ID for c in self.dof_list]

		self.quads = [True]*4

		self._set_corners()

	def set_jacobian(self,d_j_vals):
		self.J_vals = d_j_vals[self.map_type]

	def _set_corners(self):
		pass

	def _preorder_dofs(self):
		tmp_list = self.dof_list
		return self._order_dofs(tmp_list)

	def _order_dofs(self,tmp_list=None):
		if tmp_list is None:
			tmp_list = self.dof_list

		A,B,C,D,e0,e1 = tmp_list
		if self.map_type == 0:
			self.dof_list = [e0,A,B,e1,C,D]
		elif self.map_type == 1:
			self.dof_list = [e0,B,A,e1,D,C]
		elif self.map_type == 2:
			self.dof_list = [e0,e1,A,C,B,D]
		elif self.map_type == 3:
			self.dof_list = [e0,e1,B,D,A,C]
		else:
			raise ValueError('unknow map type')

		self.local_dof_ids = [dof.ID for dof in self.dof_list]

	def set_dof_ids(self,id_shift):
		for dof in self.dof_list:
			if dof.h == self.h: # fine
				self.dof_ids.append(dof.ID+id_shift)
			else:
				self.dof_ids.append(dof.ID)

	def add_dof(self,dof):
		if dof not in self.dof_list:
			self.dof_list.append(dof)
			self.local_dof_ids.append(dof.ID)

		if len(self.dof_list) == self.dof_len:
			self._order_dofs()


class TrapElement(NonCartElement):
	def __init__(self,ID,ords,corner_nodes):
		super().__init__(ID,ords,corner_nodes)

	def _set_corners(self):
		corner_nodes = [c for c in self.corners]
		xa,xb,xc,xd = [n.x for n in corner_nodes]
		ya,yb,yc,yd = [n.y for n in corner_nodes]
		self.x = xa
		self.y = ya

		x_coefs = [xa,xb-xa,xc-xa,xd+xa-xb-xc]
		y_coefs = [ya,yb-ya,yc-ya,yd+ya-yb-yc]

		def my_transform(xi,eta): # xi,eta \in [0,1]
			vander = [0*xi+1,xi,eta,xi*eta]
			x = sum([v*c for (v,c) in zip(vander,x_coefs)])
			y = sum([v*c for (v,c) in zip(vander,y_coefs)])
			return x,y

		self.transform = my_transform

		if 0 in x_coefs:
			slopes = [(yb-ya)/(xb-xa), (yd-yc)/(xd-xc)]
			self.slant = lambda x,y: (slopes[0]*(x-xa)+ya < y < slopes[1]*(x-xc)+yc)
			self.parralel = lambda x,y: xa < x < xb
		else:
			slopes = [(xc-xa)/(yc-ya), (xb-xd)/(yb-yd)]
			self.slant = lambda x,y: (slopes[0]*(y-ya)+xa < x < slopes[1]*(y-yd)+xd)
			self.parralel = lambda x,y: ya < y < yc

		self.check_loc = lambda x,y: self.parralel(x,y) and self.slant(x,y)

		def my_inv_transform(x,y):
			if 0 in x_coefs:
				xi = (x-x_coefs[0])/x_coefs[1]
				eta = (y-y_coefs[0]-y_coefs[1]*xi)/(y_coefs[-2]+y_coefs[-1]*xi)
			elif 0 in y_coefs:
				eta = (y-y_coefs[0])/y_coefs[1]
				xi = (x-x_coefs[0]-x_coefs[1]*xi)/(x_coefs[-2]+x_coefs[-1]*eta)
			return xi*self.h,eta*self.h
		self.inv_transform = my_inv_transform


class TriElement(NonCartElement):
	def __init__(self,ID,ords,corner_nodes):
		super().__init__(ID,ords,corner_nodes)
		self.dof_len -= 1
		self.tri = True

	def _set_corners(self):
		corner_nodes = [c for c in self.corners]
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

		def my_transform(xi,eta):
			if xb == xc:
				x = A*xi+xa
				y = (eta*(C-B)+B)*(x-xa)+ya
			else:
				y = A*eta+ya
				x = (xi*(C-B)+B)*(y-ya)+xa
			return x,y
		self.transform = my_transform

		def my_inv_transform(x,y):
			if xb == xc:
				eta = ((y-ya)/(x-xa) - B)/(C-B)
				xi = (x-xa)/A
			else:
				xi = ((x-xa)/(y-ya) - B)/(C-B)
				eta = (y-ya)/A
			return xi*self.h,eta*self.h

	def _preorder_dofs(self):
		A,B,C,e0,e1 = self.dof_list
		tmp_list = [A,B,A,C,e0,e1]
		return self._order_dofs(tmp_list)

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
