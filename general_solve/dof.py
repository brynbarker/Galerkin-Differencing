from general_solve.shape_functions import phi_2d_eval, phi_3d_eval

class DoF:
	def __init__(self,ID,dim,inds,loc,h,ords):#=[3,3]):
		self.ID = ID
		self.dim = dim
		self.loc = loc
		self.ind  = inds
		self.ords = ords
		if dim == 2:
			self.i,self.j = inds
			self.x,self.y = loc
			self.k,self.z = None, None
			self.phi = lambda xy: phi_2d_eval(self.ords,xy[0],xy[1],h,self.x,self.y)
		elif dim == 3:
			self.i,self.j,self.k = inds
			self.x,self.y,self.z = loc
			self.phi = lambda xyz: phi_3d_eval(self.ords,xyz[0],xyz[1],xyz[2],h,self.x,self.y,self.z)
		else:
			raise ValueError('dim must be 2 or 3')

		self.h = h
		self.elements = {}

	def add_element(self,e):
		if e.ID not in self.elements.keys():
			self.elements[e.ID] = e