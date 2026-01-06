from paper_1.shape_functions import phi3_2d_eval, phi3_3d_eval

class DoF:
	def __init__(self,ID,dim,inds,loc,h):
		self.ID = ID
		self.dim = dim
		self.loc = loc
		self.ind  = inds
		if dim == 2:
			self.i,self.j = inds
			self.x,self.y = loc
			self.k,self.z = None, None
			self.phi = lambda xy: phi3_2d_eval(xy[0],xy[1],h,self.x,self.y)
		elif dim == 3:
			self.i,self.j,self.k = inds
			self.x,self.y,self.z = loc
			self.phi = lambda xyz: phi3_3d_eval(xyz[0],xyz[1],xyz[2],h,self.x,self.y,self.z)
		else:
			raise ValueError('dim must be 2 or 3')

		self.h = h
		self.elements = {}

	def add_element(self,e):
		if e.ID not in self.elements.keys():
			self.elements[e.ID] = e