from general_solve.shape_functions import phi_2d_eval, dphi_2d_eval

class DoF:
	def __init__(self,ID,dim,inds,loc,h,ords):#=[3,3]):
		self.ID = ID
		self.dim = dim
		self.loc = loc
		self.ind  = inds
		self.ords = ords

		self.i,self.j = inds
		self.x,self.y = loc
		self.k,self.z = None, None

		self.interface = False
		self.phi = lambda xy: phi_2d_eval(self.ords,xy[0],xy[1],
								          h,self.x,self.y)
		
		self.dphi = lambda xy: dphi_2d_eval(self.ords,xy[0],xy[1],
								            h,self.x,self.y)

		self.h = h
		self.elements = {}


	def add_element(self,e):
		if e.ID not in self.elements.keys():
			self.elements[e.ID] = e