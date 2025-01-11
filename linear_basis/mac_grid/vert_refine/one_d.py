
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la
from matplotlib.animation import FuncAnimation
from IPython.display import HTML

def phi1(y,h):
	if -h < y <= 0:
		return 1+1/h*y
	elif 0 < y <= h:
		return 1-1/h*y
	else:
		return 0
	
def phi1_dy(y,h):
	if -h < y <= 0:
		return 1/h
	elif 0 < y <= h:
		return -1/h
	else:
		return 0

def phi1_interface(y,h):
	if -3/4*h < y <= 0:
		return 1+4/3/h*y
	elif 0 < y <= 3/4*h:
		return 1-4/3/h*y
	else:
		return 0
		
def phi1_interface_dy(y,h):
	if 0 < y <= 3/4*h: 
		return -4/3/h
	elif -3/4*h < y <= 0:
		return 4/3/h
	else:
		return 0
		
def phi1_dy_eval(y_in,h,y0,I=False):
	y = y_in-y0
	if I:
		return phi1_interface_dy(y,h)
	return phi1_dy(y,h)

def phi1_eval(y_in,h,y0,I=False):
	y = y_in-y0
	if I:
		return phi1_interface(y,h)
	return phi1(y,h)

def phi1_ref(y_ref,h,i,I=False):
	y = y_ref-h*i*(I*3/4)
	if I:
		return phi1_interface(y,h)
	return phi1(y,h)

def grad_phi1_ref(y_ref,h,i,I=False):
	y = y_ref-h*i*(I*3/4)
	if I:
		return phi1_interface_dy(y,h)
	return phi1_dy(y,h)

#visualization helpers

def vis_constraints(C,dofs):
	fig = plt.figure(figsize=(15,10))
	constrained = np.where(np.diag(C)==0)[0]
	h = dofs[0].h

	#to_plot = [h*1.5,h,2*h]
	#titles = ['Coarse/Fine','Fine/Fine','Coarse/Coarse']
	#plt.plot([-1.5*h,1+1.5*h],[-.1,-.1],'grey',alpha=.2)
	#plt.plot([-2*h,1+2*h],[.1,.1],'grey',alpha=.2)
	for i,ind in enumerate(constrained):
		dof_inds = np.nonzero(C[ind])[0]
		f_x = dofs[ind].x
		for c_ind in dof_inds:
			c_x = dofs[c_ind].x
	
			if dofs[ind].h != dofs[c_ind].h:
				plt.plot([f_x,f_x],[-.11,.11],'grey',alpha=.2)
				plt.plot([f_x,c_x],[-.1,.1],'C'+str(i),alpha=.8)
				plt.scatter([f_x,c_x],[-.1,.1],c='k')

	plt.ylim(-.11,.11)
	plt.xlim(-1.5*h,1+1.5*h)
	#plt.title(titles[_])
	plt.title('interface constraints')
	plt.show()

	fig = plt.figure(figsize=(10,10))
	for i,ind in enumerate(constrained):
		dof_inds = np.nonzero(C[ind])[0]
		d0x,d0y = dofs[ind].x,dofs[ind].y
		for c_ind in dof_inds:
			d1x,d1y = dofs[c_ind].x,dofs[c_ind].y
	
			if dofs[ind].h == dofs[c_ind].h:
				if dofs[ind].h == h: color0,color1 = 'C0','C1'
				elif dofs[ind].h == h/2: color0,color1 = 'C2','C4'
				else: 
					color0,color1='k','k'#raise ValueError('something is up')
				plt.scatter([d0x],[d0y],c=color0)
				plt.scatter([d1x],[d1y],c=color1)#,alpha=.2)

	plt.title('periodic constraints')
	plt.show()
#animators

def animate_2d(frames,data,size,figsize=(10,10),yesdot=True):
	fig,ax = plt.subplots(figsize=figsize)
	frame = frames[0]
	if yesdot:
		ax.set_xlim(-.2,1.2)
		ax.set_ylim(-.2,1.2)
	else:
		ax.set_xlim(frame[0][0],frame[0][-1])
		ax.set_ylim(min(data[0][0][1])-.1,max(data[0][0][1])+.1)
	
	line, = ax.plot(frame[0],frame[1],'lightgrey')
	blocks = []
	if yesdot: dot, = ax.plot([],[],'ko',linestyle='None')
	for i in range(size):
		block, = ax.plot([],[])
		blocks.append(block)

	def update(n):
		if yesdot: blocks_n, dots_n = data[n]
		else: blocks_n = data[n]
		if len(frames) > 1: frame = frames[n]
		else: frame = frames[0]
		line.set_data(frame[0],frame[1])
		for i in range(size):
			if i < len(blocks_n):
				blocks[i].set_data(blocks_n[i][0],blocks_n[i][1])
			else:
				blocks[i].set_data([],[])
		if yesdot: dot.set_data(dots_n[0],dots_n[1])
		to_return = [line]+blocks
		if yesdot: to_return += [dot]
		return to_return
	interval = 400 if yesdot else 100
	ani = FuncAnimation(fig, update, frames=len(data), interval=interval)
	plt.close()
	return HTML(ani.to_html5_video())

#integrators

def gauss(f,a,b,n):
	ymid = (a+b)/2
	yscale = (b-a)/2
	[p,w] = np.polynomial.legendre.leggauss(n)
	integral = 0.
	for i in range(n):
		integral += w[i]*f(yscale*p[i]+ymid)
	return integral*yscale

def local_stiffness(h,qpn=5,half=-1,I=False):
	K = np.zeros((4,4))
	id_to_ind = {ID:[int(ID/2),ID%2] for ID in range(4)}

	y0 = h/2*(half>0)
	y1 = h-h/2*(half==0)

	for test_id in range(4):

		test_ind = id_to_ind[test_id]
		grad_phi_test = lambda x,y: grad_phi1_ref(x,y,h,test_ind,I)

		for trial_id in range(test_id,4):

			trial_ind = id_to_ind[trial_id]
			grad_phi_trial = lambda x,y: grad_phi1_ref(x,y,h,trial_ind,I)

			func = lambda x,y: grad_phi_trial(x,y) @ grad_phi_test(x,y)
			val = gauss(func,0,h,y0,y1,qpn)

			K[test_id,trial_id] += val
			K[trial_id,test_id] += val * (test_id != trial_id)
	return K

def local_mass(h,qpn=5,half=-1,I=False):
	M = np.zeros((2,2))

	y0 = h/2*(half>0)
	y1 = h-h/2*(half==0)

	if I:
		y0, y1 = 0, 3/4*h
		
	for test_id in range(2):

		phi_test = lambda y: phi1_ref(y,h,test_id,I)

		for trial_id in range(test_id,2):

			phi_trial = lambda y: phi1_ref(y,h,trial_id,I)

			func = lambda y: phi_trial(y) * phi_test(y)
			val = gauss(func,y0,y1,qpn)
			print(test_id,trial_id,val)

			M[test_id,trial_id] += val
			M[trial_id,test_id] += val * (test_id != trial_id)
	return M


class Node:
	def __init__(self,ID,i,y,h):
		self.ID = ID
		self.i = i
		self.y = y
		self.h = h
		self.elements = {}

	def add_element(self,e):
		if e.ID not in self.elements.keys():
			self.elements[e.ID] = e

class Element:
	def __init__(self,ID,i,y,h):
		self.ID = ID
		self.i = i
		self.y = y
		self.h = h
		self.dof_ids = []
		self.dof_list = []
		self.fine = False
		self.interface = False
		self.half = False
		self.dom = [y,y+h]

		if y < 0 or y > 1-h:
			self.half = True
			self.dom = [max(0,y),max(0,y)+h/2]

	def add_dofs(self,strt):
		if len(self.dof_ids) != 0:
			return
		for ii in range(2):#4):
			self.dof_ids.append(strt++ii)
		return

	def update_dofs(self,dofs):
		if len(self.dof_list) != 0:
			return
		for dof_id in self.dof_ids:
			dof = dofs[dof_id]
			dof.add_element(self)
			self.dof_list.append(dof)
		return

	def set_fine(self):
		self.fine = True
		
	def set_interface(self):
		self.dom[-1] -= self.h/4
		self.interface = True

class Mesh:
	def __init__(self,N):
		self.N = N # number of fine elements 
				   # from y=0.5 to y=1.0
		self.h = 0.5/N
		self.dofs = {}
		self.elements = []
		self.boundaries = []
		self.interface = [[],[]]
		
		self._make_coarse()
		self._make_fine()

		self._update_elements()

	def _make_coarse(self):
		H = self.h*2
		ydom = np.linspace(0-H/2,0.5+H/2,int(self.N/2)+2)

		ylen = len(ydom)

		dof_id,e_id = 0,0
		for i,y in enumerate(ydom):
			if (i == ylen-1):
				y -= H/4
			print(y,dof_id)
			interface_element = (i == ylen-2)
			self.dofs[dof_id] = Node(dof_id,i,y,H)

			if (y<.5):
				strt = dof_id#-xlen
				element = Element(e_id,i,y,H)
				element.add_dofs(strt)
				self.elements.append(element)
				e_id += 1
				if interface_element: element.set_interface()

			if y<0:
				self.boundaries.append(dof_id)
			if (y > 0.5):
				self.interface[0].append(dof_id)

			dof_id += 1

		self.n_coarse_dofs = dof_id
		self.n_coarse_els = e_id

	def _make_fine(self):
		H = self.h
		ydom = np.linspace(0.5+H/2,1.+H/2,self.N+1)

		ylen = len(ydom)

		dof_id,e_id = self.n_coarse_dofs,self.n_coarse_els
		for i,y in enumerate(ydom):
			self.dofs[dof_id] = Node(dof_id,i,y,H)

			if (y<1.):
				strt = dof_id#-xlen
				element = Element(e_id,i,y,H)
				element.add_dofs(strt)
				element.set_fine()
				self.elements.append(element)
				e_id += 1

			if y>1.:# and (0 <= y < 1):#or y==0. or y==1:
				self.boundaries.append(dof_id)
			if (y < 0.5+H):
				self.interface[1].append(dof_id)
			dof_id += 1

	def _update_elements(self):
		for e in self.elements:
			e.update_dofs(self.dofs)

class Solver:
	def __init__(self,N,u,f=None,qpn=5):
		self.N = N
		self.ufunc = u
		self.ffunc = f #needs to be overwritten 
		self.qpn = qpn

		self.mesh = Mesh(N)
		self.h = self.mesh.h

		self.solved = False
		self.C = None
		self.Id = None

	def _build_force(self):
		num_dofs = len(self.mesh.dofs)
		self.F = np.zeros(num_dofs)

		for e in self.mesh.elements:
			y0 = e.h/2*(e.y<0)
			y1 = e.h-e.h/2*(e.y+e.h>1)
			for test_id,dof in enumerate(e.dof_list):

				phi_test = lambda y: phi1_ref(y,e.h,test_id)
				func = lambda y: phi_test(y) * self.ffunc(y+e.y)
				val = gauss(func,y0,y1,self.qpn)

				self.F[dof.ID] += val

	def _build_stiffness(self):
		num_dofs = len(self.mesh.dofs)
		self.K = np.zeros((num_dofs,num_dofs))

		k_coarse = local_stiffness(2*self.h,qpn=self.qpn)
		k_fine = local_stiffness(self.h,qpn=self.qpn)
		
		kh_coarse_top = local_stiffness(2*self.h,qpn=self.qpn,half=1)
		kh_coarse_bot = local_stiffness(2*self.h,qpn=self.qpn,half=0)
		
		kh_fine_top = local_stiffness(self.h,qpn=self.qpn,half=1)
		kh_fine_bot = local_stiffness(self.h,qpn=self.qpn,half=0)

		local_ks = [k_coarse,k_fine]
		half_ks = [[kh_coarse_top,kh_coarse_bot],
				   [kh_fine_top,kh_fine_bot]]

		for e in self.mesh.elements:
			local_k = local_ks[e.fine]
			if e.half:
				local_k = half_ks[e.fine][e.y<0]
			for test_id,dof in enumerate(e.dof_list):
				self.K[dof.ID,e.dof_ids] += local_k[test_id]


	def _build_mass(self):
		num_dofs = len(self.mesh.dofs)
		self.M = np.zeros((num_dofs,num_dofs))

		base_m = local_mass(self.h,qpn=self.qpn)
	
		top_m = local_mass(self.h,qpn=self.qpn,half=1)
		bot_m = local_mass(self.h,qpn=self.qpn,half=0)
		half_ms = [top_m,bot_m]

		interface_m = local_mass(self.h*2,qpn=self.qpn,I=True)

		for e in self.mesh.elements:
			scale = 1 if e.fine else 2
			for test_id,dof in enumerate(e.dof_list):
				#if e.half:
				#	self.M[dof.ID,e.dof_ids] += half_ms[e.y<0][test_id] * scale
				if e.interface:
					self.M[dof.ID,e.dof_ids] += interface_m[test_id]
				else:
					self.M[dof.ID,e.dof_ids] += base_m[test_id] * scale

	def _setup_constraints(self):
		num_dofs = len(self.mesh.dofs)
		self.Id = np.zeros(num_dofs)
		self.C = np.eye(num_dofs)
		self.dirichlet = np.zeros(num_dofs)

		c_inter,f_inter = self.mesh.interface
		self.Id[f_inter] = 1
		self.C[f_inter] *= 0

		# collocated are set to the coarse node
		self.C[f_inter[::2],c_inter[:]] = 1

		self.C[f_inter[1::2],np.roll(c_inter,-1)] = 1/2
		self.C[f_inter[1::2],c_inter] = 1/2

		# dirichlet
		for dof_id in self.mesh.boundaries:
			self.C[dof_id] *= 0
			self.Id[dof_id] = 1.
			y = self.mesh.dofs[dof_id].y
			self.dirichlet[dof_id] = self.ufunc(y)

		self.true_dofs = list(np.where(self.Id==0)[0])
		self.C_rect = self.C[:,self.true_dofs]
	def solve(self):
		print('virtual not overwritten')

	def vis_constraints(self):
		if self.C is not None:
			vis_constraints(self.C,self.mesh.dofs)
		else:
			print('Constraints have not been set')

	def vis_mesh(self):
		for dof in self.mesh.dofs.values():
			plt.scatter(dof.h,dof.y)
		plt.show()

	def vis_dofs(self):
		frame = [[0,0,1,1,0,0,1],[.5,1,1,0,0,.5,.5]]
		data = []
		for dof in self.mesh.dofs.values():
			blocks = []
			dots = [[dof.x],[dof.y]]
			for e in dof.elements.values():
				blocks.append(e.plot)
			data.append([blocks,dots])

		return animate_2d([frame],data,16)

	def vis_elements(self):
		frame = [[0,0,1,1,0,0,1],[.5,1,1,0,0,.5,.5]]
		data = []
		for e in self.mesh.elements:
			blocks = [e.plot]
			dots = [[],[]]
			for dof in e.dof_list:
				dots[0].append(dof.x)
				dots[1].append(dof.y)
			data.append([blocks,dots])

		return animate_2d([frame],data,16)

	def y_to_e(self,y):
		n_els = [self.N,2*self.N]
		
		y -= (y==1)*1e-14
		fine = True if y >= 0.5+self.h/2 else False

		if fine:
			y_ind = int((y-.5)/self.h-1/2)
		else:
			y_ind = int(y/2/self.h+1/2)
		el_ind = fine*self.mesh.n_coarse_els+y_ind
		e = self.mesh.elements[int(el_ind)]
		assert y >= min(e.plot[1]) and y <= max(e.plot[1])
		return e

	def sol(self, weights=None):

		if weights is None:
			assert self.solved
			weights = self.U

		def solution(y):
			e = self.y_to_e(y)

			val = 0
			for local_id, dof in enumerate(e.dof_list):
				val += weights[dof.ID]*phi1_eval(y,dof.h,dof.y,e.interface)
			
			return val
		return solution

	def error(self,qpn=5):
		uh = self.sol()
		l2_err = 0.
		for e in self.mesh.elements:
			func = lambda y: (self.ufunc(y)-uh(y))**2
			y0,y1 = e.dom
			val = gauss(func,y0,y1,qpn)
			l2_err += val
		return np.sqrt(l2_err)

class Laplace(Solver):
	def __init__(self,N,u,f,qpn=5):
		super().__init__(N,u,f,qpn)

	def solve(self):
		self._build_stiffness()
		self._build_force()
		self._setup_constraints()
		LHS = self.C_rect.T @ self.K @ self.C_rect
		RHS = self.C_rect.T @ (self.F - self.K @ self.dirichlet)
		x = la.solve(LHS,RHS)
		self.U = self.C_rect @ x + self.dirichlet
		self.solved = True


class Projection(Solver):
	def __init__(self,N,u,qpn=5):
		super().__init__(N,u,u,qpn)

	def solve(self):
		self._build_mass()
		self._build_force()
		self._setup_constraints()
		LHS = self.C_rect.T @ self.M @ self.C_rect
		RHS = self.C_rect.T @ (self.F - self.M @ self.dirichlet)
		x = la.solve(LHS,RHS)
		self.U = self.C_rect @ x + self.dirichlet
		self.solved = True
		return x


