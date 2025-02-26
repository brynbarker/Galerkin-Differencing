import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la
from matplotlib.animation import FuncAnimation
from IPython.display import HTML

def phi3(x,h):
    if -2*h < x <= -h:
        return (x+3*h)*(x+2*h)*(x+h)/6/h**3
    elif -h < x <= 0:
        return -(x+2*h)*(x+h)*(x-h)/2/h**3
    elif 0 < x <= h: 
        return (x+h)*(x-h)*(x-2*h)/2/h**3
    elif h < x <= 2*h:
        return -(x-h)*(x-2*h)*(x-3*h)/6/h**3
    else:
        return 0

def get_interface_intervals(y,h,seg):
    if seg == 0:
        shifts = [h/4]+[0]*4
        ys = [4/3*y+h/3]+[y]*3
        dys = [4/3,1,1,1]
        return (shifts,ys,dys)
    elif seg == 1:
        shifts = [h/4]*2+[0]*3
        ys = [y-h/4,4/3*y,y,y]
        dys = [1,4/3,1,1]
        return (shifts,ys,dys)
    elif seg == 2:
        shifts = [0]*3+[-h/4]*2
        ys = [y,y,4/3*y,y+h/4]
        dys = [1,1,4/3,1]
        return (shifts,ys,dys)
    elif seg == 3:
        shifts = [0]*4+[-h/4]
        ys = [y]*3+[4/3*y-h/3]
        dys = [1,1,1,4/3]
        return (shifts,ys,dys)
    else:
        return ([0]*5,[y]*4,[1]*4)

def phi3_interface(y,h,seg):
    shft,yI,dyI = get_interface_intervals(y,h,seg)

    if (-2*h+shft[0]) < y <= (-h+shft[1]):
        x = yI[0]
        return (x+3*h)*(x+2*h)*(x+h)/6/h**3
    elif (-h+shft[1]) < y <= shft[2]:
        x = yI[1]
        return -(x+2*h)*(x+h)*(x-h)/2/h**3
    elif shft[2] < y <= (h+shft[3]):
        x = yI[2]
        return (x+h)*(x-h)*(x-2*h)/2/h**3
    elif (h+shft[3]) < y <= (2*h+shft[4]):
        x = yI[3]
        return -(x-h)*(x-2*h)*(x-3*h)/6/h**3
    else:
        return 0

def phi3_dy(x,h):
    if -2*h < x <= -h:
        return (11*h**2+12*h*x+3*x**2)/6/h**3
    elif -h < x <= 0:
        return (h**2-4*h*x-3*x**2)/2/h**3
    elif 0 < x <= h:
        return -(h**2+4*h*x-3*x**2)/2/h**3
    elif h < x <= 2*h:
        return -(11*h**2-12*h*x+3*x**2)/6/h**3
    else:
        return 0

def phi3_interface_dy(y,h,seg):
    shft,yI,dyI = get_interface_intervals(y,h,seg)

    if (-2*h+shft[0]) < y <= (-h+shft[1]):
        x = yI[0]
        return dyI[0]*(11*h**2+12*h*x+3*x**2)/6/h**3
    elif (-h+shft[1]) < y <= shft[2]:
        x = yI[1]
        return dyI[1]*(h**2-4*h*x-3*x**2)/2/h**3
    elif shft[2] < y <= (h+shft[3]):
        x = yI[2]
        return -dyI[2]*(h**2+4*h*x-3*x**2)/2/h**3
    elif (h+shft[3]) < y <= (2*h+shft[4]):
        x = yI[3]
        return -dyI[3]*(11*h**2-12*h*x+3*x**2)/6/h**3
    else:
        return 0

#def phi1(y,h):
#	if -h < y <= 0:
#		return 1+1/h*y
#	elif 0 < y <= h:
#		return 1-1/h*y
#	else:
#		return 0
#	
#def phi1_dy(y,h):
#	if -h < y <= 0:
#		return 1/h
#	elif 0 < y <= h:
#		return -1/h
#	else:
#		return 0
#
#def phi1_interface(y,h):
#	if -3/4*h < y <= 0:
#		return 1+4/3/h*y
#	elif 0 < y <= 3/4*h:
#		return 1-4/3/h*y
#	else::
#		return 0
#		
#def phi1_interface_dy(y,h):
#	if 0 < y <= 3/4*h: 
#		return -4/3/h
#	elif -3/4*h < y <= 0:
#		return 4/3/h
#	else:
#		return 0
		
def phi3_dy_eval(y_in,h,y0,seg=None):
	y = y_in-y0
	if seg is not None:
		return phi3_interface_dy(y,h,seg)
	return phi3_dy(y,h)

def phi3_eval(y_in,h,y0,seg=None):
    y = y_in-y0
    if seg is not None:
        return phi3_interface(y,h,seg)
    return phi3(y,h)

def phi3_ref(y_ref,h,i,I=False):
    y = y_ref*(1+1/3*I)-h*(i-1)
    return phi3(y,h)

def grad_phi3_ref(y_ref,h,i,I=False):
    y = y_ref*(1+1/3*I)-h*(i-1)
    return phi3_dy(y,h)*(1+1/3*I)

#integrators

def gauss(f,a,b,n):
	ymid = (a+b)/2
	yscale = (b-a)/2
	[p,w] = np.polynomial.legendre.leggauss(n)
	integral = 0.
	for i in range(n):
		integral += w[i]*f(yscale*p[i]+ymid)
	return integral*yscale

def local_stiffness_1d(h,qpn=5,I=False):
	K = np.zeros((4,4))

	y0 = 0
	y1 = h

	if I:
		#y0, y1 = 0, 3/4*h
		y0 = h/2
		
	for test_id in range(4):

		grad_phi_test = lambda y: grad_phi3_ref(y,h,test_id)#,I)

		for trial_id in range(test_id,4):

			grad_phi_trial = lambda y: grad_phi3_ref(y,h,trial_id)#,I)

			func = lambda y: grad_phi_trial(y) * grad_phi_test(y)
			val = gauss(func,y0,y1,qpn)

			K[test_id,trial_id] += val
			K[trial_id,test_id] += val * (test_id != trial_id)
	return K

def local_mass_1d(h,qpn=5,I=False):
	M = np.zeros((4,4))

	y0 = 0
	y1 = h

	if I:
		#y0, y1 = 0, 3/4*h
		y0 = h/2
		
	for test_id in range(4):

		phi_test = lambda y: phi3_ref(y,h,test_id)#,I)

		for trial_id in range(test_id,4):

			phi_trial = lambda y: phi3_ref(y,h,trial_id)#,I)

			func = lambda y: phi_trial(y) * phi_test(y)
			val = gauss(func,y0,y1,qpn)

			M[test_id,trial_id] += val
			M[trial_id,test_id] += val * (test_id != trial_id)
	return M


class Node1D:
	def __init__(self,ID,i,y,h):
		self.ID = ID
		self.i = i
		self.y = y
		self.h = h
		self.elements = {}
		self.seg = None

	def add_element(self,e):
		if e.ID not in self.elements.keys():
			if e.interface:
				self.seg = len(self.elements)
			self.elements[e.ID] = e

class Element1D:
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

	def add_dofs(self,strt):
		if len(self.dof_ids) != 0:
			return
		for ii in range(4):
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
		#self.dom[-1] -= self.h/4
		self.interface = True
		self.dom[0] += self.h/2

class Mesh1D:
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
		ydom = np.linspace(0-3*H/2,0.5+3*H/2,int(self.N/2)+4)

		ylen = len(ydom)

		dof_id,e_id = 0,0
		for i,y in enumerate(ydom):
			#if (i >= ylen-2):
			#	y -= H/4
			#if y == ylen-1: y-=H/4
			#interface_element = (i == ylen-3)
			self.dofs[dof_id] = Node1D(dof_id,i,y,H)

			if (-H<y<.5):
				strt = dof_id-1#-xlen
				element = Element1D(e_id,i-1,y,H)
				element.add_dofs(strt)
				self.elements.append(element)
				#if interface_element: element.set_interface()
				e_id += 1

			if y<0:
				self.boundaries.append(dof_id)
			if (.5+H > y > 0.5):
				self.interface[0].append(dof_id)

			dof_id += 1

		self.n_coarse_dofs = dof_id
		self.n_coarse_els = e_id

	def _make_fine(self):
		H = self.h
		ydom = np.linspace(0.5-H/2,1.+3*H/2,self.N+3)

		ylen = len(ydom)

		dof_id,e_id = self.n_coarse_dofs,self.n_coarse_els
		for i,y in enumerate(ydom):
			self.dofs[dof_id] = Node1D(dof_id,i,y,H)
			interface_element = (i == 1)

			if (.5<y<1.):
				strt = dof_id-1#-xlen
				element = Element1D(e_id,i-1,y,H)
				element.add_dofs(strt)
				element.set_fine()
				self.elements.append(element)
				if interface_element: element.set_interface()
				e_id += 1

			if y>1.:# and (0 <= y < 1):#or y==0. or y==1:
				self.boundaries.append(dof_id)
			if (y < 0.5+3*H):
				self.interface[1].append(dof_id)
			dof_id += 1

	def _update_elements(self):
		for e in self.elements:
			e.update_dofs(self.dofs)

class Solver1D:
	def __init__(self,N,u,f=None,qpn=5):
		self.N = N
		self.ufunc = u
		self.ffunc = f #needs to be overwritten 
		self.qpn = qpn

		self.mesh = Mesh1D(N)
		self.h = self.mesh.h

		self.solved = False
		self.C = None
		self.Id = None
		
		self._setup_constraints()

	def _build_force(self,proj=False):
		num_dofs = len(self.mesh.dofs)
		myf = self.ufunc if proj else self.ffunc
		F = np.zeros(num_dofs)

		for e in self.mesh.elements:
			y0 = 0
			y1 = e.h
			if e.interface: y0 = e.h/2#y1 *= 3/4
			for test_id,dof in enumerate(e.dof_list):

				phi_test = lambda y: phi3_ref(y,e.h,test_id)#,e.interface)
				func = lambda y: phi_test(y) * myf(y+e.y)
				val = gauss(func,y0,y1,self.qpn)

				F[dof.ID] += val

		if proj: self.F_proj = F.copy()
		else: self.F = F.copy()

	def _build_stiffness(self):
		num_dofs = len(self.mesh.dofs)
		self.K = np.zeros((num_dofs,num_dofs))

		base_k = local_stiffness_1d(self.h,qpn=self.qpn)
	
		interface_k = local_stiffness_1d(self.h,qpn=self.qpn,I=True)
		for e in self.mesh.elements:
			scale = 1 if e.fine else 1/2
			for test_id,dof in enumerate(e.dof_list):
				if e.interface:
					self.K[dof.ID,e.dof_ids] += interface_k[test_id]
				else:
					self.K[dof.ID,e.dof_ids] += base_k[test_id] * scale

	def _build_mass(self):
		num_dofs = len(self.mesh.dofs)
		self.M = np.zeros((num_dofs,num_dofs))

		base_m = local_mass_1d(self.h,qpn=self.qpn)
	
		interface_m = local_mass_1d(self.h,qpn=self.qpn,I=True)
		for e in self.mesh.elements:
			scale = 1 if e.fine else 2
			for test_id,dof in enumerate(e.dof_list):
				if e.interface:
					self.M[dof.ID,e.dof_ids] += interface_m[test_id]
				else:
					self.M[dof.ID,e.dof_ids] += base_m[test_id] * scale

	def _setup_constraints(self):
		num_dofs = len(self.mesh.dofs)
		self.Id = np.zeros(num_dofs)
		self.C_full = np.eye(num_dofs)
		self.dirichlet = np.zeros(num_dofs)

		c_inter,f_inter = self.mesh.interface
		self.Id[f_inter[1]] = 1
		self.C_full[f_inter[1]] *= 0

		v1, v3 = phi3(1/2,1), phi3(3/2,1)
		vals = [1/v1, -v3/v1, -1,-v3/v1]
		locs = [c_inter[0],f_inter[0],f_inter[2],f_inter[3]]

		for v,loc in zip(vals,locs):
			self.C_full[f_inter[1],loc] = v

		## collocated are set to the coarse node
		#self.C_full[f_inter[::2],c_inter[:]] = 1

		#self.C_full[f_inter[1::2],np.roll(c_inter,-1)] = 1/2
		#self.C_full[f_inter[1::2],c_inter] = 1/2

		# dirichlet
		for dof_id in self.mesh.boundaries:
			self.C_full[dof_id] *= 0
			self.Id[dof_id] = 1.
			y = self.mesh.dofs[dof_id].y
			self.dirichlet[dof_id] = self.ufunc(y)

		self.true_dofs = list(np.where(self.Id==0)[0])
		self.C = self.C_full[:,self.true_dofs]
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
		fine = True if y >= 0.5+self.h else False

		if fine:
			y_ind = int((y-.5)/self.h-1/2)
		else:
			y_ind = int(y/2/self.h+1/2)
		el_ind = fine*self.mesh.n_coarse_els+y_ind
		e = self.mesh.elements[int(el_ind)]
		assert y >= min(e.dom) and y <= max(e.dom)
		return e

	def sol(self, weights=None,proj=False):

		if weights is None:
			assert self.solved
			weights = self.U_proj if proj else self.U_lap

		def solution(y):
			e = self.y_to_e(y)

			val = 0
			for local_id, dof in enumerate(e.dof_list):
				val += weights[dof.ID]*phi3_eval(y,dof.h,dof.y)#,dof.seg)
			
			return val
		return solution

	def error(self,qpn=5,proj=False):
		uh = self.sol(proj=proj)
		l2_err = 0.
		for e in self.mesh.elements:
			func = lambda y: (self.ufunc(y)-uh(y))**2
			y0,y1 = e.dom
			val = gauss(func,y0,y1,qpn)
			l2_err += val
		return np.sqrt(l2_err)

	def laplacian(self):
		self._build_stiffness()
		self._build_force()
		self._setup_constraints()
		LHS = self.C.T @ self.K @ self.C
		RHS = self.C.T @ (self.F - self.K @ self.dirichlet)
		x = la.solve(LHS,RHS)
		self.U_lap = self.C @ x + self.dirichlet
		self.solved = True
		return x


	def projection(self):
		self._build_mass()
		self._build_force(proj=True)
		self._setup_constraints()
		LHS = self.C.T @ self.M @ self.C
		RHS = self.C.T @ (self.F_proj - self.M @ self.dirichlet)
		x = la.solve(LHS,RHS)
		self.U_proj = self.C @ x + self.dirichlet
		self.solved = True
		#return x


