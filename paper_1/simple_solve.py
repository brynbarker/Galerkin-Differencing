import numpy as np
from scipy import sparse

from paper_1 import shape_functions

def gauss(f,a,b,c,d,qpn):
	xmid, ymid = (a+b)/2, (c+d)/2
	xscale, yscale = (b-a)/2, (d-c)/2
	[p,w] = np.polynomial.legendre.leggauss(qpn)
	outer = 0.
	for j in range(qpn):
		inner = 0.
		for i in range(qpn):
			inner += w[i]*f(xscale*p[j]+xmid,yscale*p[i]+ymid)
		outer += w[j]*inner
	return outer*xscale*yscale

def local_stiffness(h,qpn,quadbounds):
	x0,x1,y0,y1 = np.array(quadbounds)*h
	K = np.zeros((16,16))
	id_to_ind = {ID:[int(ID/4),ID%4] for ID in range(16)}

	for test_id in range(16):

		test_ind = id_to_ind[test_id]
		grad_phi_test = lambda x,y: shape_functions.dphi3_2d_ref(x,y,h,test_ind)

		for trial_id in range(test_id,16):

			trial_ind = id_to_ind[trial_id]
			grad_phi_trial = lambda x,y: shape_functions.dphi3_2d_ref(x,y,h,trial_ind)

			func = lambda x,y: grad_phi_trial(x,y) @ grad_phi_test(x,y)
			val = gauss(func,x0,x1,y0,y1,qpn)

			K[test_id,trial_id] += val
			K[trial_id,test_id] += val * (test_id != trial_id)
	return K

def gauss(f,a,b,c,d,qpn):
	xmid, ymid = (a+b)/2, (c+d)/2
	xscale, yscale = (b-a)/2, (d-c)/2
	[p,w] = np.polynomial.legendre.leggauss(qpn)
	outer = 0.
	for j in range(qpn):
		inner = 0.
		for i in range(qpn):
			inner += w[i]*f(xscale*p[j]+xmid,yscale*p[i]+ymid)
		outer += w[j]*inner
	return outer*xscale*yscale

class SimpleSolver:
	def __init__(self,mesh,integrator):
		self.mesh = mesh
		self.integrator = integrator
		self.dim = mesh.dim

		self.lookup = None # needs to be overwritten
		self.blocks = []

		self.U = None
		self.err = None

	def _get_blocks(self):
		if len(self.blocks) == self.dim:
			return 

		self.Bs = []

		for patch in self.mesh.patches:
			n = len(patch.dofs)
			Br, Bc, Bd = [],[],[]

			for e in patch.elements.values():
				for test_id,dof in enumerate(e.dof_list):
					for id,quad in enumerate(e.quads):
						if quad:
							Br += [dof.ID]*len(e.dof_ids)
							Bc += e.dof_ids
							Bd += list(self.lookup[id][test_id])
			spB = sparse.coo_array((Bd,(Br,Bc)),shape=(n,n))
			self.Bs.append(spB)

	def _build_system(self,scale0=1,scale1=1):
		self._get_blocks()

		self.spB = sparse.bmat(np.array(
			[[self.Bs[0]*scale0,None],
			 [None,self.Bs[1]*scale1]]),format='csc')

	def _build_force(self,ffunc):
		myFs = []

		for patch in self.mesh.patches:
			num_dofs = len(patch.dofs)
			F = np.zeros(num_dofs)

			for e in patch.elements.values():
				vol = (e.h/2)**self.dim
				for test_id,dof in enumerate(e.dof_list):
					phi_vals = self.integrator.phi_vals[test_id]
					fvals = self.integrator._evaluate_func_on_element(ffunc,e.bounds)
					for (quad,phi_val,f_val) in zip(e.quads,phi_vals,fvals):
						if quad:
							val = self.integrator._compute_product_integral(phi_val,f_val,vol)
							F[dof.ID] += val
			myFs.append(F)

		self.F = np.hstack(myFs)

	def set_U(self,U):
		self.U = U

	def set_error(self,err,Linf=False):
		if Linf:
			self.Linf_err = err
		else:
			self.err = err


class LaplaceOperator(SimpleSolver):
	def __init__(self,mesh,integrator,mu):
		super().__init__(mesh,integrator)
		self.mu = mu

		self.lookup = integrator.get_k_vals()

	#def _get_blocks(self):
	#	print('in here')
	#	quad_bounds = [[0,.5,0,.5],
	#			[.5,1,0,.5],[0,.5,.5,1],[.5,1,.5,1]]

	#	self.Bs = []

	#	for patch in self.mesh.patches:
	#		local_ks = []
	#		for quadbound in quad_bounds:
	#			local_k = local_stiffness(patch.h,self.integrator.qpn,quadbound)
	#			local_ks.append(local_k)
	#		n = len(patch.dofs)
	#		Br, Bc, Bd = [],[],[]

	#		for e in patch.elements.values():
	#			for test_id,dof in enumerate(e.dof_list):
	#				for id,quad in enumerate(e.quads):
	#					if quad:
	#						Br += [dof.ID]*len(e.dof_ids)
	#						Bc += e.dof_ids
	#						Bd += list(local_ks[id][test_id])#self.lookup[id][test_id])
	#						assert(np.linalg.norm(local_ks[id][test_id]-self.lookup[id][test_id]) < 1e-14)
	#		spB = sparse.coo_array((Bd,(Br,Bc)),shape=(n,n))
	#		self.Bs.append(spB)

	def _build_system(self):
		super()._build_system(scale0=self.mu,scale1=self.mu)

	def _build_force(self, ffunc):
		super()._build_force(ffunc)
		self.F *= -1

class ProjectionOperator(SimpleSolver):
	def __init__(self,mesh,integrator):
		super().__init__(mesh,integrator)
		scales = [p.h**self.mesh.dim for p in self.mesh.patches]
		self.scale0 = scales[0]
		self.scale1 = scales[1]

		self.lookup = integrator.get_m_vals()

	def _build_system(self):
		super()._build_system(scale0=self.scale0,scale1=self.scale1)
