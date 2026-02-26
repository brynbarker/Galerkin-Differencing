import numpy as np
from scipy import sparse

#from general_solve import shape_functions

#def gauss(f,a,b,c,d,qpn):
#	xmid, ymid = (a+b)/2, (c+d)/2
#	xscale, yscale = (b-a)/2, (d-c)/2
#	[p,w] = np.polynomial.legendre.leggauss(qpn)
#	outer = 0.
#	for j in range(qpn):
#		inner = 0.
#		for i in range(qpn):
#			inner += w[i]*f(xscale*p[j]+xmid,yscale*p[i]+ymid)
#		outer += w[j]*inner
#	return outer*xscale*yscale

#def local_stiffness(h,qpn,quadbounds,ords=[3,3]):
#	x0,x1,y0,y1 = np.array(quadbounds)*h
#	sizes = [ord+1 for ord in ords]
#	full_size = np.prod(sizes)
#	K = np.zeros((full_size,full_size))
#	id_to_ind = {ID:[int(ID/sizes[0]),ID%sizes[0]] for ID in range(full_size)}
#
#	for test_id in range(full_size):
#
#		test_ind = id_to_ind[test_id]
#		grad_phi_test = lambda x,y: shape_functions.dphi_2d_ref(ords,x,y,h,test_ind)
#
#		for trial_id in range(test_id,full_size):
#
#			trial_ind = id_to_ind[trial_id]
#			grad_phi_trial = lambda x,y: shape_functions.dphi_2d_ref(ords,x,y,h,trial_ind)
#
#			func = lambda x,y: grad_phi_trial(x,y) @ grad_phi_test(x,y)
#			val = gauss(func,x0,x1,y0,y1,qpn)
#
#			K[test_id,trial_id] += val
#			K[trial_id,test_id] += val * (test_id != trial_id)
#	return K

#def gauss(f,a,b,c,d,qpn):
#	xmid, ymid = (a+b)/2, (c+d)/2
#	xscale, yscale = (b-a)/2, (d-c)/2
#	[p,w] = np.polynomial.legendre.leggauss(qpn)
#	outer = 0.
#	for j in range(qpn):
#		inner = 0.
#		for i in range(qpn):
#			inner += w[i]*f(xscale*p[j]+xmid,yscale*p[i]+ymid)
#		outer += w[j]*inner
#	return outer*xscale*yscale

class SimpleSolver:
	def __init__(self,mesh,integrator):
		self.mesh = mesh
		self.integrator = integrator
		self.dim = mesh.dim

		self.lookup = None # needs to be overwritten
		self.blocks = []

		self.spA = None
		self.U = None
		self.F = None
		self.err = None

	def _get_blocks(self):
		if len(self.blocks) == self.dim:
			return 

		if self.lookup is None:
			print('operator quantities not specified')
			return

		self.As = []

		for patch in self.mesh.patches:
			size = len(patch.dofs)
			Ar, Ac, Ad = [],[],[]

			for e in patch.elements.values():
				for test_id,dof in enumerate(e.dof_list):
					for id,quad in enumerate(e.quads):
						if quad:
							Ar += [dof.ID]*len(e.dof_ids)
							Ac += e.dof_ids
							Ad += list(self.lookup[id][test_id])
			spA = sparse.coo_array((Ad,(Ar,Ac)),shape=(size,size))
			self.As.append(spA)

	def _build_system(self,scale0=1,scale1=1):
		if self.spA is not None:
			return 

		self._get_blocks()

		self.spA = sparse.bmat(np.array(
			[[self.As[0]*scale0,None],
			 [None,self.As[1]*scale1]]),format='csc')

	def _build_force(self,ffunc):
		if self.F is not None:
			return 

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

	def set_sys_to_check(self,sys):
		self.cTkc = sys


class LaplaceOperator(SimpleSolver):
	def __init__(self,mesh,integrator,mu):
		super().__init__(mesh,integrator)
		self.mu = mu

		self.lookup = integrator.get_k_vals()

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

class Helmholtz(SimpleSolver):
	def __init__(self,mesh,integrator,k=1):
		super().__init__(mesh,integrator)
		self.k = k
		scales = [p.h**self.mesh.dim for p in self.mesh.patches]
		self.scale0 = scales[0]
		self.scale1 = scales[1]

		lookup_m = integrator.get_m_vals()
		lookup_k = integrator.get_k_vals()

	def _build_system(self):
		super()._build_system(scale0=self.scale0,scale1=self.scale1)


	def _build_system(self):
		super()._build_system(scale0=self.mu,scale1=self.mu)

	def _build_force(self, ffunc):
		super()._build_force(ffunc)
		self.F *= -1