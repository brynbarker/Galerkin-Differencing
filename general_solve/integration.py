import numpy as np
import pickle
from general_solve import shape_functions

class Integrator:
	def __init__(self,qpn,dim,ords):#=[3,3]):
		self.qpn = qpn
		self.dim = dim
		self.ords = ords
		self.prod = np.prod([ord+1 for ord in self.ords])

		[p,w] = np.polynomial.legendre.leggauss(qpn)
		self.points = p
		self.weights = w
		self.W = np.array(w)

		if dim == 2:
			chop = self.ords[0]+1
			total = (self.ords[0]+1)*(self.ords[1]+1)
			self.id_map = {ID:[int(ID/chop),ID%chop] for ID in range(total)}
		if dim == 3:
			chop = self.ords[0]+1
			chop2 = self.ords[1]+1
			total = chop*chop2*(self.ords[2]+1)
			self.id_map = {ID:[int(ID/chop)%chop2,ID%chop,int(ID/chop/chop2)] for ID in range(total)}
		self.phi,self.dphi = shape_functions._get_phi_refs(self.ords,self.dim)

		self._compute_quad_bounds()
		self._get_phi_and_dphi_vals()
		self._get_k_vals()

	def _compute_quad_bounds(self):
		if self.dim ==2:
			self.quad_bounds = [[0,.5,0,.5],
				[.5,1,0,.5],[0,.5,.5,1],[.5,1,.5,1]]
		if self.dim ==3:
			self.quad_bounds = [[0,.5,0,.5,0,.5],
				[.5,1,0,.5,0,.5],[0,.5,.5,1,0,.5],[.5,1,.5,1,0,.5],
				[0,.5,0,.5,.5,1],[.5,1,0,.5,.5,1],
				[0,.5,.5,1,.5,1],[.5,1,.5,1,.5,1]]

	def _get_phi_and_dphi_vals(self):
		self.phi_vals = {}
		self.dphi_vals = {}
		for test_id in range(self.prod):
			test_ind = self.id_map[test_id]

			phi_test = lambda x,y: self.phi(x,y,1,test_ind)
			dphi_test = lambda x,y: self.dphi(x,y,1,test_ind)

			self.phi_vals[test_id] = []
			self.dphi_vals[test_id] = []

			for bounds in self.quad_bounds:
				vals = self._evaluate_func_at_points(phi_test,bounds)
				self.phi_vals[test_id].append(vals)

				dvals = self._evaluate_func_at_points(dphi_test,bounds,arr=True)
				self.dphi_vals[test_id].append(dvals)

	def _get_vals(self,k=True):
		lab = 'k' if k else 'm'			
		fname_prefix = '/home/bbb/Code/Galerkin-Differencing/general_solve/pickled/'
		fname = fname_prefix+'{}_vals_p{}{}_qpn{}.pickle'.format(lab,self.ords[0],self.ords[1],self.qpn)
		try:
			with open(fname,'rb') as handle:
				vals = pickle.load(handle)
		except:
			vals = {}
			size = self.prod#4**self.dim
			for id in range(len(self.quad_bounds)):
				local = np.zeros((size,size))
				for i in range(size):
					for j in range(i,size):
						if k:
							val = self._compute_k_product_integral(i,j,id)
						else:
							phi_i = self.phi_vals[i][id]
							phi_j = self.phi_vals[j][id]
							val = self._compute_product_integral(
										phi_i,phi_j,volume=1/2**self.dim)
						local[i,j] = val
						local[j,i] = val
				vals[id] = local
			with open(fname,'wb') as handle:
				pickle.dump(vals,handle,protocol=pickle.HIGHEST_PROTOCOL)
		if k:
			self.k_vals = vals
		else:
			self.m_vals = vals

	def _get_k_vals(self):
		return self._get_vals()
		fname = 'k_vals_p3_qpn{}.pickle'.format(self.qpn)
		try:
			with open(fname,'rb') as handle:
				self.k_vals = pickle.load(handle)
		except:
			self.k_vals = {}
			size = 4**self.dim
			for id in range(len(self.quad_bounds)):
				k_local = np.zeros((size,size))
				for i in range(size):
					for j in range(i,size):
						val = self._compute_k_product_integral(i,j,id)
						k_local[i,j] = val
						k_local[j,i] = val
				self.k_vals[id] = k_local
			with open(fname,'wb') as handle:
				pickle.dump(self.k_vals,handle,protocol=pickle.HIGHEST_PROTOCOL)
			
	def get_k_vals(self):
		try:
			return self.k_vals
		except:
			self._get_k_vals()
			return self.k_vals

	def _get_m_vals(self):
		return self._get_vals(k=False)
		fname = 'm_vals_p3_qpn{}.pickle'.format(self.qpn)
		try:
			with open(fname,'rb') as handle:
				self.m_vals = pickle.load(handle)
		except:
			self.m_vals = {}
			size = 4**self.dim
			for id in range(len(self.quad_bounds)):
				m_local = np.zeros((size,size))
				for i in range(size):
					phi_i = self.phi_vals[i][id]
					for j in range(i,size):
						phi_j = self.phi_vals[j][id]
						val = self._compute_product_integral(phi_i,phi_j,volume=1/2**self.dim)
						m_local[i,j] = val
						m_local[j,i] = val
				self.m_vals[id] = m_local
			with open(fname,'wb') as handle:
				pickle.dump(self.m_vals,handle,protocol=pickle.HIGHEST_PROTOCOL)
			
	def get_m_vals(self):
		try:
			return self.m_vals
		except:
			self._get_m_vals()
			return self.m_vals

	def _evaluate_func_at_points(self,func,bounds,arr=False):
		if self.dim == 2:
			a,b,c,d = bounds
			xmid, ymid = (a+b)/2, (c+d)/2
			xscale, yscale = (b-a)/2, (d-c)/2
			if arr:
				vals = np.zeros((self.qpn,self.qpn,2))
			else:
				vals = np.zeros((self.qpn,self.qpn))
			for j in range(self.qpn):
				for i in range(self.qpn):
					xinput = xscale * self.points[j] + xmid
					yinput = yscale * self.points[i] + ymid
					vals[i,j] = func(xinput,yinput)

		if self.dim == 3:
			a,b,c,d,q,r = bounds
			xmid, ymid, zmid = (a+b)/2, (c+d)/2, (q+r)/2
			xscale, yscale, zscale = (b-a)/2, (d-c)/2, (r-q)/2
			if arr:
				vals = np.zeros((self.qpn,self.qpn,self.qpn,3))
			else:
				vals = np.zeros((self.qpn,self.qpn,self.qpn))
			for j in range(self.qpn):
				for i in range(self.qpn):
					for k in range(self.qpn):
						xinput = xscale * self.points[j] + xmid
						yinput = yscale * self.points[i] + ymid
						zinput = zscale * self.points[k] + zmid
						vals[i,j,k] = func(xinput,yinput,zinput)
		return vals

	def _evaluate_func_on_element(self,func,bounds):
		lens = np.array(bounds[1::2])-np.array(bounds[::2])
		all_vals = []
		for quad in self.quad_bounds:
			quad_bound = []
			for ind,diff in enumerate(lens):
				quad_bound.append(bounds[2*ind]+quad[2*ind]*diff)
				quad_bound.append(bounds[2*ind]+quad[2*ind+1]*diff)
			quad_vals = self._evaluate_func_at_points(func,quad_bound)
			all_vals.append(quad_vals)
		return all_vals

	def _compute_k_product_integral(self,i,j,quad_id):
		vals0 = self.dphi_vals[i][quad_id]
		vals1 = self.dphi_vals[j][quad_id]
		if self.dim == 2:
			n,m,_ = vals0.shape
			prod = np.zeros((n,m))
			for ii in range(n):
				for jj in range(m):
					prod[ii,jj] = vals0[ii,jj] @ vals1[ii,jj]
		if self.dim == 3:
			n,m,p,_ = vals0.shape
			prod = np.zeros((n,m,p))
			for ii in range(n):
				for jj in range(m):
					for kk in range(p):
						prod[ii,jj,kk] = vals0[ii,jj,kk] @ vals1[ii,jj,kk]
		return self._compute_product_integral(prod, volume=1/2**self.dim)

	def _compute_product_integral(self,vals0,vals1=1,volume=1):
		if self.dim == 2:
			scale = volume/4
			return (vals0*vals1) @ self.W @ self.W * scale
		if self.dim == 3:
			scale = volume/8
			return (vals0*vals1) @ self.W @ self.W @ self.W * scale

	def _compute_error_integral(self,vals0,vals1,volume=1):
		if self.dim == 2:
			scale = volume / 4
			return ((vals0-vals1)**2)@ self.W @ self.W * scale
		if self.dim == 3:
			scale = volume / 8
			return ((vals0-vals1)**2)@ self.W @ self.W @ self.W * scale