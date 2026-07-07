import numpy as np
import os, pickle
import matplotlib.pyplot as plt
from general_solve.refinement import UniformRefinement
from general_solve.refinement import StripeRefinement
from general_solve.refinement import SquareRefinement
from general_solve.patch import Patch
from general_solve.element import TrapElement, TriElement
from general_solve.barycentric_quadrature import get_quad_pts as bary_quad

from general_solve import globals

trap_quad_id_to_bnds = {
	0: [0,1,0,.5], 1: [0,.5,0,1], 2: [0,1,.5,1], 3: [.5,1,0,1]
}

class Empty:
	def __init__(self):
		self.get_lookup_vals = lambda *args: None
		self.phi_vals = None

def get_quad_eval_locs(qpn,trap_quad_ids):
	[pts,w] = np.polynomial.legendre.leggauss(qpn)
	wts,d_pts = np.array(w),{}
	for trap_quad_id in trap_quad_ids:
		a,b,c,d = trap_quad_id_to_bnds[trap_quad_id]
		xmid, ymid = (a+b)/2, (c+d)/2
		xscale, yscale = (b-a)/2, (d-c)/2
		locs = np.zeros((qpn,qpn,2))
		for j in range(qpn):
			for i in range(qpn):
				xinput = xscale * pts[j] + xmid
				yinput = yscale * pts[i] + ymid
				locs[i,j,:] = [xinput,yinput]
		d_pts[trap_quad_id] = locs
			# locs.append((xinput+xshft,yinput+yshft))
	return wts,d_pts


def get_tri_vals_at_points(func,qpn,pts,map=None,shp=[]):
	if map is None:
		map = lambda a,b:[a,b]

	# wts,alphas,betas = bary_quad(qpn)
	alphas, betas = pts[0]
	val_shape = [qpn]+shp
	vals = np.zeros(val_shape)

	for j in range(len(alphas)):
		xi,eta = map(alphas[j],betas[j])
		vals[j] = func(xi,eta)

	return {0:vals}

def get_trap_vals_at_points(func,qpn,d_pts,shp=[],map=None):
	if map is None:
		map = lambda a,b:[a,b]

	val_shape = [qpn,qpn]+shp
	vals = {}
	
	if len(shp) == 1 and shp[0] == 3:
		shp
	for quad_id in d_pts:
		q_vals = np.zeros(val_shape)
		for j in range(qpn):
			for i in range(qpn):
				xinput,yinput = d_pts[quad_id][i,j]
				xi,eta = map(xinput,yinput)
				tmp = func(xi,eta)
				q_vals[i,j] = func(xi,eta)
		vals[quad_id] = q_vals

	return vals

def get_vals_at_points(tri,func,qpn,pts,shp=[],map=None):
	if tri:
		return get_tri_vals_at_points(func,qpn,pts,map=map,shp=shp)
	else:
		return get_trap_vals_at_points(func,qpn,pts,map=map,shp=shp)

def phi_input_trap(nu,rho,map_type):
	ind0,ind1 = map_type%2,map_type%2==0
	vars = [nu,rho]
	v0,v1 = vars[ind0],vars[ind1]
	
	if map_type < 2:
		c_main = 0
		a,b,c,d = 0,1,2,-1
		den = [2,1]
	else:
		c_main = 1
		a,b,c,d = 1,-1,1,1
		den = [3,-1]

	main = (v0+c_main)/2
	other = (a+b*v0+c*v1+d*v0*v1)/(den[0]+den[1]*v0)

	stars = [main,other]
	xi_star = stars[ind0]
	eta_star = stars[ind1]
	return xi_star,eta_star

def phi_input_tri(alpha,beta,map_type):
	tmp = (alpha+beta)/2
	main = tmp if map_type<2 else 1-tmp
	other = alpha/(2+alpha+beta)-beta/(2+beta)

	xi_ind,eta_ind = map_type%2,map_type%2==0
	vars = [main,other]
	xi_star = vars[xi_ind]
	eta_star = vars[eta_ind]
	return xi_star,eta_star


class InterfaceMapping():
	def __init__(self,mesh,integrator,ords):
		self.mesh = mesh
		self.integrator = integrator
		self.refinement = mesh.refinement
		self.patches = mesh.patches
		self.ords = ords
		self.qpn = integrator.qpn

		self.rindex = mesh.rindex
		self.sides = {}

		self.k_vals = {0:None,1:None}
		self.m_vals = {0:None,1:None}

		self.trapezoid_elements = mesh.patches[0].zigzag[0]
		self.triangle_elements = mesh.patches[0].zigzag[1]

		H = mesh.h

		Jd_trap = {0:lambda nu,rho: 1/H*np.array([[1,(4-8*rho)/(2+nu)**2+(2*rho-1)/(2+nu)],[0,2/(2+nu)]]),
				   2:lambda nu,rho: 1/H*np.array([[1,(-4+8*rho)/(3-nu)**2+(1-2*rho)/(3-nu)],[0,2/(3-nu)]]),
				   1:lambda nu,rho: 1/H*np.array([[2/(2+rho),0],[(2*nu-1)/(2+rho)+(4-8*nu)/(2+rho)**2,1]]),
				   3:lambda nu,rho: 1/H*np.array([[2/(3-rho),0],[(1-2*nu)/(3-rho)+(-4+8*nu)/(3-rho)**2,1]])}
		Jd_trap_coefs = {
			0:lambda nu,rho: 1/H**2*np.array([1,(4-8*rho)/(2+nu)**2+(2*rho-1)/(2+nu),4/(2+nu)**2+((4-8*rho)/(2+nu)**2+(2*rho-1)/(2+nu))**2]),
			2:lambda nu,rho: 1/H**2*np.array([1,(-4+8*rho)/(3-nu)**2+(1-2*rho)/(3-nu),4/(3-nu)**2+((-4+8*rho)/(3-nu)**2+(1-2*rho)/(3-nu))**2]),
			1:lambda nu,rho: 1/H**2*np.array([4/(2+rho)**2+((4-8*rho)/(2+nu)**2+(2*rho-1)/(2+nu))**2,(2*nu-1)/(2+rho)+(4-8*nu)/(2+rho)**2,1]),
			3:lambda nu,rho: 1/H**2*np.array([4/(3-rho)**2+((-4+8*rho)/(3-nu)**2+(1-2*rho)/(3-nu))**2,(1-2*nu)/(3-rho)+(-4+8*nu)/(3-rho)**2,1])}
		Jt_trap_det = {0:lambda nu,rho: H**2*(1/2-nu/4),
				       2:lambda nu,rho: H**2*(1/4+nu/4),
				       1:lambda nu,rho: H**2*(1/2-rho/4),
				       3:lambda nu,rho: H**2*(1/4+rho/4)}

		A = lambda a,b: (2+b)/(2+a+b)**2
		B = lambda a,b: -a/(2+a+b)**2-2/(2+b)**2

		Jt_tri_det = {0:lambda *args: -H**2/4,2:lambda *args: H**2/4,
					  1:lambda *args: H**2/4,3:lambda *args: -H**2/4}
		Jd_tri = {0:lambda a,b: 1/H*np.array([[1,A(a,b)+B(a,b)],[0,2*(A(a,b)-B(a,b))]]),
				  2:lambda a,b: 1/H*np.array([[1,-(A(a,b)+B(a,b))],[0,2*(A(a,b)-B(a,b))]]),
				  1:lambda a,b: 1/H*np.array([[2*(A(a,b)-B(a,b)),0],[A(a,b)+B(a,b),1]]),
				  3:lambda a,b: 1/H*np.array([[2*(A(a,b)-B(a,b)),0],[-(A(a,b)+B(a,b)),1]])}
		Jd_tri_coefs = {
			0:lambda a,b: 1/H**2*np.array([1,A(a,b)+B(a,b),3*(A(a,b)**2+B(a,b)**2)-2*A(a,b)*B(a,b)]),
			2:lambda a,b: 1/H**2*np.array([1,-(A(a,b)+B(a,b)),3*(A(a,b)**2+B(a,b)**2)-2*A(a,b)*B(a,b)]),
			1:lambda a,b: 1/H**2*np.array([3*(A(a,b)**2+B(a,b)**2)-2*A(a,b)*B(a,b),A(a,b)+B(a,b),1]),
			3:lambda a,b: 1/H**2*np.array([3*(A(a,b)**2+B(a,b)**2)-2*A(a,b)*B(a,b),-(A(a,b)+B(a,b)),1])}

		for trap_el in self.trapezoid_elements:
			trap_el.set_jacobian(
				Jd_trap[trap_el.map_type],Jt_trap_det[trap_el.map_type],
				Jd_trap_coefs[trap_el.map_type])
		for tri_el in self.triangle_elements:
			tri_el.set_jacobian(
				Jd_tri[tri_el.map_type],Jt_tri_det[tri_el.map_type],
				Jd_tri_coefs[tri_el.map_type])

		self.Jds = [Jd_trap,Jd_tri]
		self.Jd_coefs = [Jd_trap_coefs,Jd_tri_coefs]
		self.Jt_dets = [Jt_trap_det,Jt_tri_det]


		# def wrap(J):
		# 	A = J[0,0]**2+J[1,0]**2
		# 	B = J[0,0]*J[0,1]+J[1,0]*J[1,1]
		# 	C = J[1,1]**2+J[0,1]**2
		# 	return np.array([A,B,C])


		# def get_vals_at_points(func,points,arr=False,map=None):
		# 	phi_eval = map is not None
		# 	if map is None:
		# 		map = lambda a,b:[a,b]
		# 	qpn,m,dm = points.shape
		# 	assert qpn == m

		# 	if arr and phi_eval:
		# 		vals = np.zeros((qpn,qpn,2))
		# 	elif arr:
		# 		vals = [np.zeros((qpn,qpn,2,2)),np.zeros((qpn,qpn,3))]
		# 	else:
		# 		vals = np.zeros((qpn,qpn))
		# 	for j in range(qpn):
		# 		for i in range(qpn):
		# 			xinput,yinput = points[i,j]
		# 			if arr and phi_eval:
		# 				xi,eta = map(xinput,yinput)
		# 				vals[i,j,:] = func(xi,eta)
		# 			elif arr:
		# 				J = func(xinput,yinput)
		# 				vals[0][i,j] = J
		# 				vals[1][i,j] = wrap(J)
		# 			else:
		# 				xi,eta = map(xinput,yinput)
		# 				vals[i,j] = func(xi,eta)
		# 				# vals[i,j] = func(xinput,yinput)

		# 	return vals


		if self.refinement.yside:
			chop = self.ords[0]+2
			self.tri_prod = (self.ords[0]+2)*(self.ords[1]+1)
			quad_comp = 0
			lows = [False,False,False,True]
			highs = [False,True,False,False]
			both = [False,True,False,True]
		else:
			chop = self.ords[0]+1
			self.tri_prod = (self.ords[1]+2)*(self.ords[0]+1)
			quad_comp = 1
			lows = [False,False,True,False]
			highs = [True,False,False,False]
			both = [True,False,True,False]
		if self.rindex == 2:
			if self.refinement.type == 0:
				low_high = [.25,.75-H]
			else:
				low_high = [.75,.25-H]
		else:
			low_high = [0,1-H]
		tri_id_map = {ID:[int(ID/chop),ID%chop] for ID in range(self.tri_prod)}

		self.trap_wts,d_trap_pts = get_quad_eval_locs(self.qpn,[0,1,2,3])
		tri_wts,self.tri_pts = bary_quad(min(self.qpn-1,6))
		self.tri_wts = np.array(tri_wts)
		# self.tri_qpn = len(tri_wtsts)

		my_ops = [j for j in range(4) if both[j]]
		self.my_pts = [d_trap_pts,{0:self.tri_pts}]

		get_pars = [(self.integrator.prod,self.integrator.id_map,phi_input_trap),
			  		(self.tri_prod,tri_id_map,phi_input_tri)]

		self.phi_vals = [{},{}]
		self.dphi_vals = [{},{}]
		self.Jds_eval = [{},{}]
		self.J_dets_eval = [{},{}]
		self.Jd_coefs_eval = [{},{}] 

		for map_type in range(4):  # element type map
			for shape in range(2):
				dets = get_vals_at_points(
					shape,self.Jt_dets[shape][map_type],self.qpn,
					self.my_pts[shape])
				self.J_dets_eval[shape][map_type] = dets

				jds = get_vals_at_points(
					shape,self.Jds[shape][map_type],self.qpn,
					self.my_pts[shape],shp=[2,2])
				self.Jds_eval[shape][map_type] = jds

				jd_coefs = get_vals_at_points(
					shape,self.Jd_coefs[shape][map_type],self.qpn,
					self.my_pts[shape],shp=[3])
				self.Jd_coefs_eval[shape][map_type] = jd_coefs

				self.phi_vals[shape][map_type] = {q_id:[] for q_id in self.my_pts[shape]}
				self.dphi_vals[shape][map_type] = {q_id:[] for q_id in self.my_pts[shape]}
				my_prod,my_id_map,my_p_map = get_pars[shape]
				my_map = lambda a,b: my_p_map(a,b,map_type)

				phi_tmp,dphi_tmp = [],[]

				for test_id in range(my_prod):
					test_ind = my_id_map[test_id]
					phi_test = lambda x,y: self.integrator.phi(x,y,1,test_ind)
					dphi_test = lambda x,y: self.integrator.dphi(x,y,1,test_ind)
					phi_tmp.append(get_vals_at_points(
						shape,phi_test,self.qpn,self.my_pts[shape],shp=[],map=my_map))
					dphi_tmp.append(get_vals_at_points(
						shape,dphi_test,self.qpn,self.my_pts[shape],shp=[2],map=my_map))

				for phi_id,dphi_id in zip(phi_tmp,dphi_tmp):
					for q_id in self.my_pts[shape]:
						self.phi_vals[shape][map_type][q_id].append(phi_id[q_id])
						self.dphi_vals[shape][map_type][q_id].append(dphi_id[q_id])


		for i,els in enumerate([self.trapezoid_elements,self.triangle_elements]):
			for el in els:
				el.set_jacobian_eval(
					self.J_dets_eval[i],self.Jds_eval[i],self.Jd_coefs_eval[i])
				quad_check = el.h*[el.K,el.L][quad_comp]
				if el.tri:
					el.set_support([True]+[False]*3)
				elif quad_check < low_high[0]:
					el.set_support(lows)
				elif quad_check > low_high[1]:
					el.set_support(highs)
				else:
					el.set_support(both)

	def _get_vals(self,k=True,tri=False):
		lab = 'k' if k else 'm'
		shp = 'tri' if tri else 'trap'
		ord_string = '{}{}'.format(self.ords[0],self.ords[1])
		fpath = os.path.join(os.path.dirname(os.getcwd()),'pickled/')
		if globals.LAG:
			fname = fpath+'{}_{}_vals_p{}_qpn{}.pickle'.format(shp,lab,ord_string,self.qpn)
		else:
			fname = fpath+'{}_{}_spline_vals_p{}_qpn{}.pickle'.format(shp,lab,ord_string,self.qpn)

		try:
			with open(fname,'rb') as handle:
				vals = pickle.load(handle)
		except:
			vals = {}
			max_quad = 1 if tri else 4
			size = self.tri_prod if tri else self.integrator.prod
			for map_type in range(4):
				map_vals = {}
				for quad_id in range(max_quad):
					local = np.zeros((size,size))
					for i in range(size):
						for j in range(i,size):
							val = self._get_product_integral(
								k,tri,i,j,map_type,quad_id)
							local[i,j] = val
							local[j,i] = val

					map_vals[quad_id] = local
				vals[map_type] = map_vals
				
			with open(fname,'wb') as handle:
				pickle.dump(vals,handle,protocol=pickle.HIGHEST_PROTOCOL)
		if k:
			self.k_vals[tri] = vals
		else:
			self.m_vals[tri] = vals
	# def _get_transformed_vals(self,k=True):
	# 	lab = 'k' if k else 'm'
	# 	qpn = self.integrator.qpn

	# 	ord_string = '{}{}'.format(self.ords[0],self.ords[1])
	# 	fpath = os.path.join(os.path.dirname(os.getcwd()),'pickled/')
	# 	if globals.LAG:
	# 		fname = fpath+'zigzag_{}_vals_p{}_qpn{}.pickle'.format(lab,ord_string,qpn)
	# 	else:
	# 		fname = fpath+'zigzag_{}_spline_vals_p{}_qpn{}.pickle'.format(lab,ord_string,qpn)

	# 	try:
	# 		with open(fname,'rb') as handle:
	# 			vals = pickle.load(handle)
	# 	except:
	# 		xside_d = {0:[-1,0],1:[0,0],2:[1,0],3:[-1,1],4:[0,1],5:[1,1]}
	# 		yside_d = {0:[0,-1],1:[1,-1],2:[0,0],3:[1,0],4:[0,1],5:[1,1]}
	# 		vals = {}
	# 		size = self.integrator.prod
	# 		for shape in [False,True]:
	# 			shape_vals = {}
	# 			J_vals = self.J_tri_vals if shape else self.J_trap_vals
	# 			J_dets = self.J_tri_dets if shape else self.J_trap_dets
	# 			for map_type in range(4):
	# 				map_vals = {}
	# 				for quad_id in range(4):
	# 					local = np.zeros((size,size))
	# 					jac_vals = J_vals[map_type][quad_id][1]
	# 					j_det = J_dets[map_type][quad_id]
	# 					for i in range(size):
	# 						for j in range(i,size):
	# 							if k:
	# 								val = self.integrator._compute_k_product_integral(i,j,quad_id,jac=jac_vals,jdet_inv=1/j_det)
	# 							else:
	# 								phi_i = self.integrator.phi_vals[quad_id][i]
	# 								phi_j = self.integrator.phi_vals[quad_id][j]
	# 								val = self.integrator._compute_product_integral(
	# 											phi_i,phi_j,volume=1/2**self.dim,jdet=j_det)
	# 							local[i,j] = val
	# 							local[j,i] = val

	# 					map_vals[quad_id] = local
	# 				shape_vals[map_type] = map_vals	
	# 			vals[shape] = shape_vals
				
	# 		with open(fname,'wb') as handle:
	# 			pickle.dump(vals,handle,protocol=pickle.HIGHEST_PROTOCOL)
	# 	if k:
	# 		self.k_vals = vals
	# 	else:
	# 		self.m_vals = vals


	def get_lookup_vals(self,k=True):
		if k:
			for tri in [False,True]:
				if self.k_vals[tri] is None:
					self._get_vals(k,tri)
			return self.k_vals
		else:
			for tri in [False,True]:
				if self.m_vals[tri] is None:
					self._get_vals(k,tri)
			return self.m_vals

	def _get_product_integral(self,k,tri,i,j,map_type,q_id):
		d_phi = self.dphi_vals if k else self.phi_vals
		vals0 = d_phi[tri][map_type][q_id][i]
		vals1 = d_phi[tri][map_type][q_id][j]

		m0 = 1 if tri else self.qpn
		m1 = len(self.tri_wts) if tri else self.qpn
		jdet = self.J_dets_eval[tri][map_type][q_id]

		if k:
			jac = self.Jd_coefs_eval[tri][map_type][q_id]
			prod = np.zeros((m0,m1))
			for ii in range(m0):
				for jj in range(m1):
					ixi,ieta = vals0[jj] if tri else vals0[ii,jj]
					jxi,jeta = vals1[jj] if tri else vals1[ii,jj]
					A,C,B = jac[jj] if tri else jac[ii,jj]
					dj = jdet[jj] if tri else jdet[ii,jj]
					prod[ii,jj] = dj*(A*(ixi*jxi)+C*(ieta*jeta)+B*(ixi*jeta+ieta*jxi))
		else:
			prod = vals0 * vals1 * jdet

		return self._compute_product_integral(tri,prod)

 
	def evaluate_func_on_element(self,e,func):
		f_vals = get_vals_at_points(
			e.tri,func,self.qpn,self.my_pts[e.tri],
			map=e.transform)

		return f_vals

	def compute_func_integral(self,el,phi_id,f_val,q_id):
		phi_val = self.phi_vals[el.tri][el.map_type][q_id][phi_id]
		jdet = self.J_dets_eval[el.tri][el.map_type][q_id]
		prod = phi_val*f_val*jdet
		return self._compute_product_integral(el.tri,prod)

	def compute_error_integral(self,el,vals0,vals1,q_id,p=2,prev_val=0):
		diff = abs(vals0-vals1)
		jdet = self.J_dets_eval[el.tri][el.map_type][q_id]
		if p == "inf":
			new_max = max(diff.flatten())
			return max(prev_val,new_max)

		integrand = diff**p * jdet
		new_val = self._compute_product_integral(el.tri,integrand)
		return prev_val + new_val

	def _compute_product_integral(self,tri,prod):
		if tri:
			return prod @ self.tri_wts
		else:
			return prod @ self.trap_wts @ self.trap_wts / 8