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

		self.rindex = mesh.rindex
		self.sides = {}
		# self.zigzag_ghost_fills = []

		# extra_ops = {.25:[],.75:[]}
		# self.extras = {}
		self.k_vals = None
		self.m_vals = None

		# self.trapezoids = {}
		# self.triangles = {}
		self.trapezoid_elements = mesh.patches[0].zigzag[0]
		self.triangle_elements = mesh.patches[0].zigzag[1]

		# all_clear = self.rindex == 0
		# if all_clear: return
		# if self.rindex == 1: #stripe
		# 	rdim = int(self.refinement.rtype/2)
		# 	if self.refinement.xside and rdim == 1:
		# 		all_clear = True
		# 	elif self.refinement.yside and rdim == 0:
		# 		all_clear = True

		# if self.rindex == 2:
		# 	if self.refinement.xside or self.refinement.yside:
		# 		all_clear = False

		# L,R = int((max(ords)-1)/2),int(max(ords)/2)
		H = mesh.h
		# if self.refinement.rtype % 2 == 0:
		# 	first,second = .25,.75
		# else:
		# 	first,second = .75,.25
		# if R >= 1:
		# 	for r in range(R):
		# 		extra_ops[first].append(first-H*(r+1))
		# 		extra_ops[second].append(second-H*(r+2))
		# if L >= 1:
		# 	for ell in range(L):
		# 		extra_ops[first].append(first+H*(ell+2))
		# 		extra_ops[second].append(second+H*(ell+1))

		# for s in [.25,.75]:
		# 	self.extras[s] = {pos:[] for pos in extra_ops[s]}

		# if all_clear:
		# 	return

		# comp = 0 if self.refinement.xside else 1
		# self.sides = {.25:{0:[],1:[]},.75:{0:[],1:[]}}
		# sgn = 1 if self.refinement.rtype % 2 == 0 else -1
		# for p_id,p in enumerate(self.patches):
		# 	for dof_id in p.zigzag_interface:
		# 		dof = p.dofs[dof_id]
		# 		dof_comp = dof.x if comp == 0 else dof.y
		# 		if p_id == 0 or dof_comp not in [.25,.75]:
		# 			s_shfts = [sgn*dof.h,-sgn*dof.h]
		# 			for s_id,s in enumerate(self.sides):
		# 				s_shft = s + s_shfts[s_id] if p_id == 1 else s
		# 				if dof_comp == s_shft:
		# 					self.sides[s][p_id].append(dof)
		# 					if p_id == 0:
		# 						self.zigzag_ghost_fills.append(dof.ID)
		# 				elif p_id == 1 and dof_comp in self.extras[s]:
		# 					self.extras[s][dof_comp].append(dof)


		# for side in self.sides:
		# 	trap_corners,tri_corners = {},{}
		# 	c_dofs,f_dofs = self.sides[side][0],self.sides[side][1]
		# 	if (c_dofs[0].x < f_dofs[0].x) and (c_dofs[0].y<f_dofs[0].y):
		# 		c0, f0 = 0,1
		# 	else:
		# 		c0, f0 = 1,0
		# 	trap_corners[c0] = c_dofs[:-1]
		# 	trap_corners[f0] = f_dofs[::2]
		# 	trap_corners[c0+2] = c_dofs[1:]
		# 	trap_corners[f0+2] = f_dofs[1::2]

		# 	if (c_dofs[1].x > f_dofs[1].x) and (c_dofs[1].y>f_dofs[1].y):
		# 		c0, f0, f1 = 1,0,2
		# 	else:
		# 		c0, f0, f1 = 0,1,2
		# 	tri_corners[c0] = c_dofs[1:-1]
		# 	tri_corners[f0] = f_dofs[1:-1:2]
		# 	tri_corners[f1] = f_dofs[2:-1:2]

		# 	self.trapezoids[side] = trap_corners
		# 	self.triangles[side] = tri_corners

		# quad_comp = 0 if self.refinement.yside else 1

		# lows = [False,False,True,True]
		# highs = [True,True,False,False]
		# if self.rindex == 2:
		# 	if self.refinement.type == 0:
		# 		low_high = [.25,.75-2*H]
		# 	else:
		# 		low_high = [.75,.25-2*H]
		# else:
		# 	low_high = [0,1-2*H]
			
			
		# # local_id = 0
		# local_id = len(mesh.patches[0].elements)+len(mesh.patches[1].alt_el)
		# for side in self.sides:
		# 	for trap_id in range(len(self.trapezoids[side][0])):
		# 		corners = [self.trapezoids[side][i][trap_id] for i in range(4)]
		# 		my_el = TrapElement(local_id,ords,corners)
		# 		# my_el.set_global_ID(e_id_shift)
		# 		local_id += 1
		# 		for dof_comp in self.extras[side]:
		# 			my_el.add_dof(self.extras[side][dof_comp][2*trap_id])
		# 			my_el.add_dof(self.extras[side][dof_comp][2*trap_id+1])

		# 		if True:#self.rindex == 2: # could have split support
		# 			quad_check = [corners[0].x,corners[0].y][quad_comp]
		# 			if quad_check < low_high[0]:
		# 				my_el.set_support(lows)
		# 			elif quad_check > low_high[1]:
		# 				my_el.set_support(highs)

		# 		self.trapezoid_elements.append(my_el)
				
		# 	for tri_id in range(len(self.triangles[side][0])):
		# 		corners = [self.triangles[side][i][tri_id] for i in range(3)]
		# 		my_el = TriElement(local_id,ords,corners)
		# 		# my_el.set_global_ID(e_id_shift)
		# 		local_id += 1
		# 		for dof_comp in self.extras[side]:
		# 			my_el.add_dof(self.extras[side][dof_comp][2*tri_id+1])
		# 			my_el.add_dof(self.extras[side][dof_comp][2*tri_id+2])
		# 		self.triangle_elements.append(my_el)


		# we need a method for stiffness matrix on trap and tri elements
		# Jt_trap = {0:lambda nu,rho: H*np.array([[1/2,1/4-rho/2],[0,1-nu/2]]),
		# 		   2:lambda nu,rho: H*np.array([[1/2,-1/4+rho/2],[0,1/2+nu/2]]),
		# 		   1:lambda nu,rho: H*np.array([[1-rho/2,0],[1/4-nu/2,1/2]]),
		# 		   3:lambda nu,rho: H*np.array([[1/2+rho/2,0],[-1/4+nu/2,1/2]])}
		# Js_trap = {0:lambda xi,eta,i,j: H*np.array([[1,eta-j-1/2],[0,xi-i+1]]),
		# 		   2:lambda xi,eta,i,j: H*np.array([[1,-eta+j+1/2],[0,-xi+i+2]]),
		# 		   1:lambda xi,eta,i,j: H*np.array([[eta-j+1,0],[xi-i-1/2,1]]),
		# 		   3:lambda xi,eta,i,j: H*np.array([[-eta+j+2,0],[-xi+i+1/2,1]])}
		# Js_trap_inv = {0:lambda xi,eta,i,j: 1/H*np.array([[1,-(eta-j-1/2)/(xi-i+1)],[0,1/(xi-i+1)]]),
		# 		       2:lambda xi,eta,i,j: 1/H*np.array([[1,-(-eta+j+1/2)/(-xi+i+2)],[0,1/(-xi+i+2)]]),
		# 		       1:lambda xi,eta,i,j: 1/H*np.array([[1/(eta-j+1),0],[-(xi-i-1/2)/(eta-j+1),1]]),
		# 		       3:lambda xi,eta,i,j: 1/H*np.array([[1/(-eta+j+2),0],[-(-xi+i+1/2)/(-eta+j-1),1]])}

		Jd_trap = {0:lambda nu,rho: 1/H*np.array([[1,(4-8*rho)/(2+nu)**2+(2*rho-1)/(2+nu)],[0,2/(2+nu)]]),
				   2:lambda nu,rho: 1/H*np.array([[1,(-4+8*rho)/(3-nu)**2+(1-2*rho)/(3-nu)],[0,2/(3-nu)]]),
				   1:lambda nu,rho: 1/H*np.array([[2/(2+rho),0],[(2*nu-1)/(2+rho)+(4-8*nu)/(2+rho)**2,1]]),
				   3:lambda nu,rho: 1/H*np.array([[2/(3-rho),0],[(1-2*nu)/(3-rho)+(-4+8*nu)/(3-rho)**2,1]])}

		Jd_trap_coefs = {
			0:lambda nu,rho: 1/H**2*np.array([1,(4-8*rho)/(2+nu)**2+(2*rho-1)/(2+nu),0,4/(2+nu)**2+((4-8*rho)/(2+nu)**2+(2*rho-1)/(2+nu))**2]),
			2:lambda nu,rho: 1/H**2*np.array([1,(-4+8*rho)/(3-nu)**2+(1-2*rho)/(3-nu),0,4/(3-nu)**2+((-4+8*rho)/(3-nu)**2+(1-2*rho)/(3-nu))**2]),
			1:lambda nu,rho: 1/H**2*np.array([4/(2+rho)**2+((4-8*rho)/(2+nu)**2+(2*rho-1)/(2+nu))**2,0,(2*nu-1)/(2+rho)+(4-8*nu)/(2+rho)**2,1]),
			3:lambda nu,rho: 1/H**2*np.array([4/(3-rho)**2+((-4+8*rho)/(3-nu)**2+(1-2*rho)/(3-nu))**2,0,(1-2*nu)/(3-rho)+(-4+8*nu)/(3-rho)**2,1])}


		Jt_trap_det = {0:lambda nu,rho: H**2*(1/2-nu/4),
				       2:lambda nu,rho: H**2*(1/4+nu/4),
				       1:lambda nu,rho: H**2*(1/2-rho/4),
				       3:lambda nu,rho: H**2*(1/4+rho/4)}
		# Js_trap_det = {0:lambda xi,eta,i,j: H**2*(xi-i+1),
		# 		       2:lambda xi,eta,i,j: H**2*(-xi+i+2),
		# 		       1:lambda xi,eta,i,j: H**2*(eta-j+1),
		# 		       3:lambda xi,eta,i,j: H**2*(-eta+j+2)}

		A = lambda a,b: (2+b)/(2+a+b)**2
		B = lambda a,b: -a/(2+a+b)**2-2/(2+b)**2

		# Jp_tri = {0:lambda a,b: np.array([[1/2,A(a,b)],[1/2,B(a,b)]]),
		# 		  2:lambda a,b: np.array([[-1/2,A(a,b)],[-1/2,B(a,b)]]),
		# 		  1:lambda a,b: np.array([[A(a,b),1/2],[B(a,b),1/2]]),
		# 		  3:lambda a,b: np.array([[A(a,b),-1/2],[B(a,b),-1/2]])}
		# Jt_tri = {0: H*np.array([[1/2,1/4],[1/2,-1/4]]),
		# 		  2: H*np.array([[-1/2,1/4],[-1/2,-1/4]]),
		# 		  1: H*np.array([[1/4,1/2],[-1/4,1/2]]),
		# 		  3: H*np.array([[1/4,-1/2],[-1/4,-1/2]])}
		# Jt_tri_inv = {0: 1/H*np.array([[1,1],[2,-2]]),
		# 		      2: 1/H*np.array([[-1,-1],[2,-2]]),
		# 		      1: 1/H*np.array([[2,-2],[1,1]]),
		# 		      3: 1/H*np.array([[2,-2],[-1,-1]])}

		Jt_tri_det = {0:lambda *args: -H**2/4,2:lambda *args: H**2/4,
					  1:lambda *args: H**2/4,3:lambda *args: -H**2/4}
		# Jp_tri_det = {0:lambda a,b: (B(a,b)-A(a,b))/2,
		# 		      2:lambda a,b: (A(a,b)-B(a,b))/2,
		# 		      1:lambda a,b: (A(a,b)-B(a,b))/2,
		# 		      3:lambda a,b: (B(a,b)-A(a,b))/2}

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

		self.Jds_eval = [{k:{} for k in range(4)}]*2
		self.J_dets_eval = [{k:{} for k in range(4)}]*2

		# id_shift = len(self.patches[0].dofs)
		# for el in self.trapezoid_elements+self.triangle_elements:
		# 	el.set_dof_ids(id_shift)
		# return

		def wrap(J):
			A = J[0,0]**2+J[1,0]**2
			B = J[0,0]*J[0,1]+J[1,0]*J[1,1]
			C = J[1,1]**2+J[0,1]**2
			return np.array([A,B,C])


		def get_vals_at_points(func,points,arr=False,map=None):
			phi_eval = map is not None
			if map is None:
				map = lambda a,b:[a,b]
			qpn,m,dm = points.shape
			assert qpn == m

			if arr and phi_eval:
				vals = np.zeros((qpn,qpn,2))
			elif arr:
				vals = [np.zeros((qpn,qpn,2,2)),np.zeros((qpn,qpn,3))]
			else:
				vals = np.zeros((qpn,qpn))
			for j in range(qpn):
				for i in range(qpn):
					xinput,yinput = points[i,j]
					if arr and phi_eval:
						xi,eta = map(xinput,yinput)
						vals[i,j,:] = func(xi,eta)
					elif arr:
						J = func(xinput,yinput)
						vals[0][i,j] = J
						vals[1][i,j] = wrap(J)
					else:
						xi,eta = map(xinput,yinput)
						vals[i,j] = func(xi,eta)
						# vals[i,j] = func(xinput,yinput)

			return vals


		if self.refinement.yside:
			chop = self.ords[0]+2
			self.tri_prod = (self.ords[0]+2)*(self.ords[1]+1)
			quad_comp = 0
			lows = [False,True,False,True]
			highs = [True,False,True,False]
		else:
			chop = self.ords[0]+1
			self.tri_prod = (self.ords[1]+2)*(self.ords[0]+1)
			quad_comp = 1
			lows = [False,False,True,True]
			highs = [True,True,False,False]
		if self.rindex == 2:
			if self.refinement.type == 0:
				low_high = [.25,.75-H]
			else:
				low_high = [.75,.25-H]
		else:
			low_high = [0,1-H]
		tri_id_map = {ID:[int(ID/chop),ID%chop] for ID in range(tri_prod)}

		ref_eval_locs = self.integrator.quad_ref_eval_locs

		self.phi_vals = [{k:{} for k in range(4)}]*2
		self.dphi_vals = [{k:{} for k in range(4)}]*2

		for quad_id in range(4):
			points = ref_eval_locs[quad_id][-1]
			for map_type in range(4): # element map_type
				for shape in range(2):
					dets = get_vals_at_points(self.J_dets[shape][map_type],points)
					self.J_dets_eval[shape][map_type][quad_id] = dets

					jds = get_vals_at_points(self.Jds[shape][map_type],points,arr=True)
					self.Jds_eval[shape][map_type][quad_id] = jds

					self.phi_vals[shape][map_type][quad_id] = []
					self.dphi_vals[shape][map_type][quad_id] = []
					my_prod = self.tri_prod if shape else self.integrator.prod
					my_id_map = tri_id_map if shape else self.integrator.id_map
					my_p_map = phi_input_tri if shape else phi_input_trap
					my_map = lambda a,b: my_p_map(a,b,map_type)
					for test_id in range(my_prod):
						test_ind = my_id_map[test_id]
						phi_test = lambda x,y: self.integrator.phi(x,y,1,test_ind)
						dphi_test = lambda x,y: self.integrator.dphi(x,y,1,test_ind)
						self.phi_vals[shape][map_type][quad_id].append(
							get_vals_at_points(phi_test,points,map=my_map))
						self.dphi_vals[shape][map_type][quad_id].append(
							get_vals_at_points(dphi_test,points,map=my_map,arr=True))


		for i,els in enumerate([self.trapezoid_elements,self.triangle_elements]):
			for el in els:
				el.set_jacobian_eval(self.J_dets_eval[i],self.Jds_eval[i])
				quad_check = el.h*[el.K,el.L][quad_comp]
				if el.tri:
					el.set_support([True]*4)
				elif quad_check < low_high[0]:
					el.set_support(lows)
				elif quad_check > low_high[1]:
					el.set_support(highs)
				else:
					el.set_support([True]*4)

	def _get_vals(self,k=True):
		lab = 'k' if k else 'm'
		ord_string = '{}{}'.format(self.ords[0],self.ords[1])
		fpath = os.path.join(os.path.dirname(os.getcwd()),'pickled/')
		if globals.LAG:
			fname = fpath+'zigzag_{}_vals_p{}_qpn{}.pickle'.format(lab,ord_string,self.qpn)
		else:
			fname = fpath+'zigzag_{}_spline_vals_p{}_qpn{}.pickle'.format(lab,ord_string,self.qpn)

		try:
			with open(fname,'rb') as handle:
				vals = pickle.load(handle)
		except:
			base_vol = 1/2**self.dim
			vals = {}
			size = self.tri_prod if tri else self.integrator.prod
			for tri in range(2):
				shape_vals = {}
				for map_type in range(4):
					if k: dphi_dict = self.dphi_vals[tri][map_type]
					map_vals = {}
					for quad_id in range(4):
						jdet = self.J_dets_eval[tri][map_type][quad_id]
						jac = self.Jds_eval[tri][map_type][quad_id][-1]
						local = np.zeros((size,size))
						for i in range(size):
							for j in range(i,size):
								if k:
									val = self._compute_k_product_integral(
										i,j,quad_id,jac=jac,jdet=jdet,dphi_dict=dphi_dict)
								else:
									phi_i = self.phi_vals[tri][map_type][quad_id][i]
									phi_j = self.phi_vals[tri][map_type][quad_id][j]
									val = self._compute_product_integral(
												phi_i,phi_j,volume=base_vol,jdet=jdet)
								local[i,j] = val
								local[j,i] = val

						map_vals[quad_id] = local
					shape_vals[map_type] = map_vals
				vals[tri] = shape_vals
				
			with open(fname,'wb') as handle:
				pickle.dump(vals,handle,protocol=pickle.HIGHEST_PROTOCOL)
		if k:
			self.k_vals = vals
		else:
			self.m_vals = vals
	def _get_transformed_vals(self,k=True):
		lab = 'k' if k else 'm'
		qpn = self.integrator.qpn

		ord_string = '{}{}'.format(self.ords[0],self.ords[1])
		fpath = os.path.join(os.path.dirname(os.getcwd()),'pickled/')
		if globals.LAG:
			fname = fpath+'zigzag_{}_vals_p{}_qpn{}.pickle'.format(lab,ord_string,qpn)
		else:
			fname = fpath+'zigzag_{}_spline_vals_p{}_qpn{}.pickle'.format(lab,ord_string,qpn)

		try:
			with open(fname,'rb') as handle:
				vals = pickle.load(handle)
		except:
			xside_d = {0:[-1,0],1:[0,0],2:[1,0],3:[-1,1],4:[0,1],5:[1,1]}
			yside_d = {0:[0,-1],1:[1,-1],2:[0,0],3:[1,0],4:[0,1],5:[1,1]}
			vals = {}
			size = self.integrator.prod
			for shape in [False,True]:
				shape_vals = {}
				J_vals = self.J_tri_vals if shape else self.J_trap_vals
				J_dets = self.J_tri_dets if shape else self.J_trap_dets
				for map_type in range(4):
					map_vals = {}
					for quad_id in range(4):
						local = np.zeros((size,size))
						jac_vals = J_vals[map_type][quad_id][1]
						j_det = J_dets[map_type][quad_id]
						for i in range(size):
							for j in range(i,size):
								if k:
									val = self.integrator._compute_k_product_integral(i,j,quad_id,jac=jac_vals,jdet_inv=1/j_det)
								else:
									phi_i = self.integrator.phi_vals[quad_id][i]
									phi_j = self.integrator.phi_vals[quad_id][j]
									val = self.integrator._compute_product_integral(
												phi_i,phi_j,volume=1/2**self.dim,jdet=j_det)
								local[i,j] = val
								local[j,i] = val

						map_vals[quad_id] = local
					shape_vals[map_type] = map_vals	
				vals[shape] = shape_vals
				
			with open(fname,'wb') as handle:
				pickle.dump(vals,handle,protocol=pickle.HIGHEST_PROTOCOL)
		if k:
			self.k_vals = vals
		else:
			self.m_vals = vals


	def get_lookup_vals(self,k=True):
		if k:
			if self.k_vals is None:
				self._get_vals(k)
			return self.k_vals
		else:
			if self.m_vals is None:
				self._get_vals(k)
			return self.m_vals