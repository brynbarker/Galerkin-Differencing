import numpy as np
import matplotlib.pyplot as plt
from general_solve.refinement import UniformRefinement
from general_solve.refinement import StripeRefinement
from general_solve.refinement import SquareRefinement
from general_solve.patch import Patch
from general_solve.element import TrapElement, TriElement

class InterfaceMapping():
	def __init__(self,mesh,integrator,ords):
		self.mesh = mesh
		self.integrator = integrator
		self.refinement = mesh.refinement
		self.patches = mesh.patches
		self.ords = ords

		self.rindex = mesh.rindex
		self.sides = {}

		extra_ops = {.25:[],.75:[]}
		self.extras = {}

		self.trapezoids = {}
		self.triangles = {}
		self.trapezoid_elements = []
		self.triangle_elements = []

		all_clear = self.rindex == 0
		if all_clear: return
		if self.rindex == 1: #stripe
			rdim = int(self.refinement.rtype/2)
			if self.refinement.xside and rdim == 1:
				all_clear = True
			elif self.refinement.yside and rdim == 0:
				all_clear = True

		if self.rindex == 2:
			if self.refinement.xside or self.refinement.yside:
				all_clear = False

		L,R = int((max(ords)-1)/2),int(max(ords)/2)
		H = mesh.h/2
		if self.refinement.rtype % 2 == 0:
			first,second = .25,.75
		else:
			first,second = .75,.25
		if R >= 1:
			for r in range(R):
				extra_ops[first].append(first-H*(r+1))
				extra_ops[second].append(second-H*(r+2))
		if L >= 1:
			for ell in range(L):
				extra_ops[first].append(first+H*(ell+2))
				extra_ops[second].append(second+H*(ell+1))

		for s in [.25,.75]:
			self.extras[s] = {pos:[] for pos in extra_ops[s]}

		if all_clear:
			return

		comp = 0 if self.refinement.xside else 1
		self.sides = {.25:{0:[],1:[]},.75:{0:[],1:[]}}
		sgn = 1 if self.refinement.rtype % 2 == 0 else -1
		for p_id,p in enumerate(self.patches):
			for dof_id in p.full_interface:
				dof = p.dofs[dof_id]
				dof_comp = dof.x if comp == 0 else dof.y
				if p_id == 0 or dof_comp not in [.25,.75]:
					s_shfts = [sgn*dof.h,-sgn*dof.h]
					for s_id,s in enumerate(self.sides):
						s_shft = s + s_shfts[s_id] if p_id == 1 else s
						if dof_comp == s_shft:
							self.sides[s][p_id].append(dof)
						elif p_id == 1 and dof_comp in self.extras[s]:
							self.extras[s][dof_comp].append(dof)


		for side in self.sides:
			trap_corners,tri_corners = {},{}
			c_dofs,f_dofs = self.sides[side][0],self.sides[side][1]
			trap_corners[0] = c_dofs[:-1]
			trap_corners[1] = f_dofs[::2]
			trap_corners[2] = c_dofs[1:]
			trap_corners[3] = f_dofs[1::2]

			tri_corners[0] = c_dofs[1:-1]
			tri_corners[1] = f_dofs[1:-1:2]
			tri_corners[2] = f_dofs[2:-1:2]

			self.trapezoids[side] = trap_corners
			self.triangles[side] = tri_corners

		quad_comp = 0 if self.refinement.yside else 1
		if self.rindex == 2:
			lows = [False,False,True,True]
			highs = [True,True,False,False]
			if self.refinement.type == 0:
				low_high = [.25,.75-2*self.h]
			else:
				low_high = [.75,.25-2*self.h]
			
		for side in self.sides:
			for trap_id in range(len(self.trapezoids[side][0])):
				corners = [self.trapezoids[side][i][trap_id] for i in range(4)]
				my_el = TrapElement(trap_id,ords,corners)
				for dof_comp in self.extras[side]:
					my_el.add_dof(self.extras[side][dof_comp][2*trap_id])
					my_el.add_dof(self.extras[side][dof_comp][2*trap_id+1])

				if self.rindex == 2: # could have split support
					quad_check = [corners[0].x,corners[0].y][quad_comp]
					if quad_check < low_high[0]:
						my_el.set_support(lows)
					elif quad_check > low_high[0]:
						my_el.set_support(highs)

				self.trapezoid_elements.append(my_el)
				
			for tri_id in range(len(self.triangles[side][0])):
				corners = [self.triangles[side][i][tri_id] for i in range(3)]
				my_el = TriElement(tri_id,ords,corners)
				for dof_comp in self.extras[side]:
					my_el.add_dof(self.extras[side][dof_comp][2*tri_id+1])
					my_el.add_dof(self.extras[side][dof_comp][2*tri_id+2])
				self.triangle_elements.append(my_el)


		# we need a method for stiffness matrix on trap and tri elements
		H = 1
		trap_J0 = {}
		trap_J0[0] = np.array([[H,H/2],[0,2*H]])
		trap_J0[1] = np.array([[-H,H/2],[0,2*H]])
		trap_J0[2] = np.array([[H/2,H],[2*H,0]])
		trap_J0[3] = np.array([[H/2,-H],[2*H,0]])

		trap_Jxi = {}
		trap_Jxi[0] = np.array([[0,0],[0,-H]])
		trap_Jxi[1] = np.array([[0,0],[0,-H]])
		trap_Jxi[2] = np.array([[0,0],[-H,0]])
		trap_Jxi[3] = np.array([[0,0],[-H,0]])

		trap_Jeta = {}
		trap_Jeta[0] = np.array([[0,-H],[0,0]])
		trap_Jeta[1] = np.array([[0,-H],[0,0]])
		trap_Jeta[2] = np.array([[-H,0],[0,0]])
		trap_Jeta[3] = np.array([[-H,0],[0,0]])

		tri_J0 = {}
		tri_J0[0] = np.array([[H,-H/2],[-H/2,0]])
		tri_J0[1] = np.array([[-H,-H/2],[-H/2,0]])
		tri_J0[2] = np.array([[0,-H/2],[0,H]])
		tri_J0[3] = np.array([[0,-H/2],[0,-H]])

		tri_Jxi = {}
		tri_Jxi[0] = np.array([[0,0],[0,H]])
		tri_Jxi[1] = np.array([[0,0],[0,H]])
		tri_Jxi[2] = np.array([[0,H],[0,0]])
		tri_Jxi[3] = np.array([[0,H],[0,0]])

		tri_Jeta = {}
		tri_Jeta[0] = np.array([[0,0],[H,0]])
		tri_Jeta[1] = np.array([[0,0],[H,0]])
		tri_Jeta[2] = np.array([[H,0],[0,0]])
		tri_Jeta[3] = np.array([[H,0],[0,0]])

		def det_trap_J(map_type,xi,eta):
			J0 = trap_J0[map_type]
			Jxi = trap_Jxi[map_type]
			Jeta = trap_Jeta[map_type]
			J = J0 + xi*Jxi + eta*Jeta
			Jdet = J[0,0]*J[1,1]-J[0,1]*J[1,0]
			return 1/Jdet

		def det_tri_J(map_type,xi,eta):
			J0 = tri_J0[map_type]
			Jxi = tri_Jxi[map_type]
			Jeta = tri_Jeta[map_type]
			J = J0 + xi*Jxi + eta*Jeta
			Jdet = J[0,0]*J[1,1]-J[0,1]*J[1,0]
			return 1/Jdet

		def get_vals_at_points(func,points,k):
			qpn,m = points.shape
			assert qpn == m

			vals = np.zeros_like(points)
			for j in range(qpn):
				for i in range(qpn):
					xinput,yinput = points[j,i]
					vals[i,j] = func(k,xinput,yinput)

			return vals


		self.J_trap_vals = {k:{} for k in range(4)}
		self.J_tri_vals = {k:{} for k in range(4)}
		ref_eval_locs = self.integrator.quad_ref_eval_locs
		for k in range(4):
			for quad_id in range(4):
				self.J_trap_vals[k][quad_id] = []
				self.J_tri_vals[k][quad_id] = []

				for test_id in range(self.prod):
					points = ref_eval_locs[quad_id][test_id]
					trap_vals = get_vals_at_points(det_trap_J,points,k)
					tri_vals = get_vals_at_points(det_tri_J,points,k)

					self.J_trap_vals[k][quad_id].append(trap_vals)
					self.J_tri_vals[k][quad_id].append(tri_vals)


		for trap_el in self.trapezoid_elements:
			trap_el.set_jacobian(self.J_trap_vals)
		for tri_el in self.triangle_elements:
			tri_el.set_jacobian(self.J_tri_vals)

		id_shift = len(self.patches[0].dofs)
		for el in self.trapezoid_elements+self.triangle_elements:
			el.set_dof_ids(id_shift)
