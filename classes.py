import numpy as np
import matplotlib.pyplot as plt

class Node:
    def __init__(self,j,i,x,y,h):
        self.ID
        self.j = j
        self.i = i
        self.x
        self.y
        self.h

class Element:
    def __init__(self,ID,j,i,x,y,h):
        self.ID = ID
        self.j = j
        self.i = i
        self.x = x
        self.y = y
        self.h = h
        self.dof_ids = []
        self.dof_list = []
        self.stiffness = None

    def add_dofs(self,strt,xlen):
        if len(self.dof_ids) != 0:
            return
        for ii in range(4):
            for jj in range(4):
                self.dof_ids.append(strt+xlen*ii+jj)
        return

    def update_dofs(self,dofs):
        if len(self.dof_list) != 0:
            return
        for dof_id in self.dof_ids:
            dof = dofs[dof_id]
            self.dof_list.append(dof)
        return

class Mesh:
    def __init(self,N):
        self.N = N # number of fine elements 
                   # from x=0.5 to x=1.0
        self.h = 0.5/N
        self.dofs = {}
        self.elements = {}
        self.boundaries = []
        self.interface = [[],[]]
        
        self._make_coarse()
        self._make_fine()

        self._update_elements()

    def _make_coarse(self):
        H = self.h*2
        xdom = np.linspace(0-H,0.5+H,int(self.N/2)+3)
        ydom = np.linspace(0-H,1+H,self.N+3)

        xlen,ylen = len(xdom),len(ydom)

        dof_id,e_id = 0,0
        for i,y in enumerate(ydom):
            for j,x in enumerate(xdom):
                self.dofs[dof_id] = Node(j,i,x,y,H)

                if (0<=x<.5) and (0<=y<1.):
                    strt = dof_id-1-xlen
                    element = Element(e_id,j-1,i-1,x,y,H)
                    element.add_dofs(strt,xlen)
                    self.elements[e_id] = element
                    e_id += 1

                if x==0. or y==0. or y==1:
                    if (0<=x<=.5) and (0<=y<=1.):
                        self.boundaries.append(dof_id)
                elif x==0.5 and 0<y<1:
                    self.interface[0].append(dof_id)

                dof_id += 1

        self.n_coarse_dofs = dof_id
        self.n_coarse_els = e_id

    def _make_fine(self):
        H = self.h
        xdom = np.linspace(0.5-H,1.+H,self.N+3)
        ydom = np.linspace(0-H,1+H,2*self.N+3)

        dof_id,e_id = self.n_coarse_dofs,self.n_coarse_els
        for i,y in enumerate(ydom):
            for j,x in enumerate(xdom):
                self.dofs[dof_id] = Node(j,i,x,y,H)

                if (0.5<=x<1.) and (0<=y<1.):
                    strt = dof_id-1-xlen
                    element = Element(e_id,j-1,i-1,x,y,H)
                    element.add_dofs(strt,xlen)
                    self.elements[e_id] = element
                    e_id += 1

                if x==1. or y==0. or y==1:
                    if (.5<=x<=.1) and (0<=y<=1.):
                        self.boundaries.append(dof_id)
                elif x==0.5 and 0<y<1:
                    self.interface[1].append(dof_id)

                dof_id += 1

    def _update_elements(self):
        for e in self.elements:
            e.update_dofs(self.dofs)

class Laplace:
    def __init(self,N,u,f):
        self.N = N
        self.ufunc = u
        self.ffunc = f

        self.mesh = Mesh(N)
        self.h = self.mesh.h

    def _build_stiffness(self):
        k_coarse = local_stiffness(2*self.h)
        k_fine = local_stiffness(self.h)

        k_interface_0 = local_stiffness(self.h,interface=True,top=0)
        k_interface_1 = local_stiffness(self.h,interface=True,top=1)

        local_ks = [[k_coarse,k_fine],[k_interface_0,k_interface_1]]

        for e in self.mesh.elements:
            fine = e.h == self.h
            for dof in e.dof_list:
               pass 
