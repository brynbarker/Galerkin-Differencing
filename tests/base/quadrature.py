#!/usr/bin/env python3

from linear_basis.mac_grid.helpers import gauss
from scipy.integrate import dblquad
import numpy as np

output = []

def test_quad(f,a,b,c,d,qpn=5):
	func_rev = lambda y,x: f(x,y)
	scipy_val = dblquad(func_rev,a,b,c,d)[0]
	my_val = gauss(f,a,b,c,d,qpn)
	return scipy_val,my_val,abs(scipy_val-my_val)

func = lambda x,y: np.sin(x)*np.exp(y)+5*x**3*y
a,b,c,d = 3,6,4,12
v1,v2,diff = test_quad(func,a,b,c,d,15)
output.append(diff/v1 < 1e-14)

func = lambda x,y: np.sin(x)+5*x**3*y
a,b,c,d = 3,6,4,12
v1,v2,diff = test_quad(func,a,b,c,d,8)
output.append(diff/v1 < 1e-14)

func = lambda x,y: np.sin(x)*np.cos(y)+5*x**3*y
a,b,c,d = 3,6,4,12
v1,v2,diff = test_quad(func,a,b,c,d,9)
output.append(diff/v1 < 1e-14)

func = lambda x,y: x**6*y+5*x**3*y
a,b,c,d = 3,6,4,12
v1,v2,diff = test_quad(func,a,b,c,d,5)
output.append(diff/v1 < 1e-14)

### write results to output file
with open("output", 'w') as handle:
    for o in output:
        handle.write(str(o) + "\n")

