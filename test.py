import numpy
from cvxopt import matrix, solvers
c = matrix([-4., -5.])
G = matrix([[2., 1., -1., 0.], [1., 2., 0., -1.]])
h = matrix([3., 3., 0., 0.])
sol = solvers.lp(c, G, h)
print(sol['x'])