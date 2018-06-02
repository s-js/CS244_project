import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from graph import NXTopology, d_star


y_axis = []
x_axis = list(range(2,10))
for r in np.array(x_axis)/5:
    n = 40
    f = n*10
    t = NXTopology(number_of_servers=f,
                    switch_graph_degree=r, number_of_racks=n)

upper_bound = n * r / (f * d_star(n, r))
