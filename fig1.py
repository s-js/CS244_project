import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from graph import NXTopology, d_star

if __name__ == "__main__":
    y_axis = []
    x_axis = list(range(3, 33, 10))
    for r in x_axis:
        n = 40
        f = n*10
        t = NXTopology(number_of_servers=f,
                       switch_graph_degree=r, number_of_racks=n)
        # print(t.G.edges)
        # print(t.sender_to_receiver)
        ratio=t.get_max_min_throughput()
        y_axis.append(ratio)

    plt.figure()
    plt.plot(x_axis, y_axis)
    plt.savefig("1.svg")
    plt.show()