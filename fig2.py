import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from graph import NXTopology, d_star


if __name__ == "__main__":
    '''
    # figure 2 b
    y_axis1 = []
    y_axis2 = []
    x_axis = list(range(18, 199, 10))
    for n in x_axis:
        r = 10
        t = NXTopology(number_of_servers=n,
                       switch_graph_degree=r, number_of_racks=n)

        y_axis1.append(t.average_shortest_path_length())
        y_axis2.append(d_star(n, r))

    plt.figure()
    plt.plot(x_axis, y_axis1, label='Observed ASPL')
    plt.plot(x_axis, y_axis2, label='ASPL lower-bound')
    plt.xlabel('Network Size')
    plt.ylabel('Path Length')
    plt.legend()
    plt.show()
    plt.savefig('2.svg')
    '''

    # figure 1 b
    y_axis1 = []
    y_axis2 = []
    x_axis = list(range(3, 34, 2))
    for r in x_axis:
        n = 40
        t = NXTopology(number_of_servers=n,
                       switch_graph_degree=r, number_of_racks=n)

        y_axis1.append(t.average_shortest_path_length())
        y_axis2.append(d_star(n, r))

    plt.figure()
    plt.plot(x_axis, y_axis1, label='Observed ASPL')
    plt.plot(x_axis, y_axis2, label='ASPL lower-bound')
    plt.xlabel('Network Degree')
    plt.ylabel('Path Length')
    plt.legend()
    plt.show()
    plt.savefig('3.svg')
