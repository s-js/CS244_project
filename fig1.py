import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from graph import NXTopology, theoretical_upper_bound, TrafficType

if __name__ == "__main__":
    y_axis = []
    x_axis = list(range(3, 33, 2))
    number_of_switches = 40

    for r in x_axis:
        print('r = {}'.format(r))
        number_of_servers = number_of_switches * 10
        t = NXTopology(number_of_servers=number_of_servers,
                       switch_graph_degree=r, number_of_racks=number_of_switches)
        # print(t.G.edges)
        # print(t.sender_to_receiver)
        Z = t.get_max_min_throughput(TrafficType.PERMUTATION)
        ratio = Z / theoretical_upper_bound(number_of_switches, r, number_of_servers, TrafficType.PERMUTATION)
        y_axis.append(ratio)
        print('ratio = {}'.format(ratio))

    plt.figure()
    plt.plot(x_axis, y_axis)
    plt.savefig("1.svg")
    plt.show()
