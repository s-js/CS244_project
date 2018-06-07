import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from graph import NXTopology, theoretical_upper_bound, TrafficType

if __name__ == "__main__":
    x_axis = list(range(3, 33, 2))
    ys = []
    parameters = [
        {'servers_per_rack': 1,
         'traffic_type': TrafficType.ALL_TO_ALL,
         'label': 'All to All',
         'color': 'red',
         'marker': '+'
         },
        {'servers_per_rack': 10,
         'traffic_type': TrafficType.PERMUTATION,
         'label': 'Permutation (10 Servers per switch)',
         'color': 'green',
         'marker': 'o'
         },
        {'servers_per_rack': 5,
         'traffic_type': TrafficType.PERMUTATION,
         'label': 'Permutation (5 Servers per switch)',
         'color': 'blue',
         'marker': 'x'
         }
    ]
    number_of_switches = 40
    for i in range(len(parameters)):
        y_axis = []
        for r in x_axis:
            print('r = {}'.format(r))
            number_of_servers = number_of_switches * \
                parameters[i]['servers_per_rack']
            t = NXTopology(number_of_servers=number_of_servers,
                           switch_graph_degree=r, number_of_racks=number_of_switches)
            # print(t.G.edges)
            # print(t.sender_to_receiver)
            Z = t.get_max_min_throughput(parameters[i]['traffic_type'])
            ratio = Z / \
                theoretical_upper_bound(
                    number_of_switches, r, number_of_servers, parameters[i]['traffic_type'])

            y_axis.append(ratio)
            print('ratio = {}'.format(ratio))
        ys.append(y_axis)

    plt.figure()
    for i in range(len(ys)):
        plt.plot(x_axis, ys[i], marker=parameters[i]['marker'],
                 label=parameters[i]['label'], color=parameters[i]['color'])
    plt.grid(linestyle='--', color='lightgray')
    plt.ylabel('Throughput\n(Ratio to Upper-bound)')
    plt.xlabel('Network Degree')
    plt.legend(loc='lower center')
    plt.savefig("1.svg")
    plt.show()
