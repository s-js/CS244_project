import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from graph import NXTopology, theoretical_upper_bound, TrafficType, d_star

if __name__ == "__main__":
    
    ############## figure 1 a ##############
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
    plt.ylim((0, 1.1))
    plt.legend(loc='lower center')
    plt.savefig("fig1a.svg")
    plt.show()
    
    ############## figure 1 b ##############
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
    plt.plot(x_axis, y_axis1, label='Observed ASPL', marker='+', color='red')
    plt.plot(x_axis, y_axis2, label='ASPL lower-bound', color='green')
    plt.xlabel('Network Degree')
    plt.ylabel('Path Length')
    plt.legend()
    plt.ylim((1,4))
    plt.grid(linestyle='--', color='lightgray')
    plt.show()
    plt.savefig('fig1b.svg')
