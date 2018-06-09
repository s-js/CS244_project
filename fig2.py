import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from graph import NXTopology, theoretical_upper_bound, TrafficType, d_star


if __name__ == "__main__":

    ############## figure 2 a ##############
    x_axis = list(range(16, 201, 10))
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
    r = 10
    for i in range(len(parameters)):
        y_axis = []
        for n in x_axis:
            print('n = {}'.format(n))
            number_of_servers = n * \
                parameters[i]['servers_per_rack']
            t = NXTopology(number_of_servers=number_of_servers,
                           switch_graph_degree=r, number_of_racks=n)
            # print(t.G.edges)
            # print(t.sender_to_receiver)
            Z = t.get_max_min_throughput(parameters[i]['traffic_type'])
            ratio = Z / \
                theoretical_upper_bound(
                    n, r, number_of_servers, parameters[i]['traffic_type'])

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
    plt.savefig("fig2a.svg")
    plt.show()
    '''
    ############## figure 2 b ##############
    x_axis = list(range(16, 201, 10))
    y_axis1 = np.zeros(len(x_axis))
    y_axis2 = np.zeros(len(x_axis))
    for _ in range(20):
        for i in range(len(x_axis)):
            n = x_axis[i]
            r = 10
            t = NXTopology(number_of_servers=n,
                           switch_graph_degree=r, number_of_racks=n)

            y_axis1[i] += t.average_shortest_path_length()
            y_axis2[i] += d_star(n, r)

    y_axis1 /= 20
    y_axis2 /= 20

    plt.figure()
    plt.plot(x_axis, y_axis1, label='Observed ASPL', color='red', marker='+')
    plt.plot(x_axis, y_axis2, label='ASPL lower-bound', color='green')
    plt.xticks(range(0, max(x_axis), 20))
    plt.xlabel('Network Size')
    plt.ylabel('Path Length')
    plt.legend()
    plt.xlim((0, 200))
    plt.grid(linestyle='--', color='lightgray')
    plt.ylim((1.2, 2.6))
    plt.show()
    plt.savefig('fig2b.svg')
    '''