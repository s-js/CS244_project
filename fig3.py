import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from graph import NXTopology, theoretical_upper_bound, TrafficType, d_star

if __name__ == "__main__":
    x_axis = list(range(5, 1800, 2))
    ys = []
    fig, ax1 = plt.subplots()
    ax1.set_xlabel('Network Size (log scale)')
    ax1.set_ylabel('Path Length')
    ax1.set_xscale('log')
    ax1.set_xticks([17, 53, 161, 485, 1457])
    ax1.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
    plt.grid(linestyle='--', color='lightgray')
    ax1.set_xlim((5, 1800))
    ax2 = ax1.twinx()
    ax2.set_ylabel('Ratio (observed / bound)')
    ax1.set_yticks(range(1, 7))
    ax1.set_ylim((1, 6))
    ax2.set_ylim(0.96, 1.2)
    parameters = [
        {
            'label': 'Observed ASPL',
            'color': 'red',
            'marker': '',
            'linestyle': '-',
            'axis':ax1
        },
        {
            'label': 'ASPL Lower Bound',
            'color': 'green',
            'marker': '.',
            'linestyle': ':',
            'axis':ax1
        },
        {
            'label': 'Ratio',
            'color': 'blue',
            'marker': '.',
            'linestyle': '-',
            'axis':ax2
        }
    ]
    y_axis1 = []
    y_axis2 = []
    y_axis3 = []
    r = 4
    for number_of_switches in x_axis:
        print('n = {}'.format(number_of_switches))
        number_of_servers = number_of_switches
        t = NXTopology(number_of_servers=number_of_servers,
                       switch_graph_degree=r, number_of_racks=number_of_switches)
        # print(t.G.edges)
        # print(t.sender_to_receiver)
        aspl = t.average_shortest_path_length()
        lower_bound = d_star(number_of_switches, r)
        ratio = float(aspl)/lower_bound
        y_axis1.append(aspl)
        y_axis2.append(lower_bound)
        y_axis3.append(ratio)
        print('ratio = {}'.format(ratio))
    ys.append(y_axis1)
    ys.append(y_axis2)
    ys.append(y_axis3)

    
    for i in range(len(ys)):
        parameters[i]['axis'].plot(x_axis, ys[i], marker=parameters[i]['marker'],
                 label=parameters[i]['label'], color=parameters[i]['color'], linestyle=parameters[i]['linestyle'])
    
    fig.legend(loc='lower right',bbox_to_anchor=(0.9, 0.15))
    plt.savefig("fig3.svg")
    plt.show()
