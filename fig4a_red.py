import numpy as np
from graph import *

x_axis=np.array(list(range(4,9)))/5
y_ax=[]
number_of_switches=[40,20]
number_of_ports_per_switch=[15,30]
number_of_servers=510
color="Red"

for x in x_axis:
	t=NXTopology_het(number_of_servers=number_of_servers,number_of_switches=number_of_switches, number_of_ports_per_switch=number_of_ports_per_switch,ratio_of_servers_in_largest_switch_to_expected=x)
	print("4a {} plot- server bias = {}".format(color,x))
	y=t.get_max_min_throughput()
	y_ax+=[y]
y_ax=np.array(y_ax)
np.save("4a_{}".format(color),y_ax)

