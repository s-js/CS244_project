import numpy as np
from graph2 import *

x_axis=np.array(list(range(2,7)))/5
y_ax=[]
number_of_switches=[40,20]
number_of_ports_per_switch=[10,30]
number_of_servers=500
color="Green"

for x in x_axis:
	t=NXTopology_het(number_of_servers=number_of_servers,number_of_switches=number_of_switches, number_of_ports_per_switch=number_of_ports_per_switch,cross_cluster_bias=x)
	print("6c {} plot- server bias = {}".format(color,x))
	y=t.get_max_min_throughput()
	y_ax+=[y]
y_ax=np.array(y_ax)
np.save("6c_{}".format(color),y_ax)

