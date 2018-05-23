import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import networkx as nx
from itertools import islice
import numpy as np
import random
from scipy.optimize import linprog


class NXTopology:

    def __init__(self, number_of_servers=686, switch_graph_degree=14,  number_of_racks=40, number_of_links=None):
        self.number_of_servers = number_of_servers
        self.switch_graph_degree = switch_graph_degree  # k
        if number_of_links is not None:
            self.number_of_racks = (
                2 * number_of_links) // self.switch_graph_degree
        else:
            self.number_of_racks = number_of_racks

        self.number_of_servers_in_rack = int(
            np.ceil(float(self.number_of_servers) / self.number_of_racks))
        self.number_of_switch_ports = self.number_of_servers_in_rack + \
            self.switch_graph_degree  # r

        self.G = nx.random_regular_graph(
            self.switch_graph_degree, self.number_of_racks)
        # sender_to_receiver[i] = j <=> i sends message to j
        self.sender_to_receiver = self.random_derangement(
            self.number_of_servers)

        print("number_of_servers_in_rack = " +
              str(self.number_of_servers_in_rack))
        print("number_of_switch_ports = " + str(self.number_of_switch_ports))
        print("RRG has " + str(self.number_of_racks) + " nodes with degree " +
              str(self.switch_graph_degree) + " and " + str(self.G.number_of_edges()) + " edges")

    def get_rack_index(self, server_index):
        return server_index % self.number_of_racks

    def random_derangement(self, n):
        while True:
            array = list(range(n))
            for i in range(n - 1, -1, -1):
                p = random.randint(0, i)
                if array[p] == i:
                    break
                else:
                    array[i], array[p] = array[p], array[i]
            else:
                if array[0] != 0:
                    return array

    def calculate_all_paths(self):
        for sender in range(len(self.sender_to_receiver)):
            receiver = self.sender_to_receiver[sender]
            node1 = self.get_rack_index(sender)
            node2 = self.get_rack_index(receiver)
            # print(node1, node2)
            shortest_paths = list(
                islice(nx.shortest_simple_paths(self.G, node1, node2), 64))


def to_vector_index(n, i, l, k):
    return i * (n**2) + l * n + k


if __name__ == "__main__":
    n = 40
    t = NXTopology(number_of_servers=n*10,
                   switch_graph_degree=5, number_of_racks=n)
    print(t.G.edges)
    print(t.sender_to_receiver)

    D = np.zeros(shape=(n, n))

    for i in range(len(t.sender_to_receiver)):
        sender_switch = t.get_rack_index(i)
        receiver_switch = t.get_rack_index(t.sender_to_receiver[i])
        # print(sender_switch, receiver_switch)
        D[sender_switch, receiver_switch] += 1

    #np.set_printoptions(threshold=np.nan)
    print(D)
    
    C = np.zeros(shape=(n ** 3 + 1))
    C[-1] = -1
    A_eq = np.zeros(shape=(n ** 2, n ** 3+1))
    b_eq = np.zeros(shape=(n ** 2))

    for i in range(n):
        for l in range(n):
            idx = i * n + l
            for k in t.G.neighbors(l):
                A_eq[idx, to_vector_index(n, i, l, k)] = 1
                A_eq[idx, to_vector_index(n, i, k, l)] = -1
            # coefficent of Z:
            if l == i:
                A_eq[idx, -1] = -np.sum(D[i, :])
            else:
                A_eq[idx, -1] = D[i, l]

    A_up = np.zeros(shape=(len(t.G.edges), n ** 3 + 1))
    b_up = np.ones(shape=(len(t.G.edges)))  # link capacities
    idx = 0
    for (l, k) in t.G.edges:
        for i in range(n):
            A_up[idx, to_vector_index(n, i, l, k)] = 0.5
            A_up[idx, to_vector_index(n, i, k, l)] = -0.5
        idx += 1
    
    print("C = "+str(C))
    print("A_eq = "+str(A_eq))
    print("b_eq = "+str(b_eq))
    print("A_up = "+str(A_up))
    print("b_up = " + str(b_up))

    res = linprog(c=C, A_ub=A_up, b_ub=b_up, A_eq=A_eq,
                  b_eq=b_eq, bounds=[(0, 1) for _ in range(len(C))], options={"disp": True})

    print(res)
    
