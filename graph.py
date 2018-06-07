import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import networkx as nx
from itertools import islice
import numpy as np
import cvxopt as cvx
import random
from scipy.optimize import linprog
from enum import Enum


class TrafficType(Enum):
    ALL_TO_ALL = 0
    PERMUTATION = 1


def to_vector_index(n, i, l, k):
    return i * (n**2) + l * n + k


def theoretical_upper_bound(number_of_switches, degree, number_of_servers, traffic_type):

    # f is the number of flows
    if traffic_type == TrafficType.ALL_TO_ALL:
        f = number_of_servers * (number_of_servers - 1)
    elif traffic_type == TrafficType.PERMUTATION:
        f = number_of_servers

    return number_of_switches * degree / (f * d_star(number_of_switches, degree))


def d_star(N, r):
    '''
    To calculate theoretical upper bound for throughput in any regular graph network topology
    '''
    temp = 1
    p = 1
    k = 2
    while True:
        if N - 1 - r * temp < 0:
            k -= 1
            print('k = '+str(k))
            break
        R = N - 1 - r * temp
        p *= (r-1)
        temp += p
        k += 1

    R = 0
    for j in range(1, k):
        R += r * (r - 1)**(j - 1)
    R = N - 1 - R
    print('R = '+str(R))

    s = 0
    for j in range(1, k):
        s += j * r * (r - 1)**(j - 1)

    return (s+k*R)/(N-1)


def check_if_edge_possible(G, node, total_nodes, remaining_ports_per_switch_full_list):
    for idx in range(node+1, total_nodes):
        if ([node, idx] not in np.array(G.edges).tolist()) and (remaining_ports_per_switch_full_list[idx] > 0):
            return True
    return False


def random_derangement(n):
    '''
    To create the random traffic permutation
    '''
    while True:
        array = list(range(n))
        for i in range(n - 1, -1, -1):
            p = random.randint(0, i)
            if array[p] == i:  # to prevent self-directed traffic
                break
            else:
                array[i], array[p] = array[p], array[i]
        else:
            if array[0] != 0:
                return array


class NXTopology:
    '''
    NXTopology stores all information of our random topology
    '''

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

        print("number_of_servers_in_rack = " +
              str(self.number_of_servers_in_rack))
        print("number_of_switch_ports = " + str(self.number_of_switch_ports))
        print("RRG has " + str(self.number_of_racks) + " nodes with degree " +
              str(self.switch_graph_degree) + " and " + str(self.G.number_of_edges()) + " edges")

    def RandomPermutation(self, size):
        screw = 1
        ls = []

        while screw == 1:
            screw = 0
            ls = []
            for i in range(size):
                ls.append(i)
            for i in range(size):
                ok = 0
                k = 0
                cnt = 0
                while ok == 0:
                    # choose a shift in [0, size-1-i]
                    k = random.randint(0, size-i-1)
                    # check if we should swap i and i+k
                    if self.get_rack_index(i) != self.get_rack_index(ls[i+k]):
                        ok = 1
                    cnt += 1
                    if cnt > 50:
                        screw = 1
                        ok = 1

                # swap i's value and i+k's value
                buf = ls[i]
                ls[i] = ls[i+k]
                ls[i+k] = buf
        return ls

    def get_rack_index(self, server_index):
        '''
        given server index, returns the ToR switch index it is connected to
        '''
        return server_index % self.number_of_racks

    def average_shortest_path_length(self):
        '''
        To get ASPL of RRG
        '''
        s = 0
        c = 0
        for i in range(self.number_of_racks):
            for j in range(i+1, self.number_of_racks):
                s += len(nx.shortest_path(self.G, i, j))-1
                c += 1
        return float(s)/c

    def get_max_min_throughput(self, traffic_type):
        '''
        Getting the max-min throughput using a linear program
        '''
        n = self.number_of_racks
        r = self.switch_graph_degree

        # sender_to_receiver[i, j] = 1 <=> i sends message to j
        if traffic_type == TrafficType.PERMUTATION:
            self.sender_to_receiver = self.RandomPermutation(
                self.number_of_servers)
            self.sender_to_receiver = np.eye(self.number_of_servers)[
                self.sender_to_receiver]
        elif traffic_type == TrafficType.ALL_TO_ALL:
            self.sender_to_receiver = np.ones(shape=(
                self.number_of_servers, self.number_of_servers))-np.eye(self.number_of_servers)

        D = np.zeros(shape=(n, n))
        for i in range(self.sender_to_receiver.shape[0]):
            for j in range(self.sender_to_receiver.shape[1]):
                if self.sender_to_receiver[i][j] == 0:
                    continue
                sender_switch = self.get_rack_index(i)
                receiver_switch = self.get_rack_index(j)
                #receiver_switch = self.get_rack_index(self.sender_to_receiver[i])
                # print(sender_switch, receiver_switch)
                if sender_switch != receiver_switch:
                    #print(sender_switch, receiver_switch)
                    D[sender_switch, receiver_switch] += 1
                else:
                    print('switches from the same rack are sending data')

        # np.set_printoptions(threshold=np.nan)
        # D = np.eye(n)[random_derangement(n)]*self.number_of_servers_in_rack
        # print(self.sender_to_receiver)
        # print('D = '+str(D))

        # We now convert the linear programming problem to the following form:
        # Minimize:     c^T * x
        # Subject to:   A_up * x <= b_up
        #               A_eq * x == b_eq

        # ignore f_i,l,k and only minimize -Z (i.e. maximize Z):
        C = np.zeros(shape=(n**3 + 1))
        C[-1] = -1
        C = cvx.matrix(C)

        A_eq = cvx.spmatrix([], [], [], size=(n**2+1, n**3 + 1))
        b_eq = cvx.spmatrix([], [], [], size=(n**2+1, 1))

        idx = 0
        for i in range(n):
            for l in range(n):

                for k in self.G.neighbors(l):
                    A_eq[idx, to_vector_index(n, i, l, k)] = 1
                    A_eq[idx, to_vector_index(n, i, k, l)] = -1
                # coefficent of Z:
                if l == i:
                    A_eq[idx, -1] = -np.sum(D[i, :])
                else:
                    A_eq[idx, -1] = D[i, l]
                #print(A_eq[idx, -1])
                idx += 1

        for l in range(n):
            for k in range(n):
                if(l, k) not in self.G.edges():
                    for i in range(n):
                        A_eq[idx, to_vector_index(n, i, l, k)] = 1
                        A_eq[idx, to_vector_index(n, i, k, l)] = 1

        A_up = cvx.spmatrix([], [], [], size=(
            len(self.G.edges())*2 + n**3 + 1, n**3 + 1))
        b_up = cvx.spmatrix([], [], [], size=(
            len(self.G.edges())*2 + n**3 + 1, 1))
        idx = 0

        for l in range(n):
            for k in range(n):
                if(l, k) not in self.G.edges():
                    continue
                for i in range(n):
                    A_up[idx, to_vector_index(n, i, l, k)] = 1
                    # A_up[idx, to_vector_index(n, i, k, l)] = 0.5
                    b_up[idx] = 1  # C(e)

                idx += 1

        for i in range(n**3 + 1):
            A_up[idx, i] = -1
            idx += 1
        #A_up[len(self.G.edges) + n**3 + 1, n**3] = 1

        #b_up[idx:idx + n**3 + 1] = np.zeros(shape=(n**3 + 1))
        #b_up[-1] = 2

        #print("C = "+str(C))
        #print("A_eq = "+str(A_eq))
        #print("b_eq = "+str(b_eq))
        #print("A_up = "+str(A_up))
        #print("b_up = " + str(b_up))

        b_up = cvx.matrix(b_up)
        b_eq = cvx.matrix(b_eq)

        sol = cvx.solvers.lp(C, A_up, b_up, A_eq, b_eq, solver='glpk')
        '''
        # calculate and print flow of each edge
        for (l, k) in self.G.edges():
            for i in range(n):
                if np.abs(sol['x'][to_vector_index(n, i, l, k)]) > 1e-8:
                    print('flow {} from {} to {} is {}'.format(
                        i, l, k, sol['x'][to_vector_index(n, i, l, k)]))
                if np.abs(sol['x'][to_vector_index(n, i, k, l)]) > 1e-8:
                    print('flow {} from {} to {} is {}'.format(
                        i, k, l, sol['x'][to_vector_index(n, i, k, l)]))
        '''
        return sol['x'][-1]


class NXTopology_het:
    '''
    NXTopology_het stores all information of our random heteregenous topology
    '''

    def __init__(self, number_of_servers=400, number_of_switch_types=2, number_of_switches=[40,20],
            number_of_ports_per_switch=[10,30],ratio_of_servers_in_largest_switch_to_expected=1.0, cross_cluster_bias=1.0):
        
        self.number_of_switch_types = number_of_switch_types
        self.number_of_servers = number_of_servers
        assert (number_of_switch_types == len(number_of_switches)
                ), "number_of_switch_types != length of number_of_switches list"
        assert (number_of_switch_types == len(number_of_switches)
                ), "number_of_switch_types != length of number_of_ports_per_switch list"        
        self.cross_cluster_bias = cross_cluster_bias # cross_cluster just for two clusters
        
        ## Getting number of servers per switch type, with a bias factor
        number_of_switches = np.array(number_of_switches)
        number_of_ports_per_switch = np.array(number_of_ports_per_switch)
        server_dist=number_of_switches*number_of_ports_per_switch
        server_dist = server_dist/np.sum(server_dist)
        total_servers_per_switch_type = server_dist*number_of_servers
        bias_in_servers = total_servers_per_switch_type[-1]*(ratio_of_servers_in_largest_switch_to_expected-1)
        total_servers_per_switch_type_after_bias_inc=total_servers_per_switch_type[:-1] - bias_in_servers * \
            (server_dist[:-1]/sum(server_dist[:-1]))
        assert ((total_servers_per_switch_type[:-1] >= bias_in_servers*(server_dist[:-1]/sum(server_dist[:-1]))).all()), "Bias in server distribution too large"

        total_servers_per_switch_type_after_bias = np.append(
            total_servers_per_switch_type_after_bias_inc, total_servers_per_switch_type[-1]+bias_in_servers)
    
        server_per_switch_type = total_servers_per_switch_type_after_bias/number_of_switches
        
        # Rounding number of servers per switch type
        for elem in range(len(server_per_switch_type)):
            if elem==len(server_per_switch_type)-1:
                server_per_switch_type[elem] = np.ceil(server_per_switch_type[elem])
            else:
                server_per_switch_type[elem] = np.floor(server_per_switch_type[elem])
        server_per_switch_type.astype(int)

        if cross_cluster_bias>1:
            assert(len(number_of_switches)==2),"Cannot support cross-cluster bias for more than 2 switch types"
            self.generate_graph_with_cross_bias()
        else:
            self.generate_graph_without_cross_bias()
        # !!! Only works for two types of switches for now !!! #
        ## Dealing with biases in cross-cluster links
        remaining_ports_per_switch = number_of_ports_per_switch-server_per_switch_type

        # getting ratio of intra to total remaining ports per small switch, with link bias
        total_remaining_ports_per_switch_type = remaining_ports_per_switch*number_of_switches
        x = (total_remaining_ports_per_switch_type[0]
             * (total_remaining_ports_per_switch_type[0]-1))/2.0
        y = total_remaining_ports_per_switch_type[0] * \
            total_remaining_ports_per_switch_type[1]
        z = (total_remaining_ports_per_switch_type[1]
             * (total_remaining_ports_per_switch_type[1]-1))/2.0
        bias_in_cross_links=(1-cross_cluster_bias)*y
        y = y + bias_in_cross_links
        x = x - x/(x+z)*bias_in_cross_links
        z = z - z/(x+z)*bias_in_cross_links
        assert(x>0 and y>0 and z>0),"cross-cluster link bias too extreme"
        ratio_of_intra=x/(x+y)
        y_max=np.sum(total_remaining_ports_per_switch_type)/2*y/(x+y+z)

        # getting upper bound on intra and cross edges per small switch, use y_max to stop when we populate that amount of intra links in small switch
        intra_edges_per_small_switch = np.ceil(remaining_ports_per_switch[0]*ratio_of_intra)
        #cross_edges_per_small_switch = remaining_ports_per_switch[0] - intra_edges_per_small_switch
        initial_remaining_edges_per_small_switch = int(remaining_ports_per_switch[0])
        remaining_ports_per_switch_full_list = [int(remaining_ports_per_switch[0])] * number_of_switches[0]\
            + [int(remaining_ports_per_switch[1])]*number_of_switches[1]
        #used to signal we reached maximum expected intra links in small switch cluster
        reached_limit_intra_small=False
        current_intra_small=0
        
        #Populating the graph with the links, with bias
        self.G=nx.Graph()

        #start with small switches
        for switch in range(number_of_switches[0]):
            edges_list=[]

            #for each switch, start with intra-cluster links
            while ( (initial_remaining_edges_per_small_switch-remaining_ports_per_switch_full_list[switch] < intra_edges_per_small_switch) and 
            (switch != number_of_switches[0]-1) and 
            (reached_limit_intra_small==False) ):
                edge = [switch]
                switch_rcv=random.randint(switch+1,number_of_switches[0]-1)
                while (edge+[switch_rcv] in edges_list) or (initial_remaining_edges_per_small_switch-remaining_ports_per_switch_full_list[switch_rcv] >= intra_edges_per_small_switch):
                    switch_rcv = random.randint(
                        switch+1, number_of_switches[0]-1)
                edge=edge+[switch_rcv]
                edges_list.append(edge)
                remaining_ports_per_switch_full_list[switch] -= 1
                remaining_ports_per_switch_full_list[switch_rcv] -= 1
                current_intra_small+=1
                if current_intra_small == y_max:  # TODO
                    reached_limit_intra_small=True
                

            #then, for each switch, add the cross-cluster links
            while (remaining_ports_per_switch_full_list[switch] > 0):
                edge = [switch]
                switch_rcv = random.randint(
                    number_of_switches[0], number_of_switches[0]+number_of_switches[1]-1)
                while (edge+[switch_rcv] in edges_list) or (remaining_ports_per_switch_full_list[switch_rcv] == 0):
                    switch_rcv = random.randint(
                        number_of_switches[0], number_of_switches[0]+number_of_switches[1]-1)
                edge = edge+[switch_rcv]
                edges_list.append(edge)
                remaining_ports_per_switch_full_list[switch] -= 1
                remaining_ports_per_switch_full_list[switch_rcv] -= 1

            self.G.add_edges_from(edges_list)

        #now, add links for just the intra-(large switch cluster) links
        for switch_big in range(number_of_switches[0], number_of_switches[0]+number_of_switches[1]):
            # some ports in the last switches might stay unconnected because of boundary conditions
            while ( (remaining_ports_per_switch_full_list[switch_big] > 0) and 
                    (switch_big != number_of_switches[0]+number_of_switches[1]-1) and 
                    (np.sum(remaining_ports_per_switch_full_list)-remaining_ports_per_switch_full_list[switch_big]>0) and
                    check_if_edge_possible(self.G, switch_big, number_of_switches[0]+number_of_switches[1], remaining_ports_per_switch_full_list)):
                edge = [switch_big]
                switch_rcv = random.randint(
                    switch_big+1, number_of_switches[0]+number_of_switches[1]-1)
                while (edge+[switch_rcv] in np.array(self.G.edges).tolist()) or (remaining_ports_per_switch_full_list[switch_rcv] == 0): #TODO
                    switch_rcv = random.randint(switch_big+1, number_of_switches[0]+number_of_switches[1]-1)
                edge = edge+[switch_rcv]
                self.G.add_edge(switch_big, switch_rcv)
                remaining_ports_per_switch_full_list[switch_big] -= 1
                remaining_ports_per_switch_full_list[switch_rcv] -= 1
        pass

    def generate_graph_without_cross_bias(self):
        #TODO 
        pass

    def generate_graph_with_cross_bias(self):
        pass

        
    """     
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
        self.sender_to_receiver = random_derangement(self.number_of_servers)

        print("number_of_servers_in_rack = " +
              str(self.number_of_servers_in_rack))
        print("number_of_switch_ports = " + str(self.number_of_switch_ports))
        print("RRG has " + str(self.number_of_racks) + " nodes with degree " +
              str(self.switch_graph_degree) + " and " + str(self.G.number_of_edges()) + " edges") 
    """

    def get_rack_index(self, server_index):
        '''
        given server index, returns the ToR switch index it is connected to
        '''
        # return server_index % self.number_of_racks

    def average_shortest_path_length(self):
        '''
        To get ASPL of RRG
        '''
        # s = 0
        # c = 0
        # for i in range(self.number_of_racks):
        #     for j in range(i+1, self.number_of_racks):
        #         s += len(nx.shortest_path(self.G, i, j))-1
        #         c += 1
        # return float(s)/c

    def get_max_min_throughput(self):
        '''
        Getting the max-min throughput using a linear program
        '''
