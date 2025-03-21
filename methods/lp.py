from scipy.optimize import linprog
import numpy as np
import datetime

def edge_load_transfer(P,D,CA,k):
    edge_no = len(P[0])
    path_no = len(P)
    return_matrix = np.zeros(shape=(edge_no + 1, path_no + 1))

    for i in range(edge_no):
        for j in range(0, path_no):
            flow_index = j // k
            return_matrix[i][j] = P[j][i] * D[flow_index]
        return_matrix[i][len(P)] = -CA[i]
    return_matrix[edge_no][len(P)] = 1.0

    return return_matrix

def trans(R):
    edge_no = len(R) - 1
    path_no = len(R[0]) - 1
    res = np.zeros(shape=(edge_no,path_no))
    for i in range(edge_no):
        for j in range(path_no):
            res[i][j] = R[i][j]
    return res


def eq_one_transfer(MA, k):
    length = len(MA) - 1
    res = np.zeros(shape=(length//k, length + 1))
    for i in range(len(res)):
        for j in range(k):
            index = i * k + j
            res[i][index] = 1.0
    return res

def lower_bound(MA):
    res = []
    for i in range(len(MA)):
        res.append((0, None))
    return res


class RUN_LP(object):
    def __init__(self, net, path_path, traffic_path, scale, ca_path, k):
        self.count = 0
        self.max_util = 0
        self.net = net
        self.path_path = path_path
        self.traffic_path = traffic_path
        self.scale = scale
        self.ca_path = ca_path
        self.k = k
        print("verify the framework with linear optimizer.")

    def run(self):
        P = np.load(self.path_path,allow_pickle=True) # path
        D = np.load(self.traffic_path,allow_pickle=True)[4580] * self.scale # traffic
        CA = np.load(self.ca_path,allow_pickle=True) # capacity

        MA0 = np.zeros(len(P))
        MA0 = np.append(MA0, 0)
        c = np.zeros(len(MA0))
        c[len(MA0) - 1] = 1.0
        
        # traffic less than capacity 
        A_ub = edge_load_transfer(P, D, CA, self.k)
        A0 = trans(A_ub)
        b_ub = np.zeros(len(P[0]) + 1)
        b_ub[len(P[0])] = 1.0

        # Sum of splitting ratios equals to 1 for each flow 
        A_eq = eq_one_transfer(MA0, self.k)
        b_eq = np.ones(len(P)//self.k) * 1.0

        bounds = lower_bound(MA0)
        start_time = datetime.datetime.now()
        res = linprog(c=c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds)
        end_time = datetime.datetime.now()

        result = res.x
        # result = result[0:len(result)-1]
        # result = np.ones(len(result)) * 0.25
        # print(np.sum(D)/1000000000)
        # print(np.sum(A0)/1000000000)
        # print(np.sum(np.matmul(A0,result))/1000000000)

        return res