import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# making D
def transfer_traffic(a,N):
    len_a = len(a)
    res = np.zeros(shape=(len_a,N*N))
    for i in range(len_a):
        cnt = 0
        for j in range(N):
            for k in range(N):
                if j == k:
                    res[i][j*N+k] = 0
                else:
                    res[i][j*N+k] = a[i][cnt]
                    cnt += 1
    return res

a = np.load('datasource_sndlib/abilene_5min/traffic/all_demand.npy',allow_pickle=True)[800:5800]
b = transfer_traffic(a,12)
# np.save('../dataset/traffic/abilene.npy',b,allow_pickle=True)
b = np.sum(a,axis=1)
plt.plot(b,label='abilene')
plt.legend()
plt.savefig('test1.png')
# print(b[50] * 2000000)
a = np.load('datasource_sndlib/geant_15min/traffic/all_demand.npy',allow_pickle=True)[5498:10498]
b = transfer_traffic(a,22)
# np.save('../dataset/traffic/geant.npy',b,allow_pickle=True)
print(len(a))
b = np.sum(a,axis=1)
plt.plot(b,label='geant')
plt.legend()
plt.savefig('test2.png')

# Making C
def make_capacity(topo):
    net = nx.read_graphml(topo)
    res = np.ones(len(net.edges) * 2) * 1073741824.0
    print(len(res))
    return res
print('here')
np.save('../dataset/capacities/intellifiber.npy',make_capacity('../dataset/topologies/intellifiber.graphml'),allow_pickle=True)
