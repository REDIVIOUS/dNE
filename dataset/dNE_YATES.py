import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# generate yates format topology
def gen_topo(node_num,capa):
    node_num += 1
    G = nx.read_graphml('../dataset/topologies/intellifiber.graphml')
    print(len(G.nodes))
    a = {}
    for i in range(1,node_num):
        a[list(G.nodes)[i-1]] = i

    res = []
    for j in range(len(G.edges)):
        res.append((a[list(G.edges)[j][0]], a[list(G.edges)[j][1]]))
    print(len(res))
    print(len(G.edges))

    def changes(a):
        b = a % 16
        c = a//16
        dict0 = {10:'a',11:'b',12:'c',13:'d',14:'e',15:'f'}
        if b >= 10:
            res0 = dict0[b]
        else:
            res0 = str(b)
        return str(c) + res0

    with open("../../yates/data/topologies/new_intellifiber.dot","w") as f:
        f.write('digraph topology {\n')
        for i in range(1,node_num):
            f.write('s'+str(i)+'[type=switch,id='+str(i)+',mac=\"20:00:00:00:00:' +changes(i)+'\"];\n')
        for i in range(1,node_num):
            f.write('h'+str(i)+'[type=host,mac=\"00:00:00:00:00:'+changes(i)+'\",ip=\"10.0.0.'+str(i)+'\"];\n')
        for i in range(len(res)):
            f.write('s'+str(res[i][0])+' -> '+'s'+str(res[i][1]) + ' [cost=1, capacity=\"'+capa+'\"];\n')
            f.write('s'+str(res[i][1])+' -> '+'s'+str(res[i][0]) + ' [cost=1, capacity=\"'+capa+'\"];\n')
        for i in range(1,node_num):
            f.write('h'+str(i) + ' -> ' + 's'+str(i) + ' [cost=1, capacity="1000000Gbps"];\n')
            f.write('s'+str(i) + ' -> ' + 'h'+str(i) + ' [cost=1, capacity="1000000Gbps"];\n')
        f.write('}')


def gen_hosts(num_nodes):
    with open("../../yates/data/hosts/new_ittelifiber.hosts","w") as f:
        for i in range(num_nodes):
            f.write('h'+str(i)+'\n')

# generate yates format traffic
def gen_traffic(num_nodes,read_path,write_path):
    # a = np.load('datasource_sndlib/abilene_5min/traffic/all_demand.npy',allow_pickle=True)
    # plt.plot(np.sum(a,axis=1)[31000:31576])
    # plt.savefig('abilene.png')
    # plt.plot(np.sum(a,axis=1)[8500:8692])
    # plt.savefig('geant.png')

    a = np.load(read_path,allow_pickle=True)


    with open(write_path,"w") as f:
        # for i in range(5300, 5800):
        #     cnt = 0
        #     for j in range(num_nodes):
        #         for k in range(num_nodes):
        #             if j == k:
        #                 if k != num_nodes - 1:
        #                     f.write('0.0 ')
        #                 else:
        #                     f.write('0.0')
        #             else:
        #                 f.write(str(a[i][cnt])+' ')
        #                 cnt += 1
        #     f.write('\n')
        for i in range(4500, 5000):
            cnt = 0
            for j in range(num_nodes * num_nodes):
                if j == num_nodes * num_nodes - 1:
                    f.write(str(a[i][j])+'\n')
                else:
                    f.write(str(a[i][j])+' ')

# read yates result files
def read_file(topo):
    data_MLU = pd.read_table('../../yates/data/results/'+ topo +'/MaxCongestionVsIterations.dat',sep='\t')
    data_TP = pd.read_table('../../yates/data/results/'+ topo +'/TotalThroughputVsIterations.dat',sep='\t')
    data_loss = pd.read_table('../../yates/data/results/'+ topo +'/CongestionLossVsIterations.dat',sep='\t')
    data_time = pd.read_table('../../yates/data/results/'+ topo +'/TimeVsIterations.dat',sep='\t')
    # print(data_MLU['max-congestion'])
    # print(data_TP['total-throughput'])
    # print(data_loss['congestion-drop'])
    # print(data_time['time'])
    return np.array(data_MLU['max-congestion']), np.array(data_TP['total-throughput']), np.array(data_loss['congestion-drop']), np.array(data_time['time'])

# read dNE result files
def read_dNE(topo,alpha,method):
    CLoss_file = '../../results/'+method+'/'+topo+'/'+str(alpha)+'_CLoss.npy'
    MLU_file = '../../results/'+method+'/'+topo+'/'+str(alpha)+'_MLU.npy'
    TP_file = '../../results/'+method+'/'+topo+'/'+str(alpha)+'_TP.npy'
    CLoss = np.load(CLoss_file,allow_pickle=True)
    MLU = np.load(MLU_file,allow_pickle=True)
    TP = np.load(TP_file,allow_pickle=True)
    return CLoss, MLU, TP

# # read ecmp result files
# def read_ECMP(topo,alpha,method):
#     CLoss_file = '../../results/'+method+'/'+topo+'/CLoss.npy'
#     MLU_file = '../../results/'+method+'/'+topo+'/MLU.npy'
#     TP_file = '../../results/'+method+'/'+topo+'/TP.npy'
#     CLoss = np.load(CLoss_file,allow_pickle=True)
#     MLU = np.load(MLU_file,allow_pickle=True)
#     TP = np.load(TP_file,allow_pickle=True)
#     return CLoss, MLU, TP

def mluu(mlu, scale):
    mlu0 = np.zeros(500)
    for i in range(500):
        mlu0[i] = mlu[i] * scale
        if mlu0[i] > 1:
            mlu0[i] = 1
    return mlu0
def tpp():
    tp = np.ones(500)
    return tp
def closss():
    closs = np.zeros(500)
    return closs

def mlu_trans(a):
    length = len(a)
    res = []
    for i in range(length):
        if a[i] > 1:
            res.append(1)
        else:
            res.append(a[i])
    return np.array(res)


# plot yates results
def plot_yates_result(topo,length,topo_name):
    mlu, tp, closs, time = read_file(topo)
    cspf_cnt = 0
    ecmp_cnt = 1
    opt_cnt = 2

    cspf_mlu= mlu[length*cspf_cnt:length*cspf_cnt+length]
    # ecmp_mlu= mlu[length*ecmp_cnt:length*ecmp_cnt+length]
    opt_mlu = mlu[length*opt_cnt:length*opt_cnt+length]

    cspf_tp = tp[length*cspf_cnt:length*cspf_cnt+length]
    # ecmp_tp = tp[length*ecmp_cnt:length*ecmp_cnt+length]
    opt_tp = tp[length*opt_cnt:length*opt_cnt+length]

    cspf_closs = closs[length*2:length*3]
    # ecmp_closs = closs[length*ecmp_cnt:length*ecmp_cnt+length]
    opt_closs = closs[length*opt_cnt:length*opt_cnt+length]

    cspf_time = time[length*cspf_cnt:length*cspf_cnt+length]
    ecmp_time = time[length*ecmp_cnt:length*ecmp_cnt+length]
    opt_time = time[length*opt_cnt:length*opt_cnt+length]

    ecmp_closs, ecmp_mlu, ecmp_tp = read_ECMP(topo_name,0,'ECMP')
    ecmp_mlu = mlu_trans(ecmp_mlu)

    dnn_closs, dnn_mlu, dnn_tp = read_dNE(topo_name,0,'DNN')
    dnn_mlu = mlu_trans(dnn_mlu)

    lstm_closs, lstm_mlu, lstm_tp = read_dNE(topo_name,0,'LSTM')
    lstm_mlu = mlu_trans(lstm_mlu)

    cnn_closs, cnn_mlu, cnn_tp = read_dNE(topo_name,0,'CNN')
    cnn_mlu = mlu_trans(cnn_mlu)

    drl_closs, drl_mlu, drl_tp = read_dNE(topo_name,0,'DRL')
    drl_mlu = mlu_trans(drl_mlu)

    fig = plt.figure(figsize=(12,4))
    fig.subplots_adjust(hspace=0.3, wspace=0.3)
    axes = fig.subplots(nrows=1,ncols=7)

    i = 0
    axes[i].plot(cspf_mlu,label='max congestion')
    axes[i].plot(cspf_tp,label='tot throughput')
    axes[i].plot(cspf_closs,label='congestion loss')
    axes[i].set_title('cspf')

    i += 1
    axes[i].plot(ecmp_mlu,label='max congestion')
    axes[i].plot(ecmp_tp,label='tot throughput')
    axes[i].plot(ecmp_closs,label='congestion loss')
    axes[i].set_title('ecmp')
    # print(np.mean(ecmp_mlu))

    i += 1
    axes[i].plot(opt_mlu,label='max congestion')
    axes[i].plot(opt_tp,label='tot throughput')
    axes[i].plot(opt_closs,label='congestion loss')
    axes[i].set_title('optimal')

    i += 1
    axes[i].plot(dnn_mlu,label='max congestion')
    axes[i].plot(dnn_tp,label='tot throughput')
    axes[i].plot(dnn_closs,label='congestion loss')
    axes[i].set_title('dnn')

    i += 1
    axes[i].plot(lstm_mlu,label='max congestion')
    axes[i].plot(lstm_tp,label='tot throughput')
    axes[i].plot(lstm_closs,label='congestion loss')
    axes[i].set_title('lstm')

    i += 1
    axes[i].plot(cnn_mlu,label='max congestion')
    axes[i].plot(cnn_tp,label='tot throughput')
    axes[i].plot(cnn_closs,label='congestion loss')
    axes[i].set_title('cnn')

    i += 1
    axes[i].plot(drl_mlu,label='max congestion')
    axes[i].plot(drl_tp,label='tot throughput')
    axes[i].plot(drl_closs,label='congestion loss')
    axes[i].set_title('drl')
    print(np.mean(drl_mlu))

    lines,labels = fig.axes[-1].get_legend_handles_labels()
    fig.legend(lines,labels,loc='upper center',bbox_to_anchor=(0.5, 1.015),ncol=3)
    fig.savefig(topo+'.png')

    return np.array([np.mean(ecmp_mlu),np.mean(cspf_mlu),np.mean(opt_mlu),np.mean(dnn_mlu)]), np.array([np.mean(ecmp_tp),np.mean(cspf_tp),np.mean(opt_tp),np.mean(dnn_tp)]), np.array([np.mean(ecmp_closs),np.mean(cspf_closs),np.mean(opt_closs),np.mean(dnn_closs)]), np.array([np.mean(ecmp_time),np.mean(cspf_time),np.mean(opt_time)])

# read the traffic from yates
# correlaction noise is 0.2
def read_traffic(path,save_path):
    full_path = '../../' + path
    traffic = np.loadtxt(full_path)
    np.save(save_path,traffic,allow_pickle=True)
    print(traffic.shape)

# read the paths from yates
def read_path(topo, read_path, N, k):
    G = nx.read_graphml(topo)
    a = {}
    for i in range(N): # need to pay attention
        a[list(G.nodes)[i]] = i

    ress = []
    for j in range(len(G.edges)):
        ress.append((a[list(G.edges)[j][0]], a[list(G.edges)[j][1]]))
    print(ress)
    edge_no = len(G.edges) * 2
    check = {}
    for i in range(len(ress)):
        str0 = '(s'+str(ress[i][0])+',s'+str(ress[i][1])+')'
        str1 = '(s'+str(ress[i][1])+',s'+str(ress[i][0])+')'
        check[str0] = i*2
        check[str1] = i*2 + 1

    with open(read_path, "r") as f:
        data = f.readlines()
        res = np.zeros(shape=(N * N * k, edge_no))
        i = 0
        while i < len(data):
            inc = 0
            if data[i][0] == 'h':
                x = data[i].split(' -> ')
                host1 = int((x[0].split('h'))[1])
                host2 = int((((x[1].split(' :'))[0]).split('h'))[1])
                index = (host1*N + host2) * k
                Flag = True
                for j in range(k):
                    if data[i+j+1][0] == '[' and Flag:
                        inc = j
                        x = data[i+j+1].split(' @ ')[0]
                        x = x.split(', ')
                        x = x[1:len(x)-1]
                        path_len = len(x)
                        for w in range(path_len):
                            res[index+j][check[x[w]]] = 1.0
                    else:
                        Flag = False
                        x = data[i+1+inc].split(' @ ')[0]
                        x = x.split(', ')
                        x = x[1:len(x)-1]
                        path_len = len(x)
                        for w in range(path_len):
                            res[index+j][check[x[w]]] = 1.0
            i = i + inc + 1
        return res

def random_noise_generator(load_path, save_path):
    import random
    data = np.load(load_path, allow_pickle=True)
    for i in range(len(data)):
        a = random.random()
        a = a/40 - 1/20
        data[i] = data[i] * (1-a)
        for j in range(len(data[0])):
            a = random.random()
            a = a/80 - 1/40
            data[i][j] = data[i][j] * (1-a)
    np.save(save_path,data,allow_pickle=True)
    return data

# # generate yates format topology
# gen_topo(73,'1Gbps')

# # generate hosts
# gen_hosts(73)

# # generate yates format traffic
# gen_traffic(num_nodes=51,read_path="../dataset/traffic/iris-08.npy",
#             write_path="../../yates/data/demands/actual/new_iris-08.txt")

# # read yates result files
# mlu, tp, closs, time = read_file('Intellifiber-1')
# # print(len(mlu))
# # mlu = mluu(mlu,scale=185.09949097639984)
# # print('ecmp:\n','mlu: ',np.mean(mlu[500:1000]),'\ttp: ',np.mean(tp[500:1000]),'\tcloss: ',np.mean(closs[500:1000]),'\ttime: ',np.mean(time[500:1000]))
# # print('min mlu: ',np.min(mlu[500:1000]),'\tmax mlu: ',np.max(mlu[500:1000]))
# start = 500
# end = 1000
# print('optimal:\n','mlu: ',np.mean(mlu[start:end]),'\ttp: ',np.mean(tp[start:end]),'\tcloss: ',np.mean(closs[start:end]),'\ttime: ',np.mean(time[start:end])*1000)
# print('min mlu: ',np.min(mlu[start:end]),'\tmax mlu: ',np.max(mlu[start:end]))
# # print('ratio: ',np.max(mlu[start:end])/np.min(mlu[start:end]))
# # print('scale: ',0.4/np.min(mlu[start:end]))
# # print('optimal:\n','mlu: ',np.mean(mlu[1000:1500]),'\ttp: ',np.mean(tp[1000:1500]),'\tcloss: ',np.mean(closs[1000:1500]),'\ttime: ',np.mean(time[1000:1500]))
# # print('min mlu: ',np.min(mlu[1000:1500]),'\tmax mlu: ',np.max(mlu[1000:1500]))

# plot yates results
mlu, tp, closs, time = plot_yates_result('Intellifiber-1',500,'Intellifiber') 

# # read the traffic from yates
# # correlaction noise is 0.2
# read_traffic(path='yates/data/prediction/matrix/new_iris-matrix/new_iris_mergelen_12_corr_noise_0.00',save_path='../dataset/traffic/iris-0.npy')
# a = np.load('../dataset/traffic/iris-07.npy',allow_pickle=True)
# plt.plot(np.sum(a,axis=1)[4000:5000])
# print(np.min(np.sum(a,axis=1)[4500:5000]),np.max(np.sum(a,axis=1)[4500:5000]))
# plt.savefig('test.png')

# random_noise_generator(load_path='../dataset/traffic/iris-0.npy',save_path='../dataset/traffic/iris-08.npy')

# a = np.load('../dataset/traffic/iris-07.npy',allow_pickle=True)
# b = np.sum(a,axis=1)[4500:5000]
# plt.plot(b)
# plt.savefig('test.png')
# print(min(b),max(b))


# # transfer from yates format
# res = read_path(topo='../dataset/topologies/intellifiber.graphml',read_path="../dataset/paths/Smore_Format/intellifiber.txt",N=73,k=4)
# np.save('../dataset/paths/intellifiber.npy',res,allow_pickle=True)