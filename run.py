from methods.ecmp import RUN_ECMP
from methods.lp import RUN_LP
from methods.dnn import RUN_DNN
from methods.lstm import RUN_LSTM
from methods.cnn import RUN_CNN
from methods.drl import RUN_DRL
import networkx as nx
import numpy as np

data = {}

def ECMP(topo, run_flag=True, test_flag=False):
    if topo == 'abilene':
        net = nx.read_graphml('dataset/topologies/abilene.graphml')
        path_path = 'dataset/paths/abilene.npy'
        traffic_path = 'dataset/traffic/abilene.npy'
        ca_path = 'dataset/capacities/abilene.npy'
        save_dir = 'results/ECMP/Abilene/'
        train_size = 4500
        scale = 1150383
        k = 4
        # run model
        running = RUN_ECMP(net=net, path_path=path_path, save_dir=save_dir, traffic_path=traffic_path, scale=scale, ca_path=ca_path, k=k)
        if run_flag == True:
            running.run()
        # if test_flag == True:
        #     # test run time
        #     model_path = 'results/Abilene/0_model.pt'
        #     running_time, times = running.check_decision_time(model_path)
        #     print(running_time/times*1000)

        # test metrics
        mlu_path = 'results/ECMP/Abilene/MLU.npy'
        tp_path = 'results/ECMP/Abilene/TP.npy'
        closs_path = 'results/ECMP/Abilene/CLoss.npy'
        mlu, tp, closs = running.check_metric(mlu_path, tp_path, closs_path)
        print(mlu,tp,closs)

    if topo == 'geant':
        net = nx.read_graphml('dataset/topologies/geant.graphml')
        path_path = 'dataset/paths/geant.npy'
        traffic_path = 'dataset/traffic/geant.npy'
        ca_path = 'dataset/capacities/geant.npy'
        save_dir = 'results/ECMP/Geant/'
        train_size = 4500
        scale = 189955
        k = 4
        # run model
        running = RUN_ECMP(net=net, path_path=path_path, save_dir=save_dir, traffic_path=traffic_path, scale=scale, ca_path=ca_path, k=k)
        if run_flag == True:
            running.run()
        # if test_flag == True:
        #     # test run time
        #     model_path = 'results/Abilene/0_model.pt'
        #     running_time, times = running.check_decision_time(model_path)
        #     print(running_time/times*1000)

        # test metrics
        mlu_path = 'results/ECMP/Geant/MLU.npy'
        tp_path = 'results/ECMP/Geant/TP.npy'
        closs_path = 'results/ECMP/Geant/CLoss.npy'
        mlu, tp, closs = running.check_metric(mlu_path, tp_path, closs_path)
        print(mlu,tp,closs)
    if topo == 'iris':
        net = nx.read_graphml('dataset/topologies/iris.graphml')
        path_path = 'dataset/paths/iris.npy'
        traffic_path = 'dataset/traffic/iris-07.npy'
        ca_path = 'dataset/capacities/iris.npy'
        save_dir = 'results/ECMP/Iris/'
        train_size = 4500
        scale = 185.09949097639984
        k = 4
        # run model
        running = RUN_ECMP(net=net, path_path=path_path, save_dir=save_dir, traffic_path=traffic_path, scale=scale, ca_path=ca_path, k=k)
        if run_flag == True:
            running.run()
        # if test_flag == True:
        #     # test run time
        #     model_path = 'results/Abilene/0_model.pt'
        #     running_time, times = running.check_decision_time(model_path)
        #     print(running_time/times*1000)

        # test metrics
        mlu_path = 'results/ECMP/Iris/MLU.npy'
        tp_path = 'results/ECMP/Iris/TP.npy'
        closs_path = 'results/ECMP/Iris/CLoss.npy'
        mlu, tp, closs = running.check_metric(mlu_path, tp_path, closs_path)
        print(mlu,tp,closs)
    if topo == 'intellifiber':
        net = nx.read_graphml('dataset/topologies/intellifiber.graphml')
        path_path = 'dataset/paths/intellifiber.npy'
        traffic_path = 'dataset/traffic/intellifiber-1.npy'
        ca_path = 'dataset/capacities/intellifiber.npy'
        save_dir = 'results/ECMP/Intellifiber/'
        train_size = 4500
        scale = 310.80031
        k = 4
        # run model
        running = RUN_ECMP(net=net, path_path=path_path, save_dir=save_dir, traffic_path=traffic_path, scale=scale, ca_path=ca_path, k=k)
        if run_flag == True:
            running.run()
        # if test_flag == True:
        #     # test run time
        #     model_path = 'results/Abilene/0_model.pt'
        #     running_time, times = running.check_decision_time(model_path)
        #     print(running_time/times*1000)

        # test metrics
        mlu_path = 'results/ECMP/Intellifiber/MLU.npy'
        tp_path = 'results/ECMP/Intellifiber/TP.npy'
        closs_path = 'results/ECMP/Intellifiber/CLoss.npy'
        mlu, tp, closs = running.check_metric(mlu_path, tp_path, closs_path)
        print(mlu,tp,closs)

def LP(topo, run_flag=True, test_flag=False, alpha=0):
    if topo == 'abilene':
        net = nx.read_graphml('dataset/topologies/abilene.graphml')
        path_path = 'dataset/paths/abilene.npy'
        traffic_path = 'dataset/traffic/abilene.npy'
        ca_path = 'dataset/capacities/abilene.npy'
        train_size = 4500
        scale = 1150383
        k = 4
        running = RUN_LP(net=net, path_path=path_path, traffic_path=traffic_path, scale=scale, ca_path=ca_path, k=k)
        res = running.run()
        print(res.fun)

    if topo == 'geant':
        net = nx.read_graphml('dataset/topologies/geant.graphml')
        path_path = 'dataset/paths/geant.npy'
        traffic_path = 'dataset/traffic/geant.npy'
        ca_path = 'dataset/capacities/geant.npy'
        train_size = 4500
        scale = 189955
        k = 4
        running = RUN_LP(net=net, path_path=path_path, traffic_path=traffic_path, scale=scale, ca_path=ca_path, k=k)
        res = running.run()
        print(res.fun)

    if topo == 'iris':
        net = nx.read_graphml('dataset/topologies/iris.graphml')
        path_path = 'dataset/paths/iris.npy'
        traffic_path = 'dataset/traffic/iris-07.npy'
        ca_path = 'dataset/capacities/iris.npy'
        train_size = 4500
        scale = 185.09949097639984
        k = 4
        running = RUN_LP(net=net, path_path=path_path, traffic_path=traffic_path, scale=scale, ca_path=ca_path, k=k)
        res = running.run()
        print(res.fun)
    
    if topo == 'intellifiber':
        net = nx.read_graphml('dataset/topologies/intellifiber.graphml')
        path_path = 'dataset/paths/intellifiber.npy'
        traffic_path = 'dataset/traffic/intellifiber-1.npy'
        ca_path = 'dataset/capacities/intellifiber.npy'
        train_size = 4500
        scale = 310.80031
        k = 4
        running = RUN_LP(net=net, path_path=path_path, traffic_path=traffic_path, scale=scale, ca_path=ca_path, k=k)
        res = running.run()
        print(res.fun)

def DNN(topo, run_flag, test_flag, alpha, learning_rate):
    if topo == 'abilene':
        net = nx.read_graphml('dataset/topologies/abilene.graphml')
        path_path = 'dataset/paths/abilene.npy'
        traffic_path = 'dataset/traffic/abilene.npy'
        ca_path = 'dataset/capacities/abilene.npy'
        save_dir = 'results/DNN/Abilene/'
        train_size = 4500
        scale = 1150383
        k = 4
        batch_size = 64
        # run model
        running = RUN_DNN(net=net, traffic_path=traffic_path, ca_path=ca_path, path_path=path_path, train_size=train_size, save_dir=save_dir, scale=scale, k=k, batch_size=batch_size)
        if run_flag == True:
            running.run(alpha=alpha, learning_rate=learning_rate)
        if test_flag == True:
            # test run time
            model_path = '../results/DNN/Abilene/0_model.pt'
            running_time, times = running.check_decision_time(model_path)
            print(running_time/times*1000)

        alpha = alpha
        # test metrics
        mlu_path = '../results/DNN/Abilene/'+str(alpha)+'_MLU.npy'
        tp_path = '../results/DNN/Abilene/'+str(alpha)+'_TP.npy'
        closs_path = '../results/DNN/Abilene/'+str(alpha)+'_CLoss.npy'
        mlu, tp, closs = running.check_metric(mlu_path, tp_path, closs_path)
        print(mlu,tp,closs)
    
    if topo == 'geant':
        net = nx.read_graphml('dataset/topologies/geant.graphml')
        path_path = 'dataset/paths/geant.npy'
        traffic_path = 'dataset/traffic/geant.npy'
        ca_path = 'dataset/capacities/geant.npy'
        save_dir = 'results/DNN/Geant/'
        train_size = 4500
        scale = 189955
        k = 4
        batch_size = 64
        # run model
        running = RUN_DNN(net=net, traffic_path=traffic_path, ca_path=ca_path, path_path=path_path, train_size=train_size, save_dir=save_dir, scale=scale, k=k, batch_size=batch_size)
        if run_flag == True:
            running.run(alpha=alpha, learning_rate=learning_rate)
        if test_flag == True:
            # test run time
            model_path = '../results/DNN/Geant/0_model.pt'
            running_time, times = running.check_decision_time(model_path)
            print(running_time/times*1000)

        alpha = alpha
        # test metrics
        mlu_path = '../results/DNN/Geant/'+str(alpha)+'_MLU.npy'
        tp_path = '../results/DNN/Geant/'+str(alpha)+'_TP.npy'
        closs_path = '../results/DNN/Geant/'+str(alpha)+'_CLoss.npy'
        mlu, tp, closs = running.check_metric(mlu_path, tp_path, closs_path)
        print(mlu,tp,closs)

    if topo == 'iris':
        net = nx.read_graphml('dataset/topologies/iris.graphml')
        path_path = 'dataset/paths/iris.npy'
        traffic_path = 'dataset/traffic/iris-07.npy'
        ca_path = 'dataset/capacities/iris.npy'
        save_dir = 'results/DNN/Iris/'
        train_size = 4500
        scale = 185.09949097639984
        k = 4
        batch_size = 16
        # run model
        running = RUN_DNN(net=net, traffic_path=traffic_path, ca_path=ca_path, path_path=path_path, train_size=train_size, save_dir=save_dir, scale=scale, k=k, batch_size=batch_size)
        if run_flag == True:
            running.run(alpha=alpha, learning_rate=learning_rate)
        if test_flag == True:
            # test run time
            model_path = '../results/DNN/Iris/0_model.pt'
            running_time, times = running.check_decision_time(model_path)
            print(running_time/times*1000)

        alpha = alpha
        # test metrics
        mlu_path = '../results/DNN/Iris/'+str(alpha)+'_MLU.npy'
        tp_path = '../results/DNN/Iris/'+str(alpha)+'_TP.npy'
        closs_path = '../results/DNN/Iris/'+str(alpha)+'_CLoss.npy'
        mlu, tp, closs = running.check_metric(mlu_path, tp_path, closs_path)
        print(mlu,tp,closs)
    
    if topo == 'intellifiber':
        net = nx.read_graphml('dataset/topologies/intellifiber.graphml')
        path_path = 'dataset/paths/intellifiber.npy'
        traffic_path = 'dataset/traffic/intellifiber-1.npy'
        ca_path = 'dataset/capacities/intellifiber.npy'
        save_dir = 'results/DNN/Intellifiber/'
        train_size = 4500
        scale = 310.80031
        k = 4
        batch_size = 16
        # run model
        running = RUN_DNN(net=net, traffic_path=traffic_path, ca_path=ca_path, path_path=path_path, train_size=train_size, save_dir=save_dir, scale=scale, k=k, batch_size=batch_size)
        if run_flag == True:
            running.run(alpha=alpha, learning_rate=learning_rate)
        if test_flag == True:
            # test run time
            model_path = '../results/DNN/Intellifiber/0_model.pt'
            running_time, times = running.check_decision_time(model_path)
            print(running_time/times*1000)

        alpha = alpha
        # test metrics
        mlu_path = '../results/DNN/Intellifiber/'+str(alpha)+'_MLU.npy'
        tp_path = '../results/DNN/Intellifiber/'+str(alpha)+'_TP.npy'
        closs_path = '../results/DNN/Intellifiber/'+str(alpha)+'_CLoss.npy'
        mlu, tp, closs = running.check_metric(mlu_path, tp_path, closs_path)
        print(mlu,tp,closs)

def LSTM(topo, run_flag, test_flag, alpha, learning_rate):
    if topo == 'abilene':
        net = nx.read_graphml('dataset/topologies/abilene.graphml')
        path_path = 'dataset/paths/abilene.npy'
        traffic_path = 'dataset/traffic/abilene.npy'
        ca_path = 'dataset/capacities/abilene.npy'
        save_dir = 'results/LSTM/Abilene/'
        train_size = 4500
        scale = 1150383
        k = 4
        batch_size = 64
        # run model
        running = RUN_LSTM(net=net, traffic_path=traffic_path, ca_path=ca_path, path_path=path_path, train_size=train_size, save_dir=save_dir, scale=scale, k=k, batch_size=batch_size)
        if run_flag == True:
            running.run(alpha=alpha, learning_rate=learning_rate)
        if test_flag == True:
            # test run time
            model_path = '../results/LSTM/Abilene/0_model.pt'
            running_time, times = running.check_decision_time(model_path)
            print(running_time/times*1000)

        alpha = alpha
        # test metrics
        mlu_path = '../results/LSTM/Abilene/'+str(alpha)+'_MLU.npy'
        tp_path = '../results/LSTM/Abilene/'+str(alpha)+'_TP.npy'
        closs_path = '../results/LSTM/Abilene/'+str(alpha)+'_CLoss.npy'
        mlu, tp, closs = running.check_metric(mlu_path, tp_path, closs_path)
        print(mlu,tp,closs)
    
    if topo == 'geant':
        net = nx.read_graphml('dataset/topologies/geant.graphml')
        path_path = 'dataset/paths/geant.npy'
        traffic_path = 'dataset/traffic/geant.npy'
        ca_path = 'dataset/capacities/geant.npy'
        save_dir = 'results/LSTM/Geant/'
        train_size = 4500
        scale = 189955
        k = 4
        batch_size = 64
        # run model
        running = RUN_LSTM(net=net, traffic_path=traffic_path, ca_path=ca_path, path_path=path_path, train_size=train_size, save_dir=save_dir, scale=scale, k=k, batch_size=batch_size)
        if run_flag == True:
            running.run(alpha=alpha, learning_rate=learning_rate)
        if test_flag == True:
            # test run time
            model_path = '../results/LSTM/Geant/0_model.pt'
            running_time, times = running.check_decision_time(model_path)
            print("time: ",running_time/times*1000)

        alpha = alpha
        # test metrics
        mlu_path = '../results/LSTM/Geant/'+str(alpha)+'_MLU.npy'
        tp_path = '../results/LSTM/Geant/'+str(alpha)+'_TP.npy'
        closs_path = '../results/LSTM/Geant/'+str(alpha)+'_CLoss.npy'
        mlu, tp, closs = running.check_metric(mlu_path, tp_path, closs_path)
        print(alpha," :",mlu,tp,closs)
    
    if topo == 'iris':
        net = nx.read_graphml('dataset/topologies/iris.graphml')
        path_path = 'dataset/paths/iris.npy'
        traffic_path = 'dataset/traffic/iris-07.npy'
        ca_path = 'dataset/capacities/iris.npy'
        save_dir = 'results/LSTM/Iris/'
        train_size = 4500
        scale = 185.09949097639984
        k = 4
        batch_size = 16
        # run model
        running = RUN_LSTM(net=net, traffic_path=traffic_path, ca_path=ca_path, path_path=path_path, train_size=train_size, save_dir=save_dir, scale=scale, k=k, batch_size=batch_size)
        if run_flag == True:
            running.run(alpha=alpha, learning_rate=learning_rate)
        if test_flag == True:
            # test run time
            model_path = '../results/LSTM/Iris/0_model.pt'
            running_time, times = running.check_decision_time(model_path)
            print("time: ",running_time/times*1000)

        alpha = alpha
        # test metrics
        mlu_path = '../results/LSTM/Iris/'+str(alpha)+'_MLU.npy'
        tp_path = '../results/LSTM/Iris/'+str(alpha)+'_TP.npy'
        closs_path = '../results/LSTM/Iris/'+str(alpha)+'_CLoss.npy'
        mlu, tp, closs = running.check_metric(mlu_path, tp_path, closs_path)
        print(mlu,tp,closs)
    
    if topo == 'intellifiber':
        net = nx.read_graphml('dataset/topologies/intellifiber.graphml')
        path_path = 'dataset/paths/intellifiber.npy'
        traffic_path = 'dataset/traffic/intellifiber-1.npy'
        ca_path = 'dataset/capacities/intellifiber.npy'
        save_dir = 'results/LSTM/Intellifiber/'
        train_size = 4500
        scale = 310.80031
        k = 4
        batch_size = 16
         # run model
        running = RUN_LSTM(net=net, traffic_path=traffic_path, ca_path=ca_path, path_path=path_path, train_size=train_size, save_dir=save_dir, scale=scale, k=k, batch_size=batch_size)
        if run_flag == True:
            running.run(alpha=alpha, learning_rate=learning_rate)
        if test_flag == True:
            # test run time
            model_path = '../results/LSTM/Intellifiber/0_model.pt'
            running_time, times = running.check_decision_time(model_path)
            print("time: ",running_time/times*1000)

        alpha = alpha
        # test metrics
        mlu_path = '../results/LSTM/Intellifiber/'+str(alpha)+'_MLU.npy'
        tp_path = '../results/LSTM/Intellifiber/'+str(alpha)+'_TP.npy'
        closs_path = '../results/LSTM/Intellifiber/'+str(alpha)+'_CLoss.npy'
        mlu, tp, closs = running.check_metric(mlu_path, tp_path, closs_path)
        print(mlu,tp,closs)

def CNN(topo, run_flag, test_flag, alpha):
    if topo == 'abilene':
        net = nx.read_graphml('dataset/topologies/abilene.graphml')
        path_path = 'dataset/paths/abilene.npy'
        traffic_path = 'dataset/traffic/abilene.npy'
        ca_path = 'dataset/capacities/abilene.npy'
        save_dir = 'results/CNN/Abilene/'
        train_size = 4500
        scale = 1150383
        k = 4
        batch_size = 64
        # run model
        running = RUN_CNN(net=net, traffic_path=traffic_path, ca_path=ca_path, path_path=path_path, train_size=train_size, save_dir=save_dir, scale=scale, k=k, batch_size=batch_size)
        if run_flag == True:
            running.run(alpha=alpha)
        if test_flag == True:
            # test run time
            model_path = '../dNE-results/CNN/Abilene/0_model.pt'
            running_time, times = running.check_decision_time(model_path)
            print(running_time/times*1000)

        # alpha = alpha
        # # test metrics
        # mlu_path = '../results/CNN/Abilene/'+str(alpha)+'_MLU.npy'
        # tp_path = '../results/CNN/Abilene/'+str(alpha)+'_TP.npy'
        # closs_path = '../results/CNN/Abilene/'+str(alpha)+'_CLoss.npy'
        # mlu, tp, closs = running.check_metric(mlu_path, tp_path, closs_path)
        # print(mlu,tp,closs)
    
    if topo == 'geant':
        net = nx.read_graphml('dataset/topologies/geant.graphml')
        path_path = 'dataset/paths/geant.npy'
        traffic_path = 'dataset/traffic/geant.npy'
        ca_path = 'dataset/capacities/geant.npy'
        save_dir = 'results/CNN/Geant/'
        train_size = 4500
        scale = 189955
        k = 4
        batch_size = 64
        # run model
        running = RUN_CNN(net=net, traffic_path=traffic_path, ca_path=ca_path, path_path=path_path, train_size=train_size, save_dir=save_dir, scale=scale, k=k, batch_size=batch_size)
        if run_flag == True:
            running.run(alpha=alpha)
        if test_flag == True:
            # test run time
            model_path = '../dNE-results/CNN/Geant/0_model.pt'
            running_time, times = running.check_decision_time(model_path)
            print("time: ",running_time/times*1000)

        # alpha = alpha
        # # test metrics
        # mlu_path = '../results/CNN/Geant/'+str(alpha)+'_MLU.npy'
        # tp_path = '../results/CNN/Geant/'+str(alpha)+'_TP.npy'
        # closs_path = '../results/CNN/Geant/'+str(alpha)+'_CLoss.npy'
        # mlu, tp, closs = running.check_metric(mlu_path, tp_path, closs_path)
        # print(alpha," :",mlu,tp,closs)

    if topo == 'iris':
        net = nx.read_graphml('dataset/topologies/iris.graphml')
        path_path = 'dataset/paths/iris.npy'
        traffic_path = 'dataset/traffic/iris-07.npy'
        ca_path = 'dataset/capacities/iris.npy'
        save_dir = 'results/CNN/Iris/'
        train_size = 4500
        scale = 185.09949097639984
        k = 4
        batch_size = 16
        # run model
        running = RUN_CNN(net=net, traffic_path=traffic_path, ca_path=ca_path, path_path=path_path, train_size=train_size, save_dir=save_dir, scale=scale, k=k, batch_size=batch_size)
        if run_flag == True:
            running.run(alpha=alpha)
        if test_flag == True:
            # test run time
            model_path = '../dNE-results/CNN/Iris/0_model.pt'
            running_time, times = running.check_decision_time(model_path)
            print("time: ",running_time/times*1000)

        # alpha = alpha
        # # test metrics
        # mlu_path = '../results/CNN/Iris/'+str(alpha)+'_MLU.npy'
        # tp_path = '../results/CNN/Iris/'+str(alpha)+'_TP.npy'
        # closs_path = '../results/CNN/Iris/'+str(alpha)+'_CLoss.npy'
        # mlu, tp, closs = running.check_metric(mlu_path, tp_path, closs_path)
        # print(mlu,tp,closs)
    
    if topo == 'intellifiber':
        net = nx.read_graphml('dataset/topologies/intellifiber.graphml')
        path_path = 'dataset/paths/intellifiber.npy'
        traffic_path = 'dataset/traffic/intellifiber-1.npy'
        ca_path = 'dataset/capacities/intellifiber.npy'
        save_dir = 'results/CNN/Intellifiber/'
        train_size = 4500
        scale = 310.80031
        k = 4
        batch_size = 16
         # run model
        running = RUN_CNN(net=net, traffic_path=traffic_path, ca_path=ca_path, path_path=path_path, train_size=train_size, save_dir=save_dir, scale=scale, k=k, batch_size=batch_size)
        if run_flag == True:
            running.run(alpha=alpha)
        if test_flag == True:
            # test run time
            model_path = '../dNE-results/CNN/Intellifiber/0_model.pt'
            running_time, times = running.check_decision_time(model_path)
            print("time: ",running_time/times*1000)

        # alpha = alpha
        # # test metrics
        # mlu_path = '../results/CNN/Intellifiber/'+str(alpha)+'_MLU.npy'
        # tp_path = '../results/CNN/Intellifiber/'+str(alpha)+'_TP.npy'
        # closs_path = '../results/CNN/Intellifiber/'+str(alpha)+'_CLoss.npy'
        # mlu, tp, closs = running.check_metric(mlu_path, tp_path, closs_path)
        # print(mlu,tp,closs)

def DRL(topo, run_flag, test_flag, alpha, actor_lr, critic_lr, ttt, name):
    time = 0
    if topo == 'abilene':
        net = nx.read_graphml('dataset/topologies/abilene.graphml')
        path_path = 'dataset/paths/abilene.npy'
        traffic_path = 'dataset/traffic/abilene.npy'
        ca_path = 'dataset/capacities/abilene.npy'
        save_dir = 'results/DRL/Abilene/'
        train_size = 4500
        scale = 1150383
        k = 4
        batch_size = 64
        # run model
        running = RUN_DRL(net=net, traffic_path=traffic_path, ca_path=ca_path, path_path=path_path, train_size=train_size, save_dir=save_dir, scale=scale, k=k, batch_size=batch_size, name=name)
        if run_flag == True:
            time = running.run(alpha=alpha, learning_rate_actor=actor_lr, learning_rate_critic=critic_lr, gamma=0.99, tau=0.01, sigma=0.6)
        if test_flag == True:
            # test run time
            model_path = '../results/DRL/Abilene/0_model_'
            running_time, times = running.check_decision_time(model_path, learning_rate_actor=actor_lr, learning_rate_critic=critic_lr, gamma=0.99, tau=0.01, sigma=0.6)
            print("time: ",running_time/times*1000)
        # # test metrics
        # mlu_path = '../results/DRL/Abilene/'+str(alpha)+'_MLU.npy'
        # tp_path = '../results/DRL/Abilene/'+str(alpha)+'_TP.npy'
        # closs_path = '../results/DRL/Abilene/'+str(alpha)+'_CLoss.npy'
        # mlu, tp, closs = running.check_metric(mlu_path, tp_path, closs_path)
        # print(mlu,tp,closs)
    
    if topo == 'geant':
        net = nx.read_graphml('dataset/topologies/geant.graphml')
        path_path = 'dataset/paths/geant.npy'
        traffic_path = 'dataset/traffic/geant.npy'
        ca_path = 'dataset/capacities/geant.npy'
        save_dir = 'results/DRL/Geant/'
        train_size = 4500
        scale = 189955
        k = 4
        batch_size = 64
        # run model
        running = RUN_DRL(net=net, traffic_path=traffic_path, ca_path=ca_path, path_path=path_path, train_size=train_size, save_dir=save_dir, scale=scale, k=k, batch_size=batch_size, name=name)
        if run_flag == True:
            time = running.run(alpha=alpha, learning_rate_actor=actor_lr, learning_rate_critic=critic_lr, gamma=0.99, tau=0.01, sigma=0.6)
        if test_flag == True:
            # test run time
            model_path = '../results/DRL/Geant/0_model_'
            running_time, times = running.check_decision_time(model_path, learning_rate_actor=actor_lr, learning_rate_critic=critic_lr, gamma=0.99, tau=0.01, sigma=0.6)
            print("time: ",running_time/times*1000)
        # # test metrics
        # mlu_path = '../results/DRL/Geant/'+str(alpha)+'_MLU.npy'
        # tp_path = '../results/DRL/Geant/'+str(alpha)+'_TP.npy'
        # closs_path = '../results/DRL/Geant/'+str(alpha)+'_CLoss.npy'
        # mlu, tp, closs = running.check_metric(mlu_path, tp_path, closs_path)
        # print(alpha," :",mlu,tp,closs)
    
    if topo == 'iris':
        net = nx.read_graphml('dataset/topologies/iris.graphml')
        path_path = 'dataset/paths/iris.npy'
        traffic_path = 'dataset/traffic/iris-07.npy'
        ca_path = 'dataset/capacities/iris.npy'
        save_dir = 'results/DRL/Iris/'
        train_size = 4500
        scale = 185.09949097639984
        k = 4
        batch_size = 16
       # run model
        running = RUN_DRL(net=net, traffic_path=traffic_path, ca_path=ca_path, path_path=path_path, train_size=train_size, save_dir=save_dir, scale=scale, k=k, batch_size=batch_size, name=name)
        if run_flag == True:
            time = running.run(alpha=alpha, learning_rate_actor=actor_lr, learning_rate_critic=critic_lr, gamma=0.99, tau=0.01, sigma=0.6)
        if test_flag == True:
            # test run time
            model_path = '../results/DRL/Iris/0_model_'
            running_time, times = running.check_decision_time(model_path, learning_rate_actor=actor_lr, learning_rate_critic=critic_lr, gamma=0.99, tau=0.01, sigma=0.6)
            print("time: ",running_time/times*1000)
        # # test metrics
        # mlu_path = '../results/DRL/Iris/'+str(alpha)+'_MLU.npy'
        # tp_path = '../results/DRL/Iris/'+str(alpha)+'_TP.npy'
        # closs_path = '../results/DRL/Iris/'+str(alpha)+'_CLoss.npy'
        # mlu, tp, closs = running.check_metric(mlu_path, tp_path, closs_path)
        # print(mlu,tp,closs)
    
    if topo == 'intellifiber':
        net = nx.read_graphml('dataset/topologies/intellifiber.graphml')
        path_path = 'dataset/paths/intellifiber.npy'
        traffic_path = 'dataset/traffic/intellifiber-1.npy'
        ca_path = 'dataset/capacities/intellifiber.npy'
        save_dir = 'results/DRL/Intellifiber/'
        train_size = 4500
        scale = 310.80031
        k = 4
        batch_size = 16
        # run model
        running = RUN_DRL(net=net, traffic_path=traffic_path, ca_path=ca_path, path_path=path_path, train_size=train_size, save_dir=save_dir, scale=scale, k=k, batch_size=batch_size,name=name)
        if run_flag == True:
            time = running.run(alpha=alpha, learning_rate_actor=actor_lr, learning_rate_critic=critic_lr, gamma=0.99, tau=0.01, sigma=0.6)
        if test_flag == True:
            # test run time
            model_path = '../results/DRL/Intellifiber/0_model_'
            running_time, times = running.check_decision_time(model_path, learning_rate_actor=actor_lr, learning_rate_critic=critic_lr, gamma=0.99, tau=0.01, sigma=0.6)
            print("time: ",running_time/times*1000)
        # # test metrics
        # mlu_path = '../results/DRL/Intellifiber/'+str(alpha)+'_MLU.npy'
        # tp_path = '../results/DRL/Intellifiber/'+str(alpha)+'_TP.npy'
        # closs_path = '../results/DRL/Intellifiber/'+str(alpha)+'_CLoss.npy'
        # mlu, tp, closs = running.check_metric(mlu_path, tp_path, closs_path)
        # print(mlu,tp,closs)
    data[topo] = time
    np.save('runtime_empty.npy' + str(ttt),data,allow_pickle=True)

def main(method, topo, run_flag, test_flag, alpha, learning_rate, learning_rate_2=0.001):
    if method == 'ECMP':
        ECMP(topo=topo, run_flag=run_flag, test_flag=test_flag)
    elif method == 'LP':
        LP(topo=topo, run_flag=run_flag, test_flag=test_flag, alpha=alpha)
    elif method == 'DNN':
        DNN(topo=topo, run_flag=run_flag, test_flag=test_flag, alpha=alpha, learning_rate=learning_rate)
    elif method == 'LSTM':
        LSTM(topo=topo, run_flag=run_flag, test_flag=test_flag, alpha=alpha, learning_rate=learning_rate)
    elif method == 'CNN':
        CNN(topo=topo, run_flag=run_flag, test_flag=test_flag, alpha=alpha)
    elif method == 'DRL':
        DRL(topo=topo, run_flag=run_flag, test_flag=test_flag, alpha=alpha, actor_lr=learning_rate, critic_lr=learning_rate_2, ttt=2, name=topo)
    elif method in 'LARGE':
        LARGE(topo=topo, run_flag=run_flag, test_flag=test_flag, alpha=alpha)

import argparse

def main(method, topo, run_flag, test_flag, alpha, learning_rate, learning_rate_2=None):
    """
    Run the specified algorithm method and print parameter information.

    :param method: Algorithm method to use (ECMP, LP, DNN, LSTM, CNN, DRL)
    :param topo: Network topology name (e.g., abilene)
    :param run_flag: Running flag (both training and testing) (e.g., 0 or 1)
    :param test_flag: Testing only flag (e.g., 0 or 1)
    :param alpha: Hyperparameter alpha (MLU + alpha * CLoss, default: 0)
    :param learning_rate: Initial learning rate for optimization
    :param learning_rate_2: Second learning rate (only required for DRL)
    """
    print(f"Running method: {method}")
    print(f"Topology: {topo}")
    print(f"Run Flag: {run_flag}")
    print(f"Test Flag: {test_flag}")
    print(f"Alpha: {alpha}")
    print(f"Learning Rate: {learning_rate}")
    if learning_rate_2 is not None:
        print(f"Learning Rate 2: {learning_rate_2}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run network optimization algorithms with specified parameters.")

    # Define command-line arguments
    parser.add_argument("--method", type=str, required=True, choices=['ECMP', 'LP', 'DNN', 'LSTM', 'CNN', 'DRL'],
                        help="Algorithm method to use (ECMP, LP, DNN, LSTM, CNN, DRL)")
    parser.add_argument("--topo", type=str, default="abilene", help="Network topology (default: abilene)")
    parser.add_argument("--run_flag", type=int, required=True, help="Execution flag (e.g., 0 or 1)")
    parser.add_argument("--test_flag", type=int, required=True, help="Testing flag (e.g., 0 or 1)")
    parser.add_argument("--alpha", type=float, default=0, help="Alpha parameter (default: 0)")
    parser.add_argument("--learning_rate", type=float, required=True, help="Learning rate for the method")
    parser.add_argument("--learning_rate_2", type=float, default=None, help="Second learning rate (only needed for DRL)")

    # Parse command-line arguments
    args = parser.parse_args()

    # Call the main function with parsed arguments
    main(
        method=args.method,
        topo=args.topo,
        run_flag=args.run_flag,
        test_flag=args.test_flag,
        alpha=args.alpha,
        learning_rate=args.learning_rate,
        learning_rate_2=args.learning_rate_2
    )


# main(method = 'ECMP' ,topo = 'abilene' ,run_flag = run_flag ,test_flag = test_flag ,alpha = 0, learning_rate = 0.01)
# main(method = 'LP' ,topo = 'abilene' ,run_flag = run_flag ,test_flag = test_flag ,alpha = 0, learning_rate = 0.01)
# main(method = 'DNN' ,topo = 'abilene' ,run_flag = run_flag ,test_flag = test_flag ,alpha = 0, learning_rate = 0.01)
# main(method = 'LSTM' ,topo = 'abilene' ,run_flag = run_flag ,test_flag = test_flag ,alpha = 0, learning_rate = 0.01)
# main(method = 'CNN' ,topo = 'abilene' ,run_flag = run_flag ,test_flag = test_flag ,alpha = 0, learning_rate = 0.001)
# main(method = 'DRL' ,topo = 'abilene' ,run_flag = run_flag ,test_flag = test_flag ,alpha = 0, learning_rate = 0.01, learning_rate_2 = 0.01)
