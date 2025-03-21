import torch
import numpy as np
from framework import evaluation, summarization

torch.set_default_dtype(torch.float64)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("\ndevice: ",device, "\n")

class RUN_ECMP(object):
    def __init__(self, net, path_path, traffic_path, save_dir, scale, ca_path, k):
        self.count = 0
        self.max_util = 0
        self.net = net
        self.path_path = path_path
        self.save_dir = save_dir
        self.traffic_path = traffic_path
        self.scale = scale
        self.ca_path = ca_path
        self.N = len(self.net.nodes())
        self.flow_no = self.N * self.N
        self.k = k
        print("Ecmp is selected")

    def run(self):
        P = np.load(self.path_path,allow_pickle=True) # path
        P = np.array(P.tolist(), dtype=np.float64)
        D = np.load(self.traffic_path,allow_pickle=True) * self.scale # traffic
        C = np.load(self.ca_path,allow_pickle=True) # capacity
        C = np.array(C.tolist(), dtype=np.float64)
        MA = np.ones(shape=(self.flow_no,self.k)) * 1/self.k # splitting ratio

        B = np.zeros(shape=(self.flow_no, self.k * self.flow_no))
        for i in range(self.flow_no):
            for j in range(i * self.k, (i+1) * self.k):
                B[i][j] = 1.0

        P_tensor = torch.from_numpy(P).to(device)
        C_tensor = torch.from_numpy(C).to(device)
        D_tensor = torch.from_numpy(D).to(device)
        B_tensor = torch.from_numpy(B).to(device)
        MA_tensor = torch.from_numpy(MA).to(device)

        MLU = []
        TP = []
        CLoss = []
        for i in range(4500, 5000):
            L_tensor = evaluation(MA_tensor, P_tensor, B_tensor, D_tensor[i])
            TE_MLU = summarization('MLU', L_tensor, C_tensor, D_tensor[i])
            TE_Closs = summarization('CLoss', L_tensor, C_tensor, D_tensor[i])
            TE_TP = summarization('TP', L_tensor, C_tensor, D_tensor[i])
            
            MLU.append(TE_MLU.item())
            TP.append(TE_TP.item())
            CLoss.append(TE_Closs.item())

        np.save(self.save_dir + 'MLU.npy', np.array(MLU), allow_pickle=True)
        np.save(self.save_dir + 'TP.npy', np.array(TP), allow_pickle=True)
        np.save(self.save_dir + 'CLoss.npy', np.array(CLoss), allow_pickle=True)
        print("complete!")
    
    def check_decision_time(self, model_path):
        return None
    
    def check_metric(self, mlu_path, tp_path, closs_path):
        mlu = np.load(mlu_path,allow_pickle=True)
        tp = np.load(tp_path,allow_pickle=True)
        closs = np.load(closs_path,allow_pickle=True)

        new_mlu = []
        for i in range(len(mlu)):
            if mlu[i] > 1:
                cur = 1.0
            else:
                cur = mlu[i]
            new_mlu.append(cur)

        return np.mean(np.array(new_mlu)), np.mean(tp), np.mean(closs)
            

            
