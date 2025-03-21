import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch
import datetime
import random
import numpy as np
import os
import math
from framework import evaluation, summarization
from AssistFunc import adjust_learning_rate

torch.set_default_dtype(torch.float32)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("\ndevice: ",device, "\n")

def seed_everything(seed_val):
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    os.environ['PYTHONHASHSEED'] = str(seed_val)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_val)
        torch.cuda.manual_seed_all(seed_val)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.bechmark = True

# definition of the model you use
class MODLE_NAME(nn.Module):
    def __init__(self, N, output_size, kernel_size):
        super().__init__()
        # model structure
        pass

    def forward(self, x):
        # forward process
        pass

# running functions of the model
class MODLE_NAME_RUN(object):
    def __init__(self, net, path_path, traffic_path, ca_path, train_size, save_dir, scale, k, batch_size):
        self.count = 0 
        self.max_util = 0  
        self.net = net
        self.path_path = path_path
        self.traffic_path = traffic_path
        self.ca_path = ca_path
        self.train_size = train_size
        self.save_dir = save_dir
        self.scale = scale
        self.k = k
        self.batch_size = batch_size
        self.N = len(self.net.nodes())
        self.flow_no = self.N * self.N
        print("MODLE_NAME is selected.")

    # normalization of input data
    def normalization(self, data):
        _range = np.max(data) - np.min(data)
        return (data - np.min(data)) / _range
    
    # data loader from training
    def my_dataloader(self):
        # prepare data
        traffic_data = np.load(self.traffic_path, allow_pickle=True) * self.scale # load data
        traffic_data_normalized = self.normalization(traffic_data) # data normalized

        # input shape: (batch_size X seq_len X input_size)
        # input_size = 1
        # seq_len = lengths of flows
        traffic_load = []
        seq_len = len(traffic_data[0])
        for i in range(len(traffic_data)):
            res_normalized = traffic_data_normalized[i].reshape(1, self.N, self.N)
            # res = traffic_data[i].reshape(1, self.N, self.N)
            traffic_load.append((np.array(res_normalized), np.array(traffic_data[i])))
        
        train_loader = torch.utils.data.DataLoader(dataset=traffic_load[0:self.train_size], batch_size=self.batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(dataset=traffic_load[self.train_size:], batch_size=1, shuffle=False)
        return train_loader, test_loader

    # running process for both training and testing
    def run(self, alpha=0, learning_rate=0.001):
        seed = 76 # assign a seed
        seed_everything(seed)
        epoch = 100
        print('----------------------- Information -----------------------')
        print('MODLE NAME\n', self.save_dir.split('/')[-2], '\n', 'epoch: ',epoch,'\tN: ',self.N,'\tDataset size: ',self.train_size,'\tk: ',self.k,'\talpha: ',alpha,'\tseed: ',seed,'\n')
        print('----------------------- Information -----------------------')

        # Prepare paths information
        path = np.load(self.path_path,allow_pickle=True)
        P = np.array(path.tolist(), dtype=np.float64)

        # Prepare capacity information
        C = np.load(self.ca_path, allow_pickle=True)
        C = np.array(C.tolist(), dtype=np.float64)

        # transfer P and C to tensor type
        P_tensor = torch.from_numpy(P).to(device)
        C_tensor = torch.from_numpy(C).to(device)

        train_loader, test_loader = self.my_dataloader()
        # the model
        model = MODLE_NAME(N=self.N, output_size=self.k * self.flow_no).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate, weight_decay=0.001)

        train_loss_collect = []
        test_loss_collect = []

        B = np.zeros(shape=(self.flow_no, self.k * self.flow_no))
        for i in range(self.flow_no):
            for j in range(i * self.k, (i+1) * self.k):
                B[i][j] = 1.0
        B_tensor = torch.from_numpy(B).to(device)


        # Epochs
        for i in range(epoch):
            adjust_learning_rate(optimizer, i, learning_rate, 20)
            print("\nepoch: ", i)
            
            # Training
            model.train()
            train_loss = 0.0
            for batch_idx, (data1, data2) in enumerate(train_loader):
                data1, data2 = data1.to(device), data2.to(device)
                optimizer.zero_grad()
                output = model(data1)

                loss = torch.zeros(len(data1)).to(device)
                for j in range(len(data1)):
                    cur_MA_0 = output[j].view(self.flow_no, self.k)
                    cur_MA = F.softmax(cur_MA_0, dim=1)
                    L_tensor = evaluation(cur_MA, P_tensor, B_tensor, data2[j]) # evaluation stage: load on each edge
                    TE_MLU = summarization('MLU', L_tensor, C_tensor, data2[j]) # summarization stage: calculate MLU
                    TE_CLoss = summarization('CLoss', L_tensor, C_tensor, data2[j]) # summerization stage: calculate throughput
                    loss[j] = TE_MLU + alpha * TE_CLoss

                loss = torch.mean(loss)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

                if batch_idx % 10 == 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.10f}'.format(
                        i, batch_idx * len(data1), len(train_loader.dataset),
                        100. * batch_idx / len(train_loader), loss.item()
                    ))

            # Testing
            MLU = []
            TP = []
            CLoss = []
            Time = []
            model.eval()
            test_loss = 0.0
            for (data1, data2) in test_loader:
                with torch.no_grad():
                    data1, data2 = data1.to(device), data2.to(device)
                    output = model(data1)
                    MA_tensor_0 = output[0].view(self.flow_no, self.k)
                    MA_tensor = F.softmax(MA_tensor_0, dim=1)
                    L_tensor = evaluation(MA_tensor, P_tensor, B_tensor, data2[0]) # evaluation stage: load on each edge
                    TE_MLU = summarization('MLU', L_tensor, C_tensor, data2[0]) # summarization stage: calculate MLU
                    TE_TP = summarization('TP',L_tensor,C_tensor, data2[0]) # summerization stage: calculate throughput
                    TE_Closs = summarization('CLoss',L_tensor,C_tensor, data2[0]) # summerization stage: calculate congestion loss
                    loss = TE_MLU
                    test_loss += loss.item()

                    MLU.append(TE_MLU.item())
                    TP.append(TE_TP.item())
                    CLoss.append(TE_Closs.item())

            test_loss /= len(test_loader)
            train_loss /= len(train_loader)
            train_loss_collect.append(train_loss)
            test_loss_collect.append(test_loss)

            print('Test set: Average train loss: {: .10f}, Average test loss: {: .10f}'.format(train_loss, test_loss))
            np.save(self.save_dir + str(alpha) + '_' + 'train_loss' + '.npy', np.array(train_loss_collect), allow_pickle=True)
            np.save(self.save_dir + str(alpha) + '_' + 'test_loss' + '.npy', np.array(test_loss_collect), allow_pickle=True)
            np.save(self.save_dir + str(alpha) + '_' + 'MLU' + '.npy', np.array(MLU), allow_pickle=True)
            np.save(self.save_dir + str(alpha) + '_' + 'TP' + '.npy', np.array(TP), allow_pickle=True)
            np.save(self.save_dir + str(alpha) + '_' + 'CLoss' + '.npy', np.array(CLoss), allow_pickle=True)
            path_string = self.save_dir + str(alpha) + '_' +'model' + '.pt'
            torch.save(model.state_dict(),path_string)
        print("complete!")

    # check the decision time of the trained model
    def check_decision_time(self, model_path):
        model = MODLE_NAME(N=self.N, output_size=self.k * self.flow_no, kernel_size=3).to(device)
        state_dict = torch.load(model_path, map_location=torch.device(device))
        print(model_path, os.path.getsize(model_path))
        model.load_state_dict(state_dict)
        train_loader, test_loader = self.my_dataloader()
        model.eval()
        for i,j in test_loader:
            data = i
            break
        data = data.to(device)
        output = model(data)
        start_time = datetime.datetime.now()
        for i in range(100):
            output = model(data)
        end_time = datetime.datetime.now()
        time_tol = (end_time - start_time).seconds + (end_time - start_time).microseconds/1e6
        return time_tol, 100
    
    # check TE metrics of the trained model
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