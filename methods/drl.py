import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch
import datetime
import random
import numpy as np
import os
import numpy as np
from framework import evaluation, summarization
from AssistFunc import adjust_learning_rate
import datetime
# from training_time_eval import sim_tp
import time

# fix log:
# reward problem: reward not right
# dim problem: dimension of action

# check log:
# exploition shape
# reward and done size


torch.set_default_dtype(torch.float64)
time_c = 0
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

# return action
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, k):
        super().__init__()
        self.k = k
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.l1 = nn.Linear(self.state_dim, 512)
        self.l2 = nn.Linear(512, 256)
        self.l3 = nn.Linear(256, self.action_dim)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.l3(x)
        return x
    
    # needs more work here
    def get_action(self, state):
        state = torch.from_numpy(state).to(device)
        action = self.forward(state)
        if len(action.size()) > 1:
            action = action.view(len(action),self.action_dim//self.k, self.k)
            action = F.softmax(action, dim=2) # bug fixed
        else:
            action = action.view(self.action_dim//self.k, self.k)
            action = F.softmax(action, dim=1)
        return action.detach().cpu().numpy()


# return Q value
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.l1 = nn.Linear(state_dim + action_dim, 512)
        self.l2 = nn.Linear(512, 256)
        self.l3 = nn.Linear(256, 1)

    def forward(self, state, action):
        if len(action.size()) > 1:
            action = action.view(len(action),-1)
        else:
            action = action.view(-1)
        x = F.relu(self.l1(torch.cat([state, action], 1)))
        x = F.relu(self.l2(x))
        x = self.l3(x)
        return x

class Replay_buffer:
    def __init__(self, capacity):
        # tuple: (state, action, reward, next_state, done)
        self.buffer = []
        self.capacity = capacity
        self.position = 0

    # push data into replay buffer
    def push(self, state, action, reward, next_state, done):
        # if the buffer is full, pop the oldest one
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity
    
    # sample 'batch_size' actions.
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)
        
class TE_ENV(object):
    def __init__(self, traffic, net, k, P_tensor, B_tensor, C_tensor, name):
        # init the environment
        self.traffic = traffic 
        self.net = net
        self.k = k
        self.P_tensor = P_tensor
        self.B_tensor = B_tensor
        self.C_tensor = C_tensor
        self.N = len(self.net.nodes())
        self.edge_no = len(self.net.edges())
        self.flow_no = self.N * self.N
        self.state_dim = self.flow_no + self.edge_no * 2 + 1
        self.action_dim = self.k * self.flow_no
        self.state = None
        self.name = name # topology name

    def gen_state(self, action, idx):
        start_time1 = time.time()
        device0 = 'cpu'
        self.state = np.zeros(self.state_dim)
        # traffic of next state
        if idx + 1 >= len(self.traffic):
            self.state[0:self.flow_no] = self.traffic[idx]
        else:
            self.state[0:self.flow_no] = self.traffic[idx+1]
        if idx == -1:
            idx = 0
        # information of current step
        # link utilization
        traffic = torch.from_numpy(self.traffic[idx]).to(device0)
        action = torch.from_numpy(action).to(device0)
        L_tensor = evaluation(action, self.P_tensor.to(device0), self.B_tensor.to(device0), traffic)
        lu = L_tensor.cpu().numpy()
        self.state[self.flow_no: self.flow_no + self.edge_no*2] = lu
        # throughput
        TP = summarization('TP',L_tensor,self.C_tensor.to(device0), traffic)
        tp = TP.cpu().numpy()
        self.state[self.flow_no + self.edge_no*2] = tp
        # mlu and closs
        TE_MLU = summarization('MLU', L_tensor, self.C_tensor.to(device0), traffic)
        TE_Closs = summarization('CLoss',L_tensor,self.C_tensor.to(device0), traffic)
        mlu, closs = TE_MLU.cpu().numpy(), TE_Closs.cpu().numpy()
        end_time1 = time.time()
        print(end_time1-start_time1)
        # sim_tp(self.name, True)
        # global time_c
        # time_c += 1
        # mlu, closs, tp = 10, 10, 10
        # self.state = np.zeros(self.state_dim,dtype=np.float64)
        return self.state, mlu, closs, tp

    def step(self, action, idx, alpha):
        done = False # never done
        self.state, mlu, closs, _ = self.gen_state(action, idx)
        reward = 0 - (mlu + alpha * closs)
        return self.state, reward, done, {}

    # return to the situation of first traffic matrix
    def reset(self):
        action = np.ones(shape=(self.action_dim//self.k, self.k)) * 1/self.k
        self.state, _, _, _ = self.gen_state(action, -1)
        return self.state
    
    def metric(self, action, idx):
        _, mlu, closs, tp = self.gen_state(action, idx)
        return mlu, tp, closs
    
    def exploition(self, action, epoch, sigma):
        # print('action shape: ',action.shape) # To check
        random_val = random.uniform(0,1)
        decay_sigma = sigma/(2**(epoch//25))
        new_sigma = decay_sigma/(10*epoch+10)
        noise = np.random.randn(self.action_dim)
        noise = noise.reshape(self.flow_no, self.k)
        normalized_noise = 2 * (noise - np.min(noise)) / (np.max(noise) - np.min(noise)) - 1
        base_action = np.ones(self.action_dim) * 1/self.k
        base_action = base_action.reshape(self.flow_no, self.k)
        if random_val <= decay_sigma:
            action = base_action + new_sigma * normalized_noise
        else:
            action = action + decay_sigma * normalized_noise
        # re-softmax
        action = torch.from_numpy(action)
        action = F.softmax(action, dim=1)
        action = action.numpy()
        return action

class DRL(object):
    def __init__(self, state_dim, action_dim, buf_capacity, batch_size, gamma, tau, actor_lr, critic_lr, k):
        super().__init__()
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.buf_capacity = buf_capacity
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.k = k
        self.min_value = -np.inf
        self.max_value = np.inf
        # Actor network
        self.actor = Actor(state_dim, action_dim, self.k).to(device).double()
        self.actor_target = Actor(state_dim, action_dim, self.k).to(device).double()
        
        # Critic network
        self.critic = Critic(state_dim, action_dim).to(device).double()
        self.critic_target = Critic(state_dim, action_dim).to(device).double()

        # parameters
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data)
        
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)
        
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.critic_lr)
        self.criterion = nn.MSELoss()
        self.replay_buffer = Replay_buffer(self.buf_capacity)
    
    def update(self):
        state, action, reward, next_state, done = self.replay_buffer.sample(self.batch_size)
        state = torch.from_numpy(state).to(device)
        next_state = torch.from_numpy(next_state).to(device)
        action = torch.from_numpy(action).to(device)
        reward = torch.from_numpy(reward).unsqueeze(1).to(device) # needs more work
        # print('reward size: ',reward.size())
        done = torch.from_numpy(np.float64(done)).unsqueeze(1).to(device) # needs more work
        # print('reward size: ',done.size())


        actor_loss = self.critic(state, self.actor(state))
        actor_loss = -torch.mean(actor_loss)
        next_action = self.actor_target(next_state)
        target_value = self.critic_target(next_state, next_action.detach())
        expected_value = reward + (1.0-done) * self.gamma * target_value
        # expected_value = torch.clamp(expected_value, self.min_value, self.max_value) # needs more work

        value = self.critic(state, action)
        critic_loss = self.criterion(value, expected_value.detach())

        # update actor network
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # update critic network
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # update target network
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)
        
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)
        
class RUN_DRL(object):
    def __init__(self, net, path_path, traffic_path, ca_path, train_size, save_dir, scale, k, batch_size, name):
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
        self.name = name
        print("DRL is selected.")

    def run(self, alpha=0, learning_rate_actor=0.0001, learning_rate_critic=0.0001, gamma=0.99, tau=0.01, sigma=0.6):
        seed = 66 # for abilene
        seed_everything(seed)
        epoch = 3
        print('----------------------- Information -----------------------')
        print('DRL\n', self.save_dir.split('/')[-2], '\n', 'epoch: ',epoch,'\tN: ',self.N,'\tDataset size: ',self.train_size,'\tk: ',self.k,'\talpha: ',alpha,'\n')
        print('----------------------- Information -----------------------')

        # Prepare paths information
        path = np.load(self.path_path,allow_pickle=True)
        P = np.array(path.tolist(), dtype=np.float64)

        # Prepare capacity information
        C = np.load(self.ca_path, allow_pickle=True)
        C = np.array(C.tolist(), dtype=np.float64)

        # transfer P and C to tensor type
        P_tensor = torch.from_numpy(P)
        C_tensor = torch.from_numpy(C)

        B = np.zeros(shape=(self.flow_no, self.k * self.flow_no))
        for i in range(self.flow_no):
            for j in range(i * self.k, (i+1) * self.k):
                B[i][j] = 1.0
        B_tensor = torch.from_numpy(B)

        traffic_data = np.load(self.traffic_path, allow_pickle=True) * self.scale # load data

        # initialize TE environment
        env = TE_ENV(traffic=traffic_data, net=self.net, k=self.k, P_tensor=P_tensor, B_tensor=B_tensor, C_tensor=C_tensor, name=self.name)

        model = DRL(state_dim=env.state_dim, action_dim=env.action_dim, buf_capacity=10000, batch_size=self.batch_size, gamma=gamma, tau=tau, actor_lr=learning_rate_actor, critic_lr=learning_rate_critic, k=self.k)
        
        train_steps = 4500
        test_steps = 500

        train_loss_collect = []
        test_loss_collect = []

        start_time = datetime.datetime.now()
        for i in range(epoch):
            global time_c
            time_c = 0
            # adjust_learning_rate(model.actor_optimizer, i, learning_rate_actor, 200) # adjust learning rate of actor
            # adjust_learning_rate(model.critic_optimizer, i, learning_rate_critic, 200) # adjust learning rate of critic
            state = env.reset()
            model.actor.train()
            model.critic.train()
            model.actor_target.train()
            model.critic_target.train()
            train_loss = 0.0
            for step in range(train_steps):
                action = model.actor.get_action(state)
                action = env.exploition(action, i, sigma) # more work

                next_state, reward, done, _ = env.step(action, step, alpha)
                model.replay_buffer.push(state, action, reward, next_state, done)
                # batch update
                if len(model.replay_buffer) > self.batch_size:
                    model.update()
                # move to next state
                state = next_state
                train_loss -= reward

            model.actor.eval()
            model.critic.eval()
            model.actor_target.eval()
            model.critic_target.eval()
            MLU = []
            TP = []
            CLoss = []
            test_loss = 0.0
            for step in range(train_steps, train_steps+test_steps):
                action = model.actor.get_action(state)
                mlu, tp, closs = env.metric(action, step)
                MLU.append(mlu)
                TP.append(tp)
                CLoss.append(closs)
                test_loss += mlu
                if step < train_steps + test_steps - 1:
                    next_state, reward, done, _ = env.step(action, step, alpha)
                    state = next_state
            
            test_loss /= test_steps
            train_loss /= train_steps
            train_loss_collect.append(train_loss)
            test_loss_collect.append(test_loss)

            print(str(i)+': Test set: Average train loss: {: .10f}, Average test loss: {: .10f}'.format(train_loss, test_loss))
            print('time_c',time_c,'\n')
            # np.save(self.save_dir + str(alpha) + '_' + 'train_loss' + '.npy', np.array(train_loss_collect), allow_pickle=True)
            # np.save(self.save_dir + str(alpha) + '_' + 'test_loss' + '.npy', np.array(test_loss_collect), allow_pickle=True)
            # np.save(self.save_dir + str(alpha) + '_' + 'MLU' + '.npy', np.array(MLU), allow_pickle=True)
            # np.save(self.save_dir + str(alpha) + '_' + 'TP' + '.npy', np.array(TP), allow_pickle=True)
            # np.save(self.save_dir + str(alpha) + '_' + 'CLoss' + '.npy', np.array(CLoss), allow_pickle=True)
            # path_string = self.save_dir + str(alpha) + '_' +'model_actor' + '.pt'
            # torch.save(model.actor.state_dict(),path_string)
            # path_string = self.save_dir + str(alpha) + '_' +'model_actor_target' + '.pt'
            # torch.save(model.actor_target.state_dict(),path_string)
            # path_string = self.save_dir + str(alpha) + '_' +'model_critic' + '.pt'
            # torch.save(model.critic.state_dict(),path_string)
            # path_string = self.save_dir + str(alpha) + '_' +'model_critic_target' + '.pt'
            # torch.save(model.critic_target.state_dict(),path_string)
        end_time = datetime.datetime.now()
        time_count = (end_time - start_time).seconds + (end_time - start_time).microseconds/1e6
        print("complete!")
        return time_count
    
    def check_decision_time(self, model_path, learning_rate_actor=0.0001, learning_rate_critic=0.0001, gamma=0.99, tau=0.01, sigma=0.6):
        path = np.load(self.path_path,allow_pickle=True)
        P = np.array(path.tolist(), dtype=np.float64)
        C = np.load(self.ca_path, allow_pickle=True)
        C = np.array(C.tolist(), dtype=np.float64)
        P_tensor = torch.from_numpy(P)

        C_tensor = torch.from_numpy(C)
        B = np.zeros(shape=(self.flow_no, self.k * self.flow_no))
        for i in range(self.flow_no):
            for j in range(i * self.k, (i+1) * self.k):
                B[i][j] = 1.0
        B_tensor = torch.from_numpy(B)
        traffic_data = np.load(self.traffic_path, allow_pickle=True) * self.scale # load data

        # initialize TE environment and model
        env = TE_ENV(traffic=traffic_data, net=self.net, k=self.k, P_tensor=P_tensor, B_tensor=B_tensor, C_tensor=C_tensor,name=self.name)
        model = DRL(state_dim=env.state_dim, action_dim=env.action_dim, buf_capacity=10000, batch_size=64, gamma=gamma, tau=tau, actor_lr=learning_rate_actor, critic_lr=learning_rate_critic, k=4)

        state_dict1 = torch.load(model_path+'actor.pt', map_location=torch.device(device))
        state_dict2 = torch.load(model_path+'actor_target.pt', map_location=torch.device(device))
        state_dict3 = torch.load(model_path+'critic.pt', map_location=torch.device(device))
        state_dict4 = torch.load(model_path+'critic_target.pt', map_location=torch.device(device))

        model.actor.to(device).load_state_dict(state_dict1)
        model.actor_target.to(device).load_state_dict(state_dict2)
        model.critic.to(device).load_state_dict(state_dict3)
        model.critic_target.to(device).load_state_dict(state_dict4)
        
        state = env.reset()

        time_tol = 0
        for i in range(500):
            start_time = datetime.datetime.now()
            action = model.actor.get_action(state)
            end_time = datetime.datetime.now()
            cur_time = (end_time - start_time).seconds + (end_time - start_time).microseconds/1e6
            # print(cur_time)
            if i == 0:
                continue
            time_tol += cur_time
        return time_tol, 499
    
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



