import numpy as np
import random
from BlackjackSM import BlackjackSM
import pdb

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

import matplotlib.pyplot as plt

FloatTensor = torch.FloatTensor
LongTensor = torch.LongTensor
ByteTensor = torch.ByteTensor

# hyperparameters
BATCH_SIZE = 256
GAMMA = 0.999
# EPS_START = 1
EPS_START = 0.05
EPS_END = 0.05
EPS_DECAY = 20000
WEIGHT_DECAY = 0.0001

num_episodes = 300000
# start_period = 50000
start_period = 0
decay_period = num_episodes - 150000 - start_period
learning_rate = 0.0001
# how often to take gradient descent steps
C = 4

graph_interval = 10000

state_machine = BlackjackSM()
n_in = state_machine.len_state
n_out = state_machine.len_actions

# NUM_LAYERS = 11
# # k = np.rint(NUM_LAYERS / 2 + 0.5)
# k = 13
network_params = [(3,1), (3,3), (3,5), (3,7), (3,9), (3,11), (3,13), 
                  (3,15), (5,3), (5,7), (7,1), (7,3), (7,7), (7,13), 
                  (9,7), (11,7), (11,9), (11,11), (11,13), (13,7)]


class Memory():
	def __init__(self, capacity):
		self.capacity = capacity
		self.memory = []
		self.position = 0

	def push(self, transition):
		if len(self.memory) < self.capacity:
			self.memory.append(transition)
		else:
			self.memory[self.position] = transition
			self.position = (self.position + 1) % self.capacity

	def sample(self, batchsize = 10):
		'''打乱顺序 batchsize 大小的样本'''
		return random.sample(self.memory, batchsize)

	def __len__(self):
		return len(self.memory)

	def __str__(self):
		return str(self.memory)

class DQN(nn.Module):
	def __init__(self, modules):
		super(DQN, self).__init__()
		for layer, module in enumerate(modules):
			self.add_module("layer_" + str(layer), module)

	def forward(self, x):
		for layer in self.children():
			x = layer(x)
		return x


for parameters in network_params:
    NUM_LAYERS, k = parameters
    
    # build network
    layers_size = {-1: n_in}
    factor = (n_out/k/n_in)**(1/(NUM_LAYERS - 1))
    for layer in range(NUM_LAYERS):
        layers_size[layer] = int(np.rint(k*n_in * factor**(layer)))
    print(layers_size)
    
    # 构建网络模型
    modules = []
    for i in layers_size.keys():
        if i == -1: continue
    modules.append(nn.Linear(layers_size[i-1],layers_size[i]))
    if i < NUM_LAYERS - 1:
        modules.append(nn.BatchNorm1d(layers_size[i]))
        modules.append(nn.ReLU())
        # modules.append(nn.Dropout(0.15))
    
    # initialize model
    model = DQN(modules)
    try: 
        model.load_state_dict(torch.load("models/blackjack_DQN_" + str(NUM_LAYERS) + "-" + str(k) + ".pt"))
        print("loaded saved model")
    except:
        print("no saved model")
    
    # optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=WEIGHT_DECAY)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=WEIGHT_DECAY)
    memory = Memory(20000)
    model.eval() # ❓
    
    
    # training helper functions
    counter = 0 # 记录状态数
    def select_action():
        '''
		return: <tensor>
				-> 0: 要牌
				-> 1: 停牌
				-> 2: 投降
				-> 3: 双倍
				-> 4: 分牌
		'''
        global counter
        unif_draw = np.random.rand()
        if counter < start_period: # start_period❓
            return LongTensor(np.array([random.choice(state_machine.actions())]))
        
        # 探索率
        eps = EPS_END + max((EPS_START - EPS_END) * (1 - np.exp((counter - start_period - decay_period)/EPS_DECAY)), 0)
        
        scores = model(Variable(FloatTensor(np.array([state_machine.state()])), volatile=True)).data # ❓
        mask = ByteTensor(1 - state_machine.mask())
        best_action = (scores.masked_fill_(mask, -16)).max(-1)[1]
        
        if unif_draw > eps: # 利用
            return LongTensor(best_action)
        else: # 探索
            # actions = state_machine.actions()
			# p = np.abs(scores.numpy()[0])
			# p = (max(np.max(p),1) - p)[actions]
			# p = p / np.sum(p)
			# choice = np.array([np.random.choice(actions, p = p)])
			# return LongTensor(choice)
   
            actions = state_machine.actions()
            actions.remove(best_action[0]) # 移除 best_action 中概率最大的动作
            return LongTensor(np.array([random.choice(actions)]))
    
    def optimize_model():
        # 只有缓存一个 BATCH_SIZE 大小才开始优化模型
        if len(memory) < BATCH_SIZE:
            return
        model.train()
        sample = memory.sample(BATCH_SIZE) # 每次从 memory 数据池中随机选择 BATCH_SIZE 大小的样本
        
        # 将 BATCH_SIZE 大小的数据分别按 状态、动作、奖励值 batch 分类存成 list
        batch = list(zip(*sample))
        state_batch = Variable(torch.cat(batch[0]))
        action_batch = Variable(torch.cat(batch[1]))
        reward_batch = Variable(torch.cat(batch[3]))
        
        # 计算 BATCH_SIZE 大小的数据的 Q 值
        Q_sa = model(state_batch).gather(1, action_batch.view(-1,1)).squeeze() # ❓
        V_s = Variable(torch.zeros(BATCH_SIZE).type(FloatTensor))
        
        '''对不是终止状态的数据进行参数更新'''
        # 将没有下一个状态的样本数据置 0，有的置 1
        not_terminal = ByteTensor(tuple(map(lambda s: s is not None, batch[2])))
        if not_terminal.sum() > 0: 
            model.eval() # ❓
            # 将有下一状态的 state 数据提出来
            not_terminal_states = Variable(torch.cat([s for s in batch[2] if s is not None]), volatile=True)
            # 将有下一状态的数据，不是新局面和不是对子的重新标记，置为 1
            masks = ByteTensor(np.array([1 - state_machine.mask_for(s) for s in not_terminal_states.data.numpy()]))
            # 将有下一状态的数据，用 model 计算概率后，更新 V_s 表
            V_s[not_terminal] = (model(not_terminal_states).data.masked_fill_(masks, -16)).max(1)[0] # ❓
            model.train() # ❓
        observed_sa = reward_batch + (V_s * GAMMA) # 奖励值的预测值
        
        # 用 L1 计算奖励值的 loss
        loss = F.smooth_l1_loss(Q_sa, observed_sa)
        
        # pdb.set_trace()

		# ❓
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm(model.parameters(), max_norm=2, norm_type=2)
        optimizer.step()
        model.eval() # ❓

    
    # training
    iterations = []
    avg_reward_graph = []
    avg_reward = 0
    avg_score = 0
    for episode in range(num_episodes):
        state_machine.new_hand() # 发牌
        state = FloatTensor([state_machine.state()]) # 起始牌的状态
        
        # 开始一局新游戏
        while True:
            action = select_action() # 选择一个动作
            state_machine.do(int(action[0])) # 做出 action 行动
            reward = FloatTensor([state_machine.reward()])
            
            if not state_machine.terminal:
                next_state = FloatTensor([state_machine.state()])
            else:
                next_state = None
            
            memory.push([state, action, next_state, reward])
            state = next_state
            
            if counter % C == 0:
                optimize_model()
            counter += 1
            
            if state_machine.terminal:
                break
            
        if episode > (num_episodes - 5000):
            avg_score += state_machine.reward() / 5000
        
        # 记录便于画图的变量
        avg_reward += state_machine.reward()
        if episode % graph_interval == 0:
            iterations.append(episode)
            avg_reward_graph.append(avg_reward / graph_interval)
            avg_reward = 0
    
    torch.save(model.state_dict(), "models/blackjack_DQN_" + str(NUM_LAYERS) + "-" + str(k) + ".pt")
    print("saved model")
    
    print(avg_score)
    
    plt.title("Average Reward Over Time for a DQN with " + str(NUM_LAYERS) + " Layers and Size Factor " + str(k))
    plt.ylabel("return per episode")
    plt.xlabel("number of episodes")
    plt.plot(iterations, avg_reward_graph)
    # plt.show()
    plt.savefig(fname='./img/average_reward' + '_' + str(NUM_LAYERS) + '_' + str(k), dpi=150)
