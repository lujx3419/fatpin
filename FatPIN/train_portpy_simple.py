#!/usr/bin/env python3
"""
PortPy版本的FatPIN训练代码 - 简化版本
不依赖skfda，使用简化的功能数据分析
"""

import random
import numpy as np
import math
import torch.nn as nn
import torch
from torch.nn import functional as F
from torch.distributions import Categorical  
import time
import shutil
import os
import scipy.io as scio

# 导入PortPy函数
from portpy_functions import step_fda_multipatient, reset_fda_multipatient

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 强化学习参数
eplison = 0.1
criterion = nn.CrossEntropyLoss() 
Epsilon = 0.8   # greedy policy
gamma = 0.9  # reward discount factor
actor_lr = 0.001
critic_lr = 0.01
batch_size = 1
Episode = 30
TEST = 10  


def simple_func_to_state(functional_data, domin_range):
    """简化的功能数据转状态 - 不依赖skfda"""
    # 使用简单的统计特征作为状态
    num_organs = functional_data.shape[0]
    num_features = 5  # 5个特征
    
    state = np.zeros((num_organs, num_features))
    
    for i in range(num_organs):
        # 提取简单的统计特征
        data = functional_data[i, :]
        state[i, 0] = np.mean(data)  # 均值
        state[i, 1] = np.std(data)   # 标准差
        state[i, 2] = np.max(data)   # 最大值
        state[i, 3] = np.min(data)   # 最小值
        state[i, 4] = np.median(data) # 中位数
    
    return state


class TP():
    """强化学习环境 - 重构为PortPy版本"""
    
    def __init__(self, num_of_organ):
        self.action_space = np.array([0.98, 0.99, 1, 1.01, 1.02])              
        self.observation_space = np.random.randn(num_of_organ, 5)  

    def step(self, file, action, dose, V18, D95):
        """执行一步强化学习 - 使用PortPy替代MATLAB"""
        
        # 调用PortPy版本的step函数
        result = step_fda_multipatient(file, action, dose)

        f_data = np.array(result[0])                      
        d_range = np.array(result[1]).squeeze()   
        dose_ = np.array(result[2])                       
        V18_ = np.array(result[3])
        D95_ = np.array(result[4]).reshape(1, -1)

        V18 = np.array(V18)
        D95 = np.array(D95)

        # 计算奖励
        reward = 0

        if np.all(D95_[:] > D95[:]):
            reward += 10
        elif np.any(D95_[:] > D95[:]):
            reward += 6
        else:
            reward = 0

        if np.all(V18_[:] < V18[:]):
            reward += 5
        elif np.any(V18_[:] < V18[:]):
            reward += 1
            
        done = False

        if np.any(V18[:] >= 0.5) or np.any(D95_[:] < D95[:]) or np.any(D95_[:] >= 1.82):
            done = True

        info = {}

        return f_data, d_range, reward, dose_, done, V18_, D95_

    def reset(self, file, num_of_organ):
        """重置环境 - 使用PortPy替代MATLAB"""
        
        dose = [50] * num_of_organ

        # 调用PortPy版本的reset函数
        result = reset_fda_multipatient(file, dose)
        
        functional_data = np.array(result[0])                              
        domin_range = np.array(torch.FloatTensor(result[1])).squeeze()    
        D95 = np.array(torch.FloatTensor([result[3]]))
        V18 = np.array(torch.FloatTensor(result[2]).squeeze())

        return functional_data, domin_range, dose, V18, D95

    def render(self, mode='human'):
        pass

    def seed(self, seed=None):
        pass


class PGNetwork(nn.Module):
    """策略梯度网络 - 保持原始实现"""
    
    def __init__(self, n_state, n_action):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(n_state, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, n_action) 
        )

    def forward(self, x):
        action_scores = self.fc(x)
        output = F.softmax(action_scores, dim=1)  # probability
        return output


class Actor(object):
    """Actor网络 - 保持原始实现"""
    
    def __init__(self, env):
        self.n_state = env.observation_space.shape[1]  
        self.n_action = env.action_space.shape[0]         

        self.network = PGNetwork(n_state=self.n_state, n_action=self.n_action)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=actor_lr)
        self.criterion = nn.CrossEntropyLoss()

    def choose_action(self, observation):
        observation = torch.FloatTensor(observation)
        probs = self.network(observation)  

        # 生成分布
        m = Categorical(probs)

        # 贪婪算法
        if np.random.uniform() < Epsilon:
            action = m.sample()  # 从分布中采样
            action = torch.LongTensor(action).squeeze() 
        else:
            # 随机策略
            action = np.random.randint(0, self.n_action, size=(12, 1))
            action = torch.LongTensor(action).squeeze()

        return action

    def learn(self, observation, action, td_error):
        # 状态
        observation = torch.FloatTensor(observation) 

        # 概率
        softmax_input = self.network(observation)    
        action = torch.LongTensor(action)            

        neg_log_prob = self.criterion(input=softmax_input, target=action)
        loss_a = -neg_log_prob * td_error

        self.optimizer.zero_grad()
        loss_a.backward(torch.ones_like(loss_a)) 
        self.optimizer.step()


class QNetwork(nn.Module):
    """Q网络 - 保持原始实现"""
    
    def __init__(self, n_state, n_action):
        super(QNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(n_state, 128), nn.ReLU(inplace=True),
            nn.Linear(128, 64), nn.ReLU(inplace=True),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        state_value = self.fc(x)
        return state_value


class Critic(object):
    """Critic网络 - 保持原始实现"""
    
    def __init__(self, env):
        self.n_state = env.observation_space.shape[1] 
        self.n_action = env.action_space.shape[0]    

        self.network = QNetwork(n_state=self.n_state, n_action=self.n_action)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=critic_lr)
        self.loss_func = nn.MSELoss()

    def learn(self, state, reward, state_):
        # 状态 s, s_
        s = torch.FloatTensor(state)  
        s_ = torch.FloatTensor(state_) 

        # 奖励 r
        r = torch.FloatTensor([reward]) 

        # 获取 s, s_ 的值 v, v_
        v = self.network(s)    # v(s)
        v_ = self.network(s_)  # v(s_)

        # TD
        loss_q = self.loss_func(r + gamma * v_, v)

        self.optimizer.zero_grad()
        loss_q.backward()
        self.optimizer.step()

        with torch.no_grad():
            td_error = r + gamma * v_ - v

        return td_error


def main():
    """主训练函数 - 保持原始逻辑，替换MATLAB调用"""
    
    # 创建结果目录
    os.makedirs('./result/train_info', exist_ok=True)
    os.makedirs('./result/actor', exist_ok=True)
    os.makedirs('./result/critic', exist_ok=True)

    # 实例化环境
    env = TP(num_of_organ=12)

    actor = Actor(env)
    critic = Critic(env)

    # 训练轮次
    for episode in range(Episode):
        file_list = ['patient2.mat', 'patient2.mat', 'patient3.mat', 'patient4.mat', 'patient5.mat']  # 患者列表
        num_of_file = 0
        for file in file_list:
            num_of_file += 1
            print('training in the file', file)
            
            # 初始化环境
            f_data, d_range, dose, V18, D95 = env.reset(file, num_of_organ=12)  

            state = simple_func_to_state(f_data, d_range)
            
            total_reward = -300

            print("Initialization completed, Patient No. {0}, Round No. {0} begins".format(num_of_file, episode+1))
            
            for i_train in range(50):  
                env.render()  # 渲染环境

                # 选择动作
                action_index = actor.choose_action(state) 
                action = env.action_space[action_index]   

                # 与PortPy交互 (替代MATLAB)
                functional_data, domin_range, reward, dose_, done, V18_, D95_ = env.step(file, action, dose, V18, D95)
                
                # 状态转换
                state_ = simple_func_to_state(functional_data, domin_range)
                total_reward += reward

                print('Round {0} --- Patient {1} --- Step {2} --- Action {3} --- Reward {4}'.format(
                    episode+1, num_of_file, i_train+1, action, reward))

                # 训练critic
                state = torch.FloatTensor(state).unsqueeze(dim=0).unsqueeze(dim=0)
                state_ = torch.FloatTensor(state_).unsqueeze(dim=0).unsqueeze(dim=0)

                td_error = critic.learn(state, reward, state_)

                # 训练Actor
                state = state.squeeze(dim=0).squeeze(dim=0)
                state_ = state_.squeeze(dim=0).squeeze(dim=0)

                actor.learn(state, action_index, td_error)

                # 记录训练信息
                with open('./result/train_info/train_{}_rounds'.format(episode+1) + '.csv', 'a', encoding='utf-8') as f:
                    f.write('patient{}'.format(num_of_file) + 'the {}th training'.format(i_train+1)
                            + ' dose is' + ','.join(map(str, np.array(dose_).reshape(1, -1)))
                            + ' V18:' + str(np.array(V18_).reshape(1, -1)) + ' D95:' + str(np.array(D95_).reshape(1, -1))
                            + ' reward is ' + str(np.array(reward).reshape(1, -1))
                            + ' total reward is ' + str(np.array(total_reward).reshape(1, -1))
                            + '\n')

                if done:
                    state = state
                    dose = dose
                    V18 = V18
                    D95 = D95
                else:
                    state = state_
                    dose = dose_  # 更新剂量
                    V18 = V18_
                    D95 = D95_

                if i_train % 10 == 0:
                    torch.save(actor, './result/actor/multi_{0}_{1}.pth'.format(episode+1, i_train+1))
                    torch.save(critic, './result/critic/multi_{0}_{1}.pth'.format(episode+1, i_train+1))


if __name__ == "__main__":
    main()
