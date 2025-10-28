#!/usr/bin/env python3
"""
PortPy版本的FatPIN测试代码
重构自test.py，替换MATLAB依赖为PortPy
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
import skfda

# 导入PortPy函数
from portpy_functions import step_fda_multipatient, reset_fda_multipatient

# 强化学习参数
eplison = 0.1
criterion = nn.CrossEntropyLoss()  
Epsilon = 0.9  # greedy policy
gamma = 0.9  # reward discount factor
actor_lr = 0.001
critic_lr = 0.01
batch_size = 1
Episode = 5
TEST = 10  
num_of_organ = 12


def composite_approximator(f_beta, f_xt, a, b, n):
    """复合近似器 - 保持原始实现"""
    def f_x(t):
        x = t
        return eval(f_beta) * (f_xt(x).ravel()) 

    # 使用复合辛普森公式计算积分
    integ_approx = ((b-a))/6 * (f_x(a) + 4*f_x(((a+b)/2)) + f_x(b))
    return integ_approx


def intergration_form_fourier(functional_data,
                            beta_basis=None, 
                            num_fd_basis=None,  # fd基函数数量
                            num_beta_basis=None,  # beta基函数数量
                            domin_range=None):
    """傅里叶积分形式 - 保持原始实现"""
    
    #### 设置 x_i(s) 形式 ####
    grid_point = np.array(domin_range).squeeze()             
    data_matrix = np.array(functional_data)[np.newaxis, :]  
    basis = skfda.representation.basis.BSpline(n_basis=num_fd_basis)
    fd = skfda.FDataGrid(data_matrix=data_matrix, grid_points=grid_point)
    X_basis = fd.to_basis(basis)

    #### 设置 beta_(s) ####
    beta_basis_form = []
    beta_basis_form.append('1')

    for m in range(1, (num_beta_basis-1)//2+1):
        beta_basis_form.append(('np.sin(2*np.pi*x'+'*'+str(m)+'/'+str(domin_range[-1])+')'))
        beta_basis_form.append(('np.cos(2*np.pi*x'+'*'+str(m)+'/'+str(domin_range[-1])+')'))

    #### 获取近似值 ####
    integ_approximations = []
    for m in range(len(beta_basis_form)):
        # 计算 Φm(t)X(t)
        form_approximated = str(beta_basis_form[m])
        final_func = form_approximated

        integ_approximations.append(composite_approximator(
            f_beta=final_func, f_xt=X_basis, 
            a=domin_range[0], b=domin_range[-1], n=5000))

    return integ_approximations


def func_to_state(functional_data, domin_range):
    """功能数据转状态 - 保持原始实现"""
    num_fd_basis = 40
    num_beta_basis = 5 
    state = np.zeros((functional_data.shape[0], num_beta_basis))
    for i in range(functional_data.shape[0]):
        state[i, :] = intergration_form_fourier(
            functional_data[i, :], 
            num_fd_basis=num_fd_basis, 
            num_beta_basis=num_beta_basis, 
            domin_range=domin_range)

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
        if np.all(D95_[:] > D95[:]) and np.all(V18_[:] < V18[:]):
            reward = 10
        elif np.all(D95_[:] > D95[:]):
            reward = 5
        elif np.all(V18_[:] < V18[:]):
            reward = 3
        else:
            reward = 0
            
        done = False

        if np.any(V18[:] >= 0.5) or np.any(D95_[:] < D95[:]):
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
            action = np.random.randint(0, self.n_action, size=(14, 1))
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
    """主测试函数 - 保持原始逻辑，替换MATLAB调用"""
    
    # 创建结果目录
    os.makedirs('./result/test_info', exist_ok=True)

    # 实例化环境
    env = TP(num_of_organ=12)
    
    # 加载模型
    actor_test = torch.load('./result/actor/multi_20_20.pth', map_location=None)

    file = 'test_patient.mat'
    
    for episode in range(Episode):
        
        ## 初始化环境
        f_data, d_range, dose, V18, D95 = env.reset(file, num_of_organ=12) 

        state = func_to_state(f_data, d_range)
        
        reward_total = 0

        print("Initialization completed, Round No. {0} begins".format(episode))

        for i_test in range(30):
            # 选择动作
            action_index = actor_test.choose_action(state) 
            action = env.action_space[action_index]   

            # 与PortPy交互 (替代MATLAB)
            functional_data, domin_range, reward, dose_, done, V18_, D95_ = env.step(file, action, dose, V18, D95)
            
            # 状态转换
            state_ = func_to_state(functional_data, domin_range)
            reward_total += reward

            print('Step {} --- Action {}--- Reward {}'.format(i_test, action, reward))

            state = torch.FloatTensor(state).unsqueeze(dim=0).unsqueeze(dim=0)
            state_ = torch.FloatTensor(state_).unsqueeze(dim=0).unsqueeze(dim=0)

            # 记录测试信息
            with open('./result/test_info/test_{}_rounds'.format(episode) + '.csv', 'a', encoding='utf-8') as f:
                f.write(' The {}th test'.format(i_test)
                + ' dose is ' + ','.join(map(str, np.array(dose).reshape(1, -1)))
                + ' V18 is ' + str(np.array(V18_).reshape(1, -1))
                + ' D95 is ' + str(np.array(D95_).reshape(1, -1))
                + ' reward is ' + str(np.array(reward).reshape(1, -1))
                + ' total reward is ' + str(np.array(reward_total).reshape(1, -1))
                + '\n')
                
            state = state_
            dose = dose_  # 更新剂量
            V18 = V18_
            D95 = D95_


if __name__ == "__main__":
    main()
