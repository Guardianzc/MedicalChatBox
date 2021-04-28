# -*- coding:utf-8 -*-
"""
网络模型，input是一个sequence，首先经过一个LSTM层，然后在输入给DQN。
"""

import numpy as np
import math
import copy
import pickle
import sys, os
import torch.nn as nn
import torch
from torch import autograd
import torch.nn.functional as F
from torch import optim
sys.path.append(os.getcwd().replace("src/dialogue_system/policy_learning",""))


class one_layer_MLP(nn.Module):
    def __init__(self, input_size, out_ch):
        super(one_layer_MLP, self).__init__()
        self.linear1 = torch.nn.Linear(input_size, out_ch)
        self.relu1 = torch.nn.ReLU()
    
    def forward(self, x):
        x1 = self.linear1(x)
        x2 = self.relu1(x1)
        return x2

class DQN0(object):
    """
    Initial DQN written by Qianlong, one layer.
    """
    def __init__(self, input_size, hidden_size, output_size, parameter):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.checkpoint_path = parameter.get("checkpoint_path")
        self.log_dir = parameter.get("log_dir")
        self.parameter = parameter
        self.learning_rate = parameter.get("dqn_learning_rate")
        self.__build_model()

        def __build_model(self):
            device = self.parameter.get("device_for_torch")
            self.one_layer_MLP = one_layer_MLP(self.input_size, self.output_size)
            self.device = torch.device('cuda:' + str(self.parameter['cuda_idx']) if torch.cuda.is_available() else 'cpu')
            self.one_layer_MLP = self.one_layer_MLP.to(self.device)





        