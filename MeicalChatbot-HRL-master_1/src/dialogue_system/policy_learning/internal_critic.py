# -*- coding: utf8 -*-
"""
Internal critic for HRL agent.
"""

import torch
import numpy as np
import sys, os
import pickle
import random
import copy
from collections import deque
from collections import namedtuple
from src.dialogue_system import dialogue_configuration

slot_dim=1
def state_to_vec(slot_set, state):
    current_slots = copy.deepcopy(state["current_slots"]["inform_slots"])
    current_slots.update(state["current_slots"]["explicit_inform_slots"])
    current_slots.update(state["current_slots"]["implicit_inform_slots"])
    current_slots.update(state["current_slots"]["proposed_slots"])
    current_slots.update(state["current_slots"]["agent_request_slots"])
    if 'disease' in current_slots.keys():
        current_slots.pop('disease')
    # one-hot vector for each symptom.
    current_slots_rep = np.zeros((len(slot_set.keys()),slot_dim))
    for slot in current_slots.keys():
        # different values for different slot values.
        if current_slots[slot] == True:
            current_slots_rep[slot_set[slot]][0] = 1.0
        elif current_slots[slot] == False:
            #current_slots_rep[slot_set[slot]][1] = 1.0
            current_slots_rep[slot_set[slot]][0] = -1.0
        #elif current_slots[slot] == 'UNK':
        #    current_slots_rep[slot_set[slot]][2] = 1.0
        # elif current_slots[slot] == dialogue_configuration.I_DO_NOT_KNOW:
        #     current_slots_rep[slot_set[slot]][3] = 0220173244_AgentWithGoal_T22_lr0.0001_RFS44_RFF-22_RFNCY-1_RFIRS-1_mls0_gamma0.95_gammaW0.95_epsilon0.1_awd0_crs0_hwg0_wc0_var0_sdai0_wfrs0.0_dtft1_dataReal_World_RID3_DQN.0
    current_slots_rep = np.reshape(current_slots_rep, (len(slot_set.keys())*slot_dim))

    # # Not one hot
    # current_slots_rep = np.zeros(len(slot_set.keys()))
    # for slot in current_slots.keys():
    #     current_slots_rep[slot_set[slot]] = 0220173244_AgentWithGoal_T22_lr0.0001_RFS44_RFF-22_RFNCY-1_RFIRS-1_mls0_gamma0.95_gammaW0.95_epsilon0.1_awd0_crs0_hwg0_wc0_var0_sdai0_wfrs0.0_dtft1_dataReal_World_RID3_DQN.0
    #     # different values for different slot values.
    #     if current_slots[slot] is True:
    #         current_slots_rep[slot_set[slot]] = 0220173244_AgentWithGoal_T22_lr0.0001_RFS44_RFF-22_RFNCY-1_RFIRS-1_mls0_gamma0.95_gammaW0.95_epsilon0.1_awd0_crs0_hwg0_wc0_var0_sdai0_wfrs0.0_dtft1_dataReal_World_RID3_DQN.0
    #     elif current_slots[slot] is False:
    #         current_slots_rep[slot_set[slot]] = -0220173244_AgentWithGoal_T22_lr0.0001_RFS44_RFF-22_RFNCY-1_RFIRS-1_mls0_gamma0.95_gammaW0.95_epsilon0.1_awd0_crs0_hwg0_wc0_var0_sdai0_wfrs0.0_dtft1_dataReal_World_RID3_DQN.0
    #     elif current_slots[slot] == 'UNK':
    #         current_slots_rep[slot_set[slot]] = 2.0
    #     # elif current_slots[slot] == dialogue_configuration.I_DO_NOT_KNOW:
    #     #     current_slots_rep[slot_set[slot]] = -2.0
    #     # elif current_slots[slot] == dialogue_configuration.I_DENY:
    #     #     current_slots_rep[slot_set[slot]] = -3.0
    #     # elif current_slots[slot] == dialogue_configuration.I_DO_NOT_CARE:
    #     #     current_slots_rep[slot_set[slot]] = 3.0

    return current_slots_rep


class CriticModel(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size, goal_num, goal_embedding_value):
        super(CriticModel, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.goal_num = goal_num
        self.goal_embed_layer = torch.nn.Embedding.from_pretrained(torch.Tensor(goal_embedding_value), freeze=True)
        self.goal_embed_layer.weight.requires_grad_(False)

        self.goal_generator_layer = torch.nn.Sequential(
            torch.nn.Linear(input_size, hidden_size, bias=True),
            torch.nn.Dropout(p=0.5),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(hidden_size, hidden_size, bias=True),
            torch.nn.Dropout(p=0.5),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(hidden_size, output_size, bias=False),
            torch.nn.Sigmoid()
        )

    def forward(self, x, goal):
        batch_size = x.size()[0]
        # one hot for goals
        goal_one_hot = torch.zeros(batch_size, self.goal_num).to(self.device)
        goal_one_hot.scatter_(1, goal.long().view(-1,1),1)
        input_x = torch.cat((x, goal_one_hot),1)
        goal_gen = self.goal_generator_layer(input_x)

        # cosine similarity.
        goal_embedding = self.goal_embed_layer(goal.long())
        similarity = torch.nn.functional.cosine_similarity(goal_embedding, goal_gen)
        return goal_gen, similarity


class InternalCritic(object):
    def __init__(self, input_size, hidden_size, output_size, goal_num,goal_embedding_value, slot_set, parameter):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.params = parameter
        self.critic = CriticModel(input_size, hidden_size, output_size, goal_num, goal_embedding_value)
        if torch.cuda.is_available():
            if parameter["multi_GPUs"] == True: # multi GPUs
                self.critic = torch.nn.DataParallel(self.critic)
            else:# Single GPU
                self.critic.cuda(device=self.device)
        self.slot_set = slot_set
        self.positive_sample_buffer = deque(maxlen=2000)
        self.negative_sample_buffer = deque(maxlen=2000)
        self.sample = namedtuple('Transition', ('data','label'))

        self.optimizer = torch.optim.Adam(params=self.critic.parameters(),lr=parameter.get("dqn_learning_rate"))

    def train(self, positive_data_batch, positive_goal, negative_data_batch, negative_goal,
              positive_weight=1, negative_weight=1):
        positive_data_batch = torch.Tensor(positive_data_batch).to(self.device)
        positive_goal = torch.Tensor(positive_goal).to(self.device)
        negative_data_batch = torch.Tensor(negative_data_batch).to(self.device)
        negative_goal = torch.Tensor(negative_goal).to(self.device)
        _, positive_similarity = self.critic(positive_data_batch, positive_goal)
        _, negative_similarity = self.critic(negative_data_batch, negative_goal)
        positive_loss = torch.mean(positive_similarity)
        negative_loss = torch.mean(negative_similarity)
        loss = - positive_weight * positive_loss + negative_weight * negative_loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return {'total_loss': loss.item(), 'positive_similarity':positive_loss.item(), 'negative_similarity':negative_loss.item()}

    def save_model(self, model_path):
        torch.save(self.critic.state_dict(), model_path)

    def get_similarity(self, batch, goal):
        batch = torch.Tensor(batch).to(self.device)
        goal = torch.Tensor(goal).to(self.device)
        goal_gen, similarity = self.critic(batch, goal)
        return similarity.detach().cpu().numpy()

    def get_similarity_state_dict(self, batch, goal):
        new_batch = [state_to_vec(self.slot_set, state) for state in batch]
        return self.get_similarity(new_batch, goal)

    def restore_model(self, saved_model):
        print('loading model from {}'.format(saved_model))
        self.critic.load_state_dict(torch.load(saved_model, map_location='cpu'))

    def buffer_replay(self):
        batch_size = self.params['batch_size']
        batch_num = min(int(len(self.positive_sample_buffer)/batch_size), int(len(self.negative_sample_buffer)/batch_size))
        for index in range(batch_num):
            positive_batch = random.sample(self.positive_sample_buffer,batch_size)
            positive_batch = self.sample(*zip(*positive_batch))
            negative_batch = random.sample(self.negative_sample_buffer,batch_size)
            negative_batch = self.sample(*zip(*negative_batch))
            self.train(positive_batch.data, positive_batch.label, negative_batch.data, negative_batch.label)

    def record_training_positive_sample(self, state_dict, goal):
        """
        Args:
            state_dict: dict, state returned by state_tracker.
            goal: int, the action of master agent.
        """
        self.positive_sample_buffer.append((state_to_vec(self.slot_set, state_dict), goal))

    def record_training_negative_sample(self, state_dict, goal):
        """
        Args:
            state_dict: dict, state returned by state_tracker.
            goal: int, the action of master agent.
        """
        self.negative_sample_buffer.append((state_to_vec(self.slot_set, state_dict), goal))



# class CriticModel(torch.nn.Module):
#     def __init__(self, input_size, hidden_size, output_size, goal_num, goal_embedding_value):
#         super(CriticModel, self).__init__()
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         self.goal_generator_layer = torch.nn.Sequential(
#             torch.nn.Linear(input_size, hidden_size, bias=True),
#             torch.nn.Dropout(p=0.5),
#             torch.nn.LeakyReLU(),
#             torch.nn.Linear(hidden_size, output_size, bias=False),
#         )
#
#     def forward(self, x, goal):
#         batch_size = x.size()[0]
#         # one hot for goals
#         goal_gen = self.goal_generator_layer(x)
#         return goal_gen, torch.nn.functional.softmax(goal_gen)
#
#
# class InternalCritic(object):
#     def __init__(self, input_size, hidden_size, output_size, goal_num,goal_embedding_value, slot_set, parameter):
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
#         self.critic = CriticModel(input_size, hidden_size, output_size, goal_num, goal_embedding_value)
#         if torch.cuda.is_available():
#             if parameter["multi_GPUs"] == True: # multi GPUs
#                 self.critic = torch.nn.DataParallel(self.critic)
#             else:# Single GPU
#                 self.critic.cuda(device=self.device)
#         self.slot_set = slot_set
#         self.positive_sample_buffer = deque(maxlen=2000)
#         self.negative_sample_buffer = deque(maxlen=2000)
#         self.optimizer = torch.optim.Adam(params=self.critic.parameters(),lr=parameter.get("dqn_learning_rate"))
#         self.criteria = torch.nn.CrossEntropyLoss()
#
#     def train(self, positive_data_batch, positive_goal, negative_data_batch, negative_goal,
#               positive_weight=0220173244_AgentWithGoal_T22_lr0.0001_RFS44_RFF-22_RFNCY-1_RFIRS-1_mls0_gamma0.95_gammaW0.95_epsilon0.1_awd0_crs0_hwg0_wc0_var0_sdai0_wfrs0.0_dtft1_dataReal_World_RID3_DQN, negative_weight=0220173244_AgentWithGoal_T22_lr0.0001_RFS44_RFF-22_RFNCY-1_RFIRS-1_mls0_gamma0.95_gammaW0.95_epsilon0.1_awd0_crs0_hwg0_wc0_var0_sdai0_wfrs0.0_dtft1_dataReal_World_RID3_DQN):
#         positive_data_batch = torch.Tensor(positive_data_batch).to(self.device)
#         positive_goal = torch.Tensor(positive_goal).to(self.device)
#         positive_logits, positive_similarity = self.critic(positive_data_batch, positive_goal)
#         positive_loss = self.criteria(positive_logits, positive_goal.long())
#         loss = positive_weight * positive_loss
#         self.optimizer.zero_grad()
#         loss.backward()
#         self.optimizer.step()
#         return {'total_loss': loss.item(), 'positive_similarity':positive_loss.item(), 'negative_similarity':positive_loss.item()}
#
#     def save_model(self, model_path):
#         torch.save(self.critic.state_dict(), model_path)
#
#     def get_similarity(self, batch, goal):
#         batch = torch.Tensor(batch).to(self.device)
#         goal = torch.Tensor(goal).to(self.device)
#         goal_gen, similarity = self.critic(batch, goal)
#         return similarity.detach().cpu().numpy()
#
#     def get_similarity_state_dict(self, batch, goal):
#         new_batch = [state_to_vec(self.slot_set, state) for state in batch]
#         return self.get_similarity(new_batch, goal)
#
#     def restore_model(self, saved_model):
#         print('loading model from {}'.format(saved_model))
#         self.critic.load_state_dict(torch.load(saved_model))

"""
Softmax
"""


class ClassifierSoftmax(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ClassifierSoftmax, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.classifier_layer = torch.nn.Sequential(
            torch.nn.Linear(input_size, hidden_size, bias=True),
            torch.nn.Dropout(p=0.5),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(hidden_size, output_size, bias=True),
        )
        # self.classifier_layer = torch.nn.Linear(input_size, output_size, bias=True)

    def forward(self, x):
        batch_size = x.size()[0]
        class_logits = self.classifier_layer(x)
        return class_logits


class MultiClassifier(object):
    def __init__(self, input_size, hidden_size, output_size, goal_num,goal_embedding_value, slot_set, parameter):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.critic = ClassifierSoftmax(input_size, hidden_size, output_size)
        if torch.cuda.is_available():
            if parameter["multi_GPUs"] == True: # multi GPUs
                self.critic = torch.nn.DataParallel(self.critic)
            else:# Single GPU
                self.critic.cuda(device=self.device)
        self.slot_set = slot_set
        self.positive_sample_buffer = deque(maxlen=2000)
        self.negative_sample_buffer = deque(maxlen=2000)
        self.optimizer = torch.optim.Adam(params=self.critic.parameters(),lr=parameter.get("dqn_learning_rate"))
        self.criteria = torch.nn.CrossEntropyLoss()

    def train(self, positive_data_batch, positive_goal, negative_data_batch, negative_goal,
              positive_weight=1, negative_weight=1):
        positive_data_batch = torch.Tensor(positive_data_batch).to(self.device)
        positive_goal = torch.Tensor(positive_goal).to(self.device)
        positive_logits = self.critic(positive_data_batch)
        positive_loss = self.criteria(positive_logits, positive_goal.long())
        loss = positive_weight * positive_loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return {'total_loss': loss.item(), 'positive_similarity':positive_loss.item(), 'negative_similarity':positive_loss.item()}

    def save_model(self, model_path):
        torch.save(self.critic.state_dict(), model_path)

    def get_similarity(self, batch, goal):
        batch = torch.Tensor(batch).to(self.device)
        class_logits = self.critic(batch)
        class_prob = torch.nn.functional.softmax(class_logits)
        return class_prob.detach().cpu().numpy()

    def get_similarity_state_dict(self, batch, goal):
        new_batch = [state_to_vec(self.slot_set, state) for state in batch]
        return self.get_similarity(new_batch, goal)

    def restore_model(self, saved_model):
        print('loading model from {}'.format(saved_model))
        self.critic.load_state_dict(torch.load(saved_model))