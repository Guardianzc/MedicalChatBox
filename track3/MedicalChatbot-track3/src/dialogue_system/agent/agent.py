# -*- coding: utf-8 -*-
"""
Basic agent class that other complicated agent, e.g., rule-based agent, DQN-based agent.
"""

import numpy as np
import copy
import random
import json
import sys, os
from collections import deque
sys.path.append(os.getcwd().replace("src/dialogue_system/agent",""))

from src.dialogue_system import dialogue_configuration


class Agent(object):
    """
    Basic class of agent.
    """
    def __init__(self, action_set, slot_set, disease_symptom, parameter):
        self.action_set = action_set
        self.slot_set = slot_set
        # self.disease_symptom = disease_symptom
        self.disease_symptom = self.disease_symptom_clip(disease_symptom, parameter)
        self.parameter = parameter
        self.candidate_disease_list = []
        self.candidate_symptom_list = []
        self.action_space = self._build_action_space()
        self.agent_action = {
            "turn":1,
            "action":None,
            "request_slots":{},
            "inform_slots":{},
            "explicit_inform_slots":{},
            "implicit_inform_slots":{},
            "speaker":"agent"
        }

    def initialize(self):
        """
        Initializing an dialogue session.
        :return: nothing to return.
        """
        self.candidate_disease_list = []
        self.candidate_symptom_list = []
        self.agent_action = {
            "turn":None,
            "action":None,
            "request_slots":{},
            "inform_slots":{},
            "explicit_inform_slots":{},
            "implicit_inform_slots":{},
            "speaker":"agent"
        }

    def next(self, state, turn, greedy_strategy):
        """
        Taking action based on different methods, e.g., DQN-based AgentDQN, rule-based AgentRule.
        Detail codes will be implemented in different sub-class of this class.
        :param state: a vector, the representation of current dialogue state.
        :param turn: int, the time step of current dialogue session.
        :param train_mode: int, 1:for training, 0:for evaluation
        :return: a tuple consists of the selected agent action and action index.
        """
        return self.agent_action

    def train(self, batch):
        """
        Training the agent.
        Detail codes will be implemented in different sub-class of this class.
        :param batch: the sample used to training.
        :return:
        """
        pass

    def state_to_representation_last(self, state):
        """
        Mapping dialogue state, which contains the history utterances and informed/requested slots up to this turn, into
        vector so that it can be fed into the model.
        This mapping function uses informed/requested slots that user has informed and requested up to this turn .
        :param state: Dialogue state
        :return: Dialogue state representation with 1-rank, which is a vector representing dialogue state.
        """
        # Current_slots rep.
        current_slots = copy.deepcopy(state["current_slots"]["inform_slots"])
        current_slots.update(state["current_slots"]["explicit_inform_slots"])
        current_slots.update(state["current_slots"]["implicit_inform_slots"])
        current_slots.update(state["current_slots"]["proposed_slots"])
        current_slots_rep = np.zeros(len(self.slot_set.keys()))
        for slot in current_slots.keys():
            if current_slots[slot] == '1':
                current_slots_rep[self.slot_set[slot]] = 1.0
            elif current_slots[slot] == '0':
                current_slots_rep[self.slot_set[slot]] = -1.0
            else:
                current_slots_rep[self.slot_set[slot]] = -2.0
        # Turn rep.
        turn_rep = np.zeros(self.parameter["max_turn"])
        turn_rep[state["turn"]] = 1.0
        '''
        # Agent last request slot rep.
        agent_request_slots_rep = np.zeros(len(self.slot_set.keys()))
        try:
            agent_request_slots = copy.deepcopy(state["agent_action"]["request_slots"])
            for slot in agent_request_slots.keys():
                agent_request_slots_rep[self.slot_set[slot]] = 1.0
        except:
            pass
        '''
        # state_rep = np.hstack((current_slots_rep, wrong_diseases_rep, user_action_rep, user_inform_slots_rep, user_request_slots_rep, agent_action_rep, agent_inform_slots_rep, agent_request_slots_rep, turn_rep))
        # state_rep = np.hstack((current_slots_rep, agent_request_slots_rep, turn_rep))
        state_rep = np.hstack((current_slots_rep, turn_rep))
        return state_rep

    def _build_action_space(self):
        """
        Building the Action Space for the RL-based Agent.
        All diseases are treated as actions.
        :return: Action Space, a list of feasible actions.
        """
        feasible_actions = []
        #   Adding the inform actions and request actions.
        for slot in sorted(self.slot_set.keys()):
            feasible_actions.append({'action': 'request', 'inform_slots': {}, 'request_slots': {slot: dialogue_configuration.VALUE_UNKNOWN},"explicit_inform_slots":{}, "implicit_inform_slots":{}})
        # Diseases as actions.
        for disease in sorted(self.disease_symptom.keys()):
            feasible_actions.append({'action': 'inform', 'inform_slots': {"disease":disease}, 'request_slots': {},"explicit_inform_slots":{}, "implicit_inform_slots":{}})

        return feasible_actions

    def disease_symptom_clip(self, disease_symptom, parameter):
        max_turn = parameter.get('max_turn')
        temp_disease_symptom = copy.deepcopy(disease_symptom)
        for key, value in disease_symptom.items():
            symptom_list = sorted(value['Symptom'].items(),key = lambda x:x[1],reverse = True)
            symptom_list = [v[0] for v in symptom_list]
            symptom_list = symptom_list[0:min(len(symptom_list), int(max_turn / 2.5))]
            temp_disease_symptom[key]['symptom'] = symptom_list
        return temp_disease_symptom