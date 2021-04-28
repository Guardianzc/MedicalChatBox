# -*- coding:utf-8 -*-
"""
Basic user simulator, random choice action.

# Structure of agent_action:
agent_action = {
    "turn":0,
    "speaker":"agent",
    "action":"request",
    "request_slots":{},
    "inform_slots":{},
    "explicit_inform_slots":{},
    "implicit_inform_slots":{}
}

# Structure of user_action:
user_action = {
    "turn": 0,
    "speaker": "user",
    "action": "request",
    "request_slots": {},
    "inform_slots": {},
    "explicit_inform_slots": {},
    "implicit_inform_slots": {}
}

# Structure of user goal.
{
  "consult_id": "10002219",
  "disease_tag": "上呼吸道感染",
  "goal": {
    "request_slots": {
      "disease": "UNK"
    },
    "explicit_inform_slots": {
      "呼吸不畅": true,
      "发烧": true
    },
    "implicit_inform_slots": {
      "厌食": true,
      "鼻塞": true
    }
  }

"""

import random
import copy
import pickle
import sys,os
sys.path.append(os.getcwd().replace("src/dialogue_system",""))

from src.dialogue_system import dialogue_configuration


class User(object):
    def __init__(self, goal_set_path, train_mode):
        self.goal_set = pickle.load(file=open(goal_set_path, "rb"))
        self.max_turn = 11
        self._init(train_mode)

    def initialize(self, train_mode=1):
        self._init(train_mode=train_mode)

    def _init(self,train_mode=1):
        """
        used for initializing an instance or an episode.
        """
        if train_mode == 1:
            self.goal = self.goal_set["train"]
        elif train_mode == 2:
            self.goal = self.goal_set["dev"]
        else:
            self.goal = self.goal_set["test"]

        self.goal_slots = dict()
        self.request_times = dict()
        for goal in self.goal:
            ids = goal['consult_id']
            self.request_times[ids] = 0
            self.goal_slots[ids] = dict()
            for key,items in goal['goal']["explicit_inform_slots"].items():
                self.goal_slots[ids][key] = items
            for key,items in goal['goal']["implicit_inform_slots"].items():
                self.goal_slots[ids][key] = items
                            

    def request(self, disease_id, slots):
        '''
        :return: 3 --- "Exceeded the maximum number of queries."
                 2 --- not know
                 1 --- True
                 0 --- False
        '''
        self.request_times[disease_id] += 1
        if self.request_times[disease_id] > self.max_turn:
            return 3
        else:
            if slots in self.goal_slots[disease_id].keys():
                if self.goal_slots[disease_id][slots]:
                    return 1
                else:
                    return 0
            else:
                return 2

    def request_time(self, disease_id):
        return self.request_times[disease_id]
    
if __name__ == '__main__':
    user = User('.\\src\\dialogue_system\\data\\dataset\\label\\goal_set.p',train_mode = 1)
    print(user.request_time('10000832'))
    print(user.request('10000832','咳嗽'))
    print(user.request('10000832','腹泻'))
    print(user.request('10000832','支气管炎'))
    for i in range(20):
        user.request('10000832','支气管炎')
    print(user.request('10000832','咳嗽'))
    print(user.request_time('10000832'))

